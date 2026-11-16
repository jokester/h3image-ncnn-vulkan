// h3-image implemented with ncnn library
#pragma once
#include "filesystem_utils.h"
#include <cstdio>
#include <mat.h>
#include <memory>
#include <string>
#include <vector>
namespace h3image
{
class LatentBatch
{
public:
    LatentBatch(std::string& error, bool spill, int width, int height, int channels)
        : file_(spill ? std::tmpfile() : nullptr, std::fclose)
        , spill_(spill)
        , width_(width)
        , height_(height)
        , channels_(channels)
    {
        if (spill_ && !file_)
        {
            error = "cannot create batch latent storage; check temporary disk space";
            return;
        }
    }
    void append(std::string& error, ncnn::Mat latent)
    {
        if (reading_ || latent.w != width_ || latent.h != height_ || latent.c != channels_ || latent.elembits() != 32
            || latent.elempack != 1)
        {
            error = "invalid batch latent storage shape or order";
            return;
        }
        const size_t plane = static_cast<size_t>(width_) * height_;
        if (spill_)
        {
            for (int c = 0; c < channels_; c++)
                if (std::fwrite(latent.channel(c), sizeof(float), plane, file_.get()) != plane)
                {
                    error = "failed to write batch latent storage";
                    return;
                }
        }
        else
            values_.push_back(std::move(latent));
        count_++;
    }
    ncnn::Mat next(std::string& error)
    {
        if (index_ >= count_)
        {
            error = "batch latent storage exhausted";
            return {};
        }
        if (!reading_)
        {
            if (spill_)
            {
                std::rewind(file_.get());
            }

            reading_ = true;
        }
        if (!spill_)
            return std::move(values_[index_++]);
        ncnn::Mat latent(width_, height_, channels_);
        const size_t plane = static_cast<size_t>(width_) * height_;
        for (int c = 0; c < channels_; c++)
            if (std::fread(latent.channel(c), sizeof(float), plane, file_.get()) != plane)
            {
                error = "failed to read batch latent storage";
                return {};
            }
        index_++;
        return latent;
    }

private:
    std::unique_ptr<FILE, int (*)(FILE*)> file_;
    bool spill_, reading_ = false;
    int width_, height_, channels_;
    size_t count_ = 0, index_ = 0;
    std::vector<ncnn::Mat> values_;
};
}
