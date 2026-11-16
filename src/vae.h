// h3-image implemented with ncnn library
#pragma once
#include "models.h"
#include <string>
namespace h3image
{
class VAE
{
public:
    explicit VAE(H3ModelSet& models)
        : models_(models)
    {
    }
    ncnn::Mat encode(std::string& error, const ncnn::Mat& pixels);
    ncnn::Mat decode(std::string& error, const ncnn::Mat& latent);

private:
    ncnn::Mat decode_tile(std::string& error, const ncnn::Mat& latent);
    ncnn::Mat decode_stream(std::string& error, const ncnn::Mat& latent, int tile_size);
    ncnn::Mat decode_graph(std::string& error, const ncnn::Mat& latent);
    H3ModelSet& models_;
};
}
