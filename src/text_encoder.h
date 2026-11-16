// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "models.h"
#include <string>
namespace h3image
{
class TextEncoder
{
public:
    explicit TextEncoder(H3ModelSet& models)
        : models_(models)
    {
    }
    ncnn::Mat encode(std::string& error, const ncnn::Mat& token_ids, const ncnn::Mat* pixels = nullptr,
        const ncnn::Mat* grids = nullptr);

private:
    H3ModelSet& models_;
};
}
