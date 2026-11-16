// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "models.h"
#include <string>
namespace h3image
{
// Tile convolutions with their exact halos; preserve full-image GroupNorm and
// residuals. H3's spatial GroupNorm prevents whole-frontend tiling as in Qwen.
ncnn::Mat encode_vae_tiled(std::string& error, H3ModelSet& models, const ncnn::Mat& image, int tile_size);
}
