// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <mat.h>
#include <string>

namespace h3image
{
// Interleaved RGB bytes: w/h are pixels, one packed 3-byte element per pixel.
inline bool is_rgb_image(const ncnn::Mat& image)
{
    return !image.empty() && image.dims == 2 && image.w > 0 && image.h > 0 && image.elemsize == 3
        && image.elempack == 3;
}

bool image_size(const std::string& path, int& width, int& height);
std::string make_batch_output_path(const std::string& path, int b, int batch);
ncnn::Mat load_rgb_image(std::string& error, const std::string& path);
ncnn::Mat resize_rgb_lanczos(std::string& error, const ncnn::Mat& image, int width, int height);
void save_rgb_image(std::string& error, const std::string& path, const ncnn::Mat& image);
}
