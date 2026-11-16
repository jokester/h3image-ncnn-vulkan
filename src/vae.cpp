// SPDX-License-Identifier: Apache-2.0
#include "vae.h"
#include "filesystem_utils.h"
#include "vae_encoder.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <gpu.h>
#include <iostream>
namespace h3image
{
ncnn::Mat VAE::decode_graph(std::string& error, const ncnn::Mat& a)
{
    const int h = a.h, w = a.w, n = h * w + 5;
    const int cache_width = 32;
    ncnn::Mat cos(cache_width, n, 1), sin(cache_width, n, 1);
    cos.fill(0.f);
    sin.fill(0.f);
    for (int i = 0; i < n; i++)
        for (int axis = 0; axis < 2; axis++)
            for (int j = 0; j < 8; j++)
            {
                const float position = i < h * w
                    ? (axis == 0 ? 2.f * ((float)(i / w) + .5f) / h - 1.f : 2.f * ((float)(i % w) + .5f) / w - 1.f)
                    : 0.f;
                const float angle = 2.f * 3.14159265358979323846f * position * std::pow(100.f, -(float)j / 8);
                const int offset = axis * 8 + j;
                cos.channel(0).row(i)[offset] = std::cos(angle);
                sin.channel(0).row(i)[offset] = std::sin(angle);
            }
    for (int i = 0; i < n; i++)
        for (int j = 16; j < 32; j++)
            cos.channel(0).row(i)[j] = 1.f;
    auto* net = models_.prepare(error, Component::VAEDecoder);
    if (!net)
        return {};
    VulkanWorkspace workspace(*net);
    ncnn::Extractor extractor = net->create_extractor();
    workspace.bind(extractor);
    ncnn::Mat output;
    if (extractor.input("latent", a) || extractor.input("cos", cos) || extractor.input("sin", sin)
        || extractor.extract("out0", output))
    {
        error = "VAE decoder inference failed";
        return {};
    }
    return unpack_float32(error, output);
}
ncnn::Mat VAE::decode_tile(std::string& error, const ncnn::Mat& a)
{
    ncnn::Mat output = decode_graph(error, a), pixels(a.w * 16, a.h * 16, 3);
    if (!error.empty())
        return {};

    for (int y = 0; y < a.h; y++)
        for (int x = 0; x < a.w; x++)
            for (int c = 0; c < 3; c++)
                for (int py = 0; py < 16; py++)
                    for (int px = 0; px < 16; px++)
                        pixels.channel(c).row(y * 16 + py)[x * 16 + px]
                            = output.row(y * a.w + x)[c * 256 + py * 16 + px];
    return pixels;
}
namespace
{
struct TileAxis
{
    std::vector<int> starts, overlaps;
    int tile;
};
TileAxis split_axis(int length, int tile)
{
    if (length <= tile)
        return { { 0 }, {}, length };
    const int overlap = tile / 4;
    int count = (length + tile - 1) / tile;
    while (tile * count - overlap * (count - 1) < length)
        count++;
    TileAxis result;
    result.tile = tile;
    result.overlaps.assign(count - 1, overlap);
    const int remaining = tile * count - overlap * (count - 1) - length;
    for (int i = 0; i < remaining / 16; i++)
        result.overlaps[i % (count - 1)] += 16;
    result.starts.push_back(0);
    for (int value : result.overlaps)
        result.starts.push_back(result.starts.back() + tile - value);
    return result;
}
ncnn::Mat crop_latent(const ncnn::Mat& input, int y, int x, int h, int w)
{
    ncnn::Mat result(w, h, input.c);
    for (int c = 0; c < input.c; c++)
        for (int row = 0; row < h; row++)
            std::memcpy(result.channel(c).row(row), input.channel(c).row(y + row) + x, w * sizeof(float));
    return result;
}
void blend_overlap(const ncnn::Mat& previous, ncnn::Mat& tile, int extent, bool vertical)
{
    for (int c = 0; c < tile.c; c++)
        for (int y = 0; y < tile.h; y++)
            for (int x = 0; x < tile.w; x++)
            {
                const int position = vertical ? y : x;
                if (position >= extent)
                    continue;
                const float weight = (float)position / extent;
                const float value = vertical ? previous.channel(c).row(previous.h - extent + y)[x]
                                             : previous.channel(c).row(y)[previous.w - extent + x];
                float& pixel = tile.channel(c).row(y)[x];
                pixel = value * (1.f - weight) + pixel * weight;
            }
}
}
ncnn::Mat VAE::decode_stream(std::string& error, const ncnn::Mat& latent, int tile_size)
{
    if (tile_size < 64 || tile_size % 64)
    {
        error = "decoder context tile must be a multiple of 64 pixels";
        return {};
    }
    const TileAxis ys = split_axis(latent.h * 16, tile_size), xs = split_axis(latent.w * 16, tile_size);
    ncnn::Mat result(latent.w * 16, latent.h * 16, 3);
    std::vector<ncnn::Mat> previous(xs.starts.size());
    for (size_t i = 0; i < ys.starts.size(); i++)
    {
        ncnn::Mat left;
        for (size_t j = 0; j < xs.starts.size(); j++)
        {
            ncnn::Mat input = crop_latent(latent, ys.starts[i] / 16, xs.starts[j] / 16, ys.tile / 16, xs.tile / 16);
            ncnn::Mat raw = decode_tile(error, input), part = raw.clone();
            if (!error.empty())
                return {};

            if (i)
                blend_overlap(previous[j], part, ys.overlaps[i - 1], true);
            if (j)
                blend_overlap(left, part, xs.overlaps[j - 1], false);
            const int height = part.h - (i + 1 < ys.starts.size() ? ys.overlaps[i] : 0),
                      width = part.w - (j + 1 < xs.starts.size() ? xs.overlaps[j] : 0);
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < height; y++)
                    std::memcpy(result.channel(c).row(ys.starts[i] + y) + xs.starts[j], part.channel(c).row(y),
                        width * sizeof(float));
            left = raw;
            previous[j] = raw;
        }
    }
    return result;
}
ncnn::Mat VAE::encode(std::string& error, const ncnn::Mat& input)
{
    if (input.dims != 3 || input.c != 3)
    {
        error = "VAE input must be CHW RGB";
        return {};
    }
    const int pixels = models_.memory_plan().vae_encoder_tile;
    if (pixels == 0)
    {
        std::cerr << "vae encoder: whole graph, budgeted device intermediates\n";
        auto* net = models_.prepare(error, Component::VAEEncoder);
        if (!net)
            return {};
        VulkanWorkspace workspace(*net);
        ncnn::Extractor extractor = net->create_extractor();
        workspace.bind(extractor);
        ncnn::Mat output;
        if (extractor.input("in0", input) || extractor.extract("out0", output))
        {
            error = "VAE encoder inference failed";
            return {};
        }
        return unpack_float32(error, output);
    }
    std::cerr << "vae encoder tile size = " << pixels << " x " << pixels << "\n";
    return encode_vae_tiled(error, models_, input, pixels);
}
ncnn::Mat VAE::decode(std::string& error, const ncnn::Mat& input)
{
    if (input.dims != 3 || input.c != 24)
    {
        error = "VAE latent must have 24 channels";
        return {};
    }
    // H3 was trained with overlapping 256-pixel decoder context windows.
    return decode_stream(error, input, 256);
}
}
