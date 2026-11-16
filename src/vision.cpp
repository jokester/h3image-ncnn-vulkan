// SPDX-License-Identifier: Apache-2.0
#include "vision.h"
#include "filesystem_utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
namespace h3image
{
namespace
{
ncnn::Mat load_positions(std::string& error, const H3ModelSet& models)
{
    const std::string path = models.directory() + "/vision/vision_pos_embed.f32";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    constexpr int width = 1152, height = 48 * 48;
    if (!file || file.tellg() != static_cast<std::streamoff>(width * height * sizeof(float)))
    {
        error = "invalid vision position embeddings: " + path;
        return {};
    }
    file.seekg(0);
    ncnn::Mat positions(width, height);
    for (int i = 0; i < height; i++)
        file.read(reinterpret_cast<char*>(positions.row(i)), width * sizeof(float));
    if (!file)
    {
        error = "cannot read vision position embeddings: " + path;
        return {};
    }
    return positions;
}
}
VisionEncoder::VisionEncoder(std::string& error, H3ModelSet& models)
    : models_(models)
    , positions_(load_positions(error, models))
{
}
VisionResult VisionEncoder::encode(std::string& error, const ncnn::Mat& pixels, int h, int w)
{
    if (h <= 0 || w <= 0 || h % 2 || w % 2 || pixels.dims != 2 || pixels.h != h * w || pixels.w != 1536)
    {
        error = "vision pixels must have shape (H*W,1536), with even patch-grid H,W";
        return {};
    }
    ncnn::Mat position_embeds(1152, h * w);
    const int cache_width = 36;
    ncnn::Mat cos(cache_width, h * w, 1), sin(cache_width, h * w, 1);
    for (int by = 0; by < h / 2; by++)
        for (int bx = 0; bx < w / 2; bx++)
            for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++)
                {
                    const int y = by * 2 + dy, x = bx * 2 + dx, index = ((by * (w / 2) + bx) * 2 + dy) * 2 + dx;
                    const float step_y = 47.f / (h - 1), step_x = 47.f / (w - 1);
                    const float gy = y < h / 2 ? (float)y * step_y : std::fma(-(float)(h - y - 1), step_y, 47.f);
                    const float gx = x < w / 2 ? (float)x * step_x : std::fma(-(float)(w - x - 1), step_x, 47.f);
                    const int y0 = (int)gy, x0 = (int)gx, y1 = std::min(y0 + 1, 47), x1 = std::min(x0 + 1, 47);
                    const float fy = gy - y0, fx = gx - x0,
                                weights[4] = { (1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx };
                    float* out = position_embeds.row(index);
                    for (int d = 0; d < 1152; d++)
                    {
                        float value = positions_.row(y0 * 48 + x0)[d] * weights[0];
                        value += positions_.row(y0 * 48 + x1)[d] * weights[1];
                        value += positions_.row(y1 * 48 + x0)[d] * weights[2];
                        value += positions_.row(y1 * 48 + x1)[d] * weights[3];
                        out[d] = value;
                    }
                    for (int axis = 0; axis < 2; axis++)
                        for (int j = 0; j < 18; j++)
                        {
                            const float inverse = 1.f / std::pow(10000.f, (float)(2 * j) / 36);
                            const float angle = (float)(axis == 0 ? y : x) * inverse;
                            const int offset = axis * 18 + j;
                            cos.channel(0).row(index)[offset] = std::cos(angle);
                            sin.channel(0).row(index)[offset] = std::sin(angle);
                        }
                }
    auto* net = models_.prepare(error, Component::Vision);
    if (!net)
        return {};
    VulkanWorkspace workspace(*net);
    ncnn::Extractor extractor = net->create_extractor();
    workspace.bind(extractor);
    if (extractor.input("in0", pixels) || extractor.input("positions", position_embeds) || extractor.input("cos", cos)
        || extractor.input("sin", sin))
    {
        error = "invalid vision encoder inputs";
        return {};
    }
    const char* names[] = { "out0", "deepstack0", "deepstack1", "deepstack2" };
    std::array<ncnn::Mat, 4> values;
#if NCNN_VULKAN
    if (net->opt.use_vulkan_compute)
    {
        ncnn::VkCompute command(net->vulkan_device());
        auto options = net->opt;
        options.blob_vkallocator = workspace.blob.get();
        options.staging_vkallocator = workspace.staging.get();
        for (int i = 0; i < 4; i++)
        {
            ncnn::VkMat value;
            if (extractor.extract(names[i], value, command) || value.empty())
            {
                error = "vision encoder extraction failed";
                return {};
            }
            command.record_download(value, values[i], options);
        }
        if (command.submit_and_wait())
        {
            error = "vision encoder output download failed";
            return {};
        }
    }
    else
#endif
    {
        for (int i = 0; i < 4; i++)
            if (extractor.extract(names[i], values[i]))
            {
                error = "vision encoder extraction failed";
                return {};
            }
    }
    for (auto& value : values)
    {
        value = unpack_float32(error, value);
        if (!error.empty())
            return {};
    }
    VisionResult result;
    result.merged = values[0];
    for (int i = 1; i < 4; i++)
        result.deepstack.push_back(values[i]);
    return result;
}
VisionResult VisionEncoder::encode_many(std::string& error, const ncnn::Mat& pixels, const ncnn::Mat& grids)
{
    if (grids.dims != 2 || grids.w != 3 || pixels.dims != 2 || pixels.w != 1536)
    {
        error = "vision input/grid shapes are invalid";
        return {};
    }
    int total = 0, patch_count = 0;
    for (int i = 0; i < grids.h; i++)
    {
        const float* grid = grids.row(i);
        const int t = grid[0], h = grid[1], w = grid[2];
        if (t <= 0 || h <= 0 || w <= 0 || h % 2 || w % 2 || t != grid[0] || h != grid[1] || w != grid[2])
        {
            error = "invalid vision grid";
            return {};
        }
        patch_count += t * h * w;
        total += t * h * w / 4;
    }
    if (patch_count != pixels.h)
    {
        error = "pixel patch count does not match image_grid_thw";
        return {};
    }
    VisionResult result;
    result.merged.create(5120, total);
    result.deepstack.resize(3);
    for (auto& value : result.deepstack)
        value.create(5120, total);
    int patch_offset = 0, output_offset = 0;
    for (int i = 0; i < grids.h; i++)
    {
        const float* grid = grids.row(i);
        const int t = grid[0], h = grid[1], w = grid[2];
        for (int frame = 0; frame < t; frame++)
        {
            ncnn::Mat input = pixels.row_range(patch_offset, h * w);
            patch_offset += h * w;
            VisionResult value = encode(error, input, h, w);
            if (!error.empty())
                return {};

            for (int row = 0; row < h * w / 4; row++)
            {
                std::memcpy(result.merged.row(output_offset + row), value.merged.row(row), 5120 * sizeof(float));
                for (int j = 0; j < 3; j++)
                    std::memcpy(result.deepstack[j].row(output_offset + row), value.deepstack[j].row(row),
                        5120 * sizeof(float));
            }
            output_offset += h * w / 4;
        }
    }
    return result;
}
ncnn::Mat multimodal_positions(std::string& error, const ncnn::Mat& ids, const ncnn::Mat& grids)
{
    const int count = ids.w;
    int position = 0, grid_index = 0;
    ncnn::Mat result(count, 3);
    result.fill(0.f);
    for (int i = 0; i < count;)
    {
        if ((int)ids[i] != 151655)
        {
            for (int axis = 0; axis < 3; axis++)
                result.row(axis)[i] = (float)position;
            i++;
            position++;
            continue;
        }
        if (grid_index >= grids.h)
        {
            error = "image token run has no matching vision grid";
            return {};
        }
        const float* grid = grids.row(grid_index++);
        const int t = grid[0], h = (int)grid[1] / 2, w = (int)grid[2] / 2;
        if (i + t * h * w > count)
        {
            error = "image token run exceeds sequence length";
            return {};
        }
        for (int j = i; j < i + t * h * w; j++)
            if ((int)ids[j] != 151655)
            {
                error = "image tokens must form a complete contiguous grid";
                return {};
            }
        for (int tt = 0; tt < t; tt++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    const int index = i + (tt * h + y) * w + x;
                    result.row(0)[index] = position + tt;
                    result.row(1)[index] = position + y;
                    result.row(2)[index] = position + x;
                }
        i += t * h * w;
        position += std::max(h, w);
    }
    if (grid_index != grids.h)
    {
        error = "unused vision grid";
        return {};
    }
    return result;
}
}
