// SPDX-License-Identifier: Apache-2.0
#include "text_encoder.h"
#include "filesystem_utils.h"
#include "vision.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
namespace h3image
{
namespace
{
ncnn::Mat lookup_embeddings(
    std::string& error, const std::string& path, const ncnn::Mat& ids, const std::vector<int>& replaced)
{
    // Read only requested rows. The table never becomes a full FP32 allocation.
    const int vocabulary = 151936, width = 5120;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        error = "cannot open text embedding table";
        return {};
    }
    const auto bytes = file.tellg();
    file.seekg(0);
    uint32_t tag = 0;
    file.read(reinterpret_cast<char*>(&tag), sizeof(tag));
    const bool bf16 = tag == 0x01348b83;
    if (!file || (tag != 0 && !bf16))
    {
        error = "unsupported text embedding storage tag";
        return {};
    }
    const size_t row_bytes = width * (bf16 ? 2u : 4u);
    if (bytes != static_cast<std::streamoff>(sizeof(tag) + vocabulary * row_bytes))
    {
        error = "invalid text embedding table size";
        return {};
    }
    ncnn::Mat hidden(width, ids.w, bf16 ? static_cast<size_t>(2) : static_cast<size_t>(4));
    std::vector<bool> skip(ids.w, false);
    for (int row : replaced)
        skip[row] = true;
    for (int i = 0; i < ids.w; i++)
    {
        if (skip[i])
            continue;
        file.seekg(sizeof(tag) + static_cast<size_t>(ids[i]) * row_bytes);
        void* destination = static_cast<unsigned char*>(hidden.data) + static_cast<size_t>(i) * row_bytes;
        file.read(static_cast<char*>(destination), row_bytes);
        if (!file)
        {
            error = "cannot read text embedding row";
            return {};
        }
    }
    return hidden;
}
}
ncnn::Mat TextEncoder::encode(std::string& error, const ncnn::Mat& ids, const ncnn::Mat* pixels, const ncnn::Mat* grids)
{
    if (ids.dims != 1 || ids.w <= 0)
    {
        error = "token_ids must have shape (tokens,)";
        return {};
    }
    const int count = ids.w;
    for (int i = 0; i < count; i++)
        if (!std::isfinite(ids[i]) || ids[i] < 0 || ids[i] >= 151936 || ids[i] != (int)ids[i])
        {
            error = "token out of range";
            return {};
        }
    ncnn::Mat positions;
    VisionResult visual;
    std::vector<int> visual_indices;
    if (pixels)
    {
        if (!grids)
        {
            error = "vision grids are required";
            return {};
        }
        VisionEncoder vision(error, models_);
        visual = vision.encode_many(error, *pixels, *grids);
        if (!error.empty())
            return {};

        for (int i = 0; i < count; i++)
            if ((int)ids[i] == 151655)
                visual_indices.push_back(i);
        if ((int)visual_indices.size() != visual.merged.h)
        {
            error = "vision token count mismatch";
            return {};
        }
        positions = multimodal_positions(error, ids, *grids);
        if (!error.empty())
            return {};
    }
    ncnn::Mat hidden
        = lookup_embeddings(error, models_.directory() + "/text_encoder/embedding.bin", ids, visual_indices);
    if (!error.empty())
        return {};

    ncnn::Mat merged = visual.merged;
    if (!merged.empty() && hidden.elembits() == 16)
    {
        ncnn::Mat native;
        ncnn::cast_float32_to_bfloat16(merged, native);
        merged = native;
    }
    for (size_t i = 0; i < visual_indices.size(); i++)
        std::memcpy(
            static_cast<unsigned char*>(hidden.data) + static_cast<size_t>(visual_indices[i]) * 5120 * hidden.elemsize,
            static_cast<const unsigned char*>(merged.data) + i * 5120 * merged.elemsize, 5120 * hidden.elemsize);
    if (!positions.empty() && (positions.dims != 2 || positions.w != count || positions.h != 3))
    {
        error = "positions must have shape (3,tokens)";
        return {};
    }
    const int width = 64;
    ncnn::Mat cos(width, count, 1), sin(width, count, 1), mask(count, count);
    cos.fill(0.f);
    sin.fill(0.f);
    mask.fill(0.f);
    for (int i = 0; i < count; i++)
    {
        for (int j = 0; j < 64; j++)
        {
            const int axis = (j < 60 && j % 3 == 1) ? 1 : (j < 60 && j % 3 == 2) ? 2 : 0;
            const float position = positions.empty() ? (float)i : positions.row(axis)[i];
            const float angle = position / std::pow(5000000.f, (float)(2 * j) / 128);
            cos.channel(0).row(i)[j] = std::cos(angle);
            sin.channel(0).row(i)[j] = std::sin(angle);
        }
        for (int j = i + 1; j < count; j++)
            mask.row(i)[j] = -std::numeric_limits<float>::infinity();
    }
    std::vector<ncnn::Mat> additions(3);
    for (int layer = 0; layer < 3; layer++)
    {
        additions[layer].create(5120, count);
        additions[layer].fill(0.f);
        for (size_t j = 0; j < visual_indices.size(); j++)
            std::memcpy(additions[layer].row(visual_indices[j]), visual.deepstack[layer].row(j), 5120 * sizeof(float));
    }
    auto* net = models_.prepare(error, Component::Text);
    if (!net)
        return {};
    VulkanWorkspace workspace(*net);
    ncnn::Extractor extractor = net->create_extractor();
    workspace.bind(extractor);
    ncnn::Mat output;
    if (extractor.input("embeddings", hidden) || extractor.input("cos", cos) || extractor.input("sin", sin)
        || extractor.input("mask", mask) || extractor.input("deepstack0", additions[0])
        || extractor.input("deepstack1", additions[1]) || extractor.input("deepstack2", additions[2])
        || extractor.extract("out0", output))
    {
        error = "text encoder inference failed";
        return {};
    }
    return unpack_float32(error, output);
}
}
