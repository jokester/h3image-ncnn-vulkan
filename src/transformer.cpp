// SPDX-License-Identifier: Apache-2.0
#include "transformer.h"
#include "device_sampler.h"
#include "filesystem_utils.h"
#include "h3_config.h"
#include "scheduler.h"
#include <algorithm>
#include <cstring>
#if NCNN_VULKAN
#include <command.h>
#include <gpu.h>
#endif
#include <cmath>
namespace h3image
{
namespace
{
uint64_t append_identity(uint64_t hash, const void* data, size_t length)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < length; i++)
        hash = (hash ^ bytes[i]) * 1099511628211ull;
    return hash;
}
}
ncnn::Mat Transformer::forward(std::string& error, const TransformerInput& input, int step)
{
    return forward_internal(error, input, step
#if NCNN_VULKAN
        ,
        nullptr, nullptr, 0
#endif
    );
}
#if NCNN_VULKAN
void Transformer::step_vulkan(std::string& error, const TransformerInput& input, int step, ncnn::VkMat& images,
    DeviceSampler& sampler, int reference_rows)
{
    forward_internal(error, input, step, &images, &sampler, reference_rows);
}

#endif
ncnn::Mat Transformer::forward_internal(std::string& error, const TransformerInput& input, int step
#if NCNN_VULKAN
    ,
    ncnn::VkMat* device_images, DeviceSampler* sampler, int reference_count
#endif
)
{
    const PddScheduler schedule(error, models_.options().steps);
    if (!error.empty())
        return {};

    if (step < 0 || step >= schedule.step_count())
    {
        error = "DiT PDD step is outside the schedule";
        return {};
    }
    const auto& images = input.images;
    const auto& text = input.text;
    const auto& times = input.timesteps;
    const auto& ti = input.timestep_indices;
    const auto& tags = input.token_tags;
    const auto& positions = input.positions;
    const auto& ii = input.image_indices;
    const auto& ix = input.text_indices;
    const int n = positions.h, m = times.w;
    if (positions.dims != 2 || positions.w != 3 || images.dims != 2 || images.h != ii.w || images.w != 96
        || text.dims != 2 || text.h != ix.w || text.w != 5120 || ti.dims != 1 || ti.w != n || tags.dims != 1
        || tags.w != n || times.dims != 1 || ii.dims != 1 || ix.dims != 1)
    {
        error = "DiT matrix/layout shape mismatch";
        return {};
    }
    std::vector<int> used(n, 0);
    for (const ncnn::Mat* indices : { &ii, &ix })
        for (int j = 0; j < indices->w; j++)
        {
            const float value = (*indices)[j];
            const int i = (int)value;
            if (i < 0 || i >= n || i != value || used[i]++)
            {
                error = "invalid packed DiT row indices";
                return {};
            }
        }
    if (std::find(used.begin(), used.end(), 0) != used.end())
    {
        error = "DiT row indices do not partition sequence";
        return {};
    }
    for (int i = 0; i < n; i++)
        if (ti[i] < 0 || ti[i] >= m || ti[i] != (int)ti[i] || tags[i] < 0 || tags[i] > 2 || tags[i] != (int)tags[i])
        {
            error = "invalid DiT row table indices";
            return {};
        }
    const bool canonical_times = (m == 1 || m == 2) && times[0] == schedule.timestep(error, step)
        && (m == 1 || times[1] == kReferenceNoiseAug);
    if (!error.empty())
        return {};

    if (models_.condition_tables() && !canonical_times)
    {
        error = "modulation core requires the selected PDD schedule";
        return {};
    }
    bool cached = false;
    if (canonical_times)
    {
        cached = prefix_cache_.prepare(text, m);
    }
    else
        prefix_cache_.clear();
    const int width = 64;
    uint64_t metadata_key = static_cast<uint64_t>(m);
    for (const ncnn::Mat* value : { &positions, &ti, &tags, &ii, &ix })
    {
        metadata_key = append_identity(metadata_key, &value->w, sizeof(value->w));
        metadata_key = append_identity(metadata_key, &value->h, sizeof(value->h));
        metadata_key = append_identity(metadata_key, value->data, value->total() * value->elemsize);
    }
    if (selectors_.empty() || metadata_key != metadata_key_)
    {
        selectors_.create(m * 3, n);
        image_times_.create(m, ii.w);
        cos_.create(width, n, 1);
        sin_.create(width, n, 1);
        auto& selectors = selectors_;
        auto& image_times = image_times_;
        auto& cos = cos_;
        auto& sin = sin_;
        selectors.fill(0.f);
        image_times.fill(0.f);
        cos.fill(0.f);
        sin.fill(0.f);
        order_.clear();

        for (int i = 0; i < ix.w; i++)
            order_.push_back((int)ix[i]);
        for (int i = 0; i < ii.w; i++)
            order_.push_back((int)ii[i]);
        for (int i = 0; i < n; i++)
        {
            const int row = order_[i];
            selectors.row(i)[(int)ti[row] * 3 + (int)tags[row]] = 1.f;
            for (int axis = 0; axis < 3; axis++)
                for (int j = 0; j < 16; j++)
                {
                    const float angle = positions.row(row)[axis] / std::pow(10000.f, (float)(2 * j) / 32);
                    const int offset = axis * 16 + j;
                    cos.channel(0).row(i)[offset] = std::cos(angle);
                    sin.channel(0).row(i)[offset] = std::sin(angle);
                }
            for (int j = 48; j < 64; j++)
                cos.channel(0).row(i)[j] = 1.f;
        }
        for (int i = 0; i < ii.w; i++)
            image_times.row(i)[(int)ti[(int)ii[i]]] = 1.f;
        metadata_key_ = metadata_key;
    }
    ncnn::Mat selectors = selectors_, image_times = image_times_, cos = cos_, sin = sin_;
    const auto& order = order_;
    int reference_rows = 0;
    for (int i = 0; i < ii.w; i++)
        reference_rows += ti[static_cast<int>(ii[i])] != 0.f;
    uint64_t identity = prefix_cache_.text_identity();
    if (kv_cache_.enabled())
    {
        for (int i = 0; i < n; ++i)
            if (order[i] != i)
            {
                error = "rough mode requires canonical row ordering";
                return {};
            }
        for (int i = 0; i < ii.w; ++i)
            if ((ti[static_cast<int>(ii[i])] != 0.f) != (i < reference_rows))
            {
                error = "rough mode requires reference rows before generated rows";
                return {};
            }
        identity = append_identity(identity, positions.data, n * 3 * sizeof(float));
        identity = append_identity(identity, tags.data, n * sizeof(float));
        identity = append_identity(identity, ti.data, n * sizeof(float));
        identity
            = append_identity(identity, images.data, static_cast<size_t>(reference_rows) * images.w * sizeof(float));
    }
    const int omitted = kv_cache_.begin(
        error, models_.memory_plan().kv_cache, step, text.h, reference_rows, n, identity, canonical_times);
    if (!error.empty())
        return {};

    const int omitted_images = omitted ? omitted - text.h : 0;
    ncnn::Mat active_images = omitted ? images.row_range(omitted_images, images.h - omitted_images).clone() : images;
    if (omitted)
    {
        selectors = selectors.row_range(omitted, n - omitted).clone();
        image_times = image_times.row_range(omitted_images, image_times.h - omitted_images).clone();
        ncnn::Mat active_cos(width, n - omitted, 1), active_sin(width, n - omitted, 1);
        std::memcpy(active_cos.data, cos.channel(0).row(omitted), (n - omitted) * width * sizeof(float));
        std::memcpy(active_sin.data, sin.channel(0).row(omitted), (n - omitted) * width * sizeof(float));
        cos = active_cos;
        sin = active_sin;
    }
    auto* net = models_.prepare(error, Component::Transformer);
    if (!net)
        return {};
    VulkanWorkspace workspace(*net);
    ncnn::Extractor extractor = net->create_extractor();
    workspace.bind(extractor);
    if (extractor.input("text", text) || extractor.input("modality_selector", selectors)
        || extractor.input("image_time_selector", image_times) || extractor.input("cos", cos)
        || extractor.input("sin", sin))
    {
        error = "invalid transformer inputs";
        return {};
    }
#if NCNN_VULKAN
    if (!device_images)
#endif
        if (extractor.input("images", active_images))
        {
            error = "invalid transformer image input";
            return {};
        }
    if (cached)
        prefix_cache_.apply(error, extractor, step);
    if (omitted)
    {
#if NCNN_VULKAN
        if (models_.options().gpu >= 0)
        {
            const auto* device = ncnn::get_gpu_device(models_.options().gpu);
            ncnn::VkCompute command(device);
            ncnn::VkMat features;
            if (extractor.extract("image_features", features, command) || features.empty() || command.submit_and_wait())
            {
                error = "cannot project active image rows";
                return {};
            }
            if (extractor.input("packed", features))
            {
                error = "cannot bypass cached input prefix";
                return {};
            }
        }
        else
#endif
        {
            ncnn::Mat features;
            if (extractor.extract("image_features", features, 1) || features.empty()
                || extractor.input("packed", features))
            {
                error = "cannot project active CPU image rows";
                return {};
            }
        }
    }

    if (!error.empty())
        return {};
    const std::string output = "out" + std::to_string(schedule.output_head(error, step));
    if (!error.empty())
        return {};
#if NCNN_VULKAN
    if (device_images)
    {
        if (omitted || !sampler)
        {
            error = "device sampler does not support shortened rough-mode rows";
            return {};
        }
        auto options = net->opt;
        options.blob_vkallocator = workspace.blob.get();
        options.workspace_vkallocator = workspace.blob.get();
        options.staging_vkallocator = workspace.staging.get();
        ncnn::VkCompute command(net->vulkan_device());
        ncnn::VkMat native_images, velocity;
        // VkMat input skips ncnn's host upload cast. Keep the sample FP32 and
        // cast only the model input to the graph's BF16 packed row layout.
        const int rows = device_images->h * device_images->elempack;
        net->vulkan_device()->convert_packing(
            *device_images, native_images, rows % 4 == 0 ? 4 : 1, 5, command, options);
        if (native_images.empty() || extractor.input("images", native_images)
            || extractor.extract(output.c_str(), velocity, command) || velocity.empty())
        {
            error = "transformer device extraction failed";
            return {};
        }
        sampler->record_step(error, schedule, step, velocity, reference_count, *device_images, command);
        if (!error.empty())
            return {};
        if (command.submit_and_wait())
        {
            error = "transformer device submission failed";
            return {};
        }
        kv_cache_.finish(error);
        return {};
    }
#endif
    ncnn::Mat raw;
    if (extractor.extract(output.c_str(), raw))
    {
        error = "transformer extraction failed";
        return {};
    }
    auto result = unpack_float32(error, raw);
    if (!error.empty())
        return {};
    kv_cache_.finish(error);
    if (!error.empty())
        return {};
    if (omitted_images)
    {
        if (result.h != images.h - omitted_images)
        {
            error = "unexpected shortened output";
            return {};
        }
        ncnn::Mat full(result.w, images.h);
        full.fill(0.f);
        std::memcpy(full.row(omitted_images), result.data, result.w * result.h * sizeof(float));
        return full;
    }
    return result;
}
} // namespace h3image
