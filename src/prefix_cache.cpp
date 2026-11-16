// SPDX-License-Identifier: Apache-2.0
#include "prefix_cache.h"
#include "filesystem_utils.h"
#include "h3_config.h"
#include "scheduler.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#if NCNN_VULKAN
#include <command.h>
#include <gpu.h>
#endif
namespace h3image
{
namespace
{
uint64_t text_key(const ncnn::Mat& text)
{
    uint64_t hash = 14695981039346656037ull;
    for (int row = 0; row < text.h; row++)
    {
        const auto* bytes = reinterpret_cast<const unsigned char*>(text.row(row));
        for (size_t i = 0; i < static_cast<size_t>(text.w) * sizeof(float); i++)
            hash = (hash ^ bytes[i]) * 1099511628211ull;
    }
    return (hash ^ static_cast<uint64_t>(text.h)) * 1099511628211ull;
}
bool has_blob(const ncnn::Net& net, const std::string& name)
{
    for (const auto& blob : net.blobs())
        if (blob.name == name)
            return true;
    return false;
}

ncnn::Mat extract_prefix(
    std::string& error, ncnn::Net& net, const char* input_name, const ncnn::Mat& input, const char* output_name)
{
    VulkanWorkspace workspace(net);
    ncnn::Extractor extractor = net.create_extractor();
    workspace.bind(extractor);
    if (extractor.input(input_name, input))
    {
        error = "invalid prefix input";
        return {};
    }
    ncnn::Mat raw, result;
#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute)
    {
        ncnn::VkCompute command(net.vulkan_device());
        ncnn::VkMat value;
        if (extractor.extract(output_name, value, command) || value.empty())
        {
            error = "prefix extraction failed";
            return {};
        }
        auto options = net.opt;
        options.blob_vkallocator = workspace.blob.get();
        options.staging_vkallocator = workspace.staging.get();
        command.record_clone(value, raw, options);
        if (command.submit_and_wait())
        {
            error = "prefix download failed";
            return {};
        }
    }
    else
#endif
        if (extractor.extract(output_name, raw, 1))
    {
        error = "prefix extraction failed";
        return {};
    }
    ncnn::convert_packing(raw, result, 1, net.opt);
    if (result.empty())
    {
        error = "empty prefix output";
        return {};
    }
    if (result.allocator || !result.refcount)
        return result.clone();
    return result;
}

}
struct PrefixCache::Impl
{
    struct Entry
    {
        std::string name;
        ncnn::Mat host;
#if NCNN_VULKAN
        ncnn::VkMat device;
#endif
        void apply(std::string& error, ncnn::Extractor& extractor) const
        {
            int result;
#if NCNN_VULKAN
            if (!device.empty())
            {
                result = extractor.input(name.c_str(), device);
            }
            else
#endif
                result = extractor.input(name.c_str(), host);

            if (result)
            {
                error = "cannot apply prefix cache: " + name;
                return;
            }
        }
    };
    H3ModelSet& models;
    PrefixCachePlan plan;
    uint64_t host_bytes = 0;
    uint64_t device_bytes = 0;
#if NCNN_VULKAN
    // Allocators must outlive every cached VkMat.
    std::unique_ptr<VulkanWorkspace> workspace;
    std::unique_ptr<ncnn::VkCompute> uploads;
    int pending_uploads = 0;
#endif
    Entry text;
    std::vector<std::vector<Entry>> tables;

    explicit Impl(H3ModelSet& value)
        : models(value)
        , plan(value.memory_plan().prefix_cache)
        , tables(static_cast<size_t>(value.options().steps))
    {
#if NCNN_VULKAN
        if (plan.location == CacheLocation::Device)
        {
            const auto* device = ncnn::get_gpu_device(models.options().gpu);
            workspace = std::make_unique<VulkanWorkspace>(device, 4 * MiB);
        }
#endif
    }
    Entry store(std::string& error, const std::string& name, ncnn::Mat value)
    {
        if (value.empty() || value.dims != 2 || value.elempack != 1)
        {
            error = "invalid prefix cache tensor: " + name;
            return {};
        }
        Entry result;
        result.name = name;
#if NCNN_VULKAN
        if (workspace)
        {
            const auto* device = ncnn::get_gpu_device(models.options().gpu);
            ncnn::Option options;
            options.blob_vkallocator = workspace->blob.get();
            options.staging_vkallocator = workspace->staging.get();
            // Preserve the actual producer storage type; intermediate inputs must not be recast.
            options.use_bf16_storage = value.elembits() == 16;
            options.use_bf16_packed = value.elembits() == 16;
            options.use_fp16_storage = false;
            options.use_fp16_packed = false;
            if (!uploads)
                uploads = std::make_unique<ncnn::VkCompute>(device);
            uploads->record_upload(value, result.device, options);
            if (result.device.empty())
            {
                error = "cannot allocate Vulkan prefix cache: " + name;
                return {};
            }
            if (++pending_uploads == 2)
            {
                flush(error);
                if (!error.empty())
                    return {};
            }

            device_bytes += result.device.buffer_capacity();
            if (device_bytes > plan.device_bytes)
            {
                error = "Vulkan prefix cache exceeded its reserved budget";
                return {};
            }
            return result;
        }
#endif
        host_bytes += value.total() * value.elemsize;
        if (host_bytes > plan.host_bytes)
        {
            error = "RAM prefix cache exceeded its reserved budget";
            return {};
        }
        result.host = value.refcount ? std::move(value) : value.clone();
        return result;
    }
#if NCNN_VULKAN
    void flush(std::string& error)
    {
        if (uploads && uploads->submit_and_wait())
        {
            error = "prefix cache upload failed";
            return;
        }
        uploads.reset();
        pending_uploads = 0;
    }
#endif
    void populate(std::string& error, const ncnn::Mat& features, int timestep_count)
    {
        auto* loaded = models.prepare(error, Component::Transformer);
        if (!error.empty())
            return;

        if (!loaded)
            return;
        ncnn::Net& net = *loaded;
        if (!has_blob(net, "text_features") || (!models.condition_tables() && !has_blob(net, "time_embedding"))
            || !has_blob(net, "final_table"))
        {
            error = "model graph has no supported prefix cache endpoints";
            return;
        }
        for (int layer = 0; layer < kTransformerLayers; layer++)
            if (!has_blob(net, "layer" + std::to_string(layer) + "_table_flat"))
            {
                error = "model graph has incomplete prefix cache endpoints";
                return;
            }
#if NCNN_VULKAN
        if (workspace)
        {
            VulkanWorkspace forward_workspace(net);
            ncnn::Extractor extractor = net.create_extractor();
            forward_workspace.bind(extractor);
            ncnn::VkCompute command(net.vulkan_device());
            ncnn::VkMat refined;
            if (extractor.input("text", features) || extractor.extract("text_features", refined, command)
                || refined.empty())
            {
                error = "text refiner device extraction failed";
                return;
            }
            auto options = net.opt;
            options.blob_vkallocator = workspace->blob.get();
            options.staging_vkallocator = workspace->staging.get();
            command.record_clone(refined, text.device, options);
            if (text.device.empty() || command.submit_and_wait())
            {
                error = "text refiner device copy failed";
                return;
            }
            if (text.device.w != kTransformerWidth || text.device.h * text.device.elempack != features.h)
            {
                error = "text refiner returned an unsupported device shape";
                return;
            }
            text.name = "text_features";
            device_bytes += text.device.buffer_capacity();
            if (device_bytes > plan.device_bytes)
            {
                error = "device text cache exceeded its reserved budget";
                return;
            }
        }
        else
#endif
        {
            auto refined = extract_prefix(error, net, "text", features, "text_features");
            if (!error.empty())
                return;

            if (refined.w != kTransformerWidth || refined.h != features.h)
            {
                error = "text refiner returned an unsupported cache shape";
                return;
            }
            text = store(error, "text_features", std::move(refined));
            if (!error.empty())
                return;
        }
        ncnn::Mat embedding;
        if (!models.condition_tables())
        {
            embedding = extract_prefix(error, net, "time_frequencies",
                pdd_frequencies(error, timestep_count, PddScheduler(error, models.options().steps)), "time_embedding");
            if (!error.empty())
                return;
        }

        for (int layer = 0; layer <= kTransformerLayers; layer++)
        {
            const std::string name
                = layer == kTransformerLayers ? "final_table" : "layer" + std::to_string(layer) + "_table_flat";
            const ncnn::Mat all = models.condition_tables()
                ? models.condition_tables()->table(error, timestep_count, layer)
                : extract_prefix(error, net, "time_embedding", embedding, name.c_str());
            if (!error.empty())
                return;

            const int width = layer == kTransformerLayers ? 2 * kTransformerWidth : kModulationWidth;
            if (all.dims != 2 || all.w != width || all.h != models.options().steps * timestep_count)
            {
                error = "time conditioning returned an unsupported cache shape: " + name;
                return;
            }
            for (int step = 0; step < models.options().steps; step++)
            {
                tables[step].push_back(store(error, name, all.row_range(step * timestep_count, timestep_count)));
                if (!error.empty())
                    return;
            }

#if NCNN_VULKAN
            flush(error);
#endif
        }
#if NCNN_VULKAN
        // Prefill transfers do not stay alive beside the persistent cache.
        flush(error);
        if (workspace)
        {
            workspace->staging->clear();
        }
#endif
    }
};
PrefixCache::PrefixCache(H3ModelSet& models)
    : models_(models)
{
}
PrefixCache::~PrefixCache() = default;
void PrefixCache::clear()
{
    impl_.reset();
    attempted_ = false;
    key_ = 0;
    timestep_count_ = 0;
}
bool PrefixCache::prepare(const ncnn::Mat& text, int timestep_count)
{
    if (text.dims != 2 || text.w != kTextWidth || text.elempack != 1 || text.elembits() != 32 || timestep_count < 1
        || timestep_count > 2)
    {
        clear();

        return false;
    }
    const uint64_t key = text_key(text);
    if (attempted_ && key == key_ && timestep_count == timestep_count_)
        return impl_ != nullptr || models_.condition_tables();
    clear();

    attempted_ = true;
    key_ = key;
    timestep_count_ = timestep_count;
    const auto& plan = models_.memory_plan().prefix_cache;
    std::cerr << "exact prefix cache: " << cache_location_name(plan.location) << ", RAM reserve "
              << (plan.host_bytes + plan.host_scratch) / MiB << " MiB, VRAM reserve "
              << (plan.device_bytes + plan.device_scratch) / MiB << " MiB\n";
    if (plan.location == CacheLocation::Disabled)
        return models_.condition_tables() != nullptr;
    if (plan.text_tokens != static_cast<uint64_t>(text.h) || plan.timestep_count != timestep_count)
        return false;
    const auto begin = Clock::now();
    std::string error;
    {
        auto cache = std::make_unique<Impl>(models_);
        cache->populate(error, text, timestep_count);
        if (!error.empty())
        {
            std::cerr << "optional prefix cache unavailable: " << error << "\n";
            return models_.condition_tables() != nullptr;
        }
        std::cerr << "exact prefix cache ready: RAM " << cache->host_bytes / MiB << " MiB, VRAM "
                  << cache->device_bytes / MiB << " MiB, prefill " << elapsed_ms(begin) << " ms\n";
        impl_ = std::move(cache);
        return true;
    }
}
void PrefixCache::apply(std::string& error, ncnn::Extractor& extractor, int step) const
{
    if ((!impl_ && !models_.condition_tables()) || !attempted_ || step < 0 || step >= models_.options().steps)
    {
        error = "prefix cache is not ready for this PDD step";
        return;
    }
    if (impl_)
    {
        impl_->text.apply(error, extractor);
        if (!error.empty())
            return;

        for (const auto& entry : impl_->tables[step])
        {
            entry.apply(error, extractor);
            if (!error.empty())
                return;
        }
    }
    else
    {
        for (int layer = 0; layer <= kTransformerLayers; layer++)
        {
            const std::string name
                = layer == kTransformerLayers ? "final_table" : "layer" + std::to_string(layer) + "_table_flat";
            const ncnn::Mat value = models_.condition_tables()->table(error, timestep_count_, layer, step, true);
            if (!error.empty())
                return;

            if (extractor.input(name.c_str(), value))
            {
                error = "cannot bind fixed modulation table: " + name;
                return;
            }
        }
    }
}
}
