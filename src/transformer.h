// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "filesystem_utils.h"
#include "kv_cache.h"
#include "models.h"
#include "prefix_cache.h"
#include <string>
namespace h3image
{
class DeviceSampler;
struct TransformerInput
{
    ncnn::Mat images, text, timesteps, timestep_indices, token_tags;
    ncnn::Mat positions, image_indices, text_indices;
};
class Transformer
{
public:
    explicit Transformer(H3ModelSet& models)
        : models_(models)
        , prefix_cache_(models)
        , kv_cache_(models.options(), models.kv_cache_context())
    {
    }
    void prepare(const ncnn::Mat& text, int timestep_count)
    {
        prefix_cache_.prepare(text, timestep_count);
    }
    ncnn::Mat forward(std::string& error, const TransformerInput& input, int step);
#if NCNN_VULKAN
    void step_vulkan(std::string& error, const TransformerInput& input, int step, ncnn::VkMat& images,
        DeviceSampler& sampler, int reference_rows);
#endif
    void clear_cache()
    {
        prefix_cache_.clear();

        kv_cache_.clear();

        metadata_key_ = 0;
        selectors_.release();

        image_times_.release();

        cos_.release();

        sin_.release();

        order_.clear();
    }

private:
    ncnn::Mat forward_internal(std::string& error, const TransformerInput& input, int step
#if NCNN_VULKAN
        ,
        ncnn::VkMat* images, DeviceSampler* sampler, int reference_rows
#endif
    );
    H3ModelSet& models_;
    PrefixCache prefix_cache_;
    KVCache kv_cache_;
    uint64_t metadata_key_ = 0;
    ncnn::Mat selectors_, image_times_, cos_, sin_;
    std::vector<int> order_;
};
}
