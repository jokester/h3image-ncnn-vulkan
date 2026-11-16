// SPDX-License-Identifier: Apache-2.0
#include "kv_cache.h"
#include "filesystem_utils.h"
#include "h3_config.h"
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#if NCNN_VULKAN
#include <command.h>
#include <gpu.h>
#endif
namespace h3image
{
struct KVCacheState
{
    struct Pair
    {
        std::array<ncnn::Mat, 2> host;
#if NCNN_VULKAN
        std::array<ncnn::VkMat, 2> device;
#endif
    };
#if NCNN_VULKAN
    // Destroy tensors before the allocator backing them.
    std::unique_ptr<ncnn::VkBlobAllocator> allocator;
#endif
    std::array<Pair, kTransformerLayers> layers;
    CacheLocation location = CacheLocation::Disabled;
    bool enabled = false, reuse = false, ready = false;
    int text = 0, prefix = 0, sequence = 0, visited = 0, refresh_step = -1;
    uint64_t identity = 0, host_bytes = 0, device_bytes = 0, host_limit = 0, device_limit = 0;
    void visit(std::string& error, int index, int width, int rows, int heads)
    {
        if (index < 0 || index >= kTransformerLayers || index != visited++)
        {
            error = "KV layers executed out of order";
            return;
        }
        if (width != 128 || rows != sequence - (reuse ? prefix : 0) || heads != 56)
        {
            error = "KV rows disagree with the active token sequence";
            return;
        }
    }
    void count_host(std::string& error, const ncnn::Mat& value)
    {
        host_bytes += value.total() * value.elemsize;
        if (host_bytes > host_limit)
        {
            error = "KV cache exceeded its RAM reservation";
            return;
        }
    }
};
namespace
{
KVCacheState* active_state(std::string& error, KVCacheContext* context)
{
    if (!context || !context->active)
    {
        error = "KV execution context is not active";
        return {};
    }
    return context->active;
}
#if NCNN_VULKAN
std::unique_ptr<ncnn::Layer> helper(std::string& error, const char* type, const ncnn::VulkanDevice* device,
    const ncnn::ParamDict& params, const ncnn::Option& options)
{
    std::unique_ptr<ncnn::Layer> layer(ncnn::create_layer_vulkan(type));
    if (!layer)
    {
        error = "cannot create Vulkan KV helper";
        return {};
    }
    layer->vkdev = device;
    if (layer->load_param(params) || layer->create_pipeline(options))
    {
        error = "cannot create Vulkan KV pipeline";
        return {};
    }
    return layer;
}
#endif
class CacheLayer : public ncnn::Layer
{
public:
    explicit CacheLayer(KVCacheContext* value)
        : context(value)
    {
        one_blob_only = support_inplace = false;
        support_bf16_storage = true;
#if NCNN_VULKAN
        support_vulkan = support_vulkan_packing = true;
#endif
    }
    int load_param(const ncnn::ParamDict& params) override
    {
        std::string error;

        index = params.get(0, 0);

        return 0;
    }
    int forward(
        const std::vector<ncnn::Mat>& bottom, std::vector<ncnn::Mat>& top, const ncnn::Option& options) const override
    {
        std::string error;

        auto* state = active_state(error, context);
        if (!error.empty())
        {
            std::fprintf(stderr, "%s\n", error.c_str());
            return -100;
        }

        if (!state)
            return -100;
        auto& s = *state;
        if (!s.enabled)
        {
            top = bottom;
            return 0;
        }
        s.visit(error, index, bottom[0].w, bottom[0].h, bottom[0].c * bottom[0].elempack);
        if (!error.empty())
        {
            std::fprintf(stderr, "%s\n", error.c_str());
            return -100;
        }

        auto& entry = s.layers[index];
        for (int i = 0; i < 2; i++)
        {
            const auto& value = bottom[i];
            if (value.dims != 3 || value.w != 128 || value.elempack != 1 || value.c != 56
                || value.h != s.sequence - (s.reuse ? s.prefix : 0))
            {
                error = "unsupported CPU KV layout";
                std::fprintf(stderr, "%s\n", error.c_str());
                return -100;
            }
            const size_t prefix_bytes = static_cast<size_t>(s.prefix) * value.w * value.elemsize;
            if (!s.reuse)
            {
                auto& cached = entry.host[i];
                cached.create(value.w, s.prefix, value.c, value.elemsize, 1);
                if (cached.empty())
                    return -100;
                for (int c = 0; c < value.c; c++)
                    std::memcpy(cached.channel(c).data, value.channel(c).data, prefix_bytes);
                s.count_host(error, cached);
                if (!error.empty())
                {
                    std::fprintf(stderr, "%s\n", error.c_str());
                    return -100;
                }

                top[i] = value;
            }
            else
            {
                const auto& cached = entry.host[i];
                if (cached.empty() || cached.elemsize != value.elemsize)
                {
                    error = "CPU KV storage type changed";
                    std::fprintf(stderr, "%s\n", error.c_str());
                    return -100;
                }
                top[i].create(value.w, s.sequence, value.c, value.elemsize, 1, options.blob_allocator);
                if (top[i].empty())
                    return -100;
                for (int c = 0; c < value.c; c++)
                {
                    auto* out = static_cast<unsigned char*>(top[i].channel(c).data);
                    std::memcpy(out, cached.channel(c).data, prefix_bytes);
                    std::memcpy(out + prefix_bytes, value.channel(c).data,
                        static_cast<size_t>(value.w) * value.h * value.elemsize);
                }
            }
        }
        return 0;
    }
#if NCNN_VULKAN
    int create_pipeline(const ncnn::Option& options) override
    {
        std::string error;

        if (!options.use_vulkan_compute)
            return 0;
        ncnn::ParamDict params;
        params.set(3, 128);
        params.set(4, 1);
        params.set(5, 56);
        crop = helper(error, "Crop", vkdev, params, options);
        if (!error.empty())
        {
            std::fprintf(stderr, "%s\n", error.c_str());
            return -100;
        }

        params = ncnn::ParamDict();
        params.set(0, 1);
        concat = helper(error, "Concat", vkdev, params, options);
        if (!error.empty())
        {
            std::fprintf(stderr, "%s\n", error.c_str());
            return -100;
        }

        return 0;
    }
    int destroy_pipeline(const ncnn::Option& options) override
    {
        std::string error;

        if (crop)
            crop->destroy_pipeline(options);
        if (concat)
            concat->destroy_pipeline(options);
        crop.reset();
        concat.reset();
        return 0;
    }
    int forward(const std::vector<ncnn::VkMat>& bottom, std::vector<ncnn::VkMat>& top, ncnn::VkCompute& command,
        const ncnn::Option& options) const override
    {
        std::string error;

        auto* state = active_state(error, context);
        if (!error.empty())
        {
            std::fprintf(stderr, "%s\n", error.c_str());
            return -100;
        }

        if (!state)
            return -100;
        auto& s = *state;
        if (!s.enabled)
        {
            top = bottom;
            return 0;
        }
        s.visit(error, index, bottom[0].w, bottom[0].h, bottom[0].c * bottom[0].elempack);
        if (!error.empty())
        {
            std::fprintf(stderr, "%s\n", error.c_str());
            return -100;
        }

        auto& entry = s.layers[index];
        for (int i = 0; i < 2; i++)
        {
            const auto& value = bottom[i];
            if (value.dims != 3 || value.w != 128 || value.h != s.sequence - (s.reuse ? s.prefix : 0)
                || value.c * value.elempack != 56)
            {
                error = "unsupported Vulkan KV layout";
                std::fprintf(stderr, "%s\n", error.c_str());
                return -100;
            }
            if (!s.reuse)
            {
                ncnn::ParamDict params;
                params.set(3, 128);
                params.set(4, s.prefix);
                params.set(5, 56);
                crop->load_param(params);

                ncnn::VkMat prefix;
                auto store = options;
                if (s.location == CacheLocation::Device)
                {
                    store.blob_vkallocator = s.allocator.get();
                }

                if (crop->forward(value, prefix, command, store) || prefix.empty())
                    return -100;
                if (s.location == CacheLocation::Device)
                {
                    entry.device[i] = prefix;
                    s.device_bytes += prefix.buffer_capacity();
                    if (s.device_bytes > s.device_limit)
                    {
                        error = "KV cache exceeded its VRAM reservation";
                        std::fprintf(stderr, "%s\n", error.c_str());
                        return -100;
                    }
                }
                else
                {
                    // Raw clones preserve BF16 bits, packing and channel strides. All downloads
                    // finish with the graph submission, without a CPU wait at each layer.
                    command.record_clone(prefix, entry.host[i], options);
                    if (entry.host[i].empty())
                        return -100;
                    s.count_host(error, entry.host[i]);
                    if (!error.empty())
                        return -100;
                }
                top[i] = value;
            }
            else
            {
                ncnn::VkMat prefix = entry.device[i];
                if (s.location == CacheLocation::Host)
                    command.record_clone(entry.host[i], prefix, options);
                if (prefix.empty())
                    return -100;
                ncnn::VkMat suffix;
                vkdev->convert_packing(value, suffix, prefix.elempack, command, options);
                std::vector<ncnn::VkMat> merged(1);
                if (concat->forward({ prefix, suffix }, merged, command, options))
                    return -100;
                top[i] = merged[0];
            }
        }
        return 0;
    }
    mutable std::unique_ptr<ncnn::Layer> crop, concat;
#endif
    KVCacheContext* context;
    int index = 0;
};
class SelectImages : public ncnn::Layer
{
public:
    explicit SelectImages(KVCacheContext* value)
        : context(value)
    {
        one_blob_only = true;
        support_inplace = false;
        support_bf16_storage = true;
#if NCNN_VULKAN
        support_vulkan = support_vulkan_packing = true;
#endif
    }
    int forward(const ncnn::Mat& bottom, ncnn::Mat& top, const ncnn::Option& options) const override
    {
        std::string error;

        const auto* state = active_state(error, context);
        if (!error.empty())
        {
            std::fprintf(stderr, "%s\n", error.c_str());
            return -100;
        }

        if (!state)
            return -100;
        auto& s = *state;
        if (bottom.dims != 2 || bottom.elempack != 1)
        {
            error = "unsupported CPU image selection layout";
            std::fprintf(stderr, "%s\n", error.c_str());
            return -100;
        }
        top = s.reuse ? bottom : bottom.row_range(s.text, s.sequence - s.text).clone(options.blob_allocator);
        return top.empty() ? -100 : 0;
    }
#if NCNN_VULKAN
    int create_pipeline(const ncnn::Option& options) override
    {
        std::string error;

        if (!options.use_vulkan_compute)
            return 0;
        ncnn::ParamDict params;
        params.set(3, kTransformerWidth);
        params.set(4, 1);
        crop = helper(error, "Crop", vkdev, params, options);
        if (!error.empty())
        {
            std::fprintf(stderr, "%s\n", error.c_str());
            return -100;
        }

        return 0;
    }
    int destroy_pipeline(const ncnn::Option& options) override
    {
        std::string error;

        if (crop)
            crop->destroy_pipeline(options);
        crop.reset();
        return 0;
    }
    int forward(const ncnn::VkMat& bottom, ncnn::VkMat& top, ncnn::VkCompute& command,
        const ncnn::Option& options) const override
    {
        std::string error;

        const auto* state = active_state(error, context);
        if (!error.empty())
        {
            std::fprintf(stderr, "%s\n", error.c_str());
            return -100;
        }

        if (!state)
            return -100;
        auto& s = *state;
        if (s.reuse)
        {
            top = bottom;
            return 0;
        }
        ncnn::ParamDict params;
        params.set(1, s.text);
        params.set(3, kTransformerWidth);
        params.set(4, s.sequence - s.text);
        crop->load_param(params);

        return crop->forward(bottom, top, command, options);
    }
    mutable std::unique_ptr<ncnn::Layer> crop;
#endif
    KVCacheContext* context;
};
ncnn::Layer* create_cache(void* context)
{
    return new CacheLayer(static_cast<KVCacheContext*>(context));
}
ncnn::Layer* create_select(void* context)
{
    return new SelectImages(static_cast<KVCacheContext*>(context));
}
}
std::string prepare_kv_cache_graph(std::string& error, ncnn::Net& net, KVCacheContext& context, const std::string& path)
{
    if (net.register_custom_layer("H3PrefixKV", create_cache, nullptr, &context)
        || net.register_custom_layer("H3SelectImages", create_select, nullptr, &context))
    {
        error = "cannot register KV cache layers";
        return {};
    }
    std::ifstream file(path);
    std::string magic, counts, line;
    std::getline(file, magic);
    std::getline(file, counts);
    int layers = 0, blobs = 0;
    if (magic != "7767517" || !(std::istringstream(counts) >> layers >> blobs) || layers <= 0 || blobs <= 0)
    {
        error = "invalid transformer graph: " + path;
        return {};
    }
    std::ostringstream body;
    int index = 0, selections = 0;
    while (std::getline(file, line))
    {
        if (line.rfind("SDPA layer", 0) == 0)
        {
            std::istringstream in(line);
            std::string type, name, query, key, value, out, rest;
            int inputs = 0, outputs = 0;
            if (!(in >> type >> name >> inputs >> outputs >> query >> key >> value >> out) || inputs != 3
                || outputs != 1)
            {
                error = "unsupported transformer attention layout";
                return {};
            }
            std::getline(in, rest);
            const std::string prefix = "rough_kv_" + std::to_string(index);
            body << "H3PrefixKV " << prefix << " 2 2 " << key << ' ' << value << ' ' << prefix << "_k " << prefix
                 << "_v 0=" << index << '\n';
            body << "SDPA " << name << " 3 1 " << query << ' ' << prefix << "_k " << prefix << "_v " << out << rest
                 << '\n';
            index++;
        }
        else if (line.rfind("Crop select_image_rows ", 0) == 0)
        {
            std::istringstream in(line);
            std::string type, name, first, second, out;
            int inputs = 0, outputs = 0;
            if (!(in >> type >> name >> inputs >> outputs >> first >> second >> out) || inputs != 2 || outputs != 1)
            {
                error = "unsupported transformer image selection";
                return {};
            }
            body << "H3SelectImages " << name << " 1 1 " << first << ' ' << out << '\n';
            selections++;
        }
        else
            body << line << '\n';
    }
    if (index != kTransformerLayers || selections != 1)
    {
        error = "transformer graph does not support rough mode";
        return {};
    }
    return magic + "\n" + std::to_string(layers + index) + " " + std::to_string(blobs + index * 2) + "\n" + body.str();
}
KVCache::KVCache(const RuntimeConfig& config, KVCacheContext& context)
    : config_(config)
    , policy_(config.rough_mode)
    , context_(context)
{
}
KVCache::~KVCache()
{
    clear();
}
void KVCache::clear()
{
    if (state_ && context_.active == state_.get())
        context_.active = nullptr;
    state_.reset();
}
int KVCache::begin(std::string& error, const KVCachePlan& plan, int step, int text, int refs, int sequence,
    uint64_t identity, bool canonical)
{
    if (!enabled())
        return 0;
    const int refresh_step = policy_.refresh_step(error, step);
    if (!error.empty())
        return {};

    if (step == 0)
    {
        clear();
    }

    if (!state_)
        state_ = std::make_unique<KVCacheState>();
    auto& s = *state_;
    const int prefix = text + (policy_.includes_references() ? refs : 0);
    const bool same_shape = s.prefix == prefix && s.sequence == sequence;
    const bool same_placement = s.location == plan.location;
    s.enabled = canonical;
    s.reuse = canonical && step != refresh_step && s.ready && refresh_step == s.refresh_step && same_shape
        && same_placement && s.identity == identity;
    s.text = text;
    s.prefix = prefix;
    s.sequence = sequence;
    s.visited = 0;
    if (canonical && !s.reuse)
    {
        s.ready = false;
        s.layers = {};
#if NCNN_VULKAN
        if (!same_shape || !same_placement)
            s.allocator.reset();
#endif
        s.host_bytes = s.device_bytes = 0;
        s.location = plan.location;
        s.host_limit = plan.host_bytes;
        s.device_limit = plan.device_bytes;
        s.identity = identity;
        // A refresh forced by changed conditioning also starts a new valid cache bank.
        s.refresh_step = refresh_step;
        if (plan.location == CacheLocation::Disabled || prefix <= 0 || prefix >= sequence)
        {
            error = "rough mode has no valid KV cache reservation";
            return {};
        }
        if (plan.location == CacheLocation::Device && config_.gpu < 0)
        {
            error = "device KV cache requires a GPU";
            return {};
        }
#if NCNN_VULKAN
        if (plan.location == CacheLocation::Device && !s.allocator)
            s.allocator = std::make_unique<ncnn::VkBlobAllocator>(ncnn::get_gpu_device(config_.gpu), 4 * MiB);
#endif
    }
    context_.active = &s;
    if (s.enabled)
        std::cerr << "rough KV step " << step + 1 << ": " << (s.reuse ? "reuse" : "refresh") << ", computed rows "
                  << sequence - (s.reuse ? prefix : 0) << '/' << sequence << ", cache "
                  << cache_location_name(s.location) << "\n";
    return s.reuse ? prefix : 0;
}
void KVCache::finish(std::string& error)
{
    if (!enabled() || !state_)
        return;
    auto& s = *state_;
    if (s.enabled)
    {
        if (s.visited != kTransformerLayers)
        {
            error = "incomplete KV graph execution";
            return;
        }
        s.ready = true;
        std::cerr << "rough KV stored: RAM " << s.host_bytes / MiB << " MiB, VRAM " << s.device_bytes / MiB << " MiB\n";
    }
    else
        s.ready = false;
    if (context_.active == &s)
        context_.active = nullptr;
}
}
