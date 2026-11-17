// h3-image implemented with ncnn library
#include "memory_budget.h"
#include "filesystem_utils.h"
#include "h3_config.h"
#include "rough_mode.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
namespace h3image
{
namespace
{
uint64_t sum(std::string& error, std::initializer_list<uint64_t> values)
{
    uint64_t result = 0;
    for (uint64_t value : values)
    {
        if (value > std::numeric_limits<uint64_t>::max() - result)
        {
            error = "resource calculation overflow";
            return {};
        }
        result += value;
    }
    return result;
}
uint64_t product(std::string& error, std::initializer_list<uint64_t> values)
{
    uint64_t result = 1;
    for (uint64_t value : values)
    {
        if (value && result > std::numeric_limits<uint64_t>::max() / value)
        {
            error = "resource calculation overflow";
            return {};
        }
        result *= value;
    }
    return result;
}
uint64_t host_tensor(std::string& error, uint64_t bytes)
{
    return sum(error, { aligned_memory_bytes(error, sum(error, { bytes, 4 }), 64), 64 });
}
MemoryShape tensor(std::initializer_list<uint64_t> dims, uint32_t scalar = 4)
{
    return { std::vector<uint64_t>(dims), scalar, 1 };
}
struct Resources
{
    uint64_t host_load = 0, host_forward = 0, device = 0, host_visible = 0;
    std::vector<GraphMemoryUsage> calls;
    void call(
        std::string& error, const MemoryGraph& graph, const GraphMemoryRequest& request, const AllocationLayout& layout)
    {
        auto usage = plan_graph_memory(error, graph, request, layout);
        if (!error.empty())
            return;

        device = std::max(device, usage.device_pool);
        host_visible = std::max(host_visible, usage.staging);
        const uint64_t host
            = layout.gpu ? sum(error, { usage.staging, usage.host_transfer_scratch }) : usage.host_live_peak;
        if (!error.empty())
            return;

        host_forward = std::max(host_forward, host);
        calls.push_back(std::move(usage));
    }
};
AllocationLayout layout_for(const Workload& w, size_t index, bool gpu)
{
    auto layout = w.allocation_layouts[index == 0 || index == 4 ? 1 : 0];
    layout.gpu = gpu;
    layout.bf16 = index == 0 || index == 4;
    layout.cpu_threads = static_cast<uint32_t>(w.cpu_threads);
    return layout;
}
std::string table_name(int layer)
{
    return layer == kTransformerLayers ? "final_table" : "layer" + std::to_string(layer) + "_table_flat";
}
PrefixCachePlan prefix_footprint(
    std::string& error, const Workload& w, CacheLocation location, const AllocationLayout& layout)
{
    PrefixCachePlan result;
    result.location = w.prefix_cache ? location : CacheLocation::Disabled;
    result.text_tokens = w.text_tokens;
    result.timestep_count = w.reference_rows ? 2 : 1;
    const uint64_t times = static_cast<uint64_t>(result.timestep_count);
    const uint64_t scalar = 2;
    const auto text = tensor({ kTransformerWidth, w.text_tokens }, 2);
    AllocationPool pool(error, 4 * MiB, layout);
    uint64_t host = 0;
    if (result.location != CacheLocation::Disabled)
    {
        if (result.location == CacheLocation::Device)
        {
            pool.allocate(error, text.packed().storage_bytes(error));
            if (!error.empty())
                return {};
        }
        else
            host = host_tensor(error, text.storage_bytes(error));
        result.host_scratch = host_tensor(error, text.storage_bytes(error));
        if (!error.empty())
            return {};

        for (int layer = 0; layer <= kTransformerLayers; layer++)
        {
            const uint64_t width = layer == kTransformerLayers ? 2 * kTransformerWidth : kModulationWidth;
            const uint64_t entry = product(error, { times, width, scalar });
            if (!error.empty())
                return {};

            // Current all-steps layer + one step clone; old layers are persistent.
            result.host_scratch = std::max(result.host_scratch,
                sum(error,
                    { host_tensor(error, product(error, { entry, static_cast<uint64_t>(w.denoise_steps) })),
                        host_tensor(error, entry) }));
            if (!error.empty())
                return {};

            for (int step = 0; step < w.denoise_steps; step++)
            {
                if (result.location == CacheLocation::Device)
                {
                    pool.allocate(error, entry);
                    if (!error.empty())
                        return {};
                }
                else
                    host = sum(error, { host, host_tensor(error, entry) });
            }
            if (result.location == CacheLocation::Device)
            {
                result.host_scratch = std::max(result.host_scratch,
                    sum(error,
                        { host_tensor(error, product(error, { entry, static_cast<uint64_t>(w.denoise_steps) })),
                            host_tensor(error, entry),
                            product(error, { 2, layout.allocation_bytes(error, entry, true) }) }));
                if (!error.empty())
                    return {};
            }
        }
        result.device_bytes = layout.gpu ? pool.allocated_bytes() : 0;
        result.host_bytes = host;
    }
    else if (w.fixed_modulations)
    {
        for (int layer = 0; layer <= kTransformerLayers; layer++)
        {
            const uint64_t entry = product(error,
                { times, static_cast<uint64_t>(layer == kTransformerLayers ? 2 * kTransformerWidth : kModulationWidth),
                    scalar });
            if (!error.empty())
                return {};

            host = sum(error, { host, host_tensor(error, entry) });
            if (!error.empty())
                return {};
        }
        result.host_bytes = host;
        result.host_scratch = host_tensor(error, product(error, { times, kModulationWidth, scalar }));
        if (!error.empty())
            return {};
    }
    return result;
}
KVCachePlan kv_footprint(std::string& error, const Workload& w, CacheLocation location, const AllocationLayout& layout)
{
    KVCachePlan result;
    if (!w.rough_mode)
        return result;
    result.location = location;
    const RoughModePolicy policy(w.rough_mode);

    const uint64_t prefix = sum(error, { w.text_tokens, policy.includes_references() ? w.reference_rows : 0 });
    if (!error.empty())
        return {};

    const uint64_t entry = tensor({ 128, prefix, 56 }, 2).packed().storage_bytes(error);
    if (!error.empty())
        return {};

    AllocationPool pool(error, 4 * MiB, layout);
    for (int i = 0; i < 2 * kTransformerLayers; i++)
    {
        if (location == CacheLocation::Device)
        {
            pool.allocate(error, entry);
            if (!error.empty())
                return {};
        }
        else
            result.host_bytes = sum(error, { result.host_bytes, host_tensor(error, entry) });
    }
    if (location == CacheLocation::Device)
        result.device_bytes = pool.allocated_bytes();
    return result;
}
MemoryGraph rough_resource_graph(
    std::string& error, const MemoryGraph& source, bool reuse, bool host, uint64_t sequence, uint64_t prefix)
{
    MemoryGraph graph;
    for (auto node : source.nodes)
    {
        if (reuse && node.type == "Crop")
            node.type = "ResourceImageRows";
        if (node.type == "SDPA" && node.name.find("_body_") != std::string::npos)
        {
            MemoryNode cache;
            cache.type = "ResourceKV";
            cache.name = node.name + "_resources";
            cache.inputs = { checked_at(error, node.inputs, 1), checked_at(error, node.inputs, 2) };
            if (!error.empty())
                return {};

            cache.outputs = { cache.name + "_k", cache.name + "_v" };
            cache.parameters = { { 0, reuse ? "1" : "0" }, { 1, std::to_string(sequence) }, { 2, host ? "1" : "0" },
                { 3, std::to_string(prefix) } };
            node.inputs[1] = cache.outputs[0];
            node.inputs[2] = cache.outputs[1];
            graph.nodes.push_back(std::move(cache));
        }
        graph.nodes.push_back(std::move(node));
    }
    for (size_t i = 0; i < graph.nodes.size(); i++)
        for (const auto& output : graph.nodes[i].outputs)
            graph.producers.emplace(output, i);
    return graph;
}
uint64_t columns_for_decode(uint64_t width);
uint64_t stage_host_base(std::string& error, const Workload& w, size_t component)
{
    const uint64_t t = w.text_tokens, images = sum(error, { w.target_rows, w.reference_rows });
    if (!error.empty())
        return {};

    const uint64_t n = sum(error, { t, images }), m = w.reference_rows ? 2 : 1;
    if (!error.empty())
        return {};

    const uint64_t prompt = host_tensor(error, product(error, { t, 5120, 4 }));
    if (!error.empty())
        return {};

    uint64_t base = w.baseline_host_bytes;
    if (component == 0 || component == 1)
    {
        base = sum(error,
            { base, w.image_bytes, product(error, { w.vision_patches, 1536, 4 }),
                product(error, { w.vision_patches / 4, 5120, 4, 4 }), product(error, { t, 4, 4 }) });
        if (!error.empty())
            return {};

        if (component == 0)
        {
            base = sum(error,
                { base, product(error, { t, 5120, 4, 4 }), product(error, { t, 3, 4 }), product(error, { t, 64, 4, 2 }),
                    product(error, { t, t, 4 }) });
            if (!error.empty())
                return {};
        }
        else
            base = sum(error,
                { base, product(error, { 1152, 48, 48, 4 }), product(error, { w.max_vision_patches, 1152 + 72, 4 }) });
    }
    else if (component == 2)
    {
        base = sum(error, { base, prompt, w.image_bytes, product(error, { w.reference_rows, 96, 4 }) });
        if (!error.empty())
            return {};
    }
    else if (component == 3)
    {
        const uint64_t width = w.canvas_width;
        const uint64_t columns = columns_for_decode(width);
        base = sum(error,
            { base, prompt, product(error, { w.reference_rows, 96, 4 }),
                product(error, { w.batch, w.target_rows, 96, 4 }), product(error, { w.canvas_pixels, 3, 4 }),
                product(error, { columns + 3, 256, 256, 3, 4 }), product(error, { w.canvas_pixels, 3 }) });
        if (!error.empty())
            return {};

        if (w.output_pixels != w.canvas_pixels)
        {
            base = sum(error,
                { base, product(error, { w.output_width, w.canvas_height, 3, 4 }),
                    product(error, { w.output_pixels, 3 }) });
            if (!error.empty())
                return {};
        }
    }
    else
        base = sum(error,
            { base, prompt, product(error, { w.reference_rows, 96, 4 }), product(error, { n, 3 + 1 + 1, 4 }),
                product(error, { t + images + m, 4 }), product(error, { w.batch, w.target_rows, 96, 4 }),
                product(error, { images, 96, 4 }), product(error, { w.target_rows, 96, 4 }),
                product(error, { m, 256, 4 }), product(error, { n, 3 * m + 128 + 2, 4 }),
                product(error, { images, m, 4 }) });
    if (w.active_stage == static_cast<int>(component))
    {
        // Stage entry is sampled with its actual live inputs. Add allocations that
        // occur after load; take the union with the static workflow inventory.
        uint64_t future = 0;
        if (component == 1)
        {
            future = product(error, { w.vision_patches / 4, 5120, 4, 4 });
            if (!error.empty())
                return {};
        }
        else if (component == 3)
        {
            future = sum(error,
                { product(error, { w.canvas_pixels, 3, 4 }),
                    product(error, { columns_for_decode(w.canvas_width) + 3, 256, 256, 3, 4 }),
                    product(error, { w.canvas_pixels + w.output_pixels, 3 }),
                    product(error, { w.output_width, w.canvas_height, 3, 4 }) });
            if (!error.empty())
                return {};
        }
        else if (component == 4)
        {
            future = sum(error,
                { product(error, { w.batch, w.target_rows, 96, 4 }), product(error, { images, 96, 4 }),
                    product(error, { w.target_rows, 96, 4 }), product(error, { m, 256, 4 }),
                    product(error, { n, 3 * m + 128 + 2, 4 }), product(error, { images, m, 4 }) });
            if (!error.empty())
                return {};
        }

        base = std::max(base, sum(error, { w.active_host_bytes, future }));
        if (!error.empty())
            return {};
    }
    return base;
}
// Number of overlapping 256-pixel windows retained by the native decoder.
uint64_t columns_for_decode(uint64_t width)
{
    if (width <= 256)
        return 1;
    uint64_t count = (width + 255) / 256;
    while (256 * count - 64 * (count - 1) < width)
        count++;
    return count;
}
Resources resources_for(std::string& error, const Workload& w, size_t component, const MemoryGraph& graph,
    const AllocationLayout& layout, const PrefixCachePlan& cache, const KVCachePlan& kv, int tile, bool device_sampler)
{
    Resources result;
    const uint64_t t = w.text_tokens, images = sum(error, { w.target_rows, w.reference_rows });
    if (!error.empty())
        return {};

    const uint64_t n = sum(error, { t, images }), m = w.reference_rows ? 2 : 1;
    if (!error.empty())
        return {};

    GraphMemoryRequest request;
    request.outputs = { "out0" };
    if (component == 0)
    {
        request.inputs = { { "embeddings", tensor({ 5120, t }) }, { "cos", tensor({ 64, t, 1 }) },
            { "sin", tensor({ 64, t, 1 }) }, { "mask", tensor({ t, t }) } };
        for (int i = 0; i < 3; i++)
            request.inputs["deepstack" + std::to_string(i)] = tensor({ 5120, t });
        result.call(error, graph, request, layout);
        if (!error.empty())
            return {};
    }
    else if (component == 1)
    {
        for (const auto& reference : w.references)
        {
            const uint64_t patches = product(error, { reference.vision_grid_width, reference.vision_grid_height });
            if (!error.empty())
                return {};

            request.inputs = { { "in0", tensor({ 1536, patches }) }, { "positions", tensor({ 1152, patches }) },
                { "cos", tensor({ 36, patches, 1 }) }, { "sin", tensor({ 36, patches, 1 }) } };
            request.outputs = { "out0", "deepstack0", "deepstack1", "deepstack2" };
            result.call(error, graph, request, layout);
            if (!error.empty())
                return {};
        }
    }
    else if (component == 2)
    {
        for (const auto& reference : w.references)
        {
            if (tile == 0)
            {
                request.inputs["in0"] = tensor({ reference.vae_width, reference.vae_height, 3 });
                result.call(error, graph, request, layout);
                if (!error.empty())
                    return {};

                continue;
            }
            AllocationLayout host;
            GraphMemoryRequest outer;
            outer.inputs["in0"] = tensor({ reference.vae_width, reference.vae_height, 3 });
            outer.outputs = { "out0" };
            outer.external_operators = true;
            outer.preserve_host_inputs = false;
            auto live = plan_graph_memory(error, graph, outer, host);
            if (!error.empty())
                return {};

            for (const auto& node : graph.nodes)
            {
                if (node.type == "Input" || node.type == "Split")
                    continue;
                GraphMemoryRequest call;
                call.outputs = node.outputs;
                uint64_t host_crop = 0;
                for (const auto& input : node.inputs)
                {
                    call.inputs[input] = input == "in0" ? checked_at(error, outer.inputs, input)
                                                        : checked_at(error, live.output_shapes, input);
                    if (!error.empty())
                        return {};
                }

                if (node.type == "Convolution")
                {
                    auto& input = checked_at(error, call.inputs, node.inputs[0]);
                    if (!error.empty())
                        return {};

                    const auto& output = checked_at(error, live.output_shapes, node.outputs[0]);
                    if (!error.empty())
                        return {};

                    const uint64_t sw = static_cast<uint64_t>(node.integer(error, 3, 1));
                    if (!error.empty())
                        return {};

                    const uint64_t sh = static_cast<uint64_t>(node.integer(error, 13, sw));
                    if (!error.empty())
                        return {};

                    if (output.dimensions[0] > static_cast<uint64_t>(tile) / sw
                        || output.dimensions[1] > static_cast<uint64_t>(tile) / sh)
                    {
                        const uint64_t tw = std::min(output.dimensions[0], static_cast<uint64_t>(tile) / sw);
                        const uint64_t th = std::min(output.dimensions[1], static_cast<uint64_t>(tile) / sh);
                        input.dimensions[0] = (tw - 1) * sw
                            + static_cast<uint64_t>((node.integer(error, 1) - 1) * node.integer(error, 2, 1) + 1);
                        if (!error.empty())
                            return {};

                        input.dimensions[1] = (th - 1) * sh
                            + static_cast<uint64_t>((node.integer(error, 11, node.integer(error, 1)) - 1)
                                    * node.integer(error, 12, node.integer(error, 2, 1))
                                + 1);
                        if (!error.empty())
                            return {};

                        host_crop = host_tensor(error, input.storage_bytes(error));
                        if (!error.empty())
                            return {};
                    }
                }
                auto usage = plan_graph_memory(error, graph, call, layout);
                if (!error.empty())
                    return {};

                const uint64_t scratch
                    = layout.gpu ? sum(error, { usage.staging, usage.host_transfer_scratch }) : usage.host_live_peak;
                if (!error.empty())
                    return {};

                const uint64_t output = host_tensor(error, usage.host_outputs);
                if (!error.empty())
                    return {};

                // The outer executor already accounts for a full output at this node.
                const uint64_t extra = sum(error, { host_crop, scratch })
                    - std::min(sum(error, { host_crop, scratch }), host_crop ? 0 : output);
                if (!error.empty())
                    return {};

                result.host_forward = std::max(
                    result.host_forward, sum(error, { checked_at(error, live.host_node_peaks, node.name), extra }));
                if (!error.empty())
                    return {};

                result.device = std::max(result.device, usage.device_pool);
                result.host_visible = std::max(result.host_visible, usage.staging);
                result.calls.push_back(std::move(usage));
            }
        }
    }
    else if (component == 3)
    {
        const uint64_t width = std::min<uint64_t>(16, w.canvas_width / 16);
        const uint64_t height = std::min<uint64_t>(16, w.canvas_height / 16);
        request.inputs = { { "latent", tensor({ width, height, 24 }) },
            { "cos", tensor({ 32, width * height + 5, 1 }) }, { "sin", tensor({ 32, width * height + 5, 1 }) } };
        result.call(error, graph, request, layout);
        if (!error.empty())
            return {};
    }
    else
    {
        const auto make_request = [&](bool reuse)
        {
            GraphMemoryRequest call;
            const RoughModePolicy rough(w.rough_mode);
            const uint64_t prefix = t + (rough.includes_references() ? w.reference_rows : 0);
            const uint64_t rows = reuse ? n - prefix : n;
            const uint64_t image_rows = reuse && rough.includes_references() ? w.target_rows : images;
            call.inputs = { { "images", tensor({ 96, image_rows }) }, { "text", tensor({ 5120, t }) },
                { "time_frequencies", tensor({ 256, m }) }, { "modality_selector", tensor({ 3 * m, rows }) },
                { "image_time_selector", tensor({ m, image_rows }) }, { "cos", tensor({ 64, rows, 1 }) },
                { "sin", tensor({ 64, rows, 1 }) } };
            if (cache.location != CacheLocation::Disabled || w.fixed_modulations)
            {
                auto& inputs = cache.location == CacheLocation::Device ? call.device_inputs : call.inputs;
                if (cache.location != CacheLocation::Disabled)
                    inputs["text_features"] = tensor({ kTransformerWidth, t }, 2).packed(layout.gpu);
                for (int i = 0; i <= kTransformerLayers; i++)
                    inputs[table_name(i)] = tensor(
                        { static_cast<uint64_t>(i == kTransformerLayers ? 2 * kTransformerWidth : kModulationWidth),
                            m },
                        2)
                                                .packed(layout.gpu);
            }
            if (reuse)
                call.aliases["packed"] = "image_features";
            call.outputs = { "out0" };
            return call;
        };
        if (w.rough_mode)
        {
            const uint64_t prefix = t + (RoughModePolicy(w.rough_mode).includes_references() ? w.reference_rows : 0);

            for (bool reuse : { false, true })
            {
                const auto modified
                    = rough_resource_graph(error, graph, reuse, kv.location == CacheLocation::Host, n, prefix);
                if (!error.empty())
                    return {};

                result.call(error, modified, make_request(reuse), layout);
                if (!error.empty())
                    return {};
            }
        }
        else
            result.call(error, graph, make_request(false), layout);
        if (cache.location != CacheLocation::Disabled)
        {
            GraphMemoryRequest prefill;
            prefill.inputs["text"] = tensor({ 5120, t });
            prefill.outputs = { "text_features" };
            prefill.raw_outputs = true;
            result.call(error, graph, prefill, layout);
            if (!error.empty())
                return {};

            if (!w.fixed_modulations)
            {
                prefill.inputs.clear();

                prefill.inputs["time_frequencies"] = tensor({ 256, m * static_cast<uint64_t>(w.denoise_steps) });
                prefill.outputs = { "time_embedding" };
                result.call(error, graph, prefill, layout);
                if (!error.empty())
                    return {};

                const auto embedding = checked_at(error, result.calls.back().output_shapes, "time_embedding");
                if (!error.empty())
                    return {};

                for (int layer = 0; layer <= kTransformerLayers; layer++)
                {
                    prefill.inputs.clear();

                    prefill.inputs["time_embedding"] = embedding;
                    prefill.outputs = { table_name(layer) };
                    result.call(error, graph, prefill, layout);
                    if (!error.empty())
                        return {};
                }
            }
        }
        result.device = sum(error, { result.device, cache.device_bytes, kv.device_bytes });
        if (!error.empty())
            return {};

        result.host_forward = sum(error, { result.host_forward, cache.host_bytes, kv.host_bytes, cache.host_scratch });
        if (!error.empty())
            return {};

        // Immutable host rotary/selectors survive across steps. Include a second
        // set for the shortened rough-mode view, which can coexist during copies.
        result.host_forward = sum(error,
            { result.host_forward, product(error, { 2, n, 64 * 2 * 4 + m * 3 * 4 }),
                product(error, { 2, images, m, 4 }) });
        if (!error.empty())
            return {};

        if (device_sampler)
        {
            AllocationPool sample_pool(error, 4 * MiB, layout);
            const auto sample_bytes = tensor({ 96, images }).storage_bytes(error);
            if (!error.empty())
                return {};

            sample_pool.allocate(error, sample_bytes);
            if (!error.empty())
                return {};

            result.device = sum(error, { result.device, sample_pool.allocated_bytes() });
            if (!error.empty())
                return {};

            result.host_visible
                = sum(error, { result.host_visible, layout.allocation_bytes(error, sample_bytes, true) });
            if (!error.empty())
                return {};

            result.host_forward = sum(error, { result.host_forward, host_tensor(error, sample_bytes) });
            if (!error.empty())
                return {};
        }
    }
    const uint64_t base = stage_host_base(error, w, component);
    if (!error.empty())
        return {};

    result.host_load = base;
    result.host_forward = sum(error, { result.host_forward, base });
    if (!error.empty())
        return {};

    return result;
}
StageMemoryPlan solve_stage(std::string& error, const MemoryBudgets& budget, const Resources& resources,
    const MemoryGraph& graph, const AllocationLayout& layout)
{
    const auto weights = model_weight_allocations(error, graph, layout);
    if (!error.empty())
        return {};

    if (weights.empty())
    {
        error = "resource graph has no supported model weights";
        return {};
    }
    const auto fits = [&](uint64_t host, uint64_t device)
    {
        return host <= budget.ram && (!layout.gpu || device <= budget.vram)
            && (!layout.gpu || !budget.shared_heap || sum(error, { host, device }) <= budget.ram);
    };
    const auto make = [&](const WeightMemoryUsage& usage, bool paged)
    {
        StageMemoryPlan stage;
        stage.paged = paged;
        stage.device_weights = stage.device_weights_actual = usage.device_resident;
        stage.host_weights = usage.host_resident;
        stage.device_graph_workspace = resources.device;
        stage.device_streaming_reserve = paged ? usage.streaming_device_pool : 0;
        stage.device_workspace = sum(error, { resources.device, stage.device_streaming_reserve });
        stage.host_visible_workspace = std::max(usage.loading_staging_peak,
            sum(error, { resources.host_visible, paged ? usage.streaming_host_staging : 0 }));
        stage.host_loading_scratch = usage.loading_host_scratch;
        stage.host_forward_scratch = resources.host_forward - resources.host_load;
        if (paged)
            stage.host_forward_scratch
                = sum(error, { stage.host_forward_scratch, usage.streaming_host_staging, usage.streaming_host_source });
        const uint64_t forward = sum(error,
            { resources.host_forward, usage.host_resident, paged ? usage.streaming_host_staging : 0,
                paged ? usage.streaming_host_source : 0 });
        const uint64_t load = sum(error, { resources.host_load, usage.loading_host_peak });
        const uint64_t peak = std::max(forward, load);
        stage.host_workspace = peak - std::min(peak, usage.host_resident);
        stage.host_weight_loading_peak = usage.loading_host_peak;
        return stage;
    };
    const auto resident = make(plan_weight_memory(error, weights, resources.calls, layout, 0, 0, true), false);
    if (!error.empty())
        return {};

    if (fits(sum(error, { resident.host_workspace, resident.host_weights }),
            sum(error, { resident.device_workspace, resident.device_weights }))
        && resident.host_visible_workspace <= budget.host_visible
        && (!layout.gpu
            || (resident.device_weights_actual <= budget.device_weight_limit
                && resident.host_weights <= budget.host_weight_limit)))
        return resident;
    if (!layout.gpu)
    {
        error = "CPU model weights and calculated live tensors exceed the RAM budget";
        return {};
    }
    if (!fits(resources.host_forward, resources.device))
    {
        error = "calculated live tensors exceed RAM/VRAM capacity";
        return {};
    }
    uint64_t device_limit = std::min(budget.device_weight_limit, budget.vram - resources.device);
    uint64_t host_limit
        = budget.shared_heap ? 0 : std::min(budget.host_weight_limit, budget.ram - resources.host_forward);
    if (budget.shared_heap)
        device_limit = std::min(device_limit, budget.ram - resources.host_forward - resources.device);
    // Decrease by the computed byte deficit, never by a percentage. Weight tier
    // packing and the streaming pool are recalculated after each membership
    // change.
    for (;;)
    {
        const auto usage = plan_weight_memory(error, weights, resources.calls, layout, device_limit, host_limit);
        if (!error.empty())
            return {};

        auto stage = make(usage, true);
        const uint64_t host = sum(error, { stage.host_workspace, stage.host_weights });
        if (!error.empty())
            return {};

        const uint64_t device = sum(error, { stage.device_workspace, stage.device_weights });
        if (!error.empty())
            return {};

        const uint64_t visible = sum(error, { usage.host_resident, stage.host_visible_workspace });
        if (!error.empty())
            return {};

        if (fits(host, device) && visible <= budget.host_visible)
            return stage;
        const uint64_t device_deficit = device > budget.vram ? device - budget.vram : 0;
        const uint64_t host_deficit = std::max(host > budget.ram ? host - budget.ram : 0,
            visible > budget.host_visible ? visible - budget.host_visible : 0);
        const uint64_t shared_deficit
            = budget.shared_heap && sum(error, { host, device }) > budget.ram ? host + device - budget.ram : 0;
        if (!error.empty())
            return {};

        bool reduced = false;
        const uint64_t reduce_device = std::max(device_deficit, shared_deficit);
        if (reduce_device && device_limit)
        {
            device_limit -= std::min(device_limit, reduce_device);
            reduced = true;
        }
        if (host_deficit && host_limit)
        {
            host_limit -= std::min(host_limit, host_deficit);
            reduced = true;
        }
        if (!reduced)
        {
            error = "calculated allocation/transfer tensors exceed RAM/VRAM capacity";
            return {};
        }
    }
}
} // namespace
const char* component_name(Component component)
{
    static const char* names[] = { "text encoder", "vision encoder", "VAE encoder", "VAE decoder", "transformer" };
    return names[static_cast<size_t>(component)];
}
const char* cache_location_name(CacheLocation location)
{
    return location == CacheLocation::Device ? "VRAM" : location == CacheLocation::Host ? "RAM" : "disabled";
}
uint64_t available_in_memory_limit(uint64_t available, uint64_t limit, uint64_t used, uint64_t reclaimable)
{
    const uint64_t working = used - std::min(used, reclaimable);
    return std::min(available, limit > working ? limit - working : 0);
}
MemoryBudgets select_memory_budgets(const MemorySnapshot& available, uint64_t ram, uint64_t vram, bool gpu)
{
    const auto automatic = [](uint64_t bytes) { return bytes - std::min(bytes / 20, 300 * MiB); };
    MemoryBudgets result { ram ? std::min(ram, available.ram_available) : automatic(available.ram_available),
        gpu ? (vram ? std::min(vram, available.vram_available) : automatic(available.vram_available)) : 0,
        available.shared_heap };
    result.host_visible = available.host_visible_available;
    if (result.shared_heap)
        result.vram = std::min(result.vram, result.ram);
    return result;
}
ModelMemory inspect_model_memory(std::string& error, const std::string& prefix, bool cpu_bf16)
{
    ModelMemory result;
    result.stored_bytes = fsutil::file_size(error, prefix + ".ncnn.bin");
    if (!error.empty())
        return {};

    result.cpu_bf16_storage = cpu_bf16;
    const auto parsed = read_memory_graph(error, prefix + ".ncnn.param");
    if (!parsed || !error.empty())
        return {};
    auto graph = std::make_shared<MemoryGraph>(*parsed);

    inspect_weight_storage(error, prefix + ".ncnn.bin", *graph);
    if (!error.empty())
        return {};

    result.graph = std::move(graph);
    AllocationLayout layout;
    for (const auto& group : model_weight_allocations(error, *result.graph, layout))
        for (uint64_t tensor : group.tensors)
            result.largest_tensor = std::max(result.largest_tensor, tensor);
    if (!result.largest_tensor)
    {
        error = "model graph has no recognized weights: " + prefix;
        return {};
    }
    return result;
}
StageMemoryPlan plan_modulation_memory(std::string& error, const MemoryBudgets& budget, const ModelMemory& model,
    bool gpu, int steps, uint64_t retained, const AllocationLayout& supplied)
{
    if (!model.graph || steps < 1 || steps > 256)
    {
        error = "invalid modulation preparation workload";
        return {};
    }
    auto layout = supplied;
    layout.gpu = gpu;
    layout.bf16 = true;
    Resources resources;
    resources.host_load = retained;
    for (uint64_t times : { 1, 2 })
    {
        GraphMemoryRequest request;
        request.raw_outputs = true;
        request.inputs["time_frequencies"] = tensor({ 256, times * static_cast<uint64_t>(steps) });
        request.outputs = { "time_embedding" };
        resources.call(error, *model.graph, request, layout);
        if (!error.empty())
            return {};

        const auto embedding = checked_at(error, resources.calls.back().output_shapes, "time_embedding");
        if (!error.empty())
            return {};

        for (int layer = 0; layer <= kTransformerLayers; layer += 2)
        {
            request.inputs.clear();

            request.inputs["time_embedding"] = embedding;
            request.outputs = { table_name(layer) };
            if (layer < kTransformerLayers)
                request.outputs.push_back(table_name(layer + 1));
            resources.call(error, *model.graph, request, layout);
            if (!error.empty())
                return {};
        }
    }
    resources.host_forward = sum(error,
        { resources.host_forward, retained,
            product(error, { static_cast<uint64_t>(steps), 2, kModulationWidth, 2, 2 }) });
    if (!error.empty())
        return {};

    return solve_stage(error, budget, resources, *model.graph, layout);
}
MemoryPlan plan_memory(std::string& error, const MemoryBudgets& budget, const Workload& w,
    const std::array<ModelMemory, 5>& models, bool gpu)
{
    if (!RoughModePolicy(w.rough_mode).valid())
    {
        error = "rough mode must be in [0,3]";
        return {};
    }

    if (!w.text_tokens || !w.target_rows || !w.canvas_pixels || !w.canvas_width || !w.canvas_height || !w.batch
        || w.cpu_threads < 1 || w.denoise_steps < 1 || w.denoise_steps > 256)
    {
        error = "invalid generation memory workload";
        return {};
    }
    MemoryPlan result;
    result.budgets = budget;
    for (size_t component = 0; component < models.size(); component++)
    {
        if ((component == 1 || component == 2) && w.references.empty())
            continue;
        if (!models[component].graph)
        {
            error = "model resource graph is required for capacity planning";
            return {};
        }
        const auto& graph = *models[component].graph;
        const auto layout = layout_for(w, component, gpu);
        bool accepted = false;
        std::string reason;
        const auto attempt = [&](CacheLocation prefix, CacheLocation kv, int tile, bool device_sampler = false) -> bool
        {
            std::string attempt_error;
            {
                const auto pc
                    = component == 4 ? prefix_footprint(attempt_error, w, prefix, layout) : PrefixCachePlan {};
                const auto kc = component == 4 ? kv_footprint(attempt_error, w, kv, layout) : KVCachePlan {};
                const auto resources
                    = resources_for(attempt_error, w, component, graph, layout, pc, kc, tile, device_sampler);
                result.stages[component] = solve_stage(attempt_error, budget, resources, graph, layout);
                if (attempt_error.empty())
                {
                    if (component == 4)
                    {
                        result.prefix_cache = pc;
                        result.kv_cache = kc;
                        result.device_sampler = device_sampler;
                    }
                    if (component == 2)
                        result.vae_encoder_tile = tile;
                    return true;
                }
            }
            reason = attempt_error;
            if (reason.find("exceed") == std::string::npos)
                error = reason;
            return false;
        };
        if (component == 4)
        {
            const std::vector<CacheLocation> kvs = !w.rough_mode
                ? std::vector<CacheLocation> { CacheLocation::Disabled }
                : gpu ? std::vector<CacheLocation> { CacheLocation::Device, CacheLocation::Host }
                      : std::vector<CacheLocation> { CacheLocation::Host };
            for (auto kv : kvs)
            {
                for (auto prefix : { CacheLocation::Device, CacheLocation::Host, CacheLocation::Disabled })
                {
                    if (!gpu && prefix == CacheLocation::Device)
                        continue;
                    if (gpu && !w.rough_mode)
                        accepted = attempt(prefix, kv, 256, true);
                    if (accepted || (accepted = attempt(prefix, kv, 256)))
                        break;
                }
                if (accepted)
                    break;
            }
        }
        else if (component == 2)
        {
            for (int tile : { 0, 512, 256, 128, 64 })
                if ((accepted = attempt(CacheLocation::Disabled, CacheLocation::Disabled, tile)))
                    break;
        }
        else
            accepted = attempt(CacheLocation::Disabled, CacheLocation::Disabled, 256);
        if (!accepted)
        {
            if (w.batch > 1)
            {
                auto single = w;
                single.batch = 1;
                auto spilled = plan_memory(error, budget, single, models, gpu);
                if (!error.empty())
                    return {};

                spilled.spill_latents = true;
                return spilled;
            }
            {
                error = std::string(component_name(static_cast<Component>(component))) + ": " + reason;
                return {};
            }
        }
    }
    return result;
}
} // namespace h3image
