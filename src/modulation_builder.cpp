// SPDX-License-Identifier: Apache-2.0
#include "modulation_builder.h"
#include "condition_tables.h"
#include "filesystem_utils.h"
#include "h3_config.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gpu.h>
#include <iostream>
#include <memory>
#include <net.h>
#if NCNN_VULKAN
#include <command.h>
#endif
namespace h3image
{
namespace
{
std::string table_name(int layer)
{
    return layer == kTransformerLayers ? "final_table" : "layer" + std::to_string(layer) + "_table_flat";
}
template <class Consumer>
void extract_tables(std::string& error, ncnn::Net& net, const ncnn::Mat& frequencies, Consumer consume)
{
    VulkanWorkspace workspace(net);
    ncnn::Extractor extractor = net.create_extractor();
    workspace.bind(extractor);
    if (extractor.input("time_frequencies", frequencies))
    {
        error = "invalid conditioning frequencies";
        return;
    }
    ncnn::Mat embedding;
#if NCNN_VULKAN
    ncnn::VkMat device_embedding;
    if (net.opt.use_vulkan_compute)
    {
        ncnn::VkCompute command(net.vulkan_device());
        if (extractor.extract("time_embedding", device_embedding, command) || device_embedding.empty()
            || command.submit_and_wait())
        {
            error = "conditioning embedding extraction failed";
            return;
        }
    }
    else
#endif
    {
        ncnn::Mat raw;
        if (extractor.extract("time_embedding", raw, 1) || raw.empty())
        {
            error = "conditioning embedding extraction failed";
            return;
        }
        ncnn::convert_packing(raw, embedding, 1, net.opt);
        if (embedding.empty())
        {
            error = "conditioning embedding unpack failed";
            return;
        }
    }
    // Keep at most two head outputs live, as reserved by the capacity planner.
    for (int first = 0; first <= kTransformerLayers; first += 2)
    {
        int input_result;
#if NCNN_VULKAN
        if (net.opt.use_vulkan_compute)
            input_result = extractor.input("time_embedding", device_embedding);
        else
#endif
            input_result = extractor.input("time_embedding", embedding);
        if (input_result)
        {
            error = "invalid conditioning embedding";
            return;
        }
        const int count = first < kTransformerLayers ? 2 : 1;
        ncnn::Mat values[2];
#if NCNN_VULKAN
        if (net.opt.use_vulkan_compute)
        {
            ncnn::VkCompute command(net.vulkan_device());
            auto options = net.opt;
            options.blob_vkallocator = workspace.blob.get();
            options.staging_vkallocator = workspace.staging.get();
            for (int i = 0; i < count; i++)
            {
                ncnn::VkMat value;
                if (extractor.extract(table_name(first + i).c_str(), value, command) || value.empty())
                {
                    error = "conditioning table extraction failed";
                    return;
                }
                command.record_clone(value, values[i], options);
            }
            if (command.submit_and_wait())
            {
                error = "conditioning table download failed";
                return;
            }
        }
        else
#endif
            for (int i = 0; i < count; i++)
                if (extractor.extract(table_name(first + i).c_str(), values[i], 1))
                {
                    error = "conditioning table extraction failed";
                    return;
                }
        for (int i = 0; i < count; i++)
        {
            ncnn::Mat value;
            ncnn::convert_packing(values[i], value, 1, net.opt);
            if (value.empty() || value.elembits() != 16)
            {
                error = "conditioning producer did not return BF16 storage";
                return;
            }
            consume(first + i, value);

            const auto name = table_name(first + i);
            extractor.input(name.c_str(), ncnn::Mat());
#if NCNN_VULKAN
            if (net.opt.use_vulkan_compute)
                extractor.input(name.c_str(), ncnn::VkMat());
#endif
        }
    }
}
// Direct kernels must use their native 1/2-row shapes. Spool their rows to disk
// so a single extractor per step can serve all heads without retaining 51 tops.
class DirectSpool
{
public:
    explicit DirectSpool(std::string& error, const std::string& base, int steps, int times)
        : path_(base + ".rows." + std::to_string(Clock::now().time_since_epoch().count()))
        , file_(path_, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc)
        , steps_(steps)
        , times_(times)
    {
        if (!file_)
        {
            error = "cannot create modulation row spool";
            return;
        }
    }
    ~DirectSpool()
    {
        file_.close();
        std::error_code error;
        std::filesystem::remove(path_, error);
    }
    void put(std::string& error, int layer, int step, const ncnn::Mat& part)
    {
        const int width = layer == kTransformerLayers ? 2 * kTransformerWidth : kModulationWidth;
        if (part.dims != 2 || part.w != width || part.h != times_ || part.elemsize != 2)
        {
            error = "invalid direct conditioning shape";
            return;
        }
        file_.seekp(offset(layer) + static_cast<uint64_t>(step) * times_ * width * 2);
        write_mat_bytes(error, file_, part);
        if (!error.empty())
            return;
    }
    ncnn::Mat get(std::string& error, int layer)
    {
        const int width = layer == kTransformerLayers ? 2 * kTransformerWidth : kModulationWidth;
        ncnn::Mat value(width, steps_ * times_, static_cast<size_t>(2));
        file_.flush();

        file_.seekg(offset(layer));
        read_mat_bytes(error, file_, value);
        if (!error.empty())
            return {};

        return value;
    }

private:
    uint64_t offset(int layer) const
    {
        return static_cast<uint64_t>(layer) * steps_ * times_ * kModulationWidth * 2;
    }
    std::string path_;
    std::fstream file_;
    int steps_, times_;
};
} // namespace
void build_modulations(std::string& error, const std::string& directory, int gpu, int threads,
    const MemoryBudgets& budget, const std::string& path, const PddScheduler& schedule, uint64_t retained_host)
{
    const std::string prefix = condition_producer_prefix(error, directory);
    if (!error.empty())
        return;

    const auto memory = inspect_model_memory(error, prefix, gpu < 0);
    if (!error.empty())
        return;

    auto current_budget = select_memory_budgets(query_memory(error, gpu), budget.ram, budget.vram, gpu >= 0);
    if (!error.empty())
        return;

    auto plan = plan_modulation_memory(error, current_budget, memory, gpu >= 0, schedule.step_count(), retained_host,
        query_allocation_layout(error, gpu, true, threads));
    if (!error.empty())
        return;

    MemoryPressureGuard guard(current_budget.ram);
    guard.set_budget(
        current_budget.ram, plan.host_weight_loading_peak - std::min(plan.host_weight_loading_peak, plan.host_weights));
    ncnn::Net net;
    for (;;)
    {
        net.opt.num_threads = threads;
        net.opt.use_local_pool_allocator = false;
        net.opt.use_mapped_model_loading = true;
        net.opt.use_fp16_packed = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.opt.use_bf16_packed = true;
        net.opt.use_bf16_storage = true;
#if NCNN_VULKAN
        net.opt.use_vulkan_compute = gpu >= 0;
        if (gpu >= 0)
        {
            net.set_vulkan_device(gpu);
            net.opt.use_weights_in_disk = plan.paged;
            if (plan.paged)
            {
                net.opt.weight_device_memory_limit = plan.device_weights;
                net.opt.weight_host_memory_limit = plan.host_weights;
            }
        }
#endif
        std::cerr << "prepare modulations: " << prefix << ", RAM budget " << current_budget.ram / MiB
                  << " MiB, VRAM budget " << current_budget.vram / MiB << " MiB\n";
        if (net.load_param((prefix + ".ncnn.param").c_str()) || net.load_model((prefix + ".ncnn.bin").c_str()))
        {
            error = "cannot load modulation preparation graph";
            return;
        }
        if (gpu >= 0)
        {
            const auto after = query_memory(error, gpu);
            if (!error.empty())
                return;

            const uint64_t device_deficit
                = plan.device_workspace > after.vram_available ? plan.device_workspace - after.vram_available : 0;
            const uint64_t host_deficit
                = !current_budget.shared_heap && plan.host_visible_workspace > after.host_visible_available
                ? plan.host_visible_workspace - after.host_visible_available
                : 0;
            if (device_deficit || host_deficit)
            {
                const uint64_t old_device = plan.device_weights_actual, old_host = plan.host_weights;
                if ((device_deficit && (!old_device || device_deficit >= current_budget.vram))
                    || (host_deficit && (!old_host || host_deficit >= current_budget.host_visible)))
                {
                    error = "actual Vulkan allocations leave insufficient capacity for "
                            "modulation buffers";
                    return;
                }
                net.clear();

                wait_for_memory_release(error, gpu, after.device_usage - std::min(after.device_usage, old_device),
                    after.vram_available + old_device);
                if (!error.empty())
                    return;

                if (device_deficit)
                    current_budget.device_weight_limit = std::min(
                        current_budget.device_weight_limit, old_device - std::min(old_device, device_deficit));
                if (host_deficit)
                    current_budget.host_weight_limit
                        = std::min(current_budget.host_weight_limit, old_host - std::min(old_host, host_deficit));
                current_budget.vram -= device_deficit;
                current_budget.host_visible -= host_deficit;
                plan = plan_modulation_memory(error, current_budget, memory, true, schedule.step_count(), retained_host,
                    query_allocation_layout(error, gpu, true, threads));
                if (!error.empty())
                    return;

                if (plan.device_weights_actual >= old_device && plan.host_weights >= old_host)
                {
                    error = "Vulkan modulation allocation deficit "
                            "cannot reduce weight residency";
                    return;
                }
                guard.set_budget(current_budget.ram,
                    plan.host_weight_loading_peak - std::min(plan.host_weight_loading_peak, plan.host_weights));
                continue;
            }
        }
        break;
    }
    guard.set_budget(current_budget.ram, plan.host_forward_scratch);
    const std::string backend = condition_execution_identity(gpu, threads),
                      model = condition_model_identity(error, directory);
    if (!error.empty())
        return;

    ConditionTableWriter writer(error, path, backend, model, condition_source_identity(error, directory), schedule);
    if (!error.empty())
        return;
    for (bool direct : { false, true })
        for (int times = 1; times <= 2; times++)
        {
            const auto frequencies = pdd_frequencies(error, times, schedule);
            if (!error.empty())
                return;

            const auto store
                = [&](int layer, const ncnn::Mat& value) -> void { writer.append(error, times, layer, value, direct); };
            if (!direct)
            {
                extract_tables(error, net, frequencies, store);
                if (!error.empty())
                    return;
            }
            else
            {
                DirectSpool spool(error, path, schedule.step_count(), times);
                if (!error.empty())
                    return;

                for (int step = 0; step < schedule.step_count(); step++)
                {
                    extract_tables(error, net, frequencies.row_range(step * times, times).clone(),
                        [&](int layer, const ncnn::Mat& part) { spool.put(error, layer, step, part); });
                    if (!error.empty())
                        return;
                }

                for (int layer = 0; layer <= kTransformerLayers; layer++)
                {
                    store(layer, spool.get(error, layer));
                    if (!error.empty())
                        return;
                }
            }
            std::cerr << "modulations " << (direct ? "direct" : "prefill") << " m=" << times << ": all 51 tables "
                      << "baked\n";
        }
    writer.finish(error);
}
} // namespace h3image
