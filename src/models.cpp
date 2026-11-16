// h3-image implemented with ncnn library
#include "models.h"
#include "filesystem_utils.h"
#include "modulation_builder.h"
#include <filesystem>
#include <gpu.h>
#include <iostream>
#include <random>
#if NCNN_VULKAN
#include <command.h>
#endif
namespace h3image
{
GpuInstance::GpuInstance(std::string& error, bool enabled)
    : enabled_(enabled)
{
    (void)error;

#if NCNN_VULKAN
    if (enabled_)
        ncnn::create_gpu_instance();
#else
    if (enabled_)
    {
        error = "ncnn was built without Vulkan";
        return;
    }
#endif
}
GpuInstance::~GpuInstance()
{
#if NCNN_VULKAN
    if (enabled_)
        ncnn::destroy_gpu_instance();
#endif
}
bool H3ModelSet::load(std::string& error, const std::string& directory, RuntimeConfig config)
{
    unload();
    memory_guard_.reset();
    condition_tables_.reset();
    if (!temporary_condition_bank_.empty())
    {
        std::error_code code;
        std::filesystem::remove(temporary_condition_bank_, code);
        temporary_condition_bank_.clear();
    }
    configured_ = latent_storage_locked_ = latents_spilled_ = false;
    directory_ = directory;
    options_ = config;

    if (!RoughModePolicy(options_.rough_mode).valid())
    {
        error = "rough mode must be in [0,3]";
        return false;
    }
    if (!error.empty())
        return false;

    const PddScheduler schedule(error, options_.steps);
    if (!error.empty())
        return false;

    if (options_.gpu < 0 && !cpu_bf16_supported())
    {
        error = "CPU BF16 requires an ncnn build with BF16 Gemm support";
        return false;
    }
    if (options_.cache_directory.empty())
        options_.cache_directory = directory_ + "/.cache/h3image-exact";
    for (const char* ext : { ".param", ".bin", ".bin.sha256" })
        if (!fsutil::is_regular_file(error, directory_ + "/transformer/transformer-core.ncnn" + ext))
        {
            error = "missing model file: " + directory_ + "/transformer/transformer-core.ncnn" + ext;
            return false;
        }
    const std::string model = condition_model_identity(error, directory_);
    if (!error.empty())
        return false;

    if (options_.fixed_modulations)
    {
        const std::string bank = condition_bank_path(error, directory_, options_.gpu, options_.steps);
        if (fsutil::exists(error, bank))
        {
            std::string bank_error;
            {
                condition_tables_ = std::make_unique<ConditionTables>(bank_error, bank,
                    condition_execution_identity(options_.gpu, options_.threads), model,
                    condition_source_identity(bank_error, directory_), true, schedule,
                    condition_legacy_model_identity(bank_error, directory_),
                    condition_legacy_source_identity(bank_error, directory_));
                if (bank_error.empty())
                    std::cerr << "fixed modulations: compatible baked tables; reduced "
                                 "transformer core selected\n";
            }
            if (!bank_error.empty())
            {
                condition_tables_.reset();
                std::cerr << "fixed modulation bank unavailable: " << bank_error << "\n";
            }
        }
    }
    if (!condition_tables_)
    {
        const std::string producer = condition_producer_prefix(error, directory_);
        if (!error.empty())
            return false;

        for (const char* ext : { ".ncnn.param", ".ncnn.bin" })
            if (!fsutil::is_regular_file(error, producer + ext))
            {
                error = "online conditioning requires " + producer + ext
                    + "; download transformer/fixed-build/ or use --fixed-modulations 1 with a compatible bank";
                return false;
            }
        std::cerr << (options_.fixed_modulations ? "fixed modulations: no compatible bank; "
                                                 : "fixed modulations: precomputed banks disabled; ")
                  << "online preparation selected for " << options_.steps << " steps\n";
    }
    for (int i = 0; i < 5; i++)
    {
        const std::string prefix = model_prefix(static_cast<Component>(i));
        for (const char* ext : { ".ncnn.param", ".ncnn.bin" })
            if (!fsutil::is_regular_file(error, prefix + ext))
            {
                error = "missing model file: " + prefix + ext;
                return false;
            }
        model_memory_[i] = inspect_model_memory(error, prefix, options_.gpu < 0 && bf16(static_cast<Component>(i)));
        if (!error.empty())
            return false;
    }
    for (const char* name : { "processor/vocab.txt", "processor/merges.txt", "processor/added_tokens.txt",
             "vision/vision_pos_embed.f32", "text_encoder/embedding.bin" })
        if (!fsutil::is_regular_file(error, directory_ + "/" + name))
        {
            error = "missing model file: " + directory_ + "/" + name;
            return false;
        }
    return true;
}

H3ModelSet::~H3ModelSet()
{
    unload();

    condition_tables_.reset();
    if (!temporary_condition_bank_.empty())
    {
        std::error_code error;
        std::filesystem::remove(temporary_condition_bank_, error);
    }
}
void H3ModelSet::prepare_conditioning(std::string& error)
{
    if (condition_tables_)
        return;
    if (!configured_)
    {
        error = "conditioning memory has not been planned";
        return;
    }
    unload();

    const auto begin = Clock::now();
    const PddScheduler schedule(error, options_.steps);
    if (!error.empty())
        return;

    temporary_condition_bank_ = (fsutil::temp_directory_path(error)
        / ("h3-modulations-" + std::to_string(Clock::now().time_since_epoch().count()) + "-"
            + std::to_string(std::random_device {}()) + ".bin"))
                                    .string();
    if (!error.empty())
        return;

    std::cerr << "online conditioning: " << options_.steps << " steps, before DiT loop\n";
    build_modulations(error, directory_, options_.gpu, options_.threads, budgets_, temporary_condition_bank_, schedule,
        query_memory(error, options_.gpu).process_resident);
    if (!error.empty())
        return;

    condition_tables_ = std::make_unique<ConditionTables>(error, temporary_condition_bank_,
        condition_execution_identity(options_.gpu, options_.threads), condition_model_identity(error, directory_),
        condition_source_identity(error, directory_), true, schedule,
        condition_legacy_model_identity(error, directory_), condition_legacy_source_identity(error, directory_));
    if (!error.empty())
        return;

    std::cerr << "online conditioning ready: " << elapsed_ms(begin) << " ms; producer weights released\n";
}
void H3ModelSet::unload()
{
    std::string error;

    uint64_t released = 0;
    bool had_network = false;
    for (size_t i = 0; i < networks_.size(); i++)
        if (networks_[i])
        {
            had_network = true;
            released += memory_plan_.stages[i].device_weights_actual;
        }
    const MemorySnapshot before
        = had_network && options_.gpu >= 0 ? query_memory(error, options_.gpu) : MemorySnapshot {};

    for (auto& net : networks_)
        net.reset();
    if (had_network && options_.gpu >= 0)
    {
        wait_for_memory_release(error, options_.gpu, before.device_usage - std::min(before.device_usage, released),
            before.vram_available + released);
    }

    kv_cache_context_.active = nullptr;
}
void H3ModelSet::begin_generation(std::string& error)
{
    unload();

    memory_guard_.reset();
    configured_ = false;
    latent_storage_locked_ = false;
    const MemorySnapshot available = query_memory(error, options_.gpu);
    if (!error.empty())
        return;

    budgets_ = select_memory_budgets(available, options_.ram_budget, options_.vram_budget, options_.gpu >= 0);
    baseline_host_bytes_ = available.process_resident;
    if (budgets_.ram <= baseline_host_bytes_)
    {
        error = "RAM budget is below the existing runtime working set";
        return;
    }
    memory_guard_ = std::make_unique<MemoryPressureGuard>(budgets_.ram);
    std::cerr << "memory budget: RAM " << budgets_.ram / MiB << " MiB";
    if (options_.gpu >= 0)
        std::cerr << ", VRAM " << budgets_.vram / MiB << " MiB";
    std::cerr << " (available RAM " << available.ram_available / MiB << " MiB";
    if (options_.gpu >= 0)
        std::cerr << ", VRAM " << available.vram_available / MiB << " MiB";
    std::cerr << ")\n";
}
void H3ModelSet::configure_memory(std::string& error, const Workload& workload)
{
    workload_ = workload;
    workload_.baseline_host_bytes = baseline_host_bytes_;
    for (int precision = 0; precision < 2; precision++)
    {
        workload_.allocation_layouts[precision]
            = query_allocation_layout(error, options_.gpu, precision != 0, options_.threads);
        if (!error.empty())
            return;
    }

    workload_.rough_mode = options_.rough_mode;
    workload_.cpu_threads = options_.threads;
    // Both precomputed and online conditioning feed the same split DiT core.
    workload_.fixed_modulations = true;
    workload_.denoise_steps = options_.steps;
    workload_.prefix_cache = workload.prefix_cache && options_.prefix_cache;
    memory_plan_ = plan_memory(error, budgets_, workload_, model_memory_, options_.gpu >= 0);
    if (!error.empty())
        return;

    configured_ = true;
}
void H3ModelSet::lock_latent_storage(bool spilled)
{
    latent_storage_locked_ = true;
    latents_spilled_ = spilled;
    if (spilled)
        workload_.batch = 1;
}
bool H3ModelSet::bf16(Component component) const
{
    return component == Component::Text || component == Component::Transformer;
}
std::string H3ModelSet::model_prefix(Component component) const
{
    static const char* names[] = { "text_encoder/text_encoder", "vision/vision_encoder", "vae/encoder", "vae/decoder",
        "transformer/transformer" };
    if (component == Component::Transformer)
        return directory_ + "/transformer/transformer-core";
    return directory_ + "/" + names[static_cast<size_t>(component)];
}
std::string H3ModelSet::cache_identity(std::string& error, Component component) const
{
    Sha256 hash;
    hash.text("h3-deterministic-features-v1");
    hash.text(execution_identity(options_.gpu, options_.threads));
    const auto include = [&](Component value)
    {
        const std::string prefix = model_prefix(value);
        hash_file_identity(error, hash, prefix + ".ncnn.param");
        hash_file_identity(error, hash, prefix + ".ncnn.bin");
    };
    include(component);
    if (component == Component::Text)
    {
        include(Component::Vision);
        hash_file_identity(error, hash, directory_ + "/text_encoder/embedding.bin");
        if (!error.empty())
            return {};

        hash_file_identity(error, hash, directory_ + "/vision/vision_pos_embed.f32");
        if (!error.empty())
            return {};
    }
    return hex_digest(hash.finish());
}
ncnn::Net* H3ModelSet::prepare(std::string& error, Component component)
{
    const size_t index = static_cast<size_t>(component);
    if (networks_[index])
        return networks_[index].get();
    if (!configured_)
    {
        error = "generation memory has not been planned";
        return {};
    }
    const auto load_begin = Clock::now();
    unload();

    const MemorySnapshot snapshot = query_memory(error, options_.gpu);
    if (!error.empty())
        return {};

    budgets_ = select_memory_budgets(snapshot, options_.ram_budget, options_.vram_budget, options_.gpu >= 0);
    workload_.active_stage = static_cast<int>(index);
    workload_.active_host_bytes = snapshot.process_resident;
    memory_plan_ = plan_memory(error, budgets_, workload_, model_memory_, options_.gpu >= 0);
    if (!error.empty())
        return {};

    if (latent_storage_locked_ && !latents_spilled_ && memory_plan_.spill_latents)
    {
        error = "available RAM decreased after batch latent "
                "allocation; reduce batch size or retry "
                "when memory is available";
        return {};
    }
    auto plan = memory_plan_.stages[index];
    for (;;)
    {
        memory_guard_->set_budget(
            budgets_.ram, plan.host_weight_loading_peak - std::min(plan.host_weight_loading_peak, plan.host_weights));
        auto net = std::make_unique<ncnn::Net>();
        net->opt.num_threads = options_.threads;
        net->opt.use_local_pool_allocator = false;
        net->opt.use_fp16_packed = false;
        net->opt.use_fp16_storage = false;
        net->opt.use_fp16_arithmetic = false;
        net->opt.use_bf16_packed = bf16(component);
        net->opt.use_bf16_storage = bf16(component);
        net->opt.use_mapped_model_loading = true;
        const std::string prefix = model_prefix(component);
        std::cerr << "stage budget " << component_name(component) << ": RAM " << budgets_.ram / MiB << " MiB, VRAM "
                  << budgets_.vram / MiB << " MiB";
        if (options_.gpu >= 0 && !budgets_.shared_heap)
            std::cerr << ", host-visible heap " << budgets_.host_visible / MiB << " MiB";
        std::cerr << "\n";
        std::cerr << "load " << prefix << "\n";
        std::cerr << component_name(component) << " precision: " << (bf16(component) ? "BF16" : "FP32") << " storage, "
                  << (options_.gpu < 0 ? "CPU" : "Vulkan") << "\n";
#if NCNN_VULKAN
        net->opt.use_vulkan_compute = options_.gpu >= 0;
        if (options_.gpu >= 0)
        {
            net->set_vulkan_device(options_.gpu);
            net->opt.use_weights_in_disk = plan.paged;
            if (plan.paged)
            {
                net->opt.weight_device_memory_limit = plan.device_weights;
                net->opt.weight_host_memory_limit = plan.host_weights;
            }
        }
#endif
        std::cerr << component_name(component) << ": " << (plan.paged ? "paged weights" : "resident weights")
                  << ", RAM workspace " << plan.host_workspace / MiB << " MiB, VRAM workspace "
                  << plan.device_workspace / MiB << " MiB, weight caches VRAM/RAM " << plan.device_weights / MiB << "/"
                  << plan.host_weights / MiB << " MiB, streamed-weight pool bound "
                  << plan.device_streaming_reserve / MiB << " MiB, graph/persistent buffers "
                  << plan.device_graph_workspace / MiB << " MiB, host-visible buffers "
                  << plan.host_visible_workspace / MiB << " MiB\n";
        int param_result;
        if (component == Component::Transformer && options_.rough_mode)
        {
            const auto graph = prepare_kv_cache_graph(error, *net, kv_cache_context_, prefix + ".ncnn.param");
            if (!error.empty())
                return {};

            param_result = net->load_param_mem(graph.c_str());
        }
        else
            param_result = net->load_param((prefix + ".ncnn.param").c_str());
        if (param_result || net->load_model((prefix + ".ncnn.bin").c_str()))
        {
            error = "failed to load model: " + prefix;
            return {};
        }
        if (options_.gpu >= 0)
        {
            const auto after = query_memory(error, options_.gpu);
            if (!error.empty())
                return {};

            const uint64_t device_deficit
                = plan.device_workspace > after.vram_available ? plan.device_workspace - after.vram_available : 0;
            const uint64_t host_deficit
                = !budgets_.shared_heap && plan.host_visible_workspace > after.host_visible_available
                ? plan.host_visible_workspace - after.host_visible_available
                : 0;
            if (device_deficit || host_deficit)
            {
                const uint64_t old_device = plan.device_weights_actual, old_host = plan.host_weights;
                if ((device_deficit && (!old_device || device_deficit >= budgets_.vram))
                    || (host_deficit && (!old_host || host_deficit >= budgets_.host_visible)))
                {
                    error = "actual Vulkan allocations leave insufficient capacity for "
                            "calculated buffers";
                    return {};
                }
                const uint64_t target = after.device_usage - std::min(after.device_usage, old_device);
                net.reset();
                wait_for_memory_release(error, options_.gpu, target, after.vram_available + old_device);
                if (!error.empty())
                    return {};

                if (device_deficit)
                    budgets_.device_weight_limit
                        = std::min(budgets_.device_weight_limit, old_device - std::min(old_device, device_deficit));
                if (host_deficit)
                    budgets_.host_weight_limit
                        = std::min(budgets_.host_weight_limit, old_host - std::min(old_host, host_deficit));
                budgets_.vram -= device_deficit;
                budgets_.host_visible -= host_deficit;
                memory_plan_ = plan_memory(error, budgets_, workload_, model_memory_, true);
                if (!error.empty())
                    return {};

                plan = memory_plan_.stages[index];
                if (plan.device_weights_actual >= old_device && plan.host_weights >= old_host)
                {
                    error = "Vulkan allocation deficit cannot be "
                            "recovered by reducing weight residency";
                    return {};
                }
                std::cerr << "Vulkan allocation feedback: device/host heap deficits " << device_deficit << "/"
                          << host_deficit << " bytes; rebuilding the stage\n";
                continue;
            }
        }
        networks_[index] = std::move(net);
        memory_guard_->set_budget(budgets_.ram, plan.host_forward_scratch);
        break;
    }
    std::cerr << "model initialization " << component_name(component) << ": " << elapsed_ms(load_begin) << " ms\n";
    return networks_[index].get();
}
} // namespace h3image
