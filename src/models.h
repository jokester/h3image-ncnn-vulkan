// h3-image implemented with ncnn library
#pragma once
#include "condition_tables.h"
#include "filesystem_utils.h"
#include "kv_cache.h"
#include "runtime.h"
#include <array>
#include <memory>
#include <net.h>
#include <string>
#include <vector>
namespace h3image
{
class GpuInstance
{
public:
    explicit GpuInstance(std::string& error, bool enabled);
    ~GpuInstance();

private:
    bool enabled_;
};
class H3ModelSet
{
public:
    H3ModelSet() = default;
    bool load(std::string& error, const std::string& directory, RuntimeConfig config);
    ~H3ModelSet();
    void prepare_conditioning(std::string& error);
    // Loaded graph callbacks refer to this instance's cache context.
    H3ModelSet(const H3ModelSet&) = delete;
    H3ModelSet& operator=(const H3ModelSet&) = delete;
    H3ModelSet(H3ModelSet&&) = delete;
    H3ModelSet& operator=(H3ModelSet&&) = delete;
    ncnn::Net* prepare(std::string& error, Component component);
    KVCacheContext& kv_cache_context()
    {
        return kv_cache_context_;
    }
    void unload();
    void begin_generation(std::string& error);
    void configure_memory(std::string& error, const Workload& workload);
    void lock_latent_storage(bool spilled);
    const MemoryPlan& memory_plan() const
    {
        return memory_plan_;
    }
    const MemoryBudgets& budgets() const
    {
        return budgets_;
    }
    const RuntimeConfig& options() const
    {
        return options_;
    }
    const std::string& directory() const
    {
        return directory_;
    }
    std::string model_prefix(Component component) const;
    const ConditionTables* condition_tables() const
    {
        return condition_tables_.get();
    }
    std::string cache_identity(std::string& error, Component component) const;

private:
    bool bf16(Component component) const;
    std::string directory_;
    RuntimeConfig options_;
    MemoryBudgets budgets_;
    uint64_t baseline_host_bytes_ = 0;
    MemoryPlan memory_plan_;
    Workload workload_;
    std::array<ModelMemory, 5> model_memory_;
    std::unique_ptr<ConditionTables> condition_tables_;
    std::string temporary_condition_bank_;
    bool configured_ = false;
    bool latent_storage_locked_ = false;
    bool latents_spilled_ = false;
    KVCacheContext kv_cache_context_;
    std::array<std::unique_ptr<ncnn::Net>, 5> networks_;
    std::unique_ptr<MemoryPressureGuard> memory_guard_;
};
}
