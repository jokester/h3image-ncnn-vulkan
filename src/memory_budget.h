// h3-image implemented with ncnn library
#pragma once
#include "shape_memory.h"
#include <array>
#include <cstdint>
#include <limits>
#include <string>
namespace h3image
{
inline constexpr uint64_t MiB = 1024ull * 1024;
enum class Component
{
    Text,
    Vision,
    VAEEncoder,
    VAEDecoder,
    Transformer
};
struct MemorySnapshot
{
    uint64_t ram_available = 0;
    uint64_t vram_available = 0;
    bool shared_heap = false;
    uint64_t process_resident = 0;
    uint64_t process_anonymous = 0;
    uint64_t device_usage = 0;
    uint64_t device_budget = 0;
    uint64_t host_visible_available = std::numeric_limits<uint64_t>::max();
};
struct MemoryBudgets
{
    uint64_t ram = 0;
    uint64_t vram = 0;
    bool shared_heap = false;
    uint64_t host_visible = std::numeric_limits<uint64_t>::max();
    // Feedback caps physical weight residency, including allocator block tails.
    uint64_t device_weight_limit = std::numeric_limits<uint64_t>::max();
    uint64_t host_weight_limit = std::numeric_limits<uint64_t>::max();
};
struct Workload
{
    struct ReferenceShape
    {
        uint64_t vae_width = 0, vae_height = 0;
        uint64_t vision_grid_width = 0, vision_grid_height = 0;
    };
    uint64_t text_tokens = 0;
    uint64_t target_rows = 0;
    uint64_t reference_rows = 0;
    uint64_t vision_patches = 0;
    uint64_t max_vision_patches = 0;
    uint64_t reference_pixels = 0;
    uint64_t max_reference_pixels = 0;
    uint64_t image_bytes = 0;
    uint64_t canvas_pixels = 0;
    uint64_t output_pixels = 0;
    uint64_t batch = 1;
    bool prefix_cache = true;
    bool fixed_modulations = false;
    int rough_mode = 0;
    int cpu_threads = 8;
    int denoise_steps = 6;
    uint64_t canvas_width = 0, canvas_height = 0, output_width = 0, output_height = 0;
    uint64_t baseline_host_bytes = 0;
    int active_stage = -1;
    uint64_t active_host_bytes = 0;
    std::vector<ReferenceShape> references;
    std::array<AllocationLayout, 2> allocation_layouts;
};
struct ModelMemory
{
    uint64_t stored_bytes = 0;
    uint64_t largest_tensor = 0; // Largest FP32 source tensor, retained for model metadata validation.
    bool cpu_bf16_storage = false;
    std::shared_ptr<const MemoryGraph> graph;
};
struct StageMemoryPlan
{
    uint64_t host_workspace = 0;
    uint64_t device_workspace = 0;
    uint64_t device_weights = 0;
    uint64_t host_weights = 0;
    // Calculated streamed-weight pool bound, included in device_workspace.
    uint64_t device_streaming_reserve = 0;
    bool paged = false;
    uint64_t device_graph_workspace = 0;
    uint64_t host_loading_scratch = 0;
    uint64_t host_forward_scratch = 0;
    uint64_t host_weight_loading_peak = 0;
    uint64_t device_weights_actual = 0;
    uint64_t host_visible_workspace = 0;
};
enum class CacheLocation
{
    Disabled,
    Host,
    Device
};
struct PrefixCachePlan
{
    CacheLocation location = CacheLocation::Disabled;
    uint64_t text_tokens = 0;
    int timestep_count = 0;
    uint64_t host_bytes = 0;
    uint64_t device_bytes = 0;
    uint64_t host_scratch = 0;
    uint64_t device_scratch = 0;
};
struct KVCachePlan
{
    CacheLocation location = CacheLocation::Disabled;
    uint64_t host_bytes = 0;
    uint64_t device_bytes = 0;
    uint64_t host_scratch = 0;
    uint64_t device_scratch = 0;
};
struct MemoryPlan
{
    MemoryBudgets budgets;
    std::array<StageMemoryPlan, 5> stages;
    PrefixCachePlan prefix_cache;
    KVCachePlan kv_cache;
    // Zero selects budgeted whole-graph VAE encoding.
    int vae_encoder_tile = 256;
    bool device_sampler = false;
    bool spill_latents = false;
};
uint64_t available_in_memory_limit(
    uint64_t host_available, uint64_t limit, uint64_t used, uint64_t reclaimable_file_cache);
MemoryBudgets select_memory_budgets(
    const MemorySnapshot& available, uint64_t ram_request, uint64_t vram_request, bool gpu);
MemoryPlan plan_memory(std::string& error, const MemoryBudgets& budgets, const Workload& workload,
    const std::array<ModelMemory, 5>& models, bool gpu);
StageMemoryPlan plan_modulation_memory(std::string& error, const MemoryBudgets& budgets, const ModelMemory& model,
    bool gpu, int steps, uint64_t retained_host = 0, const AllocationLayout& layout = AllocationLayout {});
ModelMemory inspect_model_memory(std::string& error, const std::string& prefix, bool cpu_bf16_storage = false);
const char* component_name(Component component);
const char* cache_location_name(CacheLocation location);
} // namespace h3image
