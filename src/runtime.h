// h3-image implemented with ncnn library
#pragma once
#include "memory_budget.h"
#include <chrono>
#include <memory>
#include <net.h>
#include <string>
namespace h3image
{
// A forward's Vulkan allocations are released after its extractor and commands.
// Persistent cache and sample buffers have their own workspaces.
struct VulkanWorkspace
{
    explicit VulkanWorkspace(const ncnn::Net& net, size_t block_size = 16 * 1024 * 1024)
    {
#if NCNN_VULKAN
        if (net.opt.use_vulkan_compute)
        {
            blob = std::make_unique<ncnn::VkBlobAllocator>(net.vulkan_device(), block_size);
            staging = std::make_unique<ncnn::VkStagingAllocator>(net.vulkan_device());
        }
#else
        (void)net;
        (void)block_size;
#endif
    }
#if NCNN_VULKAN
    explicit VulkanWorkspace(const ncnn::VulkanDevice* device, size_t block_size)
        : blob(std::make_unique<ncnn::VkBlobAllocator>(device, block_size))
        , staging(std::make_unique<ncnn::VkStagingAllocator>(device))
    {
    }
    std::unique_ptr<ncnn::VkBlobAllocator> blob;
    std::unique_ptr<ncnn::VkStagingAllocator> staging;
#endif
    void bind(ncnn::Extractor& extractor) const
    {
#if NCNN_VULKAN
        if (blob)
        {
            extractor.set_blob_vkallocator(blob.get());
            extractor.set_workspace_vkallocator(blob.get());
            extractor.set_staging_vkallocator(staging.get());
        }
#else
        (void)extractor;
#endif
    }
};
struct RuntimeConfig
{
    int gpu = -1;
    int threads = 8;
    uint64_t ram_budget = 0;
    uint64_t vram_budget = 0;
    bool prefix_cache = true;
    int rough_mode = 0;
    int steps = 6;
    bool fixed_modulations = true;
    std::string cache_directory;
    uint64_t cache_limit = 512 * MiB;
};
using Clock = std::chrono::steady_clock;
inline double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
bool cpu_bf16_supported();
uint64_t process_resident_memory();
MemorySnapshot query_memory(std::string& error, int gpu);
MemorySnapshot wait_for_memory_release(
    std::string& error, int gpu, uint64_t expected_device_usage, uint64_t expected_free = 0);
AllocationLayout query_allocation_layout(std::string& error, int gpu, bool bf16, int threads);
class MemoryPressureGuard
{
public:
    explicit MemoryPressureGuard(uint64_t budget);
    ~MemoryPressureGuard();
    void set_budget(uint64_t budget, uint64_t transient_reserve = 0);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
ncnn::Mat unpack_float32(std::string& error, const ncnn::Mat& value);
}
