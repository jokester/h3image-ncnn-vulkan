// SPDX-License-Identifier: Apache-2.0
#include "filesystem_utils.h"
#include "runtime.h"
#include <algorithm>
#include <cmath>
#include <cpu.h>
#include <gpu.h>
#include <layer.h>
#include <map>
#include <numeric>
namespace h3image
{
namespace
{
// Mirrors the cache/tile arithmetic in the pinned ncnn Gemm CPU kernels.
std::array<uint64_t, 3> cpu_tiles(std::string& error, uint64_t m, uint64_t n, uint64_t k, bool bf16, uint64_t cache,
    uint32_t vector, uint32_t threads, uint32_t physical)
{
    const uint64_t sm = std::max<uint32_t>(vector, 2), sk = std::max<uint32_t>(vector, 2);
    const uint64_t sn = bf16 ? std::max<uint32_t>(vector, 1) : vector >= 16 ? 16 : vector >= 4 ? 4 : 1;
    uint64_t tile = static_cast<uint64_t>(std::sqrt(static_cast<float>(cache) / (bf16 ? 8.f : 12.f)));
    uint64_t tm = std::max(sm, tile / sm * sm), tn = std::max(sn, tile / sn * sn), tk = std::max(sk, tile / sk * sk);
    if (k)
    {
        const uint64_t parts = (k + tk - 1) / tk;
        tk = std::min(tk, aligned_memory_bytes(error, (k + parts - 1) / parts, sk));
        if (!error.empty())
            return {};

        if (parts == 1)
        {
            tile = static_cast<uint64_t>(static_cast<float>(cache) / (bf16 ? 4.f : 8.f) / tk);
            tm = std::max(sm, tile / sm * sm);
            tn = std::max(sn, tile / sn * sn);
        }
    }
    tm *= std::min(threads, physical);
    if (m)
    {
        const uint64_t parts = (m + tm - 1) / tm;
        tm = std::min(tm, aligned_memory_bytes(error, (m + parts - 1) / parts, sm));
        if (!error.empty())
            return {};
    }
    if (n)
    {
        const uint64_t parts = (n + tn - 1) / tn;
        tn = std::min(tn, aligned_memory_bytes(error, (n + parts - 1) / parts, sn));
        if (!error.empty())
            return {};
    }
    if (threads > 1)
    {
        tm = std::min(tm, aligned_memory_bytes(error, std::max<uint64_t>(1, tm / threads), sm));
        if (!error.empty())
            return {};
    }

    return { tm, tn, tk };
}
// Native x86 convolution tile rules (Winograd and im2col share the M/K
// cache equations; their N alignment is selected by the kernel family).
std::array<uint64_t, 3> cpu_convolution_tiles(std::string& error, uint64_t m, uint64_t n, uint64_t k, bool winograd,
    uint64_t cache, uint32_t vector, uint32_t threads, uint32_t physical)
{
    const uint64_t lane = std::max<uint32_t>(vector, 2);
    const auto resolve = [&](uint64_t length, uint64_t tile, uint64_t alignment)
    {
        const uint64_t count = (length + tile - 1) / tile;
        return std::min(tile, aligned_memory_bytes(error, (length + count - 1) / count, alignment));
    };
    uint64_t tm, tk;
    if (winograd)
    {
        tm = static_cast<uint64_t>(std::sqrt(static_cast<float>(cache) / 12.f));
        tm = std::max(lane, tm / lane * lane);
        tm *= std::min(threads, physical);
        tm = resolve(m, tm, lane);
        if (threads > 1)
        {
            tm = std::min(tm, aligned_memory_bytes(error, std::max<uint64_t>(1, tm / threads), lane));
            if (!error.empty())
                return {};
        }

        const float free_k = std::sqrt(static_cast<float>(cache) / 4.f) - static_cast<float>(tm);
        tk = std::max(lane, static_cast<uint64_t>(std::max(0.f, free_k)) / lane * lane);
        tk = resolve(k, tk, lane);
    }
    else
    {
        const uint64_t subtract = vector >= 16 ? 64 : vector >= 8 ? 32 : vector >= 4 ? 16 : 2;
        const uint64_t divisor = vector >= 16 ? 16 : vector >= 4 ? 8 : 3;
        const uint64_t initial = cache / 4 > subtract ? (cache / 4 - subtract) / divisor : 0;
        tk = resolve(k, std::max(lane, initial / lane * lane), lane);
        const uint64_t max_m = vector >= 16 ? 64 : vector >= 8 ? 32 : vector >= 4 ? 16 : 8;
        const uint64_t count = (m + max_m - 1) / max_m;
        tm = std::max(lane, aligned_memory_bytes(error, (m + count - 1) / count, lane));
        if (!error.empty())
            return {};

        tm *= std::min(threads, physical);
        tm = resolve(m, tm, lane);
        if (threads > 1)
        {
            tm = std::min(tm, aligned_memory_bytes(error, std::max<uint64_t>(1, tm / threads), lane));
            if (!error.empty())
                return {};
        }
    }
    uint64_t tn = 1;
    if (n)
    {
        const uint64_t align_n = vector >= 16 ? 16 : vector >= 8 ? 12 : vector >= 4 ? 4 : 1;
        const uint64_t divisor = !winograd && tk >= k ? tk : tm + tk;
        const float free_n
            = (static_cast<float>(cache) / 4.f - static_cast<float>(tm * tk)) / static_cast<float>(divisor);
        tn = std::max(align_n, static_cast<uint64_t>(std::max(0.f, free_n)) / align_n * align_n);
        tn = resolve(n, tn, align_n);
    }
    return { tm, tn, tk };
}

}
AllocationLayout query_allocation_layout(std::string& error, int gpu, bool bf16, int threads)
{
    if (threads < 1)
    {
        error = "resource planning requires a positive thread count";
        return {};
    }
    AllocationLayout result;
    result.gpu = gpu >= 0;
    result.bf16 = bf16;
    result.cpu_threads = static_cast<uint32_t>(threads);
    uint32_t vector = 1;
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    vector = ncnn::cpu_support_x86_avx512() ? 16 : ncnn::cpu_support_x86_avx() ? 8 : 4;
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
    vector = 4;
#endif
    result.cpu_vector_width = vector;
    const uint64_t l2 = static_cast<uint64_t>(ncnn::get_cpu_level2_cache_size());
    const uint32_t physical = static_cast<uint32_t>(std::max(1, ncnn::get_physical_cpu_count()));
    result.cpu_l2_cache_bytes = l2;
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    result.cpu_attention_tiled = true;
#endif
    auto operator_storage = std::make_shared<std::map<std::string, bool>>();
    result.cpu_bf16_support = [operator_storage](std::string& resource_error, const std::string& type) -> bool
    {
        auto found = operator_storage->find(type);
        if (found != operator_storage->end())
            return found->second;
        std::unique_ptr<ncnn::Layer> layer(ncnn::create_layer_cpu(type.c_str()));
        if (!layer)
        {
            resource_error = "CPU has no resource kernel for " + type;
            return {};
        }
        const bool supported = layer->support_bf16_storage;
        (*operator_storage)[type] = supported;
        return supported;
    };
    result.cpu_gemm_tile_shape
        = [bf16, l2, vector, threads, physical](std::string& resource_error, uint64_t m, uint64_t n, uint64_t k)
    { return cpu_tiles(resource_error, m, n, k, bf16, l2, vector, static_cast<uint32_t>(threads), physical); };
    result.cpu_convolution_tile_shape =
        [l2, vector, threads, physical](std::string& resource_error, uint64_t m, uint64_t n, uint64_t k, bool winograd)
    {
        return cpu_convolution_tiles(
            resource_error, m, n, k, winograd, l2, vector, static_cast<uint32_t>(threads), physical);
    };
    result.gemm_weight_shape
        = [bf16, l2, vector, threads, physical](std::string& resource_error, uint64_t m, uint64_t n, uint64_t k)
    {
        const auto tile
            = cpu_tiles(resource_error, m, n, k, bf16, l2, vector, static_cast<uint32_t>(threads), physical);
        return std::array<uint64_t, 3> { m ? aligned_memory_bytes(resource_error, m, tile[0]) : 0,
            n ? aligned_memory_bytes(resource_error, n, tile[1]) : 0,
            k ? aligned_memory_bytes(resource_error, k, tile[2]) : 0 };
    };
#if NCNN_VULKAN
    if (gpu >= 0)
    {
        const auto* device = ncnn::get_gpu_device(gpu);
        const auto& info = device->info;
        result.discrete = info.type() == 0;
        result.buffer_alignment = info.buffer_offset_alignment();
        if (info.type() == 1)
        {
            result.buffer_alignment
                = std::lcm(result.buffer_alignment, static_cast<uint64_t>(info.memory_map_alignment()));
            result.buffer_alignment
                = std::lcm(result.buffer_alignment, static_cast<uint64_t>(info.non_coherent_atom_size()));
        }
        if (info.support_VK_KHR_robustness2() || info.support_VK_EXT_robustness2())
            result.buffer_alignment = std::lcm(result.buffer_alignment,
                static_cast<uint64_t>(info.queryRobustness2Properties().robustStorageBufferAccessSizeAlignment));
        result.memory_map_alignment = std::max<uint64_t>(info.memory_map_alignment(), info.non_coherent_atom_size());
        const uint32_t reduction = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
        result.subgroup_reductions = info.subgroup_size() >= 4 && info.subgroup_size() <= 128
            && (info.support_subgroup_ops() & reduction) == reduction;
        const bool cooperative = bf16 && info.support_bf16_cooperative_matrix() && info.vendor_id() != 0x5143;
        result.flash_attention = bf16;
        result.flash_output_limit = 32 * 8; // ncnn's non-cooperative FA tile N.
        if (cooperative)
        {
            const uint32_t required = reduction | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
            result.flash_attention = (info.support_subgroup_ops() & required) == required;
            int m = 0, n = 0, k = 0, subgroup = 0;
            info.get_optimal_cooperative_matrix_mnk(1024, 1024, 1024, VK_COMPONENT_TYPE_BFLOAT16_KHR,
                VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, m, n, k, subgroup);
            if (!m || n % 8 || k % 8 || n != k || subgroup < n)
                result.flash_attention = false;
            else
            {
                result.flash_output_limit = static_cast<uint32_t>(n * 8);
                int sm = 2, wm = 2, pn = std::min(4, subgroup / n);
                for (;;)
                {
                    const uint64_t rows = static_cast<uint64_t>(sm * wm * m);
                    const uint64_t shared = 16 * (rows + static_cast<uint64_t>(pn * n)) * (k / 8)
                        + 4 * rows * static_cast<uint64_t>((pn + 1) * n + 3);
                    const uint32_t invocations = static_cast<uint32_t>(subgroup * wm);
                    if (shared <= info.max_shared_memory_size() && invocations <= info.max_workgroup_invocations()
                        && invocations <= info.max_workgroup_size_x()
                        && static_cast<uint32_t>(wm) <= info.max_compute_workgroup_subgroups())
                        break;
                    if (pn > 1)
                        pn /= 2;
                    else if (wm > 1)
                        wm = 1;
                    else if (sm > 1)
                        sm = 1;
                    else
                    {
                        result.flash_attention = false;
                        break;
                    }
                }
            }
        }
        if (info.vendor_id() == 0x5143)
            result.flash_attention = false;
        result.gemm_weight_shape
            = [device, cooperative](std::string& resource_error, uint64_t raw_m, uint64_t raw_n, uint64_t raw_k)
        {
            if (!cooperative)
                return std::array<uint64_t, 3> { raw_m, raw_n, raw_k };
            const auto& kernel_info = device->info;
            const int m0 = static_cast<int>(raw_m ? raw_m : 1024), n0 = static_cast<int>(raw_n ? raw_n : 1024),
                      k0 = static_cast<int>(raw_k ? raw_k : 1024);
            int m = 0, n = 0, k = 0, subgroup = 0;
            VkComponentTypeKHR accumulator;
            kernel_info.get_optimal_cooperative_matrix_mnk(m0, n0, k0, VK_COMPONENT_TYPE_BFLOAT16_KHR,
                VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, m, n, k, subgroup, accumulator);
            if (!m || (kernel_info.support_VK_KHR_cooperative_matrix() && ((m != 4 && m % 8) || n % 8 || k % 8)))
                return std::array<uint64_t, 3> { raw_m, raw_n, raw_k };
            int sm = std::min((m0 + m - 1) / m, 2), sn = std::min((n0 + n - 1) / n, 2),
                sk = std::min((k0 + k - 1) / k, 2);
            int wm = std::min((m0 + m * sm - 1) / (m * sm), 2), wn = std::min((n0 + n * sn - 1) / (n * sn), 2);
            for (;;)
            {
                const uint64_t a = 16ull * m * (k / 8), b = 16ull * n * (k / 8), c = 16ull * m * (n / 8);
                const uint64_t shared = a * wm * sm * sk + b * wn * sn * sk + c * wm * wn * sm * sn;
                const uint32_t invocations = static_cast<uint32_t>(subgroup * wm * wn);
                if (shared <= kernel_info.max_shared_memory_size()
                    && invocations <= kernel_info.max_workgroup_invocations()
                    && invocations <= kernel_info.max_workgroup_size_x()
                    && static_cast<uint32_t>(wm * wn) <= kernel_info.max_compute_workgroup_subgroups())
                    break;
                if (sk > 1)
                    sk = 1;
                else if (wn > 1)
                    wn = 1;
                else if (wm > 1)
                    wm = 1;
                else if (sn > 1)
                    sn = 1;
                else if (sm > 1)
                    sm = 1;
                else
                    return std::array<uint64_t, 3> { raw_m, raw_n, raw_k };
            }
            return std::array<uint64_t, 3> {
                raw_m ? aligned_memory_bytes(resource_error, raw_m, static_cast<uint64_t>(m * sm * wm)) : 0,
                raw_n ? aligned_memory_bytes(resource_error, raw_n, static_cast<uint64_t>(n * sn * wn)) : 0,
                raw_k ? aligned_memory_bytes(resource_error, raw_k, static_cast<uint64_t>(k)) : 0
            };
        };
        auto cache = std::make_shared<std::map<std::pair<uint64_t, bool>, uint64_t>>();
        result.memory_requirement
            = [device, cache](std::string& resource_error, uint64_t bytes, bool staging) -> uint64_t
        {
            const auto key = std::make_pair(bytes, staging);
            auto known = cache->find(key);
            if (known != cache->end())
                return known->second;
            VkBufferCreateInfo buffer_info {};
            buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer_info.size = bytes;
            buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            if (!staging)
                buffer_info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VkBuffer buffer = VK_NULL_HANDLE;
            if (ncnn::vkCreateBuffer(device->vkdevice(), &buffer_info, nullptr, &buffer) != VK_SUCCESS)
            {
                resource_error = "device cannot represent a calculated resource buffer";
                return {};
            }
            VkMemoryRequirements requirements;
            ncnn::vkGetBufferMemoryRequirements(device->vkdevice(), buffer, &requirements);
            ncnn::vkDestroyBuffer(device->vkdevice(), buffer, nullptr);
            (*cache)[key] = requirements.size;
            return static_cast<uint64_t>(requirements.size);
        };
    }
#else
    if (gpu >= 0)
    {
        error = "cannot plan Vulkan allocations in a CPU-only build";
        return {};
    }
#endif
    return result;
}
}
