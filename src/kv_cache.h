// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "rough_mode.h"
#include "runtime.h"
#include <net.h>
#include <string>
namespace h3image
{
struct KVCacheState;
// One context per H3ModelSet instance; cached tensors belong to the current Transformer.
struct KVCacheContext
{
    KVCacheState* active = nullptr;
};
// Adds parameter-only cache operations while retaining every original weighted layer.
std::string prepare_kv_cache_graph(
    std::string& error, ncnn::Net& net, KVCacheContext& context, const std::string& path);
class KVCache
{
public:
    KVCache(const RuntimeConfig& config, KVCacheContext& context);
    ~KVCache();
    bool enabled() const
    {
        return policy_.enabled();
    }
    int begin(std::string& error, const KVCachePlan& plan, int step, int text_rows, int reference_rows,
        int sequence_rows, uint64_t identity, bool canonical);
    void finish(std::string& error);
    void clear();

private:
    RuntimeConfig config_;
    RoughModePolicy policy_;
    KVCacheContext& context_;
    std::unique_ptr<KVCacheState> state_;
};
}
