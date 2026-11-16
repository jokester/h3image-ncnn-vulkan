// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "models.h"
#include <memory>
#include <string>
namespace h3image
{
// Exact, per-generation reuse of the text refiner and fixed PDD conditioning tables.
// H3's bidirectional attention does not permit cross-step attention K/V reuse.
class PrefixCache
{
public:
    explicit PrefixCache(H3ModelSet& models);
    ~PrefixCache();
    bool prepare(const ncnn::Mat& text, int timestep_count);
    void apply(std::string& error, ncnn::Extractor& extractor, int step) const;
    void clear();
    uint64_t text_identity() const
    {
        return key_;
    }

private:
    struct Impl;
    H3ModelSet& models_;
    std::unique_ptr<Impl> impl_;
    uint64_t key_ = 0;
    int timestep_count_ = 0;
    bool attempted_ = false;
};
}
