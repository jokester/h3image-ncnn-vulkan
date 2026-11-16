// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "artifact_hash.h"
#include <cstdint>
#include <string>
namespace h3image
{
struct TensorShape
{
    int dims = 0, w = 0, h = 1, c = 1;
    uint64_t bytes() const;
    bool matches(const ncnn::Mat& value) const;
};
// Optional, bounded persistent cache of deterministic FP32 tensors. Failures are cache misses.
class ExactCache
{
public:
    ExactCache(std::string directory, uint64_t limit)
        : directory_(std::move(directory))
        , limit_(limit)
    {
    }
    bool get(const Digest& key, const TensorShape& expected, uint64_t max_tensor_bytes, ncnn::Mat& value) const;
    bool put(const Digest& key, const ncnn::Mat& value) const;
    bool enabled() const
    {
        return limit_ != 0 && !directory_.empty();
    }

private:
    std::string directory_;
    uint64_t limit_;
};
}
