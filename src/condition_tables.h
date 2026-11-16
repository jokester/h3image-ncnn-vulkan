// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "artifact_hash.h"
#include "scheduler.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
namespace h3image
{
// Versioned separately from the optional persistent feature cache.
std::string condition_schedule_identity(std::string& error, const PddScheduler& schedule = PddScheduler());
std::string condition_bank_output_path(const std::string& directory, int steps = 6);
std::string condition_execution_identity(int gpu, int threads);
std::string condition_model_identity(std::string& error, const std::string& directory);
std::string condition_source_identity(std::string& error, const std::string& directory);
std::string condition_producer_prefix(std::string& error, const std::string& directory);
// Compatibility with banks prepared before portable checksum identities.
std::string condition_legacy_model_identity(std::string& error, const std::string& directory);
std::string condition_legacy_source_identity(std::string& error, const std::string& directory);
std::string condition_bank_path(std::string& error, const std::string& directory, int gpu, int steps = 6);
ncnn::Mat pdd_frequencies(std::string& error, int timestep_count, const PddScheduler& schedule = PddScheduler());
class ConditionTables
{
public:
    ConditionTables(std::string& error, const std::string& path, const std::string& backend, const std::string& model,
        const std::string& source = "", bool shared_bf16 = false, const PddScheduler& schedule = PddScheduler(),
        const std::string& legacy_model = "", const std::string& legacy_source = "");
    int step_count() const
    {
        return step_count_;
    }
    ncnn::Mat table(std::string& error, int timestep_count, int layer, int step = -1, bool direct = false) const;

private:
    struct Record
    {
        uint64_t offset = 0;
        int width = 0, height = 0;
        size_t elemsize = 0;
    };
    int step_count_ = 6;
    std::string path_;
    uint64_t file_size_ = 0;
    std::filesystem::file_time_type modified_;
    std::array<std::array<std::array<Record, 51>, 2>, 2> records_ {};
};
class ConditionTableWriter
{
public:
    ConditionTableWriter(std::string& error, const std::string& path, const std::string& backend,
        const std::string& model, const std::string& source = "", const PddScheduler& schedule = PddScheduler());
    ~ConditionTableWriter();
    void append(std::string& error, int timestep_count, int layer, const ncnn::Mat& value, bool direct = false);
    void finish(std::string& error);

private:
    std::string path_, temporary_;
    std::ofstream stream_;
    int count_ = 0;
    int step_count_ = 6;
    bool finished_ = false;
};
}
