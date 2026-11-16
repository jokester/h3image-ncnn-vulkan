// SPDX-License-Identifier: Apache-2.0
#include "condition_tables.h"
#include "filesystem_utils.h"
#include "h3_config.h"
#include "scheduler.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>
#ifndef H3_CONDITION_BUILD_ID
#define H3_CONDITION_BUILD_ID "h3-condition-development-v1"
#endif
namespace h3image
{
namespace
{
constexpr char magic[8] = { 'H', '3', 'F', 'I', 'X', 'E', 'D', 3 };
int table_width(int layer)
{
    return layer == kTransformerLayers ? 2 * kTransformerWidth : kModulationWidth;
}
void validate_shape(std::string& error, int times, int layer, int width, int height, size_t elemsize, int steps)
{
    if (times < 1 || times > 2 || layer < 0 || layer > kTransformerLayers || width != table_width(layer)
        || height != steps * times || (elemsize != 2 && elemsize != 4))
    {
        error = "invalid fixed modulation table shape";
        return;
    }
}
}
std::string condition_schedule_identity(std::string& error, const PddScheduler& scheduler)
{
    Sha256 hash;
    constexpr char recipe[] = "pdd-bf16-conditioning-v1";
    hash.update(recipe, sizeof(recipe) - 1);
    const auto integer = [&](uint32_t value)
    {
        unsigned char bytes[4];
        for (int i = 0; i < 4; i++)
            bytes[i] = static_cast<unsigned char>(value >> (8 * i));
        hash.update(bytes, sizeof(bytes));
    };
    const auto number = [&](float value)
    {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        integer(bits);
    };
    for (uint32_t value : { static_cast<uint32_t>(scheduler.step_count()), 256u, 128u, 5376u, 50u, 96768u, 10752u })
    {
        integer(value);
    }

    for (int i = 0; i <= scheduler.step_count(); i++)
    {
        number(scheduler.raw_sigma(error, i));
        if (!error.empty())
            return {};
    }

    number(kReferenceNoiseAug);
    number(10000.f);
    for (int step = 0; step < scheduler.step_count(); step++)
    {
        number(scheduler.timestep(error, step));
        if (!error.empty())
            return {};
    }

    return hex_digest(hash.finish());
}
std::string condition_execution_identity(int gpu, int threads)
{
    // The full cache build key covers more source than the fixed modulation arithmetic.
    // execution_identity's suffix describes the device, driver, threads and CPU ISA.
    std::string execution = execution_identity(gpu, threads);
    execution = execution.substr(execution.find(':') + 1);
    return std::string("pdd6-modulations-v2:bf16:") + H3_CONDITION_BUILD_ID + ":" + execution;
}
std::string condition_legacy_model_identity(std::string& error, const std::string& directory)
{
    Sha256 hash;
    hash.text("H3-fixed-model-v1");
    const std::string root = directory + "/transformer/";
    hash.text(hex_digest(file_digest(error, root + "transformer-core.ncnn.param")));
    if (!error.empty())
        return {};

    hash.text(hex_digest(file_digest(error, root + "transformer-core.ncnn.bin.sha256")));
    if (!error.empty())
        return {};

    const auto size = fsutil::file_size(error, root + "transformer-core.ncnn.bin");
    if (!error.empty())
        return {};

    const auto time = fsutil::last_write_time(error, root + "transformer-core.ncnn.bin").time_since_epoch().count();
    if (!error.empty())
        return {};

    hash.update(&size, sizeof(size));
    hash.update(&time, sizeof(time));
    return hex_digest(hash.finish());
}
std::string condition_legacy_source_identity(std::string& error, const std::string& directory)
{
    const std::string root = directory + "/transformer/transformer";
    if (!fsutil::exists(error, root + ".ncnn.param") && !fsutil::exists(error, root + ".ncnn.bin"))
        return "";
    Sha256 hash;
    hash.text(hex_digest(file_digest(error, root + ".ncnn.param")));
    if (!error.empty())
        return {};

    const auto size = fsutil::file_size(error, root + ".ncnn.bin");
    if (!error.empty())
        return {};

    const auto time = fsutil::last_write_time(error, root + ".ncnn.bin").time_since_epoch().count();
    if (!error.empty())
        return {};

    hash.update(&size, sizeof(size));
    hash.update(&time, sizeof(time));
    return hex_digest(hash.finish());
}

namespace
{
std::string declared_digest(std::string& error, const std::string& path)
{
    std::ifstream in(path);
    std::string digest;
    if (!(in >> digest) || digest.size() != 64)
    {
        error = "missing or invalid model checksum: " + path;
        return {};
    }
    for (char& value : digest)
    {
        if (value >= 'A' && value <= 'F')
            value += 'a' - 'A';
        if (!((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f')))
        {
            error = "invalid model checksum: " + path;
            return {};
        }
    }
    return digest;
}
}
std::string condition_model_identity(std::string& error, const std::string& directory)
{
    const std::string root = directory + "/transformer/transformer-core.ncnn";
    Sha256 hash;
    hash.text("H3-fixed-model-v2");
    hash.text(hex_digest(file_digest(error, root + ".param")));
    if (!error.empty())
        return {};

    hash.text(declared_digest(error, root + ".bin.sha256"));
    if (!error.empty())
        return {};

    hash.text(std::to_string(fsutil::file_size(error, root + ".bin")));
    if (!error.empty())
        return {};

    return hex_digest(hash.finish());
}
std::string condition_source_identity(std::string& error, const std::string& directory)
{
    const std::string root = directory + "/transformer/transformer.ncnn";
    if (!fsutil::exists(error, root + ".param") && !fsutil::exists(error, root + ".bin"))
        return "";
    std::string checksum = root + ".bin.sha256";
    if (!fsutil::exists(error, checksum))
        checksum = directory + "/transformer/fixed-build/source.sha256";
    if (!fsutil::exists(error, checksum))
        checksum = directory + "/transformer/.fixed-build/source.sha256";
    if (!fsutil::exists(error, checksum))
        return condition_legacy_source_identity(error, directory);
    Sha256 hash;
    hash.text("H3-fixed-source-v2");
    hash.text(hex_digest(file_digest(error, root + ".param")));
    if (!error.empty())
        return {};

    hash.text(declared_digest(error, checksum));
    if (!error.empty())
        return {};

    hash.text(std::to_string(fsutil::file_size(error, root + ".bin")));
    if (!error.empty())
        return {};

    return hex_digest(hash.finish());
}

std::string condition_producer_prefix(std::string& error, const std::string& directory)
{
    const std::string visible = directory + "/transformer/fixed-build/modulations";
    if (fsutil::exists(error, visible + ".ncnn.param") && fsutil::exists(error, visible + ".ncnn.bin"))
        return visible;
    const std::string legacy = directory + "/transformer/.fixed-build/modulations";
    if (fsutil::exists(error, legacy + ".ncnn.param") && fsutil::exists(error, legacy + ".ncnn.bin"))
        return legacy;
    return visible;
}

std::string condition_bank_output_path(const std::string& directory, int steps)
{
    return directory + "/transformer/fixed-modulations-bf16" + (steps == 6 ? "" : "-steps" + std::to_string(steps))
        + ".bin";
}
std::string condition_bank_path(std::string& error, const std::string& directory, int gpu, int steps)
{
    (void)gpu;
    const auto shared = condition_bank_output_path(directory, steps);
    if (steps != 6 || fsutil::exists(error, shared))
        return shared;
    // Historical filename; its BF16 payload is shared by both execution backends.
    return directory + "/transformer/fixed-modulations-vulkan.bin";
}
ncnn::Mat pdd_frequencies(std::string& error, int timestep_count, const PddScheduler& scheduler)
{
    if (timestep_count < 1 || timestep_count > 2)
    {
        error = "invalid timestep count";
        return {};
    }
    ncnn::Mat result(256, scheduler.step_count() * timestep_count);
    for (int step = 0; step < scheduler.step_count(); step++)
        for (int row = 0; row < timestep_count; row++)
            for (int j = 0; j < 128; j++)
            {
                const float time = row ? kReferenceNoiseAug : scheduler.timestep(error, step);
                if (!error.empty())
                    return {};

                const float angle = time * std::exp((-std::log(10000.f) * static_cast<float>(j)) / 128.f);
                result.row(step * timestep_count + row)[j] = std::cos(angle);
                result.row(step * timestep_count + row)[128 + j] = std::sin(angle);
            }
    return result;
}
ConditionTables::ConditionTables(std::string& error, const std::string& path, const std::string& backend,
    const std::string& model, const std::string& source, bool shared_bf16, const PddScheduler& schedule,
    const std::string& legacy_model, const std::string& legacy_source)
    : step_count_(schedule.step_count())
    , path_(path)
{
    file_size_ = fsutil::file_size(error, path);
    if (!error.empty())
        return;

    modified_ = fsutil::last_write_time(error, path);
    if (!error.empty())
        return;

    std::ifstream in(path, std::ios::binary);
    char tag[8];
    read_bytes(error, in, tag, 8);
    if (!error.empty())
        return;

    if (std::memcmp(tag, magic, 7) || (tag[7] != 2 && tag[7] != 3 && tag[7] != 4))
    {
        error = "unsupported fixed modulation bank format";
        return;
    }
    const auto producer = read_string(error, in);
    if (!error.empty())
        return;

    const std::string stored_model = read_string(error, in);
    if (!error.empty())
        return;

    const bool legacy = stored_model != model && !legacy_model.empty() && stored_model == legacy_model;
    if ((!shared_bf16 && producer != backend) || (stored_model != model && !legacy) || read_u32(error, in) != 204)
    {
        error = "fixed modulation bank does not match model or execution backend";
        return;
    }
    const std::string stored_source = read_string(error, in);
    if (!error.empty())
        return;

    const std::string& expected_source = legacy ? legacy_source : source;
    if (!expected_source.empty() && expected_source != stored_source)
    {
        error = "original checkpoint changed since fixed modulation preparation";
        return;
    }
    if (tag[7] == 4 && read_u32(error, in) != static_cast<uint32_t>(step_count_))
    {
        error = "fixed modulation bank step count mismatch";
        return;
    }
    if (tag[7] >= 3)
    {
        if (read_string(error, in) != condition_schedule_identity(error, schedule))
        {
            error = "fixed modulation bank schedule does not match this runtime";
            return;
        }
    }
    else if (shared_bf16)
    {
        if (!schedule.canonical())
        {
            error = "legacy bank requires canonical six-step schedule";
            return;
        }
        // Existing v2 Vulkan banks have no separate schedule field. Accept only the
        // standard H3 six-step recipe whose schedule code is known here.
        if (condition_schedule_identity(error) != "1f22e9e8eb77f0c09fe64acc58fef80f8f685fd2c008ce75da0aaa5acacf5964"
            || producer.rfind("pdd6-modulations-v1:bf16-gpu-fp32-cpu:", 0) != 0)
        {
            error = "legacy fixed modulation bank has no compatible schedule identity";
            return;
        }
    }
    for (int i = 0; i < 204; i++)
    {
        const int times = static_cast<int>(read_u32(error, in)), layer = static_cast<int>(read_u32(error, in));
        if (!error.empty())
            return;

        const int width = static_cast<int>(read_u32(error, in)), height = static_cast<int>(read_u32(error, in));
        if (!error.empty())
            return;

        const size_t elemsize = read_u32(error, in);
        if (!error.empty())
            return;

        const uint64_t bytes = read_u64(error, in);
        if (!error.empty())
            return;

        validate_shape(error, times, layer, width, height, elemsize, step_count_);
        if (!error.empty())
            return;

        if (shared_bf16 && elemsize != 2)
        {
            error = "shared fixed modulation bank must contain native BF16 tensors";
            return;
        }
        if (times != 1 + (i / 51) % 2 || layer != i % 51 || bytes != uint64_t(width) * height * elemsize)
        {
            error = "invalid fixed modulation bank layout";
            return;
        }
        Digest checksum;
        read_bytes(error, in, checksum.data(), checksum.size());
        if (!error.empty())
            return;

        records_[i / 102][times - 1][layer] = { static_cast<uint64_t>(in.tellg()), width, height, elemsize };
        // Validate every stored tensor before selecting the reduced runtime model.
        ncnn::Mat value(width, height, elemsize);
        if (value.empty())
        {
            error = "cannot allocate fixed table validation scratch";
            return;
        }
        read_mat_bytes(error, in, value);
        if (!error.empty())
            return;

        if (mat_digest(error, value) != checksum)
        {
            error = "fixed modulation table checksum mismatch";
            return;
        }
    }
    if (in.peek() != std::char_traits<char>::eof())
    {
        error = "trailing fixed modulation data";
        return;
    }
    if (fsutil::file_size(error, path_) != file_size_ || fsutil::last_write_time(error, path_) != modified_)
    {
        error = "fixed modulation bank changed during validation";
        return;
    }
}
ncnn::Mat ConditionTables::table(std::string& error, int times, int layer, int step, bool direct) const
{
    if (times < 1 || times > 2 || layer < 0 || layer > 50 || step < -1 || step >= step_count_)
    {
        error = "invalid fixed modulation table selection";
        return {};
    }
    if (fsutil::file_size(error, path_) != file_size_ || fsutil::last_write_time(error, path_) != modified_)
    {
        error = "fixed modulation bank changed after validation";
        return {};
    }
    const auto& r = records_[direct ? 1 : 0][times - 1][layer];
    ncnn::Mat value(r.width, step < 0 ? r.height : times, r.elemsize);
    if (value.empty())
    {
        error = "cannot allocate fixed modulation table";
        return {};
    }
    std::ifstream in(path_, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(r.offset + (step < 0 ? 0 : uint64_t(step) * times * r.width * r.elemsize)));
    read_mat_bytes(error, in, value);
    if (!error.empty())
        return {};

    if (fsutil::file_size(error, path_) != file_size_ || fsutil::last_write_time(error, path_) != modified_)
    {
        error = "fixed modulation bank changed during table read";
        return {};
    }
    return value;
}
ConditionTableWriter::ConditionTableWriter(std::string& error, const std::string& path, const std::string& backend,
    const std::string& model, const std::string& source, const PddScheduler& schedule)
    : path_(path)
    , temporary_(path + ".tmp." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "."
          + std::to_string(std::random_device {}()))
    , stream_(temporary_, std::ios::binary | std::ios::trunc)
{
    if (!stream_)
    {
        error = "cannot write fixed modulation bank";
        return;
    }
    step_count_ = schedule.step_count();
    char tag[8];
    std::memcpy(tag, magic, 8);
    if (step_count_ != 6)
        tag[7] = 4;
    stream_.write(tag, 8);
    write_string(stream_, backend);
    write_string(stream_, model);
    write_u32(stream_, 204);
    write_string(stream_, source);
    if (step_count_ != 6)
        write_u32(stream_, static_cast<uint32_t>(step_count_));
    write_string(stream_, condition_schedule_identity(error, schedule));
    if (!error.empty())
        return;
}
ConditionTableWriter::~ConditionTableWriter()
{
    stream_.close();
    if (!finished_)
    {
        std::error_code e;
        std::filesystem::remove(temporary_, e);
    }
}
void ConditionTableWriter::append(std::string& error, int times, int layer, const ncnn::Mat& value, bool direct)
{
    validate_shape(error, times, layer, value.w, value.h, value.elemsize, step_count_);
    if (!error.empty())
        return;

    if (value.dims != 2 || value.elempack != 1 || times != 1 + (count_ / 51) % 2 || layer != count_ % 51
        || count_ >= 204 || direct != (count_ >= 102))
    {
        error = "fixed modulation tables must be written in order";
        return;
    }
    write_u32(stream_, times);
    write_u32(stream_, layer);
    write_u32(stream_, value.w);
    write_u32(stream_, value.h);
    write_u32(stream_, static_cast<uint32_t>(value.elemsize));
    write_u64(stream_, uint64_t(value.w) * value.h * value.elemsize);
    const auto digest = mat_digest(error, value);
    if (!error.empty())
        return;

    stream_.write(reinterpret_cast<const char*>(digest.data()), digest.size());
    write_mat_bytes(error, stream_, value);
    if (!error.empty())
        return;

    count_++;
}
void ConditionTableWriter::finish(std::string& error)
{
    if (count_ != 204)
    {
        error = "incomplete fixed modulation bank";
        return;
    }
    stream_.close();
    if (!stream_)
    {
        error = "cannot close fixed modulation bank";
        return;
    }
#if _WIN32
    // Windows rename cannot replace an existing target. A crash here falls back to the full model.
    fsutil::remove(error, path_);
#endif
    fsutil::rename(error, temporary_, path_);
    if (!error.empty())
        return;

    finished_ = true;
}
}
