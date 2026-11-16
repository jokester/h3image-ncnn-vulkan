// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "filesystem_utils.h"
#include "runtime.h"
#include "scheduler.h"
#include <string>
#if NCNN_VULKAN
#include <pipeline.h>
namespace h3image
{
// The persistent sample is FP32. Model activations may remain packed BF16.
// RNG and posterior sampling stay on the host in their original order.
class DeviceSampler
{
public:
    explicit DeviceSampler(std::string& error, int gpu)
        : device_(ncnn::get_gpu_device(gpu))
        , workspace_(device_, 4 * 1024 * 1024)
        , pipeline_(device_)
    {
        const char* shader = R"glsl(#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0) buffer sample_blob { float sample_data[]; };
layout(binding = 1) readonly buffer velocity_blob { uint velocity_data[]; };
layout(push_constant) uniform parameter {
    int width;
    int rows;
    int reference_rows;
    int velocity_pack;
    int velocity_bits;
    float delta;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= p.width * p.rows || i < p.reference_rows * p.width) return;
    int row = i / p.width;
    int column = i % p.width;
    int j = (row / p.velocity_pack * p.width + column) * p.velocity_pack + row % p.velocity_pack;
    uint bits;
    if (p.velocity_bits == 16) {
        uint pair = velocity_data[j / 2];
        bits = ((pair >> ((j % 2) * 16)) & 65535u) << 16;
    } else bits = velocity_data[j];
    float velocity = uintBitsToFloat(bits);
    precise float update = p.delta * velocity;
    precise float value = sample_data[i] + update;
    sample_data[i] = value;
}
)glsl";
        ncnn::Option options;
        options.use_fp16_packed = options.use_fp16_storage = options.use_fp16_arithmetic = false;
        std::vector<uint32_t> spirv;
        if (ncnn::compile_spirv_module(shader, options, spirv))
        {
            error = "cannot compile FP32 PDD sampler";
            return;
        }
        pipeline_.set_local_size_xyz(64, 1, 1);
        if (pipeline_.create(spirv.data(), spirv.size() * sizeof(uint32_t), {}))
        {
            error = "cannot create FP32 PDD sampler";
            return;
        }
        if (pipeline_.shader_info().push_constant_count != 6 || pipeline_.shader_info().binding_count != 2)
        {
            error = "unexpected PDD shader interface";
            return;
        }
    }
    ncnn::VkMat upload(std::string& error, const ncnn::Mat& images)
    {
        if (images.dims != 2 || images.elembits() != 32 || images.elempack != 1)
        {
            error = "device sample requires FP32 pack1 rows";
            return {};
        }
        ncnn::VkMat result;
        auto options = transfer_options();
        ncnn::VkCompute command(device_);
        command.record_clone(images, result, options);
        if (result.empty() || command.submit_and_wait())
        {
            error = "cannot upload FP32 sample";
            return {};
        }
        return result;
    }
    void record_step(std::string& error, const PddScheduler& schedule, int index, const ncnn::VkMat& velocity,
        int reference_rows, ncnn::VkMat& images, ncnn::VkCompute& command)
    {
        const auto& v = velocity;
        const auto& x = images;
        if (x.dims != 2 || x.elembits() != 32 || x.elempack != 1 || v.dims != 2 || v.w != x.w || v.h * v.elempack != x.h
            || (v.elembits() != 16 && v.elembits() != 32) || reference_rows < 0 || reference_rows > x.h)
        {
            error = "device velocity/sample shape mismatch";
            return;
        }
        std::vector<ncnn::vk_constant_type> constants(6);
        constants[0].i = x.w;
        constants[1].i = x.h;
        constants[2].i = reference_rows;
        constants[3].i = v.elempack;
        constants[4].i = v.elembits();
        constants[5].f = schedule.delta(error, index);
        if (!error.empty())
            return;

        ncnn::VkMat dispatcher;
        dispatcher.w = x.w * x.h;
        dispatcher.h = dispatcher.c = 1;
        command.record_pipeline(&pipeline_, { x, v }, constants, dispatcher);
    }
    void step(std::string& error, const PddScheduler& schedule, int index, const ncnn::VkMat& velocity,
        int reference_rows, ncnn::VkMat& images)
    {
        ncnn::VkCompute command(device_);
        record_step(error, schedule, index, velocity, reference_rows, images, command);
        if (!error.empty())
            return;

        if (command.submit_and_wait())
        {
            error = "device PDD update failed";
            return;
        }
    }
    ncnn::Mat download(std::string& error, const ncnn::VkMat& images)
    {
        ncnn::Mat result;
        ncnn::VkCompute command(device_);
        command.record_clone(images, result, transfer_options());
        if (result.empty() || command.submit_and_wait())
        {
            error = "cannot download FP32 sample";
            return {};
        }
        return result.clone();
    }

private:
    ncnn::Option transfer_options() const
    {
        ncnn::Option options;
        options.blob_vkallocator = workspace_.blob.get();
        options.staging_vkallocator = workspace_.staging.get();
        return options;
    }
    const ncnn::VulkanDevice* device_;
    VulkanWorkspace workspace_;
    ncnn::Pipeline pipeline_;
};
}
#endif
