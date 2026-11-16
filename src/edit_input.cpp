// h3-image implemented with ncnn library
#include "edit_input.h"
#include "filesystem_utils.h"
#include "h3_config.h"
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
namespace h3image
{
int round_to_multiple(double value, int multiple)
{
    return std::max(multiple, (int)std::nearbyint(value / multiple) * multiple);
}

std::pair<int, int> bounded_short_edge(int width, int height, int min_edge, int max_edge)
{
    const int short_edge = std::min(width, height);
    if (short_edge >= min_edge && short_edge <= max_edge)
        return { width, height };
    const double scale = (double)(short_edge < min_edge ? min_edge : max_edge) / short_edge;
    return { round_to_multiple(width * scale, 32), round_to_multiple(height * scale, 32) };
}

static std::pair<int, int> vision_resize_size(std::string& error, int source_width, int source_height, int short_edge)
{
    if (short_edge <= 0 || short_edge % 32)
    {
        error = "image short-edge sizes must be positive multiples of 32";
        return {};
    }
    const auto size = bounded_short_edge(source_width, source_height, short_edge, short_edge);
    int width = round_to_multiple(size.first, 32);
    int height = round_to_multiple(size.second, 32);
    constexpr double max_vision_pixels = 16777216.0;
    if ((double)width * height > max_vision_pixels)
    {
        const double scale = std::sqrt(max_vision_pixels / ((double)width * height));
        width = round_to_multiple(width * scale, 32);
        height = round_to_multiple(height * scale, 32);
    }
    return { width, height };
}

ncnn::Mat resize_for_short_edge(std::string& error, const ncnn::Mat& image, int short_edge)
{
    const auto size = vision_resize_size(error, image.w, image.h, short_edge);
    if (!error.empty())
        return {};

    return resize_rgb_lanczos(error, image, size.first, size.second);
}

ncnn::Mat crop_rgb_to_multiple(std::string& error, const ncnn::Mat& image, int multiple)
{
    const int width = image.w / multiple * multiple;
    const int height = image.h / multiple * multiple;
    if (width <= 0 || height <= 0)
    {
        error = "image is too small for the VAE scale factor";
        return {};
    }
    if (width == image.w && height == image.h)
        return image;
    ncnn::Mat cropped(width, height, static_cast<size_t>(3), 3);
    if (cropped.empty())
    {
        error = "cannot allocate cropped RGB image";
        return {};
    }
    for (int y = 0; y < height; y++)
        std::memcpy(static_cast<unsigned char*>(cropped.data) + (size_t)y * width * 3,
            static_cast<unsigned char*>(image.data) + (size_t)y * image.w * 3, (size_t)width * 3);
    return cropped;
}

ncnn::Mat image_net_pixels(std::string& error, const ncnn::Mat& image)
{
    if (!is_rgb_image(image))
    {
        error = "invalid image for VAE encoder";
        return {};
    }
    ncnn::Mat pixels
        = ncnn::Mat::from_pixels(static_cast<unsigned char*>(image.data), ncnn::Mat::PIXEL_RGB, image.w, image.h);
    // Scale's fused multiply/add changes rounding at the trained VAE boundary.
    // All byte values have an exact lookup, so pixel conversion can use ncnn's
    // SIMD path without changing the original normalization arithmetic.
    static const auto table = []
    {
        std::array<std::array<float, 256>, 3> values {};
        constexpr float mean[3] = { .485f, .456f, .406f };
        constexpr float stddev[3] = { .229f, .224f, .225f };
        for (int c = 0; c < 3; c++)
            for (int value = 0; value < 256; value++)
                values[c][value] = (value / 255.f - mean[c]) / stddev[c];
        return values;
    }();
    for (int c = 0; c < 3; c++)
    {
        float* channel = pixels.channel(c);
        for (size_t i = 0; i < static_cast<size_t>(image.w) * image.h; i++)
            channel[i] = table[c][static_cast<unsigned int>(channel[i])];
    }
    return pixels;
}

ncnn::Mat vision_patch_pixels(std::string& error, const std::vector<Reference>& references, ncnn::Mat& grids)
{
    int total_patches = 0;
    grids.create(3, (int)references.size());
    for (size_t i = 0; i < references.size(); i++)
    {
        const auto& image = references[i].vision_image;
        if (!is_rgb_image(image) || image.w % 32 || image.h % 32)
        {
            error = "vision image dimensions must be multiples of 32";
            return {};
        }
        const int grid_h = image.h / kVisionPatch;
        const int grid_w = image.w / kVisionPatch;
        if (grid_h % kVisionMerge || grid_w % kVisionMerge)
        {
            error = "vision patch grids must be divisible by the merge size";
            return {};
        }
        grids.row((int)i)[0] = 1.f;
        grids.row((int)i)[1] = (float)grid_h;
        grids.row((int)i)[2] = (float)grid_w;
        total_patches += grid_h * grid_w;
    }
    ncnn::Mat pixels(kVisionPixelWidth, total_patches);
    int patch_offset = 0;
    for (const auto& reference : references)
    {
        const ncnn::Mat& image = reference.vision_image;
        const int grid_h = image.h / kVisionPatch;
        const int grid_w = image.w / kVisionPatch;
        for (int block_y = 0; block_y < grid_h / kVisionMerge; block_y++)
            for (int block_x = 0; block_x < grid_w / kVisionMerge; block_x++)
                for (int dy = 0; dy < kVisionMerge; dy++)
                    for (int dx = 0; dx < kVisionMerge; dx++)
                    {
                        const int patch_y = block_y * kVisionMerge + dy;
                        const int patch_x = block_x * kVisionMerge + dx;
                        const int row_index = patch_offset
                            + (block_y * (grid_w / kVisionMerge) + block_x) * kVisionMerge * kVisionMerge
                            + dy * kVisionMerge + dx;
                        float* row = pixels.row(row_index);
                        int offset = 0;
                        for (int c = 0; c < 3; c++)
                            for (int t = 0; t < 2; t++)
                                for (int py = 0; py < kVisionPatch; py++)
                                    for (int px = 0; px < kVisionPatch; px++)
                                    {
                                        const int y = patch_y * kVisionPatch + py;
                                        const int x = patch_x * kVisionPatch + px;
                                        const uint8_t value = static_cast<unsigned char*>(
                                            image.data)[((size_t)y * image.w + x) * 3 + c];
                                        row[offset++] = value / 127.5f - 1.f;
                                    }
                    }
        patch_offset += grid_h * grid_w;
    }
    return pixels;
}

EditInput prepare_edit_input(std::string& error, const GenerateRequest& request, const MemoryBudgets& budgets)
{
    std::vector<Reference> references;
    int output_width = request.width;
    int output_height = request.height;
    uint64_t retained = 0;
    if (!request.reference_images.empty())
    {
        references.reserve(request.reference_images.size());
        for (const auto& path : request.reference_images)
        {
            int source_width = 0, source_height = 0;
            if (!image_size(path, source_width, source_height) || source_width > 32768 || source_height > 32768)
            {
                error = "cannot read reference image dimensions: " + path;
                return {};
            }
            if (static_cast<uint64_t>(source_width) * source_height > INT_MAX / 4
                || fsutil::file_size(error, path) > INT_MAX)
            {
                error = "reference image exceeds codec size limits: " + path;
                return {};
            }
            const uint64_t decode_work = static_cast<uint64_t>(source_width) * source_height * 12
                + fsutil::file_size(error, path) * 2 + retained;
            if (!error.empty())
                return {};

            if (decode_work > budgets.ram)
            {
                error = "reference image decoding exceeds the RAM budget: " + path;
                return {};
            }
            const ncnn::Mat original = load_rgb_image(error, path);
            if (!error.empty())
                return {};

            if (std::min(original.w, original.h) < 32)
            {
                error = "reference images must be at least 32 pixels on each side: " + path;
                return {};
            }
            if (references.empty())
            {
                if (!request.width_set)
                    output_width = original.w;
                if (!request.height_set)
                    output_height = original.h;
            }
            Reference reference;
            const auto vae_size = bounded_short_edge(
                original.w, original.h, request.reference_min_short_edge, request.reference_max_short_edge);
            const auto vision_size = vision_resize_size(error, original.w, original.h, request.vision_short_edge);
            if (!error.empty())
                return {};

            const uint64_t vae_bytes = static_cast<uint64_t>(vae_size.first) * vae_size.second * 3;
            const uint64_t vision_bytes = static_cast<uint64_t>(vision_size.first) * vision_size.second * 3;
            const uint64_t resize_scratch
                = static_cast<uint64_t>(std::max(vae_size.first, vision_size.first)) * original.h * 3 * sizeof(float);
            const uint64_t resize_work
                = retained + original.total() * original.elemsize + vae_bytes * 2 + vision_bytes + resize_scratch;
            if (retained + vae_bytes + vision_bytes > budgets.ram || resize_work > budgets.ram)
            {
                error = "reference preprocessing exceeds the RAM budget before resizing: " + path;
                return {};
            }
            reference.vae_image = crop_rgb_to_multiple(
                error, resize_rgb_lanczos(error, original, vae_size.first, vae_size.second), kLatentScale);
            if (!error.empty())
                return {};

            reference.vision_image
                = reference.vae_image.w == vision_size.first && reference.vae_image.h == vision_size.second
                ? reference.vae_image
                : resize_rgb_lanczos(error, original, vision_size.first, vision_size.second);
            if (!error.empty())
                return {};

            reference.vision_grid_h = reference.vision_image.h / kVisionPatch;
            reference.vision_grid_w = reference.vision_image.w / kVisionPatch;
            retained += reference.vae_image.total() * reference.vae_image.elemsize;
            if (reference.vision_image.data != reference.vae_image.data)
                retained += reference.vision_image.total() * reference.vision_image.elemsize;
            if (retained > budgets.ram)
            {
                error = "reference preprocessing exceeds the RAM budget; use fewer/smaller references "
                        "or increase --ram-budget";
                return {};
            }
            std::cerr << "reference " << (references.size() + 1) << ": VAE " << reference.vae_image.w << "x"
                      << reference.vae_image.h << ", vision " << reference.vision_image.w << "x"
                      << reference.vision_image.h << "\n";
            references.push_back(std::move(reference));
        }
    }

    return { std::move(references), output_width, output_height };
}
}
