// SPDX-License-Identifier: Apache-2.0
#include "pipeline.h"
#include "filesystem_utils.h"

#include "device_sampler.h"
#include "edit_input.h"
#include "exact_cache.h"
#include "h3_config.h"
#include "image_io.h"
#include "latent_storage.h"
#include "runtime.h"
#include "scheduler.h"
#include "text_encoder.h"
#include "tokenizer.h"
#include "transformer.h"
#include "vae.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>

namespace h3image
{
namespace
{
class NormalRng
{
public:
    explicit NormalRng(uint64_t seed)
        : engine_(seed)
    {
    }

    float next()
    {
        if (has_saved_)
        {
            has_saved_ = false;
            return saved_;
        }
        const double u1 = uniform_open();
        const double u2 = uniform_open();
        const double radius = std::sqrt(-2.0 * std::log(u1));
        constexpr double tau = 6.283185307179586476925286766559;
        saved_ = (float)(radius * std::sin(tau * u2));
        has_saved_ = true;
        return (float)(radius * std::cos(tau * u2));
    }

private:
    double uniform_open()
    {
        // The 53 high bits of mt19937_64 give a reproducible open interval (0, 1).
        return ((engine_() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    }

    std::mt19937_64 engine_;
    bool has_saved_ = false;
    float saved_ = 0.f;
};

std::vector<std::string> read_lines(std::string& error, const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        error = "cannot open " + path;
        return {};
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        lines.push_back(line);
    }
    return lines;
}

BpeTokenizer load_tokenizer(std::string& error, const H3ModelSet& models)
{
    const std::string directory = models.directory() + "/processor/";
    SpecialTokensConfig special;
    auto tokenizer
        = BpeTokenizer::LoadFromFiles(directory + "vocab.txt", directory + "merges.txt", special, false, false, true);

    if (tokenizer.vocab_size() < 151669)
    {
        error = "H3 tokenizer vocabulary must include its base and added tokens";
        return {};
    }
    tokenizer.SetAdditionalSpecialTokens(read_lines(error, directory + "added_tokens.txt"), false);
    if (!error.empty())
        return {};

    for (const char* name : { "<|vision_start|>", "<|vision_end|>", "<|image_pad|>" })
        if (tokenizer.token_to_id().find(name) == tokenizer.token_to_id().end())
        {
            error = std::string("H3 tokenizer is missing special token ") + name;
            return {};
        }
    return tokenizer;
}

std::pair<int, int> resolve_canvas(std::string& error, int width, int height, int min_area)
{
    if (width <= 0 || height <= 0)
    {
        error = "output width and height must be positive";
        return {};
    }
    if (min_area < 1024 || min_area > 4096 * 4096)
    {
        error = "minimum generation area must be in [1024, 16777216]";
        return {};
    }
    const double scale = std::max(1.0, std::sqrt((double)min_area / ((double)width * height)));
    return { round_to_multiple(height * scale, 32), round_to_multiple(width * scale, 32) };
}

void append_tokens(const std::vector<int>& tokens, std::vector<float>& ids, std::vector<float>& tags, float tag)
{
    for (int token : tokens)
    {
        ids.push_back((float)token);
        tags.push_back(tag);
    }
}

int special_id(std::string& error, const BpeTokenizer& tokenizer, const char* name)
{
    const auto found = tokenizer.token_to_id().find(name);
    if (found == tokenizer.token_to_id().end())
    {
        error = std::string("missing tokenizer ID for ") + name;
        return {};
    }
    return found->second;
}

void append_special(int id, std::vector<float>& ids, std::vector<float>& tags, float tag)
{
    ids.push_back((float)id);
    tags.push_back(tag);
}

std::vector<float> frame_axis(int count, double step)
{
    std::vector<float> positions(count);
    for (int i = 0; i < count; i++)
        positions[i] = (float)(16.0 + ((double)i - (double)count / 2.0) * step);
    return positions;
}

void copy_packed_latents(std::string& error, const ncnn::Mat& latent, ncnn::Mat& rows, int row_offset)
{
    if (latent.dims != 3 || latent.c != kLatentChannels || latent.h % 2 || latent.w % 2)
    {
        error = "latent geometry must be 24 channels and divisible by 2";
        return;
    }
    int row_index = row_offset;
    for (int y = 0; y < latent.h / 2; y++)
        for (int x = 0; x < latent.w / 2; x++, row_index++)
            for (int c = 0; c < kLatentChannels; c++)
                for (int py = 0; py < 2; py++)
                    for (int px = 0; px < 2; px++)
                        rows.row(row_index)[c * 4 + py * 2 + px] = latent.channel(c).row(y * 2 + py)[x * 2 + px];
}

ncnn::Mat sample_reference_posterior(std::string& error, const ncnn::Mat& posterior,
    const std::vector<float>& latent_mean, const std::vector<float>& latent_std, NormalRng& rng)
{
    if (posterior.dims != 3 || posterior.c != 2 * kLatentChannels)
    {
        error = "H3 VAE encoder must return 48 posterior channels";
        return {};
    }
    ncnn::Mat sampled(posterior.w, posterior.h, kLatentChannels);
    for (int c = 0; c < kLatentChannels; c++)
        for (int y = 0; y < posterior.h; y++)
            for (int x = 0; x < posterior.w; x++)
            {
                const float mean = posterior.channel(c).row(y)[x];
                const float logvar = std::clamp(posterior.channel(c + kLatentChannels).row(y)[x], -30.f, 20.f);
                sampled.channel(c).row(y)[x] = mean + std::exp(.5f * logvar) * rng.next();
            }
    ncnn::Mat half, rounded;
    ncnn::cast_float32_to_float16(sampled, half);
    ncnn::cast_float16_to_float32(half, rounded);
    const int cropped_h = rounded.h / kTransformerPatch * kTransformerPatch;
    const int cropped_w = rounded.w / kTransformerPatch * kTransformerPatch;
    ncnn::Mat normalized(cropped_w, cropped_h, rounded.c);
    for (int c = 0; c < kLatentChannels; c++)
        for (int y = 0; y < cropped_h; y++)
            for (int x = 0; x < cropped_w; x++)
                normalized.channel(c).row(y)[x] = (rounded.channel(c).row(y)[x] - latent_mean[c]) / latent_std[c];
    return normalized;
}

ncnn::Mat random_latent_noise(int width, int height, NormalRng& rng)
{
    ncnn::Mat noise(width, height, kLatentChannels);
    for (int c = 0; c < kLatentChannels; c++)
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                noise.channel(c).row(y)[x] = rng.next();
            }

    return noise;
}

ncnn::Mat noise_reference(const ncnn::Mat& clean, NormalRng& rng)
{
    ncnn::Mat noise = random_latent_noise(clean.w, clean.h, rng);

    ncnn::Mat result(clean.w, clean.h, clean.c);
    const float t = kReferenceNoiseAug;
    const float noise_weight = 1.f - t;
    for (int c = 0; c < clean.c; c++)
        for (int y = 0; y < clean.h; y++)
            for (int x = 0; x < clean.w; x++)
                result.channel(c).row(y)[x]
                    = t * clean.channel(c).row(y)[x] + noise_weight * noise.channel(c).row(y)[x];
    return result;
}

ncnn::Mat unpack_generated_latents(std::string& error, const ncnn::Mat& packed, int packed_offset, int height,
    int width, const std::vector<float>& mean, const std::vector<float>& stddev)
{
    if (height % 2 || width % 2)
    {
        error = "generated latent dimensions must be even";
        return {};
    }
    const int rows = height / 2 * (width / 2);
    if (packed.dims != 2 || packed.w != kPackedWidth || packed_offset < 0 || packed_offset + rows > packed.h)
    {
        error = "packed latent rows do not match image dimensions";
        return {};
    }
    ncnn::Mat latent(width, height, kLatentChannels);
    for (int y = 0; y < height / 2; y++)
        for (int x = 0; x < width / 2; x++)
        {
            const float* row = packed.row(packed_offset + y * (width / 2) + x);
            for (int c = 0; c < kLatentChannels; c++)
                for (int py = 0; py < 2; py++)
                    for (int px = 0; px < 2; px++)
                    {
                        const float value = row[c * 4 + py * 2 + px];
                        latent.channel(c).row(y * 2 + py)[x * 2 + px] = value * stddev[c] + mean[c];
                    }
        }
    return latent;
}

ncnn::Mat decoded_rgb(std::string& error, ncnn::Mat& pixels)
{
    if (pixels.dims != 3 || pixels.c != 3)
    {
        error = "VAE decoder must return a 3-channel image";
        return {};
    }
    constexpr float mean[3] = { .485f, .456f, .406f };
    constexpr float stddev[3] = { .229f, .224f, .225f };
    ncnn::Mat image(pixels.w, pixels.h, static_cast<size_t>(3), 3);
    if (image.empty())
    {
        error = "cannot allocate output RGB image";
        return {};
    }
    // Consume the decoder buffer in place, preserving its original arithmetic
    // before ncnn's planar-to-RGB conversion and integer truncation.
    for (int c = 0; c < 3; c++)
    {
        float* channel = pixels.channel(c);
        for (size_t i = 0; i < static_cast<size_t>(pixels.w) * pixels.h; i++)
            channel[i] = std::clamp(channel[i] * stddev[c] + mean[c], 0.f, 1.f) * 255.f;
    }
    pixels.to_pixels(static_cast<unsigned char*>(image.data), ncnn::Mat::PIXEL_RGB);
    return image;
}
}

static void run_image_pipeline(
    std::string& error, H3ModelSet& models, const GenerateRequest& request, GenerateTimings& timings)
{
    if (request.prompt.empty())
    {
        error = "--prompt must not be empty";
        return;
    }
    if (request.output_image.empty())
    {
        error = "--output is required";
        return;
    }
    if (request.reference_images.empty())
    {
        if (request.width <= 0 || request.height <= 0)
        {
            error = "--width and --height must be positive";
            return;
        }
    }
    else if (request.reference_min_short_edge <= 32
        || request.reference_max_short_edge < request.reference_min_short_edge || request.vision_short_edge <= 0
        || request.vision_short_edge % 32)
    {
        {
            error = "invalid image-edit resize settings";
            return;
        }
    }

    const std::vector<float> latent_mean(kLatentMean.begin(), kLatentMean.end());

    const std::vector<float> latent_std(kLatentStd.begin(), kLatentStd.end());

    BpeTokenizer tokenizer = load_tokenizer(error, models);
    if (!error.empty())
        return;

    const std::array<int, 3> vision_specials = { special_id(error, tokenizer, "<|vision_start|>"),
        special_id(error, tokenizer, "<|vision_end|>"), special_id(error, tokenizer, "<|image_pad|>") };
    if (!error.empty())
        return;

    EditInput editing = prepare_edit_input(error, request, models.budgets());
    if (!error.empty())
        return;

    auto references = std::move(editing.references);
    const int output_width = editing.width;
    const int output_height = editing.height;

    const auto canvas = resolve_canvas(error, output_width, output_height, request.min_generation_area);
    if (!error.empty())
        return;

    const int canvas_height = canvas.first;
    const int canvas_width = canvas.second;
    if (canvas_height % 32 || canvas_width % 32)
    {
        error = "internal canvas dimensions must be multiples of 32";
        return;
    }
    const int target_latent_h = canvas_height / kLatentScale;
    const int target_latent_w = canvas_width / kLatentScale;
    const int target_grid_h = target_latent_h / kTransformerPatch;
    const int target_grid_w = target_latent_w / kTransformerPatch;
    const int target_rows = target_grid_h * target_grid_w;

    std::vector<float> ids, text_tags;
    if (!references.empty())
    {
        for (size_t i = 0; i < references.size(); i++)
        {
            const std::vector<int> label = tokenizer.encode("Picture " + std::to_string(i + 1) + ": ");

            append_tokens(
                std::vector<int>(label.begin(), label.begin() + std::min<size_t>(label.size(), kMaxTextTokens)), ids,
                text_tags, 1.f);

            append_special(vision_specials[0], ids, text_tags, 0.f);
            const int image_token_count
                = references[i].vision_grid_h / kVisionMerge * (references[i].vision_grid_w / kVisionMerge);
            for (int token = 0; token < image_token_count; token++)
                append_special(vision_specials[2], ids, text_tags, 0.f);
            append_special(vision_specials[1], ids, text_tags, 0.f);
        }
    }
    const std::vector<int> prompt_ids = tokenizer.encode(request.prompt);

    append_tokens(
        std::vector<int>(prompt_ids.begin(), prompt_ids.begin() + std::min<size_t>(prompt_ids.size(), kMaxTextTokens)),
        ids, text_tags, 1.f);

    if (ids.empty())
    {
        error = "tokenizer produced an empty prompt";
        return;
    }
    const int text_count = (int)ids.size();

    Workload workload;
    workload.text_tokens = text_count;
    workload.canvas_width = canvas_width;
    workload.canvas_height = canvas_height;
    workload.output_width = output_width;
    workload.output_height = output_height;
    workload.target_rows = target_rows;
    workload.canvas_pixels = static_cast<uint64_t>(canvas_width) * canvas_height;
    workload.output_pixels = static_cast<uint64_t>(output_width) * output_height;
    workload.batch = request.batch;
    for (const auto& reference : references)
    {
        const uint64_t patches = static_cast<uint64_t>(reference.vision_grid_h) * reference.vision_grid_w;
        const uint64_t pixels = static_cast<uint64_t>(reference.vae_image.w) * reference.vae_image.h;
        workload.references.push_back(
            { static_cast<uint64_t>(reference.vae_image.w), static_cast<uint64_t>(reference.vae_image.h),
                static_cast<uint64_t>(reference.vision_grid_w), static_cast<uint64_t>(reference.vision_grid_h) });
        workload.vision_patches += patches;
        workload.max_vision_patches = std::max(workload.max_vision_patches, patches);
        workload.reference_pixels += pixels;
        workload.max_reference_pixels = std::max(workload.max_reference_pixels, pixels);
        workload.reference_rows += (reference.vae_image.w / 32) * (reference.vae_image.h / 32);
        workload.image_bytes += reference.vae_image.total() * reference.vae_image.elemsize;
        if (reference.vision_image.data != reference.vae_image.data)
            workload.image_bytes += reference.vision_image.total() * reference.vision_image.elemsize;
    }
    models.configure_memory(error, workload);
    if (!error.empty())
        return;

    std::cerr << "workload: " << text_count << " text tokens, " << target_rows << " target rows, "
              << workload.reference_rows << " reference rows, batch " << request.batch << "\n";

    ncnn::Mat token_ids(text_count), text_token_tags(text_count);
    for (int i = 0; i < text_count; i++)
    {
        token_ids[i] = ids[i];
        text_token_tags[i] = text_tags[i];
    }
    ncnn::Mat vision_grids, vision_pixels;
    if (!references.empty())
    {
        vision_pixels = vision_patch_pixels(error, references, vision_grids);
        if (!error.empty())
            return;
    }

    const auto text_begin = Clock::now();
    ExactCache exact_cache(models.options().cache_directory, models.options().cache_limit);
    const uint64_t max_cache_tensor = std::min(models.options().cache_limit, models.budgets().ram);
    Digest text_digest {};
    if (exact_cache.enabled())
    {
        Sha256 key;
        key.text("prompt-features-v1");
        key.text(models.cache_identity(error, Component::Text));
        if (!error.empty())
            return;

        hash_mat(error, key, token_ids);
        if (!error.empty())
            return;

        if (!references.empty())
        {
            hash_mat(error, key, vision_pixels);
            if (!error.empty())
                return;

            hash_mat(error, key, vision_grids);
            if (!error.empty())
                return;
        }
        text_digest = key.finish();
    }
    ncnn::Mat prompt_features;
    const bool text_hit
        = exact_cache.get(text_digest, { 2, kTextWidth, text_count, 1 }, max_cache_tensor, prompt_features);

    if (!text_hit)
    {
        TextEncoder text_encoder(models);
        prompt_features = text_encoder.encode(error, token_ids, references.empty() ? nullptr : &vision_pixels,
            references.empty() ? nullptr : &vision_grids);
        if (!error.empty())
            return;

        exact_cache.put(text_digest, prompt_features);
    }
    std::cerr << "exact feature cache: text " << (exact_cache.enabled() ? (text_hit ? "hit" : "miss") : "disabled")
              << "\n";
    if (prompt_features.dims != 2 || prompt_features.h != text_count || prompt_features.w != kTextWidth)
    {
        error = "text encoder returned an unexpected prompt shape";
        return;
    }
    timings.text_encoder_ms = elapsed_ms(text_begin);
    std::cerr << "text encoder done\n";
    vision_pixels.release();

    vision_grids.release();

    for (auto& reference : references)
    {
        reference.vision_image.release();
    }

    NormalRng rng(request.seed);
    if (!references.empty())
    {
        const auto vae_begin = Clock::now();
        VAE vae(models);
        const std::string posterior_model
            = exact_cache.enabled() ? models.cache_identity(error, Component::VAEEncoder) : "";
        if (!error.empty())
            return;

        for (auto& reference : references)
        {
            ncnn::Mat normalized_pixels = image_net_pixels(error, reference.vae_image);
            if (!error.empty())
                return;

            const auto posterior_key = [&]()
            {
                Sha256 key;
                key.text("reference-posterior-v1");
                key.text(posterior_model);
                // Kernel tiling can change FP32 rounding; never mix plans in an exact cache.
                key.text(std::to_string(models.memory_plan().vae_encoder_tile));
                hash_mat(error, key, normalized_pixels);
                return key.finish();
            };
            Digest posterior_digest = exact_cache.enabled() ? posterior_key() : Digest {};
            ncnn::Mat posterior;
            const bool posterior_hit = exact_cache.get(posterior_digest,
                { 3, normalized_pixels.w / kLatentScale, normalized_pixels.h / kLatentScale, 2 * kLatentChannels },
                max_cache_tensor, posterior);

            if (!posterior_hit)
            {
                // Recheck available memory before selecting the encoder's actual tile size.
                models.prepare(error, Component::VAEEncoder);
                if (!error.empty())
                    return;

                if (exact_cache.enabled())
                    posterior_digest = posterior_key();
                posterior = vae.encode(error, normalized_pixels);
                if (!error.empty())
                    return;

                exact_cache.put(posterior_digest, posterior);
            }
            std::cerr << "exact feature cache: reference posterior "
                      << (exact_cache.enabled() ? (posterior_hit ? "hit" : "miss") : "disabled") << "\n";
            reference.normalized_latent = sample_reference_posterior(error, posterior, latent_mean, latent_std, rng);
            if (!error.empty())
                return;

            reference.vae_image.release();

            if (reference.normalized_latent.h % kTransformerPatch || reference.normalized_latent.w % kTransformerPatch)
            {
                error = "reference latent dimensions must be divisible by the DiT patch size";
                return;
            }
        }
        timings.vae_encoder_ms = elapsed_ms(vae_begin);
        std::cerr << "vae encoder done\n";
    }

    int reference_rows = 0;
    for (const auto& reference : references)
        reference_rows += (reference.normalized_latent.h / kTransformerPatch)
            * (reference.normalized_latent.w / kTransformerPatch);
    const int image_rows = reference_rows + target_rows;
    const int sequence_length = text_count + image_rows;

    ncnn::Mat positions(3, sequence_length), token_tags(sequence_length), timestep_indices(sequence_length);
    positions.fill(0.f);
    token_tags.fill(0.f);
    timestep_indices.fill(0.f);
    for (int i = 0; i < text_count; i++)
    {
        positions.row(i)[0] = (float)i;
        token_tags[i] = text_token_tags[i];
    }
    int image_cursor = 0;
    int reference_number = 0;
    for (const auto& reference : references)
    {
        const int latent_h = reference.normalized_latent.h;
        const int latent_w = reference.normalized_latent.w;
        const int grid_h = latent_h / kTransformerPatch;
        const int grid_w = latent_w / kTransformerPatch;
        const double step = 32.0 / std::sqrt((double)grid_h * grid_w);
        const auto y_positions = frame_axis(grid_h, step);
        const auto x_positions = frame_axis(grid_w, step);
        for (int y = 0; y < grid_h; y++)
            for (int x = 0; x < grid_w; x++)
            {
                const int row = text_count + image_cursor++;
                positions.row(row)[0] = (float)(text_count + reference_number);
                positions.row(row)[1] = y_positions[y];
                positions.row(row)[2] = x_positions[x];
            }
        reference_number++;
    }
    const double target_step = 32.0 / std::sqrt((double)target_grid_h * target_grid_w);
    const auto target_y = frame_axis(target_grid_h, target_step);
    const auto target_x = frame_axis(target_grid_w, target_step);
    for (int y = 0; y < (int)target_y.size(); y++)
        for (int x = 0; x < (int)target_x.size(); x++)
        {
            const int row = text_count + image_cursor++;
            positions.row(row)[0] = (float)(text_count + reference_number);
            positions.row(row)[1] = target_y[y];
            positions.row(row)[2] = target_x[x];
        }
    if (image_cursor != image_rows)
    {
        error = "packed image row count does not match rotary layout";
        return;
    }
    for (int i = text_count; i < sequence_length; i++)
        token_tags[i] = 0.f;

    ncnn::Mat image_indices(image_rows), text_indices(text_count), timesteps(references.empty() ? 1 : 2);
    for (int i = 0; i < image_rows; i++)
        image_indices[i] = (float)(text_count + i);
    for (int i = 0; i < text_count; i++)
        text_indices[i] = (float)i;

    TransformerInput transformer_input;
    transformer_input.text = prompt_features;
    transformer_input.timesteps = timesteps;
    transformer_input.timestep_indices = timestep_indices;
    transformer_input.token_tags = token_tags;
    transformer_input.positions = positions;
    transformer_input.image_indices = image_indices;
    transformer_input.text_indices = text_indices;

    PddScheduler scheduler(error, models.options().steps);
    if (!error.empty())
        return;

    models.prepare_conditioning(error);
    if (!error.empty())
        return;

    models.prepare(error, Component::Transformer);
    if (!error.empty())
        return;

    Transformer transformer(models);
    const auto prefix_begin = Clock::now();
    transformer.prepare(prompt_features, timesteps.w);

    timings.transformer_ms += elapsed_ms(prefix_begin);
    const bool spill = models.memory_plan().spill_latents;
    LatentBatch latents(error, spill, target_latent_w, target_latent_h, kLatentChannels);
    if (!error.empty())
        return;

    models.lock_latent_storage(spill);
    if (spill)
        std::cerr << "batch latent storage: disk\n";
#if NCNN_VULKAN
    std::unique_ptr<DeviceSampler> device_sampler;
    if (models.memory_plan().device_sampler)
    {
        device_sampler = std::make_unique<DeviceSampler>(error, models.options().gpu);
        if (!error.empty())
            return;

        std::cerr << "PDD sampler: persistent FP32 device sample\n";
    }
#endif
    for (int batch_index = 0; batch_index < request.batch; batch_index++)
    {
        ncnn::Mat packed_images(kPackedWidth, image_rows);
        ncnn::Mat target_noise = random_latent_noise(target_latent_w, target_latent_h, rng);

        int reference_offset = 0;
        for (const auto& reference : references)
        {
            ncnn::Mat noised = noise_reference(reference.normalized_latent, rng);

            const int rows = (noised.h / 2) * (noised.w / 2);
            copy_packed_latents(error, noised, packed_images, reference_offset);
            if (!error.empty())
                return;

            reference_offset += rows;
        }
        copy_packed_latents(error, target_noise, packed_images, reference_rows);
        if (!error.empty())
            return;

        transformer_input.images = packed_images;
#if NCNN_VULKAN
        ncnn::VkMat device_images;
        if (device_sampler)
        {
            device_images = device_sampler->upload(error, packed_images);
            if (!error.empty())
                return;
        }
#endif
        for (int step = 0; step < scheduler.step_count(); step++)
        {
            const float time = scheduler.timestep(error, step);
            if (!error.empty())
                return;

            transformer_input.timesteps[0] = time;
            if (!references.empty())
            {
                transformer_input.timesteps[1] = kReferenceNoiseAug;
                for (int i = 0; i < text_count; i++)
                    transformer_input.timestep_indices[i] = 0.f;
                for (int i = 0; i < reference_rows; i++)
                    transformer_input.timestep_indices[text_count + i] = 1.f;
                for (int i = reference_rows; i < image_rows; i++)
                    transformer_input.timestep_indices[text_count + i] = 0.f;
            }
            const auto dit_begin = Clock::now();
#if NCNN_VULKAN
            if (device_sampler)
            {
                transformer.step_vulkan(error, transformer_input, step, device_images, *device_sampler, reference_rows);
                if (!error.empty())
                    return;
            }
            else
#endif
            {
                ncnn::Mat velocity = transformer.forward(error, transformer_input, step);
                if (!error.empty())
                    return;

                if (velocity.dims != 2 || velocity.w != kPackedWidth || velocity.h != image_rows)
                {
                    error = "DiT returned an unexpected velocity shape";
                    return;
                }
                scheduler.step(error, step, velocity, reference_rows, transformer_input.images);
                if (!error.empty())
                    return;
            }
            timings.transformer_ms += elapsed_ms(dit_begin);
            std::cerr << "step " << (step + 1) << "/" << scheduler.step_count();
            if (!scheduler.canonical())
            {
                std::cerr << " (head " << scheduler.output_head(error, step) + 1 << "/6)";
                if (!error.empty())
                    return;
            }

            if (request.batch > 1)
                std::cerr << " of image " << (batch_index + 1) << "/" << request.batch;
            std::cerr << " done\n";
        }

#if NCNN_VULKAN
        if (device_sampler)
        {
            transformer_input.images = device_sampler->download(error, device_images);
            if (!error.empty())
                return;
        }
#endif
        ncnn::Mat latent = unpack_generated_latents(
            error, transformer_input.images, reference_rows, target_latent_h, target_latent_w, latent_mean, latent_std);
        if (!error.empty())
            return;

        latents.append(error, std::move(latent));
        if (!error.empty())
            return;
    }

    transformer.clear_cache();

#if NCNN_VULKAN
    // Release sample buffers before planning/loading the decoder stage.
    device_sampler.reset();
#endif
    models.prepare(error, Component::VAEDecoder);
    if (!error.empty())
        return;

    VAE vae(models);
    const auto decode_begin = Clock::now();
    for (int batch_index = 0; batch_index < request.batch; batch_index++)
    {
        ncnn::Mat latent = latents.next(error);
        if (!error.empty())
            return;

        ncnn::Mat decoded = vae.decode(error, latent);
        if (!error.empty())
            return;

        ncnn::Mat result = decoded_rgb(error, decoded);
        if (!error.empty())
            return;

        if (result.w != canvas_width || result.h != canvas_height)
        {
            error = "VAE decoder output size does not match generation canvas";
            return;
        }
        if (result.w != output_width || result.h != output_height)
        {
            result = resize_rgb_lanczos(error, result, output_width, output_height);
            if (!error.empty())
                return;
        }

        const std::string output = make_batch_output_path(request.output_image, batch_index, request.batch);
        save_rgb_image(error, output, result);
        if (!error.empty())
            return;

        std::cerr << "saved " << output << "\n";
    }
    timings.vae_decoder_ms = elapsed_ms(decode_begin);
    models.unload();
}

bool H3ImagePipeline::load(const std::string& model_directory, RuntimeConfig config, std::string* error)
{
    models_.reset();
    std::string diagnostic;
    auto model = std::make_unique<H3ModelSet>();
    if (!model->load(diagnostic, model_directory, config))
    {
        if (error)
            *error = diagnostic;
        return false;
    }
    models_ = std::move(model);
    return true;
}

bool H3ImagePipeline::generate(const GenerateRequest& request, GenerateTimings* timings)
{
    if (!models_ || request.batch <= 0 || request.batch > 256 || request.reference_images.size() > 10)
    {
        std::cerr << "invalid generation request or unloaded pipeline\n";
        return false;
    }
    std::string error;
    models_->begin_generation(error);
    GenerateTimings result;
    const auto start = Clock::now();
    if (error.empty())
        run_image_pipeline(error, *models_, request, result);
    if (!error.empty())
    {
        std::cerr << error << "\n";
        models_->unload();
        return false;
    }
    result.total_ms = elapsed_ms(start);
    if (timings)
        *timings = result;
    return true;
}
}
