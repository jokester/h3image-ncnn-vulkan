// h3-image implemented with ncnn library
#include "filesystem_utils.h"
#include "pipeline.h"
#include "scheduler.h"
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <gpu.h>
#include <iostream>
#include <random>
#if _WIN32
#include <windows.h>
#endif

using namespace h3image;

static void print_usage()
{
    fprintf(stdout, "Usage: h3image-ncnn-vulkan [options]...\n\n");
    fprintf(stdout, "  -h                   show this help\n");
    fprintf(stdout, "  -p prompt            prompt\n");
    fprintf(stdout, "  -o output-path       output image path (default=out.png)\n");
    fprintf(stdout, "  -i input-image       reference image for editing (repeat 1 to 10 times)\n");
    fprintf(stdout, "  -s image-size        image resolution (default=1024,1024; editing=input size)\n");
    fprintf(stdout, "  -r random-seed       random seed (default=rand)\n");
    fprintf(stdout, "  -m model-path        model directory (default=models/h3image)\n");
    fprintf(stdout, "  -g gpu-id            GPU device to use (-1=cpu, default=auto)\n");
    fprintf(stdout, "  --ram-budget MiB     process RAM planning budget (0=auto, default)\n");
    fprintf(stdout, "  --vram-budget MiB    GPU memory planning budget (0=auto, default)\n");
    fprintf(stdout, "  --rough-mode mode    0=exact, 1=text/2, 2=full/2, 3=freeze (default=0)\n");
    fprintf(stdout, "  --fixed-modulations 0|1  use precomputed conditioning banks (0=online, default=1)\n");
    fprintf(stdout, "  --cache-dir path     exact feature cache directory (default=model/.cache/h3image-exact)\n");
    fprintf(stdout, "  --cache-size MiB     disk cache limit (default=512, 0=disabled)\n");
    fprintf(stdout, "  -b batch-size        batched generation (default=1)\n");
    fprintf(stdout, "  -l steps             denoise steps (default=6; other counts experimental)\n");
    fprintf(stdout, "  -n negative-prompt   unsupported by the guidance-distilled H3 checkpoint\n");
    fprintf(stdout, "  -w guidance-scale    H3 has baked-in guidance (must be 1)\n");
}

static long long parse_integer(std::string& error, const std::string& text, long long low, long long high)
{
    char* end = nullptr;
    errno = 0;
    const long long result = std::strtoll(text.c_str(), &end, 10);
    if (errno || end == text.c_str() || *end || result < low || result > high)
    {
        error = "invalid integer: " + text;
        return {};
    }
    return result;
}

static int run_main(std::string& error, int argc, char** argv)
{
#if _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    {
        GenerateRequest request;
        request.prompt = "A glossy red ceramic teapot beside a white porcelain cup.";
        std::random_device random;
        request.seed = (static_cast<uint64_t>(random()) << 32) ^ random();
        std::string model = "models/h3image";
        std::string negative;
        float guidance = 1.f;
        int steps = 6;
        int gpu_id = -2;
        RuntimeConfig config;
        for (int i = 1; i < argc; i++)
        {
            const std::string arg = argv[i];
            if (arg == "-h" || arg == "--help")
            {
                print_usage();
                return 0;
            }
            if (arg == "--cache-dir" || arg == "--cache-size" || arg == "--fixed-modulations"
                || arg.find("--cache-dir=") == 0 || arg.find("--cache-size=") == 0
                || arg.find("--fixed-modulations=") == 0)
            {
                const auto equal = arg.find('=');
                if (equal == std::string::npos && i + 1 >= argc)
                {
                    error = "missing value for " + arg;
                    return 1;
                }
                const std::string value = equal == std::string::npos ? argv[++i] : arg.substr(equal + 1);
                const std::string name = arg.substr(0, equal);
                if (name == "--cache-dir")
                {
                    if (value.empty())
                    {
                        error = "--cache-dir must not be empty";
                        return 1;
                    }
                    config.cache_directory = value;
                }
                else if (name == "--cache-size")
                {
                    config.cache_limit = static_cast<uint64_t>(parse_integer(error, value, 0, LLONG_MAX / MiB)) * MiB;
                    if (!error.empty())
                        return 1;
                }
                else
                    config.fixed_modulations = parse_integer(error, value, 0, 1) != 0;
                continue;
            }
            if (arg == "--rough-mode" || arg.find("--rough-mode=") == 0)
            {
                const auto equal = arg.find('=');
                if (equal == std::string::npos && i + 1 >= argc)
                {
                    error = "missing value for --rough-mode";
                    return 1;
                }
                const std::string value = equal == std::string::npos ? argv[++i] : arg.substr(equal + 1);
                config.rough_mode = static_cast<int>(parse_integer(error, value, 0, 3));
                const bool valid = error.empty();
                if (!valid)
                {
                    error = "--rough-mode must be an integer in [0,3]";
                    return 1;
                }
                continue;
            }
            if (arg == "--ram-budget" || arg == "--vram-budget" || arg.find("--ram-budget=") == 0
                || arg.find("--vram-budget=") == 0)
            {
                const auto equal = arg.find('=');
                if (equal == std::string::npos && i + 1 >= argc)
                {
                    error = "missing value for " + arg;
                    return 1;
                }
                const std::string budget_value = equal == std::string::npos ? argv[++i] : arg.substr(equal + 1);
                const uint64_t amount
                    = static_cast<uint64_t>(parse_integer(error, budget_value, 0, LLONG_MAX / MiB)) * MiB;
                if (!error.empty())
                    return 1;

                if (arg.substr(0, equal) == "--ram-budget")
                    config.ram_budget = amount;
                else
                    config.vram_budget = amount;
                continue;
            }
            if (arg.size() < 2 || arg[0] != '-' || arg[1] == '-')
            {
                error = "unknown argument: " + arg;
                return 1;
            }
            const char option = arg[1];
            if (std::string("poismrgblnw").find(option) == std::string::npos)
            {
                error = "unknown option: " + arg;
                return 1;
            }
            std::string value;
            if (arg.size() > 2)
                value = arg.substr(2);
            else if (i + 1 < argc)
                value = argv[++i];
            else
            {
                error = "missing value for " + arg;
                return 1;
            }
            switch (option)
            {
            case 'p':
                request.prompt = value;
                break;
            case 'o':
                request.output_image = value;
                break;
            case 'i':
                request.reference_images.push_back(value);
                break;
            case 'm':
                model = value;
                break;
            case 'g':
                gpu_id = static_cast<int>(parse_integer(error, value, -1, INT_MAX));
                if (!error.empty())
                    return 1;

                break;
            case 'b':
                request.batch = static_cast<int>(parse_integer(error, value, 1, 256));
                if (!error.empty())
                    return 1;

                break;
            case 'l':
                steps = static_cast<int>(parse_integer(error, value, 1, PddScheduler::max_steps));
                if (!error.empty())
                    return 1;

                break;
            case 'n':
                negative = value;
                break;
            case 'w':
            {
                char* end = nullptr;
                guidance = std::strtof(value.c_str(), &end);
                if (end == value.c_str() || *end || !std::isfinite(guidance))
                {
                    error = "invalid guidance scale";
                    return 1;
                }
                break;
            }
            case 'r':
            {
                char* end = nullptr;
                errno = 0;
                if (value.empty() || value.front() == '-')
                {
                    error = "invalid seed";
                    return 1;
                }
                request.seed = std::strtoull(value.c_str(), &end, 10);
                if (errno || end == value.c_str() || *end)
                {
                    error = "invalid seed";
                    return 1;
                }
                break;
            }
            case 's':
            {
                const size_t comma = value.find(',');
                if (comma == std::string::npos)
                {
                    error = "image size must be WIDTH,HEIGHT";
                    return 1;
                }
                request.width = static_cast<int>(parse_integer(error, value.substr(0, comma), 32, 4096));
                if (!error.empty())
                    return 1;

                request.height = static_cast<int>(parse_integer(error, value.substr(comma + 1), 32, 4096));
                if (!error.empty())
                    return 1;

                request.width_set = request.height_set = true;
                break;
            }
            }
        }
        if (config.ram_budget && config.ram_budget < 256 * MiB)
        {
            error = "--ram-budget must be zero or at least 256 MiB";
            return 1;
        }
        if (request.prompt.empty())
        {
            error = "prompt must not be empty";
            return 1;
        }
        if (request.reference_images.size() > 10)
        {
            error = "at most 10 reference images are supported";
            return 1;
        }
        if (!negative.empty() || guidance != 1.f)
        {
            error
                = "H3 is guidance-distilled: negative prompts and external CFG are not supported; omit -n and use -w 1";
            return 1;
        }
        config.steps = steps;
        if (steps != 6)
            std::cerr << "experimental PDD: " << steps
                      << " steps, resampled six-interval sigma/head schedule; quality requires validation\n";
        std::string ext = std::filesystem::path(request.output_image).extension().string();
        for (char& c : ext)
            if (c >= 'A' && c <= 'Z')
                c = c - 'A' + 'a';
        if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".webp")
        {
            error = "output format must be png, jpg or webp";
            return 1;
        }
#if NCNN_VULKAN
        GpuInstance gpu(error, gpu_id != -1);
        if (gpu_id == -2)
            gpu_id = ncnn::get_default_gpu_index();
        if (gpu_id >= ncnn::get_gpu_count())
        {
            error = "invalid GPU device index";
            return 1;
        }
#else
        GpuInstance gpu(error, false);
        if (gpu_id >= 0)
        {
            error = "ncnn was built without Vulkan";
            return 1;
        }
        if (gpu_id == -2)
            gpu_id = -1;
#endif
        config.gpu = gpu_id;
        if (gpu_id < 0 && config.vram_budget)
        {
            error = "--vram-budget requires a GPU";
            return 1;
        }
        std::cerr << "prompt = " << request.prompt << "\noutput-path = " << request.output_image
                  << "\nmodel = " << model << "\nseed = " << request.seed << "\ngpu-id = " << gpu_id
                  << "\nbatch = " << request.batch << "\nrough-mode = " << config.rough_mode << "\n";
        GenerateTimings timings;
        std::string model_error;
        H3ImagePipeline pipeline;
        if (!pipeline.load(model, config, &model_error))
        {
            error = model_error;
            return 1;
        }
        const bool success = pipeline.generate(request, &timings);
        if (!success)
            return 1;
        fprintf(stdout,
            "text encoder: %.2f ms\nvae encoder: %.2f ms\ntransformer: %.2f ms\nvae decoder: %.2f ms\ntotal: %.2f ms\n",
            timings.text_encoder_ms, timings.vae_encoder_ms, timings.transformer_ms, timings.vae_decoder_ms,
            timings.total_ms);
        return 0;
    }
}

int main(int argc, char** argv)
{
    std::string error;
    const int result = run_main(error, argc, argv);
    if (!error.empty())
    {
        std::cerr << error << "\n";
        return 1;
    }
    return result;
}
