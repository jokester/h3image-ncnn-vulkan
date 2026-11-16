// h3-image implemented with ncnn library
#pragma once
#include "models.h"
#include <cstdint>
#include <string>
#include <vector>
namespace h3image
{
struct GenerateRequest
{
    std::string prompt;
    std::string output_image = "out.png";
    int width = 1024;
    int height = 1024;
    bool width_set = false;
    bool height_set = false;
    int batch = 1;
    uint64_t seed = 0;
    std::vector<std::string> reference_images;
    // Generation and reference preprocessing parameters.
    int min_generation_area = 512 * 512;
    int reference_min_short_edge = 1024;
    int reference_max_short_edge = 2048;
    int vision_short_edge = 1024;
};
struct GenerateTimings
{
    double text_encoder_ms = 0;
    double vae_encoder_ms = 0;
    double transformer_ms = 0;
    double vae_decoder_ms = 0;
    double total_ms = 0;
};
class H3ImagePipeline
{
public:
    bool load(const std::string& model_directory, RuntimeConfig config, std::string* error = nullptr);
    bool generate(const GenerateRequest& request, GenerateTimings* timings = nullptr);

private:
    std::unique_ptr<H3ModelSet> models_;
};
}
