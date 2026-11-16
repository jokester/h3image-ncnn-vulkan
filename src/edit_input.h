// h3-image implemented with ncnn library
#pragma once
#include "image_io.h"
#include "pipeline.h"
#include <string>
namespace h3image
{
struct Reference
{
    ncnn::Mat vae_image;
    ncnn::Mat vision_image;
    int vision_grid_h = 0;
    int vision_grid_w = 0;
    ncnn::Mat normalized_latent;
};
struct EditInput
{
    std::vector<Reference> references;
    int width;
    int height;
};
int round_to_multiple(double value, int multiple);
EditInput prepare_edit_input(std::string& error, const GenerateRequest& request, const MemoryBudgets& budgets);
ncnn::Mat image_net_pixels(std::string& error, const ncnn::Mat& image);
ncnn::Mat vision_patch_pixels(std::string& error, const std::vector<Reference>& references, ncnn::Mat& grids);
}
