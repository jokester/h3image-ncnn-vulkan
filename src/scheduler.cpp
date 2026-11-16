// h3-image implemented with ncnn library
#include "scheduler.h"
#include "filesystem_utils.h"
#include "h3_config.h"
#include <algorithm>
namespace h3image
{
PddScheduler::PddScheduler()
    : sigmas_(kSampleSigmas.begin(), kSampleSigmas.end())
{
}
PddScheduler::PddScheduler(std::string& error, int count)
{
    if (count < 1 || count > max_steps)
    {
        error = "PDD steps must be in [1,256]";
        return;
    }
    if (count == steps)
    {
        sigmas_.assign(kSampleSigmas.begin(), kSampleSigmas.end());

        return;
    }
    // Experimental: resample the six trained intervals, repeating/skipping their heads.
    sigmas_.resize(count + 1);
    for (int i = 0; i < count; i++)
    {
        const int coordinate = i * steps;
        const int head = coordinate / count;
        const float fraction = static_cast<float>(coordinate % count) / static_cast<float>(count);
        sigmas_[i] = kSampleSigmas[head] + (kSampleSigmas[head + 1] - kSampleSigmas[head]) * fraction;
    }
    sigmas_.back() = 0.f;
}
float PddScheduler::raw_sigma(std::string& error, int index) const
{
    if (index < 0 || index > step_count())
    {
        error = "invalid PDD sigma index";
        return {};
    }
    return sigmas_[index];
}
float PddScheduler::timestep(std::string& error, int index) const
{
    const float sigma = 1.f - (1.f - raw_sigma(error, index));
    if (!error.empty())
        return {};

    return 1.f - sigma;
}
int PddScheduler::output_head(std::string& error, int index) const
{
    if (index < 0 || index >= step_count())
    {
        error = "invalid PDD step index";
        return {};
    }
    return std::min(steps - 1, index * steps / step_count());
}
float PddScheduler::delta(std::string& error, int index) const
{
    if (index < 0 || index >= step_count())
    {
        error = "invalid PDD step index";
        return {};
    }
    const float current = 1.f - (1.f - sigmas_[index]);
    const float next = index + 1 == step_count() ? sigmas_[index + 1] : 1.f - (1.f - sigmas_[index + 1]);
    return current - next;
}
void PddScheduler::step(
    std::string& error, int index, const ncnn::Mat& velocity, int reference_rows, ncnn::Mat& images) const
{
    const float delta = this->delta(error, index);
    if (!error.empty())
        return;

    for (int row = reference_rows; row < images.h; row++)
        for (int column = 0; column < images.w; column++)
            images.row(row)[column] += delta * velocity.row(row)[column];
}
}
