// h3-image implemented with ncnn library
#pragma once
#include <mat.h>
#include <string>
#include <vector>
namespace h3image
{
class PddScheduler
{
public:
    static constexpr int steps = 6; // Number of trained output heads / canonical steps.
    static constexpr int max_steps = 256;
    PddScheduler();
    explicit PddScheduler(std::string& error, int count = steps);
    int step_count() const
    {
        return static_cast<int>(sigmas_.size()) - 1;
    }
    bool canonical() const
    {
        return step_count() == steps;
    }
    float raw_sigma(std::string& error, int index) const;
    float timestep(std::string& error, int index) const;
    int output_head(std::string& error, int index) const;
    float delta(std::string& error, int index) const;
    void step(std::string& error, int index, const ncnn::Mat& velocity, int reference_rows, ncnn::Mat& images) const;

private:
    std::vector<float> sigmas_;
};
}
