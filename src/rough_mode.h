// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "filesystem_utils.h"
#include <string>
namespace h3image
{
struct RoughModePolicy
{
    explicit RoughModePolicy(int value)
        : mode(value)
    {
    }
    bool valid() const
    {
        return mode >= 0 && mode <= 3;
    }
    bool enabled() const
    {
        return mode != 0;
    }
    bool includes_references() const
    {
        return mode >= 2;
    }
    int refresh_step(std::string& error, int step) const
    {
        if (!valid())
        {
            error = "rough mode must be in [0,3]";
            return {};
        }
        if (step < 0 || step >= 256)
        {
            error = "DiT step must be in [0,255]";
            return {};
        }
        return mode == 3 ? 0 : mode == 0 ? step : step - step % 2;
    }
    int mode;
};
}
