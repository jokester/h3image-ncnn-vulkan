// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "runtime.h"
#include "scheduler.h"
#include <string>
namespace h3image
{
void build_modulations(std::string& error, const std::string& directory, int gpu, int threads,
    const MemoryBudgets& budget, const std::string& path, const PddScheduler& schedule, uint64_t retained_host = 0);
}
