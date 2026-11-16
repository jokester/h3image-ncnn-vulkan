// h3-image implemented with ncnn library
#include "runtime.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#if defined(__linux__)
#include <fstream>
#include <sstream>
#include <sys/mman.h>
#include <unistd.h>
#ifndef MADV_PAGEOUT
#define MADV_PAGEOUT 21
#endif
#endif
namespace h3image
{
struct MemoryPressureGuard::Impl
{
    std::atomic<uint64_t> target;
    std::atomic<bool> stop { false };
    std::thread worker;
    explicit Impl(uint64_t budget)
        : target(budget)
    {
#if defined(__linux__)
        worker = std::thread(
            [this]()
            {
                const uint64_t page = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
                auto next_precise = Clock::now();
                while (!stop.load())
                {
                    const auto now = Clock::now();
                    uint64_t size = 0, resident = 0;
                    std::ifstream summary("/proc/self/statm");
                    summary >> size >> resident;
                    // Mapping walks are substantially more expensive than the
                    // summary counters. Use the latter only as an early trigger;
                    // an eviction decision always uses mapping-based RSS.
                    if (resident * page > target.load() && next_precise > now + std::chrono::milliseconds(100))
                        next_precise = now;
                    const bool sample = now >= next_precise;
                    bool pressure = false;
                    if (sample)
                    {
                        pressure = process_resident_memory() > target.load();
                        next_precise = now + std::chrono::milliseconds(resident * page > target.load() ? 100 : 1000);
                    }
                    if (pressure)
                    {
                        // PAGEOUT preserves dirty data even if an address range is
                        // reused while a model is unloaded. Do not use DONTNEED on
                        // a concurrently changing process mapping list.
                        std::ifstream maps("/proc/self/maps");
                        std::string line;
                        while (!stop.load() && std::getline(maps, line))
                        {
                            if (line.find(".ncnn.bin") == std::string::npos
                                && line.find("/ncnn-tmp-") == std::string::npos)
                                continue;
                            unsigned long long begin = 0, end = 0;
                            char permissions[5] {};
                            if (sscanf(line.c_str(), "%llx-%llx %4s", &begin, &end, permissions) == 3
                                && permissions[0] == 'r')
                                madvise(reinterpret_cast<void*>(begin), end - begin, MADV_PAGEOUT);
                        }
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
            });
#endif
    }
    ~Impl()
    {
        stop = true;
        if (worker.joinable())
            worker.join();
    }
};
MemoryPressureGuard::MemoryPressureGuard(uint64_t budget)
    : impl_(std::make_unique<Impl>(budget))
{
}
MemoryPressureGuard::~MemoryPressureGuard() = default;
void MemoryPressureGuard::set_budget(uint64_t budget, uint64_t transient_reserve)
{
    const uint64_t reserve = std::min(budget, transient_reserve);
    impl_->target = budget - reserve;
}
}
