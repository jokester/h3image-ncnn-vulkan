// SPDX-License-Identifier: Apache-2.0
#include "exact_cache.h"
#include "filesystem_utils.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>
#if _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif
namespace h3image
{
namespace
{
namespace fs = std::filesystem;
constexpr char magic[8] = { 'H', '3', 'C', 'A', 'C', 'H', 'E', 1 };
constexpr uint64_t header_bytes = 96;
uint64_t process_id()
{
#if _WIN32
    return GetCurrentProcessId();
#else
    return static_cast<uint64_t>(getpid());
#endif
}
class Lock
{
public:
    explicit Lock(const fs::path& directory)
    {
        const fs::path path = directory / ".write-lock";
#if _WIN32
        handle_ = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
            OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle_ != INVALID_HANDLE_VALUE)
        {
            owned_
                = LockFileEx(handle_, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &overlapped_) != 0;
            if (!owned_)
            {
                CloseHandle(handle_);
                handle_ = INVALID_HANDLE_VALUE;
            }
        }
#else
        descriptor_ = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (descriptor_ >= 0)
        {
            owned_ = flock(descriptor_, LOCK_EX | LOCK_NB) == 0;
            if (!owned_)
            {
                close(descriptor_);
                descriptor_ = -1;
            }
        }
#endif
    }
    Lock(const Lock&) = delete;
    Lock& operator=(const Lock&) = delete;
    ~Lock()
    {
#if _WIN32
        if (handle_ != INVALID_HANDLE_VALUE)
        {
            UnlockFileEx(handle_, 0, 1, 0, &overlapped_);
            CloseHandle(handle_);
        }
#else
        if (descriptor_ >= 0)
        {
            flock(descriptor_, LOCK_UN);
            close(descriptor_);
        }
#endif
    }
    explicit operator bool() const
    {
        return owned_;
    }

private:
    bool owned_ = false;
#if _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
    OVERLAPPED overlapped_ {};
#else
    int descriptor_ = -1;
#endif
};
bool owned_name(const std::string& s)
{
    return s.size() == 68 && s.substr(64) == ".bin"
        && s.substr(0, 64).find_first_not_of("0123456789abcdef") == std::string::npos;
}
}
uint64_t TensorShape::bytes() const
{
    if (dims < 1 || dims > 3 || w <= 0 || h <= 0 || c <= 0 || (dims == 1 && (h != 1 || c != 1))
        || (dims == 2 && c != 1))
        return 0;
    long double n = static_cast<long double>(w) * h * c * 4;
    return n > static_cast<long double>(std::numeric_limits<uint64_t>::max()) ? 0 : static_cast<uint64_t>(n);
}
bool TensorShape::matches(const ncnn::Mat& v) const
{
    return v.dims == dims && v.w == w && v.h == h && v.c == c && v.elempack == 1 && v.elemsize == 4;
}
bool ExactCache::get(const Digest& key, const TensorShape& expected, uint64_t max_bytes, ncnn::Mat& value) const
{
    value.release();

    if (!enabled() || !expected.bytes() || expected.bytes() > max_bytes || expected.bytes() + header_bytes > limit_)
        return false;
    std::string error;
    {
        const fs::path path = fs::path(directory_) / (hex_digest(key) + ".bin");
        if (fsutil::file_size(error, path) != expected.bytes() + header_bytes)
            return false;
        std::ifstream in(path, std::ios::binary);
        char tag[8];
        read_bytes(error, in, tag, 8);
        if (!error.empty())
            return {};

        if (std::memcmp(tag, magic, 8))
            return false;
        TensorShape shape;
        shape.dims = read_u32(error, in);
        if (!error.empty())
            return {};

        shape.w = read_u32(error, in);
        if (!error.empty())
            return {};

        shape.h = read_u32(error, in);
        if (!error.empty())
            return {};

        shape.c = read_u32(error, in);
        if (!error.empty())
            return {};

        if (shape.dims != expected.dims || shape.w != expected.w || shape.h != expected.h || shape.c != expected.c
            || read_u64(error, in) != expected.bytes())
            return false;
        Digest stored_key, checksum;
        read_bytes(error, in, stored_key.data(), 32);
        if (!error.empty())
            return {};

        read_bytes(error, in, checksum.data(), 32);
        if (!error.empty())
            return {};

        if (stored_key != key)
            return false;
        ncnn::Mat result;
        if (shape.dims == 1)
            result.create(shape.w);
        else if (shape.dims == 2)
            result.create(shape.w, shape.h);
        else
            result.create(shape.w, shape.h, shape.c);
        if (result.empty())
            return false;
        read_mat_bytes(error, in, result);
        if (!error.empty())
            return {};

        if (mat_digest(error, result) != checksum)
            return false;
        if (!error.empty())
            return false;
        value = std::move(result);
        in.close();
        std::error_code code;
        fs::last_write_time(path, fs::file_time_type::clock::now(), code);
        return true;
    }
}
bool ExactCache::put(const Digest& key, const ncnn::Mat& value) const
{
    TensorShape shape { value.dims, value.w, value.h, value.c };
    const uint64_t size = shape.bytes() + header_bytes;
    if (!enabled() || !shape.matches(value) || !shape.bytes() || size > limit_)
        return false;
    struct Temporary
    {
        fs::path path;
        ~Temporary()
        {
            if (!path.empty())
            {
                std::error_code code;
                fs::remove(path, code);
            }
        }
    } cleanup;
    fs::path& temporary = cleanup.path;
    std::string error;
    {
        fsutil::create_directories(error, directory_);
        if (!error.empty())
            return {};

        Lock lock(directory_);
        if (!lock)
            return false;
        const fs::path path = fs::path(directory_) / (hex_digest(key) + ".bin");
        std::error_code remove_error;
        fs::remove(path, remove_error);
        if (remove_error)
            return false;
        std::error_code scan_error;
        fs::directory_iterator scan(directory_, scan_error), end;
        for (; !scan_error && scan != end; scan.increment(scan_error))
        {
            const auto& f = *scan;
            const std::string name = f.path().filename().string();
            if (name.size() > 73 && owned_name(name.substr(0, 68)) && name.substr(68, 5) == ".tmp.")
            {
                // One writer owns the directory lock, so no live writer's temporary can exist.
                std::error_code code;
                fs::remove(f.path(), code);
            }
        }
        if (scan_error)
            return false;
        struct Entry
        {
            fs::path path;
            uint64_t bytes;
            fs::file_time_type time;
        };
        std::vector<Entry> entries;
        uint64_t total = 0, old = 0;
        scan = fs::directory_iterator(directory_, scan_error);
        for (; !scan_error && scan != end; scan.increment(scan_error))
        {
            const auto& f = *scan;
            if (!owned_name(f.path().filename().string()) || !fs::is_regular_file(f.symlink_status(scan_error)))
                continue;
            const uint64_t n = fsutil::file_size(error, f.path());
            if (!error.empty())
                return {};

            total += n;
            if (f.path() == path)
                old = n;
            else
                entries.push_back({ f.path(), n, fsutil::last_write_time(error, f.path()) });
        }
        if (scan_error || !error.empty())
            return false;
        if (total < old)
            return false;
        total -= old;
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.time < b.time; });

        for (const auto& e : entries)
        {
            if (total <= limit_ - size)
                break;
            std::error_code code;
            if (fs::remove(e.path, code))
                total -= e.bytes;
        }
        if (total > limit_ - size)
            return false;
        static std::atomic<uint64_t> sequence { 0 };
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        temporary = path.string() + ".tmp." + std::to_string(process_id()) + "." + std::to_string(tick) + "."
            + std::to_string(sequence++);
        const Digest checksum = mat_digest(error, value);
        if (!error.empty())
            return {};

        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        out.write(magic, 8);
        write_u32(out, shape.dims);
        write_u32(out, shape.w);
        write_u32(out, shape.h);
        write_u32(out, shape.c);
        write_u64(out, shape.bytes());
        out.write(reinterpret_cast<const char*>(key.data()), 32);
        out.write(reinterpret_cast<const char*>(checksum.data()), 32);
        write_mat_bytes(error, out, value);
        if (!error.empty())
            return {};

        out.close();
        if (!out)
        {
            error = "cache close failed";
            return {};
        }
        std::error_code code;
#if _WIN32
        fs::remove(path, code);
        code.clear();
#endif
        fs::rename(temporary, path, code);
        return !code;
    }
}
}
