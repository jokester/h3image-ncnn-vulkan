// SPDX-License-Identifier: Apache-2.0
#include "artifact_hash.h"
#include "filesystem_utils.h"
#include <algorithm>
#include <cfenv>
#include <cpu.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gpu.h>
#include <iomanip>
#include <sstream>
#ifndef H3_CACHE_BUILD_ID
#define H3_CACHE_BUILD_ID "h3-cache-development-v1"
#endif
namespace h3image
{
namespace
{
uint32_t rotate(uint32_t x, unsigned n)
{
    return (x >> n) | (x << (32 - n));
}
constexpr uint32_t constants[64] = { 0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152,
    0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138,
    0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70,
    0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa,
    0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };
template <typename Fn> void rows(std::string& error, const ncnn::Mat& value, Fn fn)
{
    if (value.empty() || value.elempack != 1 || value.dims < 1 || value.dims > 3)
    {
        error = "unsupported cached tensor layout";
        return;
    }
    const size_t width = static_cast<size_t>(value.w) * value.elemsize;
    if (value.dims == 1)
        fn(value.data, width);
    else if (value.dims == 2)
        for (int y = 0; y < value.h; y++)
            fn(value.row(y), width);
    else
        for (int c = 0; c < value.c; c++)
            for (int y = 0; y < value.h; y++)
                fn(value.channel(c).row(y), width);
}
}
Sha256::Sha256()
    : state_ { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 }
{
}
void Sha256::transform(const unsigned char* block)
{
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t(block[i * 4]) << 24) | (uint32_t(block[i * 4 + 1]) << 16) | (uint32_t(block[i * 4 + 2]) << 8)
            | block[i * 4 + 3];
    for (int i = 16; i < 64; i++)
    {
        uint32_t a = rotate(w[i - 15], 7) ^ rotate(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t b = rotate(w[i - 2], 17) ^ rotate(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + a + w[i - 7] + b;
    }
    uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3], e = state_[4], f = state_[5], g = state_[6],
             h = state_[7];
    for (int i = 0; i < 64; i++)
    {
        uint32_t s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25), ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + s1 + ch + constants[i] + w[i];
        uint32_t s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22), maj = (a & b) ^ (a & c) ^ (b & c), t2 = s0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    uint32_t v[8] = { a, b, c, d, e, f, g, h };
    for (int i = 0; i < 8; i++)
        state_[i] += v[i];
}
void Sha256::update(const void* data, size_t size)
{
    const auto* p = static_cast<const unsigned char*>(data);
    bytes_ += size;
    while (size)
    {
        const size_t n = std::min(size, 64 - pending_);
        std::memcpy(block_.data() + pending_, p, n);
        pending_ += n;
        p += n;
        size -= n;
        if (pending_ == 64)
        {
            transform(block_.data());
            pending_ = 0;
        }
    }
}
void Sha256::text(const std::string& value)
{
    const uint64_t n = value.size();
    update(&n, sizeof(n));
    update(value.data(), value.size());
}
Digest Sha256::finish() const
{
    Sha256 copy = *this;
    const uint64_t bits = bytes_ * 8;
    const unsigned char marker = 0x80, zero = 0;
    copy.update(&marker, 1);
    while (copy.pending_ != 56)
        copy.update(&zero, 1);
    unsigned char length[8];
    for (int i = 0; i < 8; i++)
        length[i] = static_cast<unsigned char>(bits >> (56 - i * 8));
    copy.update(length, 8);
    Digest out;
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            out[i * 4 + j] = static_cast<unsigned char>(copy.state_[i] >> (24 - j * 8));
    return out;
}
std::string hex_digest(const Digest& digest)
{
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (auto b : digest)
        out << std::setw(2) << unsigned(b);
    return out.str();
}
Digest file_digest(std::string& error, const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        error = "cannot hash " + path;
        return {};
    }
    Sha256 hash;
    char buffer[65536];
    while (in)
    {
        in.read(buffer, sizeof(buffer));
        hash.update(buffer, static_cast<size_t>(in.gcount()));
    }
    if (!in.eof())
    {
        error = "cannot read " + path;
        return {};
    }
    return hash.finish();
}
void hash_mat(std::string& error, Sha256& hash, const ncnn::Mat& value)
{
    const uint64_t shape[]
        = { uint64_t(value.dims), uint64_t(value.w), uint64_t(value.h), uint64_t(value.c), value.elemsize };
    hash.update(shape, sizeof(shape));
    rows(error, value, [&](const void* data, size_t n) { hash.update(data, n); });
    if (!error.empty())
        return;
}
Digest mat_digest(std::string& error, const ncnn::Mat& value)
{
    Sha256 h;
    hash_mat(error, h, value);
    if (!error.empty())
        return {};

    return h.finish();
}
void hash_file_identity(std::string& error, Sha256& hash, const std::string& path)
{
    namespace fs = std::filesystem;
    hash.text(fsutil::weakly_canonical(error, fs::path(path)).generic_string());
    if (!error.empty())
        return;

    const uint64_t size = fsutil::file_size(error, path);
    if (!error.empty())
        return;

    const auto time = fsutil::last_write_time(error, path).time_since_epoch().count();
    if (!error.empty())
        return;

    hash.update(&size, sizeof(size));
    hash.update(&time, sizeof(time));
    // Graphs/constants are small enough to bind by content as well as file identity.
    if (size <= 4 * 1024 * 1024)
    {
        const Digest d = file_digest(error, path);
        if (!error.empty())
            return;

        hash.update(d.data(), d.size());
    }
}
std::string execution_identity(int gpu, int threads)
{
    std::ostringstream out;
    out << H3_CACHE_BUILD_ID << ":" << threads << ":";
#if NCNN_VULKAN
    if (gpu >= 0)
    {
        const auto& info = ncnn::get_gpu_device(gpu)->info;
        out << "vk:" << info.vendor_id() << ":" << info.device_id() << ":" << info.driver_version() << ":"
            << info.device_name() << ":";
    }
#else
    (void)gpu;
#endif
    out << "cpu:" << sizeof(void*) << ":" << std::fegetround() << ":" << ncnn::get_cpu_count() << ":"
        << ncnn::get_physical_cpu_count() << ":" << ncnn::get_cpu_level2_cache_size() << ":"
        << ncnn::get_cpu_level3_cache_size() << ":";
    const int isa[] = { ncnn::cpu_support_arm_neon(), ncnn::cpu_support_arm_vfpv4(), ncnn::cpu_support_arm_asimdhp(),
        ncnn::cpu_support_arm_asimddp(), ncnn::cpu_support_arm_asimdfhm(), ncnn::cpu_support_arm_bf16(),
        ncnn::cpu_support_arm_i8mm(), ncnn::cpu_support_arm_sve(), ncnn::cpu_support_arm_sve2(),
        ncnn::cpu_support_arm_svebf16(), ncnn::cpu_support_arm_svef32mm(), ncnn::cpu_support_x86_avx(),
        ncnn::cpu_support_x86_fma(), ncnn::cpu_support_x86_fma4(), ncnn::cpu_support_x86_f16c(),
        ncnn::cpu_support_x86_avx2(), ncnn::cpu_support_x86_avx512(), ncnn::cpu_support_x86_avx512_bf16(),
        ncnn::cpu_support_x86_avx512_fp16(), ncnn::cpu_support_loongarch_lsx(), ncnn::cpu_support_loongarch_lasx(),
        ncnn::cpu_support_mips_msa(), ncnn::cpu_support_riscv_v(), ncnn::cpu_support_riscv_zfh(),
        ncnn::cpu_support_riscv_zvfh() };
    for (int flag : isa)
        out << flag;
    out << ":";
#if defined(__x86_64__) || defined(_M_X64)
    out << "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    out << "aarch64";
#else
    out << "other";
#endif
    return out.str();
}
void write_u32(std::ostream& out, uint32_t v)
{
    unsigned char b[4];
    for (int i = 0; i < 4; i++)
        b[i] = v >> (i * 8);
    out.write((char*)b, 4);
}
void write_u64(std::ostream& out, uint64_t v)
{
    unsigned char b[8];
    for (int i = 0; i < 8; i++)
        b[i] = v >> (i * 8);
    out.write((char*)b, 8);
}
void read_bytes(std::string& error, std::istream& in, void* data, size_t n)
{
    if (!in.read(static_cast<char*>(data), n))
    {
        error = "truncated cache/model artifact";
        return;
    }
}
uint32_t read_u32(std::string& error, std::istream& in)
{
    unsigned char b[4];
    read_bytes(error, in, b, 4);
    if (!error.empty())
        return {};

    uint32_t v = 0;
    for (int i = 0; i < 4; i++)
        v |= uint32_t(b[i]) << (i * 8);
    return v;
}
uint64_t read_u64(std::string& error, std::istream& in)
{
    unsigned char b[8];
    read_bytes(error, in, b, 8);
    if (!error.empty())
        return {};

    uint64_t v = 0;
    for (int i = 0; i < 8; i++)
        v |= uint64_t(b[i]) << (i * 8);
    return v;
}
void write_string(std::ostream& out, const std::string& s)
{
    write_u32(out, uint32_t(s.size()));
    out.write(s.data(), s.size());
}
std::string read_string(std::string& error, std::istream& in, size_t limit)
{
    uint32_t n = read_u32(error, in);
    if (!error.empty())
        return {};

    if (n > limit)
    {
        error = "oversized artifact string";
        return {};
    }
    std::string s(n, '\0');
    read_bytes(error, in, s.data(), n);
    if (!error.empty())
        return {};

    return s;
}
void write_mat_bytes(std::string& error, std::ostream& out, const ncnn::Mat& m)
{
    rows(error, m, [&](const void* data, size_t n) { out.write(static_cast<const char*>(data), n); });
    if (!error.empty())
        return;

    if (!out)
    {
        error = "cannot write cached tensor";
        return;
    }
}
void read_mat_bytes(std::string& error, std::istream& in, ncnn::Mat& m)
{
    rows(error, m, [&](const void* data, size_t n) { read_bytes(error, in, const_cast<void*>(data), n); });
    if (!error.empty())
        return;
}
}
