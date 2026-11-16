// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstdint>
#include <istream>
#include <mat.h>
#include <ostream>
#include <string>
namespace h3image
{
using Digest = std::array<unsigned char, 32>;
class Sha256
{
public:
    Sha256();
    void update(const void* data, size_t bytes);
    void text(const std::string& value);
    Digest finish() const;

private:
    void transform(const unsigned char* block);
    std::array<uint32_t, 8> state_;
    std::array<unsigned char, 64> block_ {};
    uint64_t bytes_ = 0;
    size_t pending_ = 0;
};
std::string hex_digest(const Digest& digest);
Digest file_digest(std::string& error, const std::string& path);
Digest mat_digest(std::string& error, const ncnn::Mat& value);
void hash_mat(std::string& error, Sha256& hash, const ncnn::Mat& value);
void hash_file_identity(std::string& error, Sha256& hash, const std::string& path);
std::string execution_identity(int gpu, int threads);
void write_u32(std::ostream& out, uint32_t value);
void write_u64(std::ostream& out, uint64_t value);
uint32_t read_u32(std::string& error, std::istream& in);
uint64_t read_u64(std::string& error, std::istream& in);
void write_string(std::ostream& out, const std::string& value);
std::string read_string(std::string& error, std::istream& in, size_t limit = 4096);
void read_bytes(std::string& error, std::istream& in, void* data, size_t bytes);
void write_mat_bytes(std::string& error, std::ostream& out, const ncnn::Mat& value);
void read_mat_bytes(std::string& error, std::istream& in, ncnn::Mat& value);
}
