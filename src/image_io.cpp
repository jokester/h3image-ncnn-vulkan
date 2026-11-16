// h3-image implemented with ncnn library

#include "image_io.h"
#include "filesystem_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#if _WIN32
#include <objbase.h>
#include <windows.h>
#endif

#include "jpeg_image.h"
#include "png_image.h"
#include "webp_image.h"
#if _WIN32
#include "wic_image.h"
#endif

namespace h3image
{
namespace
{
std::string lowercase_path(std::string value)
{
    for (char& c : value)
    {
        if (c >= 'A' && c <= 'Z')
            c = (char)(c - 'A' + 'a');
    }
    return value;
}

int read_file_bytes(const std::string& path, std::vector<unsigned char>& bytes)
{
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp)
        return -1;

    if (std::fseek(fp, 0, SEEK_END) != 0)
    {
        std::fclose(fp);
        return -1;
    }
    const long length = std::ftell(fp);
    if (length <= 0 || length > (std::numeric_limits<int>::max)() || std::fseek(fp, 0, SEEK_SET) != 0)
    {
        std::fclose(fp);
        return -1;
    }

    bytes.resize((size_t)length);
    const size_t read = std::fread(bytes.data(), 1, bytes.size(), fp);
    std::fclose(fp);
    return read == bytes.size() ? 0 : -1;
}

bool assign_rgb(const unsigned char* pixels, int width, int height, int channels, ncnn::Mat& image, bool bgr)
{
    if (width <= 0 || height <= 0 || channels < 3 || channels > 4)
        return false;
    image.create(width, height, static_cast<size_t>(3), 3);
    if (image.empty())
        return false;
    if (!bgr && channels == 3)
    {
        std::memcpy(image.data, pixels, static_cast<size_t>(width) * height * 3);
        return true;
    }
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            const unsigned char* source = pixels + ((size_t)y * width + x) * channels;
            unsigned char* destination = static_cast<unsigned char*>(image.data) + ((size_t)y * width + x) * 3;
            if (bgr)
            {
                destination[0] = source[2];
                destination[1] = source[1];
                destination[2] = source[0];
            }
            else
            {
                destination[0] = source[0];
                destination[1] = source[1];
                destination[2] = source[2];
            }
        }
    return true;
}

#if _WIN32
void ensure_wic_initialized()
{
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
}

std::wstring wide_path(const std::string& path)
{
    const int length = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), (int)path.size(), nullptr, 0);
    if (length <= 0)
        return std::wstring(path.begin(), path.end());
    std::wstring result((size_t)length, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), (int)path.size(), result.data(), length);
    return result;
}

void rgb_to_bgr(
    const unsigned char* source, int width, int height, int channels, std::vector<unsigned char>& destination)
{
    destination.resize((size_t)width * height * channels);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            const unsigned char* input = source + ((size_t)y * width + x) * channels;
            unsigned char* output = destination.data() + ((size_t)y * width + x) * channels;
            output[0] = input[2];
            output[1] = input[1];
            output[2] = input[0];
            if (channels == 4)
                output[3] = input[3];
        }
}
#endif

bool save_png(const std::string& path, int width, int height, int channels, const unsigned char* pixels)
{
#if _WIN32
    ensure_wic_initialized();
    std::vector<unsigned char> bgr;
    rgb_to_bgr(pixels, width, height, channels, bgr);
    const std::wstring output = wide_path(path);
    return wic_encode_image(output.c_str(), width, height, channels, bgr.data()) != 0;
#else
    return png_save(path.c_str(), width, height, channels, pixels) != 0;
#endif
}

} // namespace

bool image_size(const std::string& path, int& width, int& height)
{
    FILE* file = std::fopen(path.c_str(), "rb");
    if (!file)
        return false;
    unsigned char header[64] {};
    const size_t count = std::fread(header, 1, sizeof(header), file);
    if (count >= 24 && !png_sig_cmp(header, 0, 8))
    {
        width = static_cast<int>(png_get_uint_32(header + 16));
        height = static_cast<int>(png_get_uint_32(header + 20));
        std::fclose(file);
        return width > 0 && height > 0;
    }
    if (count >= 12 && WebPGetInfo(header, count, &width, &height))
    {
        std::fclose(file);
        return width > 0 && height > 0;
    }
    if (count < 2 || header[0] != 0xff || header[1] != 0xd8)
    {
        std::fclose(file);
        return false;
    }
    std::rewind(file);
    jpeg_decompress_struct decoder {};
    safe_jpeg_error_mgr error {};
    decoder.err = jpeg_std_error(&error.pub);
    decoder.err->error_exit = safe_jpeg_error_exit;
    volatile bool created = false;
    if (setjmp(error.setjmp_buffer))
    {
        if (created)
            jpeg_destroy_decompress(&decoder);
        std::fclose(file);
        return false;
    }
    jpeg_create_decompress(&decoder);
    created = true;
    jpeg_stdio_src(&decoder, file);
    const bool ok = jpeg_read_header(&decoder, TRUE) == JPEG_HEADER_OK;
    width = static_cast<int>(decoder.image_width);
    height = static_cast<int>(decoder.image_height);
    jpeg_destroy_decompress(&decoder);
    std::fclose(file);
    return ok && width > 0 && height > 0;
}

static bool decode_rgb_file(const std::string& path, ncnn::Mat& image)
{
    const std::string extension = lowercase_path(
        path.substr(path.find_last_of('.') == std::string::npos ? path.size() : path.find_last_of('.') + 1));

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* pixels = nullptr;

#if _WIN32
    if (extension != "webp")
    {
        ensure_wic_initialized();
        const std::wstring input = wide_path(path);

        pixels = wic_decode_image(input.c_str(), &width, &height, &channels);
        if (pixels)
        {
            const bool ok = assign_rgb(pixels, width, height, channels, image, true);
            std::free(pixels);
            return ok;
        }
    }
#endif

    std::vector<unsigned char> bytes;
    if (read_file_bytes(path, bytes) != 0)
        return false;

    if (extension == "png")
        pixels = png_load(bytes.data(), (int)bytes.size(), &width, &height, &channels);
    else if (extension == "jpg" || extension == "jpeg")
        pixels = jpeg_load(bytes.data(), (int)bytes.size(), &width, &height, &channels);
    else if (extension == "webp")
        pixels = webp_load(bytes.data(), (int)bytes.size(), &width, &height, &channels);

    if (!pixels)
        pixels = png_load(bytes.data(), (int)bytes.size(), &width, &height, &channels);
    if (!pixels)
        pixels = jpeg_load(bytes.data(), (int)bytes.size(), &width, &height, &channels);
    if (!pixels)
        pixels = webp_load(bytes.data(), (int)bytes.size(), &width, &height, &channels);
    if (!pixels)
        return false;

    const bool ok = assign_rgb(pixels, width, height, channels, image, false);
    std::free(pixels);
    return ok;
}

static bool write_rgb_file(const std::string& path, const ncnn::Mat& image)
{
    const size_t dot = path.find_last_of('.');
    const std::string ext = dot == std::string::npos ? "" : lowercase_path(path.substr(dot + 1));
#if _WIN32
    const std::wstring target = wide_path(path);
#else
    const std::string& target = path;
#endif
    if (ext == "jpg" || ext == "jpeg")
        return jpeg_save(target.c_str(), image.w, image.h, 3, static_cast<unsigned char*>(image.data)) != 0;
    if (ext == "webp")
        return webp_save(target.c_str(), image.w, image.h, 3, static_cast<unsigned char*>(image.data)) != 0;
    if (ext == "png")
        return save_png(path, image.w, image.h, 3, static_cast<unsigned char*>(image.data));
    return false;
}

std::string make_batch_output_path(const std::string& path, int b, int batch)
{
    if (batch <= 1)
        return path;

    const size_t slash = path.find_last_of("/\\");
    const size_t dot = path.find_last_of('.');
    const bool has_extension = dot != std::string::npos && (slash == std::string::npos || dot > slash);
    const std::string suffix = "-" + std::to_string(b);
    if (!has_extension)
        return path + suffix;
    return path.substr(0, dot) + suffix + path.substr(dot);
}

} // namespace h3image

// h3-image implemented with ncnn library
#include "filesystem_utils.h"
#include "image_io.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>
namespace h3image
{
namespace
{
float sinc(float x)
{
    if (std::abs(x) < 1e-7f)
        return 1.f;
    const float pix = 3.14159265358979323846f * x;
    return std::sin(pix) / pix;
}

float lanczos(float x)
{
    x = std::abs(x);
    return x < 3.f ? sinc(x) * sinc(x / 3.f) : 0.f;
}

struct Weight
{
    int source;
    float value;
};

std::vector<std::vector<Weight>> make_weights(std::string& error, int source_size, int target_size)
{
    const float scale = (float)source_size / target_size;
    const float filter_scale = std::max(1.f, scale);
    const float support = 3.f * filter_scale;
    std::vector<std::vector<Weight>> result(target_size);
    for (int out = 0; out < target_size; out++)
    {
        const float center = (out + .5f) * scale;
        const int first = std::max(0, (int)std::floor(center - support));
        const int last = std::min(source_size, (int)std::ceil(center + support));
        float sum = 0.f;
        auto& weights = result[out];
        weights.reserve(last - first);
        for (int i = first; i < last; i++)
        {
            const float value = lanczos(((i + .5f) - center) / filter_scale);
            if (value != 0.f)
            {
                weights.push_back({ i, value });
                sum += value;
            }
        }
        if (weights.empty() || sum == 0.f)
        {
            error = "failed to construct image resize filter";
            return {};
        }
        for (auto& weight : weights)
            weight.value /= sum;
    }
    return result;
}
}

ncnn::Mat load_rgb_image(std::string& error, const std::string& path)
{
    ncnn::Mat decoded;
    if (!decode_rgb_file(path, decoded))
    {
        error = "failed to load image: " + path;
        return {};
    }
    if (decoded.w <= 0 || decoded.h <= 0 || decoded.w > 32768 || decoded.h > 32768)
    {
        error = "invalid image dimensions: " + path;
        return {};
    }
    return decoded;
}
ncnn::Mat resize_rgb_lanczos(std::string& error, const ncnn::Mat& image, int width, int height)
{
    if (!is_rgb_image(image) || width <= 0 || height <= 0)
    {
        error = "invalid image resize dimensions";
        return {};
    }
    if (image.w == width && image.h == height)
        return image;
    if (width > 32768 || height > 32768)
    {
        error = "requested image dimensions are too large";
        return {};
    }
    const auto horizontal = make_weights(error, image.w, width);
    if (!error.empty())
        return {};

    const auto vertical = make_weights(error, image.h, height);
    if (!error.empty())
        return {};

    std::vector<float> intermediate((size_t)width * image.h * 3);
    for (int y = 0; y < image.h; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++)
            {
                float value = 0.f;
                for (const auto& weight : horizontal[x])
                    value += weight.value
                        * static_cast<unsigned char*>(image.data)[((size_t)y * image.w + weight.source) * 3 + c];
                intermediate[((size_t)y * width + x) * 3 + c] = value;
            }
    ncnn::Mat result(width, height, static_cast<size_t>(3), 3);
    if (result.empty())
    {
        error = "cannot allocate resized RGB image";
        return {};
    }
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++)
            {
                float value = 0.f;
                for (const auto& weight : vertical[y])
                    value += weight.value * intermediate[((size_t)weight.source * width + x) * 3 + c];
                static_cast<unsigned char*>(result.data)[((size_t)y * width + x) * 3 + c]
                    = (uint8_t)std::clamp(std::lround(value), 0l, 255l);
            }
    return result;
}

void save_rgb_image(std::string& error, const std::string& path, const ncnn::Mat& image)
{
    if (!is_rgb_image(image))
    {
        error = "invalid output image";
        return;
    }
    const std::filesystem::path output(path);
    if (!output.parent_path().empty())
    {
        fsutil::create_directories(error, output.parent_path());
        if (!error.empty())
            return;
    }

    if (!write_rgb_file(path, image))
    {
        error = "failed to save image: " + path;
        return;
    }
}
}
