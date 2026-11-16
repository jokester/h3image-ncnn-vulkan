#ifndef PNG_IMAGE_H
#define PNG_IMAGE_H

// png image decoder and encoder with libpng
#include "png.h"
#include <limits>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>

struct png_read_context
{
    const unsigned char* indata;
    unsigned int inlen;
    unsigned int current;
};

static void png_read_fn(png_structp png_ptr, png_bytep dst, png_size_t nread)
{
    struct png_read_context* prc = (struct png_read_context*)(png_get_io_ptr(png_ptr));
    if (prc->current > prc->inlen || nread > prc->inlen - prc->current)
        png_error(png_ptr, "truncated PNG data");

    memcpy(dst, prc->indata + prc->current, nread);
    prc->current += nread;
}

unsigned char* png_load(const unsigned char* buffer, int len, int* w, int* h, int* c)
{
    if (len < 8 || png_sig_cmp(buffer, 0, 8))
        return NULL;

    unsigned char* volatile pixeldata = 0;
    png_bytepp volatile row_pointers = NULL;

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return NULL;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return NULL;
    }

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        free(row_pointers);
        free(pixeldata);
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return NULL;
    }

    struct png_read_context prc = { buffer, (unsigned int)len, 0 };
    png_set_read_fn(png_ptr, &prc, png_read_fn);

    png_read_info(png_ptr, info_ptr);

    int width = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);

    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    // always output RGB(A) 8bit
    if (bit_depth == 16)
        png_set_strip_16(png_ptr);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);

    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);

    int has_alpha = 0;
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        has_alpha = 1;
    if (color_type == PNG_COLOR_TYPE_RGBA || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        has_alpha = 1;

    if (has_alpha)
        png_set_tRNS_to_alpha(png_ptr);

    png_read_update_info(png_ptr, info_ptr);

    int channels = has_alpha ? 4 : 3;

    if (width <= 0 || height <= 0 || (size_t)width > (std::numeric_limits<size_t>::max)() / height / channels
        || (size_t)height > (std::numeric_limits<size_t>::max)() / sizeof(png_bytep))
        png_error(png_ptr, "PNG dimensions exceed supported storage");
    pixeldata = (unsigned char*)malloc((size_t)width * height * channels);
    row_pointers = (png_bytepp)malloc((size_t)height * sizeof(png_bytep));
    if (!pixeldata || !row_pointers)
        png_error(png_ptr, "cannot allocate PNG image storage");
    for (int i = 0; i < height; i++)
        row_pointers[i] = pixeldata + (size_t)i * width * channels;

    png_read_image(png_ptr, row_pointers);

    png_read_end(png_ptr, info_ptr);

    free(row_pointers);
    row_pointers = NULL;

    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

    *w = width;
    *h = height;
    *c = channels;

    return pixeldata;
}

#if _WIN32
int png_save(const wchar_t* filepath, int w, int h, int c, const unsigned char* pixeldata)
#else
int png_save(const char* filepath, int w, int h, int c, const unsigned char* pixeldata)
#endif
{
    if (w <= 0 || h <= 0 || (c != 3 && c != 4) || !pixeldata)
        return 0;
    FILE* volatile fp = NULL;
    png_bytepp volatile row_pointers = NULL;
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 0;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        png_destroy_write_struct(&png_ptr, NULL);
        return 0;
    }

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        free(row_pointers);
        if (fp)
            fclose(fp);
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return 0;
    }

#if _WIN32
    fp = _wfopen(filepath, L"wb");
#else
    fp = fopen(filepath, "wb");
#endif
    if (!fp)
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return 0;
    }

    png_init_io(png_ptr, fp);

    png_set_IHDR(png_ptr, info_ptr, w, h, 8, c == 3 ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png_ptr, info_ptr);

    row_pointers = (png_bytepp)malloc((size_t)h * sizeof(png_bytep));
    if (!row_pointers)
        png_error(png_ptr, "cannot allocate PNG row pointers");
    for (int i = 0; i < h; i++)
        row_pointers[i] = const_cast<png_bytep>(pixeldata + (size_t)i * w * c);

    png_write_image(png_ptr, row_pointers);

    png_write_end(png_ptr, info_ptr);

    free(row_pointers);
    row_pointers = NULL;

    png_destroy_write_struct(&png_ptr, &info_ptr);

    return fclose(fp) == 0;
}
#endif // PNG_IMAGE_H
