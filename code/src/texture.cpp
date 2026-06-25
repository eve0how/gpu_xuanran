#include "texture.hpp"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>

#pragma pack(push, 1)
struct BMPHeader {
    char bfType[2];
    unsigned int bfSize;
    unsigned int bfReserved;
    unsigned int bfOffBits;
    unsigned int biSize;
    int biWidth;
    int biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int biCompression;
    unsigned int biSizeImage;
    int biXPelsPerMeter;
    int biYPelsPerMeter;
    unsigned int biClrUsed;
    unsigned int biClrImportant;
};
#pragma pack(pop)

static unsigned int hashPixel(int x, int y, int salt) {
    unsigned int h = static_cast<unsigned int>(x * 374761393 + y * 668265263 + salt * 1274126177);
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return h;
}

static float hash01(int x, int y, int salt) {
    return (hashPixel(x, y, salt) & 0xFFFFFFu) / float(0x1000000u);
}

static void encodeNormal(unsigned char *out, float nx, float ny, float nz) {
    out[0] = static_cast<unsigned char>(std::max(0, std::min(255, static_cast<int>((nx * 0.5f + 0.5f) * 255.0f))));
    out[1] = static_cast<unsigned char>(std::max(0, std::min(255, static_cast<int>((ny * 0.5f + 0.5f) * 255.0f))));
    out[2] = static_cast<unsigned char>(std::max(0, std::min(255, static_cast<int>((nz * 0.5f + 0.5f) * 255.0f))));
}

static bool writeRGBBMP(const char *filename, int w, int h, const unsigned char *rgb) {
    FILE *file = fopen(filename, "wb");
    if (file == nullptr) {
        return false;
    }

    int bytesPerLine = ((w * 3 + 3) / 4) * 4;
    BMPHeader header{};
    header.bfType[0] = 'B';
    header.bfType[1] = 'M';
    header.bfOffBits = sizeof(BMPHeader);
    header.bfSize = header.bfOffBits + bytesPerLine * h;
    header.biSize = 40;
    header.biWidth = w;
    header.biHeight = h;
    header.biPlanes = 1;
    header.biBitCount = 24;
    header.biCompression = 0;
    header.biSizeImage = bytesPerLine * h;

    fwrite(&header, sizeof(header), 1, file);
    unsigned char *line = new unsigned char[bytesPerLine];
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int src = 3 * (y * w + x);
            line[3 * x] = rgb[src];
            line[3 * x + 1] = rgb[src + 1];
            line[3 * x + 2] = rgb[src + 2];
        }
        fwrite(line, bytesPerLine, 1, file);
    }
    delete[] line;
    fclose(file);
    return true;
}

Texture *Texture::loadBMP(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == nullptr) {
        fprintf(stderr, "Cannot open texture: %s\n", filename);
        return nullptr;
    }

    BMPHeader header;
    if (fread(&header, sizeof(header), 1, file) != 1 ||
        header.bfType[0] != 'B' || header.bfType[1] != 'M' ||
        header.biBitCount != 24 || header.biCompression != 0) {
        fprintf(stderr, "Unsupported BMP texture: %s\n", filename);
        fclose(file);
        return nullptr;
    }

    int w = header.biWidth;
    int h = std::abs(header.biHeight);
    bool topDown = header.biHeight < 0;
    int rowBytes = ((w * 3 + 3) / 4) * 4;

    auto *tex = new Texture();
    tex->width = w;
    tex->height = h;
    tex->pixels = new Vector3f[w * h];

    unsigned char *row = new unsigned char[rowBytes];
    for (int y = 0; y < h; ++y) {
        fread(row, 1, rowBytes, file);
        int dstY = topDown ? y : (h - 1 - y);
        for (int x = 0; x < w; ++x) {
            unsigned char b = row[3 * x];
            unsigned char g = row[3 * x + 1];
            unsigned char r = row[3 * x + 2];
            tex->pixels[dstY * w + x] = Vector3f(r / 255.0f, g / 255.0f, b / 255.0f);
        }
    }
    delete[] row;
    fclose(file);
    return tex;
}

bool Texture::writeCheckerboardBMP(const char *filename, int size, int cells) {
    FILE *file = fopen(filename, "wb");
    if (file == nullptr) {
        return false;
    }

    int bytesPerLine = ((size * 3 + 3) / 4) * 4;
    BMPHeader header{};
    header.bfType[0] = 'B';
    header.bfType[1] = 'M';
    header.bfOffBits = sizeof(BMPHeader);
    header.bfSize = header.bfOffBits + bytesPerLine * size;
    header.biSize = 40;
    header.biWidth = size;
    header.biHeight = size;
    header.biPlanes = 1;
    header.biBitCount = 24;
    header.biCompression = 0;
    header.biSizeImage = bytesPerLine * size;

    fwrite(&header, sizeof(header), 1, file);

    unsigned char *line = new unsigned char[bytesPerLine];
    int cellSize = std::max(1, size / cells);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            bool dark = ((x / cellSize) + (y / cellSize)) % 2 == 0;
            unsigned char c = dark ? 60 : 220;
            line[3 * x] = c;
            line[3 * x + 1] = c;
            line[3 * x + 2] = c;
        }
        fwrite(line, bytesPerLine, 1, file);
    }
    delete[] line;
    fclose(file);
    return true;
}

bool Texture::writeBrickAlbedoBMP(const char *filename, int size) {
    unsigned char *rgb = new unsigned char[size * size * 3];
    int rows = 4;
    int cols = 6;
    int mortar = std::max(2, size / 96);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int cellX = x * cols % size;
            int cellY = y * rows % size;
            int localX = cellX * cols / size;
            int localY = cellY * rows / size;
            bool isMortar = localX < mortar || localY < mortar;
            float n = hash01(x, y, 11);
            unsigned char r, g, b;
            if (isMortar) {
                float m = 0.62f + 0.04f * n;
                r = static_cast<unsigned char>(m * 255);
                g = static_cast<unsigned char>((m - 0.01f) * 255);
                b = static_cast<unsigned char>((m - 0.03f) * 255);
            } else {
                r = static_cast<unsigned char>((0.58f + 0.06f * n) * 255);
                g = static_cast<unsigned char>((0.30f + 0.04f * hash01(x, y, 23)) * 255);
                b = static_cast<unsigned char>((0.22f + 0.03f * hash01(x, y, 37)) * 255);
            }
            int idx = 3 * (y * size + x);
            rgb[idx] = b;
            rgb[idx + 1] = g;
            rgb[idx + 2] = r;
        }
    }
    bool ok = writeRGBBMP(filename, size, size, rgb);
    delete[] rgb;
    return ok;
}

bool Texture::writeBrickNormalBMP(const char *filename, int size) {
    float *height = new float[size * size];
    int rows = 4;
    int cols = 6;
    int mortar = std::max(2, size / 96);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int cellX = x * cols % size;
            int cellY = y * rows % size;
            int localX = cellX * cols / size;
            int localY = cellY * rows / size;
            bool isMortar = localX < mortar || localY < mortar;
            height[y * size + x] = isMortar ? 0.15f : 0.85f + 0.08f * hash01(x, y, 91);
        }
    }
    unsigned char *rgb = new unsigned char[size * size * 3];
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float hL = height[y * size + std::max(0, x - 1)];
            float hR = height[y * size + std::min(size - 1, x + 1)];
            float hD = height[std::max(0, y - 1) * size + x];
            float hU = height[std::min(size - 1, y + 1) * size + x];
            float dx = (hR - hL) * 2.5f;
            float dy = (hU - hD) * 2.5f;
            Vector3f n(-dx, -dy, 1.0f);
            n.normalize();
            encodeNormal(rgb + 3 * (y * size + x), n[0], n[1], n[2]);
        }
    }
    bool ok = writeRGBBMP(filename, size, size, rgb);
    delete[] height;
    delete[] rgb;
    return ok;
}

bool Texture::writeWoodAlbedoBMP(const char *filename, int size) {
    unsigned char *rgb = new unsigned char[size * size * 3];
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float u = float(x) / size;
            float v = float(y) / size;
            float grain = sinf((u * 8.0f + sinf(v * 3.0f) * 1.2f) * 6.28318f);
            grain += 0.20f * sinf((u * 18.0f + v * 2.0f) * 6.28318f);
            float tone = 0.48f + 0.08f * grain + 0.03f * hash01(x, y, 51);
            unsigned char r = static_cast<unsigned char>(std::min(255.0f, tone * 1.02f * 255.0f));
            unsigned char g = static_cast<unsigned char>(std::min(255.0f, tone * 0.78f * 255.0f));
            unsigned char b = static_cast<unsigned char>(std::min(255.0f, tone * 0.48f * 255.0f));
            int idx = 3 * (y * size + x);
            rgb[idx] = b;
            rgb[idx + 1] = g;
            rgb[idx + 2] = r;
        }
    }
    bool ok = writeRGBBMP(filename, size, size, rgb);
    delete[] rgb;
    return ok;
}

bool Texture::writeStoneAlbedoBMP(const char *filename, int size) {
    unsigned char *rgb = new unsigned char[size * size * 3];
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float n1 = hash01(x / 4, y / 4, 3);
            float n2 = hash01(x / 10, y / 10, 17);
            float n3 = hash01(x / 2, y / 2, 29);
            float tone = 0.58f + 0.08f * n1 + 0.05f * n2 + 0.03f * n3;
            int idx = 3 * (y * size + x);
            rgb[idx] = static_cast<unsigned char>(std::min(255.0f, tone * 0.94f * 255.0f));
            rgb[idx + 1] = static_cast<unsigned char>(std::min(255.0f, tone * 0.97f * 255.0f));
            rgb[idx + 2] = static_cast<unsigned char>(std::min(255.0f, tone * 1.04f * 255.0f));
        }
    }
    bool ok = writeRGBBMP(filename, size, size, rgb);
    delete[] rgb;
    return ok;
}

bool Texture::writeStoneNormalBMP(const char *filename, int size) {
    float *height = new float[size * size];
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float n1 = hash01(x / 3, y / 3, 41);
            float n2 = hash01(x / 7, y / 7, 59);
            height[y * size + x] = 0.45f + 0.35f * n1 + 0.20f * n2;
        }
    }
    unsigned char *rgb = new unsigned char[size * size * 3];
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float hL = height[y * size + std::max(0, x - 1)];
            float hR = height[y * size + std::min(size - 1, x + 1)];
            float hD = height[std::max(0, y - 1) * size + x];
            float hU = height[std::min(size - 1, y + 1) * size + x];
            float dx = (hR - hL) * 1.8f;
            float dy = (hU - hD) * 1.8f;
            Vector3f n(-dx, -dy, 1.0f);
            n.normalize();
            encodeNormal(rgb + 3 * (y * size + x), n[0], n[1], n[2]);
        }
    }
    bool ok = writeRGBBMP(filename, size, size, rgb);
    delete[] height;
    delete[] rgb;
    return ok;
}

bool Texture::generateShowcaseTextures(const char *dir) {
    std::string base(dir);
    if (!base.empty() && base.back() != '/') {
        base += '/';
    }
    bool ok = true;
    ok &= writeBrickAlbedoBMP((base + "brick_albedo.bmp").c_str());
    ok &= writeBrickNormalBMP((base + "brick_normal.bmp").c_str());
    ok &= writeWoodAlbedoBMP((base + "wood_albedo.bmp").c_str());
    ok &= writeStoneAlbedoBMP((base + "stone_albedo.bmp").c_str());
    ok &= writeStoneNormalBMP((base + "stone_normal.bmp").c_str());
    return ok;
}
