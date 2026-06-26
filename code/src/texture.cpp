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

static unsigned char clampByte(float x) {
    int v = static_cast<int>(x * 255.0f + 0.5f);
    if (v < 0) {
        v = 0;
    }
    if (v > 255) {
        v = 255;
    }
    return static_cast<unsigned char>(v);
}

static unsigned int hashPixel(int x, int y, int salt) {
    unsigned int h = static_cast<unsigned int>(x * 374761393 + y * 668265263 + salt * 1274126177);
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return h;
}

static float hash01(int x, int y, int salt) {
    return (hashPixel(x, y, salt) & 0xFFFFFFu) / float(0x1000000u);
}

static float smoothNoise(float x, float y, int salt) {
    int ix = static_cast<int>(floorf(x));
    int iy = static_cast<int>(floorf(y));
    float fx = x - ix;
    float fy = y - iy;
    float sx = fx * fx * (3.0f - 2.0f * fx);
    float sy = fy * fy * (3.0f - 2.0f * fy);
    float a = hash01(ix, iy, salt);
    float b = hash01(ix + 1, iy, salt);
    float c = hash01(ix, iy + 1, salt);
    float d = hash01(ix + 1, iy + 1, salt);
    return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
}

static float fbm(float x, float y, int octaves, int salt) {
    float sum = 0.0f;
    float amp = 0.5f;
    float freq = 1.0f;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * smoothNoise(x * freq, y * freq, salt + i * 17);
        amp *= 0.5f;
        freq *= 2.0f;
    }
    return sum;
}

static void encodeNormal(unsigned char *out, float nx, float ny, float nz) {
    // BMP stores B,G,R at out[0..2]; loadBMP -> Vector3f(r,g,b); sampleNormal reads r->nx, g->ny, b->nz.
    out[0] = static_cast<unsigned char>(std::max(0, std::min(255, static_cast<int>((nz * 0.5f + 0.5f) * 255.0f))));
    out[1] = static_cast<unsigned char>(std::max(0, std::min(255, static_cast<int>((ny * 0.5f + 0.5f) * 255.0f))));
    out[2] = static_cast<unsigned char>(std::max(0, std::min(255, static_cast<int>((nx * 0.5f + 0.5f) * 255.0f))));
}

static float sampleHeightTile(const float *height, int size, int x, int y) {
    x = ((x % size) + size) % size;
    y = ((y % size) + size) % size;
    return height[y * size + x];
}

static void heightToNormalMap(const float *height, int size, unsigned char *rgb, float strength,
                              bool tile = false) {
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float hL, hR, hD, hU;
            if (tile) {
                hL = sampleHeightTile(height, size, x - 1, y);
                hR = sampleHeightTile(height, size, x + 1, y);
                hD = sampleHeightTile(height, size, x, y - 1);
                hU = sampleHeightTile(height, size, x, y + 1);
            } else {
                hL = height[y * size + std::max(0, x - 1)];
                hR = height[y * size + std::min(size - 1, x + 1)];
                hD = height[std::max(0, y - 1) * size + x];
                hU = height[std::min(size - 1, y + 1) * size + x];
            }
            float dx = (hR - hL) * strength;
            float dy = (hU - hD) * strength;
            Vector3f n(-dx, -dy, 1.0f);
            n.normalize();
            encodeNormal(rgb + 3 * (y * size + x), n[0], n[1], n[2]);
        }
    }
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
    int rows = 3;
    int cols = 5;
    int mortar = std::max(3, size / 80);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float u = float(x) / float(size);
            float v = float(y) / float(size);
            int row = static_cast<int>(v * rows) % rows;
            int col = static_cast<int>(u * cols) % cols;
            float rowShift = (row % 2 == 0) ? 0.0f : 0.5f / cols;
            float localU = u * cols + rowShift;
            localU -= floorf(localU);
            float localV = v * rows - floorf(v * rows);
            bool isMortar = localU < float(mortar) / size * cols ||
                            localV < float(mortar) / size * rows;
            float n = fbm(u * 8.0f, v * 8.0f, 3, 11);
            float r, g, b;
            if (isMortar) {
                float m = 0.68f + 0.03f * n;
                r = m;
                g = m - 0.015f;
                b = m - 0.03f;
            } else {
                r = 0.62f + 0.04f * n;
                g = 0.38f + 0.03f * fbm(u * 12.0f, v * 12.0f, 2, 23);
                b = 0.30f + 0.02f * fbm(u * 10.0f, v * 10.0f, 2, 37);
            }
            int idx = 3 * (y * size + x);
            rgb[idx] = clampByte(b);
            rgb[idx + 1] = clampByte(g);
            rgb[idx + 2] = clampByte(r);
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

bool Texture::writePlasterAlbedoBMP(const char *filename, int size) {
    unsigned char *rgb = new unsigned char[size * size * 3];
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float u = float(x) / float(size);
            float v = float(y) / float(size);
            float coarse = fbm(u * 5.0f, v * 5.0f, 4, 101);
            float fine = fbm(u * 18.0f, v * 18.0f, 3, 113);
            float speck = fbm(u * 42.0f, v * 42.0f, 2, 127);
            float tone = 0.58f + 0.24f * (coarse - 0.5f) + 0.16f * (fine - 0.5f) +
                         0.10f * (speck - 0.5f);
            float r = std::min(1.0f, tone + 0.10f);
            float g = std::min(1.0f, tone + 0.03f);
            float b = std::max(0.0f, tone - 0.12f);
            int idx = 3 * (y * size + x);
            rgb[idx] = clampByte(b);
            rgb[idx + 1] = clampByte(g);
            rgb[idx + 2] = clampByte(r);
        }
    }
    bool ok = writeRGBBMP(filename, size, size, rgb);
    delete[] rgb;
    return ok;
}

bool Texture::writePlasterNormalBMP(const char *filename, int size) {
    float *height = new float[size * size];
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float u = float(x) / float(size);
            float v = float(y) / float(size);
            float coarse = fbm(u * 5.0f, v * 5.0f, 4, 127);
            float fine = fbm(u * 18.0f, v * 18.0f, 3, 139);
            float micro = fbm(u * 42.0f, v * 42.0f, 2, 149);
            height[y * size + x] = 0.5f + 0.55f * (coarse - 0.5f) + 0.32f * (fine - 0.5f) +
                                   0.16f * (micro - 0.5f);
        }
    }
    unsigned char *rgb = new unsigned char[size * size * 3];
    heightToNormalMap(height, size, rgb, 5.5f, true);
    bool ok = writeRGBBMP(filename, size, size, rgb);
    delete[] height;
    delete[] rgb;
    return ok;
}

bool Texture::writeMarbleAlbedoBMP(const char *filename, int size) {
    unsigned char *rgb = new unsigned char[size * size * 3];
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float u = float(x) / float(size);
            float v = float(y) / float(size);
            float warp = fbm(u * 2.8f, v * 2.8f, 4, 151);
            float warp2 = fbm(u * 6.5f + warp, v * 6.5f - warp, 3, 157);
            float vein1 = sinf((u * 4.2f + warp * 2.5f) * 6.28318f +
                               sinf((v + warp) * 7.5f) * 2.0f);
            float vein2 = cosf((v * 3.8f - u * 2.2f + warp2 * 2.8f) * 6.28318f +
                               sinf(u * 11.0f + warp) * 1.4f);
            float vein3 = sinf((u * 9.0f + v * 5.5f + warp * 4.0f) * 6.28318f);
            float veinField = 0.45f * vein1 + 0.35f * vein2 + 0.20f * vein3;
            float veinMask = 0.5f + 0.5f * veinField;
            veinMask = std::max(0.0f, std::min(1.0f, (veinMask - 0.28f) * 1.65f));
            float grain = fbm(u * 22.0f, v * 22.0f, 3, 163);
            float base = 0.68f + 0.10f * (grain - 0.5f) + 0.06f * (warp - 0.5f);
            float r = base + 0.06f * (1.0f - veinMask) - 0.22f * veinMask;
            float g = base + 0.02f * (1.0f - veinMask) - 0.18f * veinMask;
            float b = base + 0.14f * (1.0f - veinMask) + 0.10f * veinMask;
            r = std::max(0.22f, std::min(0.92f, r));
            g = std::max(0.24f, std::min(0.90f, g));
            b = std::max(0.30f, std::min(0.95f, b));
            int idx = 3 * (y * size + x);
            rgb[idx] = clampByte(b);
            rgb[idx + 1] = clampByte(g);
            rgb[idx + 2] = clampByte(r);
        }
    }
    bool ok = writeRGBBMP(filename, size, size, rgb);
    delete[] rgb;
    return ok;
}

bool Texture::writeWoodAlbedoBMP(const char *filename, int size) {
    unsigned char *rgb = new unsigned char[size * size * 3];
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float u = float(x) / size;
            float v = float(y) / size;
            float grain = sinf((u * 3.5f + sinf(v * 2.0f) * 0.6f) * 6.28318f);
            grain += 0.08f * sinf((u * 9.0f + v * 1.5f) * 6.28318f);
            float tone = 0.52f + 0.05f * grain + 0.02f * fbm(u * 10.0f, v * 10.0f, 2, 51);
            float r = std::min(1.0f, tone * 1.02f);
            float g = std::min(1.0f, tone * 0.82f);
            float b = std::min(1.0f, tone * 0.55f);
            int idx = 3 * (y * size + x);
            rgb[idx] = clampByte(b);
            rgb[idx + 1] = clampByte(g);
            rgb[idx + 2] = clampByte(r);
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

bool Texture::writeEarthAlbedoBMP(const char *filename, int size) {
    unsigned char *rgb = new unsigned char[size * size * 3];
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float u = float(x) / float(size);
            float v = float(y) / float(size);
            float lon = (u - 0.5f) * 6.28318f;
            float lat = (0.5f - v) * 3.14159f;
            float n = hash01(x / 3, y / 3, 71);
            float n2 = hash01(x / 7, y / 7, 83);
            float landMask = 0.35f * sinf(lon * 2.2f + sinf(lat * 3.0f) * 0.8f) +
                             0.25f * cosf(lat * 4.5f + lon * 1.3f) +
                             0.22f * n + 0.18f * n2;
            bool land = landMask > 0.08f;
            float r, g, b;
            if (land) {
                float coast = std::max(0.0f, 1.0f - fabsf(landMask - 0.08f) * 6.0f);
                r = 0.18f + 0.22f * n + 0.08f * coast;
                g = 0.42f + 0.28f * hash01(x, y, 97) + 0.06f * coast;
                b = 0.10f + 0.08f * n2;
            } else {
                float depth = 0.55f + 0.25f * n;
                r = 0.03f + 0.04f * n2;
                g = 0.12f + 0.18f * depth;
                b = 0.38f + 0.35f * depth;
            }
            int idx = 3 * (y * size + x);
            rgb[idx] = clampByte(b);
            rgb[idx + 1] = clampByte(g);
            rgb[idx + 2] = clampByte(r);
        }
    }
    bool ok = writeRGBBMP(filename, size, size, rgb);
    delete[] rgb;
    return ok;
}

bool Texture::generateShowcaseTextures(const char *dir) {
    std::string base(dir);
    if (!base.empty() && base.back() != '/') {
        base += '/';
    }
    bool ok = true;
    ok &= writePlasterAlbedoBMP((base + "plaster_albedo.bmp").c_str());
    ok &= writePlasterNormalBMP((base + "plaster_normal.bmp").c_str());
    ok &= writeMarbleAlbedoBMP((base + "marble_albedo.bmp").c_str());
    ok &= writeBrickAlbedoBMP((base + "brick_albedo.bmp").c_str());
    ok &= writeBrickNormalBMP((base + "brick_normal.bmp").c_str());
    ok &= writeWoodAlbedoBMP((base + "wood_albedo.bmp").c_str());
    ok &= writeStoneAlbedoBMP((base + "stone_albedo.bmp").c_str());
    ok &= writeStoneNormalBMP((base + "stone_normal.bmp").c_str());
    ok &= writeEarthAlbedoBMP((base + "earth_albedo.bmp").c_str());
    return ok;
}
