#ifndef TEXTURE_H
#define TEXTURE_H

#include <vecmath.h>
#include <string>

// 2D texture loaded from 24-bit BMP; samples in [0,1]^2 with repeat wrapping.
class Texture {
public:
    Texture() : width(0), height(0), pixels(nullptr) {}

    ~Texture() {
        delete[] pixels;
    }

    Texture(const Texture &) = delete;
    Texture &operator=(const Texture &) = delete;

    static Texture *loadBMP(const char *filename);

    static bool writeCheckerboardBMP(const char *filename, int size, int cells);

    static bool writeBrickAlbedoBMP(const char *filename, int size = 512);
    static bool writeBrickNormalBMP(const char *filename, int size = 512);
    static bool writeWoodAlbedoBMP(const char *filename, int size = 512);
    static bool writeStoneAlbedoBMP(const char *filename, int size = 512);
    static bool writeStoneNormalBMP(const char *filename, int size = 512);
    static bool generateShowcaseTextures(const char *dir);

    bool valid() const {
        return pixels != nullptr && width > 0 && height > 0;
    }

    Vector3f sample(float u, float v) const {
        if (!valid()) {
            return Vector3f(1.0f, 1.0f, 1.0f);
        }
        u = u - floorf(u);
        v = v - floorf(v);
        if (u < 0.0f) {
            u += 1.0f;
        }
        if (v < 0.0f) {
            v += 1.0f;
        }
        int x = std::min(width - 1, std::max(0, static_cast<int>(u * width)));
        int y = std::min(height - 1, std::max(0, static_cast<int>((1.0f - v) * height)));
        return pixels[y * width + x];
    }

    // Tangent-space normal from RGB normal map in [0,1]^3.
    Vector3f sampleNormal(float u, float v) const {
        Vector3f c = sample(u, v);
        Vector3f n(c[0] * 2.0f - 1.0f, c[1] * 2.0f - 1.0f, c[2] * 2.0f - 1.0f);
        float len = n.length();
        if (len < 1e-8f) {
            return Vector3f(0.0f, 0.0f, 1.0f);
        }
        return n / len;
    }

private:
    int width;
    int height;
    Vector3f *pixels;
};

#endif // TEXTURE_H
