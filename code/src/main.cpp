#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "scene_parser.hpp"
#include "image.hpp"
#include "camera.hpp"
#include "group.hpp"
#include "light.hpp"
#include "raytracer.hpp"

#ifdef USE_CUDA
#include "cuda_renderer.hpp"
#endif

#include <string>

using namespace std;

static float hash01(int a, int b, int c) {
    unsigned int x = static_cast<unsigned int>(a * 374761393 + b * 668265263 + c * 1274126177);
    x = (x ^ (x >> 13)) * 1274126177u;
    x ^= x >> 16;
    return (x & 0xFFFFFFu) / float(0x1000000u);
}

static RenderMode parseMode(const string &modeStr) {
    if (modeStr == "path_guiding" || modeStr == "pathguiding" || modeStr == "guiding") {
        return RenderMode::PATH_TRACE_GUIDING;
    }
    if (modeStr == "path_mis" || modeStr == "pathmis") {
        return RenderMode::PATH_TRACE_MIS;
    }
    if (modeStr == "path_nee" || modeStr == "pathnee") {
        return RenderMode::PATH_TRACE_NEE;
    }
    if (modeStr == "path" || modeStr == "pathtrace" || modeStr == "path_trace") {
        return RenderMode::PATH_TRACE;
    }
    return RenderMode::WHITTED;
}

static bool parseFlag(int argc, char *argv[], const char *name, const char *longName) {
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], name) == 0 || (longName && strcmp(argv[i], longName) == 0)) {
            return true;
        }
    }
    return false;
}

static bool parseGammaFlag(int argc, char *argv[]) {
    return parseFlag(argc, argv, "gamma", "--gamma");
}

static bool parseOmpFlag(int argc, char *argv[]) {
    return parseFlag(argc, argv, "omp", "--omp") ||
           parseFlag(argc, argv, "parallel", "--parallel");
}

static bool parseDispersionFlag(int argc, char *argv[]) {
    return parseFlag(argc, argv, "dispersion", "--dispersion");
}

static bool parseCudaFlag(int argc, char *argv[]) {
    return parseFlag(argc, argv, "cuda", "--cuda") ||
           parseFlag(argc, argv, "gpu", "--gpu");
}

static bool parseNoBvhFlag(int argc, char *argv[]) {
    return parseFlag(argc, argv, "no_bvh", "--no-bvh");
}

static int parseTrainSppFlag(int argc, char *argv[]) {
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "train_spp") == 0 || strcmp(argv[i], "--train-spp") == 0) {
            if (i + 1 < argc) {
                int v = atoi(argv[i + 1]);
                return v > 0 ? v : 0;
            }
        }
    }
    return 0;
}

static bool isOptionToken(const char *arg) {
    return strcmp(arg, "gamma") == 0 || strcmp(arg, "--gamma") == 0 ||
           strcmp(arg, "omp") == 0 || strcmp(arg, "--omp") == 0 ||
           strcmp(arg, "parallel") == 0 || strcmp(arg, "--parallel") == 0 ||
           strcmp(arg, "dispersion") == 0 || strcmp(arg, "--dispersion") == 0 ||
           strcmp(arg, "cuda") == 0 || strcmp(arg, "--cuda") == 0 ||
           strcmp(arg, "gpu") == 0 || strcmp(arg, "--gpu") == 0 ||
           strcmp(arg, "train_spp") == 0 || strcmp(arg, "--train-spp") == 0 ||
           strcmp(arg, "no_bvh") == 0 || strcmp(arg, "--no-bvh") == 0;
}

static const char *modeName(RenderMode mode) {
    if (mode == RenderMode::PATH_TRACE_GUIDING) {
        return "path_guiding";
    }
    if (mode == RenderMode::PATH_TRACE_MIS) {
        return "path_mis";
    }
    if (mode == RenderMode::PATH_TRACE_NEE) {
        return "path_nee";
    }
    if (mode == RenderMode::PATH_TRACE) {
        return "path";
    }
    return "whitted";
}

int main(int argc, char *argv[]) {
    for (int argNum = 1; argNum < argc; ++argNum) {
        std::cout << "Argument " << argNum << " is: " << argv[argNum] << std::endl;
    }

    if (argc < 3) {
        cout << "Usage: ./PA1-2 <scene file> <output bmp> [whitted|path|path_nee|path_mis|path_guiding] [spp] [gamma] [omp|parallel] [dispersion] [cuda|gpu] [no_bvh] [train_spp N]" << endl;
        return 1;
    }

    string inputFile = argv[1];
    string outputFile = argv[2];
    RenderMode mode = RenderMode::WHITTED;
    int spp = 1;

    if (argc >= 4) {
        mode = parseMode(argv[3]);
    }
    bool applyGamma = parseGammaFlag(argc, argv);
    bool useOmp = parseOmpFlag(argc, argv);
    bool useDispersion = parseDispersionFlag(argc, argv);
    bool useCuda = parseCudaFlag(argc, argv);
    bool noBvh = parseNoBvhFlag(argc, argv);
    int trainSpp = parseTrainSppFlag(argc, argv);
    if (argc >= 5 && !isOptionToken(argv[4])) {
        spp = atoi(argv[4]);
        if (spp < 1) {
            spp = 1;
        }
    } else if (mode == RenderMode::PATH_TRACE || mode == RenderMode::PATH_TRACE_NEE ||
               mode == RenderMode::PATH_TRACE_MIS || mode == RenderMode::PATH_TRACE_GUIDING) {
        spp = 64;
    }

    SceneParser scene(inputFile.c_str());
    Camera *camera = scene.getCamera();
    Image dImg(camera->getWidth(), camera->getHeight());

    cout << "Render mode: " << modeName(mode) << ", SPP: " << spp
         << ", gamma: " << (applyGamma ? "on" : "off")
         << ", omp: " << (useOmp ? "on" : "off")
         << ", dispersion: " << (useDispersion ? "on" : "off")
         << ", cuda: " << (useCuda ? "on" : "off")
         << ", bvh: " << (noBvh ? "off" : "on");
    if (mode == RenderMode::PATH_TRACE_GUIDING && trainSpp > 0) {
        cout << ", train_spp: " << trainSpp;
    }
#ifdef _OPENMP
    if (useOmp) {
        cout << " (" << omp_get_max_threads() << " threads)";
    }
#else
    if (useOmp) {
        cout << " (OpenMP not available at build time)";
        useOmp = false;
    }
#endif
    cout << endl;

    const int height = camera->getHeight();
    const int width = camera->getWidth();

#ifdef USE_CUDA
    if (useCuda) {
        if (mode == RenderMode::PATH_TRACE_GUIDING) {
            cout << "Path guiding requires CUDA; training+render on GPU." << endl;
        }
        double cudaSec = 0.0;
        if (renderWithCuda(scene, dImg, mode, spp, useDispersion, cudaSec, trainSpp, !noBvh)) {
            cout << "Render time: " << cudaSec << " s (CUDA)" << endl;
            dImg.SaveBMP(outputFile.c_str(), applyGamma);
            cout << "Hello! Computer Graphics!" << endl;
            return 0;
        }
        cout << "CUDA rendering unavailable or failed; falling back to CPU." << endl;
        if (mode == RenderMode::PATH_TRACE_GUIDING) {
            cout << "path_guiding is GPU-only; use cuda flag." << endl;
            return 1;
        }
        useCuda = false;
    }
#endif

    if (mode == RenderMode::PATH_TRACE_GUIDING) {
        cout << "path_guiding requires CUDA. Rebuild with USE_CUDA and pass cuda flag." << endl;
        return 1;
    }

    const bool showProgress = mode == RenderMode::PATH_TRACE || mode == RenderMode::PATH_TRACE_NEE ||
                              mode == RenderMode::PATH_TRACE_MIS;

    auto t0 = chrono::high_resolution_clock::now();

    // Parallelize scanlines only: each (x,y) owns a RayTracer + seed — no shared mutable state.
    // dynamic,4 hands out 4-row chunks so sky-heavy rows don't starve geometry-heavy rows.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4) if (useOmp)
#endif
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Vector3f accum = Vector3f::ZERO;
            for (int s = 0; s < spp; ++s) {
                float jx = float(x);
                float jy = float(y);
                if (spp > 1) {
                    jx += hash01(x, y, s);
                    jy += hash01(y, x, s + 31);
                }
                unsigned int seed = 1u + static_cast<unsigned int>(x) +
                                    7919u * static_cast<unsigned int>(y) +
                                    104729u * static_cast<unsigned int>(s);
                RayTracer tracer(scene, mode, seed, useDispersion);
                Ray canRay = camera->generateRay(Vector2f(jx, jy));
                accum += tracer.trace(canRay);
            }
            dImg.SetPixel(x, y, accum * (1.0f / spp));
        }
        if (showProgress && !useOmp && (y + 1) % 64 == 0) {
            cout << "Scanline " << (y + 1) << "/" << height << endl;
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    double renderSec = chrono::duration<double>(t1 - t0).count();
    cout << "Render time: " << renderSec << " s" << endl;

    dImg.SaveBMP(outputFile.c_str(), applyGamma);
    cout << "Hello! Computer Graphics!" << endl;
    return 0;
}

