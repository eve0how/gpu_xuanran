// 文件说明：程序入口，解析命令行参数并调度 CPU 或 CUDA 路径追踪渲染。
// 原创性声明：已有代码（PA1 课程框架的 main 骨架）基础之上，命令行解析、
// OpenMP 扫描线循环与 CUDA 回退逻辑为独立实现。

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

static RenderMode resolveRenderMode(const string &modeStr) {
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
        std::cout << "[argv " << argNum << "] " << argv[argNum] << std::endl;
    }

    if (argc < 3) {
        cout << "用法: ./PA1-2 <场景文件> <输出bmp> [whitted|path|path_nee|path_mis|path_guiding] "
                "[spp] [gamma] [omp|parallel] [dispersion] [cuda|gpu] [no_bvh] [train_spp N]" << endl;
        return 1;
    }

    string inputFile = argv[1];
    string outputFile = argv[2];
    RenderMode mode = RenderMode::WHITTED;
    int spp = 1;

    if (argc >= 4) {
        mode = resolveRenderMode(argv[3]);
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
    Image outputImage(camera->getWidth(), camera->getHeight());

    cout << "渲染配置 — 模式=" << modeName(mode) << " 采样=" << spp
         << " gamma=" << (applyGamma ? "开" : "关")
         << " omp=" << (useOmp ? "开" : "关")
         << " 色散=" << (useDispersion ? "开" : "关")
         << " cuda=" << (useCuda ? "开" : "关")
         << " bvh=" << (noBvh ? "关" : "开");
    if (mode == RenderMode::PATH_TRACE_GUIDING && trainSpp > 0) {
        cout << " 训练采样=" << trainSpp;
    }
#ifdef _OPENMP
    if (useOmp) {
        cout << " (线程数 " << omp_get_max_threads() << ")";
    }
#else
    if (useOmp) {
        cout << " (编译时未启用 OpenMP，已忽略 omp 选项)";
        useOmp = false;
    }
#endif
    cout << endl;

    const int height = camera->getHeight();
    const int width = camera->getWidth();

#ifdef USE_CUDA
    if (useCuda) {
        if (mode == RenderMode::PATH_TRACE_GUIDING) {
            cout << "路径引导模式需 CUDA，将在 GPU 上完成训练与渲染。" << endl;
        }
        double cudaSec = 0.0;
        if (renderWithCuda(scene, outputImage, mode, spp, useDispersion, cudaSec, trainSpp, !noBvh)) {
            cout << "耗时 " << cudaSec << " 秒 (GPU/CUDA)" << endl;
            outputImage.SaveBMP(outputFile.c_str(), applyGamma);
            cout << "渲染完成，输出已写入。" << endl;
            return 0;
        }
        cout << "CUDA 不可用或失败，改用 CPU 渲染。" << endl;
        if (mode == RenderMode::PATH_TRACE_GUIDING) {
            cout << "path_guiding 仅支持 GPU，请加上 cuda 参数。" << endl;
            return 1;
        }
        useCuda = false;
    }
#endif

    if (mode == RenderMode::PATH_TRACE_GUIDING) {
        cout << "path_guiding 需要 CUDA：请启用 USE_CUDA 编译并传入 cuda 参数。" << endl;
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
                Ray tracingBeam = camera->generateRay(Vector2f(jx, jy));
                accum += tracer.trace(tracingBeam);
            }
            outputImage.SetPixel(x, y, accum * (1.0f / spp));
        }
        if (showProgress && !useOmp && (y + 1) % 64 == 0) {
            cout << "扫描行进度 " << (y + 1) << " / " << height << endl;
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    double renderSec = chrono::duration<double>(t1 - t0).count();
    cout << "耗时 " << renderSec << " 秒 (CPU)" << endl;

    outputImage.SaveBMP(outputFile.c_str(), applyGamma);
    cout << "渲染完成，输出已写入。" << endl;
    return 0;
}

