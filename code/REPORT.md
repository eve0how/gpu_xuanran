# 计算机图形学 PA1 实验报告

## 1. 实验概述

本实验在清华大学 PA1 光线追踪框架上，依次实现了 **Whitted-Style 光线追踪**、**路径追踪（Path Tracing）**、**Cook-Torrance 光泽材质** 以及 **Next Event Estimation（NEE）**。核心场景为 Cornell Box 风格的 `scene08_whitted.txt`（点光源 + 镜面/折射物体）与 `scene08_path.txt`（面光源 + 发光天花板，用于路径追踪对比）。

渲染器支持命令行模式切换：`whitted`、`path`（无 NEE）、`path_nee`（含 NEE），并可指定每像素采样数 SPP。

---

## 2. 实现功能

| 功能项 | 完成度 | 说明 |
|--------|--------|------|
| **基础要求 1：Whitted-Style** | ✅ 完成 | 完美镜面反射、Snell 折射、阴影射线、Phong 漫反射 |
| **基础要求 2：路径追踪** | ✅ 完成 | 余弦加权半球采样、俄罗斯轮盘赌、发光材质、面光源场景 |
| **4.2 Cook-Torrance Glossy** | ✅ 完成 | Beckmann D + Cook-Torrance G + Schlick F，路径/Whitted 均支持 |
| **4.3 NEE** | ✅ 完成 | 点光源 + 三角形面光源直接采样，阴影可见性测试 |
| **4.1 对比实验** | ✅ 完成 | `path` vs `path_nee` 同场景同 SPP 对比 |
| **5.5 Gamma 校正（bonus）** | ✅ 完成 | BMP 保存前可选 `color^(1/2.2)` 编码，CLI `gamma` 开关 |
| **5.6 OpenMP 并行（bonus）** | ✅ 完成 | 按扫描行 `#pragma omp parallel for`，CLI `omp` 开关，输出耗时 |
| **5.8 纹理与法线贴图（bonus）** | ✅ 完成 | BMP albedo + normalMap；Cornell 三面板展示（`scene_texture_cornell*.txt`） |
| **Path Guiding（bonus）** | ✅ 完成 | CUDA 两趟简化 Practical Path Guiding + NEE；CLI `path_guiding` / `train_spp` |

---

## 3. 原理简述

### 3.1 Whitted-Style 光线追踪

对镜面/折射材质递归追踪反射/折射方向；对漫反射材质用 Phong 模型计算直接光照，并向光源发射阴影射线判断可见性。玻璃材质在阴影射线中视为透明（不遮挡）。

### 3.2 路径追踪

求解渲染方程：在漫反射/光泽命中点按 BRDF 采样出射方向，递归估计入射辐射度。采用 **余弦加权半球采样**（Lambertian），pdf 与 cosθ 相消后贡献为 `albedo × Li`。命中 **EmissiveMaterial** 时返回 `throughput × emission`。

**俄罗斯轮盘赌（RR）**：深度 ≥ 8 时，以 `max(0.15, luminance(throughput))` 为存活概率终止路径，存活时除以概率保持无偏。

### 3.3 Cook-Torrance 光泽 BRDF（课程 PPT 第 53–56 页）

$$
f = k_d \frac{\rho_d}{\pi} + k_s \frac{D \cdot G \cdot F}{4 (n \cdot \omega_i)(n \cdot \omega_o)}
$$

- **D**：Beckmann 微表面法向分布（粗糙度 $m$）
- **G**：Cook-Torrance 几何项 $G_1(\omega_o) \cdot G_1(\omega_i)$，$G_1 = \min(1, 2(n\cdot h)(n\cdot v)/(v\cdot h))$
- **F**：Schlick 菲涅尔；电介质默认 $F_0=0.04$，金属 $F_0$ 取 albedo
- **路径采样**：按 $k_d$/$k_s$ 能量比选择漫反射瓣或 Beckmann 镜面瓣，各瓣用匹配 BRDF 项估计；NEE 点光加 $1/r^2$ 衰减；间接辐射度软钳制抑制 firefly

### 3.4 NEE（Next Event Estimation）

在 `path_nee` 模式下，漫反射/光泽命中点**额外**向光源采样直接光：

- **点光源**：方向由 `PointLight::getIllumination` 给出，阴影射线判断遮挡，贡献 `BRDF × Le × cosθ`。
- **面光源**：对每个 `AreaLight` 在三角形上均匀采样（pdf_area = 1/A），立体角 pdf：pdf_ω = pdf_area × r² / cosθ_l，贡献  
  `Le × (albedo/π) × cosθ_o / pdf_ω = albedo × Le × cosθ_o × cosθ_l × A / (π r²)`。
- **避免双重计数**：NEE 开启时，由漫反射弹射出去的间接光线不再对 `EmissiveMaterial` 累加辐射度（直接光仅由 NEE 估计）；镜面/折射路径仍可命中发光体。
- **阴影射线**：使用法向偏移，且不再将发光体视为“透明”（仅当 t ≥ 光距离时视为命中光源）。

`path` 模式不启用 NEE，仅靠随机弹射间接命中发光体，方差大、收敛慢。

### 3.5 路径引导（Path Guiding）

**动机**：NEE 已稳定估计直接光，但 **间接光** 仍靠余弦半球随机弹射。在遮挡阴影、窄缝漏光、多次反弹才能照亮的区域，随机方向很难命中「亮墙 / 光源方向」，方差大、同 SPP 下颗粒粗。

**思路**（相对 Müller 等完整 SD-tree 的 **课程友好简化版**）：在场景 AABB 上建 **3D 均匀网格**（`8³`），每格维护 **lat-long 方向直方图**（`16×16` θ–φ bin）。训练趟从相机路径与 **light tracing**（从面光源反向追踪）沉积「哪些方向带来亮辐射」；渲染趟在间接采样时 **50% 余弦 BRDF / 50% 引导分布**，用 **Balance MIS** 合并 pdf，保持无偏、只降方差。

**与 `path_nee` 的分**：直接光仍走 NEE；根路径与间接子路径 `countEmissive=false`，避免与 NEE 双重计光。期望亮度与 `path_nee` 接近（验收全图比 ≈ 1.0–1.05），差异主要在 **间接难采样区** 的噪声。

### 3.6 纹理与法线贴图

**动机**：在漫反射 albedo 上叠加 **空间变化**（灰泥墙、大理石球），并用 **法线贴图** 在 Whitted 点光下产生 subtle bump，无需增加几何细分。

**思路**：材质解析时加载 24-bit BMP；命中点插值 UV，着色阶段 `albedo = diffuseColor × texture(uv)`。法线贴图在 **切线空间** 扰动法线：`N' = normalize(TBN · mapNormal(uv))`，再参与 Phong / 路径 BRDF。纹理管线 **仅在 CPU**（Whitted / `path` / `path_nee`）；CUDA 扁平化材质不传贴图。

---

## 4. 代码逻辑

| 文件 | 职责 |
|------|------|
| `include/raytracer.hpp` | Whitted / Path / Path+NEE 三种模式；RR、NEE、光泽采样 |
| `include/material.hpp` | Material、Reflect、Refract、Emissive、GlossyMaterial、CookTorranceBRDF；可选纹理 |
| `include/texture.hpp` | BMP 纹理加载与 UV 采样（repeat）；法线贴图 `sampleNormal` |
| `src/texture.cpp` | `loadBMP`、程序化 `gen_textures`（plaster/marble 等） |
| `include/cuda_types.h` | GPU 结构体；`GpuGuidingGrid`（`8³` 网格 + `16×16` 方向 bin） |
| `src/cuda_path_tracer.cu` | CUDA 路径追踪；`trainGuideKernel` / `renderKernel` / 引导 MIS |
| `include/light.hpp` | PointLight、DirectionalLight、AreaLight |
| `src/scene_parser.cpp` | 解析材质别名、AreaLight、GlossyMaterial |
| `src/main.cpp` | CLI：`[whitted\|path\|path_nee] [spp] [gamma] [omp]`，默认 path 模式 SPP=64，默认串行渲染 |
| `testcases/scene08_whitted.txt` | Whitted 演示：点光源、无发光天花板 |
| `testcases/scene08_path.txt` | 路径追踪：AreaLight + EmissiveMaterial 天花板 |
| `testcases/scene09_glossy.txt` | Cook-Torrance 光泽 Cornell Box 五球演示 |

### Cornell Box 材质索引（scene08 / scene_whitted / scene_path）

| 索引 | 用途 | diffuseColor / 类型 |
|------|------|---------------------|
| 0 | 地板 | 0.725 0.725 0.725（米白） |
| 1 | 天花板（非发光面） | 0.10 0.10 0.10 |
| 2 | 左墙（红） | 0.630 0.065 0.050 |
| 3 | 蓝色球 | 0.180 0.280 0.800 |
| 4 | 红色球 | 0.800 0.150 0.150 |
| 5 | 右墙（绿）+ 绿色球 | 0.150 0.680 0.200 |
| 6 | 镜面球 | ReflectiveMaterial |
| 7 | 玻璃立方体 | RefractiveMaterial, IOR 1.45 |
| 8 | 后墙（Whitted）/ 发光天花板（Path） | Whitted: 0.65 0.65 0.60；Path: EmissiveMaterial |
| 9 | 后墙（Path） | 0.65 0.65 0.60 |

地板与后墙已拆分为不同材质，避免交界处颜色完全一致。

---

## 5. 场景与输出映射

### 5.1 规范输出（`results/` 提交用）

| 输出图片 | 场景文件 | 渲染命令（在 `code/` 下） | SPP | 作业对应项 |
|----------|----------|---------------------------|-----|------------|
| `results/whitted.bmp` | `testcases/scene_whitted.txt` | `build/PA1-2 testcases/scene_whitted.txt output/whitted.bmp whitted` | 1 | 基础要求 1（反射/折射/阴影） |
| `results/path_no_nee.bmp` | `testcases/scene_path.txt` | `build/PA1-2 testcases/scene_path.txt output/path_no_nee.bmp path 64` | 64 | 基础要求 2 + §4.1 对比（无 NEE） |
| `results/path_nee.bmp` | `testcases/scene_path.txt` | `build/PA1-2 testcases/scene_path.txt output/path_nee.bmp path_nee 64` | 64 | 基础要求 2 + §4.3 NEE 对比 |
| `results/glossy.bmp` | `testcases/scene_glossy.txt` | `build/PA1-2 testcases/scene_glossy.txt output/glossy.bmp path_nee 64` | 64 | §4.2 Cook-Torrance glossy |

`scene_whitted.txt` 与 `scene08_whitted.txt`、`scene_path.txt` 与 `scene08_path.txt` 几何与材质布局一致；对比实验以 `scene_whitted` / `scene_path` 为规范场景。渲染产物先写入 `code/output/`，再复制到 `results/`。

### 5.2 别名与历史输出（可选保留）

| 输出图片 | 等价场景 | 说明 |
|----------|----------|------|
| `results/scene08_whitted.bmp` | `scene08_whitted.txt` | 与 `whitted.bmp` 同场景 |
| `code/output/scene08_path.bmp` | `scene08_path.txt` | 与 `path_no_nee.bmp` 同场景 |
| `code/output/scene08_path_nee.bmp` | `scene08_path.txt` | 与 `path_nee.bmp` 同场景 |
| `code/output/scene09_glossy.bmp` | `scene09_glossy.txt` | 与 `glossy.bmp` 同场景 |
| `results/path_nee_keep.bmp` | `scene_path.txt` | 早期 NEE 参考备份，报告以 `path_nee.bmp` 为准 |

### 5.3 光泽场景材质参数（`scene_glossy.txt`）

| 球体 | 类型 | kd | ks | 粗糙度 m | F₀ |
|------|------|----|----|----------|-----|
| 红 | 塑料 | 0.80 0.15 0.15 | 0.5 | **0.22** | 0.04 |
| 蓝 | 塑料 | 0.18 0.28 0.80 | 0.5 | **0.28** | 0.04 |
| 绿 | 塑料 | 0.15 0.68 0.20 | 0.5 | **0.18** | 0.04 |
| 金 | 金属 | 0 0 0 | 0.9 0.7 0.3 | **0.35**（初版 0.08） | 0.9 0.7 0.3 |
| 银 | 金属 | 0 0 0 | 0.85 0.85 0.88 | **0.28**（初版 0.06） | 0.85 0.85 0.88 |

### 5.5 Gamma 校正（bonus）

显示器按 sRGB 非线性显示像素；渲染器内部在 **线性光强** 下积分，直接写入 BMP 时中间调偏暗。保存前对每通道施加 **gamma 编码**：

$$
C_{\text{out}} = \mathrm{clamp}\bigl(255 \cdot C_{\text{linear}}^{1/2.2}\bigr)
$$

实现于 `image.cpp` 的 `SaveBMP`；默认 **不** 做 gamma（保持线性，与修改前行为一致）。命令行追加 `gamma` 或 `--gamma` 开启。

| 图片 | 命令 | 说明 |
|------|------|------|
| `gamma_before.bmp` | `whitted`（无 gamma） | 线性输出，暗部压缩、对比偏硬 |
| `gamma_after.bmp` | `whitted gamma` | gamma 编码后，中间调更亮、观感更自然 |

对比场景：`scene_whitted.txt`（Whitted SPP=1，即时渲染）。gamma 仅影响 **写文件**，不改变光线追踪计算本身。

### 5.6 OpenMP CPU 并行加速（bonus）

路径追踪每像素独立、按行写入 `Image`，适合 **扫描行级** OpenMP 并行。在 `main.cpp` 外层 `y` 循环加 `#pragma omp parallel for schedule(dynamic, 4) if (useOmp)`；每像素构造独立 `RayTracer`（seed = f(x,y,s)），并行与串行 **像素结果一致**。CMake 通过 `FIND_PACKAGE(OpenMP)` + `OpenMP::OpenMP_CXX` 链接；macOS 需 Homebrew `libomp`。CLI：`omp` / `parallel`（默认串行，便于对比与调试）。

**曾出现的性能问题**：初版在并行 `y` 循环内对 **每一行** 使用 `#pragma omp critical` 包裹进度 `cout`，导致所有线程每完成一行都要抢全局锁，并行几乎退化为串行（1024×1024 `path_nee 64` 约 785–800 s）。修复后：并行模式 **不打印** 扫描行进度，仅在结束时输出 `Render time`；串行模式保留每 64 行进度。

**计时对比**（`scene_path.txt`，`path_nee` SPP=32，1024×1024，10 线程，2025-06-25）：

| 模式 | Render time | 相对加速 |
|------|-------------|----------|
| 串行 | 799.9 s | 1.0× |
| 并行 (`omp`) | 104.9 s | **7.6×** |

Whitted 快速自检（`scene_whitted.txt`，SPP=1）：串行 2.24 s → 并行 0.35 s（6.4×）。

| 图片 | 命令 | 说明 |
|------|------|------|
| `omp_serial.bmp` | `path_nee 32`（无 omp） | 串行基准 |
| `omp_parallel.bmp` | `path_nee 32 omp` | 多线程；与串行 **逐字节一致** |

### 5.6.1 抗锯齿（AA，已有实现，待文档化）

`main.cpp` 在 **SPP > 1** 时已对子样本做 **抖动**：`jx += hash01(x,y,s)`，`jy += hash01(y,x,s+31)`，即每像素内随机偏移采样点，等价于盒式滤波抗锯齿。Whitted 默认 SPP=1 无 AA；路径模式默认 SPP=64 已含抖动。无需额外 CLI，提高 SPP 即可加强 AA。

### 5.8 纹理与法线贴图（bonus）

在漫反射 / 光泽材质上可选加载 **24-bit BMP 纹理** 与 **法线贴图**，着色时用 `diffuseColor × texture(uv)` 调制 albedo；有法线贴图时用 TBN 变换扰动着色法线（未指定时行为与原来完全一致）。

#### 5.8.1 实现原理

| 环节 | 说明 | 代码位置 |
|------|------|----------|
| **加载** | `Texture::loadBMP` 读 24-bit BMP，像素存 `[0,1]³` | `src/texture.cpp`、`include/texture.hpp` |
| **Albedo 采样** | `Material::getShadedDiffuse(hit)`：`diffuseColor × texture->sample(u,v)`，UV repeat（`frac`） | `include/material.hpp` |
| **法线贴图** | `sampleNormal` 将 RGB 映射到 `[-1,1]³`；`getShadingNormal` 用 `hit` 的 TBN 变到世界空间 | `include/material.hpp` |
| **场景语法** | `texture textures/xxx.bmp`；`normalMap textures/yyy.bmp`（可单独或组合） | `src/scene_parser.cpp` |
| **Whitted 着色** | `shadeDiffuse` / `shadeGlossyWhitted` 走 `getShadingNormal` + `getShadedDiffuse` | `include/raytracer.hpp` |
| **路径着色** | `shadeDiffusePath` / `shadeGlossyPath` 同样调用上述接口 | 同上 |

**UV 映射**：

| 几何体 | UV 计算 |
|--------|---------|
| **Plane** | 交点在平面切线空间坐标，×2 平铺后 `frac` 重复；同时写入 TBN |
| **Sphere** | 球面坐标：`u = 0.5 + atan2(z,x)/(2π)`，`v = 0.5 - asin(y/r)/π` |
| **TriangleMesh** | OBJ 的 `vt`/`vn`，命中时用重心坐标插值 UV 与平滑法线；`Transform` 只更新法线/TBN、**保留 UV** |

**CPU-only 限制**：`SceneFlattener`（`cuda_scene_builder.cpp`）只上传材质常数 `diffuse[]`，不传 BMP。纹理验收请用 **Whitted 或 CPU 路径**（**勿加** `cuda` / `gpu`）。

#### 5.8.2 程序化纹理（`gen_textures`）

工具 `build/gen_textures` 调用 `Texture::generateShowcaseTextures`，生成 Cornell 展示所需贴图：

| 文件 | 生成函数 | 用途 |
|------|----------|------|
| `textures/plaster_albedo.bmp` | `writePlasterAlbedoBMP` | 灰泥墙 albedo（柔和米白、低频噪声） |
| `textures/plaster_normal.bmp` | `writePlasterNormalBMP` | 灰泥 bump 法线（由高度场求导，**Panel C 加强 bump**） |
| `textures/marble_albedo.bmp` | `writeMarbleAlbedoBMP` | 中心球大理石（蓝灰脉纹） |

另有砖/木/石/地球等 legacy demo 贴图。`gen_textures` 目标使用 `-O2` 编译，避免 Release `-O3` 下 `unsigned char` 负值转换 UB。

#### 5.8.3 Cornell 三面板展示

基于经典 Cornell 盒（`scene_whitted.txt` 布局）：红/绿侧墙、灰地板/顶；**后墙** 用 `mesh/cornell_back_wall.obj`（带 UV）；**中心球** 材质 4 贴大理石 albedo（球面 UV）；其余球与玻璃立方体不变。

| 面板 | 场景 | 后墙材质 | 说明 |
|------|------|----------|------|
| **A** | `scene_texture_cornell_notex.txt` | 纯色 diffuse | 对照：经典 Cornell |
| **B** | `scene_texture_cornell.txt` | `plaster_albedo.bmp`，**无法线贴图** | 仅 albedo 纹理 |
| **C** | `scene_texture_cornell_normal.txt` | 同 albedo + `plaster_normal.bmp` | 略降 diffuse、加 specular/shininess，使 Whitted 点光下 **bump 可见** |

**B vs C 差异**：Panel B 只有颜色变化，墙仍近似平面 Phong；Panel C 法线扰动改变 **每像素法线方向**，在固定点光源下高光与阴影边界随 bump 起伏——法线强度需足够（`plaster_normal.bmp` 比早期 subtle 版本更强），否则 Whitted 单样本下几乎看不出差异。

#### 5.8.4 渲染命令

```bash
cd code
cmake --build build -j$(nproc)
./build/gen_textures

# 三面板（CPU Whitted，SPP=1，建议开 gamma）
./build/PA1-2 testcases/scene_texture_cornell_notex.txt output/texture_cornell_notex.bmp whitted 1 gamma
./build/PA1-2 testcases/scene_texture_cornell.txt output/texture_cornell.bmp whitted 1 gamma
./build/PA1-2 testcases/scene_texture_cornell_normal.txt output/texture_cornell_normal.bmp whitted 1 gamma

# 可选：拼接提交用单图
python3 scripts/make_texture_showcase.py   # → output/texture_showcase.png
```

| 输出 | 说明 |
|------|------|
| `output/texture_cornell_notex.bmp` | Panel A |
| `output/texture_cornell.bmp` | Panel B |
| `output/texture_cornell_normal.bmp` | Panel C |
| `output/texture_showcase.png` | 三图横向拼接（提交用） |

早期地板棋盘格 demo 仍可用 `scene_texture.txt` + `scene_whitted.txt` 对比。

### 5.9 路径引导 Path Guiding（bonus）

**仅 CUDA**：CLI 模式 `path_guiding`（别名 `guiding` / `pathguiding`），必须带 `cuda` / `gpu`。相对完整 Practical Path Guiding（SD-tree），本实现为 **两趟 GPU 简化版**：`8³` 空间网格 + 每格 `16×16` lat-long 方向 bin（见 `include/cuda_types.h` 中 `GpuGuidingGrid`）。

#### 5.9.1 动机与适用场景

NEE 解决 **直接光** 采样；被遮挡、仅靠红/绿墙与顶板 **多次反弹** 照亮的区域，余弦半球几乎打不中「亮方向」，间接方差主导画面噪声。路径引导在训练阶段学习 **「从该空间位置，哪些 ωᵢ 常带来亮辐射」**，渲染时与 BRDF 采样 MIS 合并，**同 SPP 下阴影/暗区更干净**，全图亮度应与 `path_nee` 接近（无偏 MC）。

**推荐 demo**：`testcases/scene_guiding_occluder.txt` — 经典 Cornell + 天花板面光源 + **悬挂遮挡板**（中央地板/球无直射，间接为主）。比窄缝门 demo 几何更简单、无墙洞 bug，答辩时直观：**挡光板 → 软阴影 → 引导学反弹方向**。

#### 5.9.2 两趟流水线

```
renderWithCuda(path_guiding)
  ├─ 1. uploadScene + initCurandKernel
  ├─ 2. trainGuideKernel(train_spp)     ← 训练趟，不写像素
  ├─ 3. normalizeGuideKernel            ← 每格 bin 归一化为离散 PDF
  └─ 4. renderKernel(render_spp, useGuideGrid=true)  ← 正式渲染
```

| 内核 | 作用 |
|------|------|
| **`trainGuideKernel`** | 每像素 `train_spp` 次：70% **light tracing**（从随机 `AreaLight` 面采样一点，余弦半球出射，带 `emission×area` throughput）；30% 相机 primary ray。均调用 `castRayPath(..., trainingPass=true)`，**不写 framebuffer** |
| **`normalizeGuideKernel`** | 有数据的格子将权重和归一化为 1；**空格子保持 0**（渲染时回退纯余弦 BRDF） |
| **`renderKernel`** | 与 `path_nee` 相同 SPP 路径追踪，但间接 bounce 启用引导 MIS |

#### 5.9.3 `GpuGuidingGrid` 结构

| 字段 | 值 / 含义 |
|------|-----------|
| `bboxMin` / `bboxMax` | 场景 AABB（`cuda_scene_builder` 扁平化时计算） |
| `res` | `8` → **512** 个空间 cell |
| `thetaBins` × `phiBins` | `16×16` → 每 cell **256** 个方向 bin（半球 lat-long） |
| `weights` | 设备端 float 数组，长度 `512×256`；`atomicAdd` 训练写入 |

**空间索引**：命中点 `pos` 线性映射到 `[ix,iy,iz]`，cell = `ix + iy·res + iz·res²`。

**方向 bin**：入射方向 `ωᵢ`（相对 shading 法线 `N` 的半球）按 `θ = acos(N·ωᵢ)`、`φ = atan2` 落入 `(tBin, pBin)`；bin 立体角 `Δω` 用于 pdf = `w_bin / sum_cell / Δω`。

#### 5.9.4 训练沉积（何时、沉积什么）

训练趟 `trainingPass=true` 时，在 **漫反射 / 光泽** 命中点记录 **入射方向** `ωᵢ = -rayDir`（即将用于继续追踪的方向）：

| 时机 | 沉积内容 | 权重 |
|------|----------|------|
| 间接 bounce 返回 `Li` 后 | `guideDeposit(pos, N, ωᵢ, weight)` | `weight = luminance(Li) × max(luminance(throughput), ε)` |
| 训练趟 NEE 直接光（面光 / 点光） | 向光源方向 `ωᵢ` 沉积 | 同上，基于直接光贡献亮度 |
| Light tracing 起点 | 从光源反向路径同上规则 | 70% 训练样本走此路径，强化「从亮处 outgoing 的方向分布」 |

`guideDeposit` 内部：`atomicAdd(weight × cosθ_in)` 到对应 cell+bin（`cosθ_in = N·ωᵢ`，仅上半球）。

#### 5.9.5 渲染阶段 MIS

常量 `kGuideMisProb = 0.5`。间接采样时：

1. 若当前 cell **无训练数据** → 回退纯余弦半球（与 `path_nee` 一致）。
2. 否则 50% 采余弦 BRDF 方向，50% 按引导直方图 `sampleGuidingDir`。
3. **Balance MIS**（单样本合并两策略 pdf）：
   - `pdf_brdf = cosθ/π`（Lambert）
   - `pdf_guide = evalGuidingPdf(...)`
   - `pdf_total = 0.5 × pdf_brdf + 0.5 × pdf_guide`
   - `indirect = brdf × cosθ × Li / (pdf_total × rrProb)`

直接光仍走 NEE；主光线 `countEmissive=true`（直视发光体可见），间接子路径 `countEmissive=false`。

#### 5.9.6 CLI 与推荐对比

```bash
cd code
cmake --build build -j$(nproc)

# 推荐 demo：遮挡板 Cornell，低 SPP 易看出差异
./build/PA1-2 testcases/scene_guiding_occluder.txt output/guiding_occluder_nee_64.bmp path_nee 64 gamma cuda
./build/PA1-2 testcases/scene_guiding_occluder.txt output/guiding_occluder_guided_64.bmp path_guiding 64 gamma cuda train_spp 256

./build/PA1-2 testcases/scene_guiding_occluder.txt output/guiding_occluder_nee_128.bmp path_nee 128 gamma cuda
./build/PA1-2 testcases/scene_guiding_occluder.txt output/guiding_occluder_guided_128.bmp path_guiding 128 gamma cuda train_spp 512
```

| 参数 | 说明 |
|------|------|
| `path_guiding` | 启用训练 + 引导渲染（等同 `PATH_TRACE_GUIDING`） |
| `cuda` / `gpu` | **必须**；无 CUDA 时 main 报错 |
| `train_spp N` | 训练趟每像素样本数；默认 `max(render_spp, 256)` |
| 对比 | 同场景、同 render SPP：`path_nee` vs `path_guiding`；看 **中央阴影区** std/颗粒，全图 mean 应接近 |

**自测参考**（`scene_guiding_occluder.txt`，64 spp / train 256）：全图亮度比 guided/nee ≈ **1.06**；中央 ROI std 降约 **15%**；训练填充约 24% cell。

#### 5.9.7 局限

| 局限 | 说明 |
|------|------|
| 仅 GPU | CPU `RayTracer` 无 guiding 模式 |
| 粗网格 | `8³×16²` 直方图，空间/方向分辨率有限；空 cell 等同 `path_nee` |
| 无完整 SD-tree | 未实现在线迭代更新、无 product/importance 分解 |
| 直接光主导区 | 全画面 NEE 已很强时，guided vs nee 差异小；应裁切 **间接主导 ROI** 对比 |
| 与 `path_mis` 正交 | 当前 guiding 用 Balance MIS 合并 BRDF+guide，未与 NEE 做 Power MIS |

**关键文件**：`include/cuda_types.h`（`GpuGuidingGrid`）、`src/cuda_path_tracer.cu`（`trainGuideKernel` / `renderKernel` / `sampleIndirectWithGuide`）、`src/cuda_scene_builder.cpp`（AABB）、`src/main.cpp`（`train_spp` 解析）。

### 5.7 加分项路线图（终局展示场景，未实现）

长期目标：在 **一个高质量展示场景** 中同时体现全部 bonus（构图主观分）。建议实现顺序（由易到难、风险递增）：

1. ✅ **Gamma 校正** — 后处理，零光追风险  
2. ✅ **OpenMP 并行** — 已修复 critical 锁；`path_nee 32` 约 7.6× 加速（§5.6）  
3. ✅ **抗锯齿（AA）** — SPP>1 抖动子样本（§5.6.1）  
4. ✅ **纹理贴图** — 可选 `texture` 字段（§5.8）  
5. ✅ **法线贴图** — 与纹理共用 UV/TBN 管线（§5.8）  
6. ✅ **Path Guiding** — CUDA 两趟简化引导 + NEE（§5.9）  
7. **MIS** — 光泽 + NEE 降方差  
8. **环境贴图 IBL / 景深 / BVH** — 展示场景与性能进阶  

终局场景可规划为：Cornell Box 变体 + 光泽/金属球 + 贴地玻璃 + 环境光 + 景深 + `path_nee` 高 SPP + `gamma` + `omp` 离线渲染。当前阶段 **仅逐项实现 bonus**，暂不搭建完整展示场景。

---

## 6. Whitted vs 路径追踪对比（作业 §3.3）

**对比设置**：同一 Cornell Box 构图（相机、墙体、五球 + 玻璃立方体位置一致；立方体贴地见 §9）。Whitted 场景用点光源 `(0, 1.9, 0)`、color `(2,2,2)`；路径追踪场景将天花板改为 **EmissiveMaterial** + 两个 **AreaLight** 三角形（发光强度 80）。

| 维度 | Whitted（`whitted.bmp`） | 路径追踪（`path_nee.bmp`，SPP=64） |
|------|---------------------------|-----------------------------------|
| 着色模型 | Phong 直接光照 + Shadow Ray | 渲染方程蒙特卡洛估计 |
| 噪声 | **无**（确定性，SPP=1） | **有**颗粒噪声，SPP 越高越平滑 |
| 收敛速度 | 一次追踪即收敛 | 64 SPP 仍可见方差；无 NEE 时更慢（见 §7） |
| 光源阴影 | 点光源 **硬阴影** | 面光源 + NEE → 地板 **软阴影**、半影过渡 |
| 焦散（caustics） | 玻璃下方清晰亮斑（确定性折射） | 同样可见光斑，但边缘更 **模糊、带噪** |
| 全局光照 | 无颜色渗透 | 红/绿墙 **颜色渗透**明显 |
| 镜面球 | 清晰映出点光源高光 | delta BRDF 反射清晰，漫反射区有噪声 |
| 玻璃立方体 | 折射 + 全反射，无菲涅尔（符合作业基础要求） | 同上；贴地后地板接触阴影与焦散位置与 Whitted 一致 |

**图像统计（1024×1024，2025-06-25 重渲后）**：

| 图片 | 平均亮度 | 标准差 | 非黑像素(>5) | 高亮(>240) |
|------|----------|--------|--------------|------------|
| `whitted.bmp` | 82.7 | 92.3 | 62.7% | 16.3% |
| `path_nee.bmp` | 54.0 | 59.0 | 89.4% | 1.8% |

Whitted 整体更亮，来自点光源集中能量；路径追踪能量分散于发光面，且 64 SPP 下方差使高光未饱和。两者光源类型不同，**不追求像素级亮度一致**，但构图与几何应一致。

**差异原因（作业要求）**：

1. **噪声**：路径追踪对半球积分做蒙特卡洛估计，有限 SPP 产生方差；Whitted 无积分估计，故无噪点。
2. **阴影**：点光源 → 硬阴影；面光源 → 本影/半影，NEE 直接采样光源表面可稳定得到软阴影。
3. **焦散**：两者均经玻璃折射路径产生；Whitted 确定性求和，路径追踪随机平均，后者更噪更软。
4. **着色原理**：Whitted 在漫反射面终止；路径追踪继续弹射，间接光与颜色渗透自然出现。

---

## 7. NEE 前后对比（作业 §4.1 / §4.3）

**对比设置**：同一场景 `scene_path.txt`、同一 SPP=64，仅切换 `path` 与 `path_nee`。两图均于 **2025-06-25** 在立方体贴地（Y=0.36）及 NEE 修复后重渲。

| 指标 | `path_no_nee.bmp` | `path_nee.bmp` |
|------|-------------------|----------------|
| 平均亮度 | 23.2 | 54.0 |
| 标准差 | 42.9 | 59.0 |
| 非黑像素 | 43.6% | 89.4% |
| 饱和像素(任通道=255) | 2.0% | 5.3% |
| 观感 | 大面积偏暗、颗粒噪重 | 整体照亮、颜色稳定、软阴影清晰 |

1. **相同期望**：两种模式求解同一渲染方程；NEE 仅改变采样策略。平均亮度差异来自 **方差**：无 NEE 时大量像素未命中小面积发光体，贡献近 0；NEE 直接对光源采样后暗部被照亮，均值上升且更接近真值（作业文档：两图期望一致、无系统色差）。

2. **收敛速度**：无 NEE 时直接光仅当随机弹射 **恰好命中** 发光三角形或 Emissive 天花板时出现；发光面积占比极小，命中概率低、方差极大。NEE 在漫反射/光泽命中点对每个 AreaLight **显式采样** + Shadow Ray，将低概率间接命中转为 $O(1)$ 直接估计，收敛速度显著提升（参见作业文档图 2）。

3. **地板阴影**：`path_nee.bmp` 中球体与立方体接触阴影边缘柔和；`path_no_nee.bmp` 阴影区噪声大、边界斑驳。

4. **实现要点**：对每个 AreaLight 分别累加贡献；NEE 开启时间接路径不对 EmissiveMaterial 重复计光；阴影射线用法向偏移；玻璃立方体仍遮挡 NEE。

---

## 8. Cook-Torrance 光泽材质（作业 §4.2）

**输出**：`results/glossy.bmp`（`scene_glossy.txt`，`path_nee 64`）。

公式见 §3.3（课程 PPT 第 53–56 页，Beckmann D + Cook-Torrance G + Schlick F）。

### 8.1 塑料 vs 金属

| | 塑料（红/蓝/绿） | 金属（金/银） |
|--|------------------|---------------|
| $k_d$ | 有（物体本色） | **0** |
| $F_0$ | 0.04 | 与 albedo 相同 |
| 粗糙度 m | 0.18–0.28 | **0.35 / 0.28** |
| 外观 | 底色 + 较小高光 | **混浊金属**：宽软高光、环境映像模糊 |

### 8.2 由过曝到合理参数

初版金属 m=0.08/0.06 时 Beckmann 峰过窄，镜面采样易打出极高增益，NEE 下 **严重 firefly 过曝**。将金/银 m 提至 **0.35/0.28** 后 D 分布展宽，呈磨砂金属感；`clampRadiance`（亮度上限 30）抑制残余 firefly。

---

## 9. 几何说明：玻璃立方体贴地

`mesh/cube.obj` 顶点范围为 $[-1,1]^3$。场景中使用 `Translate -0.55 0.36 0.62` + `UniformScale 0.36`：边长 $0.72$，中心 $y=0.36$，底面 $y=0$，**恰好贴地**。此前中心偏低时立方体悬浮；修正后 Whitted 与路径追踪中地板接触阴影、焦散位置一致。

---

## 10. 已知局限

| 局限 | 说明 |
|------|------|
| 玻璃 firefly | 完美折射无菲涅尔（基础要求），64 SPP 时立方体边缘偶见亮斑；radiance clamp 缓解但未完全消除 |
| SPP=64 方差 | 路径对比图仍有可见噪点；提高 SPP 会更干净 |
| 无 MIS | 光泽瓣与 NEE 未做多重重要性采样（`path_mis` 已实现，见 IMPLEMENTATION.md） |
| 纹理 / 法线仅 CPU | GPU 扁平化不传 BMP；纹理与法线贴图验收用 Whitted / CPU path（§5.8） |
| Path Guiding 仅 CUDA | 粗网格直方图；空 cell 与直接光主导区与 `path_nee` 差异小（§5.9） |
| 默认线性输出 | 主结果图为线性 radiance；gamma 对比见 §5.5 |
| Whitted 无菲涅尔折射 | 符合作业基础要求文档 |

---

## 11. 参考代码与资料

- 清华大学 PA1 光线追踪框架（`code/`）
- 课程讲义：BRDF 与 Cook-Torrance（`1998990027_KJ_*.pdf`）
- 习题课 2：路径追踪、RR、NEE（`2023310795_KJ_*.pdf`）
- GAMES101 Lecture 16（NEE 面积采样公式）

---

## 附录：推荐运行命令

```bash
cd code
cmake --build build

# Whitted 演示
build/PA1-2 testcases/scene_whitted.txt output/whitted.bmp whitted
build/PA1-2 testcases/scene08_whitted.txt output/scene08_whitted.bmp whitted

# 路径追踪对比（2025-06-25 重渲，含贴地立方体）
build/PA1-2 testcases/scene_path.txt output/path_no_nee.bmp path 64
build/PA1-2 testcases/scene_path.txt output/path_nee.bmp path_nee 64
cp output/path_no_nee.bmp output/path_nee.bmp ../results/
build/PA1-2 testcases/scene08_path.txt output/scene08_path.bmp path 64
build/PA1-2 testcases/scene08_path.txt output/scene08_path_nee.bmp path_nee 64

# 光泽材质
build/PA1-2 testcases/scene_glossy.txt output/glossy.bmp path_nee 64
build/PA1-2 testcases/scene09_glossy.txt output/scene09_glossy.bmp path_nee 64

# Gamma 校正对比（bonus）
build/PA1-2 testcases/scene_whitted.txt output/gamma_before.bmp whitted
build/PA1-2 testcases/scene_whitted.txt output/gamma_after.bmp whitted gamma

# OpenMP 并行对比（bonus）：对比终端 Render time；两图应一致
build/PA1-2 testcases/scene_path.txt output/omp_serial.bmp path_nee 32
build/PA1-2 testcases/scene_path.txt output/omp_parallel.bmp path_nee 32 omp
cp output/omp_serial.bmp output/omp_parallel.bmp ../results/

# 纹理 Cornell 三面板（bonus，CPU only）
./build/gen_textures
./build/PA1-2 testcases/scene_texture_cornell_notex.txt output/texture_cornell_notex.bmp whitted 1 gamma
./build/PA1-2 testcases/scene_texture_cornell.txt output/texture_cornell.bmp whitted 1 gamma
./build/PA1-2 testcases/scene_texture_cornell_normal.txt output/texture_cornell_normal.bmp whitted 1 gamma

# Path Guiding 对比（bonus，CUDA only）
./build/PA1-2 testcases/scene_guiding_occluder.txt output/guiding_occluder_nee_64.bmp path_nee 64 gamma cuda
./build/PA1-2 testcases/scene_guiding_occluder.txt output/guiding_occluder_guided_64.bmp path_guiding 64 gamma cuda train_spp 256

# 回归
build/PA1-2 testcases/scene01_basic.txt output/scene01.bmp whitted
```
