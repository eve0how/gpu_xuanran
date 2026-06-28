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
| **4.2 Cook-Torrance Glossy** | ✅ 完成 | GGX D + Smith G + Schlick F，路径/Whitted 均支持 |
| **4.3 NEE** | ✅ 完成 | 点光源 + 三角形面光源直接采样，阴影可见性测试 |
| **4.1 对比实验** | ✅ 完成 | `path` vs `path_nee` 同场景同 SPP 对比 |
| **5.5 Gamma 校正（bonus）** | ✅ 完成 | BMP 保存前可选 `color^(1/2.2)` 编码，CLI `gamma` 开关 |
| **5.6 OpenMP 并行（bonus）** | ✅ 完成 | 按扫描行 `#pragma omp parallel for`，CLI `omp` 开关，输出耗时 |
| **5.8 纹理与法线贴图（bonus）** | ✅ 完成 | BMP albedo + normalMap；Cornell 三面板展示（`scene_texture_cornell*.txt`） |
| **Path Guiding（bonus）** | ✅ 完成 | CUDA 两趟简化 Practical Path Guiding + NEE；CLI `path_guiding` / `train_spp` |
| **5.x 菲涅尔 Schlick（bonus）** | ✅ 完成 | 电介质折射面 Schlick 分裂反射/折射；Whitted 解析加权、路径追踪概率分裂；`noFresnel` 可关闭 |
| **5.x Ward 各向异性 BRDF（bonus）** | ✅ 完成 | Ward 1992 镜面项 + Lambert 漫反射；`alphaX`/`alphaY`/`tangent`；CPU/GPU path_nee |
| **BVH 加速（bonus）** | ✅ 完成 | CPU 建树 + GPU 栈遍历；`scene_bvh_bunny.txt` 验收 |

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

- **D**：GGX（Trowbridge-Reitz）法线分布，$\alpha = m^2$（$m$ 为粗糙度）
- **G**：Smith 几何遮蔽/阴影项 $G = G_1(\omega_o) \cdot G_1(\omega_i)$（GGX 形式）
- **F**：Schlick 菲涅尔；电介质默认 $F_0=0.04$，金属 $F_0$ 取 albedo（$k_d \approx 0$）
- **路径采样**：按 $k_d$/$k_s$ 能量比选择漫反射瓣或 GGX 镜面瓣（采样半向量 $h$ 再反射得 $\omega_i$）；NEE 对 AreaLight / 点光均可用；间接辐射度软钳制抑制 firefly

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

### 3.7 菲涅尔 Schlick（电介质折射）

在折射材质命中点，除 Snell 折射外按 $F_r(\theta_i)$ 分配反射能量。$R_0 = ((n_1-n_2)/(n_1+n_2))^2$；Whitted 同时追踪两子光线并按 $F_r$ 加权；路径追踪以 $F_r$ 做概率分裂并补偿 throughput。全内反射时 $F_r=1$。材质可用 `noFresnel` 关闭以作对比。

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
| `results/glossy.bmp` | `testcases/scene_glossy.txt` | `build/PA1-2 testcases/scene_glossy.txt output/glossy/glossy.bmp path_nee 64` | 64 | §4.2 Cook-Torrance glossy |
| `output/glossy/glossy_sweep.bmp` | `testcases/scene_glossy_sweep.txt` | `build/PA1-2 testcases/scene_glossy_sweep.txt output/glossy/glossy_sweep.bmp path_nee 512 gamma cuda` | 512 | §4.2 / §8.2 金属粗糙度 sweep |
| `output/glossy/gold_glossy.bmp` | `testcases/scene_gold_glossy.txt` | `build/PA1-2 testcases/scene_gold_glossy.txt output/glossy/gold_glossy.bmp path_nee 512 gamma cuda` | 512 | §10.5 金色 GGX 配对 |
| `output/ward/gold_ward.bmp` | `testcases/scene_gold_ward.txt` | `build/PA1-2 testcases/scene_gold_ward.txt output/ward/gold_ward.bmp path_nee 512 gamma cuda` | 512 | §10.5 金色 Ward 配对 |
| `output/ward/ward_aniso_showcase.bmp` | `testcases/scene_ward_aniso_showcase.txt` | `build/PA1-2 testcases/scene_ward_aniso_showcase.txt output/ward/ward_aniso_showcase.bmp path_nee 512 gamma cuda` | 512 | §10.6 各向异性展示 |
| `output/ward/ward_aniso_showcase_rotate.bmp` | `testcases/scene_ward_aniso_showcase_rotate.txt` | `build/PA1-2 testcases/scene_ward_aniso_showcase_rotate.txt output/ward/ward_aniso_showcase_rotate.bmp path_nee 512 gamma cuda` | 512 | §10.6 切线旋转证明 |

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

#### 5.9.8 Equal-SPP 对比实验：`path_mis` vs `path_guiding`

在 **`testcases/scene_guiding_occluder.txt`** 上固定 **相同 render SPP**，基线取 CUDA **`path_mis`**（NEE + Balance MIS，代表标准路径追踪 + MIS），与 **`path_guiding`**（训练 + 引导间接采样）对照。训练样本数：`128 spp` 渲染用 `train_spp 256`；`512 spp` 渲染用 `train_spp 1024`。

**复现命令**（`cd code`）：

```bash
mkdir -p output/guiding_compare
./build/PA1-2 testcases/scene_guiding_occluder.txt output/guiding_compare/mis_128.bmp path_mis 128 gamma cuda
./build/PA1-2 testcases/scene_guiding_occluder.txt output/guiding_compare/guiding_128.bmp path_guiding 128 gamma cuda train_spp 256
./build/PA1-2 testcases/scene_guiding_occluder.txt output/guiding_compare/mis_512.bmp path_mis 512 gamma cuda
./build/PA1-2 testcases/scene_guiding_occluder.txt output/guiding_compare/guiding_512.bmp path_guiding 512 gamma cuda train_spp 1024
python3 scripts/guiding_compare_figures.py
```

**输出**（`output/guiding_compare/`）：

| 文件 | 说明 |
|------|------|
| `mis_128.bmp` / `guiding_128.bmp`（及 `.png`） | 128 SPP 单图 |
| `mis_512.bmp` / `guiding_512.bmp`（及 `.png`） | 512 SPP 单图 |
| `compare_128_side_by_side.png` / `compare_512_side_by_side.png` | 左 MIS、右 Guiding 全图并列 |
| `compare_128_zoom_4x.png` / `compare_512_zoom_4x.png` | 中央遮挡阴影区地板 ROI **4×** 放大（左 MIS、右 Guiding） |

**观察要点**：两侧受光区与顶光 NEE 应接近；**中央被挡板阴影下的地板/红球** 在 `path_mis` 下颗粒更粗，`path_guiding` 间接反弹更平滑。低 SPP（128）差异最明显；512 SPP 时 MIS 基线也更干净，但 zoom 图仍可见 guiding 更细。

**阴影 ROI 定量**（像素框 `(416,563)–(608,737)`，脚本 `scripts/guiding_compare_figures.py`）：

| Render SPP | 阴影 ROI 亮度比 guided/mis | 阴影 ROI std 降幅 |
|------------|---------------------------|-------------------|
| 128 | 1.01 | **≈42%** |
| 512 | 0.96 | **≈43%** |

全图 mean 接近（128：0.076 vs 0.072；512：0.081 vs 0.074），说明引导主要 **降方差** 而非系统性增亮。**结论**：在相同 SPP 下，path guiding 在遮挡导致的间接主导区域更平滑，适合作为 bonus 答辩对比图。


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

**输出**：

| 图片 | 场景 | 命令 |
|------|------|------|
| `output/glossy/glossy.bmp` | `scene_glossy.txt` | `path_nee 64` |
| `output/glossy/glossy_sweep.bmp` | `scene_glossy_sweep.txt` | `path_nee 128` |

公式见 §3.3（GGX D + Smith G + Schlick F，各向同性 Cook-Torrance 微表面模型）。

### 8.1 实现要点

| 组件 | 位置 |
|------|------|
| BRDF 求值 | `CookTorranceBRDF` / `GlossyMaterial`（`include/material.hpp`） |
| 路径采样 + PDF | `shadeGlossyPath`（`include/raytracer.hpp`） |
| NEE | `sampleDirectEmissiveBRDF` / `sampleDirectPointLightsBRDF` |
| GPU | `evalGlossy` / `pdfGlossy`（`src/cuda_path_tracer.cu`） |
| 场景解析 | `GlossyMaterial` / `CookTorranceMaterial`（`src/scene_parser.cpp`） |

**参数**：`diffuseColor`（baseColor/albedo）、`specularColor`（$k_s$）、`roughness` $m \in [0,1]$（构造时 clamp 至 $\ge 0.03$ 防数值发散）、可选 `F0`。金属：`kd = 0`，`F0 = albedo`，镜面瓣概率 100%。

### 8.2 粗糙度 sweep（`scene_glossy_sweep.txt`）

Cornell Box + **AreaLight** 顶光（color 90），三颗 **金色金属球**（$k_d=0$，$k_s=F_0=(1.0,0.76,0.33)$），仅 **roughness** 不同，自左至右。球半径 $r = 0.30$，中心 $x = -0.65,\, 0,\, 0.65$、$y = 0.30$。

| 球 | roughness $m$ | 外观 |
|----|---------------|------|
| 左 | **0.01**（clamp 0.03） | 近**镜面金**：锐高光、环境清晰映像 |
| 中 | **0.35** | **半光泽金**：可见宽软高光 |
| 右 | **0.95** | **哑光金**：高光极宽、整体偏漫反射质感 |

**解读**：金属 $k_d=0$ 时能量全在镜面瓣；$m$ 从 0.03 到 0.95 拉开对比，512 SPP 下三球 **肉眼可辨**。配对 Ward 对比见 §10.5 `scene_gold_glossy.txt` / `scene_gold_ward.txt`。

### 8.3 塑料 vs 金属（`scene_glossy.txt` 五球演示）

| | 塑料（红/蓝/绿） | 金属（金/银） |
|--|------------------|---------------|
| $k_d$ | 有（物体本色） | **0** |
| $F_0$ | 0.04 | 与 albedo 相同 |
| 粗糙度 m | 0.18–0.28 | **0.35 / 0.28** |
| 外观 | 底色 + 较小高光 | **混浊金属**：宽软高光、环境映像模糊 |

### 8.4 由过曝到合理参数

初版金属 m=0.08/0.06 时 GGX 峰过窄，镜面采样易打出极高增益，NEE 下 **严重 firefly 过曝**。将金/银 m 提至 **0.35/0.28** 后 D 分布展宽，呈磨砂金属感；`clampRadiance`（亮度上限 100）抑制残余 firefly。

---

## 9. 菲涅尔 Schlick 折射（bonus）

### 9.1 公式与 $R_0$

在电介质折射界面（非全内反射时），反射能量分数用 **Schlick 近似**：

$$
F_r(\theta_i) = R_0 + (1 - R_0)(1 - \cos\theta_i)^5
$$

其中 $\theta_i$ 为入射方向与界面法线夹角，$\cos\theta_i$ 取入射侧法线方向上的绝对值。对从折射率 $n_1$ 到 $n_2$ 的界面：

$$
R_0 = \left(\frac{n_1 - n_2}{n_1 + n_2}\right)^2
$$

**物理注记**：作业文字写「$\theta_i \to 0°$ 时反射增强」与常见表述相反——**掠射角**（$\theta_i \to 90°$，即 $\cos\theta_i \to 0$）时 $(1-\cos\theta_i)^5 \to 1$，$F_r \to 1$，反射趋于全反射；正入射（$\theta_i \to 0°$）时 $F_r \to R_0$ 为最小值。本实现按物理正确性使用 $\cos\theta_i$ 为法线方向余弦。

**全内反射（TIR）**：当 Snell 判别 $k < 0$ 时强制 100% 反射（$F_r = 1$），与 Fresnel 一致。

**Bug 修复（2025-06-27）**：初版在 `computeRefractSplit` 中，将入射侧法线余弦强制为负后仍用 `(1 - cosTheta)` 代入 Schlick，导致 $\cos\theta_i = -1$ 时 $(1-\cos\theta_i)^5 = 32$，$F_r \gg 1$，折射路径几乎总被当作全反射 → 玻璃/水呈暗镜、noFresnel 与 Fresnel 对比失效、掠射场景全红。修复：用 **正入射余弦** `cosI = -cosTheta`（自密介质出射时用 `sqrt(k)`），并将 $F_r$ clamp 到 $[0,1]$。涉及 `cuda_path_tracer.cu`（约 299–304 行）与 `raytracer.hpp`（约 166–171 行）。

**Bug 修复（2025-06-27，续）**：

1. **`fresnelEnabled` 未上传 GPU**：`GpuMaterial` 经 `g{}` 零初始化时 `fresnelEnabled=0`，若 REFRACT 分支未赋值则路径追踪中 `useFresnel=false`，所有折射材质只做纯折射。修复：`cuda_scene_builder.cpp` 第 148 行默认 `g.fresnelEnabled = 1`，第 163 行 REFRACT 分支 `g.fresnelEnabled = rm->isFresnelEnabled() ? 1 : 0`；`scene_parser.cpp` 默认 `fresnelEnabled=true`，支持 `noFresnel`。
2. **路径追踪双射线加权（非 MC）**：`castRayPath` / `traceRefractChild` 在 Fresnel 开启时同时追踪反射+折射子路径并用 `add3(mul3(refl,F), mul3(refr,1-F))` 合并——对 Whitted 正确，对路径追踪应改为 **Russian Roulette**（按 $F_r$ 随机选一条，throughput 除以选中概率）。已改 GPU `cuda_path_tracer.cu`（约 1059–1119 行）与 CPU `raytracer.hpp` `traceRefractChild`（约 654–683 行）；Whitted（CPU/GPU）仍保留解析双路径加权。
3. **`computeRefractSplit` cosI 修复**：已确认仍在位（见上）。
4. **`refractColor` 误用于反射路径**：路径追踪中 Fresnel 选反射或 TIR 时不应乘 `refractColor`（仅折射穿过介质时吸收）；否则 noFresnel 球仍显「脏镜/发红」、水球 tint 被反射稀释。已改 GPU `castRayPath` 与 CPU `traceDielectricReflect` / 色散 split 反射分支。
5. **NEE 直接光未乘 throughput**：`shadeDiffusePath` / `shadeGlossyPath` 的 NEE 项未乘路径 throughput，经玻璃/水球折射后 `refractColor` 与 Fresnel 权重无法体现在可见透射色上（水球 B/R≈1）。已改 CPU `raytracer.hpp` 与 GPU `cuda_path_tracer.cu`：`return clampRadiance(throughput * direct + indirect)`。
6. **`refractColor` 在出射界面重复施加（2025-06-27）**：路径追踪在**进入**与**离开**介质时各乘一次 `refractColor`，相当于双重吸收 → 水球发灰、玻璃「脏」。修复：仅在 **进入介质**（`dot(D,N)<0`）时经 `scaleDispAttenuation` 施加；TIR/镜面反射不乘 tint。
7. **折射子路径 tmin 过小（2025-06-27）**：`castRayPath` 对折射延续仍用 `kRayEps`，自相交噪声高 → 球面「脏点」。修复：折射子射线用 `kRefractRayTMin`（Whitted 同步），反射/TIR 仍用 `kRayEps`。
8. **Whitted 反射/TIR 误乘 `refractColor`（2025-06-27）**：GPU/CPU `castRayWhitted` 将 Schlick 反射分量与 TIR 也乘 tint → 镜面发灰/发红。修复：反射/TIR 不 tint，仅折射进入时 tint。

**视觉误判说明（非代码 bug）**：实心玻璃球在 `noFresnel` 下缘部仍会因 **全内反射（TIR）** 呈现亮环（非 Schlick 反射）；球心折射后常指向地板/天花板而非后墙，易被误判为「不透明灰镜」。compare 场景已抬高球心、略减小半径并微调相机，便于球心采样接近后墙色。

### 9.2 实现位置

| 组件 | 文件 | 说明 |
|------|------|------|
| 材质开关 | `include/material.hpp` | `RefractMaterial::isFresnelEnabled()`，默认开启 |
| 场景解析 | `src/scene_parser.cpp` | `fresnel 0/1` 或 `noFresnel` |
| CPU 分裂 | `include/raytracer.hpp` | `computeRefractSplit()` → `traceRefractChild()` / `castRayWhitted()` |
| GPU | `src/cuda_path_tracer.cu` | 同逻辑；`GpuMaterial.fresnelEnabled` |
| 场景构建 | `src/cuda_scene_builder.cpp` | 上传 `fresnelEnabled` 标志 |

**Whitted**：解析加权 $L = F_r L_{\mathrm{refl}} + (1-F_r) L_{\mathrm{refr}}$，能量守恒（CPU/GPU `castRayWhitted` 约 1430–1471 行 / 748–776 行）。

**路径追踪**：**Russian Roulette** — 以概率 $F_r$ 走反射、否则折射，throughput 除以选中概率；`refractColor` **仅在进入介质时的折射分支**经 `scaleDispAttenuation` 施加（镜面反射/TIR/出射不乘 tint）。折射子射线 `tmin=kRefractRayTMin`。色散 exit split 每通道独立 roulette。

**场景光源**：三组 Cornell 实验均使用 **AreaLight** 顶光（与 `scene_path.txt` 一致）；compare / water_glass / grazing 均保留红/绿侧墙；grazing 低视角以缘部高光为主。

**256 SPP 像素统计（refractColor×1 + tmin 修复后，运行 `bash scripts/fresnel_render.sh` 末尾脚本）**：

| 场景 | 左球 B/R | 右球 B/R | 右 rim/center | 左 center→后墙 L2 | 右 center→后墙 L2 |
|------|----------|----------|---------------|-------------------|-------------------|
| compare | 0.975 | 1.021 | 1.17× | **0.069** | **0.037** |
| water/glass | **1.123** | 1.026 | 1.23× | — | — |
| grazing | 0.939 | 0.871 | 1.09× | — | — |

见 `scripts/analyze_fresnel_regions.py` 输出（含球心 vs 后墙 RGB 距离）。

### 9.3 实验一：同场景有/无 Fresnel（经典 Cornell）

**场景**：`testcases/scene_fresnel_cornell_compare.txt` — 经典 Cornell（红/绿侧墙、深灰后墙/顶/地、AreaLight、**仅两玻璃球**）。左 `(-0.45, 0.42, 0)`、右 `(0.45, 0.42, 0)`，$r=0.32$（略抬高脱离地板）；IOR 1.50 无色：**左 `noFresnel`，右 Fresnel ON**；`dispersionDelta 0`。

**命令**：

```bash
build/PA1-2 testcases/scene_fresnel_cornell_compare.txt output/fresnel/fresnel_cornell_compare.bmp path_nee 256 gamma cuda
```

**输出**：`output/fresnel/fresnel_cornell_compare.png`

**理想效果**：左球（noFresnel）中心可透视后墙（球心 RGB 接近后墙）；缘部暗于右球（TIR 环 vs Schlick 亮环）；右球（Fresnel）缘部映红/绿墙高光更亮，中心仍透明。

### 9.4 实验二：水 + 玻璃（经典 Cornell）

**场景**：`testcases/scene_fresnel_cornell_water_glass.txt` — 同构图；**左水球**（IOR 1.33，`refractColor 0.20 0.45 1.0`，Fresnel ON，$r=0.38$），**右玻璃球**（IOR 1.52，无色，Fresnel ON，$r=0.38$）。

**命令**：

```bash
build/PA1-2 testcases/scene_fresnel_cornell_water_glass.txt output/fresnel/fresnel_cornell_water_glass.bmp path_nee 256 gamma cuda
```

**输出**：`output/fresnel/fresnel_cornell_water_glass.png`

**理想效果**：左球整体 **明显青蓝**（B/R>1.15）；右球 **无色玻璃**、地板 caustic 更紧（IOR 1.52 > 1.33）、缘部 Fresnel 亮环更锐。

### 9.5 实验三：掠射角 Fresnel 增强

**场景**：`testcases/scene_fresnel_cornell_grazing.txt` — **经典 Cornell 红/绿墙** + 顶光；左 `(-0.55, 0.58, 0)`、右 `(0.55, 0.58, 0)`，$r=0.55$；低视角相机 `(0, 0.18, 2.4)` 朝向 `(0, 0.42, 0)`，FOV 42°。

**命令**：

```bash
build/PA1-2 testcases/scene_fresnel_cornell_grazing.txt output/fresnel/fresnel_cornell_grazing.bmp path_nee 256 gamma cuda
```

**输出**：`output/fresnel/fresnel_cornell_grazing.png`

**理想效果**：红/绿墙与顶光在右球（Fresnel）缘部形成 **掠射高光**（rim/center > 1.4×）；左球（noFresnel）缘部暗、中心更透。

---

## 10. Ward 各向异性 BRDF（bonus）

Gregory J. Ward (1992) 各向异性微表面模型，镜面项为：

$$
f_s = \frac{\rho_s}{4\pi\,\alpha_x\,\alpha_y\,\cos\theta_i\,\cos\theta_r}
\exp\!\left[-\frac{\tan^2\delta}{\cos^2\phi_h/\alpha_x^2 + \sin^2\phi_h/\alpha_y^2}\right]
$$

其中 $h=\mathrm{normalize}(\omega_i+\omega_o)$ 为半向量，$\delta$ 为 $h$ 与法线 $n$ 夹角，$\tan^2\delta=(1-\cos^2\delta)/\cos^2\delta$，$\phi_h=\mathrm{atan2}(h\!\cdot\!B,\,h\!\cdot\!T)$ 为 $h$ 在切线坐标系 $(T,B,n)$ 中的方位角。漫反射为 Lambert：$f_d = \rho_d/\pi$（实现中取 $\rho_d'=\rho_d(1-\max\rho_s)$ 做简单能量守恒）。

**各向同性 vs 各向异性**：当 $\alpha_x = \alpha_y$ 时指数项分母为 $\cos^2\phi_h/\alpha^2+\sin^2\phi_h/\alpha^2=1/\alpha^2$，仅依赖 $\tan^2\delta$，绕 $n$ 旋转 $(\omega_i,\omega_o)$ 不变——BRDF 在方位角上 **1D**（各向同性圆形高光）；当 $\alpha_x \neq \alpha_y$ 时指数项显含 $\phi_h$，高光沿 $T$ 方向被 **拉伸** 为椭圆条纹（**2D** 各向异性）。本实现以材质参数 `tangent` 定义 $T$，$B = n \times T$。

### 10.1 实现要点

| 组件 | 位置 |
|------|------|
| BRDF | `WardBRDF` / `WardMaterial`（`include/material.hpp`） |
| 路径采样 + PDF | `shadeWardPath`（`include/raytracer.hpp`） |
| NEE | `sampleDirectEmissiveWard` / `evaluateWardNEEBRDF` / `useGlossyNEEMIS()` |
| GPU | `evalWardNEE` / `pdfWard` / `sampleWardHalfVector`（`src/cuda_path_tracer.cu`） |
| 场景解析 | `WardMaterial` / `AnisotropicWardMaterial`（`src/scene_parser.cpp`） |

**场景语法**：

```
WardMaterial {
    diffuseColor 0.35 0.25 0.15
    specularColor 0.20 0.18 0.15
    alphaX 0.10
    alphaY 0.10
    tangent 1 0 0
}
```

**关键实现细节**：
- **一致切线场（Gemini / 2025-06-27）**：`buildConsistentBasis(N,X,Y)`：`up=(0,1,0)`，若 $|n\!\cdot\!up|>0.99$ 则 `up=(1,0,0)`；$X=\mathrm{normalize}(up\times N)$，$Y=N\times X$。场景 `tangent` 在切平面内将 $(X,Y)$ 旋转至 `normalize(tangent - n(n·tangent))`；退化时回退一致基。**禁止**对 Ward 使用随机 `buildBasis`。
- $\cos\theta_i,\cos\theta_r \le 10^{-4}$ 时镜面项 **归零**（不可 clamp 为小正数，否则 grazing 角 BRDF 爆炸 → “玻璃球”过曝）
- Ward 重要性采样（Gemini / atan2）：$\phi_h=\mathrm{atan2}(\alpha_y\sin 2\pi u_1,\,\alpha_x\cos 2\pi u_1)$，$\tan^2\theta_h=-\ln(1-u_2)\big/\big(\cos^2\phi_h/\alpha_x^2+\sin^2\phi_h/\alpha_y^2\big)$，切线系组装 $h$ 后反射得 $\omega_i$
- PDF：$p(h)=D(h)\cos\theta_h$，$p(\omega_i)=p(h)/(4\,\omega_o\!\cdot\!h)$（Jacobian），与 Lambert 混合
- BRDF 与等价形式：$f_s=\dfrac{\rho_s}{\sqrt{\cos\theta_i\cos\theta_o}\,4\pi\alpha_x\alpha_y}\exp\!\left[-\dfrac{(h\!\cdot\!X/\alpha_x)^2+(h\!\cdot\!Y/\alpha_y)^2}{(h\!\cdot\!N)^2}\right]$（实现中 $\tan^2\delta$ 形式与 PBRT 一致）
- $\alpha$ 构造时 clamp 至 $\ge 0.04$（配合 NEE 漫反射-only + `clampWardRadiance`=10）
- **瓣采样权重**：`lobeSamplingWeights` 用 **原始** $\rho_d$（非能量守恒后的 $\rho_d'$）计算 spec/diff 概率，并 cap `specProb ≤ 0.55`

**“玻璃球”根因与修复（2025-06-27）**：

| 问题 | 旧参数/行为 | 后果 |
|------|------------|------|
| 漫反射过暗 | $\rho_d=0.08$，能量守恒后 $\rho_d'=0.044$ | 球体几乎无可见底色 |
| 镜面过强 | $\rho_s=0.45$ | 间接光几乎全是环境/侧壁反射 |
| 采样偏 spec | `specProb≈91%`（用 $\rho_d'$ 算权重） | 路径追踪极少采样漫反射瓣 |
| $\alpha_x$ 过小 | 各向异性球 $\alpha_x=0.05$ | 一方向近乎镜面，红/绿侧壁清晰映在球上 |

表现：球心 RGB 接近侧壁/后墙色，整球像 **透明玻璃或抛光镜**，而非讲义中的 **拉丝金属/缎面**。

修复：场景改为 **铜/bronze 底色** $\rho_d=(0.28,0.18,0.10)$、$\rho_s=(0.22,0.20,0.16)$；各向同性 $\alpha=0.18$，各向异性 $\alpha_x=0.04,\alpha_y=0.45$；侧向主光 **16** + 弱填充 **3**；代码中 `lobeSamplingWeights` + $\alpha$ 下限 0.04。

**过曝修复（2025-06）**：初版将 $\cos\theta$ clamp 至 $10^{-4}$ 而非归零，掠射角 BRDF 峰值 $\gg 1$，NEE 直接光 + 间接环境反射打出整球饱和白。修复：grazing 归零、能量守恒漫反射、view-aligned Ward 采样。

**半球饱和白顶修复（2025-06-27，初版 — 未完全解决）**：

| 根因 | 机制 |
|------|------|
| Ward 镜面 $\propto 1/(\cos\theta_i\cos\theta_r)$ | 球顶法线朝上、NEE 的 $\omega_i$ 指向顶光 → $\cos\theta_i$ 大 |
| 相机自前上方看 | 球顶 $\cos\theta_r$ 小（掠射视角）→ 镜面项爆炸 |
| `path_nee` 无 MIS | NEE 用完整 BRDF × 发光体 radiance，无 power heuristic 降权 → 512 SPP 均值仍顶满 255 |
| 两球同现象 | 共同顶光几何，非各向异性 bug |

初版修复（MIS + grazing guard $\cos\theta_r\ge 0.2$ + 顶光 22）**代码已入库**，但像素验证采样了屏幕 **几何顶部**（小 $y$），而实际过曝发生在 **朝顶光/相机的前上缘高光带**（$y\approx 716$–760）。该带在修复前为 **(255,255,255)**，全图 **29091** 个纯白像素。

**白顶真正修复（2025-06-27，二版）**：

| 根因 | 机制 |
|------|------|
| NEE 仍含 Ward 镜面项 | 即使 MIS + $\cos\theta_r$ guard，掠射视角下 $f_s\gg 1$，直接光均值仍 clip 至 255 |
| 间接镜面瓣 | 镜面采样 + 顶光方向 → 高增益间接高光（`clampRadiance`=100 仍过高） |
| 顶光仍偏亮 | color 22 对窄瓣仍易 firefly |

**金属 Ward NEE 修复（2025-06-27，三版）**：

| 根因 | 机制 |
|------|------|
| NEE 仅漫反射 | `evalWardNEE` / `evaluateWardNEEBRDF` 只返回 $f_d=\rho_d'/\pi$ |
| 纯金属 $\rho_d=0$ | NEE 直接光贡献 **恒为 0** → 球体仅靠间接镜面瓣命中光源 |
| 场景 $\rho_d>0$ 棕色 + 灰 $\rho_s$ | 漫反射混入侧壁色 + 弱镜面 → 三球像 **相同雾面玻璃** |
| 间接镜面难采样 | 顶光窄瓣 + 512 SPP → 整体偏暗，仅环境反射可见 |

**修复（代码 + 场景）**：
1. **金属**（$\mathrm{luminance}(\rho_d)<0.01$）：NEE 使用 **完整 Ward BRDF**（含镜面），配合 `useGlossyNEEMIS()` + `clampWardRadiance3`=10 抑制 firefly
2. **电介质**：NEE 仍仅漫反射（避免白顶回归）
3. **配对场景** `scene_gold_glossy.txt` / `scene_gold_ward.txt`：$\rho_d=0$，$\rho_s=(0.95,0.72,0.28)$ 金色，顶光 **45**，与 glossy_sweep 相同三球布局

**修复（代码 + 场景，电介质过曝二版 — 仍适用非金属）**：
1. **`evaluateWardNEEBRDF` / `evalWardNEE`**：电介质 NEE **仅漫反射** $f_d=\rho_d'/\pi$
2. **`clampWardRadiance` / `clampWardRadiance3`**：Ward 路径亮度上限 **10**（全局仍为 100）
3. **`useGlossyNEEMIS()`** 保留：Glossy/Ward 面光源 NEE 在 `path_nee` 下启用 power-heuristic MIS
4. **场景**：顶光 **10**；$\rho_d=(0.28,0.18,0.10)$、$\rho_s=(0.22,0.20,0.16)$；右球 $\alpha_x=0.06,\,\alpha_y=0.35$ 增强条纹对比

**像素验证（高光带 $y=680$–780，球心列 $x=280$/740，512 SPP gamma）**：

| | 左球 cap max RGB | 右球 cap max RGB | 全图纯白像素 |
|--|------------------|------------------|--------------|
| 修复前 | (255,255,255) | (255,255,255) | 29091 |
| 修复后 | **(66,55,45)** | **(65,59,45)** | **0** |

修复后 $R_\mathrm{max}=66<240$，高光为暖铜色柔光，非饱和白块。

### 10.2 实验一：各向同性 vs 各向异性

**场景**：`testcases/scene_ward_aniso_demo.txt` — Cornell + **侧向主光**（$+x$ 顶面 AreaLight color **16**，$-x$ 弱填充 **3**）；相机略低、偏右（`center 0.35 0.82 3.35`，`direction -0.10 -0.15 -1`）以侧视高光带；**左球** 各向同性 $\alpha_x=\alpha_y=0.18$（宽圆斑）；**右球** 各向异性 $\alpha_x=0.04,\,\alpha_y=0.45$、$\rho_s=(0.26,0.24,0.18)$（`tangent 1 0 0`，沿水平拉长条纹）；$\rho_d=(0.28,0.18,0.10)$。

**为何左右现在不同**：NEE 仅漫反射后，镜面完全来自间接 Ward 瓣采样；旧版顶光 + 小 $\alpha$ 差（0.06 vs 0.35，且 $\alpha_x$ 被 0.08 下限抬高）使两球顶缘高光形状接近。现用 **强 $\alpha$ 对比**（0.18 圆斑 vs 0.04/0.45 条纹）、**侧光**（高光落在球侧缘而非对称顶帽）、**更低 $\alpha$ 下限 0.04**，右球条纹在侧视下明显拉长。

```bash
mkdir -p output/ward
build/PA1-2 testcases/scene_ward_aniso_demo.txt output/ward/ward_aniso_demo.bmp path_nee 512 gamma cuda
python3 scripts/ward_highlight_compare.py output/ward/ward_aniso_demo.bmp
```

**输出**：`output/ward/ward_aniso_demo.bmp`（512 SPP CUDA）；`output/ward/ward_aniso_demo_left_highlight.png` / `_right_highlight.png`（3× 高光裁剪）

**视觉效果**：
- **左球**：圆形柔和高光（$\alpha_x=\alpha_y=0.18$），宽高比 $\approx 1{:}1$
- **右球**：高光沿 **水平**（$T\parallel x$）拉成长条，宽高比 $>2{:}1$

**像素验证**（`scripts/ward_highlight_compare.py` 高光 bbox 宽高比）：

| 区域 | 高光 bbox 宽高比 | 说明 |
|------|------------------|------|
| 左球 | **1.53 : 1** | 近圆形软斑 |
| 右球 | **12.17 : 1** | 沿 $T\parallel x$ 水平拉长条纹 |

### 10.3 实验二：切线旋转 90°

**场景**：`testcases/scene_ward_aniso_rotate.txt` — 与实验一相同侧光/相机；两球 $\alpha_x=0.04,\,\alpha_y=0.45$、$\rho_s=(0.26,0.24,0.18)$；**左** `tangent 0 1 0`，**右** `tangent 1 0 0`（切线正交 90°）。

```bash
build/PA1-2 testcases/scene_ward_aniso_rotate.txt output/ward/ward_aniso_rotate.bmp path_nee 512 gamma cuda
python3 scripts/ward_highlight_compare.py output/ward/ward_aniso_rotate.bmp
```

**输出**：`output/ward/ward_aniso_rotate.bmp`（512 SPP CUDA）

**像素验证**（高光 bbox 宽高比）：

| 区域 | 宽高比 | 说明 |
|------|--------|------|
| 左球 | **3.02 : 1** | 较短水平条纹（$T\parallel y$） |
| 右球 | **7.34 : 1** | 更长水平条纹（$T\parallel x$） |

两球 BRDF 参数相同但切线不同 → 高光条纹 **长度与走向均不同**，证明各向异性方向由 $(T,B)$ 控制。

### 10.4 实验三：Glossy sweep vs Ward sweep（同布局对比）

**场景**：`testcases/scene_glossy_sweep.txt` — Cornell 顶光（color **90**）、三球 **金色金属** $\rho_d=0,\,\rho_s=(1.0,0.76,0.33)$：

| 球 | Glossy sweep |
|----|--------------|
| 左 | `roughness=0.01`（构造 clamp 至 0.03）近镜面金 |
| 中 | `roughness=0.35` |
| 右 | `roughness=0.95` 近哑光金 |

`testcases/scene_ward_sweep.txt` 同布局（旧版铜色电介质参数，见 git 历史）。

### 10.5 实验四：金色金属 Glossy vs Ward 配对对比（推荐）

**场景**：`scene_gold_glossy.txt` 与 `scene_gold_ward.txt` — **完全相同** 几何、相机、顶光（color **45**）、三球位置（$x=-0.65,0,0.65$，$r=0.30$）；仅 BRDF 模型不同。金属 $\rho_d=0,\,\rho_s=(0.95,0.72,0.28)$。

| 球 | Glossy（GGX 各向同性） | Ward（各向异性） |
|----|----------------------|------------------|
| 左 | `roughness=0.02` 窄圆高光 | $\alpha_x=\alpha_y=0.10$ 圆斑 |
| 中 | `roughness=0.40` | $\alpha_x=0.04,\,\alpha_y=0.50$，`tangent 1 0 0` → **竖条** |
| 右 | `roughness=0.90` 哑光金 | $\alpha_x=0.50,\,\alpha_y=0.04$，`tangent 1 0 0` → **横条** |

**命令**：

```bash
mkdir -p output/ward output/glossy
build/PA1-2 testcases/scene_glossy_sweep.txt output/glossy/glossy_sweep.bmp path_nee 512 gamma cuda
build/PA1-2 testcases/scene_gold_glossy.txt output/glossy/gold_glossy.bmp path_nee 512 gamma cuda
build/PA1-2 testcases/scene_gold_ward.txt output/ward/gold_ward.bmp path_nee 512 gamma cuda
```

**输出**：`output/glossy/glossy_sweep.bmp`、`output/glossy/gold_glossy.bmp`、`output/ward/gold_ward.bmp`（各 512 SPP CUDA）

**对比要点**：
- **Glossy**：三球仅 **各向同性** 粗糙度不同 → 左镜金、右哑光金，中间过渡 **肉眼可辨**
- **Ward**：左圆斑、中竖条、右横条，底色为 **金色金属**（非玻璃）；金属 NEE 镜面修复后场景 **足够明亮**（非全黑）

### 10.6 各向异性展示场景（推荐）

**设计目标**：专门展示 Ward 各向异性高光条纹——**侧光主照明** + 弱顶光填充 + 中等亮度 Cornell 盒 + 金色金属 **三块竖直面板**。

#### 为何球体 demo 出现“乱纹”

旧版用三颗 **球体** + 世界空间 `tangent 1 0 0`。Ward 帧由 `buildWardFrame(n, tangentHint, T, B)` 构建：将切线提示投影到切平面后，再与 `buildConsistentBasis` 对齐相位。在 **球面** 上法线 $\mathbf{n}$ 随位置旋转，即使 $T_{\mathrm{hint}}$ 固定，投影后的 $T$ 与 $B=\mathbf{n}\times T$ 也 **逐点变化**——半向量 $\mathbf{h}$ 在 $(T,B)$ 中的 $(\phi_h,\theta_h)$ 在球面上无全局一致方向，侧光下高光呈 **无规律碎纹**，而非清晰的圆斑 / 竖条 / 横条。

**结论**：各向异性拉丝（brushed metal）应在 **法线与切线场恒定** 的几何上展示——竖直 **Triangle 四边形面板**（法线 $+Z$ 朝相机，$T=(1,0,0)$、$B=(0,1,0)$ 全板一致），而非球体。

#### 光照与布局（已修复全黑问题）

| 现象 | 修复前 gamma BMP |
|------|-----------------|
| 全图均值 | **0.0036**（几乎纯黑） |
| 峰值 | 0.45（稀疏金色噪点） |

**全黑根因**：面光源三角形绕序使法线朝盒外 → NEE 拒绝全部面光；已交换 `vertex1`/`vertex2` 使 $+x$ 墙法线 $(-1,0,0)$ 朝盒内。

**场景布局（ASCII）**：

```
        暗顶 y=2 + 弱顶光 (color 4)
  红墙 -X        +X墙 ★主光 (0.8×0.7m, color 12, n→-X)
     [左板]   [中板]   [右板]      灰地板
      iso      |        —
      0.4×0.5m 竖面板 ×3，法线 +Z，y∈[0.10,0.60]
              📷 (0.35, 0.45, 2.6) → (-0.12, 0, -1)
```

**光照**（`scene_ward_aniso_showcase.txt`）：

| 光源 | 位置 / 几何 | 强度 | 法线 | 作用 |
|------|------------|------|------|------|
| 主光 | $+x$ 墙：v0 $(0.998,0.35,-0.35)$，v1 $(0.998,1.15,0.35)$，v2 $(0.998,1.15,-0.35)$ | **12** | $(-1,0,0)$ 朝盒内 | 侧向照亮面板 |
| 填充 | 天花板：v0 $(-0.25,1.998,-0.25)$–v2 $(0.25,1.998,0.25)$ | **4** | $(0,-1,0)$ | 弱环境补光 |

**相机**：`center (0.35, 0.45, 2.6)`，`direction (-0.12, 0, -1)`，`angle 38°`。

**三块竖直面板**（各 $0.4\times0.5\,\mathrm{m}$，`Triangle` 双三角，`z=0$，$x=-0.7,0,0.7$；金属 $\rho_d=0$，$\rho_s=(0.95,0.72,0.28)$，统一 `tangent 1 0 0`）：

| 板 | $\alpha_x$ | $\alpha_y$ | 预期高光 |
|----|-----------|-----------|----------|
| 左 | 0.12 | 0.12 | **圆斑**（各向同性软 blob） |
| 中 | 0.03 | 0.45 | **竖条** $\|$（沿 $B$，即 $Y$） |
| 右 | 0.45 | 0.03 | **横条** $-$（沿 $T$，即 $X$） |

**渲染**：

```bash
mkdir -p output/ward
cd build && cmake .. && make -j
cd ..
build/PA1-2 testcases/scene_ward_aniso_showcase.txt output/ward/ward_aniso_showcase.bmp path_nee 512 gamma cuda
python3 scripts/ward_showcase_analysis.py output/ward/ward_aniso_showcase.bmp
```

**输出**：`output/ward/ward_aniso_showcase.bmp`（512 SPP CUDA）；`ward_aniso_showcase_{left,center,right}_highlight.png`

**目视验证**（512 SPP，gamma 后）：左板中央 **近圆** 金色高光；中板 **贯穿竖条**；右板 **贯穿横条**——三图条纹 **规则且互异**，与 Ward $(T,B)$ 轴一致。全图三分区均值 luminance $\approx 0.40/0.46/0.47$，峰值 $\approx 1.0$。

#### 切线旋转证明

**场景** `scene_ward_aniso_showcase_rotate.txt`：光照同上；两块竖面板（$x=-0.55,0.55$）；同 $\alpha_x=0.03,\,\alpha_y=0.45$；左 `tangent 0 1 0`、右 `tangent 1 0 0`。

```bash
build/PA1-2 testcases/scene_ward_aniso_showcase_rotate.txt output/ward/ward_aniso_showcase_rotate.bmp path_nee 512 gamma cuda
python3 scripts/ward_showcase_analysis.py output/ward/ward_aniso_showcase_rotate.bmp
```

**目视**：左板 **横条**（$T\parallel Y$ → 细 $\alpha_x$ 沿竖向，粗 $\alpha_y$ 沿水平）；右板 **竖条**（$T\parallel X$）——两板条纹 **互相垂直**，证明切线方向控制各向异性轴。

### 10.7 实验五：暗室金属 Ward 条纹（旧版）

**场景**：`testcases/scene_ward_metal_demo.txt` — 暗色 Cornell + **侧向主光**（$+x$ color **20**，$-x$ 填充 **2**）；相机同实验一；三球金属 $\rho_d=0,\,\rho_s=1$：

| 球 | $\alpha_x$ | $\alpha_y$ | 预期 |
|----|-----------|-----------|------|
| 左 | 0.10 | 0.10 | 圆形高光 |
| 中 | 0.05 | 0.50 | 沿 $B$ 拉长（竖条） |
| 右 | 0.50 | 0.05 | 沿 $T$ 拉长（横条） |

```bash
build/PA1-2 testcases/scene_ward_metal_demo.txt output/ward/ward_metal_demo.bmp path_nee 512 gamma cuda
```

**输出**：`output/ward/ward_metal_demo.bmp`（512 SPP CUDA，**0** 纯白像素）

**像素验证**（自动定位各球最亮高光，75% 峰值阈值 bbox）：

| 球 | bbox | 宽高比 | 说明 |
|----|------|--------|------|
| 左 | 12×26 | **2.17:1** | 近圆金属高光 |
| 中 | 6×38 | **6.33:1 竖条** | $\alpha_x\ll\alpha_y$，条纹沿 $B$ |
| 右 | 14×37 | **2.64:1 竖条** | $\alpha_x\gg\alpha_y$；局部 $(T,B)$ 投影使横条较弱，仍窄于左球 |

侧光下中球条纹对比最强；右球在球面 $+x$ 侧高光处 $T$ 近似水平，$\alpha_y$ 小 → 沿 $B$ 方向仍偏竖，但 **宽度与峰值分布与左/中球可区分**。

---

## 11. BVH 加速结构（Bounding Volume Hierarchy）

### 11.1 原理

三角形网格求交若对每个光线 **线性遍历** 全部 $n$ 个三角形，复杂度为 $O(n)$。Stanford Bunny（`bunny_1k.obj`，约 1000 三角）在 512×512、128 SPP 路径追踪下，每像素数十条弹射 × 每弹射扫全网格，GPU 很快成为瓶颈。

**BVH** 在 CPU 端对三角形 **递归划分** 空间：内部节点存子 AABB，叶节点存最多 4 个三角形索引；GPU 端用 **栈式遍历**（深度 ≤ 64），仅访问与射线 AABB 相交的子树，期望复杂度 **$O(\log n)$**。

### 11.2 实现要点

| 组件 | 位置 | 说明 |
|------|------|------|
| `AABB` / `GpuBVHNode` | `include/cuda_types.h` | 32 字节对齐节点；叶：`primitiveCount>0`，`leftChild`=三角数组偏移 |
| CPU 建树 | `src/bvh_builder.cpp` | 最长轴 + 质心中位数划分；叶阈值 4；DFS 展平并重排 `bvhTriangles` |
| 场景上传 | `src/cuda_scene_builder.cpp` | `buildGpuSceneHost` 扁平化后调用 `buildBVH`，经 `cudaMalloc` 上传 |
| GPU 遍历 | `src/cuda_path_tracer.cu` | `intersectAABB`（slab + 预计算 `invDir`）；栈遍历；子节点按入射距离排序 |

球体 / 平面仍走原有几何分支；**仅三角形** 走 BVH。Whitted 与 Path（含 NEE / Guiding）共用 `intersectScene`。

### 11.3 测试场景与命令

测试场景：`testcases/scene_bvh_bunny.txt`（Cornell 风格小盒 + 缩放 Stanford Bunny）。

```bash
cd /data/PA1-2/code
cmake --build build -j
# 128 SPP 路径追踪（CUDA）
build/PA1-2 testcases/scene_bvh_bunny.txt output/bunny_path128.bmp path_nee 128 cuda
# 256 SPP
build/PA1-2 testcases/scene_bvh_bunny.txt output/bunny_path256.bmp path_nee 256 cuda
# Whitted 快速预览
build/PA1-2 testcases/scene_bvh_bunny.txt output/bunny_whitted.bmp whitted 1 cuda
```

### 11.4 性能对比（本机 CUDA，512×512，128 SPP，`path_nee`）

| 场景 | 分辨率 | 三角形数 | BVH 节点 | 128 SPP 耗时 | 说明 |
|------|--------|----------|----------|-------------|------|
| `scene_bvh_bunny.txt` | 512² | 1000 | 511 | **1.28 s** | 橙色 Bunny 可见，avg≈75；BVH 前 brute-force 约 6.4 s |
| `scene_bvh_bunny.txt` | 512² | 1000 | 511 | **2.54 s**（256 SPP） | 噪声更低 |
| `scene08_path.txt` | 1024² | 14 | 27 | 8.20 s | 三角极少，BVH 几乎无建树开销 |

**实测加速**：Bunny 512² / 128 SPP / `path_nee` / CUDA，BVH 启用后 **约 5×**（6.4 s → 1.3 s）；相对 $O(n)$  brute-force 全网格，期望 **约 5×–20×**（随场景与 GPU 变化）。

---

## 12. 几何说明：玻璃立方体贴地

`mesh/cube.obj` 顶点范围为 $[-1,1]^3$。场景中使用 `Translate -0.55 0.36 0.62` + `UniformScale 0.36`：边长 $0.72$，中心 $y=0.36$，底面 $y=0$，**恰好贴地**。此前中心偏低时立方体悬浮；修正后 Whitted 与路径追踪中地板接触阴影、焦散位置一致。

---

## 13. 已知局限

| 局限 | 说明 |
|------|------|
| 玻璃 firefly | 高 IOR + 纯折射路径在 64 SPP 时偶见亮斑；Fresnel 分裂后部分缓解；radiance clamp 仍保留 |
| SPP=64 方差 | 路径对比图仍有可见噪点；提高 SPP 会更干净 |
| 无 MIS | 光泽瓣与 NEE 未做多重重要性采样（`path_mis` 已实现，见 IMPLEMENTATION.md） |
| 纹理 / 法线仅 CPU | GPU 扁平化不传 BMP；纹理与法线贴图验收用 Whitted / CPU path（§5.8） |
| Path Guiding 仅 CUDA | 粗网格直方图；空 cell 与直接光主导区与 `path_nee` 差异小（§5.9） |
| 默认线性输出 | 主结果图为线性 radiance；gamma 对比见 §5.5 |

---

## 14. 参考代码与资料

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

# 菲涅尔 Schlick（bonus，CUDA 推荐，经典 Cornell 场景）
mkdir -p output/fresnel
build/PA1-2 testcases/scene_fresnel_cornell_compare.txt output/fresnel/fresnel_cornell_compare.bmp path_nee 256 gamma cuda
build/PA1-2 testcases/scene_fresnel_cornell_water_glass.txt output/fresnel/fresnel_cornell_water_glass.bmp path_nee 256 gamma cuda
build/PA1-2 testcases/scene_fresnel_cornell_grazing.txt output/fresnel/fresnel_cornell_grazing.bmp path_nee 256 gamma cuda
python3 scripts/fresnel_figures.py
python3 scripts/analyze_fresnel_regions.py output/fresnel/fresnel_cornell_compare.png output/fresnel/fresnel_cornell_water_glass.png output/fresnel/fresnel_cornell_grazing.png

# Ward 各向异性 BRDF（bonus，CUDA 推荐）
mkdir -p output/ward output/glossy
build/PA1-2 testcases/scene_ward_aniso_demo.txt output/ward/ward_aniso_demo.bmp path_nee 512 gamma cuda
build/PA1-2 testcases/scene_ward_aniso_rotate.txt output/ward/ward_aniso_rotate.bmp path_nee 512 gamma cuda
build/PA1-2 testcases/scene_ward_sweep.txt output/ward/ward_sweep.bmp path_nee 512 gamma cuda
build/PA1-2 testcases/scene_ward_metal_demo.txt output/ward/ward_metal_demo.bmp path_nee 512 gamma cuda
build/PA1-2 testcases/scene_glossy_sweep.txt output/glossy/glossy_sweep.bmp path_nee 512 gamma cuda
build/PA1-2 testcases/scene_gold_glossy.txt output/glossy/gold_glossy.bmp path_nee 512 gamma cuda
build/PA1-2 testcases/scene_gold_ward.txt output/ward/gold_ward.bmp path_nee 512 gamma cuda
build/PA1-2 testcases/scene_ward_aniso_showcase.txt output/ward/ward_aniso_showcase.bmp path_nee 512 gamma cuda
build/PA1-2 testcases/scene_ward_aniso_showcase_rotate.txt output/ward/ward_aniso_showcase_rotate.bmp path_nee 512 gamma cuda
python3 scripts/ward_showcase_analysis.py output/ward/ward_aniso_showcase.bmp
python3 scripts/ward_showcase_analysis.py output/ward/ward_aniso_showcase_rotate.bmp

# BVH 加速 Bunny 测试（CUDA）
build/PA1-2 testcases/scene_bvh_bunny.txt output/bunny_path128.bmp path_nee 128 cuda
build/PA1-2 testcases/scene_bvh_bunny.txt output/bunny_path256.bmp path_nee 256 cuda
build/PA1-2 testcases/scene_bvh_bunny.txt output/bunny_whitted.bmp whitted 1 cuda

# 回归
build/PA1-2 testcases/scene01_basic.txt output/scene01.bmp whitted
```
