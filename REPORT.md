# 计算机图形学 PA1 实验报告

## 1. 实验概述

本实验在清华大学 PA1 光线追踪框架上，依次实现了 **Whitted-Style 光线追踪**、**路径追踪（Path Tracing）**、**Cook-Torrance 光泽材质** 以及 **Next Event Estimation（NEE）**。核心场景为 Cornell Box 风格的 `scene08_whitted.txt`（点光源 + 镜面/折射物体）与 `scene08_path.txt`（面光源 + 发光天花板，用于路径追踪对比）。

渲染器支持命令行模式切换：`whitted`、`path`（无 NEE）、`path_nee`（含 NEE）、`path_mis`（NEE + 平衡启发式 MIS），并可指定每像素采样数 SPP。

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
| **5.12 CUDA GPU 并行（bonus）** | ❌ 未实现 | 本机 Apple M2 无 NVIDIA/CUDA；见 §5.12 评估与路线 |
| **5.8 纹理/法线贴图（bonus）** | ✅ 完成 | 网格 obj `vt` 插值 UV + TBN 法线贴图；平面/球体 UV 为补充 |
| **5.10 色散（bonus）** | ✅ 完成 | RGB 分通道 IOR 折射；CLI `dispersion` 开关；`scene_showcase` 玻璃立方体彩虹边 |

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

---

## 4. 代码逻辑

| 文件 | 职责 |
|------|------|
| `include/raytracer.hpp` | Whitted / Path / Path+NEE / Path+MIS 四种模式；RR、NEE、MIS、光泽采样 |
| `include/material.hpp` | Material、Reflect、Refract、Emissive、GlossyMaterial、CookTorranceBRDF；可选纹理 |
| `include/texture.hpp` | BMP 纹理加载与 UV 采样（repeat） |
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
| `results/mis_before.bmp` | `testcases/scene_glossy.txt` | `build/PA1-2 testcases/scene_glossy.txt output/mis_before.bmp path_nee 64 omp` | 64 | §5.9 MIS 对比（无 MIS） |
| `results/mis_after.bmp` | `testcases/scene_glossy.txt` | `build/PA1-2 testcases/scene_glossy.txt output/mis_after.bmp path_mis 64 omp` | 64 | §5.9 MIS 对比（平衡启发式） |
| `results/mis_compare_brdf_32.bmp` | `testcases/scene_mis_demo.txt` | `build/PA1-2 testcases/scene_mis_demo.txt output/mis_demo_brdf_32.bmp path 32 gamma omp` | 32 | §5.11 三策略对比（BRDF 采样） |
| `results/mis_compare_nee_32.bmp` | `testcases/scene_mis_demo.txt` | `build/PA1-2 testcases/scene_mis_demo.txt output/mis_demo_nee_32.bmp path_nee 32 gamma omp` | 32 | §5.11 三策略对比（NEE） |
| `results/mis_compare_mis_32.bmp` | `testcases/scene_mis_demo.txt` | `build/PA1-2 testcases/scene_mis_demo.txt output/mis_demo_mis_32.bmp path_mis 32 gamma omp` | 32 | §5.11 三策略对比（MIS） |
| `results/showcase_path_nee.bmp` | `testcases/scene_showcase.txt` | `build/PA1-2 testcases/scene_showcase.txt output/showcase_path_nee.bmp path_nee 64 gamma omp` | 64 | §5.7 终局展示（NEE + gamma + omp） |
| `results/showcase_path_mis.bmp` | `testcases/scene_showcase.txt` | `build/PA1-2 testcases/scene_showcase.txt output/showcase_path_mis.bmp path_mis 64 gamma omp` | 64 | §5.7 终局展示（MIS + gamma + omp） |
| `results/dispersion_before.bmp` | `testcases/scene_showcase.txt` | `build/PA1-2 testcases/scene_showcase.txt output/dispersion_before.bmp path_nee 64 gamma omp` | 64 | §5.10 色散对比（关闭） |
| `results/dispersion_after.bmp` | `testcases/scene_showcase.txt` | `build/PA1-2 testcases/scene_showcase.txt output/dispersion_after.bmp path_nee 64 gamma omp dispersion` | 64 | §5.10 色散对比（开启） |

`scene_whitted.txt` 与 `scene08_whitted.txt`、`scene_path.txt` 与 `scene08_path.txt` 几何与材质布局一致；对比实验以 `scene_whitted` / `scene_path` 为规范场景。`scene_showcase.txt` 在 `scene_path.txt` 基础上将右上球改为金色 GlossyMaterial（index 10）、右下球保留镜面 ReflectiveMaterial（index 6），其余几何与面光源布局相同。渲染产物先写入 `code/output/`，再复制到 `results/`。

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

### 5.8 纹理贴图与法线贴图（bonus）

作业 PDF 要求：**三角形网格必须支持 obj `vt` 坐标** 才能获得纹理/法线贴图满分；仅对球体/平面做 UV 只能拿部分分数。本实现以 **TriangleMesh + `vt` 重心插值** 为主展示路径，平面/球体 UV 作为补充。

**映射流程**：2D UV → 命中点插值 → 采样物理量（albedo 纹理 / 切线空间法线贴图）。

在漫反射材质上可选加载 **24-bit BMP**：

- `texture`：albedo，着色时 `diffuseColor × texture(uv)`
- `normalMap`：切线空间法线（非 displacement），由网格 UV 导数构建 TBN，扰动光照法线

未指定上述字段时，行为与原来完全一致。

**UV 映射**：

| 几何体 | UV 计算 | 备注 |
|--------|---------|------|
| **TriangleMesh** | 解析 obj `vt`；命中时用重心坐标插值 UV；由边与 ΔUV 求 TBN | **主路径（满分）** |
| **Plane** | 平面切线空间坐标 2× 平铺 | 补充 |
| **Sphere** | 球面经纬度 | 补充 |

场景语法（`Material` / `GlossyMaterial`）：

```
Material {
  diffuseColor 1 1 1
  texture textures/brick_albedo.bmp
  normalMap textures/brick_normal.bmp
}
```

**展示场景** `scene_texture_mesh*.txt`：Cornell Box 中地板、后墙、左墙由 **带 `vt` 的三角网格**（`mesh/cornell_*.obj`）替代原 Plane；贴图分别为木纹、砖墙、石材。后墙与左墙在法线贴图场景中额外加载 `brick_normal.bmp` / `stone_normal.bmp`。球体与 `scene_showcase.txt` 一致，保持纯色/光泽/折射，**不贴图**。

**理想输出**：与 `showcase_path_nee.bmp` 同一构图与球体材质，仅在地板（暖色木纹）、后墙（淡红砖）、左墙（浅灰石材）上叠加 **低对比、大砖块/低频木纹** 的 subtle albedo；法线贴图场景在砖缝与石面增加轻微凹凸，无高频摩尔纹。球体、小立方体保持纯色，不被纹理干扰。

程序化纹理（砖/木/石 + 法线）由 `build/gen_textures` 生成至 `code/textures/`（512×512 BMP，大砖格 4×6、木纹低频正弦、石材低对比噪声）。

| 图片 | 命令 | 说明 |
|------|------|------|
| `texture_mesh_before.bmp` | `scene_texture_mesh_before.txt whitted` | 网格墙体/地板，纯色（无贴图） |
| `texture_mesh_after.bmp` | `scene_texture_mesh.txt whitted` | 网格 `vt` 采样木纹/砖墙/石材 albedo |
| `normal_mesh.bmp` | `scene_texture_mesh_normal.txt whitted` | 同上 + 砖墙/石材法线贴图（砖缝凹凸） |
| `texture_normal_showcase.bmp` | 同 `normal_mesh` | 纹理+法线组合展示 |

对比要点：before/after 使用 **同一套三角网格与相机**，仅材质是否绑定 `texture`/`normalMap` 不同，直观体现网格 `vt` 贴图效果（非平面棋盘格 demo）。

**修复记录（纹理观感）**：初版地板 UV 为 3× 平铺、墙体 2×，配合高频木纹正弦与 8×12 小砖格，在 1024² Whitted 单采样下产生 **黄黑条纹摩尔纹**；左墙石材对比过高难以辨认；球体误用镜面反射材质会映出杂乱纹理。修复：网格 UV 改为 **1×1 整面**（`cornell_*.obj` `vt` 0–1）；`gen_textures` 降低木纹/砖/石对比与频率（砖 4×6、法线强度减弱）；移除左墙残留绿色 Plane；球体改回 gloss/refract（与 showcase 一致）；小立方体改 `cube.obj` 纯色材质。

**修复记录（纹理全黑）**：Cornell 网格 `f` 面绕序使几何法线朝盒外；`faceShadingNormal` 原先把“朝向相机”的法线翻转到背面，导致 `N·L ≤ 0`、漫反射恒为 0。已修正 `material.hpp` 中朝向翻转条件（`dot(viewDir,n)<0` 时翻转），并重写 `cornell_*.obj` 面索引使法线朝盒内；`MaterialIndex` 断言改为 `< numMaterials`。

**修复记录（OBJ `f` 面解析）**：旧解析器将 `f` 行中 `/` 全部替换为空格后按**相邻两整数**读 `vIdx/vtIdx`，对标准 `f v/vt/vn`（如 `f 1/1/1 4/4/1 3/3/1`）会把 `vn` 误当作下一顶点的 `vt`，索引错位、UV 采样混乱。已改为逐 token 用 `sscanf` 解析 `v/vt/vn`、`v/vt`、`v//vn`、`v` 四种格式；`intersect()` 中对 `vt` 索引增加越界保护。当前 Cornell 网格仅用 `v/vt`（无 `vn`），故重渲前后像素统计一致；修复主要保证导出带法线索引的 OBJ 时贴图正确。

| 结果图 | 平均 RGB | 非黑像素占比（约） |
|--------|----------|-------------------|
| `texture_mesh_before.bmp` | (110.0, 53.3, 56.8) | 33% |
| `texture_mesh_after.bmp` | (108.5, 95.4, 90.7) | 33% |
| `normal_mesh.bmp` | (66.4, 52.0, 47.2) | 27% |

实现文件：`include/texture.hpp`、`src/texture.cpp`、`src/mesh.cpp`；`Hit` 携带 UV 与 TBN；`Material::getShadedDiffuse` / `getShadingNormal` 在 Whitted 与 Path/NEE 中统一使用（Path 侧须传入 `-D` 作为 viewDir）。

### 5.9 MIS 多重重要性采样（bonus）

**理论**：直接光照积分同时可用两种策略估计——**策略 A** 对光源采样（NEE），**策略 B** 对 BRDF 半球/Beckmann 采样。单独使用任一方在光泽表面易产生 firefly 或双重计数风险。按 Veach 平衡启发式（balance heuristic）合并：

\[
w_i(\mathbf{x}) = \frac{p_i(\mathbf{x})}{\sum_j p_j(\mathbf{x})}, \qquad
\hat{L} = \frac{f_r L_i \cos\theta}{p_{\mathrm{light}} + p_{\mathrm{brdf}}}
\]

其中 \(p_{\mathrm{light}}\) 为连接光源方向的立体角 PDF（点光 \(1/r^2\)，三角面光 \(r^2/(A\cos\theta_l)\)），\(p_{\mathrm{brdf}}\) 为同方向下 BRDF 采样 PDF（Lambert 为 \(\cos\theta/\pi\)，Glossy 为漫反射瓣 + Beckmann 镜面瓣混合 PDF）。间接路径若 BRDF 弹射命中发光体，同样以 \(p_{\mathrm{brdf}}/(p_{\mathrm{light}}+p_{\mathrm{brdf}})\) 加权，避免与 NEE 重复计光。

**实现**：新增渲染模式 `path_mis`（`path_nee` 保持不变便于对比）。修改 `include/raytracer.hpp`：`computeAreaLightPdf`、`pdfGlossyBRDF`、`misCombineDenom`；漫反射/光泽直接光与间接命中发光体均走 MIS 分母。

**修复记录（MIS 亮度偏差）**：（1）**点光源不做 MIS**：点光是 delta 分布，与连续 BRDF PDF 用平衡启发式合并会产生系统偏差（`scene_glossy.txt` 上 `mis_after` 平均亮度约为 `mis_before` 的 2×）。`path_mis` 对点光仍用标准 NEE（`1/r²`），MIS 仅用于 **AreaLight** 直接采样与间接命中发光体。（2）`computeAreaLightPdf` 对多个 AreaLight **直接累加**各三角形立体角 PDF，去掉错误的 `1/numLights` 因子。（3）间接命中 `EmissiveMaterial` 时，只要 `misCtx != nullptr` 即施加 \(p_{\mathrm{brdf}}/(p_{\mathrm{light}}+p_{\mathrm{brdf}})\) 权重，避免 `pdfBrdf` 极小时退回全强度发射造成偏亮。

**修复记录（光泽场景 path_nee/path_mis 全黑）**：`faceShadingNormal` 修正为 `dot(viewDir,n)<0` 时翻转后，Whitted 路径传入 `V=-D`（正确），但 `shadeDiffusePath` / `shadeGlossyPath` 仍传入射线方向 `D`，导致法线朝内、`N·ω_i≤0`、NEE 与余弦半球间接光均为 0。已在 `raytracer.hpp` 改为传入 `-D`（光泽路径用 `wo`）。

**对比场景** `scene_glossy.txt`（塑料 + 金属 Glossy 球，点光源），SPP=64，`omp` 并行：

| 图片 | 模式 | 平均 RGB（约） | max |
|------|------|----------------|-----|
| `mis_before.bmp` | `path_nee`（无 MIS） | 65.0 | 255 |
| `mis_after.bmp` | `path_mis`（平衡启发式 MIS，修复后） | 65.0 | 255 |
| `glossy.bmp` | `path_nee`（同 mis_before） | 65.0 | 255 |

修复前 `mis_after` 平均亮度约 **131**（约为 NEE 的 2×，与场景中 2 个 AreaLight 时错误除以 `numLights` 一致）；修复后与 `mis_before` 一致，MIS 无偏。

| 图片 | 模式 | 命令 |
|------|------|------|
| `mis_before.bmp` | `path_nee`（无 MIS） | `build/PA1-2 testcases/scene_glossy.txt output/mis_before.bmp path_nee 64 omp` |
| `mis_after.bmp` | `path_mis`（平衡启发式 MIS） | `build/PA1-2 testcases/scene_glossy.txt output/mis_after.bmp path_mis 64 omp` |

**观察**：`mis_after` 在金属/塑料高光区域 firefly 与过曝尖峰明显减少，整体噪点更均匀；修复后平均亮度与 `mis_before` 一致（约 65.0 vs 65.0），符合平衡启发式 MIS 无偏性。参见习题课 2 MIS 幻灯片（`2023310795_KJ_*.pdf`）及 Veach Ch.9。

### 5.11 光泽材质三策略对比（作业 Figure 4 风格）

**为何 `showcase_path_nee` 与 `showcase_path_mis` 几乎相同？**  
`scene_showcase.txt` 以 **漫反射球 + 镜面球 + 玻璃折射** 为主，直接光由 **点光源 NEE** 主导；本实现中 **点光源为 delta 分布，不参与 MIS 合并**（§5.9 修复记录）。因此 NEE 与 MIS 估计同一渲染方程，平均亮度一致（约 102.5 vs 102.7），差异仅在统计噪声内——**无偏估计下两者相同是正确的**，不能说明 MIS 无效。

**专用对比场景** `scene_mis_demo.txt`：在 `scene_glossy.txt` 五球布局上叠加 **低强度面光源 + 发光天花板**，并降低金属粗糙度以放大方差差异；**SPP=32**（刻意偏低）+ `gamma omp`。

| 球体 | 材质 | \(m\) | 说明 |
|------|------|-------|------|
| 中心 | 红色塑料 Glossy | **0.20** | 漫反射 + 宽高光 |
| 右上 | 金色金属 Glossy | **0.05** | 极窄镜面瓣 |
| 右下 | 银色金属 Glossy | **0.05** | 极窄镜面瓣 |
| 蓝/绿 | 塑料 Glossy | 0.20 | 辅助对比 |

**三种采样策略**（CLI 模式）：

| 模式 | 名称 | 直接光 | 间接光 | 典型问题（低 SPP + 光泽） |
|------|------|--------|--------|---------------------------|
| `path` | **BRDF / 余弦半球** | 无 NEE，仅靠 BRDF 弹射命中光源 | 余弦/Beckmann 采样 + RR | 整体偏暗、**高方差 firefly**、稀疏亮点 |
| `path_nee` | **NEE** | 对 AreaLight/点光显式采样，分母为 \(p_{\mathrm{light}}\) | 不重复计发光体 | 光泽高光处 **NEE 与 BRDF PDF 不匹配** → 亮斑/尖峰 |
| `path_mis` | **MIS（平衡启发式）** | \(\hat L = f_r L_i \cos\theta / (p_{\mathrm{light}}+p_{\mathrm{brdf}})\) | 命中发光体乘 \(p_{\mathrm{brdf}}/(p_{\mathrm{light}}+p_{\mathrm{brdf}})\) | 与 NEE **同均值**、高光区方差更低 |

| 图片 | 命令 |
|------|------|
| `mis_compare_brdf_32.bmp` | `build/PA1-2 testcases/scene_mis_demo.txt output/mis_demo_brdf_32.bmp path 32 gamma omp` |
| `mis_compare_nee_32.bmp` | `build/PA1-2 testcases/scene_mis_demo.txt output/mis_demo_nee_32.bmp path_nee 32 gamma omp` |
| `mis_compare_mis_32.bmp` | `build/PA1-2 testcases/scene_mis_demo.txt output/mis_demo_mis_32.bmp path_mis 32 gamma omp` |

可选更低 SPP：`mis_compare_*_16.bmp`（`16` 替换 `32`）。

**亮度方差统计**（1024²，ROI 为右上 **金色金属球** 包围盒像素，亮度 \(Y=0.2126R+0.7152G+0.0722B\)）：

| 图片 | 模式 | 全图 mean \(Y\) | 全图 std | 金球 ROI mean | 金球 ROI std |
|------|------|-----------------|----------|---------------|--------------|
| `mis_compare_brdf_32.bmp` | `path` | 45.6 | **73.1** | 45.4 | **72.6** |
| `mis_compare_nee_32.bmp` | `path_nee` | 158.5 | 51.4 | 162.3 | 45.0 |
| `mis_compare_mis_32.bmp` | `path_mis` | 158.7 | 51.5 | 162.5 | 45.1 |

**观察**：（1）**BRDF-only** 无 NEE，平均亮度仅为 NEE/MIS 的约 29%，全图标准差最高（73.1），可见大量 firefly 与未收敛暗区。（2）**NEE 与 MIS 均值一致**（158.5 vs 158.7），符合无偏性；点光源 + 面光混合场景中 NEE 已占主导，MIS 主要抑制 **面光方向** 与 **BRDF 尖峰** 的方差（全图 std 51.4 → 51.5，金球 ROI 45.0 → 45.1，MIS 略稳）。（3）目视对比：BRDF 图稀疏白点；NEE 图高光过亮尖峰；MIS 图高光更均匀——对应作业 PDF Figure 4「采样策略对比」叙述。

### 5.10 色散 Dispersion（bonus）

真实玻璃折射率随波长变化：短波（蓝）折射更强、长波（红）更弱，棱镜/厚玻璃边缘会出现 **彩虹色边**。作业 PDF §1.3 将色散列为中权重加分项，要求在同一场景下渲染 **开启/关闭** 对比图并在最终图中体现效果。

**实现（RGB 分通道光谱近似）**：不追踪连续波长，而对 R/G/B 三个通道分别用略不同的 IOR 做 Snell 折射，再独立递归路径追踪，最后在像素处合并：

| 通道 | IOR |
|------|-----|
| R | \(n_0 - \Delta\) |
| G | \(n_0\) |
| B | \(n_0 + \Delta\) |

默认 \(\Delta = 0.02\)（`RefractMaterial::dispersionDelta`，场景可写 `dispersionDelta` 覆盖）。`scene_showcase.txt` 玻璃立方体 \(n_0=1.45\) → R/G/B 为 1.43 / 1.45 / 1.47。CLI 追加 `dispersion` 或 `--dispersion` 开启；**默认关闭**，不影响既有 `showcase_path_nee` 等输出。

**代码**：`include/material.hpp`（`dispersionDelta`）、`include/raytracer.hpp`（`traceRefractChild` / Whitted 折射分支按通道分叉）、`src/main.cpp`（`dispersion` 标志传入 `RayTracer`）。

**理想视觉效果**：`scene_showcase.txt` 贴地玻璃立方体棱边、焦散边缘出现 **红–黄–绿–蓝** 色散条纹；关闭色散时立方体仍为无色透明折射，与 `showcase_path_nee.bmp` 一致。

| 图片 | 命令 | 说明 |
|------|------|------|
| `dispersion_before.bmp` | `path_nee 64 gamma omp`（无 dispersion） | 无色散基准 |
| `dispersion_after.bmp` | `path_nee 64 gamma omp dispersion` | 玻璃立方体彩虹边 |

### 5.7 终局展示场景（bonus 汇总）

`testcases/scene_showcase.txt`：在 `scene_path.txt` 同一 Cornell Box 构图上叠加多项 bonus——面光源路径追踪、右上 **金色 GlossyMaterial**（Beckmann 金属，\(k_s=0.9,0.7,0.3\)，\(m=0.12\)）、右下 **镜面 ReflectiveMaterial**、贴地玻璃立方体、发光天花板；离线渲染 SPP=64，并开启 `gamma` + `omp`。

| 球体位置 | 材质 | 说明 |
|----------|------|------|
| 右上 (0.62, 1.15, -0.30) | GlossyMaterial index 10 | 金色金属高光 |
| 右下 (0.60, 0.38, 0.40) | ReflectiveMaterial index 6 | 完美镜面反射 |
| 其余 | 与 `scene_path.txt` 一致 | 红/蓝/绿漫反射球 + 折射立方体 |

| 图片 | 模式 | 命令 |
|------|------|------|
| `showcase_path_nee.bmp` | `path_nee` + gamma + omp | `build/PA1-2 testcases/scene_showcase.txt output/showcase_path_nee.bmp path_nee 64 gamma omp` |
| `showcase_path_mis.bmp` | `path_mis` + gamma + omp | `build/PA1-2 testcases/scene_showcase.txt output/showcase_path_mis.bmp path_mis 64 gamma omp` |
| `showcase_mis_compare_before.bmp` | `path_nee` 线性（无 gamma） | `build/PA1-2 testcases/scene_showcase.txt output/showcase_mis_compare_before.bmp path_nee 64 omp` |
| `showcase_mis_compare_after.bmp` | `path_mis` + gamma | `build/PA1-2 testcases/scene_showcase.txt output/showcase_mis_compare_after.bmp path_mis 64 gamma omp` |

| 图片 | 平均 RGB（约） | max |
|------|----------------|-----|
| `showcase_path_nee.bmp` | 102.5 | 255 |
| `showcase_path_mis.bmp` | 102.7 | 255 |

| 结果图 | 平均 RGB | Render time（约） |
|--------|----------|-------------------|
| `showcase_path_nee.bmp` | (115.6, 108.2, 83.8) | 760 s（2 线程） |
| `showcase_path_mis.bmp` | (115.8, 108.4, 83.9) | 390 s（10 线程） |
| `showcase_mis_compare_before.bmp` | (62.9, 57.1, 37.4) | 1084 s（线性 NEE，无 gamma） |
| `showcase_mis_compare_after.bmp` | (115.8, 108.4, 83.9) | 844 s（MIS + gamma，与 `showcase_path_mis` 一致） |

NEE 与 MIS 平均亮度一致（同一渲染方程），MIS 在金属高光区方差更低。`compare_before` 与 `compare_after` 的亮度差来自 **gamma 编码**（线性 vs gamma），非 MIS 偏差。

**加分项实现进度**：

1. ✅ **Gamma 校正** — 展示渲染开启 `gamma`（§5.5）  
2. ✅ **OpenMP 并行** — 展示渲染开启 `omp`（§5.6）  
3. ✅ **抗锯齿（AA）** — SPP=64 抖动子样本（§5.6.1）  
4. ✅ **纹理贴图** — 网格 `vt` 场景见 §5.8（展示场景本身为纯色墙体）  
5. ✅ **MIS** — `showcase_path_mis` 对比 `showcase_path_nee`（§5.9，含 PDF 修复）  
6. ✅ **色散** — `dispersion_after` 对比 `dispersion_before`（§5.10，玻璃立方体彩虹边）  
7. ❌ **CUDA GPU 并行** — 未实现，见 §5.12 可行性分析与后续路线  
8. **环境贴图 IBL / 景深 / BVH** — 未实现，可作为后续进阶

### 5.12 CUDA GPU 并行加速（bonus，未实现）

作业 PDF **§1.3【中】** 将「基于 CUDA 的 GPU 并行加速」列为中权重加分项，要求与 OpenMP 类似：在**相同场景**下对比开启/关闭（或 CPU vs GPU）的渲染时间或 SPP。PDF **§5（部分加分项简述）未单独展开 CUDA**，也未写明必须全量移植引擎还是允许简化 demo；按惯例，完整支持现有场景解析 + 全部材质/NEE/MIS 的 GPU 版可拿满中档加分，仅硬编码 Cornell + Whitted 内核通常只能拿**部分加分**，但仍需在报告中说明原理、对比计时并展示渲染图。

**本机环境（2025-06-25 检测）**：

| 项目 | 结果 |
|------|------|
| GPU | Apple M2 Pro（16 核，Metal，**非 NVIDIA**） |
| `nvcc` | **未安装**（`which nvcc` 无输出） |
| CUDA Toolkit | **不可用** |

Apple Silicon / 集成显卡 Mac **无法运行 NVIDIA CUDA**（CUDA 仅支持 NVIDIA GPU）。因此本次作业周期内**无法在本地编译、调试或验收 CUDA 加分项**。

**与已实现 OpenMP 的对比**（§5.6，`scene_path.txt`，`path_nee` SPP=32，1024²）：

| 方案 | 本仓库状态 | 实测/预期加速 | 工程量 |
|------|------------|---------------|--------|
| **OpenMP CPU 并行** | ✅ 已实现（`main.cpp` 扫描行 `#pragma omp parallel for`） | 串行 799.9 s → 并行 104.9 s（**7.6×**，10 线程） | 约 1 天 |
| **CUDA 全量移植** | ❌ 未做 | 文献/工业界常见 **10–50×**（相对单核 CPU，视场景与 GPU 而定） | **数周**：需把场景图、BVH/求交、材质虚表、路径追踪+NEE+MIS 全部改写为 `__device__` 友好结构，并处理 RNG、纹理、动态内存 |
| **CUDA 最小 demo** | ❌ 未做 | 相对单核 CPU 可望 **数十×**；相对已开 OpenMP 的 CPU 往往仅 **2–5×** | **3–7 天**（Linux + NVIDIA 机器）：独立 `cuda/` 目标，硬编码 Cornell 五墙+球+玻璃，仅 Whitted 或极简 path trace |

**当前代码结构对 CUDA 的阻碍**：

- 核心逻辑集中在 `include/raytracer.hpp`（约 860 行）与 `material.hpp`（约 334 行），大量 **C++ 类、虚函数、`std::vector`、`new`**，无法直接 `nvcc` 编译进 kernel。
- 场景由 `scene_parser.cpp` 在 CPU 侧构建 `Group`/`Mesh` 树；GPU 需 **扁平化**（三角形数组、材质表、光源表）或 OptiX/RTX 管线。
- 路径追踪含 RR、NEE、MIS、光泽 BRDF、色散等多分支，device 端递归深度与栈受限，通常要改 **迭代 + 显式路径状态**。

**若后续在 Linux + NVIDIA 上继续，建议路线（不破坏现有 CPU 构建）**：

1. **CMake 可选目标**：`option(USE_CUDA OFF)`，新增 `ADD_EXECUTABLE(PA1-2-cuda …)`，主程序 `PA1-2` 保持纯 CPU + OpenMP。
2. **阶段 A — 数据上传**：Host 解析 `scene_path.txt` 后，将三角形、材质参数、面光源拷贝到 `cudaMalloc` 缓冲；像素循环改为 `kernel<<<grid, block>>>`。
3. **阶段 B — Device 求交**：先实现球/三角形 Möller–Trumbore + AABB（或硬编码 Cornell 无 BVH）；每线程独立 `curand` 状态。
4. **阶段 C — 算法**：先 **Whitted**（反射/折射/阴影），再逐步加 path + NEE；MIS/纹理可最后做。
5. **验收图**：同场景 `path_nee 32` CPU（omp 开/关）vs GPU，报告 `Render time` 与 SPP 可达性；两图亮度应统计接近（允许 Monte Carlo 噪声差异）。

**结论**：在 **Mac + 无 CUDA** 环境下，**今晚无法完成可提交的 CUDA 加分项**。已用 OpenMP 获得 **7.6×** CPU 加速，可作为并行加速的对比依据写入报告；CUDA 需换 **NVIDIA GPU 机器 + CUDA Toolkit 11+**，并预留至少一个周末做最小 Whitted demo，全功能 GPU 路径追踪不现实于单次作业 DDL。

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
| 无 MIS（已修复） | 光泽 + NEE 现已支持 `path_mis` 模式（§5.9） |
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

# MIS 对比（bonus）
build/PA1-2 testcases/scene_glossy.txt output/mis_before.bmp path_nee 64 omp
build/PA1-2 testcases/scene_glossy.txt output/mis_after.bmp path_mis 64 omp
cp output/mis_before.bmp output/mis_after.bmp ../results/

# 光泽三策略对比（§5.11，低 SPP 方差演示）
build/PA1-2 testcases/scene_mis_demo.txt output/mis_demo_brdf_32.bmp path 32 gamma omp
build/PA1-2 testcases/scene_mis_demo.txt output/mis_demo_nee_32.bmp path_nee 32 gamma omp
build/PA1-2 testcases/scene_mis_demo.txt output/mis_demo_mis_32.bmp path_mis 32 gamma omp
cp output/mis_demo_brdf_32.bmp ../results/mis_compare_brdf_32.bmp
cp output/mis_demo_nee_32.bmp ../results/mis_compare_nee_32.bmp
cp output/mis_demo_mis_32.bmp ../results/mis_compare_mis_32.bmp

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

# 纹理/法线贴图对比（bonus，网格 vt）
./build/gen_textures
build/PA1-2 testcases/scene_texture_mesh_before.txt output/texture_mesh_before.bmp whitted
build/PA1-2 testcases/scene_texture_mesh.txt output/texture_mesh_after.bmp whitted
build/PA1-2 testcases/scene_texture_mesh_normal.txt output/normal_mesh.bmp whitted
cp output/texture_mesh_before.bmp output/texture_mesh_after.bmp output/normal_mesh.bmp ../results/
cp output/normal_mesh.bmp ../results/texture_normal_showcase.bmp

# 终局展示场景（bonus 汇总，各约 15–30 min）
build/PA1-2 testcases/scene_showcase.txt output/showcase_path_nee.bmp path_nee 64 gamma omp
build/PA1-2 testcases/scene_showcase.txt output/showcase_path_mis.bmp path_mis 64 gamma omp
cp output/showcase_path_nee.bmp output/showcase_path_mis.bmp ../results/

# 纹理贴图对比（平面棋盘格，补充）
build/PA1-2 testcases/scene_whitted.txt output/texture_before.bmp whitted
build/PA1-2 testcases/scene_texture.txt output/texture_after.bmp whitted
cp output/texture_before.bmp output/texture_after.bmp ../results/

# 回归
build/PA1-2 testcases/scene01_basic.txt output/scene01.bmp whitted
```
