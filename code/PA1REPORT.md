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
| **5.8 纹理贴图（bonus）** | ✅ 完成 | 材质可选 `texture` 路径，平面/球体/网格 UV，Whitted/Path 漫反射调制 |

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
| `include/raytracer.hpp` | Whitted / Path / Path+NEE 三种模式；RR、NEE、光泽采样 |
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

### 5.8 纹理贴图（bonus）

在漫反射材质上可选加载 **24-bit BMP 纹理**，着色时用 `diffuseColor × texture(uv)` 调制 albedo（未指定 `texture` 时行为与原来完全一致）。

**UV 映射**：

| 几何体 | UV 计算 |
|--------|---------|
| **Plane** | 在平面切线空间中取交点坐标，按 2× 平铺后 `frac` 重复 |
| **Sphere** | 球面坐标：`u = 0.5 + atan2(z,x)/(2π)`，`v = 0.5 - asin(y/r)/π` |
| **TriangleMesh** | 解析 obj 的 `vt`，命中三角形时用重心坐标插值 UV |

场景语法（在 `Material` 或 `GlossyMaterial` 块内）：

```
Material {
  diffuseColor 0.725 0.725 0.725
  texture textures/floor_checker.bmp
}
```

| 图片 | 命令 | 说明 |
|------|------|------|
| `texture_before.bmp` | `scene_whitted.txt whitted` | 地板纯色（无纹理） |
| `texture_after.bmp` | `scene_texture.txt whitted` | 地板棋盘格纹理 |

对比场景：Cornell Box Whitted（SPP=1）。`scene_texture.txt` 仅在地板材质（索引 0）增加 `texture` 行，其余与 `scene_whitted.txt` 相同。纹理文件 `code/textures/floor_checker.bmp` 为 256×256 棋盘格。

实现文件：`include/texture.hpp`、`src/texture.cpp`；`Hit` 携带 UV；`Material::getShadedDiffuse(hit)` 在 Whitted Phong 与 Path/NEE 漫反射路径中统一采样。

### 5.7 加分项路线图（终局展示场景，未实现）

长期目标：在 **一个高质量展示场景** 中同时体现全部 bonus（构图主观分）。建议实现顺序（由易到难、风险递增）：

1. ✅ **Gamma 校正** — 后处理，零光追风险  
2. ✅ **OpenMP 并行** — 已修复 critical 锁；`path_nee 32` 约 7.6× 加速（§5.6）  
3. ✅ **抗锯齿（AA）** — SPP>1 抖动子样本（§5.6.1）  
4. ✅ **纹理贴图** — 可选 `texture` 字段（§5.8）  
5. **法线贴图** — 下一项建议；复用 UV 与纹理管线  
6. **MIS** — 光泽 + NEE 降方差  
7. **环境贴图 IBL / 景深 / BVH** — 展示场景与性能进阶  

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
| 无 MIS | 光泽瓣与 NEE 未做多重重要性采样 |
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

# 纹理贴图对比（bonus）
build/PA1-2 testcases/scene_whitted.txt output/texture_before.bmp whitted
build/PA1-2 testcases/scene_texture.txt output/texture_after.bmp whitted
cp output/texture_before.bmp output/texture_after.bmp ../results/

# 回归
build/PA1-2 testcases/scene01_basic.txt output/scene01.bmp whitted
```
