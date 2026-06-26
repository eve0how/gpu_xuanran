# PA1 基础功能实现说明（验收答辩用）

> 工作目录：`code/`  
> 本文档面向 **今日验收答辩**，按「原理 → 文件/函数 → 关键逻辑 → 常见坑」组织；§1–§6 为 **基础要求**，§7 展开 **Gamma 校正** 与 **OpenMP 并行** 两个加分项，其余加分功能见 §7.3 简表。

---

## 目录

1. [Whitted-Style 光线追踪（基础要求 1）](#1-whitted-style-光线追踪基础要求-1)
2. [路径追踪（基础要求 2）](#2-路径追踪基础要求-2)
3. [Next Event Estimation — NEE（§4.3）](#3-next-event-estimation--nee-43)
4. [Cook-Torrance 光泽材质（§4.2）](#4-cook-torrance-光泽材质-42)
5. [场景文件与命令行](#5-场景文件与命令行)
6. [关键文件对照表](#6-关键文件对照表)
7. [加分项](#7-加分项)
   - [7.1 Gamma 校正](#71-gamma-校正)
   - [7.2 CPU 加速（OpenMP）](#72-cpu-加速openmp)
   - [7.3 其他加分项简表](#73-其他加分项简表)
   - [7.4 GPU/CUDA 并行加速](#74-gpucuda-并行加速)
   - [7.5 MIS（多重重要性采样）](#75-mis多重重要性采样)

---

## 1. Whitted-Style 光线追踪（基础要求 1）

### 1.1 原理是什么

Whitted-Style 是一种 **递归确定性** 光线追踪：

- **漫反射（Diffuse）**：在命中点用 **Phong 模型** 计算直接光照；对每个光源发射 **阴影射线（Shadow Ray）**，若中间无遮挡则累加光照贡献。
- **完美镜面（Reflect）**：按反射定律计算反射方向，递归追踪子光线，结果乘以 `reflectColor`。
- **折射（Refract）**：按 **Snell 定律** 计算折射方向（全内反射时退化为镜面反射），递归追踪子光线，结果乘以 `refractColor`。
- **发光体（Emissive）**：直接返回 `emission`，不再弹射。

Whitted 在漫反射面 **终止** 递归，不做间接光积分；玻璃在 Whitted 阴影检测中视为 **透明**（不遮挡点光源）。

### 1.2 在哪些文件/函数实现

| 组件 | 位置 |
|------|------|
| 主入口 | `include/raytracer.hpp` — `RayTracer::trace()` → `castRayWhitted()` |
| 递归追踪 | `castRayWhitted(ray, depth, tmin)` |
| 漫反射着色 | `shadeDiffuse()` |
| 镜面反射 | `castRayWhitted` 内 `MaterialType::REFLECT` 分支 |
| 折射 | `computeRefractDirection()` + `castRayWhitted` 内 `REFRACT` 分支 |
| 阴影检测 | `isInShadow(p, N, light)` |
| 材质类 | `include/material.hpp` — `Material`, `ReflectMaterial`, `RefractMaterial`, `EmissiveMaterial` |
| 场景解析 | `src/scene_parser.cpp` — `ReflectiveMaterial` / `RefractiveMaterial` |

### 1.3 关键代码逻辑

**入口与模式分发**（`raytracer.hpp`）：

```cpp
Vector3f trace(const Ray &ray) const {
    if (mode == RenderMode::WHITTED) {
        return castRayWhitted(ray, 0, RAY_EPSILON);
    }
    return castRayPath(ray, 0, Vector3f(1,1,1), true);
}
```

**`castRayWhitted` 伪代码**：

```
castRayWhitted(ray, depth, tmin):
    if depth > MAX_TRACE_DEPTH: return background
    if no hit: return background
    if EMISSIVE: return emission
    if opaque back face: return black
    if GLOSSY: shadeGlossyWhitted (Whitted 下仍走 Phong 直接光)
    if DIFFUSE: shadeDiffuse (Phong + shadow ray)
    if REFLECT:
        R = reflect(D, faceNormal)
        origin = offsetAlongNormal(hitPoint, N, ORIGIN_OFFSET)
        return reflectColor * castRayWhitted(Ray(origin, R), depth+1)
    if REFRACT:
        T = computeRefractDirection(D, geomN, ior)
        origin = offsetAlongRay(hitPoint, T, REFRACT_ORIGIN_OFFSET)  // 沿折射方向偏移
        return refractColor * castRayWhitted(Ray(origin, T), depth+1, REFRACT_RAY_TMIN)
```

**Snell 折射**（`computeRefractDirection`，封闭网格、几何法线朝外）：

```
dot(D, geomN) < 0  →  进入介质，η = 1/ior
dot(D, geomN) > 0  →  离开介质，η = ior/1，法线取反

k = 1 - η²(1 - cos²θ)
k < 0  →  全内反射，返回镜面反射方向
否则   →  T = normalize(ηD + (η cosθ - √k) N)
```

对应代码（节选）：

```97:114:code/include/raytracer.hpp
    static Vector3f computeRefractDirection(const Vector3f &D, const Vector3f &geomN, float ior) {
        float etai = 1.0f;
        float etat = ior;
        Vector3f n = geomN;
        float cosTheta = Vector3f::dot(D, n);
        if (cosTheta > 0.0f) {
            etai = ior;
            etat = 1.0f;
            n = -n;
            cosTheta = -cosTheta;
        }
        float eta = etai / etat;
        float k = 1.0f - eta * eta * (1.0f - cosTheta * cosTheta);
        if (k < 0.0f) {
            return (D - 2.0f * Vector3f::dot(D, n) * n).normalized();
        }
        return (eta * D + (eta * cosTheta - sqrtf(k)) * n).normalized();
    }
```

**阴影射线**（`isInShadow`）：

```
L = 光源方向（由 light->getIllumination 给出）
shadowOrigin = p + N_shadow * SHADOW_EPSILON   // 沿与 L 同侧的法线偏移，防自相交
maxT = light->getDistance(p)                   // 点光源为到光源距离
若 intersect 且 t < maxT - ε:
    若 blocker 为 REFRACT → 不遮挡（玻璃透明）
    否则 → 在阴影中
```

**材质类型**（`material.hpp`）：

- `ReflectMaterial`：仅 `reflectColor`，漫反射色为 0。
- `RefractMaterial`：`refractColor` + `refractIndex`（IOR）。
- `EmissiveMaterial`：`emission`，Whitted 下直接返回。

### 1.4 常见坑与修复

| 问题 | 原因 | 修复 |
|------|------|------|
| **镜面球/地板变黑（黑镜）** | 反射/折射子光线起点用 **法线偏移** `p + N*ε`，但折射后光线在 **介质内部**，沿法线偏移可能仍落在同一三角形内，立刻再次自相交 | **折射** 改用 `offsetAlongRay(p, newDir, ε)` 沿 **折射方向** 推出；反射仍用 `offsetAlongNormal` |
| 折射全黑/闪烁 | 进入/离开介质时 η 取反错误 | 用 **几何法线** `geomN`（不 flip 成 face normal）配合 `computeRefractDirection` 的进出判定 |
| 阴影 acne | 阴影射线从表面出发无偏移 | `p + shadowN * SHADOW_EPSILON`，且 `shadowN` 与 `L` 同侧 |
| 玻璃挡光 | Whitted 阴影应允许光穿过玻璃 | `isInShadow` 中 `blocker == REFRACT` 时返回 **false**（不遮挡） |
| 背面漏光 | 从物体内部看到背面 | `isOpaqueBackFace`：非折射/非发光材质且 `dot(D,N)>0` 时返回黑色 |

**偏移函数对比**（核心修复点）：

```81:87:code/include/raytracer.hpp
    static Vector3f offsetAlongRay(const Vector3f &p, const Vector3f &dir, float eps) {
        return p + dir.normalized() * eps;
    }

    static Vector3f offsetAlongNormal(const Vector3f &p, const Vector3f &n, float eps) {
        return p + n.normalized() * eps;
    }
```

- **反射**：`offsetAlongNormal(hitPoint, N, ORIGIN_OFFSET)` — 推出到反射侧半空间。
- **折射**：`offsetAlongRay(hitPoint, newDir, REFRACT_ORIGIN_OFFSET)` — 沿折射光线进入玻璃/空气，避免黑镜。

---

## 2. 路径追踪（基础要求 2）

### 2.1 原理是什么

路径追踪求解 **渲染方程**（蒙特卡洛估计）：

$$
L_o(x, \omega_o) = L_e(x, \omega_o) + \int_{\Omega} f_r(x, \omega_i, \omega_o)\, L_i(x, \omega_i)\, \cos\theta_i \,\mathrm{d}\omega_i
$$

实现要点：

- **余弦加权半球采样**（Lambertian）：pdf = cosθ/π，贡献简化为 `albedo × Li`（pdf 与 cos 相消）。
- **Throughput（路径权重）**：沿路径累积 `reflectColor` / `refractColor`；命中发光体返回 `throughput × emission`。
- **俄罗斯轮盘赌（RR）**：深度 ≥ `RR_START_DEPTH(8)` 时，以 `max(0.15, luminance(throughput))` 概率继续，存活时除以该概率保持 **无偏**。
- **深度限制**：`MAX_TRACE_DEPTH = 12`。
- **发光材质 + 面光源场景**：`scene_path.txt` 中天花板为 `EmissiveMaterial`，并配 `AreaLight` 三角形（几何与光源一致）。

### 2.2 在哪些文件/函数实现

| 组件 | 位置 |
|------|------|
| 路径主函数 | `castRayPath(ray, depth, throughput, countEmissive)` |
| 漫反射路径 | `shadeDiffusePath()` |
| 镜面/折射路径 | `traceReflectChild()`, `traceRefractChild()` |
| 半球采样 | `sampleCosineHemisphere()` |
| RR | `survivalProbability()`, `shadeDiffusePath` 内 depth ≥ 8 分支 |
| 发光体 | `EmissiveMaterial`，`castRayPath` 中 `countEmissive` 控制是否计光 |
| 主循环 / SPP | `src/main.cpp` |

### 2.3 关键代码逻辑

**`castRayPath` 伪代码**：

```
castRayPath(ray, depth, throughput, countEmissive):
    if depth > MAX: return 0
    if no hit: return 0
    if EMISSIVE:
        if !countEmissive: return 0    // NEE 模式下间接路径不计发光体（见 §3）
        return clamp(throughput * emission)
    if opaque back face: return 0
    if REFLECT: return traceReflectChild(...)
    if REFRACT: return traceRefractChild(...)
    if GLOSSY: return shadeGlossyPath(...)
    return shadeDiffusePath(...)
```

**漫反射间接光**（`shadeDiffusePath`）：

```
direct = 0
if useNEE(): direct = sampleDirectEmissive + sampleDirectPointLights   // §3

wi, pdf = sampleCosineHemisphere(N)
if depth >= RR_START_DEPTH:
    rrProb = max(0.15, luminance(throughput * albedo))
    if random > rrProb: skip indirect
indirect = albedo * Li / rrProb
return clamp(direct + indirect)
```

**余弦加权采样**（pdf = cosθ/π）：

```139:153:code/include/raytracer.hpp
    Vector3f sampleCosineHemisphere(const Vector3f &normal, float &pdf) const {
        float u1 = uniform();
        float u2 = uniform();
        float phi = 2.0f * M_PI_F * u1;
        float cosTheta = sqrtf(u2);
        float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
        // ... 构建正交基 ...
        pdf = cosTheta / M_PI_F;
        return tangent * (cosf(phi) * sinTheta) + bitangent * (sinf(phi) * sinTheta) + normal * cosTheta;
    }
```

**main.cpp SPP 与抖动（抗锯齿）**：

```132:149:code/src/main.cpp
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
                unsigned int seed = 1u + x + 7919u * y + 104729u * s;
                RayTracer tracer(scene, mode, seed);
                accum += tracer.trace(camera->generateRay(Vector2f(jx, jy)));
            }
            dImg.SetPixel(x, y, accum * (1.0f / spp));
        }
    }
```

- 路径模式默认 **SPP=64**；Whitted 默认 SPP=1。
- 每像素独立 `RayTracer` + 确定性 seed，保证可复现。
- SPP>1 时子样本随机抖动，等价盒式滤波抗锯齿。

### 2.4 常见坑与修复

| 问题 | 修复 |
|------|------|
| 路径爆炸 / firefly | `clampRadiance`：亮度超过 30 时按比例缩放 |
| 无限递归 | `MAX_TRACE_DEPTH` + RR |
| 路径模式背景过亮 | 未命中返回 **0**（非 background），因路径追踪不采样环境贴图 |
| 镜面/玻璃在路径中行为 | 仍确定性反射/折射（`traceReflectChild` / `traceRefractChild`），throughput 乘材质色 |
| 与 NEE / MIS 配合 | `path_nee` 靠 NEE 算直接光；`path_mis` 在发光命中处加 MIS 权（§7.5） |

---

## 3. Next Event Estimation — NEE（§4.3）

### 3.1 原理是什么

NEE 在每次 **漫反射/光泽** 命中点，**显式向光源采样** 直接光照，而不是仅靠随机弹射碰巧命中小面积发光体。

**渲染方程直接光项**（单光源）：

$$
L_o \mathrel{+}= f_r(x,\omega_i,\omega_o)\, L_e(x,\omega_i)\, \cos\theta_o
$$

**点光源**：方向固定，需 shadow ray；贡献含 **距离平方反比** `1/r²`（由 `getIllumination` 与 BRDF 积分得到）。

**三角形面光源**：

1. 在三角形上 **均匀采样** 一点，面积 pdf：\(p_A = 1/A\)。
2. 转换到 **立体角 pdf**：
   \[
   p_\omega = p_A \cdot \frac{r^2}{\cos\theta_l}
   \]
   其中 \(r = \|x - x_l\|\)，\(\theta_l\) 为光源法线与 \(-\omega_i\) 夹角。
3. Lambertian 贡献（代码中已化简）：
   \[
   L = \frac{\text{albedo}}{\pi} \cdot L_e \cdot \cos\theta_o \cdot \frac{\cos\theta_l}{p_\omega}
     = \text{albedo} \cdot L_e \cdot \cos\theta_o \cdot \cos\theta_l \cdot \frac{A}{\pi r^2}
   \]

**实现架构说明**：代码中 **没有** 单独的 `castRayPathNEE` 函数；`RenderMode::PATH_TRACE_NEE` 时 `useNEE()` 为真，在 `shadeDiffusePath` / `shadeGlossyPath` 内累加直接光项。`PATH_TRACE` 与 `PATH_TRACE_NEE` 共用 `castRayPath`。

**path vs path_nee 对比目的**（§4.1）：同场景 `scene_path.txt`、同 SPP，仅差 NEE。两者 **期望相同**，但无 NEE 时直接光只能靠低概率间接命中发光体，方差极大、收敛慢、画面偏暗噪重；NEE 将直接光变为 O(1) 估计，软阴影与整体亮度显著改善。

> 说明：`path_nee` 用 NEE 估计直接光；`path_mis` 在 NEE 与 BRDF 采样之间加 MIS 权重（详见 [§7.5](#75-mis多重重要性采样)）。

### 3.2 在哪些文件/函数实现

| 组件 | 位置 |
|------|------|
| 模式开关 | `useNEE()` ← `RenderMode::PATH_TRACE_NEE` |
| 面光源 NEE（Lambert） | `sampleDirectEmissive()` → `sampleOneAreaLightDiffuse()` |
| 面光源 NEE（Glossy） | `sampleDirectEmissiveBRDF()` → `sampleOneAreaLightGlossy()` |
| 点光源 NEE | `sampleDirectPointLights()` / `sampleDirectPointLightsBRDF()` |
| 三角形采样 | `sampleTriangle()` |
| 可见性 | `isSegmentOccluded(from, to, N)` |
| 避免双重计光 | `castRayPath(..., countEmissive=!useNEE())` |

### 3.3 关键代码逻辑

**Lambert 面光源 NEE**（注释与实现一致）：

```209:244:code/include/raytracer.hpp
    // L = Le * (albedo/pi) * cos(theta_o) / pdf_omega, pdf_omega = (1/A) * r^2 / cos(theta_l)
    bool sampleOneAreaLightDiffuse(...) {
        lightPoint = sampleTriangle(v0, v1, v2);
        wi = normalize(lightPoint - hitPoint);
        // cosO, cosL 检查，isSegmentOccluded
        contrib += albedo * emission * cosO * cosL * lightArea / (M_PI_F * dist2);
    }
```

**NEE 与间接弹射的配合**（当前源码）：

```718:732:code/include/raytracer.hpp
            if (traceIndirect) {
                Vector3f origin = offsetAlongNormal(hitPoint, N, ORIGIN_OFFSET);
                MisIndirectCtx misCtx;
                const MisIndirectCtx *misPtr = nullptr;
                if (useMIS()) {
                    misCtx.pdfBrdf = pdf;
                    misCtx.wi = wi;
                    misCtx.shadingPoint = hitPoint;
                    misCtx.N = N;
                    misCtx.wo = -D;
                    misCtx.glossyMat = nullptr;
                    misPtr = &misCtx;
                }
                Vector3f Li = castRayPath(Ray(origin, wi), depth + 1, throughput, true, misPtr,
                                          dispChannel);
```

- NEE 开启：**直接光** 由 `sampleDirectEmissive` 等估计。
- **间接** 子路径传入 `countEmissive=true`（允许命中发光体）；`path_mis` 时额外传入 `MisIndirectCtx`，在发光命中处用 MIS 权重（§7.5）。
- 镜面/折射子路径仍 `countEmissive=true`（`traceReflectChild` / `traceRefractChild`）。

**`isSegmentOccluded`（NEE 专用段可见性）**：

```
shadowOrigin = from + shadowN * SHADOW_EPSILON
segDist = |to - shadowOrigin|
若最近交点 t >= segDist - ε → 未遮挡（含精确命中光源三角形）
若 blocker == EMISSIVE → 未遮挡（命中光源本身）
否则 → 遮挡（含 REFRACT 玻璃 — NEE 下玻璃仍挡光）
```

与 Whitted `isInShadow` 的区别：**NEE 下玻璃不透明**；**发光体不视为 blocker**。

### 3.4 常见坑与修复

| 问题 | 修复 |
|------|------|
| NEE 后画面过亮 | `path_mis` 用 MIS 合并；`path_nee` 下若间接也命中同一面光源，理论上可能轻微双重计光，高 SPP 下通常可接受 |
| 面光源 NEE 全黑 | 检查 cosO、cosL>0；光源三角形法线方向 |
| Shadow 自相交 | 法线侧向 `SHADOW_EPSILON` 偏移 |
| 命中光源三角形被判遮挡 | `t >= segDist - ε` 视为可见；`EMISSIVE` blocker 放行 |
| 无 NEE 极暗 | 正常现象：小面积发光体间接命中概率极低，用于对比实验 |

---

## 4. Cook-Torrance 光泽材质（§4.2）

### 4.1 原理是什么

**Cook-Torrance 微表面 BRDF**（课程 PPT 53–56 页）：

$$
f_r = k_d \frac{\rho_d}{\pi} + k_s \frac{D \cdot G \cdot F}{4\,(n\cdot\omega_i)(n\cdot\omega_o)}
$$

| 符号 | 含义 | 实现 |
|------|------|------|
| \(D\) | Beckmann 法线分布（粗糙度 \(m\)） | `CookTorranceBRDF::beckmannD` |
| \(G\) | Cook-Torrance 几何遮蔽 | `cookTorranceG` = \(G_1(\omega_o) G_1(\omega_i)\) |
| \(F\) | Schlick 菲涅尔 | `schlickF(F0, cosθ)` |
| \(F_0\) | 法向入射反射率 | 电介质 0.04；金属取 albedo/ks |

**路径追踪采样策略**（`shadeGlossyPath`）：

- 按 \(k_d\)、\(k_s\) **亮度能量比** 选择 **漫反射瓣** 或 **镜面瓣**（金属 \(k_d \approx 0\) 时 100% 镜面）。
- 镜面瓣：Beckmann 采样 **半向量 h**，再 `wi = reflect(wo, h)`，pdf 含 Jacobian \(1/(4\,\omega_o\cdot h)\)。
- 漫反射瓣：余弦加权半球，pdf 乘 \((1-p_\text{spec})\)。

Whitted 模式下光泽物体走 `shadeGlossyWhitted`：对每个光源 Phong 式 **直接光** + shadow ray（与 Diffuse 类似，但 `GlossyMaterial::Shade` 用完整 BRDF）。

### 4.2 在哪些文件/函数实现

| 组件 | 位置 |
|------|------|
| BRDF 数学 | `include/material.hpp` — `CookTorranceBRDF`, `GlossyMaterial` |
| 路径光泽 | `shadeGlossyPath()` |
| Whitted 光泽 | `shadeGlossyWhitted()` |
| Beckmann 半向量采样 | `sampleBeckmannHalfVector()` |
| 场景 | `testcases/scene_glossy.txt` |

### 4.3 关键代码逻辑

**Beckmann D**：

```28:34:code/include/material.hpp
    static float beckmannD(const Vector3f &n, const Vector3f &h, float roughness) {
        float cosTheta = std::max(0.001f, Vector3f::dot(n, h));
        float cosTheta2 = cosTheta * cosTheta;
        float tanTheta2 = (1.0f - cosTheta2) / cosTheta2;
        float m2 = roughness * roughness;
        return expf(-tanTheta2 / m2) / (CT_PI * m2 * cosTheta2 * cosTheta2);
    }
```

**Glossy 路径采样分支**（节选）：

```540:563:code/include/raytracer.hpp
        float specProb = isMetal ? 1.0f : ksLum / std::max(1e-4f, kdLum + ksLum);
        bool specularLobe = isMetal || uniform() < specProb;
        if (specularLobe) {
            h = sampleBeckmannHalfVector(N, wo, roughness, pdfH);
            wi = (2 * dot(wo,h)*h - wo).normalized();
            pdf = pdfH / (4 * dot(wo,h) * specProb);
            brdf = mat->evaluateSpecular(N, wo, wi);
        } else {
            wi = sampleCosineHemisphere(N, pdf);
            pdf *= (1 - specProb);
            brdf = mat->evaluateDiffuse(N, wi, kd);
        }
        indirect = brdf * cosO * Li / (pdf * rrProb);
```

### 4.4 `scene_glossy.txt` 场景要点

- Cornell Box + **5 个 Glossy 球**（无玻璃/镜面球，突出 BRDF）。
- 1 个点光源 `(0, 1.9, 0)`，color `(2,2,2)`。
- 推荐渲染：`path_nee 64`。

| 球 | 类型 | roughness m | F₀ |
|----|------|-------------|-----|
| 红/蓝/绿 | 塑料 | 0.22 / 0.28 / 0.18 | 0.04 |
| 金/银 | 金属（kd=0） | 0.35 / 0.28 | 与 ks 同色 |

### 4.5 常见坑与修复

| 问题 | 修复 |
|------|------|
| 金属高光 firefly 过曝 | 增大 roughness（0.35/0.28）；`clampRadiance` 上限 30 |
| 粗糙度为 0 数值不稳定 | `GlossyMaterial` 构造时 `roughness = max(0.03, roughness)` |
| 采样到半球下方 | `dot(N, wi) <= 0` 时仅返回 direct，不追间接 |

---

## 5. 场景文件与命令行

### 5.1 渲染模式（`main.cpp`）

```bash
./PA1-2 <scene.txt> <output.bmp> [whitted|path|path_nee|path_mis] [spp] [gamma] [omp|parallel] [dispersion] [cuda|gpu]
```

| 模式 | `RenderMode` | 默认 SPP | 用途 |
|------|--------------|----------|------|
| `whitted` | `WHITTED` | 1 | 基础要求 1：反射/折射/阴影 |
| `path` | `PATH_TRACE` | 64 | 路径追踪，**无 NEE**（§4.1 对比） |
| `path_nee` | `PATH_TRACE_NEE` | 64 | 路径追踪 + NEE（§4.3） |
| `path_mis` | `PATH_TRACE_MIS` | 64 | 路径追踪 + NEE + MIS（§7.5） |

可选标志（顺序任意，从第 4 个参数起）：`gamma`、`omp`/`parallel`、`dispersion`、`cuda`/`gpu`（详见 §7.1、§7.2、§7.4）。

### 5.2 规范场景文件

| 文件 | 路径 | 说明 |
|------|------|------|
| `scene_whitted.txt` | `code/testcases/` | Whitted 演示：点光源、镜面球+玻璃立方体 |
| `scene_path.txt` | `code/testcases/` | 路径追踪：AreaLight + Emissive 天花板 |
| `scene_glossy.txt` | `code/testcases/` | Cook-Torrance 五球 |
| `scene08_whitted.txt` | `code/testcases/` | 与 `scene_whitted` 同布局（Cornell 编号版） |
| `scene08_path.txt` | `code/testcases/` | 与 `scene_path` 同布局 |

**`submit_scenes/`**（仓库根目录）：已恢复与 `testcases/` 中上述场景 **一致** 的提交副本，便于打包验收：

- `submit_scenes/scene_whitted.txt`
- `submit_scenes/scene_path.txt`
- `submit_scenes/scene_glossy.txt`
- `submit_scenes/scene08_whitted.txt`
- `submit_scenes/scene08_path.txt`

### 5.3 推荐渲染命令（在 `code/` 目录）

```bash
cmake -B build && cmake --build build

# 基础要求 1
build/PA1-2 testcases/scene_whitted.txt output/whitted.bmp whitted

# 基础要求 2 + NEE 对比
build/PA1-2 testcases/scene_path.txt output/path_no_nee.bmp path 64
build/PA1-2 testcases/scene_path.txt output/path_nee.bmp path_nee 64

# §4.2 光泽
build/PA1-2 testcases/scene_glossy.txt output/glossy.bmp path_nee 64

# 加分项：gamma / OpenMP / CUDA / MIS（详见 §7）
build/PA1-2 testcases/scene_path.txt output/path_gamma.bmp path_nee 64 gamma
build/PA1-2 testcases/scene_path.txt output/path_omp.bmp path_nee 64 omp
build/PA1-2 testcases/scene_glossy.txt output/glossy_fast.bmp path_nee 64 gamma omp
build/PA1-2 testcases/scene_mis_demo.txt output/mis_demo_mis.bmp path_mis 32 gamma
build/PA1-2 testcases/scene_showcase.txt output/showcase_cuda.bmp path_nee 128 gamma cuda
build/PA1-2 testcases/scene_showcase.txt output/dispersion_after_cuda.bmp path_nee 256 gamma dispersion cuda
```

### 5.4 两场景关键差异（答辩常问）

| 项目 | `scene_whitted.txt` | `scene_path.txt` |
|------|---------------------|------------------|
| 光源 | 1× PointLight | 2× AreaLight + Emissive 天花板三角形 |
| 后墙材质 | 普通 Diffuse (0.65…) | 索引 8: Emissive，索引 9: 后墙 Diffuse |
| 预期效果 | 硬阴影、无噪声 | 软阴影、路径噪声（SPP 越高越干净） |

**几何**：玻璃立方体 `Translate(-0.55, 0.36, 0.62)` + `UniformScale(0.36)`，底面 **贴地** y=0。

---

## 6. 关键文件对照表

| 文件 | 职责 |
|------|------|
| `include/raytracer.hpp` | **核心**：Whitted / Path / NEE、RR、阴影、Snell、采样 |
| `include/material.hpp` | 材质类型、Cook-Torrance BRDF、Phong Shade |
| `include/light.hpp` | PointLight、DirectionalLight、AreaLight |
| `include/camera.hpp` | PerspectiveCamera、`generateRay` |
| `include/hit.hpp` | 交点信息（法线、UV、材质指针） |
| `include/scene_parser.hpp` + `src/scene_parser.cpp` | 场景/材质/光源解析 |
| `src/main.cpp` | CLI、SPP 循环、像素抖动、计时；`cuda` 时走 `renderWithCuda` |
| `src/cuda_path_tracer.cu` | CUDA 内核：`renderKernel`、`castRayPath`、`castRayWhitted`、MIS、色散 |
| `src/cuda_scene_builder.cpp` | 场景树 → GPU 扁平数组（`buildGpuSceneHost`） |
| `include/cuda_types.h` | GPU 侧结构体与 `GpuRenderMode` 枚举 |
| `include/cuda_renderer.hpp` | `renderWithCuda` / `cudaAvailable` 声明 |
| `include/cuda_device.hpp` | `__device__` / `__host__` 宏 |
| `include/cuda_alloc.hpp` | Unified Memory 辅助（`cudaMallocManaged`）；**主渲染路径未使用** |
| `src/image.cpp` | BMP 输出（可选 gamma，加分项） |
| `testcases/scene_*.txt` | 测试场景 |
| `submit_scenes/scene_*.txt` | 提交用场景副本 |

**常量一览**（`raytracer.hpp`）：

| 常量 | 值 | 含义 |
|------|-----|------|
| `MAX_TRACE_DEPTH` | 12 | 最大递归深度 |
| `RR_START_DEPTH` | 8 | 开始俄罗斯轮盘赌的深度 |
| `RR_MIN_SURVIVAL` | 0.15 | RR 最小存活概率 |
| `ORIGIN_OFFSET` | 1e-3 | 反射/漫反射法线偏移 |
| `REFRACT_ORIGIN_OFFSET` | 2e-3 | 折射 **沿光线** 偏移 |
| `SHADOW_EPSILON` | 1e-3 | 阴影射线偏移 |
| `PATH_RADIANCE_CLAMP` | 30 | 辐射度软钳制 |

---

## 7. 加分项

以下功能 **不属于基础验收核心**，答辩时一笔带过即可；其中 **Gamma** 与 **OpenMP** 已在当前代码中完整实现，本节按与 §1–§4 相同的结构展开。

### 7.1 Gamma 校正

#### 7.1.1 原理是什么

路径追踪在帧缓冲中存储的是 **线性辐射度**（radiance），数值往往偏暗、对比度低。标准显示器按 **sRGB / gamma ≈ 2.2** 编码：像素值 \(V\) 与感知亮度近似满足 \(L \propto V^{2.2}\)。

因此在 **写 BMP 之前** 对 RGB 做 **gamma 编码**（也称 display gamma）：

\[
V = C^{1/2.2}, \quad C \ge 0
\]

其中 \(C\) 为渲染得到的线性颜色分量（已按 SPP 平均，范围通常在 \([0,1]\) 附近，高亮处可能略超 1）。编码后再映射到 8-bit：`round(clamp(V,0,1) × 255)`。

**注意**：gamma 仅作用于 **最终输出**，追踪过程中的 BRDF、NEE、throughput 仍在 **线性空间** 计算；不在 `SetPixel` 或 `trace()` 内做 gamma。

#### 7.1.2 在哪些文件/函数实现

| 组件 | 位置 |
|------|------|
| CLI 开关解析 | `src/main.cpp` — `parseGammaFlag()` / `parseFlag()` |
| 选项 token 识别 | `isOptionToken()`（避免把 `gamma` 误当成 SPP 数字） |
| 保存时编码 | `src/image.cpp` — `EncodeColorComponent()`, `Image::SaveBMP(..., applyGamma)` |
| 接口声明 | `include/image.hpp` — `SaveBMP` 第二参数默认 `false` |

#### 7.1.3 关键代码逻辑

**CLI 解析**（`gamma` / `--gamma` 可出现在第 4 个参数及之后的任意位置）：

```50:52:code/src/main.cpp
static bool parseGammaFlag(int argc, char *argv[]) {
    return parseFlag(argc, argv, "gamma", "--gamma");
}
```

```93:94:code/src/main.cpp
    bool applyGamma = parseGammaFlag(argc, argv);
    bool useOmp = parseOmpFlag(argc, argv);
```

```160:160:code/src/main.cpp
    dImg.SaveBMP(outputFile.c_str(), applyGamma);
```

**编码与写盘**（负值先钳为 0，再 pow，再 8-bit 钳制）：

```24:30:code/src/image.cpp
static float EncodeColorComponent( float c, bool applyGamma )
{
    if ( applyGamma ) {
        c = std::pow( c < 0.0f ? 0.0f : c, 1.0f / 2.2f );
    }
    return c;
}
```

```289:291:code/src/image.cpp
            line[3*j] = ClampColorComponent(EncodeColorComponent(rgb[ipos][2], applyGamma));
            line[3*j+1] = ClampColorComponent(EncodeColorComponent(rgb[ipos][1], applyGamma));
            line[3*j+2] = ClampColorComponent(EncodeColorComponent(rgb[ipos][0], applyGamma));
```

**数据流**：

```
trace() → 线性 Vector3f → SetPixel 累加/平均 → SaveBMP
                                              ↓ applyGamma=true
                                    EncodeColorComponent (pow 1/2.2)
                                              ↓
                                    ClampColorComponent → BMP 字节
```

#### 7.1.4 CLI 用法示例

```bash
# 路径追踪，默认 SPP=64，开启 gamma（中间调变亮，更符合屏幕观感）
build/PA1-2 testcases/scene_path.txt output/path_gamma.bmp path_nee gamma

# 显式指定 SPP，gamma 与 spp 顺序可互换
build/PA1-2 testcases/scene_path.txt output/path_gamma64.bmp path_nee 64 gamma
build/PA1-2 testcases/scene_path.txt output/path_gamma64.bmp path_nee gamma 64

# Whitted 对比：无 gamma 偏暗，有 gamma 高光与中间调更接近常见参考图
build/PA1-2 testcases/scene_whitted.txt output/whitted_linear.bmp whitted
build/PA1-2 testcases/scene_whitted.txt output/whitted_gamma.bmp whitted gamma
```

启动时会打印 `gamma: on/off`，便于确认开关状态。

#### 7.1.5 常见坑与修复

| 问题 | 原因 | 说明 / 修复 |
|------|------|-------------|
| 开 gamma 后仍偏暗 | 线性 radiance 本身未收敛或 NEE 未开 | 先保证 `path_nee` + 足够 SPP；gamma 只改 **显示映射**，不增加真实能量 |
| 与参考图亮度不一致 | 对方可能在线性空间 tonemap，或用了不同 γ | 本实现固定 **1/2.2**；对比实验应统一是否加 `gamma` |
| 高光发灰/过曝 | 线性值 >1 经 pow 后仍可能顶满 255 | `clampRadiance` 限制追踪亮度；`ClampColorComponent` 再钳 8-bit |
| 误以为追踪中要 gamma | 概念混淆 | **仅** `SaveBMP` 路径调用 `EncodeColorComponent`；`SaveTGA` 等其它导出未接 gamma 开关 |

---

### 7.2 CPU 加速（OpenMP）

#### 7.2.1 原理是什么

路径追踪 **像素之间无数据依赖**（每像素独立 RNG seed、独立 `RayTracer`），天然适合 **数据并行**。实现上对 **外层扫描线循环** `for (y …)` 使用 OpenMP `parallel for`，多线程同时处理不同行，从而缩短墙钟时间。

调度策略为 `schedule(dynamic, 4)`：每轮动态分配 4 行一块，减轻各行 SPP/材质复杂度不均导致的 **负载失衡**（例如上半屏空天空、下半屏复杂几何）。

编译期通过 CMake `find_package(OpenMP)` 链接 `OpenMP::OpenMP_CXX`；运行时由 CLI `omp` / `parallel` 开关决定是否启用（`if (useOmp)`），便于同一二进制对比串行与并行耗时。

#### 7.2.2 在哪些文件/函数实现

| 组件 | 位置 |
|------|------|
| CMake 检测与链接 | `code/CMakeLists.txt` — `FIND_PACKAGE(OpenMP)` |
| CLI 开关 | `src/main.cpp` — `parseOmpFlag()`（`omp` / `--omp` / `parallel` / `--parallel`） |
| 并行渲染循环 | `src/main.cpp` — `#pragma omp parallel for` 包裹 `y` 循环 |
| 线程数查询 | `omp_get_max_threads()`（仅 `#ifdef _OPENMP` 编译时） |
| 可复现 RNG | 每像素 `seed = f(x, y, s)`，与是否并行无关 |

#### 7.2.3 关键代码逻辑

**CMake**（找到 OpenMP 则链接，否则串行构建并打印提示）：

```42:48:code/CMakeLists.txt
FIND_PACKAGE(OpenMP)
IF(OpenMP_CXX_FOUND)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} OpenMP::OpenMP_CXX)
    MESSAGE(STATUS "OpenMP enabled: ${OpenMP_CXX_FLAGS}")
ELSE()
    MESSAGE(STATUS "OpenMP not found; build is serial-only (pass omp/parallel flag has no effect)")
ENDIF()
```

**运行时开关与降级**：

```54:57:code/src/main.cpp
static bool parseOmpFlag(int argc, char *argv[]) {
    return parseFlag(argc, argv, "omp", "--omp") ||
           parseFlag(argc, argv, "parallel", "--parallel");
}
```

```111:119:code/src/main.cpp
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
```

**并行循环**（仅并行 `y`；内层 `x` 与 SPP 仍在单线程内顺序执行）：

```129:154:code/src/main.cpp
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4) if (useOmp)
#endif
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // ... 每像素新建 RayTracer(scene, mode, seed) ...
            dImg.SetPixel(x, y, accum * (1.0f / spp));
        }
        if (showProgress && !useOmp && (y + 1) % 64 == 0) {
            cout << "Scanline " << (y + 1) << "/" << height << endl;
        }
    }
```

**线程安全要点**：

- `SceneParser scene` 在并行区域 **之前** 构造，各线程 **只读** 场景与相机。
- 每像素独立 `RayTracer` + 确定性 `seed`，无共享 RNG 状态。
- `Image::SetPixel(x,y,…)` 每个 `(x,y)` 只被一个线程写入，无写冲突。
- 开启 OpenMP 时 **关闭** 扫描线进度输出，避免 `cout` 交错乱序。

#### 7.2.4 CLI 用法示例

```bash
# 路径追踪 + 64 SPP + 多线程（线程数由 OpenMP 默认或环境变量决定）
build/PA1-2 testcases/scene_path.txt output/path_omp.bmp path_nee 64 omp

# 等价长选项
build/PA1-2 testcases/scene_path.txt output/path_omp.bmp path_nee 64 parallel

# 与 gamma 组合（标志顺序任意）
build/PA1-2 testcases/scene_glossy.txt output/glossy_fast.bmp path_nee 64 gamma omp

# 限制线程数（标准 OpenMP 环境变量，非程序自定义参数）
export OMP_NUM_THREADS=4
build/PA1-2 testcases/scene_path.txt output/path_omp4.bmp path_nee 64 omp
```

启动日志示例：`Render mode: path_nee, SPP: 64, gamma: on, omp: on (8 threads)`，随后打印 `Render time: … s` 可与不加 `omp` 对比加速比。

#### 7.2.5 常见坑与修复

| 问题 | 原因 | 说明 / 修复 |
|------|------|-------------|
| 加了 `omp` 仍无加速 | 构建时未找到 OpenMP | 查看 cmake 配置输出；macOS 可安装 `libomp` 后重配 `cmake -B build` |
| 日志显示 “OpenMP not available at build time” | 未定义 `_OPENMP` | 程序自动将 `useOmp=false`；需修复 CMake/工具链 |
| 并行与串行图像不一致 | 不应发生（确定性 seed） | 若出现差异，检查是否在 `trace` 内引入共享可变状态 |
| 加速比低于核数 | Amdahl 瓶颈、动态调度开销、内存带宽 | 高 SPP、大分辨率时收益更明显；Whitted SPP=1 时并行仍有效但绝对耗时短 |
| 看不到 Scanline 进度 | 设计如此 | `useOmp` 为真时禁用进度打印；用 `Render time` 判断完成 |
| `OMP_NUM_THREADS` 无效 | 未用 OpenMP 编译 | 先确认 cmake 输出 `OpenMP enabled` |

---

### 7.3 其他加分项简表

| 功能 | 位置 | 说明 |
|------|------|------|
| **纹理 / 法线贴图** | `texture.hpp`, `material.hpp` | 场景 `texture` / 法线贴图字段 |
| **色散（Dispersion）** | `raytracer.hpp` `traceRefractChild`；GPU `castRayPath` 折射分支 | CLI `dispersion`；RGB 分通道 IOR（`channelIor`），详见 §7.4.6 |
| **MIS (`path_mis`)** | `raytracer.hpp`；`cuda_path_tracer.cu` | Power heuristic 合并 NEE 与 BRDF 采样，详见 §7.5 |
| **GPU/CUDA** | `cuda_path_tracer.cu`, `cuda_scene_builder.cpp` | CLI `cuda`/`gpu`，详见 §7.4 |

---

### 7.4 GPU/CUDA 并行加速

#### 7.4.1 为什么需要 GPU？（给零基础同学）

路径追踪的代价主要在：**每个像素要独立发射很多条光线（SPP）**，每条光线又要递归弹射十几次，每次弹射都要和场景里所有球/平面/三角形求交。

- **CPU**：核心少（8～16 个），但单核很强，适合复杂分支逻辑。
- **GPU**：核心极多（数千个），每个核心较弱，但 **同一时间能处理成千上万个彼此无关的任务**。

渲染恰好是「每个像素互不干扰」——像素 A 算到什么深度，跟像素 B 无关。这种任务叫 **数据并行（embarrassingly parallel）**，非常适合 GPU：**一个 CUDA 线程负责一个像素**，所有线程同时跑 `castRayPath`。

类比：CPU 像 8 个高级工程师各画一幅画；GPU 像 4096 个实习生每人只画一个点，但所有人同时动笔，总墙钟时间往往短很多。

#### 7.4.2 整体架构：场景扁平化 + 显式显存上传（不是 Unified Memory 主路径）

CPU 侧场景是一棵 **对象树**（`Group` → `Transform` → `Sphere`/`Mesh`…），带虚函数、`dynamic_cast`，GPU 无法直接遍历。

因此采用 **场景扁平化（Scene Flattening）**：

1. **`cuda_scene_builder.cpp`** 递归遍历场景树，把几何体变成 **纯数组**：`GpuSphere[]`、`GpuPlane[]`、`GpuTriangle[]`；材质变成 `GpuMaterial[]`；光源变成 `GpuAreaLight[]` 等。
2. **`uploadScene`**（在 `cuda_path_tracer.cu`）用 `cudaMalloc` + `cudaMemcpy(HostToDevice)` 把这些数组拷到 **GPU 显存**。
3. 内核里 `intersectScene` 用 **for 循环扫数组** 求最近交点（与 CPU `Group::intersect` 逻辑等价，但无指针跳转）。

**与 Unified Memory 的区别**（`include/cuda_alloc.hpp` 提供了 `cudaMallocManaged`，但 **当前主渲染管线未使用**）：

| 方式 | 做法 | 本项目的选用 |
|------|------|--------------|
| **显式上传** | Host 数组 → `cudaMemcpy` → Device 指针 | ✅ 实际使用；数据布局清晰、可预测 |
| **Unified Memory** | `cudaMallocManaged`，CPU/GPU 共用同一指针，由驱动按需迁移页 | 仅头文件预留；未接入 `renderWithCuda` |

扁平化后，设备端用一个小结构体 `GpuSceneDevice` 保存各数组指针与数量，内核只读这一份「快照」。

#### 7.4.3 新增文件清单

| 文件 | 职责 |
|------|------|
| `include/cuda_types.h` | GPU 结构体：`GpuMaterial`、`GpuTriangle`、`GpuCamera`、`GpuRenderMode` 等 |
| `include/cuda_renderer.hpp` | 对外 API：`cudaAvailable()`、`renderWithCuda()`、`freeCudaSceneCache()` |
| `include/cuda_device.hpp` | `HOST_DEVICE` / `DEVICE` 宏，供 `.cu` 与 C++ 共用 |
| `include/cuda_alloc.hpp` | Unified Memory 的 `new`/`delete` 重载（备用，主路径未用） |
| `src/cuda_scene_builder.cpp` | `SceneFlattener`：解析后场景 → `GpuSceneHost` |
| `src/cuda_path_tracer.cu` | 全部 `__device__` 追踪逻辑 + `renderKernel` + `renderWithCuda` 宿主代码 |
| `CMakeLists.txt` | 检测 `CMAKE_CUDA_COMPILER`，链接 `CUDA::cudart`、`CUDA::curand`，定义 `USE_CUDA=1` |

#### 7.4.4 关键函数对照表

| 函数 | 文件 | 作用 |
|------|------|------|
| `buildGpuSceneHost` | `cuda_scene_builder.cpp` | 入口：扁平化整个 `SceneParser` |
| `SceneFlattener::flattenObject` | 同上 | 递归展开 Group/Transform/Sphere/Plane/Mesh |
| `uploadScene` | `cuda_path_tracer.cu` | `cudaMalloc`/`cudaMemcpy` 上传各数组 |
| `renderWithCuda` | 同上 | 宿主侧：建场景、launch 内核、回读像素 |
| `initCurandKernel` | 同上 | 每像素初始化 `curandState` |
| `renderKernel` | 同上 | **主内核**：一线程一像素，内层 SPP 循环 |
| `castRayPath` | 同上 | GPU 路径追踪（NEE/MIS/色散/RR） |
| `castRayWhitted` | 同上 | GPU Whitted 追踪 |
| `intersectScene` | 同上 | 遍历球/面/三角形求最近 hit |
| `parseCudaFlag` | `main.cpp` | 解析 CLI `cuda` / `gpu` |

#### 7.4.5 数据流（从命令行到 BMP）

```
main.cpp
  ├─ parseCudaFlag → useCuda=true
  ├─ SceneParser 读场景（仍在 CPU）
  └─ renderWithCuda(scene, image, mode, spp, dispersion, time)
        ├─ buildGpuSceneHost(scene)     // CPU：树 → 扁平 vector
        ├─ uploadScene(host, w, h)      // cudaMalloc + cudaMemcpy
        ├─ dim3 block(16,16); dim3 grid(...)
        ├─ initCurandKernel<<<grid,block>>>   // 固定种子，每像素一条 curand 序列
        ├─ renderKernel<<<grid,block>>>       // 并行渲染 → g_device.pixels
        ├─ cudaMemcpy(DeviceToHost) → hostPixels
        └─ Image::SetPixel → SaveBMP（gamma 仍在 CPU 的 image.cpp）
```

内核内单像素逻辑（与 CPU `main` 循环对应）：

```971:1004:code/src/cuda_path_tracer.cu
__global__ void renderKernel(const GpuSceneDevice scene, float *output, curandState *rngStates,
                             int width, int height, int spp, int mode, bool dispersion) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // ...
    for (int s = 0; s < spp; ++s) {
        // 子像素抖动 hash01（与 CPU 相同公式）
        // castRayWhitted 或 castRayPath
    }
    // 写 output[(y*width+x)*3 + c]
}
```

#### 7.4.6 怎么用：编译与运行

**编译**（需本机安装 CUDA Toolkit，CMake ≥ 3.16）：

```bash
cd code
cmake -B build && cmake --build build
# 配置时应看到：CUDA compiler found: ...
# 且链接 cudart、curand
```

`CMakeLists.txt` 在检测到 `nvcc` 时自动加入 `.cu` 源文件并定义 `USE_CUDA=1`；未检测到则 **仅 CPU** 构建，`cuda` 标志无效。

**运行示例**：

```bash
# GPU 路径追踪 + NEE
build/PA1-2 testcases/scene_path.txt output/path_cuda.bmp path_nee 64 cuda

# GPU + MIS + gamma
build/PA1-2 testcases/scene_mis_demo.txt output/mis_cuda.bmp path_mis 32 gamma cuda

# GPU + 色散（折射材质 dispersionDelta > 0）
build/PA1-2 testcases/scene_dispersion.txt output/disp_cuda.bmp path_nee 256 gamma dispersion cuda
```

若 `cudaAvailable()` 失败或内核报错，`main.cpp` 会打印 `CUDA rendering unavailable or failed; falling back to CPU.` 并自动走 CPU 循环。

**注意**：`cuda` 与 `omp` 互斥使用——开了 `cuda` 时直接 `return`，不会进入 OpenMP 像素循环。

#### 7.4.7 curand、固定种子、Block/Grid

**随机数（curand）**

- 每个像素一个 `curandState`，存在 `g_device.rngStates[width*height]`。
- 启动前 `initCurandKernel` 调用 `curand_init(seed, idx, 0, &states[idx])`：
  - `seed` = 常量 `kCudaRenderSeed = 104729`（固定，便于复现、对比 dispersion/MIS）
  - `idx` = 像素线性下标 `y*width+x`（每像素独立子序列）
- 路径追踪内用 `gpuUniform(rng)` → `curand_uniform`；子像素 **抖动** 仍用确定性 `hash01(x,y,s)`（与 CPU 一致，不消耗 curand 状态）。

**Block / Grid**

```1193:1194:code/src/cuda_path_tracer.cu
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
```

- **Block**：16×16 = 256 线程（常见选择，占满一个 warp 组的整数倍）。
- **Grid**：覆盖整个图像，边缘 block 内部分线程 `x>=width` 会 early return。
- 映射：`thread (x,y)` ↔ 像素坐标，**并行维度是像素**，SPP 在线程内 **串行 for 循环**（避免过多寄存器/spawn）。

**栈限制**：`cudaDeviceSetLimit(cudaLimitStackSize, 65536)` — 路径递归在设备函数内深度可达 12，需加大默认栈。

#### 7.4.8 GPU 与 CPU 的逻辑对齐

| 项目 | CPU | GPU |
|------|-----|-----|
| 追踪核心 | `raytracer.hpp` `castRayPath` | `cuda_path_tracer.cu` `castRayPath` |
| 场景求交 | `Group::intersect` | `intersectScene` 扫扁平数组 |
| 模式枚举 | `RenderMode` | `GpuRenderMode`（`GPU_PATH_NEE` 等） |
| RR / 深度 / clamp | 同名常量 | `kMaxDepth`、`kRrStartDepth`、`kRadianceClamp` |
| 色散 | `channelIor` + 分通道折射 | 同公式，`dispersion` 标志传入内核 |

色散原理简述：玻璃 IOR 随波长略变，实现上对 R/G/B **各追踪一条折射子路径**（`dispChannel` 0/1/2），IOR 分别为 `base−δ`、`base`、`base+δ`，最后在像素处合成 RGB。CPU 见 `traceRefractChild`；GPU 见 `castRayPath` 中 `GPU_MAT_REFRACT` 分支。

#### 7.4.9 常见坑

| 问题 | 说明 |
|------|------|
| 构建无 CUDA | 未装 `nvcc` 或 CMake 找不到；只能 CPU |
| 图像与 CPU 不完全逐像素相同 | RNG 发生器不同（CPU 为 xorshift，GPU 为 curand），统计分布应一致 |
| `scene_glossy` 上 `path_nee` vs `path_mis` 无差别 | 该场景只有 **点光源**，无面光源；MIS 主要合并 **面光源 NEE vs BRDF**。请用 `scene_mis_demo.txt` |
| 纹理/法线贴图在 GPU 路径 | 扁平化材质只拷贝常数 `diffuse`/`roughness`；带贴图场景 GPU 可能与 CPU 有差异 |

---

### 7.5 MIS（多重重要性采样）

#### 7.5.1 原理：为什么需要 MIS？（直觉版）

估计像素颜色时，我们常常有 **两种（或多种）合法采样手段**：

1. **按材质采样（BSDF/BRDF）**：在命中点随机选一个出射方向 \(\omega_i\)，看能不能打到灯。
2. **按光源采样（NEE）**：直接在小灯上随机选一点，看能不能无遮挡地连到命中点。

单独用任何一种，在「另一种策略更划算」的情形下都会 **方差很大**（画面噪点、亮点/firefly）。

**多重重要性采样（MIS）** 的思想：同一点的光照贡献，可以用多种策略各采一次，再用 **权重** \(w_i\) 合并，使得只要有一种策略的 pdf 合理，整体估计就稳定。合并后仍保持 **无偏**（期望正确）。

#### 7.5.2 Balance Heuristic vs Power Heuristic

对两个采样策略，pdf 分别为 \(p_1, p_2\)。某次样本由策略 1 产生，pdf 为 \(p\)。

| 启发式 | 权重公式 | 特点 |
|--------|----------|------|
| **Balance** | \(w = \dfrac{p}{p_1 + p_2}\) | 简单；当两种 pdf 差几个数量级时，方差仍可能偏大 |
| **Power（β=2）** | \(w = \dfrac{p^2}{p_1^2 + p_2^2}\) | 更 **压制** 极小 pdf 那一方的贡献；实践常用 β=2 |

**本项目实现 Power heuristic（β=2）**：

```291:301:code/include/raytracer.hpp
    static float misPowerDenom(float pdfLight, float pdfBrdf) {
        return pdfLight * pdfLight + pdfBrdf * pdfBrdf;
    }

    static float misWeightPower(float pdf, float pdfLight, float pdfBrdf) {
        float denom = misPowerDenom(pdfLight, pdfBrdf);
        if (denom < 1e-8f) {
            return 0.0f;
        }
        return pdf * pdf / denom;
    }
```

- **直接光（NEE）**：样本由光源采样产生，\(p = p_\text{light}\)，权重 `misWeightPower(pdfLight, pdfLight, pdfBrdf)`。
- **间接命中发光体**：样本由 BRDF 采样产生，\(p = p_\text{brdf}\)，权重 `misWeightPower(pdfBrdf, pdfLight, pdfBrdf)`，并乘 `scale = misW / pdfBrdf` 修正贡献。

#### 7.5.3 三种模式：`path` / `path_nee` / `path_mis`

| 模式 | NEE 直接光 | 间接命中 Emissive | MIS 权重 | 典型场景 |
|------|------------|------------------------|----------|----------|
| `path` | ❌ | ✅ 全额 `throughput×emission` | ❌ | 与 `path_nee` 对比方差 |
| `path_nee` | ✅ 标准 NEE 公式 | ✅ `countEmissive=true`（无 MIS 缩放） | ❌ | 日常渲染、Cornell 软阴影 |
| `path_mis` | ✅ NEE × Power 权 | ✅ 命中时 × MIS 权 | ✅ | 光泽 + 面光源；`scene_mis_demo` |

**重要**：`path_mis` 下 `useNEE()` 与 `useMIS()` 均为真。与 `path_nee` 的核心差别是：**是否在 NEE 样本与间接命中发光体时套用 Power heuristic 权重**，而不是简单关闭间接发光。

**对比实验场景**：

- `scene_path.txt`：`path` vs `path_nee`（NEE 价值）
- `scene_mis_demo.txt`：`path_nee` vs `path_mis`（MIS 价值；**不要用** `scene_glossy.txt`——只有点光源，NEE 与 MIS 几乎一样）

#### 7.5.4 CPU 实现：函数与代码位置

| 组件 | 位置 |
|------|------|
| 模式开关 | `useMIS()`、`useNEE()`（`raytracer.hpp`） |
| 上下文 | `MisIndirectCtx`：记录 `pdfBrdf, wi, shadingPoint, N, wo, glossyMat` |
| 面光源 pdf（给定方向） | `pdfAreaLightDirection` / `computeAreaLightPdf` |
| BRDF pdf | `pdfDiffuseBRDF`（Lambert）；`pdfGlossyBRDF`（Cook-Torrance 混合瓣） |
| NEE + MIS（漫反射） | `sampleOneAreaLightDiffuse` 内 `useMIS()` 分支 |
| NEE + MIS（光泽） | `sampleOneAreaLightGlossy` |
| 间接命中发光体 | `castRayPath` 命中 `EMISSIVE` 且 `misCtx!=nullptr` |

**直接光 MIS（以光泽为例）**——先按灯采样，再算「若用 BRDF 采样得到同一 \(\omega_i\)」的 pdf，套 Power 权：

```410:417:code/include/raytracer.hpp
        if (useMIS()) {
            float pdfLight = dist2 / (lightArea * cosL);
            float pdfBrdf = pdfGlossyBRDF(N, wo, wi, mat);
            float misW = misWeightPower(pdfLight, pdfLight, pdfBrdf);
            // ...
            contrib += brdf * area->getColor() * cosO / pdfLight * misW;
```

**间接光 MIS**——在 `shadeDiffusePath` / `shadeGlossyPath` 向子路径传入 `MisIndirectCtx`：

```722:732:code/include/raytracer.hpp
                if (useMIS()) {
                    misCtx.pdfBrdf = pdf;
                    misCtx.wi = wi;
                    misCtx.shadingPoint = hitPoint;
                    // ...
                    misPtr = &misCtx;
                }
                Vector3f Li = castRayPath(Ray(origin, wi), depth + 1, throughput, true, misPtr,
                                          dispChannel);
```

子路径若打在发光体上：

```656:666:code/include/raytracer.hpp
            if (misCtx != nullptr) {
                float pdfLight = computeAreaLightPdf(misCtx->shadingPoint, misCtx->wi);
                float misW = misWeightPower(misCtx->pdfBrdf, pdfLight, misCtx->pdfBrdf);
                float scale = misW / misCtx->pdfBrdf;
                return clampRadiance(throughput * emission * scale);
            }
```

#### 7.5.5 GPU 实现：`MisCtx` 与 CPU 对应关系

GPU 侧结构体（字段略少，无 `wo`/`glossyMat` 指针，光泽 pdf 在内核用 `pdfGlossy` 重算）：

```609:616:code/src/cuda_path_tracer.cu
struct MisCtx {
    float pdfBrdf;
    float3 wi;
    float3 shadingPoint;
    float3 N;
    bool active;
    bool glossyPath;
};
```

| CPU | GPU |
|-----|-----|
| `MisIndirectCtx` | `MisCtx` |
| `misWeightPower` / `misPowerDenom` | 同名 `__device__` 函数 |
| `sampleOneAreaLightGlossy` | `sampleAreaLightGlossy` |
| `computeAreaLightPdf` | `computeAreaLightPdf` |
| `useMIS()` | `mode == GPU_PATH_MIS` |

直接光：`sampleAreaLightDiffuse` / `sampleAreaLightGlossy` 在 `useMis==true` 时与 CPU 相同公式。

间接发光命中（注意参数顺序与 CPU 一致，均为 Power 权）：

```646:654:code/src/cuda_path_tracer.cu
        if (misCtx != nullptr && misCtx->active) {
            float pdfLight = computeAreaLightPdf(scene, misCtx->shadingPoint, misCtx->wi);
            float misW = misWeightPower(misCtx->pdfBrdf, misCtx->pdfBrdf, pdfLight);
            float scale = misW / misCtx->pdfBrdf;
            return clampRadiance3(mul3(mul3v(throughput, emission), scale));
        }
```

`renderKernel` 将 `RenderMode::PATH_TRACE_MIS` 映射为 `GPU_PATH_MIS` 传入 `castRayPath`。

#### 7.5.6 验收常见问答（Q&A）

**Q1：MIS 和 NEE 是什么关系？**

A：NEE 是「多了一种采样直接光的手段」。MIS 是「当 **NEE 采样** 和 **BRDF 随机采样** 都能解释同一条光路时，用权重把两方合并，避免只信一方导致方差爆炸」。`path_nee` 只有 NEE；`path_mis` = NEE + 对 BRDF 采样路径的权重修正。

**Q2：`path_nee` 和 `path_mis` 在代码里差在哪？**

A：两者都开 NEE、间接子路径都传 `countEmissive=true`。`path_mis` 额外在（1）面光源 NEE 的 `sampleOneAreaLightDiffuse/Glossy` 里乘 MIS 权；（2）间接弹射前写入 `MisIndirectCtx`，子路径打在发光体上时用 `misWeightPower` 缩放。`path_nee` 不做这两步权重。

**Q3：Power heuristic 比 Balance 好在哪里？**

A：当 \(p_\text{light} \ll p_\text{brdf}\) 或反过来时，Balance 仍可能给 **极差策略** 不小权重；Power（β=2）让权重与 pdf **平方** 成正比，小 pdf 方权重被压得更狠，高光/firefly 更少。本实现固定 β=2，与 PBRT 默认实践一致。

**Q4：为什么对比 MIS 要用 `scene_mis_demo.txt` 而不是 `scene_glossy.txt`？**

A：`scene_glossy` 只有 **点光源**，面光源 MIS 分支几乎不执行，`path_nee` 与 `path_mis` 图像几乎相同。`scene_mis_demo` 含 **面光源 + 光泽/漫反射**，才能看出 MIS 在软阴影边缘、亮斑处的降噪。

**Q5：CPU `path_mis` 和 GPU `path_mis cuda` 结果为何不完全一致？**

A：随机数发生器不同（CPU xorshift vs GPU curand），Monte Carlo 噪声图案不同；**期望**应接近。验收可比相同 SPP 下的平均亮度、软阴影形态，或提高 SPP 后目视收敛。

**Q6：点光源做 MIS 了吗？**

A：**面光源**在 NEE 与间接命中发光体两条路径上做了 MIS。**点光源**仍用常规 shadow ray + BRDF 点乘（`sampleDirectPointLights`），未纳入双策略 pdf 合并——答辩可如实说明范围。

#### 7.5.7 推荐渲染命令

```bash
# CPU MIS 对比（32 spp 即可看出差异）
build/PA1-2 testcases/scene_mis_demo.txt output/mis_nee.bmp path_nee 32 gamma
build/PA1-2 testcases/scene_mis_demo.txt output/mis_mis.bmp path_mis 32 gamma

# GPU 同上
build/PA1-2 testcases/scene_mis_demo.txt output/mis_mis_cuda.bmp path_mis 32 gamma cuda
```

色散与 MIS 可叠加：`... path_mis 128 gamma dispersion cuda`（见 `scene_showcase.txt` / `scene_dispersion.txt`）。

## 答辩速查：基础点 + 加分项一句话

1. **Whitted**：递归反射/折射 + Phong 直接光 + shadow ray；折射用 **沿光线偏移** 修黑镜。
2. **Path**：余弦半球 + RR + throughput；`scene_path` 发光天花板 + 面光源。
3. **NEE**：`path_nee` 在命中点直接采样光源；面光源 pdf 面积→立体角；`path_mis` 再加 MIS 权重。
4. **Glossy**：Cook-Torrance = Beckmann D × G × Schlick F；路径按 kd/ks 分瓣采样。
5. **MIS**：`path_mis` 用 Power heuristic 合并 NEE 与 BRDF 采样；对比用 `scene_mis_demo.txt`。
6. **CUDA**：场景扁平化 + `renderKernel` 一像素一线程；CLI 加 `cuda`；与 CPU 同算法、curand 随机。
7. **CLI/场景**：`whitted` / `path` / `path_nee` / `path_mis` + `gamma` / `omp` / `dispersion` / `cuda`。

---

*文档生成依据当前 `code/` 源码；GPU 与 MIS 详见 §7.4–§7.5。*
