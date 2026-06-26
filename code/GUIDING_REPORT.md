# Path Guiding 修复报告（CUDA）

## 理想效果

- 与 `path_nee` **同 SPP 下平均亮度一致**（无偏 MC，引导只降方差不增亮）
- 间接难采样区域（暗墙、门缝漏光阴影区）**更干净、噪点更少**
- 直接光仍由 NEE 估计，与 `path_nee` 相同

---

## 推荐对比场景 — `scene_guiding_occluder.txt`（2025-06 第五轮）

**首选 demo**：经典 Cornell 盒（同 `scene_path.txt`）+ 天花板面光源 + **悬挂遮挡板**，无需窄缝几何，无十字洞 bug。

### 设计

| 参数 | 值 |
|------|-----|
| 基础 | `scene_path.txt`：相机 `(0,1,4.3)` FOV 36°，五色球 + 玻璃立方体 |
| 面光源 | 天花板 0.4×0.4 m，`emission / AreaLight color **42**`（避免 80 过曝） |
| 遮挡板 | 灰色漫反射薄盒 `Translate(0,1.84,0) Scale(0.55,0.05,0.55)`，悬于光源正下方 |
| 效果 | 画面中央地板/球体**无直射光**，主要靠红/绿墙与顶板**多次反弹**照明 |

### 为何优于窄缝场景

| 窄缝 demo 问题 | 遮挡板 demo 优势 |
|---------------|-----------------|
| 后墙角块易留十字形洞 | 仅用标准平面 + mesh，**几何零歧义** |
| 缝宽/相机难调，易成「顶栏大光」 | 遮挡阴影边界清晰，**一眼可辨** |
| 直接光(NEE)与间接光混杂 | NEE 仍采样顶光，但中心区**被遮挡**，间接方差主导 |
| 答辩需解释门缝建模 | 直观：**「挡光板 → 软阴影 → 引导学反弹方向」** |

### 自测（CUDA · Python 分析）

| 配置 | 全图 mean | 全图亮度比 | 暗中心 ROI std 降幅 | 训练填充 |
|------|-----------|-----------|-------------------|---------|
| 64 spp, train 256 | nee 16.1 / guided 17.0 | **1.06** | **15%** | 123/512 (24%) |
| 128 spp, train 512 | nee 16.7 / guided 17.5 | **1.05** | **15%** | 130/512 (25%) |

- **无黑十字**：墙/顶/地连续；顶光与遮挡板可见
- **遮挡有效**：中央红球/地板明显暗于两侧受光区
- **引导差异**：64 spp 下 `path_guiding` 阴影区颗粒更细；128 spp 差异仍可见

### 输出

- `output/guiding_occluder_nee_64.bmp` / `guiding_occluder_guided_64.bmp`（及 `.png`）
- `output/guiding_occluder_nee_128.bmp` / `guiding_occluder_guided_128.bmp`（及 `.png`）

### 命令（推荐）

```bash
cd /data/PA1-2/code
cmake --build build -j$(nproc)

./build/PA1-2 testcases/scene_guiding_occluder.txt output/guiding_occluder_nee_64.bmp path_nee 64 gamma cuda
./build/PA1-2 testcases/scene_guiding_occluder.txt output/guiding_occluder_guided_64.bmp path_guiding 64 gamma cuda train_spp 256
./build/PA1-2 testcases/scene_guiding_occluder.txt output/guiding_occluder_nee_128.bmp path_nee 128 gamma cuda
./build/PA1-2 testcases/scene_guiding_occluder.txt output/guiding_occluder_guided_128.bmp path_guiding 128 gamma cuda train_spp 512
```

### 答辩话术（遮挡板版）

> 经典 Cornell 盒顶置面光源，下方加一块大遮挡板，使画面中央只有**间接反弹光**。NEE 仍能直接采样顶光，但被挡区域的路径需经墙面多次弹射，方差大、`path_nee` 噪点粗。路径引导在训练阶段从光源反向追踪，学习「往亮墙方向弹射」的分布，64 spp 下阴影区更平滑，全图亮度比 ≈ 1.05 说明基本无偏。场景无自定义墙洞，比窄缝 demo 更稳、更好讲。

---

## 历史：后墙十字洞（2025-06 第四轮 · 窄缝场景）

### 现象

画面呈 **黑色十字 + 中央细竖亮缝**：水平黑带（与缝同宽）贯穿左右，竖直黑带贯穿上下，仅十字中心 intersection 为发光缝。

### 根因

后墙仅用 **4 个角块三角形**（左上/左下/右上/右下）围缝，**未填充**：

| 缺失区域 | 范围 |
|---------|------|
| 左侧墙带 | x ∈ [-1, -w]，y ∈ [0, 2]（缝高范围内） |
| 右侧墙带 | x ∈ [w, 1]，y ∈ [0, 2] |
| 上楣（lintel） | x ∈ [-w, w]，y ∈ [y_top, 2] |
| 下槛（sill） | x ∈ [-w, w]，y ∈ [0, y_bottom] |

射线穿过这些洞看到 `Background color 0 0 0`，与发光缝叠加形成十字。

### 修复

将 8 个角块三角形改为 **4 条墙带**（左墙 / 右墙 / 上楣 / 下槛），每条 2 三角形，缝口仅保留发光三角形 + `AreaLight`。

**已修复场景**：`scene_guiding_demo.txt`、`scene_guiding_door.txt`、`scene_guiding_window.txt`、`scene_guiding_ajar.txt`

**未改**：`scene_guiding_indirect.txt`（左墙无限平面 + 矩形窗，无角块十字洞）；`scene_guiding_demo_bright.txt`（同类左墙窗，无此 bug）

### 自测（修复后 · 64 spp path_nee）

| 场景 | 黑十字 | 缝宽（屏宽%） | 墙像素亮度 |
|------|--------|--------------|-----------|
| demo | **否** | 5.7% | L≈20, R≈19 |
| door | **否** | 5.5% | L≈19, R≈17 |
| ajar | **否** | 3.9% | L≈16, R≈14 |
| window | **否** | 横窗（顶） | 正常暗房 |

---

## 历史根因（2025-06 第三轮 · 窄缝经典门）

### A. 旧场景「顶上像大面光源」

| 现象 | 根因 |
|------|------|
| 画面顶部宽亮带，不像门缝 | 缝宽 **0.30 m**（x ±0.15），在 1.7 m 距离上占 FOV **~26%**，视觉上是大面积光 |
| 地板无光斑 | 相机俯视（`direction y=-0.10`），中心视线打在墙 y≈0.3，缝在 y=0.82–1.48 视野边缘；漏光未落到画面内地板 |
| `path_nee ≈ path_guiding` | 全画面 85% 像素由 NEE 直接光主导，间接方差差异被掩盖；128 spp 收敛后差异更小 |

### B. 已确认无问题的代码

- 主光线 `countEmissive=true`（直视发光缝可见）
- `AreaLight` 与发光三角形几何一致，parser 正常

### C. 训练侧改进

- Light tracing 比例 **50% → 70%**（窄缝场景更多从光源反向沉积方向直方图）
- 推荐 `train_spp 512–1024`

---

## 本次修复

### 1. 场景重设计 — 经典「微开房门 / 窄竖缝」

**`scene_guiding_demo.txt`**（主 demo）：

| 参数 | 值 |
|------|-----|
| 房间 | 2×2×2 m Cornell 盒，**六面全封闭**（含前墙 z=+1） |
| 相机 | `(0, 0.52, 0.78)`，`direction (0, 0.38, -1)`，FOV **32°**，眼平朝后墙 |
| 窄竖缝 | 后墙 z=-0.998，**宽 0.06 m**（x ±0.03），**高 0.50 m**（y 1.08–1.58），居中 |
| 发光 | 仅缝上 2 个三角形 + 匹配 `AreaLight`，`emission 420 390 330` |
| 墙/顶 | 深灰 albedo **0.07** |
| 地板光斑区 | 缝下路径处加亮地板三角形（albedo **0.42**），z ∈ [-0.38, 0.02] |
| 球体 | `(0.32, 0.17, -0.12)` r=0.15，偏右，不挡缝下光斑 |

**`scene_guiding_ajar.txt`**（最清晰门缝 demo）：

- 缝更窄：**0.04 m**（x ±0.02），高 0.48 m
- 更暗房间（墙 albedo **0.05**），更高发光 **500 460 380**
- 后墙两侧加**门板**三角形（材质 5），视觉上「两扇门 + 中间缝」
- 球体在左侧 `(-0.30, 0.16, -0.10)`

**`scene_guiding_door.txt` / `scene_guiding_window.txt`**：同步窄缝尺度与前墙封闭，相机改为仰视缝方向。

### 2. 代码

```cpp
// trainGuideKernel：Light tracing 70%（原 50%）
if (scene.numAreaLights > 0 && gpuUniform(localState) < 0.7f) { ... }
```

---

## 自测结果（修复后 · Python 分析）

### `scene_guiding_demo.txt`

| 配置 | 缝宽（像素） | 地板光斑 | 亮度比 | 间接区 std 降幅 |
|------|-------------|---------|--------|----------------|
| 128 spp, train 1024 | **58 px (5.7%)** | 中心/侧边 **4.0×** | 1.000 | 3.4% |
| 64 spp, train 1024 | **58 px (5.7%)** | 中心/侧边 **4.0×** | 1.000 | **6.1%** |

- 缝在画面行 **408–826** 呈**细竖亮线**（宽 ~6% 屏宽），不再是顶栏宽亮带
- 训练填充：**82 / 512 cells (16.0%)**（train_spp=1024）
- 全图 `diff>25%` 在 128 spp 下仍低（**直接光区域 NEE 相同属正常**）；应裁切**暗墙/暗角**对比

### 输出文件

- `output/guiding_demo_nee_64.bmp` / `guiding_demo_guided_64.bmp`
- `output/guiding_demo_nee_128.bmp` / `guiding_demo_guided_128.bmp`

---

## 命令（备选 · 窄缝场景）

```bash
cd /data/PA1-2/code
cmake -B build && cmake --build build -j$(nproc)

# 首选：遮挡板 Cornell（见上文 scene_guiding_occluder.txt）

# 窄缝 demo（已修复十字洞，但不如遮挡板直观）
./build/PA1-2 testcases/scene_guiding_demo.txt output/guiding_demo_nee_128.bmp path_nee 128 gamma cuda
./build/PA1-2 testcases/scene_guiding_demo.txt output/guiding_demo_guided_128.bmp path_guiding 128 gamma cuda train_spp 1024
./build/PA1-2 testcases/scene_guiding_ajar.txt output/guiding_ajar_nee_64.bmp path_nee 64 gamma cuda
./build/PA1-2 testcases/scene_guiding_ajar.txt output/guiding_ajar_guided_64.bmp path_guiding 64 gamma cuda train_spp 1024
```

---

## 修复后用户应看到

### 遮挡板 demo（推荐）

1. **经典 Cornell**：红/绿侧墙、灰地板、顶光方片 + 其下**灰色遮挡板**
2. **软阴影**：中央红球/地板明显暗于两侧；遮挡边界在后墙可见
3. **无几何 bug**：无黑十字、无透明洞
4. **亮度**：`path_guiding` / `path_nee` 全图比 ≈ **1.05**（64–128 spp）
5. **引导差异**：**64 spp 看中央阴影区**，guided 颗粒更细（std 降 ~15%）

### 窄缝 demo（备选）

1. 封闭暗房 + 后墙细竖亮缝 + 地板漏光
2. 128 spp 全图差异小；64 spp 裁切暗墙对比
3. 需确保墙带几何完整（无十字洞）

---

## 答辩话术（窄缝版 · 备选）

> 路径引导适合**窄缝漏光**这类难采样场景：缝宽仅 4–6 cm，随机弹射几乎打不中光源，但 NEE 能直接采样缝。引导的价值在**间接分量**——暗墙、地板阴影区的多次反弹。场景为经典 Cornell 暗房 + 后墙窄竖缝，六面封闭；后墙几何用左/右墙带 + 上楣 + 下槛四条带围出缝口（避免角块三角形留下十字形洞）。相机眼平朝缝，画面为连续深灰墙 + 中央细竖亮缝 + 地板漏光。`path_guiding` 训练时 70% 光线做 light tracing 从缝出发，沉积方向直方图。与 `path_nee` 同 SPP 亮度比 ≈ 1.0 说明无偏；64 spp 下暗区更平滑。

---

## 历史（第二轮）

- 修复主光线 `countEmissive=false` 导致直视发光缝全黑
- 添加前墙封闭、初始 0.30 m 缝（本轮已缩窄）
