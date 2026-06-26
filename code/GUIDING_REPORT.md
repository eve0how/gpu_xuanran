# Path Guiding 修复报告（CUDA）

## 理想效果

- 与 `path_nee` **同 SPP 下平均亮度一致**（无偏 MC，引导只降方差不增亮）
- 间接难采样区域（墙面色溢、门缝漏光等）**更干净、噪点更少**
- 直接光仍由 NEE 估计，与 `path_nee` 相同

## 旧实现根因（含子代理 c01f9d37 部分修复）

| 问题 | 根因 |
|------|------|
| **偏亮** | ① 训练核误用 `GPU_PATH_NEE`，`useGuide=false`，沉积从未执行或行为不一致；② MIS 用 Power heuristic + 错误的 `misW/pdfStrategy` 组合，未按 `pdf_total=0.5·pdf_brdf+0.5·pdf_guide`；③ 光泽漫反射引导分支漏乘 `cosθ` 和 `1/π`；④ 根路径 `countEmissive=true` 与 NEE 双重计光 |
| **仍然很噪** | 训练无有效沉积（上表①）；空格子被填成均匀分布后仍强行走引导分支；`train_spp` 不足 |
| **c01f9d37 已做** | 引入 `sampleIndirectWithGuide`、`train_spp` CLI、`normalizeGuide` 空格子不改；但将训练改为 `GPU_PATH_NEE` **反而禁用沉积**，且 MIS 公式仍不正确 |

## 本次修复要点

1. **训练**：`trainGuideKernel` 使用 `GPU_PATH_GUIDING` + `trainingPass`；`useGuide = guideGrid && (guidingMode \|\| trainingPass)`；沉积 `luminance(throughput)`、方向 `ω_i=-rayDir`
2. **MIS（Gemini Step 4）**：`pdf_total = 0.5*pdf_brdf + 0.5*pdf_guide`，`contrib = brdf·cosθ·Li / pdf_total`
3. **countEmissive**：`path_nee` / `path_guiding` 根路径 `countEmissive=false`；间接仍为 `false`
4. **空格子**：权重保持 0，无数据时回退余弦 BRDF（等同 `path_nee`）
5. **`sampleGuidingDir`**：引导策略失败时不偷偷回退余弦（避免破坏 MIS）

## 自测结果（修复后）

```
A/B scene_path.txt 128 SPP:
  path_nee    mean=0.4391  std=0.2389
  path_guiding mean=0.4383 std=0.2396  ratio=0.998  ✓ (5% 内)
```

输出：`output/guiding_nee_128.bmp/.png`、`output/guiding_guided_128.bmp/.png`

## 命令

```bash
cd /data/PA1-2/code
cmake -B build && cmake --build build -j$(nproc)

# A/B 对比
./build/PA1-2 testcases/scene_path.txt output/guiding_nee_128.bmp path_nee 128 gamma cuda
./build/PA1-2 testcases/scene_path.txt output/guiding_guided_128.bmp path_guiding 128 gamma cuda train_spp 128

# 自动化自测
python3 test_guiding.py

# 可选：门缝场景
./build/PA1-2 testcases/scene_guiding_door.txt output/guiding_door.bmp path_guiding 128 gamma cuda train_spp 256
```

## 答辩话术

> 我们实现了 Practical Path Guiding 的简化版：16³ 空间网格 + 每格 16×16 方向直方图。第一趟用与 path_nee 相同的路径追踪在线累积入射方向分布；第二趟渲染时对间接漫反射做 50/50 BRDF 与引导采样，用 `pdf_total=0.5·pdf_brdf+0.5·pdf_guide` 做无偏 MIS。与 path_nee 同 SPP 下亮度比约 1.0，说明无能量偏差；引导的作用是在间接区域降低方差而不是提高亮度。SPP 越高画面越干净，但**无偏情况下不应越来越亮**——若曾出现偏亮，是 MIS/双重计光/训练未生效的实现 bug，不是算法特性。

## 简化说明

未实现完整 SD-tree / 在线迭代更新；空间用均匀 16³ 网格、方向用 lat-long 直方图。MIS 与训练沉积公式按 Müller 2017 / 课程 spec 实现，保证无偏。
