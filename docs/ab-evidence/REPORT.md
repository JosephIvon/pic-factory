# 生图模型 A/B 实测报告

> **口径指路（2026-10-24 追加，正文一字未改——本文件是历史证据快照）**：本报告数字全部为
> **实验台口径**（YuNet 按面积取脸），已被后续审计订正。生产口径逐模型复算值见
> `model-sim-recompute.json`；生产人脸闸阈值以同域样本标定的 **T=0.57** 为准
> （`threshold-calibration-production.json`）。文中"陌生人基线 0.341"是 n=1 孤例（已作废）、
> "0.45 漂移线"为未标定经验值——两者都**不得再被引用**。订正依据见本目录
> `model-sim-recompute.json` 与 `threshold-calibration-production.json`。

> 同一角色卡（定妆照 1 张作参考图）、同一批 10 条英文画面 prompt、同一 3:4 画幅（1536×2048）、同一后处理（仅保留平台水印，盲测素材已裁掉底部水印）。

> 一致性口径：输出图与定妆照的 ArcFace 余弦相似度（YuNet 5 点对齐 + buffalo_l w600k_r50）。陌生人基线实测 **0.341**。


## 1. 汇总

| 模型 | 成功 | 一致性均值 | 剔除遮挡 | 最低 | 标准差 | 无人脸 | P50 | P95 | 花费 | 有效单张成本 | 参考图上限 | RPM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Seedream 4.0 | 10/10 | 0.729 | **0.757** | 0.503 | 0.105 | 0 | 10.2s | 17.4s | ¥2.0 | ¥0.2 | 10 张 | 未公开 |
| Seedream 5.0 Pro | 10/10 | 0.669 | **0.724** | 0.234 | 0.214 | 0 | 35.0s | 88.9s | ¥3.0 | ¥0.3 | 未公开 | 未公开 |
| Seedream 5.0 Flash | 10/10 | 0.683 | **0.706** | -0.005 | 0.264 | 0 | 11.2s | 18.8s | ¥1.0 | ¥0.1 | 未公开 | 未公开 |
| Qwen-Image 3.0 | 10/10 | 0.775 | **0.844** | 0.226 | 0.201 | 0 | 22.1s | 30.4s | ¥1.8 | ¥0.18 | 1–3 张 | 20 |
| Qwen-Image 3.0 Pro | 10/10 | 0.878 | **0.888** | 0.759 | 0.059 | 0 | 28.2s | 29.5s | ¥4.9 | ¥0.49 | 1–3 张 | 5 |

> 「剔除遮挡」= 去掉 `p09`（贴面膜，面部被遮挡，ArcFace 不可测）后的均值。判断模型请以这列为主。

## 2. 逐条对照


### p01_waterlight_bed

| ark-4.0 | ark-5.0-pro | ark-5.0-flash | qwen-3.0 | qwen-3.0-pro |
|---|---|---|---|---|
| ![](out/ab/ark-4.0/p01_waterlight_bed.jpg) | ![](out/ab/ark-5.0-pro/p01_waterlight_bed.jpg) | ![](out/ab/ark-5.0-flash/p01_waterlight_bed.jpg) | ![](out/ab/qwen-3.0/p01_waterlight_bed.png) | ![](out/ab/qwen-3.0-pro/p01_waterlight_bed.png) |

### p02_waterlight_closeup

| ark-4.0 | ark-5.0-pro | ark-5.0-flash | qwen-3.0 | qwen-3.0-pro |
|---|---|---|---|---|
| ![](out/ab/ark-4.0/p02_waterlight_closeup.jpg) | ![](out/ab/ark-5.0-pro/p02_waterlight_closeup.jpg) | ![](out/ab/ark-5.0-flash/p02_waterlight_closeup.jpg) | ![](out/ab/qwen-3.0/p02_waterlight_closeup.png) | ![](out/ab/qwen-3.0-pro/p02_waterlight_closeup.png) |

### p03_waterlight_standing

| ark-4.0 | ark-5.0-pro | ark-5.0-flash | qwen-3.0 | qwen-3.0-pro |
|---|---|---|---|---|
| ![](out/ab/ark-4.0/p03_waterlight_standing.jpg) | ![](out/ab/ark-5.0-pro/p03_waterlight_standing.jpg) | ![](out/ab/ark-5.0-flash/p03_waterlight_standing.jpg) | ![](out/ab/qwen-3.0/p03_waterlight_standing.png) | ![](out/ab/qwen-3.0-pro/p03_waterlight_standing.png) |

### p04_photon_headband

| ark-4.0 | ark-5.0-pro | ark-5.0-flash | qwen-3.0 | qwen-3.0-pro |
|---|---|---|---|---|
| ![](out/ab/ark-4.0/p04_photon_headband.jpg) | ![](out/ab/ark-5.0-pro/p04_photon_headband.jpg) | ![](out/ab/ark-5.0-flash/p04_photon_headband.jpg) | ![](out/ab/qwen-3.0/p04_photon_headband.png) | ![](out/ab/qwen-3.0-pro/p04_photon_headband.png) |

### p05_photon_consult

| ark-4.0 | ark-5.0-pro | ark-5.0-flash | qwen-3.0 | qwen-3.0-pro |
|---|---|---|---|---|
| ![](out/ab/ark-4.0/p05_photon_consult.jpg) | ![](out/ab/ark-5.0-pro/p05_photon_consult.jpg) | ![](out/ab/ark-5.0-flash/p05_photon_consult.jpg) | ![](out/ab/qwen-3.0/p05_photon_consult.png) | ![](out/ab/qwen-3.0-pro/p05_photon_consult.png) |

### p06_pico_mirror

| ark-4.0 | ark-5.0-pro | ark-5.0-flash | qwen-3.0 | qwen-3.0-pro |
|---|---|---|---|---|
| ![](out/ab/ark-4.0/p06_pico_mirror.jpg) | ![](out/ab/ark-5.0-pro/p06_pico_mirror.jpg) | ![](out/ab/ark-5.0-flash/p06_pico_mirror.jpg) | ![](out/ab/qwen-3.0/p06_pico_mirror.png) | ![](out/ab/qwen-3.0-pro/p06_pico_mirror.png) |

### p07_pico_seated

| ark-4.0 | ark-5.0-pro | ark-5.0-flash | qwen-3.0 | qwen-3.0-pro |
|---|---|---|---|---|
| ![](out/ab/ark-4.0/p07_pico_seated.jpg) | ![](out/ab/ark-5.0-pro/p07_pico_seated.jpg) | ![](out/ab/ark-5.0-flash/p07_pico_seated.jpg) | ![](out/ab/qwen-3.0/p07_pico_seated.png) | ![](out/ab/qwen-3.0-pro/p07_pico_seated.png) |

### p08_generic_price_talk

| ark-4.0 | ark-5.0-pro | ark-5.0-flash | qwen-3.0 | qwen-3.0-pro |
|---|---|---|---|---|
| ![](out/ab/ark-4.0/p08_generic_price_talk.jpg) | ![](out/ab/ark-5.0-pro/p08_generic_price_talk.jpg) | ![](out/ab/ark-5.0-flash/p08_generic_price_talk.jpg) | ![](out/ab/qwen-3.0/p08_generic_price_talk.png) | ![](out/ab/qwen-3.0-pro/p08_generic_price_talk.png) |

### p09_generic_aftercare

| ark-4.0 | ark-5.0-pro | ark-5.0-flash | qwen-3.0 | qwen-3.0-pro |
|---|---|---|---|---|
| ![](out/ab/ark-4.0/p09_generic_aftercare.jpg) | ![](out/ab/ark-5.0-pro/p09_generic_aftercare.jpg) | ![](out/ab/ark-5.0-flash/p09_generic_aftercare.jpg) | ![](out/ab/qwen-3.0/p09_generic_aftercare.png) | ![](out/ab/qwen-3.0-pro/p09_generic_aftercare.png) |

### p10_scene_clinic

| ark-4.0 | ark-5.0-pro | ark-5.0-flash | qwen-3.0 | qwen-3.0-pro |
|---|---|---|---|---|
| ![](out/ab/ark-4.0/p10_scene_clinic.jpg) | ![](out/ab/ark-5.0-pro/p10_scene_clinic.jpg) | ![](out/ab/ark-5.0-flash/p10_scene_clinic.jpg) | ![](out/ab/qwen-3.0/p10_scene_clinic.png) | ![](out/ab/qwen-3.0-pro/p10_scene_clinic.png) |

## 3. 成本口径

- **有效单张成本** = 批次总花费 / 质检通过张数，把失败与返工都摊进去。
- 阿里侧 3.0 系列按「输出像素档」计费：面积 ≤ 2.25M 记 1K 档，> 2.25M 记 2K 档；本次统一 1536×2048（3.15M）→ 全部落在 2K 档。
- 火山侧按张计费，分辨率档位以控制台账单为准，本表单价为估算值，**需以开通日价目复核**。
- 阿里侧图片输入另计 $0.00275/张，本表未摊入（每张输出仅 1 张输入，影响 <1%）。

## 4. 盲测（真实感）

被试标为 AI：C, J, O, Q；整体可辨识率 **20%**。

| 模型 | 被认出 | 抽出示意 | 可辨识率 |
|---|---|---|---|
| Seedream 4.0 | 0 | 4 | 0% |
| Seedream 5.0 Pro | 1 | 4 | 25% |
| Seedream 5.0 Flash | 1 | 4 | 25% |
| Qwen-Image 3.0 | 1 | 4 | 25% |
| Qwen-Image 3.0 Pro | 1 | 4 | 25% |

- 被试（1 人，业务视角）标为 AI 的 4 张：C=5.0Pro 的镜面场景、Q=5.0Flash 的同题镜面、J=Qwen3.0 的贴面膜、O=Qwen3Pro 的空诊室。
- 误判集中在「镜面反射」「面部遮挡」「无人空间」三类结构性难点，没有一张是因为人脸的塑料感被认出。
- 1 人 20 张属初筛，结论是方向性的；扩到 5-8 人才能给置信区间。

## 5. 补充实验：多参考图 & 5.0 根因

| 实验 | 条件 | 题 | 相似度 | 耗时 |
|---|---|---|---|---|
| E1 | E1_ark4_1ref | p01 | 0.508 | 9.3s |
| E1 | E1_ark4_5ref | p01 | 0.683 | 30.7s |
| E1 | E1_qwen1ref | p01 | 0.824 | 55.9s |
| E1 | E1_qwen3ref | p01 | 0.607 | 75.5s |
| E1 | E1_ark4_1ref | p05 | 0.818 | 16.4s |
| E1 | E1_ark4_5ref | p05 | 0.791 | 30.9s |
| E1 | E1_qwen1ref | p05 | 0.897 | 66.9s |
| E1 | E1_qwen3ref | p05 | 0.854 | 81.0s |
| E2 | E2_str | p01 | 0.752 | 92.1s |
| E2 | E2_arr | p01 | 0.885 | 103.5s |
| E2 | E2_seq | p01 | - | 0.3s |
| E2 | E2_str | p05 | 0.850 | 108.6s |
| E2 | E2_arr | p05 | 0.819 | 106.4s |
| E2 | E2_seq | p05 | - | 0.3s |

**E1 多参考图（反直觉，但需重复确认）**：
- Qwen 3.0 加图到 3 张：p01 **0.824 → 0.607**、p05 0.897 → 0.854——多图融合疑似稀释身份。
- Seedream 4.0 加图到 5 张：p01 0.508 → 0.683、p05 0.818 → 0.791，方向不一致。
- 两边都显著变慢：Ark 9s→31s，Qwen 56s→76s。
- ⚠️ **同一配置跨次方差可达 0.2**（Ark p01 在 A/B 是 0.733，本轮 1ref 是 0.508），所以以上只是信号，需每条件重复 3 次取均值才可下定论。

**E2 5.0 换脸根因：不是参数，是方差**：
- `image` 传数组 vs 字符串：p01 0.752 → 0.885，p05 0.85 → 0.819，无一致收益。
- `sequential_image_generation` 在 5.0 上直接 **400 InvalidParameter**（与官方「不支持」一致），说明请求体没写错。
- 同配置在 A/B 与本次相差 ~0.2（p01：0.55 vs 0.752）→ 根因是**模型方差大**。排除 5.0 的理由从「会换脸」修正为「一致性不可控」。

## 6. 结论与边界

- **一致性最好**：Qwen-Image 3.0 Pro（剔除遮挡 0.888）
- **有效单张最便宜**：Seedream 5.0 Flash（¥0.1）
- **最快**：Seedream 4.0（P50 10.2s）

- **低于 0.45 视为漂移风险**（陌生人基线 0.341，阈值须另行标定）。
- **新版本不等于更适合锁脸**：5.0 系列的低分样本（0.234 / -0.005）说明按任务验收比按版本号选型重要。
- **遮挡脸样本不可用**：贴面膜/背面/侧脸会让 ArcFace 失效，生产管线里这类图必须走「人工复核」而不是「自动判死」。
- 真实感：需配合盲测（`out/ab/blind/sheet.png`），本报告不含主观分。
- **参考图策略按模型定**：Qwen 单图最稳，Seedream 4.0 难构图加图有收益；不要一套策略套两家。
- 5.0 的换脸是**方差问题**，不是参数问题；把它放进人物主线等于赌运气。
- 样本量：每臂 10 张，属方向性验证；失败率为零不代表长跑为零。
- 提示词里 `damp glowing fresh skin` 被字面执行为油光，属提示词缺陷，下一轮修正。