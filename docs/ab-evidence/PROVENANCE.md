# 选型证据的来路与现状（2026-10-24）

## 一句话

源实验台 `D:\workspace\picfactory` **已于 2026-10-24 删除**；本目录是那批实验的
唯一现场，实验代码归档在 `tools/ab/`。

## 这里有什么

| 路径 | 内容 | 能不能用来复算 |
|---|---|---|
| `lab-images/` | 50 张选型产出 + 参考图的**压缩版**（长边 1024、JPEG q88，66 件 / 5.65 MB） | ❌ 只供展示与人工复核 |
| `calib-samples/` | 人脸阈值标定的刻度样本（正样本 + 同域难负，54 件 / 4.69 MB） | ✅ 生产尺 T=0.57 可在其上复现 |
| `model-sim-recompute.json` | 每模型逐张相似度（生产度量：insightface buffalo_l + BGR + gallery 选脸） | ✅ 一致性数字的当前唯一真源 |
| `threshold-calibration-production.json` | T=0.57 的标定过程（正/负分布、间隙、样本量） | ✅ 阈值来源 |
| `face-crosscheck.json` | 两种取脸规则的逐张对照（审计"面积取脸"伪影用） | 🟡 数值可复核，但里面的绝对路径是历史记录 |
| `REPORT.md` / `score.json` / `results.jsonl` / `blind-result.json` | 实验台口径的报告与原始行 | 🟡 部分是旧口径，引用前先看 REPORT 顶部指路块 |

## 三个必须知道的边界

1. **分数以 JSON 为准，不以压缩图为准**。压缩图是给眼睛看的；重新算分会因为
   JPEG 重编码而略有差异。要复算请用 `calib-samples/`（无损语义）或原始行。
2. **`face-crosscheck.json` 里的 `D:\workspace\picfactory\...` 路径已失效**。
   保留原文是因为它是"当时从哪个文件算出来的"证据；文件本身在 `lab-images/` 里有压缩版。
3. **实验台代码是归档，不是生产代码**。`tools/ab/README.md` 写明：不参与流水线、
   不再修 bug，其报告口径（0.341 基线 / 0.45 漂移线）已废弃。

## 复算入口

源实验台的**未压缩原图没有入库**，所以下面两个脚本必须显式传 `--lab-dir`
指向你自备的实验台目录；日常引用数字直接看 JSON 即可，不必重跑：

```bash
python tools/recompute_lab_sims.py --lab-dir <实验台/ab>        # 逐模型相似度
python tools/recalibrate_face_threshold.py --lab-dir <实验台/ab> # 阈值标定
python tools/check_doc_numbers.py                               # 文档数字与证据一致性门禁
```

> 2026-10-24 起这两个脚本**不再有默认路径**：源目录已删，留一个指向不存在路径的
> 默认值只会制造"看起来能跑"的假象。`tools/ab/` 里是实验台代码归档，同样只读。
