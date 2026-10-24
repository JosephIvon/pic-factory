# tools/ab — 实验台归档（只读）

选型期（2026-10-24 前后）在 DeepSeek 源仓库跑的 A/B 实验代码原样迁入，
按复算裁决（见 `docs/项目史.md` §4）**降级为归档**：不参与生产流水线，不再修 bug，
`run_ab.py` 重试链等已知坑在生产侧由 http_api 计费断言 + `successful_keys` 挡住。

**已补全**：源实验台 `D:\workspace\picfactory` 于 2026-10-24 删除前，把当时没迁过来的
`calibrate.py`（Olivetti/同域阈值标定）、`check_docs.py`（文档数字对照）、
`verify_faces.py`（逐脸 det/sim 枚举）、`requirements.txt`、`config.json`、`.env.example`
一并归档到这里。现在实验代码**不再依赖源仓库**。

**数字口径已废弃**：本目录脚本（如 `report.py`）仍会生成"0.341 基线 / 0.45 漂移线"
的**实验台口径**报告（YuNet 按面积取脸，且 0.341 是 n=1 孤例）。引用一致性数字请只看：

- 逐模型生产口径复算值：`docs/ab-evidence/model-sim-recompute.json`
- 生产人脸闸阈值 T=0.57（同域标定）：`docs/ab-evidence/threshold-calibration-production.json`
- 裁决过程与订正链：`docs/项目史.md` §4

保留理由：复算方法可复核（`tools/recompute_lab_sims.py`、
`tools/recalibrate_face_threshold.py` 以此为对照对象），也是"错误如何被发现"的复盘案例。
证据来路与失效路径见 `docs/ab-evidence/PROVENANCE.md`。
