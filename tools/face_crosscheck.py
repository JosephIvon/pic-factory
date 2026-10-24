"""P5 收口：两套人脸度量对账（外部锚点证据）。

同一批已生成图（tools/ab 实验台产出的 50 张）上**双跑**两套度量：

1. **生产尺子** insightface buffalo_l —— 直接复用上游生产模块
   `src/aesthetics_pipeline/qc/face.py` 的 embedding()，与一致性闸门口径完全一致；
2. **校验尺子** YuNet 检测 + w600k_r50.onnx 识别 —— 复用 DeepSeek 实验台
   `tools/ab/face.py` 的 FaceMetric（另一套自写度量）。

两者对同一张图 vs 角色锚图（tools/ab/refs/persona_lin.png）各出一个余弦分，
输出逐图对比 + 汇总（均值差、Pearson/Spearman、最大偏差、
0.35/0.5/0.58 各阈值下"是否同人"的分歧率），证明阈值标定只认一把尺子。

裁决背景：生产只留 insightface；YuNet 版降级为校验工具（就是本脚本）。
若 YuNet 权重在本机取不到，汇总里如实写 "yunet_unavailable"，不造假第二列。

用法（须用装了 insightface/cv2/onnxruntime 的 venv）：
    D:/Tools/venvs/aesthetics/Scripts/python.exe tools/face_crosscheck.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # GBK 控制台防崩（P2 同类缺陷）

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "docs" / "ab-evidence" / "results.jsonl"
ANCHOR = REPO / "tools" / "ab" / "refs" / "persona_lin.png"
OUT = REPO / "docs" / "ab-evidence" / "face-crosscheck.json"

# 生产口径与实验台口径都按源码引用，不改其行为
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "ab"))

THRESHOLDS = [0.35, 0.5, 0.58]


# ---------- 度量 A：insightface buffalo_l（生产尺子） ----------
def insightface_factory():
    import numpy as np

    from aesthetics_pipeline.qc import face as prod_face

    def embed(path: str | Path):
        e = prod_face.embedding(path)  # list[float] | None；None=检不到脸/模型不可用
        return None if e is None else np.asarray(e, dtype=np.float64)

    available = embed(ANCHOR) is not None
    return embed, available, "insightface buffalo_l (src/aesthetics_pipeline/qc/face.py)"


# ---------- 度量 B：YuNet + w600k_r50（实验台校验尺子） ----------
def yunet_factory():
    from face import ARCFACE, YUNET, FaceMetric  # tools/ab/face.py

    if not ARCFACE.exists():
        raise RuntimeError(f"缺少 ArcFace 权重: {ARCFACE}")
    if not YUNET.exists():
        # YuNet 权重已随仓库入库（tools/ab/），不再从外部实验台目录拷贝——
        # 本仓库与两个源仓库完全解耦，源仓库删除后此工具必须照跑。
        raise RuntimeError(f"缺少 YuNet 权重: {YUNET}（应在仓库内，检出损坏？）")
    metric = FaceMetric()

    def embed(path: str | Path):
        return metric.embedding(path)  # np.ndarray | None

    return embed, "YuNet 检测 + w600k_r50.onnx (tools/ab/face.py FaceMetric)"


def cos(a, b) -> float | None:
    if a is None or b is None:
        return None
    return round(float(sum(x * y for x, y in zip(a, b))), 4)


def rank(vals: list[float]) -> list[float]:
    """平均名次（处理并列），用于 Spearman。"""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
        return None
    return statistics.correlation(xs, ys)


def main() -> int:
    # 1) 定位图片：同一 (provider, prompt_id) 只留最后一条（与实验台 score.py 同规则）
    raw = [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    latest: dict[tuple[str, str], dict] = {}
    for r in raw:
        latest[(r["provider"], r["prompt_id"])] = r
    todo = []
    for r in latest.values():
        p = Path(r.get("path") or "")
        if not r.get("ok") or not p.exists():  # 历史台账记 .png，实际可能被改成 .jpg
            alt = [c for c in p.parent.glob(p.stem + ".*") if c.is_file()] if p.parent.exists() else []
            p = alt[0] if alt else p
        if p.exists():
            todo.append((r, p))

    print(f"results.jsonl: {len(raw)} 行，去重后 {len(latest)} 张，可定位 {len(todo)} 张")

    # 2) 加载两套度量（各自失败互不牵连）
    ins_embed, ins_ok, ins_desc = insightface_factory()
    if not ins_ok:
        print("错误：insightface 锚图 embedding 取不到（模型或锚图不可用），无法出分")
        return 2
    ins_anchor = ins_embed(ANCHOR)

    yunet_desc = "YuNet 检测 + w600k_r50.onnx (tools/ab/face.py FaceMetric)"
    yunet_status = "ok"
    yunet_reason = ""
    yunet_embed = yunet_anchor = None
    try:
        yunet_embed, yunet_desc = yunet_factory()
        yunet_anchor = yunet_embed(ANCHOR)
        if yunet_anchor is None:
            yunet_status = "yunet_unavailable"
            yunet_reason = "锚图上 YuNet 检不到人脸"
            yunet_embed = None
    except Exception as e:  # 权重缺失/依赖缺失：如实记录，不造假第二列
        yunet_status = "yunet_unavailable"
        yunet_reason = f"{type(e).__name__}: {e}"
        yunet_embed = None

    # 3) 双跑
    images = []
    n_ins = n_yun = n_pair = 0
    for r, p in todo:
        s_ins = cos(ins_anchor, ins_embed(p))
        s_yun = cos(yunet_anchor, yunet_embed(p)) if yunet_embed else None
        n_ins += s_ins is not None
        n_yun += s_yun is not None
        n_pair += s_ins is not None and s_yun is not None
        images.append({
            "provider": r["provider"],
            "model": r.get("model"),
            "prompt_id": r["prompt_id"],
            "mode": r.get("mode"),
            "path": str(p),
            "insightface_sim": s_ins,
            "insightface_no_face": s_ins is None,
            "yunet_sim": s_yun if yunet_embed else None,
            "yunet_no_face": (s_yun is None) if yunet_embed else None,
        })
        print(f"  {r['provider']}/{r['prompt_id']}: insightface={s_ins} yunet={s_yun if yunet_embed else 'n/a'}")

    # 4) 汇总：只对两套都出分的图算统计
    paired = [im for im in images if im["insightface_sim"] is not None and im["yunet_sim"] is not None]
    summary: dict = {
        "records_total": len(latest),
        "records_located": len(todo),
        "insightface_scored": n_ins,
        "yunet_status": yunet_status,
        "yunet_reason": yunet_reason or None,
        "yunet_scored": n_yun if yunet_embed else 0,
        "both_scored": n_pair,
    }
    if paired:
        xs = [im["insightface_sim"] for im in paired]
        ys = [im["yunet_sim"] for im in paired]
        diffs = [x - y for x, y in zip(xs, ys)]
        summary.update({
            "insightface_mean": round(statistics.mean(xs), 4),
            "yunet_mean": round(statistics.mean(ys), 4),
            "mean_diff": round(statistics.mean(diffs), 4),
            "mean_abs_diff": round(statistics.mean([abs(d) for d in diffs]), 4),
            "max_abs_diff": round(max(abs(d) for d in diffs), 4),
            "pearson_r": round(pearson(xs, ys), 4),
            "spearman_rho": round(pearson(rank(xs), rank(ys)), 4) if len(xs) > 2 else None,
        })
        disagree = {}
        for thr in THRESHOLDS:
            dn = sum(1 for x, y in zip(xs, ys) if (x >= thr) != (y >= thr))
            disagree[str(thr)] = {
                "pairs": len(paired),
                "disagree": dn,
                "disagree_rate": round(dn / len(paired), 4),
            }
        summary["same_person_disagreement"] = disagree
        summary["null_notes"] = {
            "insightface_null": len(todo) - n_ins,
            "yunet_null": (len(todo) - n_yun) if yunet_embed else None,
            "说明": "检不到脸=null，按两套均有分样本统计；两套分数不可直接互换阈值",
        }

    payload = {
        "task": "P5 两套人脸度量对账（外部锚点）",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "anchor": str(ANCHOR),
        "anchor_note": "tools/ab/refs/persona_lin.png，与上游 evidence/01_t2i_baseline.png md5 相同（18009cc9a87703d7b07e15c10c1dc775）",
        "metrics": {"insightface": ins_desc, "yunet": yunet_desc},
        "images": images,
        "summary": summary,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 汇总（两套均有分 n={len(paired)}）===")
    for k in ("insightface_scored", "yunet_status", "yunet_scored", "both_scored",
              "insightface_mean", "yunet_mean", "mean_diff", "mean_abs_diff",
              "max_abs_diff", "pearson_r", "spearman_rho"):
        if k in summary:
            print(f"{k}: {summary[k]}")
    for thr in THRESHOLDS:
        if "same_person_disagreement" in summary:
            d = summary["same_person_disagreement"][str(thr)]
            print(f"阈值 {thr}: 分歧 {d['disagree']}/{d['pairs']} ({d['disagree_rate']:.1%})")
    print(f"\n已写出: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
