"""阈值标定 v2：用**同域**负样本定阈值，跨域真人只做 sanity check。

样本构成：
  P   域内正样本  45 张：定妆照 vs 本仓库生成图（同一角色，intended same）
  N1  域内负样本   7 张：定妆照 vs batch 图（同风格但**画出来是别人**——最难的一类负样本）
  N2  跨域负样本 100 张：定妆照 vs Olivetti 真人（64×64 灰度，域差大，只作 sanity）

阈值规则：取「正样本最低分（剔除遮挡）」与「负样本最高分」的中点；同时报告该点的 TPR/FAR。

用法：python calibrate.py            # 首次跑会算 Olivetti 嵌入并缓存
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np

from face import FaceMetric

ROOT = Path(__file__).resolve().parent
FACES = ROOT / "_calib" / "faces"
IMPOSTORS = ROOT / "_calib" / "impostors"
REF = ROOT / "refs" / "persona_lin.png"
CACHE = ROOT / "_calib" / "olivetti_emb.npz"
OUT = ROOT / "out" / "ab" / "calibration.json"
OCCLUDED = "p09_generic_aftercare"


def generated_pairs(fm: FaceMetric, ref_v: np.ndarray) -> list[tuple[str, float]]:
    rows = [json.loads(x) for x in (ROOT / "out/ab/results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    latest = {}
    for r in rows:
        latest[(r["provider"], r["prompt_id"])] = r
    out = []
    for (prov, pid), r in sorted(latest.items()):
        if not r["ok"] or r["mode"] != "i2i":
            continue
        p = Path(r["path"])
        if not p.exists():
            alt = list(p.parent.glob(p.stem + ".*"))
            if not alt:
                continue
            p = alt[0]
        v = fm.embedding(p)
        if v is not None:
            out.append((pid, float(np.dot(v, ref_v))))
    return out


def olivetti_sims(fm: FaceMetric, ref_v: np.ndarray, limit: int = 100) -> list[float]:
    if CACHE.exists():
        return list(np.load(CACHE, allow_pickle=False)["sims"])
    files = sorted(FACES.glob("*.png"))[:limit]
    sims = []
    for f in files:
        v = fm.embedding(f)
        if v is not None:
            sims.append(float(np.dot(v, ref_v)))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, sims=np.array(sims, dtype=np.float32))
    return sims


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fm = FaceMetric()
    ref_v = np.asarray(fm.embedding(REF), dtype=np.float32)

    gen = generated_pairs(fm, ref_v)
    pos_all = [s for _, s in gen]
    pos_clean = [s for pid, s in gen if pid != OCCLUDED]

    imp = []
    for p in sorted(IMPOSTORS.glob("*.jpg")):
        v = fm.embedding(p)
        if v is not None:
            imp.append(float(np.dot(v, ref_v)))

    oliv = olivetti_sims(fm, ref_v)

    lo = max(imp) if imp else 0.0
    hi = min(pos_clean)
    t = round((lo + hi) / 2, 2)
    tpr = sum(1 for s in pos_clean if s >= t) / len(pos_clean)
    far_in = sum(1 for s in imp if s >= t) / len(imp) if imp else 0.0
    far_out = sum(1 for s in oliv if s >= t) / len(oliv) if oliv else 0.0

    summary = {
        "rule": "T = (同域负样本最高分 + 正样本最低分(剔遮挡)) / 2",
        "positive_n": len(pos_all),
        "positive_mean": round(statistics.mean(pos_all), 3),
        "positive_min_clean": round(hi, 3),
        "positive_min_all": round(min(pos_all), 3),
        "positive_occluded_pid": OCCLUDED,
        "indomain_negative_n": len(imp),
        "indomain_negative_scores": [round(x, 3) for x in sorted(imp)],
        "crossdomain_negative_n": len(oliv),
        "crossdomain_negative_max": round(max(oliv), 3) if oliv else None,
        "threshold": t,
        "tpr_at_threshold": round(tpr, 3),
        "far_indomain": round(far_in, 3),
        "far_crossdomain": round(far_out, 3),
        "caveat": "同域负样本仅 7 个，FAR 的点估计没有统计意义；要声称 FAR≤2% 需 ≥150 个同域负样本（约 ¥15 生成成本）。",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"正样本 n={len(pos_all)}  mean={summary['positive_mean']}  min(剔遮挡)={summary['positive_min_clean']}  min(含遮挡)={summary['positive_min_all']}")
    print(f"同域负样本 n={len(imp)}  max={round(lo,3)}  分数={summary['indomain_negative_scores']}")
    print(f"跨域真人 n={len(oliv)}  max={summary['crossdomain_negative_max']}")
    print(f"\n=> T={t}  TPR={tpr:.1%}  FAR(同域)={far_in:.0%}  FAR(跨域)={far_out:.0%}")
    print(f"写入 {OUT}")


if __name__ == "__main__":
    main()
