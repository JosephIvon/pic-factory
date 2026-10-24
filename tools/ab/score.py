"""打分：一致性（ArcFace 余弦）、失败率、延迟分位、成本，输出 Markdown 表。"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from face import FaceMetric

if hasattr(sys.stdout, "reconfigure"):
    # GBK 控制台下 ¥/中文会 UnicodeEncodeError；入口即锁 UTF-8，
    # 且打印一律放在 write_text 之后——崩也不能让 score.json 没落盘（P2 修复）。
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out" / "ab"

# 面部被遮挡的样本：ArcFace 测不准，不能拿来判模型好坏
OCCLUDED = {"p09_generic_aftercare"}


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((len(xs) - 1) * q))))
    return xs[k]


def main() -> None:
    raw = [json.loads(line) for line in (OUT / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    # 同一 (provider, prompt) 只保留最后一次，避免重跑把样本量算重
    latest: dict[tuple[str, str], dict] = {}
    for r in raw:
        latest[(r["provider"], r["prompt_id"])] = r
    rows = list(latest.values())
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    ref = ROOT / cfg["ref_image"]
    fm = FaceMetric()

    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["provider"], []).append(r)

    table = []
    for name, rs in by.items():
        ok = [r for r in rs if r["ok"]]
        i2i = [r for r in ok if r["mode"] == "i2i"]
        sims, fails = [], 0
        sims_clean: list[float] = []
        for r in i2i:
            p = Path(r["path"])
            if not p.exists():  # 历史台账里记的是服务端扩展名，实际可能被改成 .jpg
                alt = [c for c in p.parent.glob(p.stem + ".*") if c.is_file()]
                p = alt[0] if alt else p
            s = fm.similarity(ref, p)
            if s is None:
                fails += 1
            else:
                sims.append(s)
                if r["prompt_id"] not in OCCLUDED:
                    sims_clean.append(s)
        latency = [r["seconds"] for r in ok]
        cost = sum(r["cost_cny"] for r in rs)
        table.append({
            "provider": name,
            "label": rs[0]["label"],
            "n": len(rs),
            "ok": len(ok),
            "fail_rate": round(1 - len(ok) / max(len(rs), 1), 3),
            "sim_mean": round(statistics.mean(sims), 3) if sims else None,
            "sim_mean_clean": round(statistics.mean(sims_clean), 3) if sims_clean else None,
            "sim_min": round(min(sims), 3) if sims else None,
            "sim_std": round(statistics.pstdev(sims), 3) if len(sims) > 1 else None,
            "no_face": fails,
            "p50_s": round(pct(latency, 0.5), 1),
            "p95_s": round(pct(latency, 0.95), 1),
            "cost_cny": round(cost, 2),
            "cost_per_ok": round(cost / len(ok), 2) if ok else None,
        })

    # 先落盘再打印：stdout 编码/重定向异常绝不能影响 score.json 生成（P2 修复）
    (OUT / "score.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")

    print("| 模型 | 成功/总数 | 失败率 | 一致性均值 | 剔除遮挡 | 最低 | 标准差 | 无人脸 | P50 | P95 | 花费 | 有效单张成本 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in table:
        print(
            f"| {t['label']} | {t['ok']}/{t['n']} | {t['fail_rate']:.0%} | "
            f"{t['sim_mean'] if t['sim_mean'] is not None else '-'} | "
            f"{t['sim_mean_clean'] if t.get('sim_mean_clean') is not None else '-'} | "
            f"{t['sim_min'] if t['sim_min'] is not None else '-'} | "
            f"{t['sim_std'] if t['sim_std'] is not None else '-'} | {t['no_face']} | "
            f"{t['p50_s']}s | {t['p95_s']}s | ¥{t['cost_cny']} | "
            f"{'¥' + str(t['cost_per_ok']) if t['cost_per_ok'] is not None else '-'} |"
        )

    print(f"\n明细: {OUT / 'score.json'}")


if __name__ == "__main__":
    main()
