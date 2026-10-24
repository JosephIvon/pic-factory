# 用 pic-factory 生产度量（insightface + BGR + gallery 选脸）复算实验台 50 张。
#
# 出处与生命周期：输入来自**旧 DeepSeek 实验台**（源目录已于 2026-10-24 删除，
# 未入库的 85MB 原图随之删除）。因此这里**不再指向任何路径**：
# 结论以已入库的 docs/ab-evidence/model-sim-recompute.json 为准，脚本留档方法学。
# 真要重跑，必须显式 --lab-dir 指向你保留的实验台原图目录。
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="复算实验台 50 张一致性（生产度量口径）")
parser.add_argument(
    "--lab-dir",
    required=True,
    help="旧实验台 ab/ 目录（含 refs/persona_lin.png 与 out/ab/results.jsonl）；"
    "源仓库已删除，需自备原图；只有压缩版时无法重跑（分数以 JSON 为准）",
)
parser.add_argument(
    "--out",
    default=str(
        Path(__file__).resolve().parents[1] / "docs" / "ab-evidence" / "model-sim-recompute.json"
    ),
)
args = parser.parse_args()

LAB = Path(args.lab_dir)
REF = LAB / "refs" / "persona_lin.png"
LEDGER = LAB / "out" / "ab" / "results.jsonl"
OUT = Path(args.out)
if not LEDGER.exists():
    raise SystemExit(f"找不到实验台台账 {LEDGER}（源仓库可能已删除；结论看 {OUT}）")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aesthetics_pipeline.qc.face import similarity  # noqa: E402

rows = [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]
ok_rows = [r for r in rows if r.get("ok")]

def real_path(p: str) -> Path | None:
    q = Path(p)
    if q.exists():
        return q
    cands = list(q.parent.glob(q.stem + ".*"))
    return cands[0] if cands else None

per: dict[str, dict[str, list]] = {}
missing = []
for r in ok_rows:
    prov = r["provider"]
    img = real_path(r["path"])
    if img is None:
        missing.append(r["path"])
        continue
    slot = per.setdefault(prov, {})
    if r["prompt_id"] in slot:  # 重复扣费对：同 prompt 取后一次
        pass
    slot[r["prompt_id"]] = similarity(img, REF)

result = {}
for prov, slot in sorted(per.items()):
    scores = {k: v for k, v in slot.items()}
    vals = [v for v in scores.values() if v is not None]
    clean = {k: v for k, v in scores.items() if v is not None and "p09" not in k}
    cv = list(clean.values())
    result[prov] = {
        "n": len(slot),
        "sim_mean_all": round(sum(vals) / len(vals), 4) if vals else None,
        "sim_mean_clean_excl_p09": round(sum(cv) / len(cv), 4) if cv else None,
        "sim_min_all": min(vals) if vals else None,
        "per_prompt": scores,
    }
payload = {
    "method": "pic-factory production metric: insightface buffalo_l, BGR convert, gallery selection (max cos pair)",
    "reference": str(REF),
    "missing_files": missing,
    "models": result,
}
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("WROTE", OUT)
