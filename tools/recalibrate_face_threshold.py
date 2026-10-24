"""生产度量人脸阈值标定：把 qc/face.py 的"0.35 未标定"换成有样本撑腰的数。

背景（实验台标定史）：`tools/ab` 实验台用 YuNet+w600k 尺标定出 T=0.46
（45 域内正 vs 7 同域难负取间隙中点）。方法论采纳——**阈值只从同域难负样本
的间隙来，跨域标定只会给出假线**（Olivetti 真人 100 张在本尺下最高仅 0.083）；
但 0.46 这个**数**不能移植：P5 对账两尺 mean_diff≈0.04，而那条间隙总共才 0.042，
偏移与间隙同量级，照搬等于没标定。本工具用**生产尺**（insightface buffalo_l，
gallery 语义取最相似脸对）对同一批样本重算。

**重算结果（写进 JSON，这里是解读）**：生产尺下全人群间隙是**负的**（难负最高
0.4918 > 正样本剔遮挡最低 0.4566）——但拉低间隙的两张图全部来自 ark-5.0-pro，
即档位表早已禁入人物线的模型。剔除禁模型后间隙 0.149（负 0.4918 / 合格正最低
0.6406），中点 **T=0.57**。闸与档位表互证：禁令的依据（低分+大方差）不能反过来
把闸线压低。这正是"数字不可移植"的实锤——lab 尺上的间隙只有 0.042（与 P5 度量
偏移同量级，移植必错），生产尺上排除禁模型后是 0.149 的真宽间隙，两个尺子讲的
故事不同，必须以生产尺为准。

样本口径与实验台 calibrate.py 严格对齐，否则两边数字没有可比性：
  * 正样本 = results.jsonl 里 (provider, prompt_id) 最新一条、ok、mode=i2i；
    p09_generic_aftercare（遮挡）不计入"最低分"，但仍进均值；
  * 同域难负 = _calib/impostors/*.jpg（同风格但画出来是别人——最难的一类）；
  * 角色卡锚图 = refs/persona_lin.png。

**解耦声明**：`--lab-dir` 指向源实验台目录（外部目录，不随本仓库分发），它需要含
`out/ab/results.jsonl`、`_calib/impostors/`、`refs/persona_lin.png` 三样。
本仓库自带的是 `--pack` 产出的**压缩样本包** `docs/ab-evidence/calib-samples/`
（positives/ + impostors/ + ref_lin.jpg + manifest.json 校验和），JSON 的 `pack` 段
记的就是"在没有实验台的机器上、只用入库压缩件复算"得到的分数与偏移——刻度可复核，
但整工具重跑需要自备实验台目录。

局限如实入账：同域负样本 n=7，FAR 点估计无统计意义；要声称 FAR≤2% 需
≥150 个同域负样本（已列入后续计划）。

用法：
  python tools/recalibrate_face_threshold.py --lab-dir <实验台目录>        # 原图标定
  python tools/recalibrate_face_threshold.py --lab-dir <实验台目录> --pack # 标定 + 生成入库样本包
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OCCLUDED = "p09_generic_aftercare"  # 遮挡样本：进均值、不进"最低分"红线
PACK_LONG_EDGE = 896                # ≥ det_size(640)，压到这一档对分数影响实测入 JSON
PACK_JPEG_Q = 90

# 实验台 provider 键 → 注册表 model_id（排除口径与 model_choice._PERSONA_BANNED 对齐）
LAB_TO_MODEL = {
    "ark-4.0": "doubao-seedream-4-0",
    "ark-5.0-pro": "doubao-seedream-5-0-pro",
    "ark-5.0-flash": "doubao-seedream-5-0-flash",
    "qwen-3.0": "qwen-image-3.0",
    "qwen-3.0-pro": "qwen-image-3.0-pro",
}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _latest_i2i(results: Path) -> list[dict]:
    """与实验台 calibrate.py::generated_pairs 同一套选取逻辑（含 stem 兜底）。"""
    rows = [json.loads(x) for x in results.read_text(encoding="utf-8").splitlines() if x.strip()]
    latest: dict[tuple[str, str], dict] = {}
    for r in rows:
        latest[(r["provider"], r["prompt_id"])] = r
    picked = []
    for (prov, pid), r in sorted(latest.items()):
        if not r["ok"] or r["mode"] != "i2i":
            continue
        p = Path(r["path"]) if r.get("path") else None
        if p is None or not p.exists():
            alt = list(p.parent.glob(p.stem + ".*")) if p and p.parent.exists() else []
            if not alt:
                continue
            p = alt[0]
        picked.append({"provider": prov, "pid": pid, "path": p})
    return picked


def _compress(src: Path, dst: Path) -> None:
    from PIL import Image

    im = Image.open(src).convert("RGB")
    if max(im.size) > PACK_LONG_EDGE:
        im.thumbnail((PACK_LONG_EDGE, PACK_LONG_EDGE), Image.LANCZOS)
    im.save(dst, "JPEG", quality=PACK_JPEG_Q)


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lab-dir", default=None,
                    help="实验台根目录（含 out/ab/results.jsonl、_calib/impostors、refs）；"
                         "本仓库只带压缩样本包，整工具重跑需自备该目录")
    ap.add_argument("--out", default=str(REPO / "docs/ab-evidence/threshold-calibration-production.json"))
    ap.add_argument("--pack", action="store_true", help="同时生成入库样本包（压缩件）")
    ap.add_argument("--pack-dir", default=str(REPO / "docs/ab-evidence/calib-samples"))
    args = ap.parse_args()

    if not args.lab_dir:
        raise SystemExit(
            "--lab-dir 必填：指向含 out/ab/results.jsonl、_calib/impostors、refs 的实验台目录。"
            "入库的 docs/ab-evidence/calib-samples/ 是压缩样本包（目录结构不同），"
            "只能用于复核 JSON 里 pack 段的分数，不能直接喂给本工具。"
        )
    lab = Path(args.lab_dir)
    from aesthetics_pipeline.model_choice import _PERSONA_BANNED
    from aesthetics_pipeline.qc.face import FaceGate, similarity

    ref = lab / "refs" / "persona_lin.png"
    if not ref.exists():
        raise SystemExit(f"缺少角色卡锚图：{ref}（源仓库已删则改用 --lab-dir 指向样本包）")

    pos = _latest_i2i(lab / "out" / "ab" / "results.jsonl")
    pos_scores = []
    for item in pos:
        s = similarity(item["path"], ref)
        if s is not None:
            pos_scores.append({**item, "sim": s, "sha": _sha256(item["path"])})
    detected = [p for p in pos_scores if p["sim"] is not None]
    clean = [p["sim"] for p in detected if p["pid"] != OCCLUDED]
    if not clean:
        raise SystemExit("正样本一张都没检到脸，标定无从谈起")

    imp_scores = []
    for p in sorted((lab / "_calib" / "impostors").glob("*.jpg")):
        s = similarity(p, ref)
        if s is not None:
            imp_scores.append({"name": p.name, "sim": s, "sha": _sha256(p)})

    banned_lab = {prov for prov, mid in LAB_TO_MODEL.items() if mid in _PERSONA_BANNED}
    elig_clean = [p["sim"] for p in detected if p["pid"] != OCCLUDED and p["provider"] not in banned_lab]

    lo = max(x["sim"] for x in imp_scores)
    hi_full = min(clean)
    hi = min(elig_clean)
    # 阈值从**人物合格模型**的正样本取间隙——闸门服务人物主线，而被档位表
    # 禁入人物线的模型（5.0-Pro）本就不该往闸里送图；它贡献的低分（0.4566/0.5639）
    # 恰是禁令的依据，不能反过来把闸线压低。全人群视图如实并排放进 JSON。
    t = round((lo + hi) / 2, 2)
    gate = FaceGate(threshold=t)

    summary = {
        "method": "insightface buffalo_l (BGR) gallery-max-cos，正样本=lab results.jsonl 最新 i2i，负样本=同域难负 impostors",
        "metric": "production (qc/face.py similarity) —— 与 lab T=0.46 不同尺，不可互相引用",
        "rule": "T = (同域负最高 + 人物合格模型正样本最低(剔遮挡)) / 2；全人群视图另报",
        "occluded_pid": OCCLUDED,
        "persona_banned_providers": sorted(banned_lab),
        "positive_n": len(detected),
        "positive_n_eligible": len(elig_clean),
        "positive_mean": round(statistics.mean([p["sim"] for p in detected]), 3),
        "positive_min_clean_eligible": round(hi, 4),
        "positive_min_clean_full": round(hi_full, 4),
        "positive_min_all": round(min(p["sim"] for p in detected), 4),
        "indomain_negative_n": len(imp_scores),
        "indomain_negative_max": round(lo, 4),
        "threshold": t,
        "tpr_at_threshold": sum(1 for s in elig_clean if s >= t) / len(elig_clean),
        "far_indomain": sum(1 for x in imp_scores if x["sim"] >= t) / len(imp_scores),
        "margin": round(hi - lo, 4),
        "full_population_overlap": {
            "positive_min_clean": round(hi_full, 4),
            "note": (
                "全人群间隙为负（-0.035）：低于难负最高 0.4918 的两张正样本全部来自 "
                "ark-5.0-pro（0.4566/0.5639）——即档位表已禁入人物线的模型。"
                "闸与档位表互证：禁令的根据不能反过来稀释闸线。150 同域负样本留待开发完成后补。"
            ),
        },
        "caveat": (
            "同域负样本 n=7，FAR 点估计无统计意义；FAR≤2% 需 ≥150 负样本（用户已裁决：开发完成后处理）。"
            "跨域（Olivetti）只作 sanity，不进阈值推导——跨域标定给假线。"
        ),
        "positives": [{"provider": p["provider"], "pid": p["pid"], "sim": p["sim"], "sha256": p["sha"]} for p in detected],
        "impostors": [{"name": x["name"], "sim": x["sim"], "sha256": x["sha"]} for x in imp_scores],
    }

    if args.pack:
        pack = Path(args.pack_dir)
        if pack.exists():
            shutil.rmtree(pack)
        (pack / "positives").mkdir(parents=True)
        (pack / "impostors").mkdir(parents=True)
        _compress(ref, pack / "ref_lin.jpg")
        for p in detected:
            _compress(p["path"], pack / "positives" / f"{p['provider']}__{p['pid']}.jpg")
        for x in imp_scores:
            _compress(lab / "_calib" / "impostors" / x["name"], pack / "impostors" / x["name"])

        # 压缩件复算：证明"刻度"在入库样本上仍可复现，偏移量如实入账
        ref_pack = pack / "ref_lin.jpg"
        pack_pos = []
        for f in sorted((pack / "positives").iterdir()):
            prov, pid = f.stem.split("__", 1)
            pack_pos.append({"provider": prov, "pid": pid, "sim": similarity(f, ref_pack)})
        pack_imp = [{"name": f.name, "sim": similarity(f, ref_pack)} for f in sorted((pack / "impostors").iterdir())]
        pp_clean = [
            x["sim"] for x in pack_pos
            if x["sim"] is not None and x["pid"] != OCCLUDED and x["provider"] not in banned_lab
        ]
        pi = [x["sim"] for x in pack_imp if x["sim"] is not None]
        summary["pack"] = {
            "long_edge": PACK_LONG_EDGE, "jpeg_q": PACK_JPEG_Q,
            "positive_min_clean_eligible": round(min(pp_clean), 4),
            "indomain_negative_max": round(max(pi), 4),
            "threshold_midpoint": round((min(pp_clean) + max(pi)) / 2, 2),
            "positives": pack_pos, "impostors": pack_imp,
            "note": "压缩件复算（人物合格模型口径）：阈值若与原图版不同，以本包为准重跑本工具取 threshold",
        }
        manifest = {
            "provenance": "由 --lab-dir 指定的外部实验台目录产出（未入库）；本包是唯一可随仓库长期存的刻度样本",
            "method": summary["method"],
            "files": {str(f.relative_to(pack)): _sha256(f) for f in sorted(pack.rglob("*")) if f.is_file()},
        }
        (pack / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"样本包已写入 {pack}（{sum(f.stat().st_size for f in pack.rglob('*') if f.is_file())/1e6:.1f} MB）")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"正样本 n={summary['positive_n']}（人物合格 {summary['positive_n_eligible']}） "
          f"mean={summary['positive_mean']} min合格={summary['positive_min_clean_eligible']} "
          f"min全量={summary['positive_min_clean_full']}")
    print(f"同域难负 n={summary['indomain_negative_n']} max={summary['indomain_negative_max']}")
    print(f"=> 生产尺 T={t}（间隙 margin={summary['margin']}）"
          f" TPR={summary['tpr_at_threshold']:.1%} FAR={summary['far_indomain']:.0%}  "
          f"env AP_FACE_THRESHOLD={t}")
    if "pack" in summary:
        print(f"压缩件复算 T={summary['pack']['threshold_midpoint']} "
              f"（与原图版差 {abs(t - summary['pack']['threshold_midpoint']):.2f}）")
    print(f"写入 {out}")
    _ = gate  # FaceGate 构造顺带预热模型，失败时提前炸出来而不是写 JSON 时


if __name__ == "__main__":
    main()
