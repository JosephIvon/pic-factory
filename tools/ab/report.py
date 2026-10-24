"""把台账 + 打分拼成一份可交付的对比报告（Markdown，图片用相对路径引用）。

用法：python report.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out" / "ab"

META = {
    "ark-4.0": {"refs": "10 张", "rpm": "未公开", "tier": "2K", "note": "4.0 快照 20260415"},
    "ark-5.0-flash": {"refs": "未公开", "rpm": "未公开", "tier": "2K", "note": "5.0 轻量档"},
    "ark-5.0-pro": {"refs": "未公开", "rpm": "未公开", "tier": "2K", "note": "5.0 旗舰，不支持 n>1/组图"},
    "qwen-3.0": {"refs": "1–3 张", "rpm": "20", "tier": "2K", "note": "单价不随尺寸变"},
    "qwen-3.0-pro": {"refs": "1–3 张", "rpm": "5", "tier": "2K", "note": "2K 档单价翻倍"},
}


def main() -> None:
    rows = [json.loads(x) for x in (OUT / "results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    latest: dict[tuple[str, str], dict] = {}
    for r in rows:
        latest[(r["provider"], r["prompt_id"])] = r
    rows = list(latest.values())

    score = {s["provider"]: s for s in json.loads((OUT / "score.json").read_text(encoding="utf-8"))}
    providers = [p for p in score]

    lines: list[str] = []
    lines.append("# 生图模型 A/B 实测报告\n")
    lines.append("> 同一角色卡（定妆照 1 张作参考图）、同一批 10 条英文画面 prompt、同一 3:4 画幅（1536×2048）、同一后处理（仅保留平台水印，盲测素材已裁掉底部水印）。\n")
    lines.append("> 一致性口径：输出图与定妆照的 ArcFace 余弦相似度（YuNet 5 点对齐 + buffalo_l w600k_r50）。陌生人基线实测 **0.341**。\n")

    lines.append("\n## 1. 汇总\n")
    lines.append("| 模型 | 成功 | 一致性均值 | 剔除遮挡 | 最低 | 标准差 | 无人脸 | P50 | P95 | 花费 | 有效单张成本 | 参考图上限 | RPM |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in providers:
        s = score[p]
        m = META.get(p, {})
        sim = f"{s['sim_mean']:.3f}" if s["sim_mean"] is not None else "-"
        clean = f"{s.get('sim_mean_clean'):.3f}" if s.get("sim_mean_clean") is not None else "-"
        mn = f"{s['sim_min']:.3f}" if s["sim_min"] is not None else "-"
        sd = f"{s['sim_std']:.3f}" if s["sim_std"] is not None else "-"
        cpo = f"¥{s['cost_per_ok']}" if s["cost_per_ok"] is not None else "-"
        lines.append(
            f"| {s['label']} | {s['ok']}/{s['n']} | {sim} | **{clean}** | {mn} | {sd} | {s['no_face']} | "
            f"{s['p50_s']}s | {s['p95_s']}s | ¥{s['cost_cny']} | {cpo} | {m.get('refs','-')} | {m.get('rpm','-')} |"
        )
    lines.append("\n> 「剔除遮挡」= 去掉 `p09`（贴面膜，面部被遮挡，ArcFace 不可测）后的均值。判断模型请以这列为主。")

    lines.append("\n## 2. 逐条对照\n")
    pids = sorted({r["prompt_id"] for r in rows})
    for pid in pids:
        lines.append(f"\n### {pid}\n")
        lines.append("| " + " | ".join(providers) + " |")
        lines.append("|" + "---|" * len(providers))
        cells = []
        for p in providers:
            r = next((x for x in rows if x["provider"] == p and x["prompt_id"] == pid), None)
            if not r:
                cells.append("—")
            elif not r["ok"]:
                cells.append("失败")
            else:
                fp = Path(r["path"])
                if not fp.exists():
                    alt = [c for c in fp.parent.glob(fp.stem + ".*") if c.is_file()]
                    fp = alt[0] if alt else fp
                rel = fp.relative_to(ROOT).as_posix() if fp.is_relative_to(ROOT) else str(fp)
                cells.append(f"![]({rel})")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n## 3. 成本口径\n")
    lines.append("- **有效单张成本** = 批次总花费 / 质检通过张数，把失败与返工都摊进去。")
    lines.append("- 阿里侧 3.0 系列按「输出像素档」计费：面积 ≤ 2.25M 记 1K 档，> 2.25M 记 2K 档；本次统一 1536×2048（3.15M）→ 全部落在 2K 档。")
    lines.append("- 火山侧按张计费，分辨率档位以控制台账单为准，本表单价为估算值，**需以开通日价目复核**。")
    lines.append("- 阿里侧图片输入另计 $0.00275/张，本表未摊入（每张输出仅 1 张输入，影响 <1%）。")

    lines.append("\n## 4. 盲测（真实感）\n")
    blind = OUT / "blind" / "result.json"
    if blind.exists():
        b = json.loads(blind.read_text(encoding="utf-8"))
        lines.append(f"被试标为 AI：{', '.join(b['picked'])}；整体可辨识率 **{b['overall_rate']:.0%}**。\n")
        lines.append("| 模型 | 被认出 | 抽出示意 | 可辨识率 |")
        lines.append("|---|---|---|---|")
        for label, v in b["by_model"].items():
            lines.append(f"| {label} | {v['hit']} | {v['total']} | {v['hit'] / v['total']:.0%} |")
        lines.append("")
        for n in b.get("notes", []):
            lines.append(f"- {n}")
    else:
        lines.append("（尚未做盲测，跑 `blind.py` 后把字母传给 `blind_score.py`）")

    lines.append("\n## 5. 补充实验：多参考图 & 5.0 根因\n")
    exp_file = OUT / "exp" / "results.json"
    if exp_file.exists():
        exp = json.loads(exp_file.read_text(encoding="utf-8"))
        lines.append("| 实验 | 条件 | 题 | 相似度 | 耗时 |")
        lines.append("|---|---|---|---|---|")
        for r in exp:
            sim = "-" if r["sim"] is None else f"{r['sim']:.3f}"
            lines.append(f"| {r['exp']} | {r['tag']} | {r['prompt']} | {sim} | {r['sec']}s |")
        lines.append("")
        lines.append("**E1 多参考图（反直觉，但需重复确认）**：")
        lines.append("- Qwen 3.0 加图到 3 张：p01 **0.824 → 0.607**、p05 0.897 → 0.854——多图融合疑似稀释身份。")
        lines.append("- Seedream 4.0 加图到 5 张：p01 0.508 → 0.683、p05 0.818 → 0.791，方向不一致。")
        lines.append("- 两边都显著变慢：Ark 9s→31s，Qwen 56s→76s。")
        lines.append("- ⚠️ **同一配置跨次方差可达 0.2**（Ark p01 在 A/B 是 0.733，本轮 1ref 是 0.508），所以以上只是信号，需每条件重复 3 次取均值才可下定论。")
        lines.append("")
        lines.append("**E2 5.0 换脸根因：不是参数，是方差**：")
        lines.append("- `image` 传数组 vs 字符串：p01 0.752 → 0.885，p05 0.85 → 0.819，无一致收益。")
        lines.append("- `sequential_image_generation` 在 5.0 上直接 **400 InvalidParameter**（与官方「不支持」一致），说明请求体没写错。")
        lines.append("- 同配置在 A/B 与本次相差 ~0.2（p01：0.55 vs 0.752）→ 根因是**模型方差大**。排除 5.0 的理由从「会换脸」修正为「一致性不可控」。")
    else:
        lines.append("（未跑补充实验）")

    lines.append("\n## 6. 结论与边界\n")
    valid = [(p, s) for p, s in score.items() if s.get("sim_mean_clean") is not None]
    if valid:
        best = max(valid, key=lambda x: x[1]["sim_mean_clean"])
        cheap = min(valid, key=lambda x: x[1]["cost_per_ok"] or 9e9)
        fast = min(valid, key=lambda x: x[1]["p50_s"])
        lines.append(f"- **一致性最好**：{best[1]['label']}（剔除遮挡 {best[1]['sim_mean_clean']:.3f}）")
        lines.append(f"- **有效单张最便宜**：{cheap[1]['label']}（¥{cheap[1]['cost_per_ok']}）")
        lines.append(f"- **最快**：{fast[1]['label']}（P50 {fast[1]['p50_s']}s）")
    lines.append("")
    lines.append("- **低于 0.45 视为漂移风险**（陌生人基线 0.341，阈值须另行标定）。")
    lines.append("- **新版本不等于更适合锁脸**：5.0 系列的低分样本（0.234 / -0.005）说明按任务验收比按版本号选型重要。")
    lines.append("- **遮挡脸样本不可用**：贴面膜/背面/侧脸会让 ArcFace 失效，生产管线里这类图必须走「人工复核」而不是「自动判死」。")
    lines.append("- 真实感：需配合盲测（`out/ab/blind/sheet.png`），本报告不含主观分。")
    lines.append("- **参考图策略按模型定**：Qwen 单图最稳，Seedream 4.0 难构图加图有收益；不要一套策略套两家。")
    lines.append("- 5.0 的换脸是**方差问题**，不是参数问题；把它放进人物主线等于赌运气。")
    lines.append("- 样本量：每臂 10 张，属方向性验证；失败率为零不代表长跑为零。")
    lines.append("- 提示词里 `damp glowing fresh skin` 被字面执行为油光，属提示词缺陷，下一轮修正。")

    report = OUT / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告: {report}")


if __name__ == "__main__":
    main()
