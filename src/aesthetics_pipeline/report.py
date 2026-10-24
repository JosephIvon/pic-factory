"""报表：拼图 + CSV + JSON。

拼图（contact sheet）是演示里最直观的东西：一眼看出哪张被拒、哪组人物一致。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .config import OUT_DIR, unit_cost_for

THUMB = 320
LABEL_H = 26
COLS = 4
BG = (24, 24, 28)
FG = (232, 232, 236)

# 中文字体候选：PIL 默认位图字体渲染不了中文，标签会变成方块。
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyh.ttf",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def load_font(size: int):
    """拿一个能渲染中文的字体；找不到就退回默认位图字体。"""
    from PIL import ImageFont

    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def collage(
    items: list[tuple[str, Path]],
    out_path: Path,
    cols: int = COLS,
    title: str | None = None,
) -> Path | None:
    """items = [(标签, 图片路径)]。生成带标签的拼图。"""
    items = [(label, p) for label, p in items if p and Path(p).exists()]
    if not items:
        return None

    rows = (len(items) + cols - 1) // cols
    head = 40 if title else 0
    canvas = Image.new(
        "RGB",
        (cols * THUMB, rows * (THUMB + LABEL_H) + head),
        BG,
    )
    draw = ImageDraw.Draw(canvas)
    font = load_font(15)
    title_font = load_font(19)
    if title:
        draw.text((12, 10), title, fill=FG, font=title_font)

    for index, (label, path) in enumerate(items):
        col = index % cols
        row = index // cols
        x = col * THUMB
        y = head + row * (THUMB + LABEL_H)
        try:
            with Image.open(path) as img:
                thumb = img.convert("RGB").copy()
                thumb.thumbnail((THUMB - 8, THUMB - 8), Image.Resampling.LANCZOS)
        except Exception:
            continue
        offset = ((THUMB - thumb.width) // 2, (THUMB - thumb.height) // 2)
        canvas.paste(thumb, (x + offset[0], y + offset[1]))
        draw.text((x + 6, y + THUMB + 5), str(label)[:42], fill=FG, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def export_csv(rows: list[dict[str, Any]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "idempotency_key", "city", "project", "theme", "channel", "character",
        "route", "provider", "route_rule_id", "route_reason",
        "file_name", "local_path", "swapped_path",
        "gen_seconds", "est_cost",
        "qc_verdict", "face_similarity", "face_similarity_after", "text_found",
        "adopted", "adopted_at", "adopt_note", "created_at",
    ]
    # 表头写清耗时语义：API 后端是批内均摊，本地栈是每张实测
    header_alias = {
        "gen_seconds": "gen_seconds（API=批内均摊 / 本地栈=单张实测）",
    }
    with out_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writerow({f: header_alias.get(f, f) for f in fields})
        writer.writerows(rows)
    return out_path


def cost_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按后端汇总成本。

    关键口径是**单张合格图综合成本 = 总花费 / 合格张数**：
    被拒收的图同样花了钱（也占了机时），必须算进去，否则低成本后端会因为
    "拒得多所以看起来便宜"而失真。返工次数 × 单次成本才是真成本。

    单价未配置时 total_cost / cost_per_accepted 为 None —— **不编造金额**。
    """
    by_provider: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_provider.setdefault(row.get("provider") or "unknown", []).append(row)

    out: list[dict[str, Any]] = []
    for provider, items in sorted(by_provider.items()):
        # 成本口径：优先用台账里生成时记下的金额；没记（当时没配单价）就按
        # **实测秒数 × 当前成本假设**现算——改价格假设后重跑 report 就能更新，
        # 不用重新生成图。台账记的是事实（秒数），假设是可以换的。
        costs: list[float] = []
        for item in items:
            cost = item.get("est_cost")
            if cost is None:
                cost = unit_cost_for(provider, item.get("gen_seconds") or 0.0)
            if cost is not None:
                costs.append(float(cost))
        priced = bool(costs)
        total_cost = sum(costs)
        accepted = sum(1 for item in items if item.get("qc_verdict") == "accepted")
        rejected = sum(1 for item in items if item.get("qc_verdict") == "rejected")
        seconds = sum(item.get("gen_seconds") or 0.0 for item in items)
        out.append(
            {
                "provider": provider,
                "calls": len(items),
                "accepted": accepted,
                "rejected": rejected,
                "accept_rate": round(accepted / len(items), 4) if items else None,
                "total_seconds": round(seconds, 1),
                "avg_seconds": round(seconds / len(items), 1) if items else 0.0,
                "priced": priced,
                "total_cost": round(total_cost, 4) if priced else None,
                # 单张合格图综合成本：总花费 ÷ 合格张数
                "cost_per_accepted": (
                    round(total_cost / accepted, 4) if (priced and accepted) else None
                ),
            }
        )
    return out


def adoption_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """采纳率摘要：adopted / 实际落盘产出，按后端分组。

    与 ledger.adoption_summary 同一口径，但吃已读出的资产行列表，
    供 report 组装（报表拿到的就是行列表，不再开一次库）。
    采纳率必须是**抽检台点击**写进 adopted 列的事实，不拿"放行"冒充。
    """
    generated = [r for r in rows if r.get("local_path")]

    def slot(items: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(items)
        adopted = sum(1 for r in items if r.get("adopted") == "adopted")
        rejected = sum(1 for r in items if r.get("adopted") == "rejected")
        return {
            "generated": n,
            "adopted": adopted,
            "rejected": rejected,
            "adoption_rate": round(adopted / n, 4) if n else None,
        }

    out = slot(generated)
    by_provider: dict[str, list[dict[str, Any]]] = {}
    for r in generated:
        by_provider.setdefault(r.get("provider") or "unknown", []).append(r)
    out["by_provider"] = {p: slot(items) for p, items in sorted(by_provider.items())}
    return out


def export_json(payload: dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def report_dir(out_dir: Path | None = None) -> Path:
    path = Path(out_dir or OUT_DIR) / "report"
    path.mkdir(parents=True, exist_ok=True)
    return path
