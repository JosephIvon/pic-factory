"""盲测物料：每模型随机抽 N 张，打乱成一张对照表 + 答案键。

用法：
  python blind.py --per 5
然后让人看 out/blind/sheet.png，标出"哪些是 AI"（字母），
再用答案键 out/blind/key.json 算可辨识率。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out" / "ab"
BLIND = OUT / "blind"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cell", type=int, default=300)
    ap.add_argument("--crop-bottom", type=float, default=0.12,
                    help="裁掉底部比例，去掉平台'AI生成'水印（否则盲测失效）")
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in (OUT / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by: dict[str, list[dict]] = {}
    for r in rows:
        if r["ok"]:
            by.setdefault(r["provider"], []).append(r)

    rnd = random.Random(args.seed)
    picked: list[dict] = []
    for name, rs in by.items():
        rnd.shuffle(rs)
        picked.extend(rs[: args.per])
    rnd.shuffle(picked)

    BLIND.mkdir(parents=True, exist_ok=True)
    for f in BLIND.glob("*.png"):
        f.unlink()

    cols = 5
    rows_n = (len(picked) + cols - 1) // cols
    cell_w, cell_h = args.cell, int(args.cell * 4 / 3)
    sheet = Image.new("RGB", (cols * cell_w, rows_n * (cell_h + 28)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    key = {}
    for i, r in enumerate(picked):
        letter = LETTERS[i]
        p = Path(r["path"])
        if not p.exists():
            alt = [c for c in p.parent.glob(p.stem + ".*") if c.is_file()]
            p = alt[0] if alt else p
        img = Image.open(p).convert("RGB")
        if args.crop_bottom > 0:
            w, h = img.size
            img = img.crop((0, 0, w, int(h * (1 - args.crop_bottom))))
        img.thumbnail((cell_w - 8, cell_h - 8))
        x = (i % cols) * cell_w + 4
        y = (i // cols) * (cell_h + 28) + 4
        sheet.paste(img, (x, y))
        draw.text((x + 4, y + cell_h), letter, fill="black", font=font)
        key[letter] = {"provider": r["provider"], "label": r["label"], "prompt_id": r["prompt_id"]}

    sheet.save(BLIND / "sheet.png")
    (BLIND / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"对照表: {BLIND / 'sheet.png'}  共 {len(picked)} 张")
    print(f"答案键: {BLIND / 'key.json'}（评分前不要给被测试者看）")


if __name__ == "__main__":
    main()
