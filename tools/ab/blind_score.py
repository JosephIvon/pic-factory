"""盲测计分：把"被试认为哪些是 AI"的字母列表换成每个模型的可辨识率。

用法：
  python blind_score.py "A,C,F,H"
输出：
  每个模型被正确认出 / 抽出的张数 → 可辨识率
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out" / "ab" / "blind"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    key = json.loads((OUT / "key.json").read_text(encoding="utf-8"))
    picked = {x.strip().upper() for x in (sys.argv[1] if len(sys.argv) > 1 else "").split(",") if x.strip()}

    total: dict[str, int] = defaultdict(int)
    hit: dict[str, int] = defaultdict(int)
    for letter, meta in key.items():
        label = meta["label"]
        total[label] += 1
        if letter in picked:
            hit[label] += 1

    print(f"被试标为 AI 的字母: {sorted(picked) or '（空）'}\n")
    print("| 模型 | 被判为 AI | 抽出示意 | 可辨识率 |")
    print("|---|---|---|---|")
    for label in total:
        rate = hit[label] / total[label]
        print(f"| {label} | {hit[label]} | {total[label]} | {rate:.0%} |")

    all_rate = len(picked) / max(len(key), 1)
    print(f"\n整体可辨识率: {all_rate:.0%}（{len(picked)}/{len(key)}）")
    print("判读：越低越真实。≥60% = 一眼假；30–60% = 有 AI 感；≤30% = 接近实拍。")


if __name__ == "__main__":
    main()
