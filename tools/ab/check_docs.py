"""一致性检查：score.json 与两份报告里的表格数字是否逐项对得上。

用法：python check_docs.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCORE = ROOT / "out" / "ab" / "score.json"
DOCS = [ROOT / "out" / "ab" / "REPORT.md", ROOT.parent / "MODEL-CHOICE.md"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    score = {s["label"]: s for s in json.loads(SCORE.read_text(encoding="utf-8"))}

    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        problems = []
        for label, s in score.items():
            row = next((ln for ln in text.splitlines() if ln.startswith(f"| {label} |")), None)
            if row is None:
                problems.append(f"{label}: 表里没有这一行")
                continue
            cells = [c.strip().replace("**", "") for c in row.strip("|").split("|")]
            nums = [c for c in cells if re.fullmatch(r"-?\d+\.\d{3}", c)]
            want = [f"{s['sim_mean']:.3f}", f"{s['sim_mean_clean']:.3f}", f"{s['sim_min']:.3f}"]
            missing = [w for w in want if w not in nums]
            if missing:
                problems.append(f"{label}: 缺 {missing}，行内数字 {nums}")
        print(f"{doc.name}: {'OK' if not problems else problems}")


if __name__ == "__main__":
    main()
