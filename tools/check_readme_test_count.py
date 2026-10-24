"""校验 README 里的"测试数量"是否等于实际收集数（可 --update 自动改写）。

为什么要有它：文档里的数字会漂——本仓库实测出现过 README 写 62/124、实际 152/155。
靠人肉同步必错，所以做成一条命令：CI 里跑它当门禁，或者改完测试后跑 `--update`。

用法：
  python tools/check_readme_test_count.py            # 只检查，不一致退出码 1
  python tools/check_readme_test_count.py --update   # 按实际数量改写 README
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

# README 里三处会出现测试数的固定句式（改了措辞就要同步改这里）
PATTERNS = [
    (re.compile(r"\*\*(\d+) 项全过\*\*"), "{n} 项全过"),
    (re.compile(r"# (\d+) 项回归测试"), "# {n} 项回归测试"),
    (re.compile(r"(pytest -q\s+# )(\d+)( 项)"), r"\g<1>{n}\g<3>"),
]


def collect_count() -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    match = re.search(r"(\d+) tests? collected", proc.stdout or "")
    if not match:
        print("无法解析 pytest 输出：\n" + (proc.stdout or "")[-800:], file=sys.stderr)
        raise SystemExit(2)
    return int(match.group(1))


def main() -> None:
    update = "--update" in sys.argv[1:]
    actual = collect_count()
    text = README.read_text(encoding="utf-8")

    found: list[tuple[str, int]] = []
    for pattern, template in PATTERNS:
        for m in pattern.finditer(text):
            number = int(m.group(2) if template.startswith("\\g") else m.group(1))
            found.append((m.group(0), number))

    if not found:
        print("README 里没找到测试数句式（PATTERNS 需要同步更新）", file=sys.stderr)
        raise SystemExit(2)

    mismatched = [(snippet, n) for snippet, n in found if n != actual]
    label = f"实际收集 {actual} 项；README 有 {len(found)} 处声明"

    if update:
        for pattern, template in PATTERNS:
            def _sub(m, template=template):
                if template.startswith("\\g"):
                    return m.group(1) + str(actual) + m.group(3)
                return m.group(0).replace(m.group(1), str(actual), 1)

            text = pattern.sub(_sub, text)
        README.write_text(text, encoding="utf-8")
        print(f"已更新：{label} → 全部改为 {actual}")
        return

    if mismatched:
        print(f"不一致：{label}")
        for snippet, n in mismatched:
            print(f"  - README 写 {n}：{snippet.strip()[:60]}")
        print("修复：python tools/check_readme_test_count.py --update")
        raise SystemExit(1)
    print(f"一致：{label}")


if __name__ == "__main__":
    main()
