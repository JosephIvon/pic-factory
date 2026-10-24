"""交付前检查：域包里每个角色都必须有可商用的授权记录。

这是 `compliance.audit` 的**离线版**：不依赖台账，直接对域包做静态检查，
适合在交给甲方/发布前跑一遍（CI 或人工验收都能用）。

用法：
  python tools/check_asset_licenses.py                 # 默认医美包
  python tools/check_asset_licenses.py --pack example-retail
退出码：0=全部有授权；1=有缺失/不可商用/已过期。
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Windows 控制台默认代码页会把 ✓/✗ 这类符号写崩（UnicodeEncodeError）：
# 这个检查经常是人在自己机器上手跑的，不能因为输出编码退化成"看起来像失败"。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from aesthetics_pipeline.pack import available_packs, load_pack  # noqa: E402


def main() -> int:
    args = sys.argv[1:]
    pack_name = None
    if "--pack" in args:
        pack_name = args[args.index("--pack") + 1]
    # 默认与运行口径一致：AP_PACK → medical-aesthetics（主包），都没有才取第一个可用包。
    pack_name = pack_name or os.environ.get("AP_PACK") or (
        "medical-aesthetics" if "medical-aesthetics" in available_packs() else None
    )
    pack_name = pack_name or (available_packs()[0] if available_packs() else None)
    if not pack_name:
        print("没有可用域包（packs/ 为空）", file=sys.stderr)
        return 2

    pack = load_pack(pack_name)
    characters = list((pack.matrix.get("characters") or {}).keys())
    entries = (pack.licenses or {}).get("characters") or {}
    print(f"域包：{pack.name}  角色 {len(characters)} 个  授权记录 {len(entries)} 条\n")

    failures: list[str] = []
    today = date.today()
    for name in characters:
        entry = entries.get(name)
        if not entry:
            failures.append(f"{name}: 缺少授权记录（licenses.yaml）")
            print(f"  ✗ {name}: 缺少授权记录")
            continue
        problems: list[str] = []
        if entry.get("commercial_use") is False:
            problems.append("标注为不可商用")
        if not entry.get("source"):
            problems.append("缺少来源（真人/AI 生成）")
        if entry.get("kind") == "real" and not entry.get("consent"):
            problems.append("真人角色缺少肖像同意说明")
        valid_until = entry.get("valid_until")
        if valid_until:
            try:
                if date.fromisoformat(str(valid_until)) < today:
                    problems.append(f"授权已过期（{valid_until}）")
            except ValueError:
                problems.append(f"valid_until 不是日期：{valid_until}")
        if problems:
            failures.append(f"{name}: " + "；".join(problems))
            print(f"  ✗ {name}: " + "；".join(problems))
        else:
            print(f"  ✓ {name}: {entry.get('license', '-')}")

    unused = sorted(set(entries) - set(characters))
    if unused:
        print(f"\n提示：授权表里有未被子引用的角色：{', '.join(unused)}")

    if failures:
        print(f"\n共 {len(failures)} 个角色不可交付。")
        return 1
    print("\n全部角色授权检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
