"""工具脚本不得指向已删除的实验台路径（2026-10-24 删除后的防回归）。

背景：源实验台 `D:\\workspace\\picfactory` 已删，两个复算脚本曾把
`--lab-dir` 的默认值写成该路径——源没了，默认值就是一条死路径，
跑起来只会得到"找不到台账"的误导性报错。现在它们**必须显式传参**。

这条测试挡两件事：
1. 工具脚本里再出现 `picfactory`（死路径回潮）；
2. `--lab-dir` 又被加回默认值（"看起来能跑"的假象）。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _tool_sources() -> list[Path]:
    return [
        p for p in TOOLS.rglob("*.py")
        if "__pycache__" not in p.parts and "ab" not in p.relative_to(TOOLS).parts[:1]
    ]


def test_no_tool_references_deleted_lab_repo() -> None:
    offenders = []
    for path in _tool_sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        if "picfactory" in text:
            offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, f"这些工具仍引用已删除的实验台：{offenders}"


def test_lab_dir_is_required_not_defaulted() -> None:
    """`--lab-dir` 必须是必填；有 default 就说明又埋了一条死路径。"""
    for name in ("recompute_lab_sims.py", "recalibrate_face_threshold.py"):
        text = (TOOLS / name).read_text(encoding="utf-8")
        block = re.search(r'"--lab-dir",(.{0,240}?)\)', text, re.S)
        assert block, f"{name} 里找不到 --lab-dir 参数"
        body = block.group(1)
        assert "required=True" in body, f"{name} 的 --lab-dir 必须 required=True"
        assert "default=" not in body, f"{name} 的 --lab-dir 不应有默认值"
