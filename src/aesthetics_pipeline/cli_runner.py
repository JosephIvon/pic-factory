r"""在 Windows 上安全地调用外部 CLI。

实测坑（都在这里收口）：
  1. `mcode-tools` 是 .cmd 批处理，Python 的 subprocess **列表模式直接执行会 WinError 2**
     （"系统找不到指定的文件"）。必须经 cmd.exe /d /s /c 转发。
  2. `cmd /s /c` 会剥掉命令行最外层的一对引号，所以外面只能包**一层**；
     路径里有空格时由 list2cmdline 自己给程序名加引号，不要手动再包一层。
  3. JSON 参数不要用 `--args '<json>'` 内联，引号会被解析破坏；统一走 `--args-file`。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def resolve_cli(cli: str) -> str:
    """定位可执行文件；找不到就明确报错，不静默继续。"""
    found = shutil.which(cli)
    if found:
        return found
    if Path(cli).exists():
        return cli
    raise FileNotFoundError(
        f"找不到 CLI {cli!r}。请确认它在 PATH 上，或用环境变量指定绝对路径"
        "（例如 AP_MATRIX_CLI）。"
    )


def run_cli(cli: str, args: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    exe = resolve_cli(cli)
    if exe.lower().endswith((".cmd", ".bat")):
        cmdline = subprocess.list2cmdline([exe, *args])
        return subprocess.run(
            f'cmd.exe /d /s /c "{cmdline}"',
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    return subprocess.run(
        [exe, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def extract_json(text: str) -> dict[str, Any]:
    """从 CLI 输出里取出 JSON 对象。

    实测两种格式都要认：
      - 单行紧凑 JSON（connector call 的返回）
      - 多行美化 JSON（connector tools 的返回）
    """
    text = (text or "").strip()
    if not text:
        raise RuntimeError("CLI 没有输出内容")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise RuntimeError(f"无法从输出解析 JSON: {text[:500]}")
