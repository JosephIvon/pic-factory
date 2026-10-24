"""读取 ab/.env 到环境变量字典（不覆盖已存在的真实环境变量）。"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    p = root / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env
