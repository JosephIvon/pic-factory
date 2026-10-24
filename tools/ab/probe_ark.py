"""探针：确认方舟 key 在标准 /api/v3 下能访问哪些生图模型。

用法：python probe_ark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

from envload import load_env

BASE = "https://ark.cn-beijing.volces.com/api/v3"
CANDIDATES = [
    "doubao-seedream-5-0-lite-260628",
    "doubao-seedream-4-5-251128",
    "doubao-seedream-4-0-250828",
    "doubao-seedream-3-0-t2i-250415",
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    env = load_env(Path(__file__).resolve().parent)
    h = {"Authorization": "Bearer " + env["ARK_API_KEY"], "Content-Type": "application/json"}

    try:
        r = httpx.get(f"{BASE}/models", headers=h, timeout=30)
        print(f"models list: {r.status_code} {r.text[:300]}")
    except Exception as e:
        print(f"models list: ERR {type(e).__name__} {e}")

    for m in CANDIDATES:
        body = {"model": m, "prompt": "a white ceramic cup on a wooden table, soft daylight", "size": "1024x1024"}
        try:
            rr = httpx.post(f"{BASE}/images/generations", headers=h, json=body, timeout=90)
            text = rr.text.replace("\n", " ")[:200]
            print(f"{m:34s} {rr.status_code} {text}")
        except Exception as e:
            print(f"{m:34s} ERR {type(e).__name__} {e}")


if __name__ == "__main__":
    main()
