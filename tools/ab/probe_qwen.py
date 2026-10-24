"""探针：确认百炼 key 在三条可能路径上的真实状态（401=key 无效 / 404=路径不对 / 200=通）。

用法：python probe_qwen.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

from envload import load_env

BODY = {
    "model": "qwen-image-3.0",
    "input": {"messages": [{"role": "user", "content": [{"text": "a white cup on a table"}]}]},
    "parameters": {"n": 1, "size": "1024*1024"},
}
PATHS = [
    ("sync-multimodal", "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"),
    ("async-imagegen", "https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation"),
    ("compat-images", "https://dashscope.aliyuncs.com/compatible-mode/v1/images/generations"),
    ("tasks-fake", "https://dashscope.aliyuncs.com/api/v1/tasks/00000000-0000-0000-0000-000000000000"),
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    env = load_env(Path(__file__).resolve().parent)
    h = {
        "Authorization": "Bearer " + env["DASHSCOPE_API_KEY"],
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    for name, url in PATHS:
        try:
            if name == "tasks-fake":
                r = httpx.get(url, headers=h, timeout=30)
            else:
                r = httpx.post(url, headers=h, json=BODY, timeout=60)
            text = r.text.replace("\n", " ")[:220]
            print(f"{name:18s} {r.status_code} {text}")
        except Exception as e:
            print(f"{name:18s} ERR {type(e).__name__} {str(e)[:90]}")


if __name__ == "__main__":
    main()
