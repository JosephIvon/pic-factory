"""全通路探针：验证两把 key 在各自可用端点上的真实状态。

用法：python probe_all.py [workspace_id]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import httpx

from envload import load_env

WS = sys.argv[1] if len(sys.argv) > 1 else "llm-k0ips249in5m1uxc"
ARK = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
Q_T2I = {"model": "qwen-image-3.0", "prompt": "a white cup on a wooden table", "size": "1024x1024", "n": 1}
Q_NATIVE = {
    "model": "qwen-image-3.0",
    "input": {"messages": [{"role": "user", "content": [{"text": "a white cup on a wooden table"}]}]},
    "parameters": {"n": 1, "size": "1024*1024"},
}


def h12(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    env = load_env(Path(__file__).resolve().parent)
    ark, ds = env.get("ARK_API_KEY", ""), env.get("DASHSCOPE_API_KEY", "")
    print(f"ARK len={len(ark)} sha={h12(ark)}")
    print(f"DASHSCOPE len={len(ds)} sha={h12(ds)}  workspace={WS}")

    print("\n--- Ark ---")
    for model in ("doubao-seedream-4-0-20260415", "doubao-seedream-5-0-pro-260628"):
        try:
            r = httpx.post(
                ARK,
                headers={"Authorization": f"Bearer {ark}", "Content-Type": "application/json"},
                json={"model": model, "prompt": "a white cup", "size": "1024x1024"},
                timeout=90,
            )
            print(f"{model:34s} {r.status_code} {r.text.replace(chr(10), ' ')[:110]}")
        except Exception as e:
            print(f"{model:34s} ERR {type(e).__name__} {str(e)[:70]}")

    print("\n--- DashScope ---")
    dh = {"Authorization": f"Bearer {ds}", "Content-Type": "application/json"}
    tests = [
        ("classic-native", "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation", Q_NATIVE),
        ("ws-compat", f"https://{WS}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/images/generations", Q_T2I),
        ("ws-native", f"https://{WS}.cn-beijing.maas.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation", Q_NATIVE),
    ]
    for name, url, body in tests:
        try:
            r = httpx.post(url, headers=dh, json=body, timeout=180)
            print(f"{name:16s} {r.status_code} {r.text.replace(chr(10), ' ')[:130]}")
        except Exception as e:
            print(f"{name:16s} ERR {type(e).__name__} {str(e)[:70]}")


if __name__ == "__main__":
    main()
