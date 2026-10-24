"""Qwen-Image 3.0 适配层（百炼 DashScope 原生同步接口）。

踩坑记录：
1. `dashscope.aliyuncs.com/compatible-mode/v1/images/generations` 在经典域名下**不存在**（404），
   OpenAI 兼容模式只给了 `{WorkspaceId}.cn-beijing.maas.aliyuncs.com` 这种业务空间域名。
2. 直接把 1.3MB PNG 转 base64（≈1.8MB）塞进请求体，会被服务端重置连接（10054）。
   → 参考图统一压到长边 1024、JPEG q92，约 200KB。
3. 原生接口 size 用 `W*H`，且 i2i 的图要放在 content 数组里，与 text 平级。
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from providers import Result, _download

QWEN_SYNC = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


def _ref_data_uri(path: Path, max_edge: int = 1024, quality: int = 92) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class QwenProvider:
    def __init__(self, name: str, cfg: dict, ref: Path | None, key: str) -> None:
        self.name = name
        self.cfg = cfg
        self.ref = ref
        self.key = key
        self.ep = cfg.get("base_url") or QWEN_SYNC
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=httpx.Timeout(600, connect=20),
        )

    def generate(self, prompt: str, out_path: Path, mode: str) -> Result:
        content: list[dict[str, Any]] = []
        if mode == "i2i" and self.ref is not None:
            content.append({"image": _ref_data_uri(self.ref)})
        content.append({"text": prompt})

        body: dict[str, Any] = {
            "model": self.cfg["model"],
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": {
                "n": 1,
                "size": self.cfg.get("size", "1536*2048").replace("x", "*"),
                **(self.cfg.get("extras") or {}),
            },
        }

        t0 = time.perf_counter()
        try:
            r = self.client.post(self.ep, json=body)
            elapsed = time.perf_counter() - t0
            if r.status_code != 200:
                return Result(False, seconds=elapsed, model=self.cfg["model"],
                              error=f"HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            choices = (data.get("output") or {}).get("choices") or []
            url = None
            if choices:
                for item in (choices[0].get("message") or {}).get("content") or []:
                    if item.get("image"):
                        url = item["image"]
                        break
            if not url:
                return Result(False, seconds=elapsed, model=self.cfg["model"],
                              error=f"no image url: {str(data)[:300]}")
            real = _download(url, out_path, self.client)
            usage = data.get("usage") or {}
            size = f"{usage['output_width']}x{usage['output_height']}" if usage.get("output_width") else ""
            return Result(
                True, path=str(real), seconds=elapsed,
                cost_cny=float(self.cfg.get("price_cny", 0.0)),
                request_id=data.get("request_id"),
                model=self.cfg["model"], size=size,
                meta={"usage": usage},
            )
        except Exception as exc:
            return Result(False, seconds=time.perf_counter() - t0,
                          model=self.cfg["model"], error=f"{type(exc).__name__}: {exc}")
