"""A/B 的 provider 适配层：火山方舟 Seedream（同步）与阿里百炼万相（异步轮询）。

只依赖 httpx，不引 SDK——保证在客户环境里也能跑。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

ARK_EP = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
WAN_I2I_EP = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis"
WAN_T2I_EP = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
WAN_TASK_EP = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"


@dataclass
class Result:
    ok: bool
    path: str | None = None
    seconds: float = 0.0
    cost_cny: float = 0.0
    request_id: str | None = None
    model: str = ""
    size: str = ""
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def _sniff_ext(data: bytes) -> str:
    """服务端返回的扩展名不可信，按魔数定：PNG/JPEG/WEBP/GIF。"""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    return ".bin"


def _download(url: str, dest: Path, client: httpx.Client | None = None) -> Path:
    """下载并按真实格式落盘，返回最终路径（可能改过扩展名）。

    必须用不带 Authorization 的干净 client：阿里云 OSS 的签名 URL 会因为
    多带一个 Authorization 头而返回 403（签名不匹配）。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=httpx.Timeout(180, connect=20), follow_redirects=True) as bare:
        r = bare.get(url)
        r.raise_for_status()
        data = r.content
    real = dest.with_suffix(_sniff_ext(data))
    real.write_bytes(data)
    if real != dest and dest.exists():
        dest.unlink()
    return real


class ArkProvider:
    """Seedream：同步接口，一个请求一张图（4.0 可另开组图）。"""

    def __init__(self, name: str, cfg: dict, ref: Path | None, key: str) -> None:
        self.name = name
        self.cfg = cfg
        self.ref = ref
        self.key = key
        self.ep = cfg.get("base_url") or ARK_EP  # 支持中转服务
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=httpx.Timeout(300, connect=20),
        )

    def generate(self, prompt: str, out_path: Path, mode: str) -> Result:
        body: dict[str, Any] = {
            "model": self.cfg["model"],
            "prompt": prompt,
            "size": self.cfg.get("size", "1536x2048"),
            "response_format": "url",
        }
        if mode == "i2i" and self.ref is not None:
            body["image"] = _data_uri(self.ref)
        body.update(self.cfg.get("extras") or {})

        t0 = time.perf_counter()
        try:
            r = self.client.post(self.ep, json=body)
            elapsed = time.perf_counter() - t0
            if r.status_code != 200:
                return Result(False, seconds=elapsed, model=self.cfg["model"],
                              error=f"HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            if not data.get("data"):
                return Result(False, seconds=elapsed, model=self.cfg["model"],
                              error=f"empty data: {json.dumps(data)[:300]}")
            item = data["data"][0]
            url = item.get("url")
            if not url:
                return Result(False, seconds=elapsed, model=self.cfg["model"],
                              error="no url in response")
            real = _download(url, out_path, self.client)
            return Result(
                True, path=str(real), seconds=elapsed,
                cost_cny=float(self.cfg.get("price_cny", 0.0)),
                request_id=r.headers.get("x-request-id"),
                model=self.cfg["model"], size=str(item.get("size", "")),
                meta={"usage": data.get("usage")},
            )
        except Exception as exc:  # 网络/解析异常都归为失败，交给上层统计
            return Result(False, seconds=time.perf_counter() - t0,
                          model=self.cfg["model"], error=f"{type(exc).__name__}: {exc}")


class WanProvider:
    """万相：异步任务，创建后轮询 task_id。"""

    def __init__(self, name: str, cfg: dict, ref: Path | None, key: str) -> None:
        self.name = name
        self.cfg = cfg
        self.ref = ref
        self.key = key
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
            },
            timeout=httpx.Timeout(120, connect=20),
        )

    def generate(self, prompt: str, out_path: Path, mode: str) -> Result:
        params = {
            "n": 1,  # 关键：万相默认 n=4，不显式设就是 4 倍费用
            "size": self.cfg.get("size", "960*1280"),
            **(self.cfg.get("extras") or {}),
        }
        if mode == "i2i" and self.ref is not None:
            ep = WAN_I2I_EP
            body = {
                "model": self.cfg["model"],
                "input": {"prompt": prompt, "images": [_data_uri(self.ref)]},
                "parameters": params,
            }
        else:
            ep = WAN_T2I_EP
            body = {
                "model": self.cfg["model"],
                "input": {"prompt": prompt},
                "parameters": params,
            }

        t0 = time.perf_counter()
        try:
            r = self.client.post(ep, json=body)
            if r.status_code != 200:
                return Result(False, seconds=time.perf_counter() - t0, model=self.cfg["model"],
                              error=f"HTTP {r.status_code}: {r.text[:300]}")
            task_id = r.json().get("output", {}).get("task_id")
            if not task_id:
                return Result(False, seconds=time.perf_counter() - t0, model=self.cfg["model"],
                              error=f"no task_id: {r.text[:300]}")

            deadline = time.perf_counter() + 480
            while time.perf_counter() < deadline:
                time.sleep(8)
                q = self.client.get(WAN_TASK_EP.format(task_id=task_id))
                if q.status_code != 200:
                    continue
                out = q.json().get("output", {})
                status = out.get("task_status")
                if status == "SUCCEEDED":
                    results = out.get("results") or []
                    url = next((x.get("url") for x in results if x.get("url")), None)
                    if not url:
                        return Result(False, seconds=time.perf_counter() - t0,
                                      model=self.cfg["model"], error="succeeded but no url")
                    real = _download(url, out_path, self.client)
                    return Result(
                        True, path=str(real), seconds=time.perf_counter() - t0,
                        cost_cny=float(self.cfg.get("price_cny", 0.0)),
                        request_id=task_id, model=self.cfg["model"],
                        size=self.cfg.get("size", ""),
                    )
                if status in ("FAILED", "CANCELED", "UNKNOWN"):
                    return Result(False, seconds=time.perf_counter() - t0,
                                  model=self.cfg["model"],
                                  error=f"task {status}: {json.dumps(out)[:300]}")
            return Result(False, seconds=time.perf_counter() - t0, model=self.cfg["model"],
                          error="poll timeout")
        except Exception as exc:
            return Result(False, seconds=time.perf_counter() - t0,
                          model=self.cfg["model"], error=f"{type(exc).__name__}: {exc}")


def build(name: str, cfg: dict, ref: Path | None, env: dict[str, str]):
    kind = cfg["kind"]
    if kind == "ark":
        key = env.get("ARK_API_KEY")
        if not key:
            raise RuntimeError("缺少 ARK_API_KEY")
        return ArkProvider(name, cfg, ref, key)
    if kind == "wan":
        key = env.get("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY")
        return WanProvider(name, cfg, ref, key)
    if kind == "qwen":
        from providers_qwen import QwenProvider

        key = env.get("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY")
        return QwenProvider(name, cfg, ref, key)
    raise ValueError(f"未知 provider 类型: {kind}")
