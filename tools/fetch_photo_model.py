"""分块下载写实向 SDXL checkpoint（diffusers 目录结构，fp16 变体）。

为什么分块：实测在本机网络下，**单个长连接下载大文件会卡死**
（连接建立后 0 字节、十几分钟无进展；HF 官方客户端和 requests 单流都一样），
而小文件（235MB / 160MB）能正常下完。判断是长连接被中间链路掐断，
所以改成**固定大小分块、每块独立 Range 请求**，单块失败只重试那块。

用法：python tools/fetch_photo_model.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests

DEST = Path(os.environ.get("AP_PHOTO_MODEL_DIR") or r"D:\Tools\models\juggernaut-xl-v9")
REPO = "RunDiffusion/Juggernaut-XL-v9"
BASE = f"https://huggingface.co/{REPO}/resolve/main"

FILES = [
    ("model_index.json", "model_index.json"),
    ("scheduler/scheduler_config.json", "scheduler/scheduler_config.json"),
    ("tokenizer/merges.txt", "tokenizer/merges.txt"),
    ("tokenizer/special_tokens_map.json", "tokenizer/special_tokens_map.json"),
    ("tokenizer/tokenizer_config.json", "tokenizer/tokenizer_config.json"),
    ("tokenizer/vocab.json", "tokenizer/vocab.json"),
    ("tokenizer_2/merges.txt", "tokenizer_2/merges.txt"),
    ("tokenizer_2/special_tokens_map.json", "tokenizer_2/special_tokens_map.json"),
    ("tokenizer_2/tokenizer_config.json", "tokenizer_2/tokenizer_config.json"),
    ("tokenizer_2/vocab.json", "tokenizer_2/vocab.json"),
    ("text_encoder/config.json", "text_encoder/config.json"),
    ("text_encoder/model.fp16.safetensors", "text_encoder/model.fp16.safetensors"),
    ("text_encoder_2/config.json", "text_encoder_2/config.json"),
    ("text_encoder_2/model.fp16.safetensors", "text_encoder_2/model.fp16.safetensors"),
    ("unet/config.json", "unet/config.json"),
    ("unet/diffusion_pytorch_model.fp16.safetensors", "unet/diffusion_pytorch_model.fp16.safetensors"),
    ("vae/config.json", "vae/config.json"),
    ("vae/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors"),
]

BLOCK = 32 << 20  # 32MB 一块


def remote_size(url: str, attempts: int = 10) -> int:
    """取远端文件大小。

    HF 的 HEAD 响应不一定带 content-length，所以用 Range 请求读 Content-Range。
    实测本机网络到 us.aws.cdn.hf.co 会被掐断（SSL: UNEXPECTED_EOF_WHILE_READING），
    所以这里要重试——连接是**间歇性**坏的，不是完全不通。
    """
    last = ""
    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(
                url, headers={"Range": "bytes=0-0"}, stream=True, timeout=(15, 30), allow_redirects=True
            )
            r.raise_for_status()
            content_range = r.headers.get("content-range", "")
            if "/" in content_range:
                return int(content_range.rsplit("/", 1)[1])
            return int(r.headers.get("content-length", 0))
        except Exception as exc:
            last = f"{type(exc).__name__}: {str(exc)[:120]}"
            print(f"  取文件大小第 {attempt} 次失败: {last}", flush=True)
            time.sleep(3)
    raise SystemExit(f"拿不到远端文件大小：{url}\n{last}")


def fetch(remote: str, target: Path, attempts: int = 8) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{remote}"
    size = remote_size(url)
    if target.exists() and target.stat().st_size == size:
        print(f"  已有 {remote}", flush=True)
        return

    tmp = target.with_suffix(target.suffix + ".part")
    with tmp.open("wb") as fh:
        offset = 0
        while offset < size:
            end = min(offset + BLOCK, size) - 1
            data = None
            for attempt in range(1, attempts + 1):
                try:
                    r = requests.get(
                        url,
                        headers={"Range": f"bytes={offset}-{end}"},
                        stream=True,
                        timeout=(15, 45),
                    )
                    r.raise_for_status()
                    data = r.content
                    break
                except Exception as exc:
                    print(
                        f"  {remote} 块 @{offset} 第 {attempt} 次失败: {type(exc).__name__}",
                        flush=True,
                    )
                    time.sleep(2)
            if data is None:
                raise SystemExit(f"下载失败：{remote} @{offset}")
            fh.write(data)
            offset += len(data)
            print(f"  {remote}: {offset}/{size} ({offset * 100 // size}%)", flush=True)
    tmp.replace(target)
    print(f"  ✓ {remote}  {target.stat().st_size / 1048576:.0f}MB", flush=True)


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"下载 {REPO} → {DEST}（fp16 变体，约 7GB，32MB 分块）", flush=True)
    for remote, rel in FILES:
        fetch(remote, DEST / rel)
    total = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file())
    print(f"完成，合计 {total / 1073741824:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
