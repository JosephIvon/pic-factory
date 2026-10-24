"""本地栈工作进程。

跑在独立的 CUDA venv 里，只依赖 torch + diffusers + PIL（不依赖主应用其它包）。
协议：stdin 收 JSON，stdout 回一行 JSON。

三个**必须记住的坑**（都是实测踩出来的）：

1. `model_index.json` 声明的是 `_class_name: StableDiffusionXLPipeline`（文生图类）。
   用 `AutoPipelineForImage2Image.from_pipe(pipe)` 会继承到这个 T2I 类，
   `image` / `strength` 被 `__call__(**kwargs)` 静默吞掉 —— 跑出来是纯文生图，
   **且不同 strength 的产物字节完全相同**。必须显式加载
   `StableDiffusionXLImg2ImgPipeline`。

2. 两个由同一份组件 `from_pipe` 出来的 pipeline **不能各自 enable_model_cpu_offload()**：
   共享模块上的 offload 钩子会互相踩，实测直接 0xC0000005 访问冲突。
   解法：一次调用只加载一种管线类（mode 决定），调用方按 mode 分组。

3. 8GB 显存装不下全量上卡（实测峰值 10.49GB 会 OOM）；offload 模式峰值 5.07GB、
   单张 5.62s。另外 **CLIP 上下文只有 77 token**，超长 prompt 会被静默截断，
   所以这里按 token 数裁剪（API 后端没有这个限制，是本地栈独有的坑）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

TOKEN_LIMIT = 77


def _fit_tokens(tokenizer, text: str, limit: int = TOKEN_LIMIT) -> str:
    """按 token 数裁剪 prompt，避免 CLIP 静默截断丢掉关键描述。"""
    if not text:
        return text
    try:
        ids = tokenizer(text, truncation=False, add_special_tokens=True).input_ids
        if len(ids) <= limit:
            return text
        return tokenizer.decode(ids[: limit - 1], skip_special_tokens=True).strip()
    except Exception:
        return text


def main() -> int:
    payload = json.loads(sys.stdin.read())
    model_dir = payload["model_dir"]
    mode = payload.get("mode", "t2i")
    items = payload.get("items") or []

    import torch
    from diffusers import (
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLPipeline,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    t0 = time.perf_counter()
    if mode == "i2i":
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_dir, torch_dtype=dtype, local_files_only=True, variant="fp16"
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_dir, torch_dtype=dtype, local_files_only=True, variant="fp16"
        )
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    load_seconds = time.perf_counter() - t0

    tokenizer = getattr(pipe, "tokenizer_2", None) or getattr(pipe, "tokenizer", None)

    results = []
    for item in items:
        started = time.perf_counter()
        out_path = Path(item["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            width, height = int(item["width"]), int(item["height"])
            generator = torch.Generator(device=device).manual_seed(int(item.get("seed") or 0))
            prompt = _fit_tokens(tokenizer, item["prompt"])
            negative = _fit_tokens(tokenizer, item.get("negative") or "")
            # 采样参数随模型族走：turbo 系 4 步 / guidance 0，
            # 普通 SDXL checkpoint 要 25~30 步 / guidance 7~8，否则发灰、结构崩坏。
            steps = int(item.get("steps") or payload.get("steps") or 4)
            guidance = item.get("guidance")
            if guidance is None:
                guidance = payload.get("guidance", 0.0)
            guidance = float(guidance)

            if mode == "i2i":
                from PIL import Image

                image = Image.open(item["input_path"]).convert("RGB").resize((width, height))
                strength = float(item.get("strength") or 0.45)
                # 步数可按项指定（参数扫描用）；默认给 turbo 模型 4 步
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative or None,
                    image=image,
                    strength=strength,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                )
            else:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative or None,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                )
            result.images[0].save(out_path)
            results.append(
                {
                    "ok": True,
                    "path": str(out_path),
                    "seconds": round(time.perf_counter() - started, 2),
                    "error": None,
                }
            )
        except Exception as exc:  # 单张失败不影响同批其它张
            results.append(
                {
                    "ok": False,
                    "path": None,
                    "seconds": round(time.perf_counter() - started, 2),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    payload_out = {
        "device": device,
        "mode": mode,
        "load_seconds": round(load_seconds, 2),
        "results": results,
    }
    sys.stdout.write(json.dumps(payload_out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
