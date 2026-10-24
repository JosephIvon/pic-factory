"""本地栈后端：独立 venv 的 CUDA torch + SDXL-Turbo。

为什么单独一个 venv：主应用保持纯 Python 轻依赖，本地栈的重依赖（CUDA torch）
被隔离在子进程里，主程序通过 stdin/stdout 的 JSON 协议调用，崩了也不影响主线。

本地栈买到的三样东西（闭源 API 给不了）：
  1. 参考强度可控（img2img 的 strength 参数）
  2. seed 固定即可复现
  3. 数据不出网 + 单张边际成本接近电费

按 mode 分组调用 worker：文生图与图生图分两次进程，避开共享组件的 offload 钩子冲突。
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ..config import OUT_DIR, REPO_ROOT, local_model_dir, local_python, local_sampling
from ..models import GenItem, GenOutcome
from .cache import write_cache

# SDXL 系按 64 对齐的尺寸桶
DIMENSIONS: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "21:9": (1536, 672),
}

WORKER = Path(__file__).with_name("local_worker.py")


def _resolve_ref(ref_asset: str | None) -> str | None:
    """参考图路径要绝对路径：子进程的 cwd 不是仓库根。"""
    if not ref_asset:
        return None
    for candidate in (REPO_ROOT / ref_asset, REPO_ROOT / "evidence" / ref_asset):
        if candidate.exists():
            return str(candidate)
    return None


class LocalProvider:
    name = "local"

    def __init__(self, model_path: Path | None = None, timeout: int = 1800) -> None:
        self.model_path = model_path or local_model_dir()
        self.timeout = timeout

    def _run(self, mode: str, items: list[GenItem]) -> list[GenOutcome]:
        out_dir = Path(OUT_DIR) / "local_images"
        sampling = local_sampling()
        payload = {
            "model_dir": str(self.model_path),
            "mode": mode,
            "steps": sampling["steps"],
            "guidance": sampling["guidance"],
            "items": [
                {
                    "prompt": item.prompt,
                    "negative": item.negative_prompt,
                    "width": DIMENSIONS.get(item.aspect_ratio, (1024, 1024))[0],
                    "height": DIMENSIONS.get(item.aspect_ratio, (1024, 1024))[1],
                    "out_path": str(out_dir / f"{item.task.idempotency_key}.png"),
                    "input_path": _resolve_ref(item.task.ref_asset),
                    "strength": float(os.environ.get("AP_LOCAL_STRENGTH") or "0.45"),
                    "seed": int(item.task.idempotency_key[:8], 16) % 2**31,
                }
                for item in items
            ],
        }
        proc = subprocess.run(
            [str(local_python()), str(WORKER)],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout)[-800:]
            return [GenOutcome(task=i.task, ok=False, error=f"本地栈失败: {err}") for i in items]
        try:
            results = json.loads(proc.stdout.strip().splitlines()[-1])["results"]
        except Exception as exc:
            return [
                GenOutcome(task=i.task, ok=False, error=f"本地栈返回无法解析: {exc}") for i in items
            ]

        outcomes: list[GenOutcome] = []
        for item, res in zip(items, results):
            ok = bool(res.get("ok"))
            local_path = res.get("path") if ok else None
            if ok and local_path:
                # 同样写回放缓存：本地栈的产出也必须能离线复现
                write_cache(
                    Path(OUT_DIR) / "cache",
                    key=item.task.idempotency_key,
                    provider=self.name,
                    node_id=None,
                    file_name=Path(local_path).name,
                    local_path=local_path,
                    prompt=item.prompt,
                    path=item.task.path,
                    seconds=float(res.get("seconds", 0.0)),
                )
            outcomes.append(
                GenOutcome(
                    task=item.task,
                    ok=ok,
                    node_id=None,
                    file_name=Path(local_path).name if local_path else None,
                    local_path=local_path,
                    seconds=float(res.get("seconds", 0.0)),
                    error=res.get("error"),
                    source_provider=self.name,
                )
            )
        return outcomes

    def generate(self, items: list[GenItem]) -> list[GenOutcome]:
        """文生图与图生图分两次进程调用（避开共享组件的 offload 钩子冲突）。"""
        outcomes: list[GenOutcome] = []
        for mode, group in (
            ("t2i", [i for i in items if i.task.path != "i2i"]),
            ("i2i", [i for i in items if i.task.path == "i2i"]),
        ):
            if group:
                outcomes.extend(self._run(mode, group))
        return outcomes
