"""生图后端适配层。

统一契约：
    generate(items) -> list[GenOutcome]

实现四种后端：
    matrix    真实调用（mcode-tools CLI），同时是 replay 缓存的写入方
    replay    离线回放（读 matrix 写下的缓存），断网/没配额时照常跑完整流水线
    local     本地栈（独立 venv 的 CUDA torch + SDXL-Turbo），子进程隔离
    http_api  直连 Ark(同步)/万相(异步) HTTP 接口，一个执行器 + 每族一个适配器
"""

from __future__ import annotations

from .base import Provider
from .http_api import StandardApiProvider
from .local import LocalProvider
from .matrix import MatrixProvider
from .replay import ReplayProvider

__all__ = [
    "Provider", "MatrixProvider", "ReplayProvider", "LocalProvider",
    "StandardApiProvider", "build_provider",
]


def build_provider(name: str, **kwargs) -> Provider:
    providers = {
        "matrix": MatrixProvider,
        "replay": ReplayProvider,
        "local": LocalProvider,
        "http_api": StandardApiProvider,
    }
    if name not in providers:
        raise ValueError(f"未知后端 {name!r}，可选：{', '.join(providers)}")
    return providers[name](**kwargs)
