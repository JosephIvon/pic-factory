"""测试级隔离：让用例不依赖开发者 shell 里的环境变量。

背景（实测踩过）：`AP_PACK` 一旦在 shell 里设上，`expand()` 会优先读域包，
而 `test_library` / `test_cli_cost_wiring` 用 monkeypatch 指到临时矩阵、
期望走旧路径——于是同一套测试在"设了 AP_PACK"的机器上红、在"没设"的机器上绿。

测试必须与开发者的 shell 无关：默认清空，需要的用例自己 `monkeypatch.setenv`。
"""

from __future__ import annotations

import pytest

PACK_ENV = ("AP_PACK", "AP_PACKS_DIR")


@pytest.fixture(autouse=True)
def _isolate_pack_env(monkeypatch):
    for name in PACK_ENV:
        monkeypatch.delenv(name, raising=False)
    yield
