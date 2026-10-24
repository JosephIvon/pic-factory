"""后端契约。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import GenItem, GenOutcome


@runtime_checkable
class Provider(Protocol):
    name: str

    def generate(self, items: list[GenItem]) -> list[GenOutcome]:
        """批量生成。返回与 items 等长、顺序一致的 outcomes。"""
        ...
