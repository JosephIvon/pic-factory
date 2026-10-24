"""离线回放后端。

读真实后端写下的缓存（out/cache/<provider>/<key>.json + out/images/…），
不调任何外部服务。用途：断网 / 账号配额异常时，照样跑完整条流水线，
演示的流程、质检、台账都不打折。

只回放缓存里**真实生成过**的图，不伪造任何素材。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import OUT_DIR, REPO_ROOT
from ..models import GenItem, GenOutcome
from .cache import read_cache, resolve_local_path


class ReplayProvider:
    """离线回放。

    依次在多个缓存目录里找：`out/cache/`（本机 live 产物）→
    `fixtures/cache/`（随仓库提交的压缩副本）。
    所以新 clone 的机器没有 `out/` 也能复现整条流水线。

    `prefer_provider` 指定复现哪条路线（matrix / http_api / local）。
    不指定时按 PROVIDER_ORDER 取先命中的那条；**不会拿别路线的图冒充**。
    """

    name = "replay"

    def __init__(
        self,
        out_dir: Path | None = None,
        extra_cache_dirs: tuple[Path, ...] = (),
        prefer_provider: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self.out_dir = Path(out_dir or OUT_DIR)
        self.cache_dirs = [self.out_dir / "cache", *extra_cache_dirs]
        if not extra_cache_dirs:
            self.cache_dirs.append(REPO_ROOT / "fixtures" / "cache")
        self.prefer_provider = prefer_provider
        # 指定档位时只回放该档的缓存（多档并存时不会拿别的档冒充）。
        self.model_id = model_id

    def _find(self, key: str, model_id: str | None = None) -> dict | None:
        wanted = model_id or self.model_id
        for cache_dir in self.cache_dirs:
            entry = read_cache(cache_dir, key, self.prefer_provider, wanted)
            if entry is not None:
                return entry
        return None

    def generate(
        self, items: list[GenItem], model_ids: list[str | None] | None = None
    ) -> list[GenOutcome]:
        outcomes: list[GenOutcome] = []
        for index, item in enumerate(items):
            key = item.task.idempotency_key
            wanted = model_ids[index] if model_ids else None
            entry = self._find(key, wanted)
            if entry is None:
                outcomes.append(
                    GenOutcome(task=item.task, ok=False, error="回放缓存缺失（该任务尚未真实生成过）")
                )
                continue
            local_path = resolve_local_path(entry["local_path"])
            if not local_path.exists():
                outcomes.append(
                    GenOutcome(task=item.task, ok=False, error=f"回放素材缺失: {local_path}")
                )
                continue
            outcomes.append(
                GenOutcome(
                    task=item.task,
                    ok=True,
                    node_id=entry.get("node_id"),
                    file_name=entry.get("file_name"),
                    local_path=str(local_path),
                    seconds=float(entry.get("seconds", 0.0)),
                    # 回放不改归属：仍是原后端生成的同一张图；
                    # 档位照搬缓存条目——http_api 的图回放后仍按实际档位
                    # 落账（同键同 provider 是 upsert，不会双计，Trae S4）。
                    model_id=entry.get("model_id"),
                    source_provider=entry.get("provider") or "matrix",
                )
            )
        return outcomes
