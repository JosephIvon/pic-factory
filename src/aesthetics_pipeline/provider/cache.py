"""回放缓存：记录每张图"谁生成的、怎么生成的"。

回放（replay）读这个缓存离线复现全流程，所以各写入后端格式必须一致。
缓存里记的是**真正生成的后端**——回放只是传输方式，不是新的生成后端。

路径存**仓库相对路径**（`out/images/xxx.png`），这样缓存可以随仓库搬到别的机器；
`fixtures/cache/` 里另有一份提交进 git 的副本，指向 `fixtures/images/`，
新 clone 后没有 `out/` 也能离线回放。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT


def cache_path(
    cache_dir: Path, key: str, provider: str | None = None, model_id: str | None = None
) -> Path:
    """缓存文件路径。

    **provider 进目录名**：matrix 与 local 共用同一幂等键，扁平命名会互相覆盖
    （与 `out/swapped/` 换脸产物同一个坑：台账复合主键不冲突，磁盘会冲突）。
    **档位进文件名**：同一后端换模型跑同一任务是两条资产，缓存也不能互相覆盖。
    """
    cache_dir = Path(cache_dir)
    if provider:
        name = f"{key}__{model_id}.json" if model_id else f"{key}.json"
        return cache_dir / provider / name
    return cache_dir / f"{key}.json"


def relative_to_repo(path: str | Path) -> str:
    """仓库内路径转相对路径（可移植）；仓库外的保留绝对路径。"""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def resolve_local_path(raw: str) -> Path:
    """把缓存里的路径还原成本地可读路径。"""
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def write_cache(
    cache_dir: Path,
    *,
    key: str,
    provider: str,
    node_id: str | None,
    file_name: str | None,
    local_path: str,
    prompt: str,
    path: str,
    seconds: float,
    model_id: str | None = None,
    portable: bool = True,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "idempotency_key": key,
        "provider": provider,
        "node_id": node_id,
        "file_name": file_name,
        "local_path": relative_to_repo(local_path) if portable else local_path,
        "prompt": prompt,
        "path": path,
        "seconds": round(seconds, 3),
        # 档位随行：http_api 的价目/台账 model_id 都挂在档位上，
        # 缓存不带它，回放线就会退化成"不知道哪档生成的图"（Trae S4）。
        "model_id": model_id,
    }
    target = cache_path(cache_dir, key, provider, model_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


# 回放时的后端优先顺序（确定性优先，不看写入时间）。
# http_api 必须入表：它是交付主线，此前表里只有 matrix/local——API 真实
# 产出的缓存默认回放口径下根本不可见（断网兜底对交付线名不副实，Trae S4）。
# 插在 matrix 之后：既有 matrix/local 的相对优先级一字不动，只补漏。
PROVIDER_ORDER = ("matrix", "http_api", "local")


def _load(target: Path) -> dict[str, Any] | None:
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def read_cache(
    cache_dir: Path,
    key: str,
    provider: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any] | None:
    """读缓存条目。

    provider 给定 → 只认该后端的条目（不拿别的路线的图冒充）。
    provider 为空 → 按 PROVIDER_ORDER 依次找，最后退回旧的扁平布局。

    model_id 给定 → 优先读该档位的缓存；旧的无档位条目只在"没记档位"时兜底，
    绝不拿 A 档的图冒充 B 档。不指定 model_id 时：先旧布局，再取该 key
    最新的档位缓存（回放"最近一次"语义）。
    """
    cache_dir = Path(cache_dir)
    targets: list[Path] = []
    providers = [provider] if provider else list(PROVIDER_ORDER)

    if model_id:
        targets += [cache_path(cache_dir, key, p, model_id) for p in providers]
    # 旧布局（无档位）排在档位文件之后：有精确档位就先用它
    targets += [cache_path(cache_dir, key, p) for p in providers]
    targets.append(cache_path(cache_dir, key))

    seen: set[Path] = set()
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        entry = _load(target)
        if entry is None:
            continue
        if provider and entry.get("provider") not in (provider, None, ""):
            continue
        entry_model = entry.get("model_id")
        if model_id and entry_model not in (model_id, None, ""):
            continue  # 有档位记录且对不上 → 不冒充
        return entry

    # 没指定档位时，档位文件可能只有 __<model>.json：取该 key 最新的一份
    if not model_id:
        for candidate in providers:
            folder = cache_dir / candidate
            if not folder.is_dir():
                continue
            files = sorted(folder.glob(f"{key}__*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for target in files:
                entry = _load(target)
                if entry is not None and (provider is None or entry.get("provider") == provider):
                    return entry
    return None
