"""把本机 live 产物导出成随仓库走的 fixtures。

为什么需要：`out/` 与 `data/*.db` 都在 .gitignore 里，新 clone 的机器既没有
回放缓存也没有台账，演示脚本里的"幂等证明"和"断网 replay"两步会直接失效。
本脚本把 out/cache 里的条目复制成 `fixtures/`（图片缩放压缩为 JPEG，
缓存里的路径改指向仓库内），提交后任何机器都能离线回放。

用法：python tools/export_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "src"))

from PIL import Image  # noqa: E402

from aesthetics_pipeline.config import REPO_ROOT  # noqa: E402
from aesthetics_pipeline.provider.cache import read_cache, resolve_local_path  # noqa: E402

MAX_SIDE = 1024
JPEG_QUALITY = 88


def main() -> int:
    live_cache = REPO_ROOT / "out" / "cache"
    fix_cache = REPO_ROOT / "fixtures" / "cache"
    fix_images = REPO_ROOT / "fixtures" / "images"
    fix_cache.mkdir(parents=True, exist_ok=True)
    fix_images.mkdir(parents=True, exist_ok=True)

    if not live_cache.exists():
        print(f"没有 live 缓存目录：{live_cache}\n先跑一次 generate 再导出。")
        return 1

    exported = 0
    # 缓存按后端分目录（cache/<provider>/<key>.json），素材名也带后端，
    # 否则 matrix 与 local 的同一幂等键会互相覆盖。
    for cache_file in sorted(live_cache.rglob("*.json")):
        provider = cache_file.parent.name if cache_file.parent != live_cache else ""
        entry = read_cache(live_cache, cache_file.stem, provider or None)
        if entry is None:
            continue
        provider = entry.get("provider") or provider or "unknown"
        source = resolve_local_path(entry["local_path"])
        if not source.exists():
            print(f"  跳过（素材缺失）: {source}")
            continue

        target = fix_images / f"{provider}_{cache_file.stem}.jpg"
        with Image.open(source) as img:
            img = img.convert("RGB")
            img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
            img.save(target, "JPEG", quality=JPEG_QUALITY, optimize=True)

        fixture_entry = dict(entry)
        fixture_entry["local_path"] = target.relative_to(REPO_ROOT).as_posix()
        fixture_entry["file_name"] = target.name
        fixture_entry["fixture_note"] = (
            f"缩放压缩副本（长边≤{MAX_SIDE}px, JPEG q{JPEG_QUALITY}），仅供离线回放；"
            "原始 2K 产物用 live 模式重新生成"
        )
        fix_entry = fix_cache / provider / f"{cache_file.stem}.json"
        fix_entry.parent.mkdir(parents=True, exist_ok=True)
        fix_entry.write_text(
            json.dumps(fixture_entry, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        exported += 1
        print(f"  {provider:<8} {cache_file.stem[:8]} -> {target.name} "
              f"({target.stat().st_size // 1024}KB)")

    total = sum(p.stat().st_size for p in fix_images.glob("*")) if fix_images.exists() else 0
    print(f"\n导出 {exported} 条，fixtures/images 合计 {total // 1024}KB")
    print("提交 fixtures/ 后，新 clone 的机器可直接 `generate --provider replay` 复现。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
