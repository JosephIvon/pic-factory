"""fixtures 压缩副本 vs 原图：身份分漂移了多少。

fixtures/ 是随仓库提交的离线回放素材（长边 ≤1024px、JPEG q88），
压缩会让身份分轻微漂移 —— 这条要用**当前口径**重新测，
不能沿用旧选脸规则下测的数字。

用法：python tools/fixture_drift.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "src"))

from aesthetics_pipeline.qc.face import similarity  # noqa: E402


def main() -> int:
    import sqlite3

    ref = WS / "evidence" / "01_t2i_baseline.png"
    fixtures = WS / "fixtures"
    cache_dir = fixtures / "cache"

    # 原图路径以**台账**为准（台账是唯一事实来源）；fixtures 缓存里指的是压缩副本自己
    db = WS / "data" / "pipeline.db"
    originals: dict[tuple[str, str], str] = {}
    if db.exists():
        conn = sqlite3.connect(db)
        for key, provider, lp in conn.execute(
            "SELECT idempotency_key, provider, local_path FROM assets"
        ):
            if lp:
                originals[(provider, key)] = lp

    rows = []
    for entry_path in sorted(cache_dir.rglob("*.json")):
        entry = json.loads(entry_path.read_text(encoding="utf-8"))
        key = entry_path.stem
        provider = entry_path.parent.name
        fixture = fixtures / "images" / f"{provider}_{key}.jpg"
        original = originals.get((provider, key))
        if not fixture.exists():
            print(f"缺压缩副本 {fixture}")
            continue
        s_fix = similarity(fixture, ref)
        s_org = similarity(original, ref) if original else None
        rows.append((f"{provider}/{key}", s_org, s_fix))

    print(f"{'任务':<28}{'原图':>10}{'压缩副本':>12}{'漂移':>10}")
    print("-" * 62)
    drifts = []
    for name, org, fix in rows:
        drift = (fix - org) if (org is not None and fix is not None) else None
        if drift is not None:
            drifts.append(drift)
        fmt = lambda v: f"{v:.4f}" if v is not None else "-"  # noqa: E731
        print(f"{name:<28}{fmt(org):>10}{fmt(fix):>12}{fmt(drift):>10}")
    if drifts:
        print(
            f"\n漂移范围 {min(drifts):+.4f} ~ {max(drifts):+.4f}"
            f"（n={len(drifts)}，长边 ≤1024px + JPEG q88）"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
