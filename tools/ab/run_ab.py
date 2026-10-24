"""跑 A/B：固定 10 条 prompt × N 个模型，落图 + 落台账（耗时/成本/错误）。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from providers import build

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out" / "ab"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="*", default=None, help="默认全跑")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    prompts = json.loads((ROOT / "prompts.json").read_text(encoding="utf-8"))["prompts"]
    ref = ROOT / cfg["ref_image"]
    if not ref.exists():
        raise SystemExit(f"参考图不存在: {ref}（先复制角色定妆照到 ab/refs/persona_lin.png）")

    if args.limit:
        prompts = prompts[: args.limit]

    names = args.providers or list(cfg["providers"].keys())
    env = dict(os.environ)
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "results.jsonl"

    for name in names:
        pcfg = cfg["providers"][name]
        print(f"\n=== {name} ({pcfg['label']}) {pcfg['model']} ===")
        if args.dry_run:
            for p in prompts:
                print(f"  [dry] {p['id']} mode={p['mode']}")
            continue
        prov = build(name, pcfg, ref, env)
        for p in prompts:
            dest = OUT / name / f"{p['id']}.png"
            if dest.exists() and dest.stat().st_size > 1000:
                print(f"  [skip] {p['id']}")
                continue
            if args.dry_run:
                print(f"  [dry] {p['id']} mode={p['mode']}")
                continue

            res = None
            for attempt in range(1, args.retries + 2):
                res = prov.generate(p["prompt"], dest, p["mode"])
                if res.ok:
                    break
                print(f"  [retry {attempt}] {p['id']}: {res.error}")
                time.sleep(5)

            row = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "provider": name,
                "label": pcfg["label"],
                "model": pcfg["model"],
                "prompt_id": p["id"],
                "mode": p["mode"],
                "project": p["project"],
                "theme": p["theme"],
                "ok": bool(res and res.ok),
                "path": res.path if res else None,
                "seconds": round(res.seconds, 2) if res else 0,
                "cost_cny": res.cost_cny if res else 0,
                "size": res.size if res else "",
                "request_id": res.request_id if res else None,
                "error": res.error if res else "no result",
            }
            with log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            mark = "ok " if row["ok"] else "FAIL"
            print(f"  [{mark}] {p['id']}  {row['seconds']}s  {row['size']}  {row['error'] or ''}")

    print(f"\n台账: {log}")


if __name__ == "__main__":
    main()
