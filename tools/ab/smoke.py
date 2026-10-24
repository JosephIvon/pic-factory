"""冒烟：用一条 prompt 验证某个 provider 的请求形状，不烧批量预算。

用法：python smoke.py ark-4.0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from envload import load_env
from providers import build

ROOT = Path(__file__).resolve().parent
PROMPT = (
    "a bright modern aesthetic clinic treatment room, white treatment bed, "
    "large window with soft daylight, no people, 24mm lens, "
    "realistic architectural photography, no text"
)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    name = sys.argv[1] if len(sys.argv) > 1 else "ark-4.0"
    mode = sys.argv[2] if len(sys.argv) > 2 else "t2i"
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    env = load_env(ROOT)

    for k in ("ARK_API_KEY", "DASHSCOPE_API_KEY"):
        v = env.get(k) or ""
        print(f"{k}: {'present len=' + str(len(v)) if v else 'MISSING'}")

    prov = build(name, cfg["providers"][name], ROOT / cfg["ref_image"], env)
    res = prov.generate(PROMPT, ROOT / "out" / f"smoke_{name}.png", mode)
    print(f"ok={res.ok} size={res.size} seconds={round(res.seconds, 1)}")
    if res.error:
        print(f"error={res.error}")


if __name__ == "__main__":
    main()
