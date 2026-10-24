"""本地栈真实感对比：SDXL-Turbo 4 步 vs 写实 checkpoint（PhotoReal 系）25~30 步。

同一 prompt、同一 seed、同一尺寸，**只换模型与采样参数**，出并排对照图，
供人工盲评（真实感是主观验收项，没有自动判分）。

用法：
    python tools/compare_photo_vs_turbo.py

依赖：
    AP_LOCAL_PYTHON        本地栈解释器（torch+diffusers）
    AP_LOCAL_MODEL_DIR     turbo 权重目录（默认 models/sdxl-turbo）
    AP_PHOTO_MODEL_DIR     写实权重目录（默认 D:\\Tools\\models\\juggernaut-xl-v9）
    AP_PHOTO_STEPS         写实模型步数（默认 28）
    AP_PHOTO_GUIDANCE      写实模型 guidance（默认 7.5）
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "src"))

from aesthetics_pipeline.config import REPO_ROOT, local_python  # noqa: E402

PHOTO_DIR = Path(os.environ.get("AP_PHOTO_MODEL_DIR") or r"D:\Tools\models\juggernaut-xl-v9")
TURBO_DIR = Path(os.environ.get("AP_LOCAL_MODEL_DIR") or (REPO_ROOT / "models" / "sdxl-turbo"))
PHOTO_STEPS = int(os.environ.get("AP_PHOTO_STEPS") or "28")
PHOTO_GUIDANCE = float(os.environ.get("AP_PHOTO_GUIDANCE") or "7.5")
OUT = REPO_ROOT / "out" / "photo_compare"
REF = REPO_ROOT / "evidence" / "01_t2i_baseline.png"

PROMPT = (
    "in Wuhan, a female client in her 30s resting on a treatment bed in a bright modern "
    "aesthetic clinic, damp glowing fresh skin, calm satisfied expression, healthy even skin, "
    "relaxed shoulders, documentary editorial photography, 35mm lens, shallow depth of field, "
    "realistic skin texture with visible pores, unretouched, high detail"
)
NEG = "illustration, anime, 3d render, cgi, plastic skin, over-smoothed, text, lettering, watermark"
SEED = 20261024


def run_worker(model_dir: Path, mode: str, out_path: Path, steps: int, guidance: float) -> dict:
    payload = {
        "model_dir": str(model_dir),
        "mode": mode,
        "steps": steps,
        "guidance": guidance,
        "items": [
            {
                "prompt": PROMPT,
                "negative": NEG,
                "width": 896,
                "height": 1152,
                "out_path": str(out_path),
                "input_path": str(REF) if mode == "i2i" else None,
                "strength": 0.45,
                "seed": SEED,
            }
        ],
    }
    proc = subprocess.run(
        [str(local_python()), str(WS / "src" / "aesthetics_pipeline" / "provider" / "local_worker.py")],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    if proc.returncode != 0:
        raise SystemExit(f"worker 失败 ({model_dir.name}/{mode}): {(proc.stderr or proc.stdout)[-800:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    if not PHOTO_DIR.exists():
        print(f"找不到写实权重目录 {PHOTO_DIR}\n先跑 tools/fetch_photo_model.py 下载。")
        return 1
    if not REF.exists():
        print(f"找不到参考图 {REF}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    from aesthetics_pipeline.qc.face import similarity
    from aesthetics_pipeline.report import collage

    items: list[tuple[str, Path]] = []
    table_rows: list[tuple[str, str, str]] = []
    for mode in ("t2i", "i2i"):
        for tag, model_dir, steps, guidance in (
            ("turbo", TURBO_DIR, 4, 0.0),
            ("photo", PHOTO_DIR, PHOTO_STEPS, PHOTO_GUIDANCE),
        ):
            out_path = OUT / f"{mode}_{tag}.png"
            print(f"→ {mode}/{tag}（{model_dir.name}, steps={steps}, guidance={guidance}）")
            result = run_worker(model_dir, mode, out_path, steps, guidance)
            if not result["results"][0].get("ok"):
                print(f"   失败: {result['results'][0].get('error')}")
                continue
            secs = result["results"][0]["seconds"]
            sim = similarity(out_path, REF) if mode == "i2i" else None
            items.append((f"{mode} · {tag}", out_path))
            table_rows.append(
                (f"{mode}/{tag}", f"{secs}s", f"{sim}" if sim is not None else "-")
            )

    sheet = collage(
        items,
        OUT / "photo_vs_turbo.png",
        cols=2,
        title="本地栈真实感对比：SDXL-Turbo 4 步  vs  写实 checkpoint 25~30 步（同 prompt / 同 seed）",
    )
    print("\n模式/模型   耗时     身份相似度")
    for mode, secs, sim in table_rows:
        print(f"{mode:<10} {secs:<8} {sim}")
    print(f"\n并排对照图: {sheet}")
    print("真实感是主观验收项——这张图是给人工盲评用的素材，不是自动判分。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
