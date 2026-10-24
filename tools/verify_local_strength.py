"""验证本地栈的 strength 真的可控。

背景：worker 之前用 AutoPipelineForImage2Image.from_pipe，继承了 T2I 类，
image/strength 被静默吞掉——三档 strength 产出字节完全相同（图生图没生效）。
现在显式加载 StableDiffusionXLImg2ImgPipeline，并固定 seed 只变 strength。
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
# 仓库外依赖走环境变量，缺失时明确报错退出（不静默按当前目录找）
PY = os.environ.get("AP_LOCAL_PYTHON") or str(WS / ".venv-local" / "Scripts" / "python.exe")
MODEL = os.environ.get("AP_LOCAL_MODEL_DIR") or str(WS / "models" / "sdxl-turbo")
if not Path(PY).exists():
    raise SystemExit(f"找不到本地栈解释器：{PY}\n请设置 AP_LOCAL_PYTHON 指向装有 torch(CUDA)+diffusers 的虚拟环境。")
WORKER = WS / "src" / "aesthetics_pipeline" / "provider" / "local_worker.py"
REF = WS / "evidence" / "01_t2i_baseline.png"
OUT = WS / "out" / "local_images"
OUT.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "in Wuhan, a female client in her 30s resting on a treatment bed in a bright modern "
    "aesthetic clinic, damp glowing fresh skin, calm satisfied expression, healthy even skin, "
    "relaxed shoulders, a 28 year old Chinese woman with shoulder length straight black hair "
    "parted in the middle, oval face, soft almond shaped eyes, a small mole on her left cheek"
)
NEG = "illustration, anime, 3d render, cgi, plastic skin, over-smoothed, text, lettering, watermark"


def run(mode, items):
    proc = subprocess.run(
        [PY, str(WORKER)],
        input=json.dumps({"model_dir": MODEL, "mode": mode, "items": items}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    print(f"[{mode}] returncode:", proc.returncode)
    if proc.returncode != 0:
        print("STDERR tail:", proc.stderr[-1500:])
        raise SystemExit(1)
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    print(f"[{mode}] device:", result.get("device"), "load_seconds:", result.get("load_seconds"))
    for r in result["results"]:
        print("   ", r.get("ok"), r.get("seconds"), Path(r.get("path") or "").name, r.get("error"))
    return result


# 1) 文生图（1 张）
run("t2i", [
    {
        "prompt": PROMPT, "negative": NEG, "width": 896, "height": 1152,
        "out_path": str(OUT / "local_t2i.png"), "input_path": None,
        "strength": None, "seed": 12345,
    }
])

# 2) 图生图 strength 扫描（固定 seed，只变 strength）
items = [
    {
        "prompt": PROMPT, "negative": NEG, "width": 896, "height": 1152,
        "out_path": str(OUT / f"strength_{int(s*100):02d}.png"),
        "input_path": str(REF), "strength": s, "seed": 12345,
    }
    for s in (0.25, 0.45, 0.70)
]
run("i2i", items)

print("\n--- strength 是否真的改变输出 ---")
digests = {}
for s in (0.25, 0.45, 0.70):
    path = OUT / f"strength_{int(s*100):02d}.png"
    if path.exists():
        digests[s] = hashlib.md5(path.read_bytes()).hexdigest()
        print(f"  strength={s}: {digests[s]} ({path.stat().st_size} bytes)")
if len(set(digests.values())) == len(digests) == 3:
    print("结论：三档 strength 产出互不相同 -> strength 生效（这是闭源 API 给不了的能力）")
else:
    print("结论：产出存在重复 -> strength 仍未生效")
