"""本地栈图生图的参数扫描：找能把身份相似度拉到可用的配置。

背景：默认配置（4 步 / strength 0.45 / 长 prompt 截到 77 token）下，
本地栈的 I2I 身份相似度只有 0.28/0.15，而闭源 API 是 0.58/0.81。
（这组数字来自**固定 seed 的扫描实验**，与主流程那一轮不是同一批产物，
不可直接混比。）
这轮扫描验证"是不是参数问题"：固定 seed，只变 steps / strength / prompt 长度。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "src"))

# 仓库外依赖走环境变量，缺失时明确报错退出（不静默按当前目录找）
PY = os.environ.get("AP_LOCAL_PYTHON") or str(WS / ".venv-local" / "Scripts" / "python.exe")
MODEL = os.environ.get("AP_LOCAL_MODEL_DIR") or str(WS / "models" / "sdxl-turbo")
if not Path(PY).exists():
    raise SystemExit(f"找不到本地栈解释器：{PY}\n请设置 AP_LOCAL_PYTHON 指向装有 torch(CUDA)+diffusers 的虚拟环境。")
WORKER = WS / "src" / "aesthetics_pipeline" / "provider" / "local_worker.py"
REF = WS / "evidence" / "01_t2i_baseline.png"
OUT = WS / "out" / "local_images"
OUT.mkdir(parents=True, exist_ok=True)

SHORT = (
    "a 28 year old Chinese woman with shoulder length straight black hair parted in the middle, "
    "oval face, soft almond shaped eyes, a small mole on her left cheek, natural thin eyebrows, "
    "in a bright modern aesthetic clinic, documentary photography, natural daylight"
)
NEG = "illustration, 3d render, plastic skin, text, watermark, distorted hands"

# (steps, strength, prompt) 组合
CONFIGS = [
    (4, 0.45, "short"),
    (8, 0.35, "short"),
    (12, 0.30, "short"),
    (8, 0.25, "short"),
    (12, 0.30, "long"),
]

items = []
for steps, strength, ptag in CONFIGS:
    items.append(
        {
            "prompt": SHORT if ptag == "short" else SHORT + ", in Wuhan, resting on a treatment bed, damp glowing fresh skin, calm satisfied expression, healthy even skin, relaxed shoulders, documentary editorial photography, 35mm lens, shallow depth of field, realistic skin texture with visible pores, unretouched, high detail",
            "negative": NEG,
            "width": 896,
            "height": 1152,
            "out_path": str(OUT / f"sweep_{steps}_{int(strength*100)}_{ptag}.png"),
            "input_path": str(REF),
            "strength": strength,
            "seed": 20261024,
            "steps": steps,
        }
    )

proc = subprocess.run(
    [PY, str(WORKER)],
    input=json.dumps({"model_dir": MODEL, "mode": "i2i", "items": items}),
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    timeout=3600,
)
print("returncode:", proc.returncode)
if proc.returncode != 0:
    print("STDERR:", proc.stderr[-1500:])
    raise SystemExit(1)

from aesthetics_pipeline.qc.face import similarity

result = json.loads(proc.stdout.strip().splitlines()[-1])
print(f"device={result.get('device')} load={result.get('load_seconds')}s\n")
print(f"{'steps':>5} {'strength':>8} {'prompt':>6} {'sec':>6} {'similarity':>11}")
for cfg, res in zip(CONFIGS, result["results"]):
    steps, strength, ptag = cfg
    if not res.get("ok"):
        print(f"{steps:>5} {strength:>8} {ptag:>6}   FAIL {res.get('error')}")
        continue
    sim = similarity(res["path"], REF)
    print(f"{steps:>5} {strength:>8} {ptag:>6} {res['seconds']:>6} {sim!s:>11}")
