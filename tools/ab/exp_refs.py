"""两个补充实验：

E1 多参考图：Qwen（上限 3 张）vs Seedream（10 张）喂多图后，一致性怎么变
E2 换脸根因：Seedream 5.0 的 image 用「字符串 / 数组 / 加 sequential=disabled」三种传法

参考图来自同一角色：定妆照 + A/B 里相似度最高的 4 张同人图。
"""

from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

from envload import load_env
from face import FaceMetric

ROOT = Path(__file__).resolve().parent
ARK_EP = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
QWEN_EP = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
SIZE = "1536x2048"

R = {
    "R1": ROOT / "refs" / "persona_lin.png",
    "R2": ROOT / "out/ab/ark-4.0/p03_waterlight_standing.jpg",
    "R3": ROOT / "out/ab/ark-4.0/p07_pico_seated.jpg",
    "R4": ROOT / "out/ab/ark-4.0/p02_waterlight_closeup.jpg",
    "R5": ROOT / "out/ab/ark-4.0/p05_photon_consult.jpg",
}
PROMPTS = {
    "p01": "a female client in her 30s resting on a treatment bed in a bright modern aesthetic clinic, natural window light, 35mm lens, visible skin texture and pores, no makeup",
    "p05": "a female client in her 30s in conversation with a clinician in a bright clinic room, over-the-shoulder composition, natural window light, 35mm lens, candid",
}
OUT = ROOT / "out" / "ab" / "exp"


def uri(path: Path, max_edge: int = 1024) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def download(url: str, dest: Path) -> Path:
    with httpx.Client(timeout=180, follow_redirects=True) as c:
        r = c.get(url)
        r.raise_for_status()
        data = r.content
    real = dest.with_suffix(".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg")
    real.parent.mkdir(parents=True, exist_ok=True)
    real.write_bytes(data)
    return real


def call_ark(key: str, model: str, prompt: str, images, extra: dict | None, dest: Path, as_array: bool = False):
    body = {"model": model, "prompt": prompt, "size": SIZE, "response_format": "url"}
    if images:
        if isinstance(images, list) and len(images) == 1 and not as_array:
            body["image"] = images[0]          # 单图传字符串（与 A/B 基线一致）
        else:
            body["image"] = images
    if extra:
        body.update(extra)
    t0 = time.perf_counter()
    r = httpx.post(ARK_EP, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=body, timeout=300)
    if r.status_code != 200:
        return None, time.perf_counter() - t0, f"HTTP {r.status_code}: {r.text[:160]}"
    url = r.json().get("data", [{}])[0].get("url")
    if not url:
        return None, time.perf_counter() - t0, "no url"
    return download(url, dest), time.perf_counter() - t0, None


def call_qwen(key: str, model: str, prompt: str, images, extra: dict | None, dest: Path):
    content = [{"image": u} for u in images] + [{"text": prompt}]
    body = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": content}]},
        "parameters": {"n": 1, "size": SIZE.replace("x", "*"), **(extra or {})},
    }
    t0 = time.perf_counter()
    r = httpx.post(QWEN_EP, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=body, timeout=300)
    if r.status_code != 200:
        return None, time.perf_counter() - t0, f"HTTP {r.status_code}: {r.text[:160]}"
    choices = r.json().get("output", {}).get("choices") or []
    url = None
    if choices:
        for c in choices[0].get("message", {}).get("content", []):
            if c.get("image"):
                url = c["image"]
                break
    if not url:
        return None, time.perf_counter() - t0, "no image"
    return download(url, dest), time.perf_counter() - t0, None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    env = load_env(ROOT)
    ark, ds = env["ARK_API_KEY"], env["DASHSCOPE_API_KEY"]
    fm = FaceMetric()
    ref_main = R["R1"]
    rows = []

    def run(tag, fn, key, model, refs, prompt_id, extra=None):
        dest = OUT / f"{tag}_{prompt_id}"
        img, sec, err = fn(key, model, PROMPTS[prompt_id], [uri(R[x]) for x in refs] if refs else None, extra, dest)
        sim = fm.similarity(ref_main, img) if img else None
        rows.append({"exp": tag.split("_")[0], "tag": tag, "model": model, "refs": len(refs) if refs else 0,
                     "prompt": prompt_id, "sim": round(sim, 3) if sim is not None else None,
                     "sec": round(sec, 1), "err": err})
        print(f"{tag:38s} {prompt_id} sim={sim if sim is None else round(sim,3)} {round(sec,1)}s {err or ''}")

    # E1 多参考图
    for pid in ("p01", "p05"):
        run("E1_ark4_1ref", call_ark, ark, "doubao-seedream-4-0-20260415", ["R1"], pid)
        run("E1_ark4_5ref", call_ark, ark, "doubao-seedream-4-0-20260415", ["R1", "R2", "R3", "R4", "R5"], pid)
        run("E1_qwen1ref", call_qwen, ds, "qwen-image-3.0", ["R1"], pid)
        run("E1_qwen3ref", call_qwen, ds, "qwen-image-3.0", ["R1", "R2", "R3"], pid)

    # E2 5.0 换脸根因（只测 Pro，flash 同架构）
    for pid in ("p01", "p05"):
        one = [uri(R["R1"])]
        for tag, images, extra, as_array in (
            ("E2_str", one, None, False),
            ("E2_arr", one, None, True),
            ("E2_seq", one, {"sequential_image_generation": "disabled"}, False),
        ):
            dest = OUT / f"{tag}_{pid}"
            img, sec, err = call_ark(ark, "doubao-seedream-5-0-pro-260628", PROMPTS[pid], images, extra, dest, as_array)
            sim = fm.similarity(ref_main, img) if img else None
            rows.append({"exp": "E2", "tag": tag, "model": "seedream-5.0-pro", "refs": 1,
                         "prompt": pid, "sim": round(sim, 3) if sim is not None else None,
                         "sec": round(sec, 1), "err": err})
            print(f"{tag:38s} {pid} sim={sim if sim is None else round(sim,3)} {round(sec,1)}s {err or ''}")

    (OUT / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细: {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
