"""通道顺序验证：insightface 的 app.get() 到底期望 RGB 还是 BGR。

现象：一张大笑的人脸特写（evidence/15_local_a004118539d12515.png）用 RGB 传入时
检不到任何脸。insightface 的检测器是按 OpenCV 惯例训练的（BGR），
传 RGB 等于把肤色通道反了 —— 检测率会掉。

本脚本对同一批图分别用 RGB / BGR 传入，比较：
  * 检出的人脸数
  * 与定妆照的相似度

用法：python tools/check_channel_order.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "src"))

from aesthetics_pipeline.config import REPO_ROOT  # noqa: E402

REFERENCE = REPO_ROOT / "evidence" / "01_t2i_baseline.png"


def main() -> int:
    from insightface.app import FaceAnalysis

    root = os.environ.get("AP_INSIGHTFACE_ROOT") or str(Path.home() / ".insightface")
    app = FaceAnalysis(name="buffalo_l", root=root)
    app.prepare(ctx_id=0, det_size=(640, 640))

    def faces_of(path: Path, order: str):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        img = rgb[:, :, ::-1].copy() if order == "bgr" else rgb
        faces = app.get(img)
        out = []
        for f in faces:
            out.append(
                {
                    "det": float(getattr(f, "det_score", 0.0)),
                    "emb": np.asarray(f.normed_embedding, dtype=np.float64),
                }
            )
        return out

    ref_rgb = faces_of(REFERENCE, "rgb")
    ref_bgr = faces_of(REFERENCE, "bgr")
    print(f"参考照 {REFERENCE.name}: RGB 检出 {len(ref_rgb)} 张 / BGR 检出 {len(ref_bgr)} 张")
    ref = (ref_bgr or ref_rgb)[0]["emb"]

    # 全部指向仓内证据：原图曾放在上游会话工作区的 out/ 下，
    # 那是临时目录、不随仓库走——现在统一收进 evidence/。
    targets = [
        REPO_ROOT / "evidence" / "15_local_a004118539d12515.png",
        REPO_ROOT / "evidence" / "12_local_i2i_generated.png",
        REPO_ROOT / "evidence" / "10_api_i2i_generated.png",
        REPO_ROOT / "evidence" / "02_i2i_same_person.jpg",
        REPO_ROOT / "evidence" / "batch" / "batch_03_chengdu_photon.jpg",
    ]

    print(f"\n{'图片':<34}{'RGB 脸数':>10}{'RGB 最相似':>12}{'BGR 脸数':>10}{'BGR 最相似':>12}")
    print("-" * 80)
    for path in targets:
        if not path.exists():
            print(f"{path.name:<34} (缺文件)")
            continue
        fr, fb = faces_of(path, "rgb"), faces_of(path, "bgr")

        def best(fs):
            if not fs:
                return None
            return max(float(np.dot(f["emb"], ref)) for f in fs)

        sr, sb = best(fr), best(fb)
        print(
            f"{path.name:<34}{len(fr):>10}{(f'{sr:.4f}' if sr is not None else '-'):>12}"
            f"{len(fb):>10}{(f'{sb:.4f}' if sb is not None else '-'):>12}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
