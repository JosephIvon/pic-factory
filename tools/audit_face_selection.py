"""人脸选脸策略审计：为什么"检测分最高的脸"不能用来做身份比对。

实测争议点（来自复审）：`evidence/02_i2i_same_person.jpg` 画面里有**两个人**
（顾客 + 医生）。原实现取 `det_score` 最高的那张脸做比对；如果那张脸是医生，
分数会掉到 0.0x —— 于是"同一人分数低"可能是**取错人**造成的假象，
而不是 ArcFace 不具判别力。

本脚本把三种选脸策略在同一批图上并排算出来：
  A. det_score 最高      —— 原实现（embedding() / faceswap 的选脸方式）
  B. 面积最大            —— "画面里最大的人"
  C. 与参考照最相似      —— 新实现（gallery 语义：这张图里有没有目标那个人）

用法：
    python tools/audit_face_selection.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from aesthetics_pipeline.config import REPO_ROOT  # noqa: E402

REFERENCE = REPO_ROOT / "evidence" / "01_t2i_baseline.png"


_APP = None


def _app():
    """insightface 只加载一次（每次重新加载要好几秒）。"""
    global _APP
    if _APP is None:
        from insightface.app import FaceAnalysis

        root = os.environ.get("AP_INSIGHTFACE_ROOT") or str(Path.home() / ".insightface")
        _APP = FaceAnalysis(name="buffalo_l", root=root)
        _APP.prepare(ctx_id=0, det_size=(640, 640))
    return _APP


def _faces(path: Path) -> list[dict]:
    """检出人脸，给出检测分、面积占比、与参考照的相似度。"""
    app = _app()

    img = np.asarray(Image.open(path).convert("RGB"))
    img = img[:, :, ::-1].copy()  # insightface 期望 BGR（与生产代码保持一致）
    faces = app.get(img)
    h, w = img.shape[:2]
    out = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        area = max(0.0, float(x2 - x1) * float(y2 - y1)) / float(h * w)
        out.append(
            {
                "det_score": round(float(getattr(f, "det_score", 0.0)), 4),
                "area": round(area, 4),
                "emb": np.asarray(f.normed_embedding, dtype=np.float64),
                "box": [int(x1), int(y1), int(x2), int(y2)],
            }
        )
    return out


def main() -> int:
    if not REFERENCE.exists():
        print(f"找不到参考照 {REFERENCE}")
        return 1

    ref_faces = _faces(REFERENCE)
    if not ref_faces:
        print("参考照里检不到人脸")
        return 1
    # 参考照是单人定妆照：取面积最大那张作为"目标人物"
    ref = max(ref_faces, key=lambda f: f["area"])
    print(f"参考照 {REFERENCE.name}: {len(ref_faces)} 张脸，采用面积最大的一张 "
          f"(det={ref['det_score']}, area={ref['area']})\n")

    targets: list[Path] = []
    for pattern in ("evidence/*.png", "evidence/*.jpg", "evidence/batch/*"):
        targets += sorted((REPO_ROOT / pattern).parent.glob(Path(pattern).name))
    targets += sorted((REPO_ROOT / "out" / "images").glob("*"))
    targets += sorted((REPO_ROOT / "out" / "swapped").rglob("*.png"))
    targets += sorted((REPO_ROOT / "out" / "local_images").glob("*"))
    targets = [p for p in targets if p.exists() and p != REFERENCE]
    # 去掉归档副本（拼图/对照图），它们本身是多脸拼版，不代表单张产出
    targets = [
        p for p in targets
        if "contact_sheet" not in p.name and "photo_vs_turbo" not in p.name
    ]

    print(f"{'图片':<42}{'脸数':>4}{'A det最高':>12}{'B 面积最大':>12}{'C 最相似':>10}   说明")
    print("-" * 100)
    rows = []
    for path in targets:
        faces = _faces(path)
        if not faces:
            print(f"{path.name:<42}{0:>4}{'-':>12}{'-':>12}{'-':>10}   画面无人脸")
            continue

        sims = [float(np.dot(f["emb"], ref["emb"])) for f in faces]
        a = sims[max(range(len(faces)), key=lambda i: faces[i]["det_score"])]
        b = sims[max(range(len(faces)), key=lambda i: faces[i]["area"])]
        c = max(sims)
        note = ""
        if len(faces) > 1 and abs(a - c) > 0.05:
            note = "⚠ det最高 ≠ 最相似（多脸图取错人）"
        print(f"{path.name:<42}{len(faces):>4}{a:>12.4f}{b:>12.4f}{c:>10.4f}   {note}")
        if len(faces) > 1:
            for f, s in sorted(
                zip(faces, sims), key=lambda x: x[0]["det_score"], reverse=True
            ):
                print(
                    f"    └ face det={f['det_score']:.4f} area={f['area']:.4f} "
                    f"box={f['box']} sim={s:.4f}"
                )
        rows.append((path.name, len(faces), a, b, c))

    if not rows:
        return 0

    def stats(idx: int, label: str) -> None:
        vals = [r[idx] for r in rows]
        print(f"\n{label}: n={len(vals)} min={min(vals):.4f} max={max(vals):.4f} "
              f"均值={sum(vals) / len(vals):.4f}")

    print()
    stats(2, "A 选脸（det 最高）分数分布")
    stats(4, "C 选脸（与参考照最相似）分数分布")
    print(
        "\n注意：这些是**本项目自有素材**上的分布，属于「自洽性验证」，\n"
        "不是外部基准的 ROC/FPR —— 要谈判别力需要外部标注样本（见 README 第 6 节）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
