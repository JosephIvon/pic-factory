"""复核：对每张图逐脸做 5 点对齐嵌入，检查台账里的 sim 是"主脸"还是"副作用脸"算出来的。

输出四列：det 分数、框、该脸相似度；并给出"按最高分取脸"的结果（台账口径）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from face import ARC_TEMPLATE, FaceMetric

ROOT = Path(__file__).resolve().parent
REF = ROOT / "refs" / "persona_lin.png"


def faces_with_sims(fm: FaceMetric, path: Path, ref_v: np.ndarray):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    fm.det.setInputSize((w, h))
    _, faces = fm.det.detect(img)
    if faces is None:
        return []
    out = []
    for f in faces:
        score = float(f[-1])
        box = tuple(int(v) for v in f[:4])
        pts = f[4:14].reshape(5, 2).astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(pts, ARC_TEMPLATE, method=cv2.LMEDS)
        if M is None:
            continue
        aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
        blob = cv2.dnn.blobFromImage(aligned, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        v = fm.sess.run(None, {fm.input_name: blob})[0].reshape(-1)
        v = v / np.linalg.norm(v)
        out.append({"det": round(score, 3), "box": box, "sim": round(float(v @ ref_v), 3)})
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fm = FaceMetric()
    ref_v = np.asarray(fm.embedding(REF), dtype=np.float32)

    rows = [json.loads(x) for x in (ROOT / "out/ab/results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    latest = {}
    for r in rows:
        latest[(r["provider"], r["prompt_id"])] = r

    print(f"{'provider':14s} {'prompt':26s} {'台账sim':>7s} {'主脸sim':>7s} {'脸数':>3s}  全部脸(det/sim)")
    for (prov, pid), r in sorted(latest.items()):
        if not r["ok"] or r["mode"] != "i2i":
            continue
        p = Path(r["path"])
        if not p.exists():
            alt = list(p.parent.glob(p.stem + ".*"))
            if not alt:
                continue
            p = alt[0]
        infos = faces_with_sims(fm, p, ref_v)
        if not infos:
            print(f"{prov:14s} {pid:26s} {r['sim'] if 'sim' in r else '-':>7}      无脸")
            continue
        best = max(infos, key=lambda x: x["det"])
        detail = "; ".join(f"{i['det']:.2f}/{i['sim']:+.2f}" for i in infos)
        print(f"{prov:14s} {pid:26s} {'':>7s} {best['sim']:>+7.3f} {len(infos):>3d}  {detail}")


if __name__ == "__main__":
    main()
