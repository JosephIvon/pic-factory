"""ArcFace 人脸一致性度量：YuNet 检测 5 点 → 仿射对齐 112x112 → w600k_r50 ONNX 嵌入。

不依赖 insightface 包（本机装不上），只用 opencv + onnxruntime + 本机已有的 buffalo_l 权重。
输出与标准 ArcFace 同口径的余弦相似度，用于 A/B 的一致性对比。
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HOME = Path.home()
MODEL_DIR = HOME / ".insightface" / "models" / "buffalo_l"
ARCFACE = MODEL_DIR / "w600k_r50.onnx"
YUNET = Path(__file__).with_name("face_detection_yunet_2023mar.onnx")
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# ArcFace 标准 5 点模板（112x112）
ARC_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],  # 左眼
        [73.5318, 51.5014],  # 右眼
        [56.0252, 71.7366],  # 鼻尖
        [41.5493, 92.3655],  # 左嘴角
        [70.7299, 92.2041],  # 右嘴角
    ],
    dtype=np.float32,
)


class FaceMetric:
    def __init__(self, det_size: int = 640) -> None:
        if not ARCFACE.exists():
            raise RuntimeError(f"缺少 ArcFace 权重: {ARCFACE}")
        _ensure_yunet()
        self.det = cv2.FaceDetectorYN.create(str(YUNET), "", (det_size, det_size), 0.6, 0.3, 5000)
        self.sess = ort.InferenceSession(str(ARCFACE), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    # ---------- 检测 + 对齐 ----------
    def _align(self, img_bgr: np.ndarray) -> np.ndarray | None:
        h, w = img_bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None
        # 取面积最大的人脸
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        pts = faces[0][4:14].reshape(5, 2).astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(pts, ARC_TEMPLATE, method=cv2.LMEDS)
        if M is None:
            return None
        return cv2.warpAffine(img_bgr, M, (112, 112), borderValue=0.0)

    # ---------- 嵌入 ----------
    def embedding(self, path: str | Path) -> np.ndarray | None:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        aligned = self._align(img)
        if aligned is None:
            return None
        blob = cv2.dnn.blobFromImage(
            aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
        )
        out = self.sess.run(None, {self.input_name: blob})[0].reshape(-1)
        norm = np.linalg.norm(out)
        return out / norm if norm > 0 else None

    def similarity(self, a: str | Path, b: str | Path) -> float | None:
        ea, eb = self.embedding(a), self.embedding(b)
        if ea is None or eb is None:
            return None
        return float(np.dot(ea, eb))


def _ensure_yunet() -> None:
    if YUNET.exists():
        return
    YUNET.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(YUNET_URL, YUNET)


if __name__ == "__main__":
    import sys

    fm = FaceMetric()
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if base is None:
        print(json.dumps({"error": "usage: face.py <dir>"}))
        raise SystemExit(2)
    ref = base / "01_t2i_baseline.png"
    others = ["02_i2i_same_person.jpg", "03_t2i_same_description_differs.png"]
    out = {"ref": ref.name}
    for name in others:
        out[name] = fm.similarity(ref, base / name)
    print(json.dumps(out, ensure_ascii=False))
