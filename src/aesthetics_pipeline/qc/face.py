"""路2：人脸 embedding 数值闸（人物一致性）。

用 insightface 的 ArcFace 嵌入算余弦相似度。

**选脸规则（修掉的一处机制缺陷）**：早先的实现对每张图取「检测分最高」的那张脸。
画面里有多个人时（实测 `evidence/02_i2i_same_person.jpg` 同时有顾客和医生），
检测分最高的那张未必是目标人物，同一个人会被算成"不像"——
分布重叠里有一部分是**取错人**造出来的假象。
现在改成 gallery 语义：**这张图里有没有目标那个人**
（取所有脸里与参考照最相似的那一对）。
复算工具：`tools/audit_face_selection.py`（三种选脸策略并排出数）。

阈值不拍脑袋：由 `tools/recalibrate_face_threshold.py` 在**同域**样本上标定——
45 张域内正样本（剔遮挡、排除人物禁模型后 32 张，最低 0.6406）与 7 张同域难负
（同风格但画出来是别人，最高 0.4918）取间隙中点 **T=0.57**（证据：
`docs/ab-evidence/threshold-calibration-production.json`；跨域样本只作 sanity，
跨域标定给假线）。**n=7 的 FAR 无统计意义**，扩到 ≥150 负样本的复核留待开发完成后；
可用 `AP_FACE_THRESHOLD` 覆盖。

模型权重首次使用时自动下载（insightface 的 buffalo_l），离线环境请提前预热。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_app = None
_load_failed = False

# 本项目同域样本标定的生产尺阈值（tools/recalibrate_face_threshold.py 产出）。
CALIBRATED_THRESHOLD = 0.57


def _face_app():
    """懒加载 insightface。失败时返回 None，闸门标记为 skipped 而不是伪造通过。"""
    global _app, _load_failed
    if _app is not None or _load_failed:
        return _app
    try:
        from insightface.app import FaceAnalysis

        root = os.environ.get("AP_INSIGHTFACE_ROOT") or str(Path.home() / ".insightface")
        app = FaceAnalysis(name="buffalo_l", root=root)
        app.prepare(ctx_id=0 if _cuda() else -1, det_size=(640, 640))
        _app = app
    except Exception:
        _load_failed = True
        _app = None
    return _app


def _cuda() -> bool:
    try:
        import onnxruntime

        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False


def _detect(path: str | Path) -> list[dict[str, Any]] | None:
    """检出人脸。

    返回 None = 人脸模型不可用；返回 [] = 画面里确实没有人脸。
    两者必须区分：前者是"没法判断"，后者是"判断为不合格"。
    """
    app = _face_app()
    if app is None:
        return None
    import numpy as np

    # **必须转成 BGR 再送进 insightface**：它的检测器按 OpenCV 惯例训练（输入是 BGR）。
    # 传 RGB 等于把肤色通道反了，实测会**整张脸都检不到**
    # （evidence/15_local_a004118539d12515.png：RGB 检 0 张 / BGR 检 1 张）。
    # 换脸那条链路用 cv2.imread（本来就是 BGR），两边必须一致。
    rgb = np.asarray(_pil(path).convert("RGB"))
    img = rgb[:, :, ::-1].copy()
    faces = app.get(img)
    h, w = img.shape[:2]
    out: list[dict[str, Any]] = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        out.append(
            {
                "emb": np.asarray(f.normed_embedding, dtype=np.float64),
                "det_score": float(getattr(f, "det_score", 0.0)),
                "area": max(0.0, float(x2 - x1) * float(y2 - y1)) / float(h * w),
            }
        )
    return out


def _cos(a: Any, b: Any) -> float:
    """insightface 的 normed_embedding 已归一化，点积即余弦相似度。"""
    import numpy as np

    return float(np.dot(a, b))


def _pil(path: str | Path):
    from PIL import Image

    return Image.open(path)


def similarity(a: str | Path, b: str | Path) -> float | None:
    """两张图之间**最相似的一对脸**的余弦相似度；任一张检不到人脸则 None。

    语义是 gallery 检索：**候选图里只要有一张脸像参考照那个人，就算命中**。
    单脸图与旧实现等价；多脸图这是修掉"取错人"的关键。
    """
    fa, fb = _detect(a), _detect(b)
    if not fa or not fb:
        return None
    return round(max(_cos(x["emb"], y["emb"]) for x in fa for y in fb), 4)


class FaceGate:
    """人脸一致性闸门。"""

    def __init__(self, threshold: float | None = None) -> None:
        # 阈值可由环境变量覆盖；默认走标定文件，缺省则用模型常用参考值（会标注来源）。
        env = os.environ.get("AP_FACE_THRESHOLD")
        if env:
            self.threshold = float(env)
            self.threshold_source = "env AP_FACE_THRESHOLD"
        elif threshold is not None:
            self.threshold = float(threshold)
            self.threshold_source = "caller"
        else:
            self.threshold = CALIBRATED_THRESHOLD
            self.threshold_source = (
                "本项目同域样本标定（T=0.57，负最高0.4918/合格正最低0.6406 取中点；"
                "docs/ab-evidence/threshold-calibration-production.json；n=7 待扩样复核）"
            )

    def detect(self, candidate: str | Path) -> dict[str, Any]:
        """只做"画面里有没有人脸/有几张脸"的检查。

        用于文生图任务：那些任务本来就没有指定人物，拿定妆照去比相似度
        没有意义（实测会把随机路人判成"不像目标人物"而误杀）。

        `skipped` 与 `check()` 同一把尺：**没检成**（模型不可用）时为 True，
        区别于"检了、确实没有"（count == 0）。
        """
        faces = _detect(candidate)
        if faces is None:
            return {
                "available": False,
                "count": None,
                "skipped": True,
                "reason": "人脸模型不可用",
            }
        return {
            "available": True,
            "count": len(faces),
            "skipped": False,
            "reason": "" if faces else "画面中未检出人脸",
        }

    def check(self, candidate: str | Path, reference: str | Path) -> dict[str, Any]:
        """比对候选图与参考照。

        返回里带 `count`（候选图检出的人脸数）：
        **人物任务画面里没有人脸 ≠ 通过**——那是"没法校验身份"，
        属于硬规则要拦的情况（见 cli.qc 的"人物任务必须有人脸"）。
        """
        faces = _detect(candidate)
        refs = _detect(reference)
        base: dict[str, Any] = {
            "ok": True,
            "skipped": True,
            "similarity": None,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
        }
        if faces is None or refs is None:
            return {**base, "count": None, "reason": "人脸模型不可用，本项跳过（不伪造通过分数）"}
        if not faces:
            return {**base, "count": 0, "reason": "画面中未检出人脸"}
        if not refs:
            return {**base, "count": len(faces), "reason": "参考照检不到人脸，本项跳过"}
        score = round(max(_cos(x["emb"], y["emb"]) for x in faces for y in refs), 4)
        ok = score >= self.threshold
        return {
            "ok": ok,
            "skipped": False,
            "similarity": score,
            "count": len(faces),
            "reason": "" if ok else f"人脸相似度 {score} 低于阈值 {self.threshold}",
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
        }


def missing_face_reason(has_character: bool, count: int | None) -> str | None:
    """人物任务画面里没有人脸 → 拒收理由；不该拒时返回 None。

    **「检不到就跳过」不等于「合格」**：那是"没法校验身份"，
    不是"身份没问题"。人物任务画面里没有人脸 = 交付物里没有那个人，
    必须拦。（count 为 None 表示人脸模型不可用 —— 那是"没法判断"，
    这时不当成"没有人脸"，不误杀；模型不可用由 `face_gate_unavailable_reason` 拦。）
    """
    if has_character and count == 0:
        return "人物任务画面未检出人脸：身份一致性无从校验（不放行）"
    return None


def face_gate_unavailable_reason(
    has_character: bool, face_detail: dict[str, Any]
) -> str | None:
    """人物任务但人脸闸**没给出判定** → 拒收理由；不该拒时返回 None。

    与 OCR 闸同口径（`qc/rules.py`：「文字闸未执行，不能记为合格」）：
    **闸门没跑 ≠ 合格**。人物任务的身份一致性没被校验过，就不能记为合格。

    三种"没法判断"要分得开（别把"没检成"说成"没有人脸"）：
      * count is None  —— 人脸模型不可用（连检都没检）；
      * count == 0     —— 画面里确实没有人脸（`missing_face_reason` 已给理由）；
      * count > 0 且 skipped —— 参考照检不到人脸。

    非人物任务不适用：场景图不需要身份校验，不能因为缺人脸模型被整批拒收。
    """
    if not has_character or not face_detail.get("skipped"):
        return None
    count = face_detail.get("count")
    if count is None:
        return "人脸模型不可用：人脸闸未执行，不能记为合格（装好模型后重跑 qc --reset）"
    if count == 0:
        return None  # missing_face_reason 已覆盖
    return "参考照检不到人脸：身份一致性无从校验（不放行）"


def should_reject_on_identity(face_result: dict[str, Any], phase: str) -> bool:
    """身份分是否构成拒收理由。

    phase:
      "generate" —— **永不**因身份分拒收。阈值在「生成图 vs 定妆照」这个口径上
                    不具判别力（实测同一人生成图 0.2022~0.7350、不同人生成图
                    0.3182~0.4922，分布重叠），拿它拦中间产物会误杀；
                    此阶段只标记漂移，交给后置换脸兜底。
                    注意：这不是"ArcFace 分不出人"——同一人的**真人照片**
                    能到 0.6551，明显高于不同人的 0.3463（见 README 第 6 节），
                    真正的问题是**生成方差**：模型没把同一张脸稳定画出来。
      "post_swap" —— 出厂判定。换脸后仍低于阈值才拒收（此时比对的是最终产物）。
                    换脸后分数实测 0.8787~0.9016，与阈值差得很远，
                    所以这一步实际校验的是"**换脸有没有生效**"，不是身份判别。

    **skipped（闸门没跑成）在出厂阶段必须拒收**：与 OCR 闸同口径——
    「文字闸未执行，不能记为合格」；人脸闸同理，"换脸换不出脸"或
    "人脸模型不可用"都意味着最终产物**没被校验过**，不能出厂放行。
    生成阶段仍然不拦（那里本来就不因身份拒收）。
    """
    if face_result.get("skipped"):
        return phase == "post_swap"
    if phase == "generate":
        return False
    score = face_result.get("similarity")
    if score is None:
        return False
    threshold = face_result.get("threshold", CALIBRATED_THRESHOLD)
    return score < threshold
