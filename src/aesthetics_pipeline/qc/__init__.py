"""质检三路闸门。

路1 硬规则（确定性）：可解码 / 尺寸 / 比例 / 画面内文字 / 重复图
路2 数值闸（一致性）：人脸 embedding 余弦相似度 vs 定妆照
路3 VLM 语义（软信号）：写实感 / 场景匹配度

**为什么不能只靠 VLM**：实测 describe_images 能读出画面里的乱码文字，却判定
"garbled: false"、"worst_issue: 无明显问题"。所以文字类问题必须走路1 的 OCR 硬规则，
VLM 的分数只做软信号，不参与出厂拦截（除非显式开启严格模式）。
"""

from __future__ import annotations

from .face import FaceGate, similarity
from .rules import dhash, image_has_text, rule_findings
from .vlm import VlmGate

__all__ = [
    "FaceGate",
    "VlmGate",
    "rule_findings",
    "image_has_text",
    "similarity",
    "dhash",
]
