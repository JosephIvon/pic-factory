"""路1：确定性硬规则。

这些判定不依赖任何模型输出，同样的输入必得同样的结论，所以可以直接用于"出厂拦截"。

最要紧的一条是**画面内文字检测**：实测主题词会被模型画成黑板字、屏幕字、桌面名牌，
还带乱码汉字。医美内容里出现乱码招牌是直接翻车，所以 OCR 命中即拒。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

MIN_SIDE = 512
OCR_MIN_SCORE = 0.5

_ocr_engine = None
_ocr_failed = False


def _ocr():
    """懒加载 OCR 引擎。加载失败时降级为"未检测"，而不是伪造通过。"""
    global _ocr_engine, _ocr_failed
    if _ocr_engine is not None or _ocr_failed:
        return _ocr_engine
    try:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()
    except Exception:
        _ocr_failed = True
        _ocr_engine = None
    return _ocr_engine


def image_has_text(path: str | Path, min_score: float = OCR_MIN_SCORE) -> list[str]:
    """返回画面里识别到的文字列表；空列表表示没检测到。"""
    engine = _ocr()
    if engine is None:
        return []
    try:
        result, _ = engine(str(path))
    except Exception:
        return []
    texts: list[str] = []
    for item in result or []:
        try:
            text, score = item[1], float(item[2])
        except (IndexError, TypeError, ValueError):
            continue
        if score >= min_score and text.strip():
            texts.append(text.strip())
    return texts


def dhash(path: str | Path, hash_size: int = 8) -> int:
    """差值感知哈希，用于近似重复检测。"""
    with Image.open(path) as img:
        small = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(small.getdata())
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hash_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def rule_findings(
    path: str | Path,
    expected_aspect: str | None = None,
    known_hashes: dict[str, int] | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    """跑一遍硬规则。返回 {ok, reasons, details}。"""
    reasons: list[str] = []
    details: dict[str, Any] = {}

    path = Path(path)
    if not path.exists():
        return {"ok": False, "reasons": [f"文件不存在: {path}"], "details": {}}

    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
    except Exception as exc:
        return {"ok": False, "reasons": [f"图片无法解码: {exc}"], "details": {}}

    details["width"] = width
    details["height"] = height

    if min(width, height) < MIN_SIDE:
        reasons.append(f"短边 {min(width, height)}px 小于 {MIN_SIDE}px")

    if expected_aspect:
        want = expected_aspect.replace(":", "/")
        try:
            want_w, want_h = (float(x) for x in want.split("/"))
        except ValueError:
            want_w = want_h = None
        if want_w and want_h:
            got = width / height
            target = want_w / want_h
            if abs(got - target) / target > 0.02:
                reasons.append(f"比例 {width}x{height} 与渠道要求的 {expected_aspect} 不符")

    texts = image_has_text(path)
    details["text_found"] = texts
    # 文字闸"没跑"与"跑了没查到"必须可区分：rapidocr 属 [qc] extra，
    # 缺依赖时 image_has_text 返回 []，静默绿等于最强硬闸凭空消失——
    # 与人脸闸"检不到人脸=没校验，不是合格"同一条教义（Trae S2）。
    details["ocr_available"] = _ocr() is not None
    if texts:
        reasons.append(f"画面内出现文字（医美内容翻车点）: {'; '.join(texts[:5])}")
    elif not details["ocr_available"]:
        reasons.append("OCR 引擎不可用：文字闸未执行，不能记为合格（安装 [qc] extra 后重跑 qc）")

    if known_hashes:
        current = dhash(path)
        details["dhash"] = current
        for other_key, other_hash in known_hashes.items():
            if other_key == key:
                continue
            distance = hash_distance(current, other_hash)
            if distance <= 6:
                reasons.append(f"与已有产出 {other_key} 近似重复（dhash 距离 {distance}）")
                break

    return {"ok": not reasons, "reasons": reasons, "details": details}
