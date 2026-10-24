"""硬规则闸门。用仓库里的真实产出做样本。"""

from __future__ import annotations

import pytest

from aesthetics_pipeline import config
from aesthetics_pipeline.qc.rules import image_has_text, rule_findings

EVIDENCE = config.EVIDENCE_DIR
CLEAN = EVIDENCE / "01_t2i_baseline.png"
WITH_TEXT = EVIDENCE / "batch" / "batch_02_wuhan_rf_microneedle.jpg"


def test_readable_image_passes_basic_rules() -> None:
    result = rule_findings(CLEAN, expected_aspect="3:4")
    assert result["details"]["width"] > 0
    assert "图片无法解码" not in " ".join(result["reasons"])


def test_corrupt_file_is_rejected(tmp_path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image at all")
    result = rule_findings(broken)
    assert not result["ok"]
    assert any("无法解码" in r for r in result["reasons"])


def test_missing_file_is_rejected(tmp_path) -> None:
    result = rule_findings(tmp_path / "nope.png")
    assert not result["ok"]


def test_ocr_detects_text_when_ocr_available() -> None:
    """带黑板字的真实产出必须被 OCR 检出——这是硬闸门的核心价值。"""
    pytest.importorskip("rapidocr_onnxruntime")
    texts = image_has_text(WITH_TEXT)
    assert texts, "OCR 应该在带黑板字的样本上检出文字"


def test_missing_ocr_engine_is_flagged_not_silent(tmp_path, monkeypatch) -> None:
    """没装 [qc] extra 时文字闸"没跑"必须留痕并拦下这条行——
    静默绿与人脸闸"检不到≠合格"的教义双标。"""
    from PIL import Image

    from aesthetics_pipeline.qc import rules

    img = tmp_path / "clean.png"
    Image.new("RGB", (900, 1200), "white").save(img)
    # 模拟引擎加载失败后的稳态（懒加载已尝试过）
    monkeypatch.setattr(rules, "_ocr_engine", None)
    monkeypatch.setattr(rules, "_ocr_failed", True)
    result = rules.rule_findings(img)
    assert result["details"]["ocr_available"] is False
    assert not result["ok"]
    assert any("OCR" in r for r in result["reasons"])


def test_engine_present_marks_ocr_available() -> None:
    pytest.importorskip("rapidocr_onnxruntime")
    result = rule_findings(CLEAN, expected_aspect="3:4")
    assert result["details"]["ocr_available"] is True
