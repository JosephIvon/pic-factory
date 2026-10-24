"""路由层的回归测试。

重点是两条**用实测换来的不变量**：
  1. 原始中文标签不得进入画面 prompt（会被渲染成乱码招牌）
  2. 需要同一个人 -> 图生图；不需要 -> 文生图。两条路都要真的被走到。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aesthetics_pipeline import config
from aesthetics_pipeline.router import Router
from aesthetics_pipeline.taskspec import expand, load_yaml, scene_config


@pytest.fixture(scope="module")
def router() -> Router:
    return Router(load_yaml(config.DATA_DIR / "tasks.yaml"), scene_config())


def test_raw_chinese_labels_never_leak_into_visual_prompt(router: Router) -> None:
    for spec in expand():
        prompt, _negative, _warnings = router.compile(spec)
        assert spec.theme not in prompt, f"主题标签泄漏进 prompt: {spec.theme}"
        assert spec.project not in prompt, f"项目标签泄漏进 prompt: {spec.project}"


def test_unknown_theme_falls_back_without_leaking(router: Router) -> None:
    spec = expand()[0]
    spec.theme = "一个不存在的主题"
    prompt, _negative, warnings = router.compile(spec)
    assert "一个不存在的主题" not in prompt
    assert warnings, "缺视觉映射时必须给出告警"


def test_route_covers_both_paths(router: Router) -> None:
    paths = {router.decide_path(spec)[0] for spec in expand()}
    assert paths == {"t2i", "i2i"}


def test_i2i_route_always_has_a_real_reference_file(router: Router) -> None:
    for spec in expand():
        path, ref = router.decide_path(spec)
        if path == "i2i":
            assert ref, "选了图生图却没有参考图"
            assert Path(ref).exists(), f"参考图不存在: {ref}"


def test_negative_prompt_suppresses_text(router: Router) -> None:
    """画面内文字是医美内容的翻车点，negative 必须压住。"""
    for spec in expand()[:5]:
        _prompt, negative, _warnings = router.compile(spec)
        for token in ("text", "lettering", "signage", "watermark"):
            assert token in negative
