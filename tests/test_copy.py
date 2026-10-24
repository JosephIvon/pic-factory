"""文案模板与禁用词闸。"""

from __future__ import annotations

from aesthetics_pipeline.copy import make_copy, scan


def test_scan_finds_absolute_and_guarantee_terms() -> None:
    hits = scan("本院拥有全网最好的技术，保证一次见效")
    assert any("最好" in h for h in hits)
    assert any("保证" in h for h in hits)
    assert any("一次见效" in h for h in hits)


def test_scan_is_clean_for_normal_copy() -> None:
    assert scan("武汉水光针补水术后第 7 天的真实状态记录") == []


def test_generated_copy_passes_its_own_banned_word_gate() -> None:
    """模板自己产出的文案必须能过自己的禁用词闸（否则闸门形同虚设）。"""
    for channel in ("xiaohongshu", "wechat"):
        for theme in ("术后第7天真实反馈", "价格透明拆解", "真实素颜无滤镜"):
            draft = make_copy(
                city="武汉",
                project="水光针补水",
                theme=theme,
                copy_angle="术后第 7 天的真实状态记录",
                channel=channel,
                character="lin",
            )
            assert draft.banned_hits == [], f"{channel}/{theme} 命中禁用词 {draft.banned_hits}"
            assert draft.ok
            assert "武汉" in draft.title


def test_negated_banned_word_is_not_a_hit() -> None:
    """'不保证疗效' 是合规写法，不能被同一个词表拦下来。"""
    assert scan("本院不保证疗效") == []
    assert scan("无任何承诺") == []
    assert scan("保证一次见效") != []
