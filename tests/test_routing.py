"""自动路由规则表（provider 进规则表，决策理由落台账）。"""

from __future__ import annotations

import pytest

from aesthetics_pipeline.models import TaskSpec
from aesthetics_pipeline.routing import decide_provider, summarize


def spec(**kw) -> TaskSpec:
    base = dict(
        city="武汉", project="水光针补水", theme="术后第7天真实反馈",
        channel="xiaohongshu", character=None, aspect_ratio="3:4",
        resolution="2K", idempotency_key="k1",
    )
    base.update(kw)
    return TaskSpec(**base)  # type: ignore[arg-type]


ALWAYS_OK = lambda name: (True, "ok")  # noqa: E731


def test_cloud_prefers_deliverable_api_when_key_set(monkeypatch) -> None:
    """配了标准 key：命中"走云"的任务应改走可交付的 api，且理由写明交付优先。"""
    monkeypatch.setenv("AP_ARK_KEY", "ark-test")
    d = decide_provider(spec(character="lin"), available=ALWAYS_OK)
    assert d.provider == "http_api"
    assert d.rule_id == "R3_STRONG_IDENTITY"  # 命中的规则不变
    assert "交付优先" in d.reason


def test_cloud_falls_back_to_matrix_reference_without_key(monkeypatch) -> None:
    """没配 key：云端仍是 matrix（演示参照），保持既有行为。"""
    monkeypatch.delenv("AP_ARK_KEY", raising=False)
    monkeypatch.delenv("AP_DASHSCOPE_KEY", raising=False)
    d = decide_provider(spec(character="lin"), available=ALWAYS_OK)
    assert d.provider == "matrix"


def test_matrix_explicitly_forced_by_flag(monkeypatch) -> None:
    """AP_PREFER_MATRIX=1：即便有 key 也留在 matrix（对比基线场景）。"""
    monkeypatch.setenv("AP_ARK_KEY", "ark-test")
    monkeypatch.setenv("AP_PREFER_MATRIX", "1")
    d = decide_provider(spec(character="lin"), available=ALWAYS_OK)
    assert d.provider == "matrix"


def test_no_cloud_beats_deliverable_api(monkeypatch) -> None:
    """不出网压过一切：no_cloud 即便有 key 也走本地。"""
    monkeypatch.setenv("AP_ARK_KEY", "ark-test")
    d = decide_provider(spec(character="lin", no_cloud=True), available=ALWAYS_OK)
    assert d.provider == "local"
    assert d.rule_id == "R1_NO_CLOUD"


def test_bound_character_goes_to_api_for_identity() -> None:
    """强人物一致 → API：实测 API 身份保持 0.32~0.83 优于本地栈 0.24~0.47。"""
    d = decide_provider(spec(character="lin"), available=ALWAYS_OK)
    assert d.provider == "matrix"
    assert d.rule_id == "R3_STRONG_IDENTITY"


def test_generic_task_goes_to_local_for_cost() -> None:
    """泛化人物 / 铺量 → 本地栈：边际成本≈电费。"""
    d = decide_provider(spec(), available=ALWAYS_OK)
    assert d.provider == "local"
    assert d.rule_id == "R4_VOLUME_LOCAL"


def test_no_cloud_beats_identity_priority() -> None:
    """不出网是合规要求，优先级高于"强一致→API"。"""
    d = decide_provider(spec(character="lin", no_cloud=True), available=ALWAYS_OK)
    assert d.provider == "local"
    assert d.rule_id == "R1_NO_CLOUD"


def test_no_cloud_beats_text_overlay() -> None:
    """规则表顺序即优先级：R1 > R2 > R3 > R4。"""
    d = decide_provider(spec(character="lin", no_cloud=True, needs_text_overlay=True), available=ALWAYS_OK)
    assert d.rule_id == "R1_NO_CLOUD"


def test_text_overlay_goes_to_api() -> None:
    d = decide_provider(spec(needs_text_overlay=True), available=ALWAYS_OK)
    assert d.provider == "matrix"
    assert d.rule_id == "R2_TEXT_OVERLAY"


def test_global_no_cloud_overrides_everything() -> None:
    d = decide_provider(spec(character="lin"), no_cloud=True, available=ALWAYS_OK)
    assert d.provider == "local"
    assert d.rule_id == "R0_NO_CLOUD_GLOBAL"


def test_unavailable_backend_falls_back_and_says_why() -> None:
    """后端不可用要回退并写明原因，不能静默换掉。"""
    def available(name: str) -> tuple[bool, str]:
        if name == "local":
            return False, "找不到本地模型目录"
        return True, "ok"

    d = decide_provider(spec(), available=available)
    assert d.provider == "matrix"
    assert d.rule_id == "R_FALLBACK_UNAVAILABLE"
    assert "找不到本地模型目录" in d.reason


def test_both_backends_unavailable_raises() -> None:
    def available(name: str) -> tuple[bool, str]:
        return False, "都没有"

    with pytest.raises(RuntimeError, match="两个后端都不可用"):
        decide_provider(spec(), available=available)


def test_summarize_counts_by_provider_and_rule() -> None:
    decisions = [
        decide_provider(spec(character="lin", idempotency_key="a"), available=ALWAYS_OK),
        decide_provider(spec(idempotency_key="b"), available=ALWAYS_OK),
        decide_provider(spec(idempotency_key="c"), available=ALWAYS_OK),
    ]
    summary = summarize(decisions)
    assert summary["matrix"]["count"] == 1
    assert summary["local"]["count"] == 2
    assert summary["local"]["rules"]["R4_VOLUME_LOCAL"] == 2
