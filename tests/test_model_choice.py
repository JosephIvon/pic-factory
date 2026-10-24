"""模型档位路由 + 声明式注册表的契约测试（整合方自有层）。"""

from __future__ import annotations

import pytest

from aesthetics_pipeline.model_choice import IDENTITY_FLOOR, _PERSONA_BANNED, choose_model
from aesthetics_pipeline.models import TaskSpec
from aesthetics_pipeline.provider.models_spec import REGISTRY, get


def spec(**kw) -> TaskSpec:
    base = dict(
        city="武汉", project="水光针", theme="记录", channel="xhs",
        character=None, aspect_ratio="3:4", resolution="2K", idempotency_key="k",
        path="t2i", ref_asset=None,
    )
    base.update(kw)
    return TaskSpec(**base)


def test_registry_keys_are_consistent_with_model_choice() -> None:
    """model_choice 里出现的 model_id 必须都在注册表里（否则路由会指向不存在的模型）。"""
    from aesthetics_pipeline import model_choice as mc

    for m in mc._MEAN_SIM:
        assert m in REGISTRY, f"{m} 在档位表但不在注册表"


def test_persona_task_never_selects_banned_swap_models() -> None:
    """人物任务无论如何在候选里都不能出现换脸黑名单模型。"""
    d = choose_model(spec(character="lin", path="i2i", ref_asset="refs/lin.png"))
    assert d.model_id not in _PERSONA_BANNED
    assert get(d.model_id)  # 选出来的必须是已注册模型


def test_persona_picks_cheapest_safe_tier() -> None:
    """合格档里取最低价：5.0-Flash(¥0.10, 订正后均值 0.815) 先于 Qwen-3.0(¥0.18)。

    取脸口径订正（model-sim-recompute.json）后 Flash 平反回可选池；
    一致性由"两层身份 + 换脸兜底"保证，价格成为区分项。
    """
    d = choose_model(spec(character="lin", path="i2i", ref_asset="refs/lin.png"))
    assert d.model_id == "doubao-seedream-5-0-flash"
    assert d.rule_id == "M2_PERSONA_SAFE"
    # 负向锚点：均值同样过线的 5.0-Pro 因非遮挡低分/方差被独立排除
    assert "doubao-seedream-5-0-pro" in _PERSONA_BANNED


def test_pro_fallback_when_only_expensive_passes_floor(monkeypatch) -> None:
    """把下限抬到只有 Pro 能过：应升档 Pro，而不是选不达标的便宜档。"""
    from aesthetics_pipeline import model_choice as mc

    monkeypatch.setattr(mc, "IDENTITY_FLOOR", 0.87)  # qwen-3.0=0.863 被卡掉
    d = choose_model(spec(character="lin", path="i2i", ref_asset="refs/lin.png"))
    assert d.model_id == "qwen-image-3.0-pro"
    assert d.rule_id == "M2_PERSONA_PRO_FALLBACK" or mc._MEAN_SIM[d.model_id] >= mc.IDENTITY_FLOOR


def test_pure_scene_goes_cheapest_seedream() -> None:
    """无人物纯场景：成本优先，Seedream 4.0。"""
    d = choose_model(spec(character=None, path="t2i"))
    assert d.model_id == "doubao-seedream-4-0"
    assert d.rule_id in {"M1_SCENE_CHEAP", "M3_VOLUME_CHEAP"}


def test_floor_value_matches_measured_anchor() -> None:
    """下限 0.75 是从实测最低分反推的红线，钉住防手滑改动。"""
    assert IDENTITY_FLOOR == 0.75


def test_get_unknown_model_raises_with_hint() -> None:
    with pytest.raises(KeyError, match="未注册"):
        get("gpt-image-9")


def test_async_wan_models_flag_watermark_note() -> None:
    """阿里默认裸图：注册表应把'需强制开水印'标进 notes，防批量里漏。"""
    s = get("qwen-image-3.0-pro")
    assert s.mode == "async" and not s.watermark_default_on
    assert "强制显式开水印" in s.notes
