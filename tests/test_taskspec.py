"""任务矩阵展开与幂等键。"""

from __future__ import annotations

from aesthetics_pipeline.taskspec import expand


def test_expansion_is_deterministic() -> None:
    first = [spec.idempotency_key for spec in expand()]
    second = [spec.idempotency_key for spec in expand()]
    assert first == second


def test_idempotency_keys_are_unique() -> None:
    keys = [spec.idempotency_key for spec in expand()]
    assert len(set(keys)) == len(keys)


def test_matrix_covers_all_dimensions() -> None:
    specs = expand()
    assert {s.city for s in specs} == {"武汉", "成都", "深圳"}
    assert {s.channel for s in specs} == {"xiaohongshu", "wechat"}
    assert len({s.aspect_ratio for s in specs}) == 2


def test_fixed_character_tasks_carry_reference() -> None:
    """绑了角色的任务必须带得上参考图，否则会静默降级成文生图。"""
    for spec in expand():
        if spec.character:
            assert spec.ref_asset, "绑角色的任务缺少参考图配置"
