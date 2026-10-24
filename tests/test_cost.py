"""成本账：单张合格图综合成本 = 总花费 ÷ 合格张数。"""

from __future__ import annotations

from aesthetics_pipeline import config
from aesthetics_pipeline.report import cost_summary


def test_api_unit_cost_unset_is_none(monkeypatch) -> None:
    """没配单价就不给数字——不编造金额。"""
    monkeypatch.delenv("AP_COST_API", raising=False)
    monkeypatch.delenv("AP_UNIT_COST", raising=False)
    assert config.api_unit_cost() is None


def test_api_unit_cost_from_env(monkeypatch) -> None:
    monkeypatch.setenv("AP_COST_API", "0.2")
    assert config.api_unit_cost() == 0.2


def test_local_unit_cost_needs_all_three_params(monkeypatch) -> None:
    """缺一个参数就返回 None，不做部分估算。"""
    monkeypatch.setenv("AP_LOCAL_WATT", "150")
    monkeypatch.delenv("AP_KWH_PRICE", raising=False)
    monkeypatch.delenv("AP_LOCAL_DEPR_PER_HOUR", raising=False)
    assert config.local_unit_cost(15) is None


def test_local_unit_cost_formula(monkeypatch) -> None:
    """电费 + 折旧，按秒折算：15/3600 × (150/1000×0.6 + 2) = 0.008708"""
    monkeypatch.setenv("AP_LOCAL_WATT", "150")
    monkeypatch.setenv("AP_KWH_PRICE", "0.6")
    monkeypatch.setenv("AP_LOCAL_DEPR_PER_HOUR", "2")
    assert config.local_unit_cost(15) == 0.008708


def test_unit_cost_for_dispatches_by_provider(monkeypatch) -> None:
    monkeypatch.setenv("AP_COST_API", "0.2")
    monkeypatch.setenv("AP_LOCAL_WATT", "150")
    monkeypatch.setenv("AP_KWH_PRICE", "0.6")
    monkeypatch.setenv("AP_LOCAL_DEPR_PER_HOUR", "2")
    assert config.unit_cost_for("matrix", 15) == 0.2
    assert config.unit_cost_for("local", 15) == 0.008708
    assert config.unit_cost_for("replay", 15) is None


def test_cost_per_accepted_counts_rework() -> None:
    """被拒收的图也花了钱，必须算进总花费。"""
    rows = [
        {"provider": "matrix", "est_cost": 0.2, "qc_verdict": "accepted", "gen_seconds": 10},
        {"provider": "matrix", "est_cost": 0.2, "qc_verdict": "rejected", "gen_seconds": 10},
        {"provider": "local", "est_cost": 0.01, "qc_verdict": "accepted", "gen_seconds": 10},
        {"provider": "local", "est_cost": 0.01, "qc_verdict": "rejected", "gen_seconds": 10},
    ]
    summary = {c["provider"]: c for c in cost_summary(rows)}
    matrix = summary["matrix"]
    assert matrix["calls"] == 2 and matrix["accepted"] == 1 and matrix["rejected"] == 1
    assert matrix["total_cost"] == 0.4
    # 2 张花了 0.4 元，只合格 1 张 → 单张合格图综合成本 0.4（不是 0.2）
    assert matrix["cost_per_accepted"] == 0.4
    assert summary["local"]["cost_per_accepted"] == 0.02


def test_cost_summary_without_prices_is_none() -> None:
    """没配单价 → 金额列为 None，报表显示"未配置单价"。"""
    rows = [
        {"provider": "matrix", "est_cost": None, "qc_verdict": "accepted", "gen_seconds": 10},
    ]
    summary = {c["provider"]: c for c in cost_summary(rows)}
    assert summary["matrix"]["priced"] is False
    assert summary["matrix"]["total_cost"] is None
    assert summary["matrix"]["cost_per_accepted"] is None


def test_cost_summary_recomputes_from_seconds_when_price_set_later(monkeypatch) -> None:
    """生成时没配单价 → 报表按实测秒数 × 当前假设现算（改假设不用重新生成）。"""
    monkeypatch.setenv("AP_COST_API", "0.2")
    rows = [
        {"provider": "matrix", "est_cost": None, "qc_verdict": "accepted", "gen_seconds": 10},
    ]
    summary = {c["provider"]: c for c in cost_summary(rows)}
    assert summary["matrix"]["priced"] is True
    assert summary["matrix"]["total_cost"] == 0.2
    assert summary["matrix"]["cost_per_accepted"] == 0.2
