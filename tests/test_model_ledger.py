"""模型档位 → GenOutcome → 台账的管线断言（集成收口）。

负向重点：http_api 行的 est_cost 必须来自 ModelSpec 实测单价，
**不是** AP_COST_API 的一刀切旧口径——两档差 5 倍时混算就是假账。
"""

from __future__ import annotations

import sqlite3

from aesthetics_pipeline import ledger
from aesthetics_pipeline.cli import _est_cost_for
from aesthetics_pipeline.models import GenOutcome, TaskSpec
from aesthetics_pipeline.provider.models_spec import get as get_spec


def _task(key: str = "k1") -> TaskSpec:
    return TaskSpec(
        city="武汉", project="光子嫩肤", theme="术后修复", channel="小红书",
        character=None, aspect_ratio="3x4", resolution="1024x1365",
        idempotency_key=key,
    )


def test_migration_adds_model_columns(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE assets (idempotency_key TEXT NOT NULL, provider TEXT NOT NULL,"
        " route_reason TEXT, qc_verdict TEXT, adopted TEXT,"
        " PRIMARY KEY (idempotency_key, provider))"
    )
    conn.commit()
    conn.close()
    with ledger.connect(db) as c:
        cols = {row[1] for row in c.execute("PRAGMA table_info(assets)")}
    for col in ("model_id", "model_rule_id", "model_reason"):
        assert col in cols, f"旧库迁移后应补上 {col}"


def test_record_asset_roundtrip_model_fields(tmp_path):
    db = tmp_path / "t.db"
    with ledger.connect(db) as conn:
        ledger.record_asset(conn, {
            "idempotency_key": "k1", "provider": "http_api",
            "model_id": "doubao-seedream-4-0",
            "model_rule_id": "M1_SCENE_CHEAP",
            "model_reason": "无人物纯场景",
        })
        ledger.record_asset(conn, {
            "idempotency_key": "k1", "provider": "http_api", "qc_verdict": "accepted",
        })
        rows = ledger.assets(conn, "idempotency_key = 'k1'")
    assert len(rows) == 1
    # 局部更新（质检）不得把 model 三列覆盖成 NULL——与上游 record_asset 同一不变量
    assert rows[0]["model_id"] == "doubao-seedream-4-0"
    assert rows[0]["model_rule_id"] == "M1_SCENE_CHEAP"
    assert rows[0]["qc_verdict"] == "accepted"


def test_est_cost_uses_model_pricelist_not_env_fallback(monkeypatch):
    # 即便给 matrix 配了一刀切单价，带 model_id 的行也必须按档位价目表算
    monkeypatch.setenv("AP_COST_API", "0.5")
    outcome = GenOutcome(
        task=_task(), ok=True, source_provider="http_api",
        model_id="doubao-seedream-4-0", seconds=3.0,
    )
    assert _est_cost_for(outcome, "http_api") == get_spec("doubao-seedream-4-0").unit_cost_cny
    assert _est_cost_for(outcome, "http_api") != 0.5

    # 无 model_id（matrix/local）沿用旧口径
    legacy = GenOutcome(task=_task("k2"), ok=True, source_provider="matrix", seconds=3.0)
    assert _est_cost_for(legacy, "matrix") == 0.5

    # 未配置单价 → None，不编造
    monkeypatch.delenv("AP_COST_API")
    monkeypatch.delenv("AP_UNIT_COST", raising=False)
    assert _est_cost_for(legacy, "matrix") is None
