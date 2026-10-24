"""generate 与成本闸的接线回归（Trae 评审 M1/M2 裁决落点）。

M1：http_api 后端的 BudgetGuard 必须由 cli 用**台账连接**构造后注入。
provider 内部兜底的 `BudgetGuard.from_env()`（无 conn）是零基数守卫——
能力测试（test_cost_guard）测的是"守卫会算"，测不出"跑批传了 conn"，
跨批熔断正好从这条缝里漏掉。
M2：重试耗尽的失败行 est_cost 必须为 None，不是正数——上游没出图就不该
计费，且失败行的正 est_cost 会污染 M1 播种的熔断基数（_spent_from_ledger
只跳 None）。None=没花钱 / 0.0=复用免费，两个语义不许混。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from typer.testing import CliRunner

from aesthetics_pipeline import config, ledger, taskspec
from aesthetics_pipeline.cli import app
from aesthetics_pipeline.cost_guard import BudgetGuard
from aesthetics_pipeline.model_choice import choose_model
from aesthetics_pipeline.models import GenItem, GenOutcome

runner = CliRunner()

MATRIX = {
    "version": "test-cost-wiring",
    "cities": ["武汉"],
    "projects": {
        "光子嫩肤": {"visual": "a female client seated in front of a light device",
                    "use_character": True},
    },
    "themes": {
        "术后修复": {"visual": "calm satisfied expression, healthy even skin",
                     "copy_angle": "术后修复记录"},
    },
    "channels": {"xiaohongshu": {"aspect_ratio": "3:4", "resolution": "2K"}},
    "characters": {
        "lin": {
            "identity_block": "a 28 year old Chinese woman with straight black hair",
            "ref": "evidence/01_t2i_baseline.png",
        }
    },
}

SCENE = {
    "version": "test-cost-wiring",
    "city_english": {"武汉": "Wuhan"},
    "photography": "documentary editorial photography, natural window daylight",
    "negative": "text, watermark",
}


def _env(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "tasks.yaml").write_text(
        yaml.safe_dump(MATRIX, allow_unicode=True), encoding="utf-8"
    )
    (data_dir / "scene.yaml").write_text(
        yaml.safe_dump(SCENE, allow_unicode=True), encoding="utf-8"
    )
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "PROMPT_DIR", data_dir)
    monkeypatch.setattr(taskspec, "DEFAULT_MATRIX", data_dir / "tasks.yaml")
    monkeypatch.setattr(taskspec, "DEFAULT_SCENE", data_dir / "scene.yaml")
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")


class FakeApiBackend:
    """http_api 假后端：永远失败（带 model_id），钉死 M2 的失败行口径。"""

    name = "http_api"

    def generate(self, items: list[GenItem], model_ids: list[str | None] | None = None):
        return [
            GenOutcome(
                task=i.task, ok=False, source_provider="http_api",
                model_id=(model_ids[j] if model_ids else None),
                seconds=0.1, error="模拟上游失败（未出图=没花钱）",
            )
            for j, i in enumerate(items)
        ]


class FakeLocalBackend:
    name = "local"

    def generate(self, items: list[GenItem], model_ids: list[str | None] | None = None):
        return [
            GenOutcome(task=i.task, ok=True, source_provider="local",
                       file_name=f"{i.task.idempotency_key}.png", seconds=1.0)
            for i in items
        ]


def _patched_build(monkeypatch, captured: dict[str, Any], backend: Any):
    def fake_build(name: str, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return backend
    import aesthetics_pipeline.cli as cli_mod
    monkeypatch.setattr(cli_mod, "build_provider", fake_build)


def test_api_backend_gets_ledger_seeded_guard(tmp_path, monkeypatch) -> None:
    """M1：跑批构造 http_api 时必须注入 from_env(conn) 的守卫——
    历史台账花费（35 元）对熔断基数可见。"""
    db = tmp_path / "t.db"
    _env(tmp_path, monkeypatch)
    monkeypatch.setenv("AP_BUDGET_CNY", "50")
    with ledger.connect(db) as conn:
        ledger.record_asset(conn, {"idempotency_key": "seed-a", "provider": "http_api", "est_cost": 10.0})
        ledger.record_asset(conn, {"idempotency_key": "seed-b", "provider": "http_api", "est_cost": 25.0})

    captured: dict[str, Any] = {}
    _patched_build(monkeypatch, captured, FakeApiBackend())

    res = runner.invoke(app, ["generate", "--provider", "http_api", "--retry", "0",
                              "--db", str(db)])
    assert res.exit_code == 0, res.output
    guard = captured.get("budget_guard")
    assert isinstance(guard, BudgetGuard), "http_api 必须收到 cli 注入的 budget_guard"
    assert guard.budget == 50.0
    assert guard.spent == 35.0  # 播种自台账，而不是零基数兜底


def test_non_api_backend_gets_no_guard_kwarg(tmp_path, monkeypatch) -> None:
    """改动波及面钉桩：matrix/local/replay 的构造路径原样（不带 budget_guard kwarg）。"""
    db = tmp_path / "t.db"
    _env(tmp_path, monkeypatch)
    captured: dict[str, Any] = {}
    _patched_build(monkeypatch, captured, FakeLocalBackend())

    res = runner.invoke(app, ["generate", "--provider", "local", "--retry", "0",
                              "--db", str(db)])
    assert res.exit_code == 0, res.output
    assert "budget_guard" not in captured


def test_failed_row_est_cost_is_none_not_positive(tmp_path, monkeypatch) -> None:
    """M2：重试耗尽的失败行 est_cost=None（不是档位正价），
    且不进熔断基数——否则坏行会提前引爆下一次跑批。"""
    db = tmp_path / "t.db"
    _env(tmp_path, monkeypatch)

    captured: dict[str, Any] = {}
    _patched_build(monkeypatch, captured, FakeApiBackend())

    res = runner.invoke(app, ["generate", "--provider", "http_api", "--retry", "2",
                              "--db", str(db)])
    assert res.exit_code == 0, res.output
    with ledger.connect(db) as conn:
        rows = ledger.assets(conn, "provider = 'http_api'")
    assert rows, "失败也要留台账行（可追溯）"
    for row in rows:
        assert row["est_cost"] is None
        assert row["model_id"], "失败行仍带实际档位，供追溯"
        # 熔断基数口径：None 不计钱
    with ledger.connect(db) as conn:
        assert BudgetGuard._spent_from_ledger(conn) == 0.0


class RecordingOkBackend:
    """记录本批实际送进后端的任务与档位，并全部成功（用于测幂等跳过口径）。"""

    name = "http_api"

    def __init__(self) -> None:
        self.sent: list[tuple[str, str | None]] = []

    def generate(self, items: list[GenItem], model_ids: list[str | None] | None = None):
        self.sent.extend(
            (i.task.idempotency_key, model_ids[j] if model_ids else None)
            for j, i in enumerate(items)
        )
        return [
            GenOutcome(
                task=i.task, ok=True, source_provider="http_api",
                model_id=(model_ids[j] if model_ids else None),
                file_name=f"{i.task.idempotency_key}.jpg",
                local_path=f"out/{i.task.idempotency_key}.jpg",
                seconds=0.1,
            )
            for j, i in enumerate(items)
        ]


def _seed(db: Path, key: str, model_id: str | None) -> None:
    with ledger.connect(db) as conn:
        ledger.record_asset(
            conn,
            {"idempotency_key": key, "provider": "http_api",
             "model_id": model_id, "local_path": "out/seed.jpg"},
        )


def test_skip_existing_is_per_model(tmp_path, monkeypatch) -> None:
    """同档位已产出 → 跳过；换档位 → 必须重新生成，并打印提示。"""
    db = tmp_path / "t.db"
    _env(tmp_path, monkeypatch)
    backend = RecordingOkBackend()
    _patched_build(monkeypatch, {}, backend)

    spec = taskspec.expand()[0]
    key = spec.idempotency_key
    chosen = choose_model(spec).model_id

    _seed(db, key, chosen)  # 当前档已有产出
    res = runner.invoke(app, ["generate", "--provider", "http_api", "--retry", "0",
                              "--db", str(db)])
    assert res.exit_code == 0, res.output
    assert backend.sent == [], "同档位已产出必须跳过（不能重复花钱）"

    # 换档位：同一幂等键、另一个模型 → 旧产出不能算已完成
    with ledger.connect(db) as conn:
        conn.execute(
            "UPDATE assets SET model_id = 'qwen-image-3.0-pro' WHERE idempotency_key = ?",
            (key,),
        )
    res = runner.invoke(app, ["generate", "--provider", "http_api", "--retry", "0",
                              "--db", str(db)])
    assert res.exit_code == 0, res.output
    assert [k for k, _ in backend.sent] == [key], "换模型后必须重新生成"
    assert backend.sent[0][1] == chosen
    # rich 会按终端宽度折行，断言只取不会被折断的短片段
    assert "没有产出" in res.output


def test_legacy_row_without_model_is_regenerated_with_notice(tmp_path, monkeypatch) -> None:
    """升级前的行没记 model_id：精确匹配不命中 → 重出一次并明确提示。"""
    db = tmp_path / "t.db"
    _env(tmp_path, monkeypatch)
    backend = RecordingOkBackend()
    _patched_build(monkeypatch, {}, backend)

    key = taskspec.expand()[0].idempotency_key
    _seed(db, key, None)

    res = runner.invoke(app, ["generate", "--provider", "http_api", "--retry", "0",
                              "--db", str(db)])
    assert res.exit_code == 0, res.output
    assert [k for k, _ in backend.sent] == [key]
    assert "model_id" in res.output
    with ledger.connect(db) as conn:
        rows = ledger.assets(conn, "provider = 'http_api'")
    # 旧行（无档位）仍在，重出行是**另一条**（主键含 variant）——这正是"多档并存"。
    assert len(rows) == 2
    assert any(row["model_id"] for row in rows), "重出后必须记下实际档位"
