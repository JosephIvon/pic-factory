"""抽检台人工采纳（adopted / rejected）。

两条核心回归：
  1. 采纳写入只动 adopted/adopted_at/adopt_note 三列，**不覆盖** qc_verdict /
     local_path —— 这靠 record_asset 的 ON CONFLICT DO UPDATE 局部语义
     （换成 INSERT OR REPLACE 会把其它列清成 NULL，台账自我销毁）。
  2. 面板 POST /api/assets/adopt 的写必须**经 CLI 子进程**落到台账，
     用 spy 记录真实 subprocess 命令行 + 读盘验效果，证明"写走 CLI"。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from aesthetics_pipeline import ledger
from aesthetics_pipeline.cli import app
from aesthetics_pipeline import report as report_mod

runner = CliRunner()

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from aesthetics_pipeline import config, panel  # noqa: E402


def _seed(db: Path, key: str, provider: str, **overrides: Any) -> None:
    row: dict[str, Any] = {
        "idempotency_key": key,
        "provider": provider,
        "city": "武汉",
        "project": "水光针补水",
        "theme": "术后第7天真实反馈",
        "channel": "xiaohongshu",
        "route": "i2i",
        "local_path": f"out/{provider}/{key}.png",
        "gen_seconds": 12.0,
        "est_cost": 0.2,
        "qc_verdict": "accepted",
        "qc_reasons": [],
        "route_rule_id": "R3_STRONG_IDENTITY",
        "route_reason": "强人物一致：API 身份保持更好",
    }
    row.update(overrides)
    with ledger.connect(db) as conn:
        ledger.record_asset(conn, row)


# ---------------------------------------------------------------- 台账局部更新


def test_record_adoption_only_touches_adoption_columns(tmp_path) -> None:
    """关键回归：采纳后 adopted/adopted_at 写入，qc_verdict/local_path 原样不动。"""
    db = tmp_path / "t.db"
    _seed(db, "k1", "matrix")
    with ledger.connect(db) as conn:
        assert ledger.record_adoption(conn, "k1", decision="adopted") == 1
        row = ledger.assets(conn)[0]
    assert row["adopted"] == "adopted"
    assert row["adopted_at"]  # 时间戳由 ledger 补
    assert row["adopt_note"] is None  # 没传 note 不写出鬼值
    # 局部更新不许碰别的列（ON CONFLICT DO UPDATE 只动 touched）
    assert row["qc_verdict"] == "accepted"
    assert row["local_path"] == "out/matrix/k1.png"
    assert row["city"] == "武汉"
    assert row["est_cost"] == 0.2
    assert row["route_rule_id"] == "R3_STRONG_IDENTITY"


def test_rejected_with_note_is_stored(tmp_path) -> None:
    """质检放行 ≠ 人工采纳：rejected + 原因单独存三列，判定列保持 accepted。"""
    db = tmp_path / "t.db"
    _seed(db, "k2", "matrix")
    with ledger.connect(db) as conn:
        ledger.record_adoption(conn, "k2", decision="rejected", note="脸部塑料感太强")
        row = ledger.assets(conn)[0]
    assert row["adopted"] == "rejected"
    assert row["adopt_note"] == "脸部塑料感太强"
    assert row["qc_verdict"] == "accepted"


def test_record_adoption_validates_and_reports_missing(tmp_path) -> None:
    db = tmp_path / "t.db"
    _seed(db, "k3", "matrix")
    with ledger.connect(db) as conn:
        with pytest.raises(ValueError):
            ledger.record_adoption(conn, "k3", decision="maybe")
        # 采纳只能点在真实存在的产出上：查无此键返回 0，不插孤儿行
        assert ledger.record_adoption(conn, "ghost", decision="adopted") == 0
        assert len(ledger.assets(conn)) == 1


# ---------------------------------------------------------------- CLI 写入口


def test_cli_adopt_writes_via_command(tmp_path) -> None:
    db = tmp_path / "t.db"
    _seed(db, "k1", "matrix")
    _seed(db, "k2", "local", qc_verdict="pending")

    res = runner.invoke(app, ["adopt", "k1", "--decision", "rejected", "--note", "手指畸形", "--db", str(db)])
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["adopt", "k2", "--db", str(db)])  # 默认 adopted
    assert res.exit_code == 0, res.output

    with ledger.connect(db) as conn:
        rows = {r["idempotency_key"]: r for r in ledger.assets(conn)}
    assert rows["k1"]["adopted"] == "rejected"
    assert rows["k1"]["adopt_note"] == "手指畸形"
    assert rows["k1"]["qc_verdict"] == "accepted"  # 未被覆盖
    assert rows["k2"]["adopted"] == "adopted"
    assert rows["k2"]["local_path"] == "out/local/k2.png"


def test_cli_adopt_error_codes(tmp_path) -> None:
    db = tmp_path / "t.db"
    _seed(db, "k1", "matrix")
    # 非法决定 → 2；键不存在 → 1（面板据此回 422，绝不静默成功）
    assert runner.invoke(app, ["adopt", "k1", "--decision", "maybe", "--db", str(db)]).exit_code == 2
    assert runner.invoke(app, ["adopt", "ghost", "--db", str(db)]).exit_code == 1


def test_cli_adoption_prints_rate(tmp_path) -> None:
    db = tmp_path / "t.db"
    _seed(db, "k1", "matrix")
    _seed(db, "k2", "local")
    with ledger.connect(db) as conn:
        ledger.record_adoption(conn, "k1", decision="adopted")
    res = runner.invoke(app, ["adoption", "--db", str(db)])
    assert res.exit_code == 0
    assert "1/2 = 50.0%" in res.output


# ---------------------------------------------------------------- 采纳率口径


def test_adoption_rate_by_provider(tmp_path) -> None:
    db = tmp_path / "t.db"
    _seed(db, "k1", "matrix")
    _seed(db, "k2", "matrix")
    _seed(db, "k3", "local")
    _seed(db, "k5", "replay")
    # 失败没落盘的行（local_path NULL）不进分母：没人能点它
    _seed(db, "k4", "local", local_path=None)
    with ledger.connect(db) as conn:
        ledger.record_adoption(conn, "k1", decision="adopted")
        ledger.record_adoption(conn, "k2", decision="rejected")
        ledger.record_adoption(conn, "k3", decision="adopted")
        ledger.record_adoption(conn, "k4", decision="adopted")
        stats = ledger.adoption_summary(conn)
        summary = ledger.summary(conn)
        rows = ledger.assets(conn)

    assert stats["generated"] == 4  # k4 无文件被排除
    assert stats["adopted"] == 2
    assert stats["rejected"] == 1
    assert stats["adoption_rate"] == 0.5
    assert stats["by_provider"]["matrix"] == {
        "generated": 2, "adopted": 1, "rejected": 1, "adoption_rate": 0.5,
    }
    assert stats["by_provider"]["local"] == {
        "generated": 1, "adopted": 1, "rejected": 0, "adoption_rate": 1.0,
    }
    assert stats["by_provider"]["replay"]["adoption_rate"] == 0.0
    # 面板 /api/summary 复用的 ledger.summary 也带上同一口径
    assert summary["adoption_rate"] == 0.5
    assert summary["adoption"] == stats
    # 报表的纯函数版与 SQL 版同口径（report 拿的是已读出的行列表）
    report_stats = report_mod.adoption_stats(rows)
    assert report_stats["generated"] == stats["generated"]
    assert report_stats["adopted"] == stats["adopted"]
    assert report_stats["adoption_rate"] == stats["adoption_rate"]
    assert report_stats["by_provider"] == stats["by_provider"]


# ---------------------------------------------------------------- 面板：写走 CLI


@pytest.fixture()
def panel_env(tmp_path, monkeypatch):
    """把面板进程与 CLI 子进程都指向 tmp 台账，隔离仓库真实数据。

    子进程是全新解释器：它的 DB 位置由环境变量 AP_DB 决定（config 在子进程
    导入时读取），进程内读则 monkeypatch ledger.DB_PATH —— 两者必须同文件。
    """
    db = tmp_path / "pipeline.db"
    src_dir = Path(panel.__file__).resolve().parents[1]
    ppath = str(src_dir)
    if os.environ.get("PYTHONPATH"):
        ppath += os.pathsep + os.environ["PYTHONPATH"]
    monkeypatch.setenv("PYTHONPATH", ppath)
    monkeypatch.setenv("AP_DB", str(db))
    monkeypatch.setattr(ledger, "DB_PATH", db)
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    return db


def test_panel_adopt_post_goes_through_cli_subprocess(panel_env, monkeypatch) -> None:
    db = panel_env
    _seed(db, "abc123", "matrix")
    client = TestClient(panel.app)

    # spy 放行真实 subprocess.run：既记录命令行参数，又让子进程真的写库
    calls: list[list[str]] = []
    real_run = subprocess.run

    def spy_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(panel.subprocess, "run", spy_run)

    resp = client.post(
        "/api/assets/adopt",
        json={"key": "abc123", "provider": "matrix", "decision": "rejected",
              "note": "抽检：手指数不对"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["ok"] is True

    # 证据 1：这笔写是从 CLI 子进程出去的（面板没有旁路直接 import ledger 写库）
    assert calls == [[
        sys.executable, "-m", "aesthetics_pipeline.cli",
        "adopt", "abc123", "--decision", "rejected", "--note", "抽检：手指数不对",
        "--provider", "matrix",
    ]]

    # 证据 2：CLI 子进程真的改了 tmp 台账，且只动采纳三列
    with ledger.connect(db) as conn:
        row = ledger.assets(conn)[0]
    assert row["adopted"] == "rejected"
    assert row["adopt_note"] == "抽检：手指数不对"
    assert row["adopted_at"]
    assert row["qc_verdict"] == "accepted"
    assert row["local_path"] == "out/matrix/abc123.png"

    # 面板读侧同口径：/api/summary 报出采纳率与计数，画廊带上 adopted
    summary = client.get("/api/summary").json()
    assert summary["adoption"]["rejected"] == 1
    assert summary["adoption_rate"] == 0.0
    item = client.get("/api/assets").json()["items"][0]
    assert item["adopted"] == "rejected"


def test_panel_adopt_maps_cli_failure_and_writes_nothing(panel_env, monkeypatch) -> None:
    """CLI 失败（退出码非 0）→ 422；面板不得绕过 CLI 自己补写台账。"""
    db = panel_env
    _seed(db, "abc123", "matrix")

    class _FakeProc:
        returncode = 1
        stdout = "台账里没有这条产出：ghost"
        stderr = ""

    monkeypatch.setattr(panel.subprocess, "run", lambda *a, **k: _FakeProc())
    client = TestClient(panel.app)
    resp = client.post("/api/assets/adopt", json={"key": "ghost", "decision": "adopted"})
    assert resp.status_code == 422
    with ledger.connect(db) as conn:
        rows = ledger.assets(conn)
    assert len(rows) == 1  # 没插孤儿行
    assert rows[0]["adopted"] is None  # 已有行也没被顺手改


def test_panel_adopt_rejects_bad_payload(panel_env) -> None:
    client = TestClient(panel.app)
    assert client.post("/api/assets/adopt", json={"decision": "adopted"}).status_code == 400
    assert client.post(
        "/api/assets/adopt", json={"key": "abc123", "decision": "maybe"}
    ).status_code == 400
