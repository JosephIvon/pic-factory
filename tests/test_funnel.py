"""回传闭环的 CLI 回归：发布登记 → 效果录入 → 漏斗。"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from aesthetics_pipeline import ledger
from aesthetics_pipeline.cli import app

runner = CliRunner()


def test_publish_record_metrics_then_funnel(tmp_path: Path) -> None:
    db = tmp_path / "f.db"
    with ledger.connect(db) as conn:
        ledger.record_asset(
            conn,
            {"idempotency_key": "k1", "provider": "http_api",
             "local_path": "out/k1.jpg", "qc_verdict": "accepted", "adopted": "adopted"},
        )

    res = runner.invoke(app, [
        "publish-record", "k1", "--provider", "http_api",
        "--channel", "xiaohongshu", "--url", "https://example.com/p/1", "--db", str(db),
    ])
    assert res.exit_code == 0, res.output

    res = runner.invoke(app, [
        "metrics", "k1", "--channel", "xiaohongshu",
        "--impressions", "2000", "--clicks", "120", "--saves", "12", "--db", str(db),
    ])
    assert res.exit_code == 0, res.output

    res = runner.invoke(app, ["funnel", "--db", str(db)])
    assert res.exit_code == 0, res.output
    assert "已发布" in res.output
    assert "2000" in res.output
    assert "CTR" in res.output


def test_funnel_empty_ledger_is_honest(tmp_path: Path) -> None:
    res = runner.invoke(app, ["funnel", "--db", str(tmp_path / "empty.db")])
    assert res.exit_code == 0, res.output
    assert "还没有效果回填" in res.output
