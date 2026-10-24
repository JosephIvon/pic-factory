"""台账与幂等。"""

from __future__ import annotations

from aesthetics_pipeline import ledger


def test_record_and_skip_existing(tmp_path) -> None:
    db = tmp_path / "t.db"
    with ledger.connect(db) as conn:
        ledger.record_asset(
            conn,
            {
                "idempotency_key": "abc", "city": "武汉", "qc_verdict": "pending",
                "local_path": "out/abc.jpg",
            },
        )
        assert "abc" in ledger.successful_keys(conn)


def test_failed_row_does_not_block_retry(tmp_path) -> None:
    """失败的行（没落盘）不算"已完成"——否则一次失败就永久挡住重试。"""
    db = tmp_path / "t.db"
    with ledger.connect(db) as conn:
        ledger.record_asset(
            conn,
            {"idempotency_key": "boom", "provider": "replay", "local_path": None},
        )
        assert "boom" not in ledger.successful_keys(conn)
        ledger.record_asset(
            conn,
            {"idempotency_key": "boom", "provider": "replay", "local_path": "out/x.jpg"},
        )
        assert "boom" in ledger.successful_keys(conn)


def test_accepted_filter(tmp_path) -> None:
    db = tmp_path / "t.db"
    with ledger.connect(db) as conn:
        ledger.record_asset(
            conn,
            {"idempotency_key": "ok", "qc_verdict": "accepted", "local_path": "out/ok.jpg"},
        )
        ledger.record_asset(
            conn,
            {"idempotency_key": "no", "qc_verdict": "rejected", "local_path": "out/no.jpg"},
        )
        assert ledger.successful_keys(conn, only_accepted=True) == {"ok"}


def test_summary_counts(tmp_path) -> None:
    db = tmp_path / "t.db"
    with ledger.connect(db) as conn:
        ledger.record_asset(conn, {"idempotency_key": "a", "qc_verdict": "accepted", "gen_seconds": 1.0})
        ledger.record_asset(conn, {"idempotency_key": "b", "qc_verdict": "rejected", "gen_seconds": 2.0})
        summary = ledger.summary(conn)
    assert summary["assets"] == 2
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert summary["gen_seconds_total"] == 3.0
