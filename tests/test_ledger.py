"""台账与幂等。"""

from __future__ import annotations

import sqlite3

import pytest

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


def test_connection_enables_wal_and_busy_timeout(tmp_path) -> None:
    """面板（子进程）与手工 CLI 可能同时写库：必须开 WAL + busy_timeout。

    默认 rollback journal 下并发写会直接抛 `database is locked`；
    开了 WAL 后读不阻塞写，busy_timeout 把冲突变成短暂等待。
    """
    with ledger.connect(tmp_path / "w.db") as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert str(mode).lower() == "wal"
    assert timeout == 5000


def test_successful_keys_matches_model_exactly(tmp_path) -> None:
    """幂等跳过按模型精确匹配：换档位后旧档产出不算"已完成"。

    只按 provider 过滤时，换模型重跑会一张不生成、却看起来像"模型跑了"——
    这是模型迭代场景里最贵的静默错误。
    """
    db = tmp_path / "m.db"
    with ledger.connect(db) as conn:
        ledger.record_asset(
            conn,
            {"idempotency_key": "k1", "provider": "http_api",
             "model_id": "doubao-seedream-5-0-flash", "local_path": "out/k1.jpg"},
        )
        assert ledger.successful_keys(conn, provider="http_api",
                                      model_id="doubao-seedream-5-0-flash") == {"k1"}
        assert ledger.successful_keys(conn, provider="http_api",
                                      model_id="qwen-image-3.0-pro") == set()
        # 不传 model_id 时保持旧口径（任何档位都算）
        assert ledger.successful_keys(conn, provider="http_api") == {"k1"}


def test_legacy_rows_without_model_do_not_match_exactly(tmp_path) -> None:
    """升级前的行没有 model_id：精确匹配下不算命中，重跑一次（会花钱）。

    这是有意的取舍：宁可多花一次，也不能拿"不知道哪档"的旧图冒充新模型。
    """
    db = tmp_path / "legacy.db"
    with ledger.connect(db) as conn:
        ledger.record_asset(
            conn,
            {"idempotency_key": "old", "provider": "http_api", "local_path": "out/old.jpg"},
        )
        assert ledger.successful_keys(conn, model_id="doubao-seedream-5-0-flash") == set()
        assert ledger.successful_keys(conn) == {"old"}


def test_jobs_persist_and_recover_interrupted(tmp_path) -> None:
    """面板任务落库：重启后能读回；残留 running 标成 interrupted，
    否则提交闸会被一条"内存里没有、库里却 running"的任务永久锁死。"""
    db = tmp_path / "j.db"
    with ledger.connect(db) as conn:
        ledger.upsert_job(
            conn,
            {"id": "j1", "status": "running",
             "steps": [{"name": "generate", "status": "running"}], "log": ["启动"]},
        )
    with ledger.connect(db) as conn:
        jobs = ledger.jobs_all(conn)
        assert jobs[0]["id"] == "j1"
        assert jobs[0]["steps"][0]["name"] == "generate"
        assert jobs[0]["log"] == ["启动"]
        assert ledger.recover_interrupted_jobs(conn) == 1
        assert ledger.jobs_all(conn)[0]["status"] == "interrupted"
        assert ledger.recover_interrupted_jobs(conn) == 0  # 幂等：没有 running 就不再改


def test_funnel_counts_publish_and_metrics(tmp_path) -> None:
    """漏斗：产出 → 放行 → 采纳 → 发布 → 有数据；CTR 只按回填的事实算。"""
    db = tmp_path / "f.db"
    with ledger.connect(db) as conn:
        ledger.record_asset(conn, {"idempotency_key": "a", "local_path": "out/a.jpg",
                                   "qc_verdict": "accepted", "adopted": "adopted"})
        ledger.record_asset(conn, {"idempotency_key": "b", "local_path": "out/b.jpg",
                                   "qc_verdict": "accepted"})
        ledger.record_publication(conn, idempotency_key="a", channel="xiaohongshu",
                                  url="https://example.com/a")
        ledger.record_metrics(conn, idempotency_key="a", channel="xiaohongshu",
                              impressions=1000, clicks=50, saves=8)
        stats = ledger.funnel_summary(conn)
    assert (stats["generated"], stats["accepted"], stats["adopted"]) == (2, 2, 1)
    assert (stats["published"], stats["measured"]) == (1, 1)
    assert stats["totals"]["ctr"] == 0.05
    assert stats["by_channel"][0]["channel"] == "xiaohongshu"


def test_funnel_without_reflow_reports_zero_not_fake(tmp_path) -> None:
    """没回填就是 0——不拿采纳率冒充发布率。"""
    db = tmp_path / "f0.db"
    with ledger.connect(db) as conn:
        ledger.record_asset(conn, {"idempotency_key": "a", "local_path": "out/a.jpg",
                                   "qc_verdict": "accepted", "adopted": "adopted"})
        stats = ledger.funnel_summary(conn)
    assert stats["published"] == 0 and stats["measured"] == 0
    assert stats["totals"]["ctr"] is None


def test_variants_coexist_and_adoption_requires_model(tmp_path) -> None:
    """同一任务同一后端换模型跑 = 两条独立资产，互不覆盖。

    采纳时若有多档而不指定 model，必须报错——"一起采纳"等于把没看过的图
    也标成用了。
    """
    db = tmp_path / "v.db"
    with ledger.connect(db) as conn:
        for model in ("doubao-seedream-5-0-flash", "qwen-image-3.0-pro"):
            ledger.record_asset(
                conn,
                {"idempotency_key": "k", "provider": "http_api", "model_id": model,
                 "local_path": f"out/{model}.jpg", "qc_verdict": "accepted"},
            )
        rows = ledger.assets(conn, "provider = 'http_api'")
        assert len(rows) == 2, "换档重跑必须是两条资产，不能覆盖"
        assert {r["variant"] for r in rows} == {
            "doubao-seedream-5-0-flash", "qwen-image-3.0-pro"
        }

        with pytest.raises(ValueError, match="--model"):
            ledger.record_adoption(conn, "k", "adopted", provider="http_api")

        assert ledger.record_adoption(
            conn, "k", "adopted", provider="http_api", model_id="doubao-seedream-5-0-flash"
        ) == 1
        rows = ledger.assets(conn, "provider = 'http_api'")
        by_variant = {r["variant"]: r["adopted"] for r in rows}
    assert by_variant["doubao-seedream-5-0-flash"] == "adopted"
    assert by_variant["qwen-image-3.0-pro"] is None, "只采纳点过的那一档"


def test_migration_backfills_variant_from_model(tmp_path) -> None:
    """老库主键是 (key, provider)：升级后重建主键，variant 用 model_id 回填。"""
    db = tmp_path / "old.db"
    raw = sqlite3.connect(db)
    raw.execute(
        "CREATE TABLE assets ("
        " idempotency_key TEXT NOT NULL,"
        " provider TEXT NOT NULL DEFAULT 'unknown',"
        " model_id TEXT, local_path TEXT,"
        " PRIMARY KEY (idempotency_key, provider))"
    )
    raw.execute("INSERT INTO assets VALUES ('k', 'http_api', 'm1', 'out/k.jpg')")
    raw.commit()
    raw.close()

    with ledger.connect(db) as conn:
        rows = conn.execute("SELECT variant, model_id FROM assets").fetchall()
        pk = [r[1] for r in sorted(conn.execute("PRAGMA table_info(assets)"),
                                   key=lambda r: r[5]) if r[5]]
    assert [tuple(r) for r in rows] == [("m1", "m1")]
    assert pk == ["idempotency_key", "provider", "variant"]
