"""SQLite 台账。

为什么用 SQLite：零部署成本，一条命令起；schema 是标准 SQL，
将来迁 Postgres 只换连接层。

台账是"成本/耗时/通过率"这类问题的唯一答案来源，不能靠感觉报数。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import DB_PATH, resolve_repo_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    idempotency_key TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'unknown',
    city TEXT, project TEXT, theme TEXT, channel TEXT, character TEXT,
    route TEXT, node_id TEXT, file_name TEXT, local_path TEXT,
    prompt TEXT,
    gen_seconds REAL, est_cost REAL,
    qc_verdict TEXT, qc_reasons TEXT, qc_rules TEXT, qc_soft TEXT,
    face_similarity REAL, text_found TEXT,
    swapped_path TEXT, face_similarity_after REAL,
    route_rule_id TEXT, route_reason TEXT,
    model_id TEXT, model_rule_id TEXT, model_reason TEXT,
    adopted TEXT, adopted_at TEXT, adopt_note TEXT,
    created_at TEXT,
    PRIMARY KEY (idempotency_key, provider)
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT, finished_at TEXT, provider TEXT,
    requested INT, success INT, failed INT, rejected INT,
    est_cost REAL, notes TEXT
);
CREATE TABLE IF NOT EXISTS copies (
    idempotency_key TEXT PRIMARY KEY,
    title TEXT, body TEXT, tags TEXT, banned_hits TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_assets_qc ON assets(qc_verdict);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# 轻量迁移：给已存在的库补上后来新增的列。
# CREATE TABLE IF NOT EXISTS 不会改动已存在的表，所以列的演进要显式迁移。
MIGRATIONS: list[tuple[str, str, str]] = [
    ("assets", "swapped_path", "TEXT"),
    ("assets", "face_similarity_after", "REAL"),
    ("assets", "route_rule_id", "TEXT"),
    ("assets", "route_reason", "TEXT"),
    # 模型档位维度（model_choice 选定）：后端规则管"哪条线"，这三列管"线上用哪个模型"。
    ("assets", "model_id", "TEXT"),
    ("assets", "model_rule_id", "TEXT"),
    ("assets", "model_reason", "TEXT"),
    # 抽检台人工采纳：adopted 取值 'adopted' / 'rejected' / NULL（未处理）。
    # 与 qc_verdict 是两回事——质检放行是机器的判断，采纳是人在抽检台上点的。
    ("assets", "adopted", "TEXT"),
    ("assets", "adopted_at", "TEXT"),
    ("assets", "adopt_note", "TEXT"),
]


def _needs_pk_rebuild(conn: sqlite3.Connection) -> bool:
    """主键要从单列 (idempotency_key) 升级成 (idempotency_key, provider)。

    不升级的话，API 与本地栈跑同一个任务会互相覆盖对方的产出，A/B 对比做不了。
    """
    info = list(conn.execute("PRAGMA table_info(assets)"))
    if not info:
        return False
    pk_cols = [row[1] for row in sorted(info, key=lambda r: r[5]) if row[5]]
    return pk_cols == ["idempotency_key"]


def _rebuild_assets(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE assets RENAME TO assets_legacy")
    conn.executescript(SCHEMA)
    legacy_cols = {row[1] for row in conn.execute("PRAGMA table_info(assets_legacy)")}
    target_cols = {row[1] for row in conn.execute("PRAGMA table_info(assets)")}
    shared = [c for c in ("idempotency_key", "provider", "city", "project", "theme",
                          "channel", "character", "route", "node_id", "file_name",
                          "local_path", "prompt", "gen_seconds", "est_cost",
                          "qc_verdict", "qc_reasons", "qc_rules", "qc_soft",
                          "face_similarity", "text_found", "swapped_path",
                          "face_similarity_after", "route_rule_id", "route_reason",
                          "model_id", "model_rule_id", "model_reason",
                          "adopted", "adopted_at", "adopt_note",
                          "created_at") if c in legacy_cols and c in target_cols]
    conn.execute(
        f"INSERT OR REPLACE INTO assets ({', '.join(shared)}) "
        f"SELECT {', '.join(shared)} FROM assets_legacy"
    )
    conn.execute("DROP TABLE assets_legacy")


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, decl in MIGRATIONS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if existing and column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    if _needs_pk_rebuild(conn):
        _rebuild_assets(conn)


@contextmanager
def connect(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def successful_keys(
    conn: sqlite3.Connection, only_accepted: bool = False, provider: str | None = None
) -> set[str]:
    """已成功产出的幂等键。用于跳过，避免重复花钱。

    只算**真正落过盘**的行（local_path 非空）：失败的行也留在表里，
    如果把它们也算成"已完成"，一次失败就会永久挡住这条任务的后续重试。

    按后端过滤：同一批任务用不同后端跑（API vs 本地栈做 A/B 对比）时，
    另一条路线的产出不算"已完成"，否则本地栈永远跑不出来。
    """
    sql = "SELECT idempotency_key FROM assets"
    clauses: list[str] = ["local_path IS NOT NULL"]
    params: list[Any] = []
    if only_accepted:
        clauses.append("qc_verdict = 'accepted'")
    if provider:
        clauses.append("provider = ?")
        params.append(provider)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    return {row["idempotency_key"] for row in conn.execute(sql, params)}


def record_asset(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    """写入或**局部更新**一条产出记录。

    注意：不能用 INSERT OR REPLACE —— 那样只传 qc_verdict 的局部更新会把
    city/project/prompt 等其它列整体覆盖成 NULL（台账自我销毁，实测踩过）。
    这里用 ON CONFLICT DO UPDATE，只更新本次真正传入的列。
    """
    cols = [
        "idempotency_key", "city", "project", "theme", "channel", "character",
        "route", "provider", "node_id", "file_name", "local_path", "prompt",
        "gen_seconds", "est_cost", "qc_verdict", "qc_reasons", "qc_rules",
        "qc_soft", "face_similarity", "text_found", "swapped_path",
        "face_similarity_after", "route_rule_id", "route_reason",
        "model_id", "model_rule_id", "model_reason",
        "adopted", "adopted_at", "adopt_note", "created_at",
    ]
    data = {c: row.get(c) for c in cols}
    data.setdefault("created_at", now())
    if not data.get("provider"):
        # 局部更新（质检/换脸）没带 provider：沿用该任务既有的后端，
        # 否则复合主键对不上，会插出一条 provider=NULL 的孤儿记录。
        existing = conn.execute(
            "SELECT provider FROM assets WHERE idempotency_key = ? ORDER BY created_at LIMIT 1",
            (data["idempotency_key"],),
        ).fetchone()
        data["provider"] = existing["provider"] if existing else "unknown"
    for key in ("qc_reasons", "qc_rules", "qc_soft", "text_found"):
        if isinstance(data.get(key), (list, dict)):
            data[key] = json.dumps(data[key], ensure_ascii=False)

    # 本次真正要写入的列（created_at 只在新建时给）
    touched = [c for c in cols if c in row]
    if "created_at" not in touched:
        touched.append("created_at")

    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c} = excluded.{c}" for c in touched if c not in ("idempotency_key", "provider"))
    conn.execute(
        f"INSERT INTO assets ({', '.join(cols)}) VALUES ({placeholders})"
        f" ON CONFLICT(idempotency_key, provider) DO UPDATE SET {updates}",
        [data[c] for c in cols],
    )


# 抽检台"用/不用"的合法取值；NULL 表示人还没处理过。
ADOPT_DECISIONS = ("adopted", "rejected")


def record_adoption(
    conn: sqlite3.Connection,
    idempotency_key: str,
    decision: str = "adopted",
    note: str | None = None,
    provider: str | None = None,
) -> int:
    """记录一次人工采纳决定，返回写入的行数（0 = 台账里没有这条产出）。

    走 record_asset 的 ON CONFLICT DO UPDATE 通道：patch 里**只带采纳三列**，
    qc_verdict / local_path / prompt 等其它列不在 touched 里，一概不动。
    provider 缺省时该任务的全部行一起改（面板卡片自带 provider，逐行点）。
    """
    if decision not in ADOPT_DECISIONS:
        raise ValueError(f"未知采纳决定 {decision!r}，只接受 {ADOPT_DECISIONS}")
    where = "idempotency_key = ?"
    params: list[Any] = [idempotency_key]
    if provider:
        where += " AND provider = ?"
        params.append(provider)
    rows = conn.execute(f"SELECT provider FROM assets WHERE {where}", params).fetchall()
    if not rows:
        return 0
    for row in rows:
        patch: dict[str, Any] = {
            "idempotency_key": idempotency_key,
            "provider": row["provider"],
            "adopted": decision,
            "adopted_at": now(),
        }
        # note 没传就不带这一列：重改"用/不用"不该把上次写的备注抹掉
        if note is not None:
            patch["adopt_note"] = note
        record_asset(conn, patch)
    return len(rows)


def adoption_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    """采纳率 = adopted / generated。

    分母口径与 successful_keys 一致：只算**真正落过盘**的产出行
    （local_path 非空）。失败的行没人能点"用"，混进分母会稀释真实采纳率。
    """
    overall = conn.execute(
        "SELECT COUNT(*) generated,"
        " SUM(CASE WHEN adopted = 'adopted' THEN 1 ELSE 0 END) adopted,"
        " SUM(CASE WHEN adopted = 'rejected' THEN 1 ELSE 0 END) rejected"
        " FROM assets WHERE local_path IS NOT NULL"
    ).fetchone()
    by_provider: dict[str, Any] = {}
    for row in conn.execute(
        "SELECT provider, COUNT(*) generated,"
        " SUM(CASE WHEN adopted = 'adopted' THEN 1 ELSE 0 END) adopted,"
        " SUM(CASE WHEN adopted = 'rejected' THEN 1 ELSE 0 END) rejected"
        " FROM assets WHERE local_path IS NOT NULL"
        " GROUP BY provider ORDER BY provider"
    ):
        generated = row["generated"] or 0
        by_provider[row["provider"]] = {
            "generated": generated,
            "adopted": row["adopted"] or 0,
            "rejected": row["rejected"] or 0,
            "adoption_rate": round((row["adopted"] or 0) / generated, 4) if generated else None,
        }
    generated = overall["generated"] or 0
    return {
        "generated": generated,
        "adopted": overall["adopted"] or 0,
        "rejected": overall["rejected"] or 0,
        "adoption_rate": round((overall["adopted"] or 0) / generated, 4) if generated else None,
        "by_provider": by_provider,
    }


def record_run(conn: sqlite3.Connection, row: dict[str, Any]) -> int:
    cols = ["started_at", "finished_at", "provider", "requested", "success", "failed", "rejected", "est_cost", "notes"]
    data = {c: row.get(c) for c in cols}
    cur = conn.execute(
        f"INSERT INTO runs ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
        [data[c] for c in cols],
    )
    return int(cur.lastrowid or 0)


def record_copy(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    data = {
        "idempotency_key": row["idempotency_key"],
        "title": row.get("title"),
        "body": row.get("body"),
        "tags": json.dumps(row.get("tags", []), ensure_ascii=False),
        "banned_hits": json.dumps(row.get("banned_hits", []), ensure_ascii=False),
        "created_at": now(),
    }
    conn.execute(
        "INSERT OR REPLACE INTO copies (idempotency_key, title, body, tags, banned_hits, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [data["idempotency_key"], data["title"], data["body"], data["tags"], data["banned_hits"], data["created_at"]],
    )


def assets(conn: sqlite3.Connection, where: str = "", params: tuple = ()) -> list[dict[str, Any]]:
    sql = "SELECT * FROM assets"
    if where:
        sql += f" WHERE {where}"
    sql += " ORDER BY created_at, idempotency_key"
    return [dict(row) for row in conn.execute(sql, params)]


def asset_path(row: dict[str, Any], field: str = "local_path") -> Path | None:
    """取产出文件的**当前可用路径**。

    台账里存的是绝对路径（生成当时的路径）。仓库整体搬到别处后（换机器、
    把整个文件夹拷过去）这些绝对路径会失效，画廊就会变成一片空图。
    这里按仓库内子目录（out/ evidence/ fixtures/）把路径重新锚定到当前仓库根。
    解析不出来返回 None —— 调用方要能接受"文件不在了"。
    """
    return resolve_repo_path(row.get(field))


def summary(conn: sqlite3.Connection) -> dict[str, Any]:
    total = conn.execute("SELECT COUNT(*) c FROM assets").fetchone()["c"]
    accepted = conn.execute(
        "SELECT COUNT(*) c FROM assets WHERE qc_verdict = 'accepted'"
    ).fetchone()["c"]
    rejected = conn.execute(
        "SELECT COUNT(*) c FROM assets WHERE qc_verdict = 'rejected'"
    ).fetchone()["c"]
    pending = total - accepted - rejected
    secs = conn.execute("SELECT COALESCE(SUM(gen_seconds), 0) s FROM assets").fetchone()["s"]
    cost = conn.execute("SELECT COALESCE(SUM(est_cost), 0) c FROM assets").fetchone()["c"]
    cost_n = conn.execute("SELECT COUNT(est_cost) c FROM assets WHERE est_cost IS NOT NULL").fetchone()["c"]
    adoption = adoption_summary(conn)
    return {
        "assets": total,
        "accepted": accepted,
        "rejected": rejected,
        "pending": pending,
        "accept_rate": round(accepted / total, 4) if total else None,
        "gen_seconds_total": round(secs, 2),
        "est_cost_total": round(cost, 4) if cost_n else None,
        "priced_assets": cost_n,
        # 人工采纳口径（抽检台点击）：机器放行 ≠ 人决定要用，两套数分开报。
        # 注意与上面的 accepted/rejected（质检判定）不是一回事。
        "adoption_rate": adoption["adoption_rate"],
        "adoption": adoption,
    }
