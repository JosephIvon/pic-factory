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
    -- 档位维度（主键的一部分）：同一任务在同一后端用不同模型跑，是**两条独立资产**。
    -- 不能只靠 model_id 做列：列可以 NULL，而主键列必须非空且唯一。
    variant TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (idempotency_key, provider, variant)
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
-- 面板任务（P0-2 落库）：进程重启后仍能回答"这批跑到哪一步"。
-- steps/log 用 JSON 存：面板只读取展示，不做 SQL 级查询，没必要拆表。
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    steps TEXT, log TEXT,
    created_at TEXT, updated_at TEXT,
    note TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
-- 回传表（发布）：素材发到哪个渠道、什么时候、什么链接。
CREATE TABLE IF NOT EXISTS publications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'unknown',
    channel TEXT,
    url TEXT,
    published_at TEXT,
    note TEXT,
    created_at TEXT,
    UNIQUE (idempotency_key, provider, channel, url)
);
-- 效果数据：同一素材同一渠道可按日期多次测量（meas_id 幂等）。
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'unknown',
    channel TEXT,
    measured_at TEXT,
    impressions INTEGER, clicks INTEGER, saves INTEGER, comments INTEGER, shares INTEGER,
    note TEXT,
    created_at TEXT,
    UNIQUE (idempotency_key, provider, channel, measured_at)
);
"""

# 索引单独执行：老库可能缺某个被索引的列（例如极早期版本没有 qc_verdict），
# 建表脚本里直接建索引会让 connect() 在迁移之前就炸掉。顺序改为
# 建表 → 补列/迁主键 → 建索引。
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_assets_qc ON assets(qc_verdict);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_publications_key ON publications(idempotency_key);
CREATE INDEX IF NOT EXISTS idx_metrics_key ON metrics(idempotency_key);
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
    """主键要升级成 (idempotency_key, provider, variant)。

    演进史：单列 (key) → (key, provider)（API 与本地栈不互相覆盖）→
    (key, provider, variant)（同一后端换模型也不互相覆盖——"换档重跑"本该
    产出两条独立资产，而不是把上一条覆盖掉）。
    """
    info = list(conn.execute("PRAGMA table_info(assets)"))
    if not info:
        return False
    pk_cols = [row[1] for row in sorted(info, key=lambda r: r[5]) if row[5]]
    return pk_cols != ["idempotency_key", "provider", "variant"]


def _rebuild_assets(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE assets RENAME TO assets_legacy")
    conn.executescript(SCHEMA)
    legacy_cols = {row[1] for row in conn.execute("PRAGMA table_info(assets_legacy)")}
    target_cols = {row[1] for row in conn.execute("PRAGMA table_info(assets)")}
    shared = [c for c in ("idempotency_key", "provider", "variant",
                          "city", "project", "theme",
                          "channel", "character", "route", "node_id", "file_name",
                          "local_path", "prompt", "gen_seconds", "est_cost",
                          "qc_verdict", "qc_reasons", "qc_rules", "qc_soft",
                          "face_similarity", "text_found", "swapped_path",
                          "face_similarity_after", "route_rule_id", "route_reason",
                          "model_id", "model_rule_id", "model_reason",
                          "adopted", "adopted_at", "adopt_note",
                          "created_at")
              if c in target_cols and (c in legacy_cols or c == "variant")]
    # variant 的历史值：老库没有这列 → 用 model_id 回填（API 行有档位；
    # matrix/local 行 model_id 为 NULL → 空串，与新建行一致）。
    select: list[str] = []
    for col in shared:
        if col == "variant":
            if "variant" in legacy_cols:
                select.append("COALESCE(NULLIF(variant, ''), model_id, '') AS variant")
            else:
                select.append("COALESCE(model_id, '') AS variant")
        else:
            select.append(col)
    conn.execute(
        f"INSERT OR REPLACE INTO assets ({', '.join(shared)}) "
        f"SELECT {', '.join(select)} FROM assets_legacy"
    )
    conn.execute("DROP TABLE assets_legacy")


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, decl in MIGRATIONS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if existing and column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    if _needs_pk_rebuild(conn):
        _rebuild_assets(conn)
    conn.executescript(INDEXES)


@contextmanager
def connect(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    # 面板（CLI 子进程）与手工命令可能同时写同一个库：WAL 让"读"不阻塞"写"，
    # busy_timeout 把"database is locked"从随机 500 变成短暂等待后重试。
    # 单机 SQLite 不开这两项时，并发写会直接抛 OperationalError。
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def successful_keys(
    conn: sqlite3.Connection,
    only_accepted: bool = False,
    provider: str | None = None,
    model_id: str | None = None,
) -> set[str]:
    """已成功产出的幂等键。用于跳过，避免重复花钱。

    只算**真正落过盘**的行（local_path 非空）：失败的行也留在表里，
    如果把它们也算成"已完成"，一次失败就会永久挡住这条任务的后续重试。

    按后端过滤：同一批任务用不同后端跑（API vs 本地栈做 A/B 对比）时，
    另一条路线的产出不算"已完成"，否则本地栈永远跑不出来。

    按模型过滤（`model_id` 非空时**精确匹配**）：同一后端换档位重跑时，
    另一档的产出不算已完成——否则"换模型重跑"会静默复用旧图（幂等键里
    不含模型，只按 provider 过滤时这是必然结果）。历史行的 model_id 为
    NULL，精确匹配下**不算命中**：升级后第一次重跑会重新生成并计费，
    cli 会打印提示，这是有意的取舍（宁可多花一次，不拿旧图冒充新模型）。
    """
    sql = "SELECT idempotency_key FROM assets"
    clauses: list[str] = ["local_path IS NOT NULL"]
    params: list[Any] = []
    if only_accepted:
        clauses.append("qc_verdict = 'accepted'")
    if provider:
        clauses.append("provider = ?")
        params.append(provider)
    if model_id is not None:
        clauses.append("model_id = ?")
        params.append(model_id)
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
        "route", "provider", "variant", "node_id", "file_name", "local_path", "prompt",
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

    # 档位维度（主键的一部分）：显式 variant > model_id > 从既有行推断。
    # 同一任务在同一后端可能有多条（不同模型），局部更新不指明档位就会
    # 改错行或插出孤儿行——这里宁可报错，也不猜。
    if "variant" in row:
        data["variant"] = str(row.get("variant") or "")
    elif "model_id" in row:
        data["variant"] = str(row.get("model_id") or "")
    else:
        variants = [
            r[0]
            for r in conn.execute(
                "SELECT variant FROM assets WHERE idempotency_key = ? AND provider = ?",
                (data["idempotency_key"], data["provider"]),
            )
        ]
        if len(variants) == 1:
            data["variant"] = variants[0]
        elif not variants:
            data["variant"] = ""
        else:
            raise ValueError(
                "该任务有多个模型档位产出，局部更新必须指定 model_id"
                f"（现有档位：{sorted(variants)}）"
            )
    for key in ("qc_reasons", "qc_rules", "qc_soft", "text_found"):
        if isinstance(data.get(key), (list, dict)):
            data[key] = json.dumps(data[key], ensure_ascii=False)

    # 本次真正要写入的列（created_at 只在新建时给）
    touched = [c for c in cols if c in row]
    if "created_at" not in touched:
        touched.append("created_at")

    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(
        f"{c} = excluded.{c}"
        for c in touched
        if c not in ("idempotency_key", "provider", "variant")
    )
    conn.execute(
        f"INSERT INTO assets ({', '.join(cols)}) VALUES ({placeholders})"
        f" ON CONFLICT(idempotency_key, provider, variant) DO UPDATE SET {updates}",
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
    model_id: str | None = None,
) -> int:
    """记录一次人工采纳决定，返回写入的行数（0 = 台账里没有这条产出）。

    走 record_asset 的 ON CONFLICT DO UPDATE 通道：patch 里**只带采纳三列**，
    qc_verdict / local_path / prompt 等其它列不在 touched 里，一概不动。
    provider 缺省时该任务的全部行一起改（面板卡片自带 provider，逐行点）。

    同一后端可能有多个**模型档位**产出（variant 不同）：这时不指定 model_id
    就不知道该采纳哪一张——直接报错，绝不"一起采纳"（那等于把没看过的图也标成用了）。
    """
    if decision not in ADOPT_DECISIONS:
        raise ValueError(f"未知采纳决定 {decision!r}，只接受 {ADOPT_DECISIONS}")
    where = "idempotency_key = ?"
    params: list[Any] = [idempotency_key]
    if provider:
        where += " AND provider = ?"
        params.append(provider)
    if model_id is not None:
        where += " AND COALESCE(model_id, '') = ?"
        params.append(model_id)
    rows = conn.execute(f"SELECT provider, variant, model_id FROM assets WHERE {where}", params).fetchall()
    if not rows:
        return 0
    variants = {(r["provider"], r["variant"]) for r in rows}
    if len(variants) > 1:
        options = sorted({str(r["model_id"] or "（无档位）") for r in rows})
        raise ValueError(
            f"该任务有 {len(variants)} 个档位产出，采纳必须用 --model 指定：{options}"
        )
    for row in rows:
        patch: dict[str, Any] = {
            "idempotency_key": idempotency_key,
            "provider": row["provider"],
            "variant": row["variant"],
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
    整个文件夹拷到别的机器时）这些绝对路径会失效，画廊就会变成一片空图。
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


# ---------------------------------------------------------------- 面板任务（落库）

JOB_STATUSES = ("running", "done", "failed", "interrupted")


def upsert_job(conn: sqlite3.Connection, job: dict[str, Any]) -> None:
    """写入/更新一条面板任务。steps/log 用 JSON 存。"""
    conn.execute(
        "INSERT INTO jobs (id, status, steps, log, created_at, updated_at, note)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(id) DO UPDATE SET status=excluded.status, steps=excluded.steps,"
        " log=excluded.log, updated_at=excluded.updated_at, note=excluded.note",
        [
            job["id"],
            job.get("status") or "running",
            json.dumps(job.get("steps") or [], ensure_ascii=False),
            json.dumps(job.get("log") or [], ensure_ascii=False),
            job.get("created_at") or now(),
            job.get("updated_at") or now(),
            job.get("note"),
        ],
    )


def jobs_all(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", [limit]
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        job = dict(row)
        job["steps"] = json.loads(job.get("steps") or "[]")
        job["log"] = json.loads(job.get("log") or "[]")
        out.append(job)
    return out


def job_get(conn: sqlite3.Connection, job_id: str) -> dict[str, Any] | None:
    for job in jobs_all(conn, limit=1000):
        if job["id"] == job_id:
            return job
    return None


def recover_interrupted_jobs(conn: sqlite3.Connection) -> int:
    """把上次进程留下的 running 任务标成 interrupted。

    没有这一步，面板重启后"已有任务在跑"的 409 会永久生效（内存里其实没有
    任何线程）——任务没在跑，闸门却一直关着。单机部署下按"进程重启 =
    上次的批已中断"处理，是最诚实的口径。
    """
    cur = conn.execute(
        "UPDATE jobs SET status = 'interrupted', updated_at = ? WHERE status = 'running'",
        [now()],
    )
    return int(cur.rowcount or 0)


# ---------------------------------------------------------------- 发布与效果（回传表）

METRIC_FIELDS = ("impressions", "clicks", "saves", "comments", "shares")


def record_publication(
    conn: sqlite3.Connection,
    *,
    idempotency_key: str,
    provider: str = "unknown",
    channel: str | None = None,
    url: str | None = None,
    published_at: str | None = None,
    note: str | None = None,
) -> None:
    """登记一次发布。同 (key, provider, channel, url) 视为同一条记录。"""
    conn.execute(
        "INSERT OR REPLACE INTO publications"
        " (idempotency_key, provider, channel, url, published_at, note, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        [idempotency_key, provider, channel, url, published_at or now(), note, now()],
    )


def publications(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in conn.execute("SELECT * FROM publications ORDER BY published_at, id")
    ]


def record_metrics(
    conn: sqlite3.Connection,
    *,
    idempotency_key: str,
    provider: str = "unknown",
    channel: str | None = None,
    measured_at: str | None = None,
    impressions: int | None = None,
    clicks: int | None = None,
    saves: int | None = None,
    comments: int | None = None,
    shares: int | None = None,
    note: str | None = None,
) -> None:
    """登记一次效果测量（同 key/渠道/日期 覆盖，便于"数据更新后补录"）。"""
    conn.execute(
        "INSERT OR REPLACE INTO metrics"
        " (idempotency_key, provider, channel, measured_at, impressions, clicks,"
        "  saves, comments, shares, note, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            idempotency_key, provider, channel, measured_at or now(),
            impressions, clicks, saves, comments, shares, note, now(),
        ],
    )


def metrics(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute("SELECT * FROM metrics ORDER BY measured_at, id")]


def funnel_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    """漏斗：产出 → 放行 → 人采纳 → 已发布 → 有数据。

    "发布"和"效果"是**人回填的事实**，不是系统推断：没有回传就如实显示 0，
    不拿"采纳率"冒充"发布率"。
    """
    def count(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0] or 0)

    generated = count("SELECT COUNT(*) FROM assets WHERE local_path IS NOT NULL")
    accepted = count("SELECT COUNT(*) FROM assets WHERE qc_verdict = 'accepted'")
    adopted = count("SELECT COUNT(*) FROM assets WHERE adopted = 'adopted'")
    published = count("SELECT COUNT(DISTINCT idempotency_key) FROM publications")
    measured = count("SELECT COUNT(DISTINCT idempotency_key) FROM metrics")

    totals = conn.execute(
        "SELECT COALESCE(SUM(impressions), 0) AS impressions,"
        " COALESCE(SUM(clicks), 0) AS clicks,"
        " COALESCE(SUM(saves), 0) AS saves,"
        " COALESCE(SUM(comments), 0) AS comments,"
        " COALESCE(SUM(shares), 0) AS shares FROM metrics"
    ).fetchone()
    impressions = int(totals["impressions"] or 0)
    clicks = int(totals["clicks"] or 0)

    by_channel: list[dict[str, Any]] = []
    for row in conn.execute(
        "SELECT COALESCE(channel, '-') AS channel, COUNT(DISTINCT idempotency_key) AS assets,"
        " COALESCE(SUM(impressions), 0) AS impressions, COALESCE(SUM(clicks), 0) AS clicks,"
        " COALESCE(SUM(saves), 0) AS saves"
        " FROM metrics GROUP BY COALESCE(channel, '-') ORDER BY impressions DESC"
    ):
        item = dict(row)
        item["ctr"] = round(item["clicks"] / item["impressions"], 4) if item["impressions"] else None
        by_channel.append(item)

    return {
        "generated": generated,
        "accepted": accepted,
        "adopted": adopted,
        "published": published,
        "measured": measured,
        "publish_rate": round(published / adopted, 4) if adopted else None,
        "totals": {
            "impressions": impressions,
            "clicks": clicks,
            "saves": int(totals["saves"] or 0),
            "comments": int(totals["comments"] or 0),
            "shares": int(totals["shares"] or 0),
            "ctr": round(clicks / impressions, 4) if impressions else None,
        },
        "by_channel": by_channel,
    }
