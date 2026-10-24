"""L1 素材资产化+复用层：发布量 ≠ 生成量。

台账里已有的合格素材是**花过钱买回来的资产**：同一 (角色×项目×主题) 的素材
换个城市/渠道发布，语义上还是那张图，没必要再花一次生成费。docs/COST-MODEL.md
把这条列为最大降本杠杆（复用 3:1 → 砍 2/3），本模块就是它的机制落点：
按素材语义（而非幂等键）找已落盘的最优产出，本地复制顶上，零边际成本。

**为什么复用层放在台账之上，而不是塞进路由里**：路由（routing.py）是一张
静态规则表——只读 TaskSpec 的特征就给出后端与理由，无副作用、可预测、可逐条
追溯，这是它能进台账当"这张图为什么走这条线"证据的前提。而"能不能复用"依赖
的是**运行时状态**：库里此刻哪些行真的落了盘、质检放没放行、人在抽检台上点没
点"用"、哪条最新——这些只有台账知道。把复用判断写进路由，等于让静态政策去
读可变数据库：规则不再可重放，A/B 对比的归属口径也会被搅乱（复用行仍归属
当前后端，源图的 provider 只在 route_reason 里留证）。所以分层是：路由照常
决定"这条任务交给谁"，复用层在**路由之后、后端之前**拦截——命中就把源文件
复制成该后端的产物命名，走现有 record_asset 通道落一行 est_cost=0.0。
一行 SQL 都不用建新表：复用行与生成行同库同 schema，成本账（est_cost 求和）
天然把"省下的生成费"记成 0 而非漏记，这正是这套账的不变量。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from . import ledger
from .models import TaskSpec
from .routing import RouteDecision

# 复用匹配的语义键：素材"长什么样"由 角色 + 项目 + 主题 决定，城市/渠道不进键——
# 幂等键含城市/渠道/版本，同键匹配就等于没复用（那正是现状 generate 的口径）。
# 具体口径写在 find_reusable 的 SQL 条件里（有角色时多一个 character 条件），
# 不额外维护一份常量——曾经有过 `_MATCH_COLUMNS`，但它从未被读取，只会与 SQL 漂移。


def find_reusable(conn, spec: TaskSpec) -> dict[str, Any] | None:
    """给一条待生成任务找**已存在、已落盘、质量最优**的可复用产出。

    匹配口径是素材语义而非幂等键：
      - 有角色：character + project + theme（"同一个人同一个叙事"才是同一张素材）；
      - 角色为空：退化到 project + theme（泛化人物，任何一条合格素材都能顶）。

    优先级（tier 小者胜，同档取 created_at 最新）：
      0. adopted='adopted' 且 qc_verdict='accepted' —— 机器放行且人决定要用；
      1. qc_verdict='accepted' —— 机器放行、人还没处理；
      2. 其余落盘行（pending / 未质检）。

    永不复用：qc_verdict='rejected' 或 adopted='rejected' 的行（被拦/被人弃用的
    素材再复制一遍只是把废品搬回家）；local_path 为空的行（失败没落盘，无源可拷）。
    """
    clauses = ["local_path IS NOT NULL", "project = ?", "theme = ?"]
    params: list[Any] = [spec.project, spec.theme]
    if spec.character:
        clauses.append("character = ?")
        params.append(spec.character)
    # 排除两类 rejected：质检拦截（机器）与抽检台弃用（人）
    clauses.append("COALESCE(qc_verdict, '') != 'rejected'")
    clauses.append("COALESCE(adopted, '') != 'rejected'")

    sql = (
        "SELECT *, "
        "CASE WHEN adopted = 'adopted' AND qc_verdict = 'accepted' THEN 0 "
        "     WHEN qc_verdict = 'accepted' THEN 1 "
        "     ELSE 2 END AS reuse_tier "
        "FROM assets WHERE " + " AND ".join(clauses) +
        # 同档取最新：created_at 为 NULL（极少见）排最后；键做稳定兜底
        " ORDER BY reuse_tier ASC, created_at DESC, idempotency_key DESC LIMIT 1"
    )
    row = conn.execute(sql, params).fetchone()
    return dict(row) if row else None


def reuse_dest_name(key: str, backend_name: str, source_path: Path) -> str:
    """复用产物的落盘文件名：本后端产物命名，扩展名跟随源文件。

    带 __backend 后缀的原因与 faceswap 一致：同一幂等键可能被多条后端各自
    产出一行，DB 复合主键不冲突但磁盘会冲突；复用行也不能覆盖本后端
    真实生成过的同名文件。
    """
    return f"{key}__{backend_name}{source_path.suffix or '.png'}"


def reuse_asset(
    conn,
    *,
    spec: TaskSpec,
    backend_name: str,
    prompt: str,
    decision: RouteDecision | None,
    out_dir: Path,
) -> dict[str, Any] | None:
    """命中复用则：复制源文件 → 走 record_asset 落一行，返回该行的写入字段。

    返回 None 表示没有可复用素材（或源文件已不在磁盘上——台账记过但文件丢了，
    宁可重新花钱也不复制一张不存在的图），调用方照常把任务发给后端。

    复用行的账目口径：
      - provider = 本后端（复用行的归属与生成行同构，A/B 分组、跳过口径都不破）；
      - est_cost = 0.0 且 gen_seconds = 0.0 —— 本地复制零边际成本，这是 L1 的
        存在理由，也是测试要钉死的不变量；
      - qc_verdict = 'pending' —— 复制品没被本管线质检过，判定不能白拿源行的；
      - route_reason 带上源（key@provider），逐条可查"这张是从哪复用的"。
    """
    source = find_reusable(conn, spec)
    if source is None:
        return None
    src_path = ledger.asset_path(source)
    if src_path is None or not src_path.exists():
        return None

    images_dir = Path(out_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    dest = images_dir / reuse_dest_name(spec.idempotency_key, backend_name, src_path)
    shutil.copyfile(src_path, dest)

    reason = (decision.reason if decision else "复用已落盘素材").strip()
    row: dict[str, Any] = {
        "idempotency_key": spec.idempotency_key,
        "city": spec.city, "project": spec.project, "theme": spec.theme,
        "channel": spec.channel, "character": spec.character,
        "route": spec.path,
        "provider": backend_name,
        "node_id": None,
        "file_name": dest.name,
        "local_path": str(dest),
        "prompt": prompt,
        "gen_seconds": 0.0,
        "est_cost": 0.0,
        "qc_verdict": "pending",
        "route_rule_id": decision.rule_id if decision else None,
        "route_reason": (
            f"{reason}｜复用 {source['idempotency_key']}@{source['provider']}"
            "（本地复制，零边际成本）"
        ),
        # 复用行没有"本后端实际用的模型"可言（图是别人生成的），
        # model 三列留空；源档位顺带记进 model_reason 供追溯。
        "model_id": None,
        "model_rule_id": None,
        "model_reason": f"复用自 {source['idempotency_key']}@{source['provider']}"
        + (f"，源档位 {source['model_id']}" if source.get("model_id") else ""),
        # 显式时间戳：复用行要能参与"同档取最新"排序（重跑刷新为最新一条）。
        "created_at": ledger.now(),
    }
    ledger.record_asset(conn, row)
    return row
