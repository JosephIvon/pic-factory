"""命令行入口。

一条条命令对应演示脚本里的一步：
    plan            展开任务矩阵，看看会产出什么
    generate        批量生成（含幂等跳过、失败精确重试、成本台账）
    qc              跑三路质检，判定出厂/拦截
    adopt           抽检台人工采纳：质检放行后再由人点"用/不用"
    adoption        打印采纳率（adopted/generated，按后端分组）
    library         复用素材盘点（L1）：按角色×项目×主题看可复用/已采纳的资产
    copy            生成文案并过禁用词闸
    report          出拼图 + CSV + JSON 报表
    compliance      合规审计（免责声明/图文成对/画面文字/禁用词；不过则退出码 1）
    publish         打发布包（fail-closed；manifest 即 AI 隐式标识）
    calibrate-face  在标注样本上标定人脸阈值
    packs           列出可用域包（换行业不改代码）
    compare         后端可用性预检（对比前置；真正的对比见 README 第 7 节）
    panel           起轻量 Web 面板（任务提交 + 画廊 + 明细）
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    import typer
    from rich.console import Console
    from rich.table import Table
except ImportError as exc:  # pragma: no cover —— 依赖没装时给可执行的提示
    # 裸 ModuleNotFoundError 在陌生机器上最难查：先说清缺什么、怎么装。
    missing = getattr(exc, "name", None) or "依赖"
    print(
        f"缺依赖 {missing}。先装本项目：pip install -e .\n"
        "（要起 Web 面板再加：pip install -e \".[panel]\"）",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

# Windows 控制台默认代码页会把中文/符号写崩（UnicodeEncodeError）。
# 演示要在别人机器上跑，这一步必须自己做。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from . import compliance as compliance_mod
from . import config, copy as copy_mod, ledger, library, report as report_mod, routing
from .cost_guard import BudgetGuard
from .model_choice import ModelDecision, choose_model
from .models import GenItem, GenOutcome, TaskSpec
from .pack import try_load_pack
from .provider import build_provider
from .provider.models_spec import get as get_model_spec
from .qc.face import (
    FaceGate,
    face_gate_unavailable_reason,
    missing_face_reason,
    should_reject_on_identity,
)
from .qc.rules import rule_findings
from .router import Router
from .routing import RouteDecision
from .taskspec import character_profiles, expand, filter_specs, load_yaml, pack_matrix, pack_scene, scene_config

app = typer.Typer(add_completion=False, help="医美 AI 图文闭环 · MVP 演示 CLI")
console = Console()


def _router() -> Router:
    # 走域包（AP_PACK 未设时 pack_matrix/pack_scene 回退到 data/prompts，行为不变）
    matrix = pack_matrix()
    scene = pack_scene()
    return Router(matrix, scene)


def _specs() -> list[TaskSpec]:
    return expand()


def _print_route_summary(
    decisions: dict[str, RouteDecision],
    specs: list[TaskSpec],
    title: str,
    with_reason: bool = True,
) -> None:
    """打印路由计划：每条任务走哪条线 + 命中哪条规则。

    理由是**规则的常量**，所以按规则列一次就够，不必每行重复
    （重复既占宽度又难读）。
    """
    table = Table(title=title)
    for col in ("项目", "主题", "角色", "后端", "路由规则"):
        table.add_column(col, overflow="ellipsis", no_wrap=True)
    for spec in specs:
        d = decisions[spec.idempotency_key]
        table.add_row(
            spec.project, spec.theme, spec.character or "-", d.provider, d.rule_id
        )
    console.print(table)

    summary = routing.summarize(list(decisions.values()))
    for provider, slot in sorted(summary.items()):
        rules = "、".join(f"{k}×{v}" for k, v in sorted(slot["rules"].items()))
        console.print(f"[dim]{provider}: {slot['count']} 条（{rules}）[/dim]")
    if with_reason:
        console.print("[dim]命中的规则与理由：[/dim]")
        seen: dict[str, str] = {}
        for d in decisions.values():
            seen.setdefault(d.rule_id, d.reason)
        for rule_id, reason in seen.items():
            console.print(f"[dim]  {rule_id} → {reason}[/dim]")


def _factory_reject_reason(after_result: dict[str, Any], threshold: float) -> str:
    """出厂复检的拒收理由——把三种"不合格"分开说，别混成一句。

    * 没检成（人脸模型不可用 / 参考照没人脸）→ 身份一致性**没被校验过**，
      与 OCR 闸「文字闸未执行，不能记为合格」同口径；
    * 检了、换脸后画面里没有人脸 → **换脸没生效**，交付物里没有那个人；
    * 检了、分数低于阈值 → 身份仍不达标。
    """
    if after_result.get("skipped"):
        count = after_result.get("count")
        if count is None:
            return "换脸后复检：人脸模型不可用，人脸闸未执行，不能记为合格"
        if count == 0:
            return "换脸后复检：换脸后画面未检出人脸（换脸未生效），不放行"
        return "换脸后复检：参考照检不到人脸，身份一致性无从校验（不放行）"
    score = after_result.get("similarity")
    return f"换脸后身份一致性 {score} 仍低于阈值 {threshold}"


def _est_cost_for(outcome: GenOutcome, provider_name: str) -> float | None:
    """单张成本口径。

    HTTP API 后端：**按实际选定模型档位的实测单价**（models_spec 价目表），
    不再用 AP_COST_API 的一刀切口径——不同档位差 5 倍，混着算成本账是假账。
    其余后端沿用 config.unit_cost_for 旧口径；未配置单价照旧返回 None（不编造）。
    """
    if outcome.model_id:
        return get_model_spec(outcome.model_id).unit_cost_cny
    return config.unit_cost_for(provider_name, outcome.seconds)


@app.command()
def plan(
    city: str = typer.Option(None, help="只看某个城市"),
    limit: int = typer.Option(None, help="只预览前 N 条"),
    no_cloud: bool = typer.Option(False, help="按不出网口径预览路由（全部走本地栈）"),
) -> None:
    """展开任务矩阵：城市 × 项目 × 主题 × 渠道 × 角色，并预览路由到哪个后端。"""
    router = _router()
    specs = filter_specs(_specs(), city=city, limit=limit)

    table = Table(title=f"任务矩阵展开（{len(specs)} 条）")
    for col in ("项目", "主题", "角色", "路径", "后端", "路由规则"):
        table.add_column(col, overflow="ellipsis", no_wrap=True)
    decisions: dict[str, RouteDecision] = {}
    for spec in specs:
        path, _ = router.decide_path(spec)
        # 预览用"不检查可用性"的决策：后端没装也要能看见"本该走哪条线、命中哪条规则"，
        # 可用性单独提示（否则两个后端都缺时整个命令直接崩掉）。
        decision = routing.decide_provider(spec, no_cloud=no_cloud, require_available=False)
        decisions[spec.idempotency_key] = decision
        table.add_row(
            spec.project, spec.theme, spec.character or "-", path,
            decision.provider, decision.rule_id,
        )
    console.print(table)
    for provider, slot in sorted(routing.summarize(list(decisions.values())).items()):
        rules = "、".join(f"{k}×{v}" for k, v in sorted(slot["rules"].items()))
        console.print(f"[dim]{provider}: {slot['count']} 条（{rules}）[/dim]")
    for provider in sorted({d.provider for d in decisions.values()}):
        ok, why = routing.backend_available(provider)
        if not ok:
            console.print(f"[yellow]! {provider} 后端当前不可用：{why}[/yellow]")

    with console.status("编译提示词并校验不变量……"):
        for spec in specs:
            router.compile(spec)
    console.print("[green]✓ 全部提示词通过「原始中文标签不入画面 prompt」不变量校验[/green]")


@app.command()
def generate(
    provider: str = typer.Option("matrix", help="生图后端：matrix | replay | local | auto（按规则表自动分流）"),
    replay_provider: str = typer.Option(None, help="回放时复现哪条路线：matrix | http_api | local"),
    replay_model: str = typer.Option(None, help="回放时只取该模型档位的缓存（多档并存时指定）"),
    no_cloud: bool = typer.Option(False, help="本次运行不出网：全部交给本地栈（合规优先）"),
    city: str = typer.Option(None),
    project: str = typer.Option(None),
    theme: str = typer.Option(None),
    limit: int = typer.Option(None),
    retry: int = typer.Option(config.MAX_RETRY, help="失败项重试次数"),
    skip_existing: bool = typer.Option(True, help="跳过该后端已产出的任务（幂等）；用 --no-skip-existing 强制重出"),
    reuse_adopted: bool = typer.Option(
        False,
        "--reuse-adopted/--no-reuse-adopted",
        help="L1 复用：同素材（角色×项目×主题）已有合格产出时本地复制顶上，"
        "不再发后端、不花生成费（est_cost=0.0 落台账）；默认关，行为与旧一致",
    ),
    db: Path = typer.Option(None, help="台账路径"),
) -> None:
    """批量生成：编译提示词 → 路由定后端 → 调后端 → 下载 → 记台账。

    `--provider auto` 时按规则表**逐条任务**决定后端（见 routing.py），
    决策理由（rule_id + reason）会落台账，事后能逐条追溯"这张图为什么走这条线"。
    `--reuse-adopted` 打开 L1 复用层（见 library.py）：发布量≠生成量，
    同素材已有合格产出时本地复制顶上，默认关。
    """
    router = _router()
    specs = filter_specs(_specs(), city=city, project=project, theme=theme, limit=limit)
    if not specs:
        console.print("[yellow]没有匹配的任务[/yellow]")
        return

    # 1) 路由决策：auto 按规则表逐条定后端；手动模式统一后端
    # 手动模式下规则表被绕过，"合规优先"这层保护也没了：把不出网的运行/任务
    # 塞给云端后端＝静默把素材发出去。这种自相矛盾的输入在起跑前就拒，一张图都不发。
    if provider != "auto" and provider in (routing.API, routing.MATRIX):
        blocked = [spec for spec in specs if spec.no_cloud]
        if no_cloud or blocked:
            if no_cloud:
                who = "本次运行带了 --no-cloud（全局不出网）"
            else:
                themes = sorted({spec.theme for spec in blocked})
                who = (
                    f"有 {len(blocked)} 条任务标记了不出网"
                    f"（主题：{'、'.join(themes[:5])}{'…' if len(themes) > 5 else ''}）"
                )
            console.print(
                f"[red]后端已手动指定为 {provider}（云端），但{who}。"
                "要么去掉 --provider 让路由按规则表分流，要么改用 --provider local。"
                "不出网是合规约束，不接受被命令行静默覆盖。[/red]"
            )
            raise typer.Exit(code=1)

    decisions: dict[str, RouteDecision] = {}
    groups: dict[str, list[TaskSpec]] = {}
    for spec in specs:
        if provider == "auto":
            decision = routing.decide_provider(spec, no_cloud=no_cloud)
        else:
            decision = RouteDecision(
                provider=provider, rule_id="R_MANUAL", reason=f"命令行指定后端 {provider}"
            )
        decisions[spec.idempotency_key] = decision
        groups.setdefault(decision.provider, []).append(spec)

    if provider == "auto":
        _print_route_summary(decisions, specs, "自动路由计划（规则表：先命中先用）")

    prompts: dict[str, str] = {}
    model_decisions: dict[str, ModelDecision] = {}
    outcomes: list[GenOutcome] = []
    stats: dict[str, dict[str, float]] = {}

    def _model_for(spec: TaskSpec) -> str:
        """当前任务的模型档位（带缓存）。

        幂等跳过与真正生成**必须用同一个值**：跳过按模型精确匹配，
        如果两处各算一次（万一 choose_model 的输入变了），就会出现
        "跳过用 A 档、记账用 B 档"的错位。这里只算一次、两处共用。
        """
        md = model_decisions.get(spec.idempotency_key)
        if md is None:
            md = choose_model(spec)
            model_decisions[spec.idempotency_key] = md
        return md.model_id

    with ledger.connect(db) as conn:
        for backend_name, group_specs in groups.items():
            if backend_name == "replay":
                backend = build_provider(
                    backend_name, prefer_provider=replay_provider, model_id=replay_model
                )
            elif backend_name == routing.API:
                # 跨批预算熔断必须从台账播种：不传 conn 的话 provider 内部
                # 兜底 from_env() 是零基数守卫，历史花费对它不可见，
                # AP_BUDGET_CNY 每批都会重新归零——熔断形同虚设。
                backend = build_provider(
                    backend_name, budget_guard=BudgetGuard.from_env(conn)
                )
            else:
                backend = build_provider(backend_name)
            items: list[GenItem] = []
            for spec in group_specs:
                prompt, negative, warnings = router.compile(spec)
                for warning in warnings:
                    console.print(f"[yellow]! {spec.idempotency_key} {warning}[/yellow]")
                path, ref = router.decide_path(spec)
                spec.path = path
                input_urls: list[str] = []
                if path == "i2i" and ref and hasattr(backend, "upload"):
                    input_urls = [backend.upload(ref)]  # type: ignore[attr-defined]
                prompts[spec.idempotency_key] = prompt
                items.append(
                    GenItem(
                        task=spec,
                        prompt=prompt,
                        negative_prompt=negative,
                        aspect_ratio=spec.aspect_ratio,
                        resolution=spec.resolution,
                        input_urls=input_urls,
                    )
                )

            if skip_existing:
                # 幂等跳过：**按 (后端, 模型档位) 精确匹配**。
                # 只按后端的旧口径会把"换了模型"的任务当成已完成——同一批
                # 换档位重跑时一张不生成，却看起来像"模型跑了"，这是模型
                # 迭代场景里最贵的静默错误。非 API 后端没有档位概念，维持原口径。
                done_all = ledger.successful_keys(conn, provider=backend.name)
                done_by_model: dict[str, set[str]] = {}
                keep: list[GenItem] = []
                regenerated = 0
                regenerated_cost = 0.0
                for item in items:
                    key = item.task.idempotency_key
                    if key not in done_all:
                        keep.append(item)
                        continue
                    if backend_name != routing.API:
                        continue
                    model_id = _model_for(item.task)
                    if model_id not in done_by_model:
                        done_by_model[model_id] = ledger.successful_keys(
                            conn, provider=backend.name, model_id=model_id
                        )
                    if key in done_by_model[model_id]:
                        continue
                    # 该键在本后端产出过，但**不是当前档位**（换档 / 旧行没记模型）
                    regenerated += 1
                    regenerated_cost += get_model_spec(model_id).unit_cost_cny
                    keep.append(item)
                skipped = len(items) - len(keep)
                if skipped:
                    console.print(f"[dim]{backend.name}：幂等跳过 {skipped} 条已产出任务[/dim]")
                if regenerated:
                    console.print(
                        f"[yellow]{backend.name}：{regenerated} 条任务在当前模型下没有产出"
                        "（历史行属于其它档位，或升级前没记 model_id）→ 将重新生成。"
                        f"预计额外花费约 ¥{regenerated_cost:.2f}；"
                        "这是『换模型不再静默复用旧图』的预期行为。"
                        "[dim]建议先用 --limit 小批试跑；不想重出请先确认档位或清理对应台账行。[/dim]"
                    )
                items = keep

            # L1 复用（默认关）：在发给后端之前拦一道——同素材已有合格产出的，
            # 本地复制成"本后端产物"落台账（est_cost=0.0），不再花生成的钱。
            # 放在 skip_existing 之后：同后端同键的幂等跳过口径不变，
            # 复用只处理"这个后端还没产出过"的那部分。
            if reuse_adopted and items:
                remaining: list[GenItem] = []
                for item in items:
                    reused = library.reuse_asset(
                        conn,
                        spec=item.task,
                        backend_name=backend.name,
                        prompt=item.prompt,
                        decision=decisions[item.task.idempotency_key],
                        out_dir=config.OUT_DIR,
                    )
                    if reused is None:
                        remaining.append(item)
                if len(remaining) != len(items):
                    console.print(
                        f"[dim]{backend.name}：复用 {len(items) - len(remaining)} 条"
                        "已落盘素材（--reuse-adopted，零生成成本）[/dim]"
                    )
                items = remaining

            if not items:
                continue

            # 模型档位与后端路由正交：只有 http_api 需要逐条定档（model_choice），
            # matrix/local 不涉及——所以非 API 组传 None，走各 provider 原契约。
            # 走 _model_for 复用跳过阶段算过的档位，保证"跳过判定"与"记账档位"一致。
            model_ids: list[str | None] | None = None
            if backend_name == routing.API:
                model_ids = [_model_for(item.task) for item in items]

            started = time.perf_counter()
            group_outcomes = (
                backend.generate(items, model_ids=model_ids)
                if model_ids is not None
                else backend.generate(items)
            )
            attempt = 1
            while attempt <= retry:
                failed_idx = [i for i, o in enumerate(group_outcomes) if not o.ok]
                if not failed_idx:
                    break
                console.print(f"[yellow]第 {attempt} 次重试 {len(failed_idx)} 个失败项[/yellow]")
                retry_items = [items[i] for i in failed_idx]
                retry_outcomes = (
                    backend.generate(retry_items, model_ids=[model_ids[i] for i in failed_idx])
                    if model_ids is not None
                    else backend.generate(retry_items)
                )
                # 只把成功的替换回原位，保持 outcomes 与 items 顺序长度一致
                for idx, outcome in zip(failed_idx, retry_outcomes):
                    if outcome.ok:
                        group_outcomes[idx] = outcome
                attempt += 1
            elapsed = time.perf_counter() - started
            stats[backend_name] = {
                "requested": float(len(group_outcomes)),
                "success": float(sum(1 for o in group_outcomes if o.ok)),
                "elapsed": elapsed,
            }
            if model_ids is not None:
                # HTTP API 跑批按**实际档位的价目表**计成本；失败的不上游计费，不计。
                stats[backend_name]["cost"] = round(
                    sum(
                        get_model_spec(o.model_id).unit_cost_cny
                        for o in group_outcomes
                        if o.ok and o.model_id
                    ),
                    4,
                )
                dist: dict[str, int] = {}
                for mid in model_ids:
                    dist[str(mid)] = dist.get(str(mid), 0) + 1
                console.print(
                    f"[dim]{backend_name}：模型档位分布 {dist}"
                    "（理由见台账 model_rule_id/model_reason）[/dim]"
                )
            outcomes.extend(group_outcomes)

        for outcome in outcomes:
            spec = outcome.task
            decision = decisions.get(spec.idempotency_key)
            md = model_decisions.get(spec.idempotency_key)
            provider_name = outcome.source_provider or (decision.provider if decision else "unknown")
            ledger.record_asset(
                conn,
                {
                    "idempotency_key": spec.idempotency_key,
                    "city": spec.city, "project": spec.project, "theme": spec.theme,
                    "channel": spec.channel, "character": spec.character,
                    "route": spec.path,
                    # 归属"真正生成"的后端：回放时沿用原后端，不重复计数
                    "provider": provider_name,
                    "node_id": outcome.node_id, "file_name": outcome.file_name,
                    "local_path": outcome.local_path,
                    "prompt": prompts.get(spec.idempotency_key, ""),
                    "gen_seconds": round(outcome.seconds, 3),
                    # 失败行记 None 而非 0.0：None=上游没花钱、不进熔断基数；
                    # 0.0=花过钱但免费（L1 复用）——两个账目语义不能混。
                    "est_cost": _est_cost_for(outcome, provider_name) if outcome.ok else None,
                    "qc_verdict": "pending",
                    "route_rule_id": decision.rule_id if decision else None,
                    "route_reason": decision.reason if decision else None,
                    # 模型档位维度：outcome 带的是**实际用**的档（失败也带），
                    # 决策理由带的是选档规则——两个维度都能从台账逐条追溯。
                    "model_id": outcome.model_id or (md.model_id if md else None),
                    "model_rule_id": md.rule_id if md else None,
                    "model_reason": md.reason if md else None,
                },
            )

        for backend_name, st in stats.items():
            if "cost" in st:
                # http_api：按各条实际档位价目表求和（与 asset 行 est_cost 同口径）
                run_cost = st["cost"]
            else:
                unit = config.unit_cost_for(backend_name, 0.0)
                run_cost = (unit or 0) * st["success"] if unit else None
            ledger.record_run(
                conn,
                {
                    "started_at": ledger.now(), "finished_at": ledger.now(),
                    "provider": backend_name,
                    "requested": int(st["requested"]), "success": int(st["success"]),
                    "failed": int(st["requested"] - st["success"]), "rejected": 0,
                    "est_cost": run_cost,
                    "notes": f"elapsed={st['elapsed']:.1f}s",
                },
            )

    # 耗时语义按后端说清楚：API 批次接口只给整批耗时，单张数字是**批内均摊**；
    # 本地栈逐张计时，台账里是真实单张耗时（首张还含管线加载）。
    seconds_note = {
        "matrix": "（批次接口只给整批耗时，单张数字是**批内均摊**，非单张实测）",
        "local": "（本地栈逐张计时，台账里是真实单张耗时，首张含管线加载）",
        "replay": "（回放不产生新图，秒数只是读缓存的耗时，不代表生成成本）",
    }
    if not stats:
        console.print("[dim]没有需要生成的任务（全部已产出过）[/dim]")
        return
    table = Table(title="生成结果（按后端）")
    for col in ("后端", "请求", "成功", "墙钟", "摊/张", "耗时口径"):
        table.add_column(col, overflow="fold")
    for backend_name, st in stats.items():
        n = max(st["requested"], 1)
        table.add_row(
            backend_name, str(int(st["requested"])), str(int(st["success"])),
            f"{st['elapsed']:.1f}s", f"{st['elapsed'] / n:.1f}s",
            seconds_note.get(backend_name, ""),
        )
    console.print(table)


@app.command()
def qc(
    db: Path = typer.Option(None),
    use_face: bool = typer.Option(True, help="启用人脸一致性闸门"),
    face_threshold: float = typer.Option(None, help="覆盖人脸阈值"),
    reference: Path = typer.Option(None, help="定妆照路径（默认取矩阵里的角色参考图）"),
    reset: bool = typer.Option(False, help="重置已有判定，全部重新质检"),
) -> None:
    """质检：硬规则（必拦）+ 人脸一致性（数值闸）。VLM 为软信号，另开 --vlm。"""
    router = _router()
    gate = FaceGate(face_threshold)
    ref = reference or (config.EVIDENCE_DIR / "01_t2i_baseline.png")
    profiles = character_profiles()

    with ledger.connect(db) as conn:
        if reset:
            conn.execute("UPDATE assets SET qc_verdict = 'pending' WHERE qc_verdict IS NOT NULL")
        rows = ledger.assets(conn, "qc_verdict IS NULL OR qc_verdict = 'pending'")
        if not rows:
            console.print("[yellow]没有待质检的产出[/yellow]")
            return

        known_hashes: dict[str, int] = {}
        from .qc.rules import dhash

        for row in ledger.assets(conn, "qc_verdict = 'accepted'"):
            path = ledger.asset_path(row)
            if path:
                known_hashes[row["idempotency_key"]] = dhash(path)

        table = Table(title=f"质检结果（{len(rows)} 条）")
        for col in ("后端", "城市", "项目", "主题", "路径", "判定", "理由", "人脸相似度"):
            table.add_column(col, overflow="fold")

        for row in rows:
            reasons: list[str] = []
            details: dict[str, Any] = {}
            # 台账存的是绝对路径；仓库搬过家就重新锚定（找不到就当"没有本地文件"）
            local_path = ledger.asset_path(row)
            if local_path:
                findings = rule_findings(
                    local_path,
                    expected_aspect=None,
                    known_hashes=known_hashes,
                    key=row["idempotency_key"],
                )
                reasons.extend(findings["reasons"])
                details = findings["details"]
            else:
                reasons.append("没有本地文件")

            face_score = None
            face_detail: dict[str, Any] = {}
            identity_drift = False
            if use_face and local_path:
                if row.get("route") == "i2i" and row.get("character"):
                    # 绑了角色的任务：与该角色的定妆照比身份
                    profile = profiles.get(row["character"]) or {}
                    ref_candidate = config.REPO_ROOT / (profile.get("ref") or "")
                    # is_file() 不是 exists()：profile 缺 ref 时 `REPO_ROOT / ""`
                    # 就是仓库根目录，exists() 为真 → 拿目录当参考图 → Image.open
                    # 抛 PermissionError，整条 qc 命令中断、事务回滚（判定全丢）。
                    reference_for_task = ref_candidate if ref_candidate.is_file() else ref
                    face = gate.check(local_path, reference_for_task)
                    face_score = face.get("similarity")
                    face_detail = dict(face)
                    if should_reject_on_identity(face, "generate"):
                        reasons.append(face["reason"])
                    elif not face.get("ok") and not face.get("skipped"):
                        # **生成阶段只标记、不拦截**。
                        # 阈值区间本身不具判别力（拿它拦中间产物会误杀），
                        # 出厂判定放在换脸之后复检（见 faceswap 段）。
                        identity_drift = True
                        face_detail["identity_drift"] = True
                        face_detail["drift_reason"] = face["reason"]
                else:
                    # 文生图任务：没有指定人物，只记录检出人脸数，不做身份比对
                    face_detail = gate.detect(local_path)

                # **人物任务必须检出人脸**：画面里没有人脸 = 身份一致性无从校验，
                # 不能因为"检不到就跳过"而放行（那是"没法判断"，不是"合格"）。
                missing = missing_face_reason(
                    bool(row.get("character")), face_detail.get("count")
                )
                if missing:
                    reasons.append(missing)
                # **人脸闸没跑 ≠ 合格**（与 OCR 闸同口径）：人物任务的身份一致性
                # 没被校验过就不能记为合格——"没法判断"不是"身份没问题"。
                unverified = face_gate_unavailable_reason(
                    bool(row.get("character")), face_detail
                )
                if unverified:
                    reasons.append(unverified)

            verdict = "rejected" if reasons else "accepted"
            if face_detail:
                details["face"] = face_detail
            ledger.record_asset(
                conn,
                {
                    "idempotency_key": row["idempotency_key"],
                    # 必须带上 provider 与 model_id：同一任务可能有多条产出
                    # （不同后端 / 同一后端不同档位），不指定档位就会改错行。
                    "provider": row.get("provider"),
                    "model_id": row.get("model_id"),
                    "qc_verdict": verdict,
                    "qc_reasons": reasons,
                    "qc_rules": details,
                    "face_similarity": face_score,
                    "text_found": details.get("text_found", []),
                    # 拒收的行不再指向换脸产物：源图可能已经重跑过，
                    # 那张换脸图是**上一轮的旧产物**，留着会让人看到一张"不存在"的成品。
                    **(
                        {"swapped_path": None, "face_similarity_after": None}
                        if verdict == "rejected"
                        else {}
                    ),
                },
            )
            face_cell = str(face_score) if face_score is not None else (
                f"{face_detail.get('count')} 张脸" if face_detail.get("count") is not None else "-"
            )
            if identity_drift:
                face_cell += " ⚠漂移"
            table.add_row(
                row.get("provider", "-"), row["city"], row["project"], row["theme"],
                row.get("route", "-"),
                "[red]拦截[/red]" if verdict == "rejected" else "[green]放行[/green]",
                "；".join(reasons)[:110] or "-",
                face_cell,
            )

    console.print(table)


@app.command("qc-file")
def qc_file(path: Path = typer.Argument(..., help="任意图片路径")) -> None:
    """对单张图跑硬规则（不经过台账）。

    用途：演示时现场拿一张翻车样本（如 `evidence/batch/` 里的黑板字图）
    当场证明 OCR 硬规则**真的会拦**，而不是只报"这条规则存在"。
    """
    findings = rule_findings(path)
    if findings["ok"]:
        console.print(f"[green]放行[/green]：{path} 未触发硬规则")
        return
    console.print(f"[red]拦截[/red]：{path}")
    for reason in findings["reasons"]:
        console.print(f"  - {reason}")
    raise typer.Exit(code=1)


@app.command()
def adopt(
    key: str = typer.Argument(..., help="产出任务的 idempotency_key"),
    decision: str = typer.Option("adopted", help="采纳决定：adopted | rejected"),
    note: str = typer.Option(None, help="备注；rejected 时写明不用原因"),
    provider: str = typer.Option(None, help="只改某个后端的产出；缺省改该任务全部行"),
    model: str = typer.Option(None, help="只改某个模型档位的产出（同键多档时必须指定）"),
    db: Path = typer.Option(None),
) -> None:
    """抽检台人工采纳：质检放行后，**人**在抽检台上点"用/不用"。

    这是系统内部的一次真实点击，不是事后统计：采纳只写 adopted/adopted_at/
    adopt_note 三列（走 record_asset 的 ON CONFLICT DO UPDATE 局部语义），
    不碰 qc_verdict / local_path —— 机器放行 ≠ 人决定要用，两回事两列存证。

    同一后端同一任务可能有多个模型档位的产出（variant）：此时**必须**用
    `--model` 指定采纳哪一张，否则会报错——"一起采纳"等于把没看过的图也标成用了。
    """
    if decision not in ledger.ADOPT_DECISIONS:
        console.print(f"[red]未知决定 {decision!r}，只接受 adopted / rejected[/red]")
        raise typer.Exit(code=2)
    try:
        with ledger.connect(db) as conn:
            updated = ledger.record_adoption(
                conn, key, decision=decision, note=note, provider=provider, model_id=model
            )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2)
    if not updated:
        console.print(f"[red]台账里没有这条产出：{key}（采纳只能点在真实存在的产出上）[/red]")
        raise typer.Exit(code=1)
    label = "采纳" if decision == "adopted" else "弃用"
    console.print(
        f"[green]✓ 已{label}（{updated} 行）[/green] [dim]{key}"
        f"{' @' + provider if provider else ''}{' [' + model + ']' if model else ''}[/dim]"
        + (f"\n[dim]备注：{note}[/dim]" if note else "")
    )


@app.command()
def adoption(db: Path = typer.Option(None)) -> None:
    """采纳率：adopted / generated，按后端分组。分母只算真正落过盘的产出。"""
    with ledger.connect(db) as conn:
        stats = ledger.adoption_summary(conn)
    rate = f"{stats['adoption_rate'] * 100:.1f}%" if stats["adoption_rate"] is not None else "—"
    untouched = stats["generated"] - stats["adopted"] - stats["rejected"]
    console.print(
        f"采纳率 adopted/generated = {stats['adopted']}/{stats['generated']} = {rate}"
        f"（弃用 {stats['rejected']}，未处理 {untouched}）"
    )
    if stats["by_provider"]:
        table = Table(title="按后端")
        for col in ("后端", "产出", "采纳", "弃用", "采纳率"):
            table.add_column(col)
        for provider, s in stats["by_provider"].items():
            r = f"{s['adoption_rate'] * 100:.1f}%" if s["adoption_rate"] is not None else "—"
            table.add_row(provider, str(s["generated"]), str(s["adopted"]), str(s["rejected"]), r)
        console.print(table)


@app.command("library")
def library_cmd(db: Path = typer.Option(None)) -> None:
    """复用素材盘点（L1）：按 角色 × 项目 × 主题 汇总台账里的可复用资产。

    只读台账、不建新表：口径与 library.find_reusable 一致——
    落过盘（local_path 非空）才算资产，rejected（机器拦/人弃）不算可复用。
    看"哪组素材覆盖广、采纳多"，就是下一次 generate --reuse-adopted 能省多少。
    """
    with ledger.connect(db) as conn:
        rows = conn.execute(
            "SELECT COALESCE(character, '') AS character, project, theme,"
            " COUNT(*) AS reusable,"
            " SUM(CASE WHEN qc_verdict = 'accepted' THEN 1 ELSE 0 END) AS accepted,"
            " SUM(CASE WHEN adopted = 'adopted' THEN 1 ELSE 0 END) AS adopted"
            " FROM assets"
            " WHERE local_path IS NOT NULL"
            " AND COALESCE(qc_verdict, '') != 'rejected'"
            " AND COALESCE(adopted, '') != 'rejected'"
            " GROUP BY COALESCE(character, ''), project, theme"
            " ORDER BY adopted DESC, reusable DESC, character, project, theme"
        ).fetchall()

    if not rows:
        console.print("[yellow]台账里没有可复用的落盘产出[/yellow]")
        return
    table = Table(title=f"可复用素材（{len(rows)} 组，读台账不建新表）")
    for col in ("角色", "项目", "主题", "落盘", "质检放行", "人工采纳"):
        table.add_column(col, overflow="fold")
    total_reusable = total_adopted = 0
    for r in rows:
        total_reusable += r["reusable"]
        total_adopted += r["adopted"]
        table.add_row(
            r["character"] or "（泛化）", r["project"], r["theme"],
            str(r["reusable"]), str(r["accepted"]), str(r["adopted"]),
        )
    console.print(table)
    console.print(
        f"[dim]合计 {total_reusable} 份可复用素材（其中已采纳 {total_adopted}）；"
        "generate --reuse-adopted 时同素材任务将优先本地复制、不再花生成费。[/dim]"
    )


@app.command("faceswap")
def faceswap_cmd(db: Path = typer.Option(None)) -> None:
    """把定妆照的脸确定性贴回成品，并**在换脸后复检出厂判定**。

    这是人物一致性的**工业解法**：实测证明"靠生成模型记住脸"不可靠
    （同一人 0.2836~0.4284 / 不同人 0.3399~0.4147，分布重叠），
    所以身份最终由后处理兜底。

    因为生成阶段的人脸分只标记不拦截（阈值不具判别力），
    **出厂判定放在这里**：换脸后复检，仍低于阈值才判拒收。
    """
    from .postprocess.faceswap import FaceSwapper, swap_available, swapped_output_path
    from .qc.face import similarity

    available, message = swap_available()
    if not available:
        console.print(f"[red]{message}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[dim]{message}[/dim]")

    profiles = character_profiles()
    gate = FaceGate()
    with ledger.connect(db) as conn:
        # **只处理已质检放行的图生图产出**：没质检过的图连 OCR 文字闸都没跑过，
        # 换完脸直接标合格＝拿没验过的产物出厂。
        unchecked = ledger.assets(
            conn, "route = 'i2i' AND (qc_verdict IS NULL OR qc_verdict = 'pending')"
        )
        if unchecked:
            console.print(
                f"[red]有 {len(unchecked)} 条图生图产出还没质检，先跑 `aesthetics qc` "
                "再换脸（不部分处理）。[/red]"
            )
            raise typer.Exit(code=1)
        # 只处理图生图路线；被硬规则（画面内文字等）拒收的图不必换脸
        rows = ledger.assets(conn, "route = 'i2i' AND qc_verdict = 'accepted'")
        table = Table(title="人脸替换与出厂复检（换脸前 → 换脸后 → 判定）")
        for col in ("后端", "城市", "项目", "换脸前", "换脸后", "判定", "产出"):
            table.add_column(col, overflow="fold")

        swapper = None
        for row in rows:
            if not row.get("character"):
                continue
            local_path = ledger.asset_path(row)
            if not local_path:
                continue
            profile = profiles.get(row["character"]) or {}
            reference = config.REPO_ROOT / (profile.get("ref") or "")
            if not reference.exists():
                continue
            if swapper is None:
                with console.status("加载换脸模型……"):
                    swapper = FaceSwapper()
            # provider 必须进路径：matrix 与 local 两条线共用同一幂等键，
            # 不带 provider 会互相覆盖（DB 复合主键不冲突，磁盘会冲突）
            out_path = swapped_output_path(
                str(row.get("provider") or "unknown"),
                row["idempotency_key"],
                config.OUT_DIR,
                row.get("model_id"),
            )
            result = swapper.swap(local_path, reference, out_path)
            if not result.get("ok"):
                # 换脸失败 = 身份一致性没被施加。这条行**绝不能留在 accepted**：
                # 合规导出/发布只取 accepted 的行，于是没换脸的原图会被当成品发出去。
                ledger.record_asset(
                    conn,
                    {
                        "idempotency_key": row["idempotency_key"],
                        "provider": row.get("provider"),
                        "model_id": row.get("model_id"),
                        "swapped_path": None,
                        "face_similarity_after": None,
                        "qc_verdict": "rejected",
                        "qc_reasons": [
                            f"换脸失败：{result['reason']}"
                            "（修好后 `qc --reset` 重跑质检再 faceswap）"
                        ],
                    },
                )
                table.add_row(
                    row.get("provider", "-"), row["city"], row["project"], "-", "-",
                    "[red]拦截[/red]", result["reason"],
                )
                continue

            # 换脸前的分数**现算**：台账里那个可能是上一轮、旧选脸规则下写的
            before = similarity(local_path, reference)
            # 出厂判定跑在**最终产物**上：人脸闸 + 硬规则（OCR 文字闸等）。
            # 交付出去的是这张换脸图，不是源图——闸门必须跑在它身上。
            after_result = gate.check(out_path, reference)
            after = after_result.get("similarity")
            threshold = gate.threshold
            findings = rule_findings(out_path)
            reasons: list[str] = list(findings["reasons"])
            if should_reject_on_identity(after_result, "post_swap"):
                reasons.append(_factory_reject_reason(after_result, threshold))
            accepted = not reasons
            verdict = "accepted" if accepted else "rejected"

            ledger.record_asset(
                conn,
                {
                    "idempotency_key": row["idempotency_key"],
                    "provider": row.get("provider"),
                    "model_id": row.get("model_id"),
                    # 拒收的行不再指向换脸产物：留着会让人看到一张"不存在"的成品
                    "swapped_path": None if not accepted else str(out_path),
                    "face_similarity": before,
                    "face_similarity_after": after,
                    "qc_verdict": verdict,
                    "qc_reasons": reasons,
                },
            )
            table.add_row(
                row.get("provider", "-"), row["city"], row["project"],
                str(before), str(after),
                "[green]放行[/green]" if accepted else f"[red]拦截[/red] {reasons}",
                str(out_path),
            )
    console.print(table)


@app.command("reset-db")
def reset_db(db: Path = typer.Option(None), confirm: bool = typer.Option(False, help="确认清空")) -> None:
    """清空台账（不改素材、不动缓存）。

    为什么要这个命令：改质检规则或换后端后要能从头重跑一遍；
    用 SQL 清表而不是删库文件，避免把缓存和已生成素材一起搞没。
    """
    if not confirm:
        console.print("[yellow]加 --confirm 才会执行清空[/yellow]")
        return
    with ledger.connect(db) as conn:
        conn.execute("DELETE FROM assets")
        conn.execute("DELETE FROM runs")
        conn.execute("DELETE FROM copies")
    console.print("[green]台账已清空（素材与回放缓存保留）[/green]")


@app.command("copy")
def copy_cmd(db: Path = typer.Option(None)) -> None:
    """生成文案（规则模板）并过禁用词闸。"""
    router = _router()
    with ledger.connect(db) as conn:
        rows = ledger.assets(conn, "qc_verdict = 'accepted'")
        # 同一任务被两个后端各产出一份时，文案按幂等键去重：
        # 文案只由城市/项目/主题决定，与后端无关，台账 copies 表主键也是幂等键。
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for row in rows:
            if row["idempotency_key"] in seen:
                continue
            seen.add(row["idempotency_key"])
            unique.append(row)
        hits = 0
        for row in unique:
            draft = copy_mod.make_copy(
                city=row["city"],
                project=row["project"],
                theme=row["theme"],
                copy_angle=router.copy_angle(row["theme"]),
                channel=row["channel"],
                character=row.get("character"),
            )
            ledger.record_copy(conn, copy_mod.to_row(row["idempotency_key"], draft))
            hits += 1 if draft.banned_hits else 0
        console.print(
            f"已生成 {len(unique)} 条文案（覆盖 {len(rows)} 份产出，同任务多后端只存一份），"
            f"禁用词命中 {hits} 条"
        )


@app.command()
def report(db: Path = typer.Option(None)) -> None:
    """出拼图 + CSV + JSON 报表 + 成本对比。"""
    with ledger.connect(db) as conn:
        rows = ledger.assets(conn)
        summary = ledger.summary(conn)
        costs = report_mod.cost_summary(rows)
        summary["cost_by_provider"] = costs
        report_mod.export_csv(rows, report_mod.report_dir() / "assets.csv")
        report_mod.export_json(summary, report_mod.report_dir() / "summary.json")

        # 合规审计随报表一起出：发布前必须知道"哪些放行的图其实不能发"
        comp = compliance_mod.audit(conn, pack=try_load_pack())
        report_mod.export_json(comp, report_mod.report_dir() / "compliance.json")

        verdict_cn = {"accepted": "放行", "rejected": "拦截", "pending": "待检"}
        # 拼图展示**最终产物**：换过脸的用 swapped_path，
        # 否则拼图上还是换脸前那张脸，却标着"放行"，又变成自相矛盾。
        # 路径走 ledger.asset_path：台账存的是绝对路径，仓库搬过家要能重新锚定。
        items: list[tuple[str, Path]] = []
        for r in rows:
            path = ledger.asset_path(r, "swapped_path") or ledger.asset_path(r)
            if not path:
                continue
            items.append(
                (
                    f"{r.get('provider')} {r['city']}{r['project'][:4]} {verdict_cn.get(r['qc_verdict'], '-')}",
                    path,
                )
            )
        sheet = report_mod.collage(
            items,
            report_mod.report_dir() / "contact_sheet.png",
            title="医美图文 MVP 产出（后端 城市项目 判定）",
        )

    console.print(json.dumps(summary, ensure_ascii=False, indent=2))

    if comp["ok"]:
        console.print(
            f"[green]合规校验通过[/green]（检查 {comp['checked']} 条放行素材，域包 {comp['pack'] or '未启用'}）"
        )
    else:
        console.print(
            f"[yellow]合规校验未通过[/yellow]：{comp['by_code']} —— "
            "明细见 report/compliance.json，或跑 `aesthetics compliance`"
        )

    # 成本对比：单张**合格**图综合成本 = 总花费 ÷ 合格张数（返工也算钱）
    cost_table = Table(title="后端成本对比（单张合格图综合成本 = 总花费 ÷ 合格张数）")
    for col in ("后端", "调用", "合格", "拒收", "通过率", "总耗时", "单张耗时", "总成本", "单张合格图综合成本"):
        cost_table.add_column(col, overflow="fold")
    for c in costs:
        money = f"¥{c['total_cost']:.4f}" if c["priced"] and c["total_cost"] is not None else "未配置单价"
        per = (
            f"¥{c['cost_per_accepted']:.4f}"
            if c["priced"] and c["cost_per_accepted"] is not None
            else "—"
        )
        rate = f"{c['accept_rate'] * 100:.1f}%" if c["accept_rate"] is not None else "-"
        cost_table.add_row(
            c["provider"], str(c["calls"]), str(c["accepted"]), str(c["rejected"]),
            rate, f"{c['total_seconds']}s", f"{c['avg_seconds']}s", money, per,
        )
    console.print(cost_table)
    if not any(c["priced"] for c in costs):
        console.print(
            "[dim]未配置单价：台账只记调用次数与耗时，不编造金额。"
            "设置 AP_COST_API（API 元/张）与 AP_LOCAL_WATT + AP_KWH_PRICE + AP_LOCAL_DEPR_PER_HOUR"
            "（本地栈电费+折旧）后本表会给出金额。[/dim]"
        )

    # 采纳率单独成行：这是"质检放行后人在抽检台上点了用"的比例，
    # 与上面的通过率（机器判定）分开口径，报表里不能混成一个数。
    ad = report_mod.adoption_stats(rows)
    ad_rate = f"{ad['adoption_rate'] * 100:.1f}%" if ad["adoption_rate"] is not None else "—"
    console.print(
        f"[bold]采纳率（人工采纳/实际产出）[/bold] adopted/generated = "
        f"{ad['adopted']}/{ad['generated']} = {ad_rate}（弃用 {ad['rejected']}）"
    )
    for provider_name, s in sorted(ad["by_provider"].items()):
        r = f"{s['adoption_rate'] * 100:.1f}%" if s["adoption_rate"] is not None else "—"
        console.print(
            f"[dim]  {provider_name}: {s['adopted']}/{s['generated']} = {r}[/dim]"
        )

    if sheet:
        console.print(f"[green]拼图[/green]: {sheet}")


@app.command("calibrate-face")
def calibrate_face() -> None:
    """看身份分的分布（**这不是标定**）。

    三组分开看，才是这件事的真相：
      A 同一人 · 真人照片 —— 人工 I2I，不是模型生成的
      B 不同人 · 真人照片 —— 同样描述换了个人生成
      C 同一人 · **生成图** —— 流水线自己产的

    A 与 B 可分（说明模型**分得出人**）；C 与 B 重叠（说明**生成方差**才是问题）。
    这才是"必须后置换脸兜底"的真正理由 —— 不是"阈值根本没用"。

    样本量 n=4 且全是本项目自有素材：只能当**自洽性验证**，
    不能当有效性证明（要谈判别力得用 ≥10 组外部标注样本出 ROC/FPR，本项目未做）。
    """
    from .qc.face import similarity

    base = config.EVIDENCE_DIR
    ref = base / "01_t2i_baseline.png"
    groups = (
        ("A 同一人 · 真人照片", [(base / "02_i2i_same_person.jpg",)]),
        (
            "B 不同人 · 真人照片",
            [
                (base / "03_t2i_same_description_differs.png",),
                (base / "batch" / "batch_01_wuhan_shuiguang.png",),
                (base / "batch" / "batch_03_chengdu_photon.jpg",),
            ],
        ),
        (
            "C 同一人 · 生成图",
            [
                (base / "10_api_i2i_generated.png",),
                (base / "12_local_i2i_generated.png",),
            ],
        ),
    )

    table = Table(title="身份相似度分布（对照定妆照 · 选脸=与参考照最相似的一对）")
    table.add_column("组", overflow="fold")
    table.add_column("样本", overflow="fold")
    table.add_column("分数")
    for label, paths in groups:
        for (path,) in paths:
            if not path.exists():
                table.add_row(label, path.name, "（缺文件）")
                continue
            score = similarity(path, ref)
            table.add_row(label, path.name, "-" if score is None else f"{score}")
    console.print(table)
    console.print(
        "[dim]A 与 B 可分 = 模型分得出人；C 掉进 B 的区间 = 生成方差。"
        "生产阈值已标定 T=0.57（45 域内正 vs 7 同域难负取间隙中点，"
        "tools/recalibrate_face_threshold.py）；本视图是分布自检，"
        "n=4 且全为自有素材，不替代标定样本集。[/dim]"
    )


@app.command()
def compare() -> None:
    """后端可用性预检：matrix 与 local 是否都就绪（对比前置）。

    真正的对比不是这个命令做的：用 `generate --provider matrix` 与
    `generate --provider local` 各跑同一批，再 `report` 看成本/一致性对比。
    这里只回答"现在能不能开始对比"。

    注意：本命令**不带任何任务参数**。曾有一版保留了 city/project/theme 的默认值，
    其中 theme 的默认值取自业务话术——它既不被代码使用，又会被 `--help` 展示出来，
    而那句话属于合规词表里的诱导性表述。结论：**默认值不要用业务话术**。
    """
    for name in ("matrix", "local"):
        try:
            build_provider(name)
        except Exception as exc:
            console.print(f"[yellow]{name} 后端不可用：{exc}[/yellow]")
            return
    console.print("[green]两个后端都可用。用 generate --provider matrix / --provider local 分别跑同一批任务后，"
                  "再执行 report 对比。[/green]")


@app.command()
def panel(
    host: str = typer.Option("127.0.0.1", help="监听地址（默认只听本机，演示不做公网暴露）"),
    port: int = typer.Option(8000),
) -> None:
    """起轻量 Web 面板：任务提交 + 画廊 + 每张的质检明细/路由理由/成本。

    面板是 CLI 的一层皮：**读**走台账（进程内 SQLite），**写**走 CLI 子进程
    （`python -m aesthetics_pipeline.cli generate ...`），流水线逻辑只有一份。
    """
    try:
        from .panel import serve
    except ImportError as exc:
        console.print(f"[red]面板依赖没装：{exc}[/red]")
        console.print("[dim]安装：pip install fastapi uvicorn[/dim]")
        raise typer.Exit(code=1)
    console.print(f"[green]面板启动：http://{host}:{port}[/green]")
    serve(host=host, port=port)


@app.command("packs")
def packs_cmd() -> None:
    """列出可用域包（当前生效的会标出来）。

    域包 = 一个行业的全部"数据"：任务矩阵、画面模板、文案模板、合规要求。
    换行业不改代码：`AP_PACK=<name> aesthetics plan` 即可。
    """
    from .pack import active_pack_name, available_packs, load_pack

    active = active_pack_name()
    names = available_packs()
    if not names:
        console.print("[yellow]没找到域包：packs/ 目录为空或不存在[/yellow]")
        console.print("[dim]未启用域包时系统走旧路径 data/tasks.yaml + prompts/scene.yaml[/dim]")
        return

    table = Table(title=f"域包（当前生效：{active or '未启用 → 旧路径'}）")
    for col in ("包名", "显示名", "版本", "城市", "项目", "文案模板", "合规声明"):
        table.add_column(col, overflow="fold")
    for name in names:
        try:
            pack = load_pack(name)
        except Exception as exc:  # 单个包坏了不影响列出其它包
            table.add_row(name, "[red]加载失败[/red]", "-", "-", "-", "-", str(exc)[:60])
            continue
        marker = " ←" if name == active else ""
        table.add_row(
            f"[bold]{name}[/bold]{marker}",
            str(pack.manifest.get("display") or "-"),
            pack.version,
            str(len(pack.matrix.get("cities") or [])),
            str(len(pack.matrix.get("projects") or {})),
            str(len((pack.copy_cfg.get("templates") or {}))),
            "有" if pack.compliance.get("required_disclaimer") else "缺席",
        )
    console.print(table)
    if not active:
        console.print("[dim]启用方式：AP_PACK=medical-aesthetics（Windows: $env:AP_PACK=...）[/dim]")


@app.command("compliance")
def compliance_cmd(
    db: Path = typer.Option(None),
    adopted_only: bool = typer.Option(False, help="只审已被人采纳的素材"),
) -> None:
    """合规审计：免责声明 / 图文成对 / 画面文字 / 禁用词。

    这是**发布前**的闸：不通过则退出码 1，可直接挂进 CI 或发布会话脚本。
    """
    with ledger.connect(db) as conn:
        result = compliance_mod.audit(conn, adopted_only=adopted_only)

    table = Table(
        title=f"合规审计（检查 {result['checked']} 条；域包 {result['pack'] or '未启用'}）"
    )
    for col in ("级别", "代码", "素材", "后端", "说明"):
        table.add_column(col, overflow="fold")
    for issue in result["issues"]:
        color = "red" if issue["severity"] == "error" else "yellow"
        table.add_row(
            f"[{color}]{issue['severity']}[/{color}]",
            issue["code"],
            issue["key"][:12],
            issue["provider"],
            issue["detail"],
        )
    if result["issues"]:
        console.print(table)
    if result["ok"]:
        console.print("[green]通过[/green]：没有发现发布阻断项")
    else:
        console.print(f"[red]未通过[/red]：{result['by_code']}")
        raise typer.Exit(1)


@app.command("publish")
def publish_cmd(
    out: Path = typer.Option(None, help="输出 zip 路径（默认 out/publish/ 下带时间戳）"),
    db: Path = typer.Option(None),
    adopted_only: bool = typer.Option(True, help="只打包被人采纳的素材（默认）"),
    allow_violations: bool = typer.Option(False, help="合规不过也强制打包（会写进 compliance.json）"),
    batch: str = typer.Option("", help="批次名，写进 manifest"),
) -> None:
    """打发布包：图 + 文案 + manifest（含 AI 标识）+ 合规结果。

    **默认 fail-closed**：合规审计不通过就拒绝打包。发布包里的 manifest.json
    就是《AI 生成合成内容标识办法》要求的"隐式标识"元数据。
    """
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out or (config.OUT_DIR / "publish" / f"publish-{stamp}.zip")
    try:
        with ledger.connect(db) as conn:
            path = compliance_mod.export_publish(
                conn,
                out_path,
                pack=try_load_pack(),
                adopted_only=adopted_only,
                allow_violations=allow_violations,
                batch=batch,
            )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    console.print(f"[green]发布包：{path}[/green]")
    console.print(
        "[dim]发出去之后回填：`aesthetics publish-record <key> --channel ... --url ...`，"
        "再 `aesthetics metrics <key> --impressions ... --clicks ...`，最后 `aesthetics funnel`[/dim]"
    )


@app.command("publish-record")
def publish_record(
    key: str = typer.Argument(..., help="幂等键（台账里的 idempotency_key）"),
    provider: str = typer.Option("unknown", help="产出后端"),
    channel: str = typer.Option(None, help="渠道：xiaohongshu / wechat / ..."),
    url: str = typer.Option(None, help="发布后的链接"),
    published_at: str = typer.Option(None, help="发布时间（默认现在）"),
    note: str = typer.Option(None, help="备注"),
    db: Path = typer.Option(None),
) -> None:
    """登记一次发布（回传表的入口）：素材真的发到哪个渠道了。"""
    with ledger.connect(db) as conn:
        ledger.record_publication(
            conn,
            idempotency_key=key,
            provider=provider,
            channel=channel,
            url=url,
            published_at=published_at,
            note=note,
        )
    console.print(f"[green]已登记发布[/green]：{key} → {channel or '-'} {url or ''}")


@app.command("metrics")
def metrics_cmd(
    key: str = typer.Argument(..., help="幂等键"),
    channel: str = typer.Option(None, help="渠道"),
    impressions: int = typer.Option(0, help="曝光量"),
    clicks: int = typer.Option(0, help="点击/互动量"),
    saves: int = typer.Option(0, help="收藏"),
    comments: int = typer.Option(0, help="评论"),
    shares: int = typer.Option(0, help="分享"),
    measured_at: str = typer.Option(None, help="数据日期（默认现在；同日期覆盖）"),
    provider: str = typer.Option("unknown", help="产出后端"),
    note: str = typer.Option(None, help="备注"),
    db: Path = typer.Option(None),
) -> None:
    """登记一次效果数据（回传表）：曝光/点击/收藏/评论/分享。"""
    with ledger.connect(db) as conn:
        ledger.record_metrics(
            conn,
            idempotency_key=key,
            provider=provider,
            channel=channel,
            measured_at=measured_at,
            impressions=impressions,
            clicks=clicks,
            saves=saves,
            comments=comments,
            shares=shares,
            note=note,
        )
    console.print(f"[green]已登记效果[/green]：{key} 曝光 {impressions} / 点击 {clicks}")


@app.command("funnel")
def funnel_cmd(db: Path = typer.Option(None)) -> None:
    """漏斗：产出 → 放行 → 人采纳 → 已发布 → 有数据。

    "发布/效果"是**人回填的事实**：没回填就是 0，不拿采纳率冒充发布率。
    """
    with ledger.connect(db) as conn:
        stats = ledger.funnel_summary(conn)

    table = Table(title="内容生产 → 发布 → 效果")
    for col in ("环节", "数量", "转化率"):
        table.add_column(col)

    def rate(num: int, den: int) -> str:
        return f"{num / den * 100:.1f}%" if den else "—"

    table.add_row("产出（落盘）", str(stats["generated"]), "—")
    table.add_row("质检放行", str(stats["accepted"]), rate(stats["accepted"], stats["generated"]))
    table.add_row("人工采纳", str(stats["adopted"]), rate(stats["adopted"], stats["accepted"]))
    table.add_row("已发布", str(stats["published"]), rate(stats["published"], stats["adopted"]))
    table.add_row("有回传数据", str(stats["measured"]), rate(stats["measured"], stats["published"]))
    console.print(table)

    totals = stats["totals"]
    if totals["impressions"]:
        console.print(
            f"效果合计：曝光 {totals['impressions']} / 点击 {totals['clicks']}"
            f" / 收藏 {totals['saves']} / 评论 {totals['comments']} / 分享 {totals['shares']}"
            f" / CTR {totals['ctr'] * 100:.2f}%"
        )
        if stats["by_channel"]:
            channel_table = Table(title="按渠道")
            for col in ("渠道", "素材", "曝光", "点击", "收藏", "CTR"):
                channel_table.add_column(col)
            for row in stats["by_channel"]:
                ctr = f"{row['ctr'] * 100:.2f}%" if row["ctr"] is not None else "—"
                channel_table.add_row(
                    str(row["channel"]), str(row["assets"]), str(row["impressions"]),
                    str(row["clicks"]), str(row["saves"]), ctr,
                )
            console.print(channel_table)
    else:
        console.print(
            "[dim]还没有效果回填。发布后用 `aesthetics publish-record` + "
            "`aesthetics metrics` 录入，这里才会出现 CTR。[/dim]"
        )


if __name__ == "__main__":
    app()
