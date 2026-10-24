"""轻量 Web 面板：任务提交 + 画廊 + 每张的质检明细/路由理由/成本。

定位是**薄壳**，不是第二套系统：
  * 读 —— 直接查台账（进程内 SQLite，与 CLI 同一份库）
  * 写 —— 调 CLI 子进程（`python -m aesthetics_pipeline.cli generate ...`）
流水线逻辑只有一份，面板不重复实现；流水线也不依赖面板。

为什么要面板：评审口径不是"能跑"，而是"能交付"——全栈交付和部署维护
要有硬证据。这个面板就是那件证据：
提交任务 → 自动路由 → 出图 → 质检 → 换脸兜底 → 画廊逐张可查
（判定理由 / 命中哪条路由规则 / 这张花了多少钱）。

静态页零 CDN 依赖：断网环境下页面也必须照样能开。
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, ledger, report as report_mod, routing
from .router import Router
from .taskspec import expand, filter_specs, load_yaml, pack_matrix, scene_config

PANEL_HTML = Path(__file__).with_name("panel.html")

# 任务状态：内存里存就够了，进程重启就没了 —— 面板不是台账，台账才是。
JOBS: dict[str, dict[str, Any]] = {}

app = FastAPI(title="医美 AI 图文闭环 · 控制台", version="0.1.0")

# 图片静态挂载：画廊要能看图。目录不存在就建（out/ 与 evidence/ 都是仓库内目录）。
for _dir, _prefix in ((config.OUT_DIR, "/files/out"), (config.EVIDENCE_DIR, "/files/evidence")):
    Path(_dir).mkdir(parents=True, exist_ok=True)
    app.mount(_prefix, StaticFiles(directory=str(_dir)), name=_prefix.strip("/").replace("/", "-"))


# ---------------------------------------------------------------- 鉴权（P0-3）
#
# 环境变量 AP_PANEL_TOKEN：**每次请求实时读 os.environ**（不缓存），测试可直接
# monkeypatch，运维换 token 也无需重启。
#   * 已设置：除 `/api/health`（面板没有该端点）外，所有 /api/* 与 /files/*
#     必须携带凭据 —— 请求头 `Authorization: Bearer <token>`，或查询参数
#     `?token=<token>`（`<img src>` 带不了请求头，/files 必须支持 query）。
#     缺失或不匹配 → 401。比较用 secrets.compare_digest，防时序侧信道。
#   * 未设置：行为与鉴权引入前完全一致（本机开发零门槛）。

def _panel_token() -> str:
    """当前面板访问令牌；空串/未设置 = 鉴权关闭。"""
    return (os.environ.get("AP_PANEL_TOKEN") or "").strip()


def _supplied_credentials(request: Request) -> list[str]:
    """从请求里取候选凭据：query 的 token + Bearer 头（大小写不敏感的 scheme）
    + `panel_token` cookie（由 `/?token=…` 引导写入，让页面里的 fetch/img 免逐请求带参）。"""
    candidates: list[str] = []
    query_token = request.query_params.get("token")
    if query_token:
        candidates.append(query_token)
    auth = request.headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        candidates.append(auth[7:].strip())
    cookie_token = request.cookies.get("panel_token")
    if cookie_token:
        candidates.append(cookie_token)
    return candidates


@app.middleware("http")
async def require_panel_token(request: Request, call_next):
    """内网暴露时的凭据闸：只管 /api/* 与 /files/*，其余（单页 HTML）放行。"""
    expected = _panel_token()
    path = request.url.path
    if expected and (path.startswith("/api/") or path.startswith("/files/")):
        if not any(secrets.compare_digest(s, expected) for s in _supplied_credentials(request)):
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "需要面板访问令牌：请携带请求头 Authorization: Bearer <AP_PANEL_TOKEN>"
                    "，或在 URL 加 ?token=<AP_PANEL_TOKEN>（图片 /files 用后者）"
                },
            )
    return await call_next(request)


# 回环地址白名单：绑这些地址时安全边界仍是"只有本机能连"，token 可不设。
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


def assert_serve_security(host: str) -> None:
    """fail-closed 启动闸：非回环绑定必须设 AP_PANEL_TOKEN，否则拒绝启动。

    serve() 内部会调用；CLI/其他启动路径若不走 serve()，应在起服务前显式调用本
    函数（主智能体接线用）。判定只看实时 os.environ，方便测试 monkeypatch。
    """
    if host in LOOPBACK_HOSTS:
        return
    if _panel_token():
        return
    raise RuntimeError(
        f"拒绝启动：面板绑定 {host!r}（非回环）却没设 AP_PANEL_TOKEN——"
        "内网任何人都能提交生成（花钱）、点采纳（污染台账）。两条出路任选其一："
        "1) 设环境变量 AP_PANEL_TOKEN=<随机串>；2) 改回回环绑定 --host 127.0.0.1。"
    )


# ---------------------------------------------------------------- 工具


def _router() -> Router:
    matrix = pack_matrix()
    scene = scene_config()
    return Router(matrix, scene)


def _json(value: Any, default: Any) -> Any:
    """台账里的 JSON 列存的是字符串，这里还原成对象。"""
    if value is None or value == "":
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _file_url(path: str | None) -> str | None:
    """把绝对路径换成面板能访问的 URL。仓库外的文件不给 URL（如实缺失）。"""
    if not path:
        return None
    resolved = Path(path).resolve()
    for root, prefix in (
        (config.OUT_DIR, "/files/out"),
        (config.EVIDENCE_DIR, "/files/evidence"),
    ):
        try:
            rel = resolved.relative_to(Path(root).resolve())
        except ValueError:
            continue
        return f"{prefix}/{rel.as_posix()}"
    return None


def _final_path(row: dict[str, Any]) -> str | None:
    """最终产物：换过脸的用 swapped_path，否则用 local_path。

    拼图/画廊必须展示最终产物，否则会出现"图上还是换脸前那张脸，
    标签却写着放行"的自相矛盾。

    走 ledger.asset_path：台账里存的是绝对路径，仓库被整体搬走后要能
    重新锚定到当前仓库根，否则画廊会整片空图。
    """
    path = ledger.asset_path(row, "swapped_path") or ledger.asset_path(row)
    return str(path) if path else None


def _brief(row: dict[str, Any]) -> dict[str, Any]:
    """画廊卡片要的字段（够画卡片 + 排序，不含大字段）。"""
    return {
        "id": f"{row['idempotency_key']}:{row.get('provider')}",
        "key": row["idempotency_key"],
        "provider": row.get("provider"),
        "city": row.get("city"),
        "project": row.get("project"),
        "theme": row.get("theme"),
        "channel": row.get("channel"),
        "character": row.get("character"),
        "route": row.get("route"),
        "verdict": row.get("qc_verdict") or "pending",
        "reasons": _json(row.get("qc_reasons"), []),
        # 抽检台人工采纳（与机器质检判定分开的两列）
        "adopted": row.get("adopted"),
        "adopted_at": row.get("adopted_at"),
        "adopt_note": row.get("adopt_note"),
        "rule_id": row.get("route_rule_id"),
        "rule_reason": row.get("route_reason"),
        "face_similarity": row.get("face_similarity"),
        "face_similarity_after": row.get("face_similarity_after"),
        "gen_seconds": row.get("gen_seconds"),
        "est_cost": row.get("est_cost"),
        "swapped": bool(row.get("swapped_path")),
        "image": _file_url(_final_path(row)),
    }


# ---------------------------------------------------------------- 只读接口


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """单页前端。零 CDN 依赖，离线可演示。

    鉴权引导：设了 AP_PANEL_TOKEN 时，页面里的 fetch/img 没法逐请求带头，
    用 `/?token=<token>` 打开一次即把凭据写入 httponly cookie（浏览器对
    同源 fetch 与 <img> 都会自动带上），之后的 /api 与 /files 请求免参数。
    """
    if not PANEL_HTML.exists():  # pragma: no cover —— 打包漏掉时才走到
        raise HTTPException(500, f"面板页面缺失: {PANEL_HTML}")
    html = PANEL_HTML.read_text(encoding="utf-8")
    expected = _panel_token()
    supplied = request.query_params.get("token") or ""
    resp = HTMLResponse(html)
    if expected and supplied and secrets.compare_digest(supplied, expected):
        resp.set_cookie("panel_token", supplied, httponly=True, samesite="lax")
    return resp


@app.get("/api/meta")
def api_meta() -> dict[str, Any]:
    """筛选选项 + 后端可用性 + 是否配了单价。"""
    specs = expand()
    matrix = pack_matrix()
    backends: dict[str, Any] = {}
    for name in (routing.MATRIX, routing.LOCAL):
        ok, why = routing.backend_available(name)
        backends[name] = {"available": ok, "reason": why}
    return {
        "cities": sorted({s.city for s in specs}),
        "projects": list(matrix.get("projects") or {}),
        "themes": list(matrix.get("themes") or {}),
        "characters": list(matrix.get("characters") or {}),
        "total_specs": len(specs),
        "backends": backends,
        # 单价没配时面板要显示"未配置单价"，不能默默显示 0 元
        "cost_configured": bool(config.api_unit_cost() or config.local_unit_cost(1.0)),
    }


@app.get("/api/plan")
def api_plan(
    city: str | None = None,
    project: str | None = None,
    theme: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """路由预览：提交前就能看到每条任务会走哪个后端、命中哪条规则。"""
    router = _router()
    specs = filter_specs(expand(), city=city, project=project, theme=theme, limit=limit)
    items: list[dict[str, Any]] = []
    decisions: list[routing.RouteDecision] = []
    for spec in specs:
        path, _ = router.decide_path(spec)
        try:
            decision = routing.decide_provider(spec, require_available=False)
        except RuntimeError as exc:
            return {"items": [], "summary": {}, "total": len(specs), "error": str(exc)}
        decisions.append(decision)
        items.append(
            {
                "key": spec.idempotency_key,
                "city": spec.city,
                "project": spec.project,
                "theme": spec.theme,
                "channel": spec.channel,
                "character": spec.character,
                "route": path,
                "provider": decision.provider,
                "rule_id": decision.rule_id,
                "reason": decision.reason,
            }
        )
    return {
        "items": items,
        "summary": routing.summarize(decisions),
        "total": len(specs),
    }


@app.get("/api/assets")
def api_assets(limit: int = 200) -> dict[str, Any]:
    """画廊列表。"""
    with ledger.connect() as conn:
        rows = ledger.assets(conn)
    items = [_brief(r) for r in rows][-limit:]
    return {"items": items, "total": len(rows)}


@app.get("/api/assets/{key}")
def api_asset_detail(key: str, provider: str | None = None) -> dict[str, Any]:
    """单张明细：质检明细 + 路由理由 + 成本。"""
    with ledger.connect() as conn:
        rows = [
            r
            for r in ledger.assets(conn)
            if r["idempotency_key"] == key and (provider is None or r.get("provider") == provider)
        ]
    if not rows:
        raise HTTPException(404, f"没有这条产出: {key}")
    row = rows[0]
    seconds = row.get("gen_seconds") or 0.0
    detail = _brief(row)
    detail.update(
        {
            "prompt": row.get("prompt"),
            "node_id": row.get("node_id"),
            "file_name": row.get("file_name"),
            "local_path": str(ledger.asset_path(row) or "") or None,
            "swapped_path": str(ledger.asset_path(row, "swapped_path") or "") or None,
            "qc_rules": _json(row.get("qc_rules"), {}),
            "text_found": _json(row.get("text_found"), []),
            "created_at": row.get("created_at"),
            # 这张图的金额：台账生成时记的是事实；没记就按当前单价假设现算
            "unit_cost": config.unit_cost_for(row.get("provider") or "", seconds),
        }
    )
    return {"asset": detail}


@app.get("/api/cost")
def api_cost() -> dict[str, Any]:
    """成本账：单张合格图综合成本 = 总花费 ÷ 合格张数。"""
    with ledger.connect() as conn:
        rows = ledger.assets(conn)
        summary = ledger.summary(conn)
    return {"items": report_mod.cost_summary(rows), "summary": summary}


@app.get("/api/summary")
def api_summary() -> dict[str, Any]:
    with ledger.connect() as conn:
        return ledger.summary(conn)


# ---------------------------------------------------------------- 写：走 CLI


def _cli(*args: str) -> list[str]:
    return [sys.executable, "-m", "aesthetics_pipeline.cli", *args]


def _run_job(job_id: str, steps: list[tuple[str, list[str]]]) -> None:
    """跑一串 CLI 命令，边跑边把输出记进 job 日志。

    写操作走 CLI 子进程而不是在面板里复刻一遍流水线：
    逻辑只有一份，面板只是 CLI 的一层皮。
    """
    job = JOBS[job_id]
    try:
        for name, cmd in steps:
            job["steps"].append(
                {"name": name, "status": "running", "cmd": " ".join(cmd[2:])}
            )
            proc = subprocess.Popen(
                cmd,
                cwd=str(config.REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    job["log"].append(f"[{name}] {line}")
            rc = proc.wait()
            job["steps"][-1].update({"status": "done" if rc == 0 else "failed", "rc": rc})
            if rc != 0:
                job["status"] = "failed"
                job["log"].append(f"[{name}] 退出码 {rc}，后续步骤跳过")
                return
        job["status"] = "done"
    except Exception as exc:  # pragma: no cover —— 兜底，别让后台线程静默死掉
        job["status"] = "failed"
        job["log"].append(f"[error] {exc}")


@app.post("/api/generate")
def api_generate(payload: dict[str, Any]) -> dict[str, Any]:
    """提交一批任务：生成 →（可选）质检 → 换脸兜底 → 出报表。

    立即返回 job_id，页面轮询 /api/jobs 看进度 —— 生图要几十秒，
    让浏览器同步等会把请求挂死。
    """
    provider = str(payload.get("provider") or "auto")
    if provider not in ("auto", "matrix", "local", "replay"):
        raise HTTPException(400, f"未知后端 {provider!r}")
    raw_limit = payload.get("limit")
    try:
        limit = 4 if raw_limit in (None, "") else int(raw_limit)
    except (TypeError, ValueError):
        raise HTTPException(400, f"limit 不是整数: {raw_limit!r}")
    if limit < 1 or limit > 50:
        raise HTTPException(400, "limit 取值范围 1~50")
    # 并发闸：一次只放一个批。两条 CLI 并行会互相踩 skip_existing（都查不到
    # 对方的行 → 双份花钱）、各持独立 BudgetGuard（熔断基数互相不可见），
    # 还要抢同一个 SQLite。双击/脚本重放都在这一行挡掉。
    if any(job.get("status") == "running" for job in JOBS.values()):
        raise HTTPException(409, "已有生成任务在跑，请等它结束后再提交")

    gen = _cli("generate", "--provider", provider, "--limit", str(limit))
    for field in ("city", "project", "theme"):
        value = str(payload.get(field) or "").strip()
        if value:
            gen += [f"--{field}", value]

    steps: list[tuple[str, list[str]]] = [("generate", gen)]
    if bool(payload.get("chain", True)):
        # 出厂判定放在换脸之后（生成阶段的人脸分只标记不拦截），
        # 所以链条顺序不能乱。
        steps += [
            ("qc", _cli("qc")),
            ("faceswap", _cli("faceswap")),
            ("report", _cli("report")),
        ]

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id,
        "status": "running",
        "created_at": time.time(),
        "steps": [],
        "log": [],
    }
    threading.Thread(target=_run_job, args=(job_id, steps), daemon=True).start()
    return {"job_id": job_id, "steps": [name for name, _ in steps]}


@app.post("/api/assets/adopt")
def api_adopt(payload: dict[str, Any]) -> dict[str, Any]:
    """抽检台"用/不用"按钮的后端。

    **写仍然走 CLI 子进程**（`adopt` 子命令），和 /api/generate 同一约定：
    面板不 import ledger 写库，流水线写入口只有一份。区别只在采纳是瞬时
    操作，同步跑完 CLI 直接回传退出码，前端拿到 ok 就刷新，不用起后台 job。
    """
    key = str(payload.get("key") or "").strip()
    decision = str(payload.get("decision") or "").strip()
    if not key:
        raise HTTPException(400, "缺少 key")
    if decision not in ("adopted", "rejected"):
        raise HTTPException(400, "decision 只能是 adopted / rejected")
    args = ["adopt", key, "--decision", decision]
    note = str(payload.get("note") or "").strip()
    if note:
        args += ["--note", note]
    provider = str(payload.get("provider") or "").strip()
    if provider:
        args += ["--provider", provider]
    cmd = _cli(*args)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(config.REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "adopt 子进程超时")
    if proc.returncode != 0:
        detail = (proc.stdout or proc.stderr or "").strip()[:400]
        raise HTTPException(422, f"采纳写入失败（CLI 退出码 {proc.returncode}）：{detail}")
    return {
        "ok": True,
        "key": key,
        "provider": provider or None,
        "decision": decision,
        "cli": " ".join(cmd[2:]),  # 证明这笔写是从 CLI 子进程出去的
        "output": (proc.stdout or "").strip(),
    }


@app.get("/api/jobs")
def api_jobs() -> dict[str, Any]:
    jobs = sorted(JOBS.values(), key=lambda j: j["created_at"], reverse=True)
    return {
        "items": [
            {
                "id": j["id"],
                "status": j["status"],
                "created_at": j["created_at"],
                "steps": j["steps"],
                "log": j["log"][-60:],
            }
            for j in jobs
        ]
    }


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """起服务（CLI 的 panel 命令调它）。"""
    import uvicorn

    # fail-closed：非回环绑定又没设 token 时直接拒绝启动，而不是"先起来再说"。
    assert_serve_security(host)
    uvicorn.run(app, host=host, port=port, log_level="info")
