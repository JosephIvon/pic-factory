"""Web 面板 API 冒烟。

不真的生图：POST /api/generate 只验"提交 → 起后台任务"这条路径，
子进程用 monkeypatch 换掉；流水线本身由 CLI / provider 的测试覆盖。
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.routing import Mount  # noqa: E402

from aesthetics_pipeline import config, ledger, panel  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """隔离台账 + 隔离 out 目录，测试不碰仓库里的真实数据。"""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(config, "OUT_DIR", out_dir)
    monkeypatch.setattr(ledger, "DB_PATH", tmp_path / "pipeline.db")

    img = out_dir / "abc123.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")  # 只测路径映射，不解码

    with ledger.connect() as conn:
        ledger.record_asset(
            conn,
            {
                "idempotency_key": "abc123",
                "provider": "matrix",
                "city": "武汉",
                "project": "水光针补水",
                "theme": "术后第7天真实反馈",
                "channel": "xiaohongshu",
                "character": "lin",
                "route": "i2i",
                "local_path": str(img),
                "prompt": "a photo of a woman in a clinic",
                "gen_seconds": 18.4,
                "est_cost": 0.2,
                "qc_verdict": "accepted",
                "qc_reasons": [],
                "qc_rules": {"width": 896, "height": 1152, "text_found": []},
                "face_similarity": 0.42,
                "face_similarity_after": 0.81,
                "route_rule_id": "R3_STRONG_IDENTITY",
                "route_reason": "强人物一致：API 身份保持更好",
            },
        )
    return TestClient(panel.app)


def test_index_serves_single_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "画廊" in resp.text


def test_meta_lists_filters_and_backends(client):
    data = client.get("/api/meta").json()
    assert "武汉" in data["cities"]
    assert "水光针补水" in data["projects"]
    assert set(data["backends"]) == {"matrix", "local"}
    assert data["total_specs"] > 0


def test_plan_shows_backend_and_rule(client):
    data = client.get("/api/plan", params={"city": "武汉"}).json()
    assert data["items"], "路由预览不能为空"
    item = data["items"][0]
    assert item["provider"] in ("matrix", "local")
    assert item["rule_id"].startswith("R")
    assert item["reason"]
    # 后端没装也要能预览规则（面板不能整页报错）
    assert "error" not in data


def test_gallery_and_detail_expose_qc_route_and_cost(client):
    items = client.get("/api/assets").json()["items"]
    assert len(items) == 1
    item = items[0]
    assert item["verdict"] == "accepted"
    assert item["rule_id"] == "R3_STRONG_IDENTITY"
    assert item["image"] == "/files/out/abc123.png"

    detail = client.get("/api/assets/abc123", params={"provider": "matrix"}).json()["asset"]
    assert detail["qc_rules"]["width"] == 896
    assert detail["rule_reason"].startswith("强人物一致")
    assert detail["face_similarity_after"] == 0.81
    assert "unit_cost" in detail
    assert detail["prompt"]


def test_cost_endpoint_does_not_invent_money(client, monkeypatch):
    for var in ("AP_COST_API", "AP_UNIT_COST", "AP_LOCAL_WATT", "AP_KWH_PRICE",
                "AP_LOCAL_DEPR_PER_HOUR"):
        monkeypatch.delenv(var, raising=False)
    # 把台账里记的金额清掉：单价没配时面板只能报"未配置单价"，不能编一个 0 元
    with ledger.connect() as conn:
        ledger.record_asset(
            conn, {"idempotency_key": "abc123", "provider": "matrix", "est_cost": None}
        )
    data = client.get("/api/cost").json()
    item = next(c for c in data["items"] if c["provider"] == "matrix")
    assert item["priced"] is False
    assert item["cost_per_accepted"] is None


def test_generate_submits_job_without_generating(client, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(job_id, steps):
        calls.append([name for name, _ in steps])
        panel.JOBS[job_id]["status"] = "done"

    monkeypatch.setattr(panel, "_run_job", fake_run)
    resp = client.post(
        "/api/generate",
        json={"city": "武汉", "provider": "auto", "limit": 2, "chain": True},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert job_id
    # 生成后自动质检 → 换脸兜底 → 出报表，顺序不能乱
    assert calls == [["generate", "qc", "faceswap", "report"]]

    jobs = client.get("/api/jobs").json()["items"]
    assert jobs[0]["id"] == job_id


def test_generate_rejects_bad_input(client):
    assert client.post("/api/generate", json={"limit": 0}).status_code == 400
    assert client.post("/api/generate", json={"provider": "bogus"}).status_code == 400


def test_generate_rejects_concurrent_submission(client, monkeypatch):
    """并发闸（Trae S1）：已有 running 批时第二发提交必须 409。
    两条 CLI 并行会互相踩 skip_existing、各持独立成本守卫、抢同一个 SQLite。"""
    monkeypatch.setattr(
        panel, "JOBS", {"busy": {"id": "busy", "status": "running", "steps": [], "log": []}}
    )
    resp = client.post("/api/generate", json={"provider": "auto", "limit": 2})
    assert resp.status_code == 409
    assert "在跑" in resp.json()["detail"]

    # 前一批结束后立刻恢复正常提交
    panel.JOBS["busy"]["status"] = "done"

    def fake_run(job_id, steps):
        panel.JOBS[job_id]["status"] = "done"

    monkeypatch.setattr(panel, "_run_job", fake_run)
    assert client.post("/api/generate", json={"provider": "auto", "limit": 2}).status_code == 200


# ---------------------------------------------------------------- P0-3 面板鉴权

TEST_TOKEN = "s3cret-token"


@pytest.fixture(autouse=True)
def _no_ambient_panel_token(monkeypatch):
    """测试环境隔离：宿主 shell 里残留的 AP_PANEL_TOKEN 不能把无鉴权用例打成 401。

    面板每次请求实时读 os.environ，所以不隔离的话整套测试的通过率取决于跑测试的
    机器有没有设这个变量——CI（task #17 门禁）上尤其危险。
    """
    monkeypatch.delenv("AP_PANEL_TOKEN", raising=False)


@pytest.fixture()
def auth_client(client, tmp_path, monkeypatch):
    """带令牌的 client：设 AP_PANEL_TOKEN，并把 /files/out 重指到 fixture 的 tmp 目录。

    app 导入时 StaticFiles 挂的是当时的 config.OUT_DIR（仓库真实 out/），client
    fixture 换了 config.OUT_DIR 并不会带着挂载一起换 —— 这里原地替换 Mount 路由，
    让 /files/out/abc123.png 真能从 tmp 目录出图（teardown 还原，别污染别的用例）。
    """
    monkeypatch.setenv("AP_PANEL_TOKEN", TEST_TOKEN)
    original = None
    for index, route in enumerate(panel.app.routes):
        if isinstance(route, Mount) and route.path == "/files/out":
            original = (index, route)
            panel.app.routes[index] = Mount(
                "/files/out",
                app=StaticFiles(directory=str(tmp_path / "out")),
                name="files-out",
            )
    assert original is not None, "panel 的 /files/out 挂载不见了，测试前提失效"
    yield client
    panel.app.routes[original[0]] = original[1]


def test_api_requires_credentials_when_token_set(auth_client):
    resp = auth_client.get("/api/assets")
    assert resp.status_code == 401
    assert "需要面板访问令牌" in resp.json()["detail"]


def test_api_bearer_header_passes(auth_client):
    resp = auth_client.get("/api/assets", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
    assert resp.status_code == 200
    assert resp.json()["items"][0]["key"] == "abc123"


def test_api_rejects_wrong_token_and_write_endpoints_guarded(auth_client):
    # 错 token：头和 query 两条路都必须 401
    assert auth_client.get(
        "/api/assets", headers={"Authorization": f"Bearer {TEST_TOKEN}x"}
    ).status_code == 401
    assert auth_client.get("/api/assets", params={"token": "wrong"}).status_code == 401
    # 花钱/污染数据的写接口同样在闸后（generate 走 monkeypatch，不该起 job）
    assert auth_client.post("/api/generate", json={"provider": "auto", "limit": 1}).status_code == 401
    assert not panel.JOBS or all(j.get("status") != "running" for j in panel.JOBS.values())
    # 单页 HTML 不在 /api、/files 范围内：不设防（审计口径只要求数据面）
    assert auth_client.get("/").status_code == 200


def test_files_accepts_query_token_and_requires_it(auth_client):
    # <img src> 带不了请求头，/files 必须支持 ?token=
    ok = auth_client.get("/files/out/abc123.png", params={"token": TEST_TOKEN})
    assert ok.status_code == 200
    assert auth_client.get("/files/out/abc123.png").status_code == 401
    assert auth_client.get("/files/out/abc123.png", params={"token": "nope"}).status_code == 401
    # Bearer 头对 /files 也有效（两条凭据路对 /api 与 /files 一致）
    assert auth_client.get(
        "/files/out/abc123.png", headers={"Authorization": f"Bearer {TEST_TOKEN}"}
    ).status_code == 200


def test_no_token_keeps_behavior_unchanged(client):
    """未设置 AP_PANEL_TOKEN：零门槛，鉴权引入前的所有面都照常（本机开发体验）。"""
    assert client.get("/api/assets").status_code == 200
    assert client.get("/api/meta").status_code == 200
    assert client.get("/").status_code == 200


def test_serve_gate_blocks_public_bind_without_token(monkeypatch):
    """fail-closed：非回环绑定 + 没设 token = 拒绝启动，报错给两条出路。"""
    monkeypatch.delenv("AP_PANEL_TOKEN", raising=False)
    with pytest.raises(RuntimeError) as err:
        panel.assert_serve_security("0.0.0.0")
    msg = str(err.value)
    assert "AP_PANEL_TOKEN" in msg
    assert "127.0.0.1" in msg


def test_serve_gate_allows_public_bind_with_token(monkeypatch):
    monkeypatch.setenv("AP_PANEL_TOKEN", TEST_TOKEN)
    panel.assert_serve_security("0.0.0.0")  # 不抛即通过


def test_serve_gate_allows_loopback_without_token(monkeypatch):
    monkeypatch.delenv("AP_PANEL_TOKEN", raising=False)
    panel.assert_serve_security("127.0.0.1")
    panel.assert_serve_security("localhost")


def test_token_bootstrap_cookie_makes_ui_work(auth_client):
    """/?token=<token> 引导一次即写 httponly cookie，页面里 fetch/img 免逐请求带参。
    没有这条闭环，设了 token 后面板整页 401——"能锁"和"能用"必须同时成立。"""
    bootstrap = auth_client.get("/", params={"token": TEST_TOKEN})
    assert bootstrap.status_code == 200
    assert bootstrap.cookies.get("panel_token") == TEST_TOKEN
    assert auth_client.get("/api/assets").status_code == 200
    assert auth_client.get("/files/out/abc123.png").status_code == 200


def test_wrong_token_on_index_sets_no_cookie(auth_client):
    resp = auth_client.get("/", params={"token": "nope"})
    assert resp.status_code == 200
    assert "panel_token" not in resp.cookies
    assert auth_client.get("/api/assets").status_code == 401
