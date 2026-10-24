"""StandardApiProvider 离线测试：一个执行器 + ark/wan 两个族适配器。

全部走注入的假 httpx 客户端：绝不打真实 API、绝不需要 key。
覆盖 MODEL-CHOICE §2/§9 里那些"只有读过接口文档才知道"的坑。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aesthetics_pipeline.cost_guard import BudgetGuard
from aesthetics_pipeline.models import GenItem, TaskSpec
from aesthetics_pipeline.provider import http_api as http_api_mod
from aesthetics_pipeline.provider.http_api import (
    ARK_EP,
    StandardApiProvider,
    fold_negative,
    sniff_ext,
)
from aesthetics_pipeline.provider.models_spec import get as get_spec
from aesthetics_pipeline import ledger

# 一段假 JPEG（魔数 FFD8）与假 PNG（魔数 89504E47）
FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"jpegbody" * 16
FAKE_PNG = b"\x89\x50\x4e\x47\r\n\x1a\n" + b"pngbody" * 16

SEEDREAM = "doubao-seedream-4-0"
SEEDREAM_UNIT = get_spec(SEEDREAM).unit_cost_cny


class FakeResp:
    def __init__(self, status=200, payload=None, content=b"", text="", headers=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.content = content
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeClient:
    """记录 post/get 调用与所用请求头；按队列返回预设响应。

    并发跑批下"按队列弹出"不再可控（哪个线程先 pop 哪条是调度决定的），
    所以额外支持 posts_fn/gets_fn：给一个 (url, json)->resp / (url)->resp 的工厂，
    响应由**请求内容**决定，与线程时序无关。队列与工厂二选一，既有测试不受影响。
    """

    def __init__(self, headers=None, posts=None, gets=None, posts_fn=None, gets_fn=None):
        self.headers = headers or {}
        self._posts = list(posts or [])
        self._gets = list(gets or [])
        self._posts_fn = posts_fn
        self._gets_fn = gets_fn
        self.post_calls: list[dict] = []
        self.get_calls: list[dict] = []

    def post(self, url, json=None):
        self.post_calls.append({"url": url, "json": json, "headers": dict(self.headers)})
        if self._posts_fn is not None:
            return self._posts_fn(url=url, json=json)
        return self._posts.pop(0)

    def get(self, url):
        self.get_calls.append({"url": url, "headers": dict(self.headers)})
        if self._gets_fn is not None:
            return self._gets_fn(url=url)
        return self._gets.pop(0)


def make_item(
    key: str = "k1",
    negative: str = "",
    refs: list[str] | None = None,
    prompt: str = "a bright clinical photo of a woman",
) -> GenItem:
    spec = TaskSpec(
        city="武汉", project="光子嫩肤", theme="夏日透亮", channel="小红书",
        character=None, aspect_ratio="3:4", resolution="1536x2048",
        idempotency_key=key, path="t2i" if not refs else "i2i",
    )
    return GenItem(
        task=spec, prompt=prompt,
        negative_prompt=negative, aspect_ratio="3:4", resolution="1536x2048",
        input_urls=refs or [], output_file=f"{key}.png",
    )


def ark_post_ok() -> FakeResp:
    return FakeResp(200, payload={"data": [{"url": "https://ark/x.jpg", "size": "1536x2048"}],
                                  "usage": {"generated_images": 1}},
                     headers={"x-request-id": "req-ark-1"})


def wan_create(task_id="t-1") -> FakeResp:
    return FakeResp(200, payload={"output": {"task_id": task_id}})


def wan_poll(results_urls, status="SUCCEEDED") -> FakeResp:
    return FakeResp(200, payload={"output": {
        "task_status": status, "results": [{"url": u} for u in results_urls]}})


def build_provider(tmp_path, *, ark=None, wan=None, download=None, budget=None,
                   concurrency=None):
    dl = download or FakeClient(headers={}, gets=[FakeResp(200, content=FAKE_JPEG)])
    p = StandardApiProvider(
        out_dir=tmp_path,
        ark_client=ark if ark is not None else FakeClient(headers={"Authorization": "Bearer x"}, posts=[ark_post_ok()]),
        wan_client=wan if wan is not None else FakeClient(
            headers={"Authorization": "Bearer x", "X-DashScope-Async": "enable"},
            posts=[wan_create()],
            gets=[wan_poll(["https://oss.signed/y.jpg?sig=abc"])],
        ),
        download_client=dl,
        budget_guard=budget if budget is not None else BudgetGuard(None),
        sleep_func=lambda *_: None,
        concurrency=concurrency,
    )
    return p, dl


# ---------------- 魔数 / 负向词纯函数 ----------------

def test_sniff_ext_magic_bytes() -> None:
    assert sniff_ext(FAKE_JPEG) == ".jpg"
    assert sniff_ext(FAKE_PNG) == ".png"
    assert sniff_ext(b"garbage") == ".bin"


def test_fold_negative_appends_and_no_field() -> None:
    folded = fold_negative("a woman portrait", "text, watermark")
    assert "text" in folded and "watermark" in folded
    assert folded.startswith("a woman portrait")


# ---------------- ark 同步 ----------------

def test_ark_sync_success(tmp_path) -> None:
    p, dl = build_provider(tmp_path)
    out = p.generate([make_item("arkok")], model_ids=["doubao-seedream-4-0"])[0]
    assert out.ok is True
    assert out.local_path and Path(out.local_path).exists()
    assert out.source_provider == "http_api"
    # 假 JPEG 落盘扩展名必须是 .jpg（不信请求里的 .png）
    assert out.file_name.endswith("__http_api.jpg")
    # 缓存带档位（Trae S4）：断网回放要能还原"这张是哪个档生成的"
    entry = json.loads((tmp_path / "cache" / "http_api" / "arkok.json").read_text(encoding="utf-8"))
    assert entry["model_id"] == "doubao-seedream-4-0"


def test_ark_failure_not_billed_not_cached(tmp_path) -> None:
    """ark 失败：不计费、不写缓存（也就不会进 successful_keys）。"""
    bad = FakeClient(headers={"Authorization": "Bearer x"},
                     posts=[FakeResp(500, text="rate limited")])
    budget = BudgetGuard(None)
    p, _ = build_provider(tmp_path, ark=bad, budget=budget)
    out = p.generate([make_item("arkfail")], model_ids=["doubao-seedream-4-0"])[0]
    assert out.ok is False
    assert out.local_path is None
    assert budget.spent == 0.0                       # 没扣预算
    db = tmp_path / "t.db"
    with ledger.connect(db) as conn:
        # 缓存目录里没有该 provider 的成功条目可回放 → 不会被认成已完成
        assert (tmp_path / "cache" / "http_api" / "arkfail.json").exists() is False


def test_ark_folds_negative_into_prompt_no_field(tmp_path) -> None:
    ark = FakeClient(headers={"Authorization": "Bearer x"}, posts=[ark_post_ok()])
    p, _ = build_provider(tmp_path, ark=ark)
    p.generate([make_item("arkneg", negative="text, watermark, plastic skin")],
               model_ids=["doubao-seedream-4-0"])
    body = ark.post_calls[0]["json"]
    assert "negative_prompt" not in body                       # ark 族无独立字段
    assert "plastic skin" in body["prompt"]                    # 折进正向


# ---------------- wan 异步 ----------------

def test_wan_async_success_forces_n1_and_watermark(tmp_path) -> None:
    wan = FakeClient(
        headers={"Authorization": "Bearer x", "X-DashScope-Async": "enable"},
        posts=[wan_create()],
        gets=[
            FakeResp(200, payload={"output": {"task_status": "PENDING"}}),
            wan_poll(["https://oss.signed/z.jpg?sig=1"]),
        ],
    )
    p, dl = build_provider(tmp_path, wan=wan)
    out = p.generate([make_item("wanok")], model_ids=["qwen-image-3.0"])[0]
    assert out.ok is True
    assert out.file_name.endswith("__http_api.jpg")
    params = wan.post_calls[0]["json"]["parameters"]
    assert params["n"] == 1                    # 关键坑：默认 n=4 必须强制 1
    assert params["watermark"] is True         # 默认 false → 交付红线显式开
    assert wan.post_calls[0]["headers"].get("X-DashScope-Async") == "enable"


def test_wan_separate_negative_field(tmp_path) -> None:
    wan = FakeClient(
        headers={"Authorization": "Bearer x"},
        posts=[wan_create()],
        gets=[wan_poll(["https://oss.signed/z.jpg?sig=1"])],
    )
    p, _ = build_provider(tmp_path, wan=wan)
    p.generate([make_item("wanneg", negative="text, watermark")], model_ids=["qwen-image-3.0"])
    inp = wan.post_calls[0]["json"]["input"]
    assert inp["negative_prompt"] == "text, watermark"         # wan 族走独立字段


def test_wan_signed_url_download_has_no_auth_header(tmp_path) -> None:
    """核心坑：下载 OSS 签名 URL 绝不能带 Authorization（否则 403）。"""
    dl = FakeClient(headers={}, gets=[FakeResp(200, content=FAKE_PNG)])  # 干净客户端
    p, _ = build_provider(tmp_path, download=dl)
    p.generate([make_item("wandl")], model_ids=["qwen-image-3.0"])
    assert dl.get_calls, "下载没有被调用"
    assert "Authorization" not in dl.get_calls[0]["headers"]
    # 落盘按 PNG 魔数定扩展名
    files = list((tmp_path / "images").glob("wandl__http_api.*"))
    assert files and files[0].suffix == ".png"


def test_wan_billing_mismatch_rejected(tmp_path) -> None:
    """wan 返回 2 张（n 没被强制到位）→ 计费断言失败，拒绝落账、不计费。"""
    wan = FakeClient(
        headers={"Authorization": "Bearer x"},
        posts=[wan_create()],
        gets=[wan_poll(["https://oss/a.jpg", "https://oss/b.jpg"])],
    )
    budget = BudgetGuard(None)
    p, _ = build_provider(tmp_path, wan=wan, budget=budget)
    out = p.generate([make_item("wanbill")], model_ids=["qwen-image-3.0"])[0]
    assert out.ok is False
    assert "计费断言" in (out.error or "")
    assert budget.spent == 0.0                       # 拒绝落账 → 不扣预算


# ---------------- 参考图上限 ----------------

def test_ref_over_limit_errors_without_request(tmp_path) -> None:
    """qwen ref_max=3，传 4 张 → 该 item 报错，且绝不发请求。"""
    wan = FakeClient(headers={"Authorization": "Bearer x"}, posts=[wan_create()], gets=[])
    p, _ = build_provider(tmp_path, wan=wan)
    refs = ["https://a/1.jpg", "https://a/2.jpg", "https://a/3.jpg", "https://a/4.jpg"]
    out = p.generate([make_item("wanref", refs=refs)], model_ids=["qwen-image-3.0"])[0]
    assert out.ok is False
    assert "上限" in (out.error or "")
    assert wan.post_calls == []                       # 超限前就拦下，没打 API


# ---------------- 预算熔断 ----------------

def test_budget_fuse_blocks_request_before_api(tmp_path) -> None:
    wan = FakeClient(headers={"Authorization": "Bearer x"}, posts=[wan_create()], gets=[])
    guard = BudgetGuard(100.0, spent=85.0)            # 已达 85% > 80%
    p, _ = build_provider(tmp_path, wan=wan, budget=guard)
    out = p.generate([make_item("wanfuse")], model_ids=["qwen-image-3.0"])[0]
    assert out.ok is False
    assert "预算熔断" in (out.error or "")
    assert wan.post_calls == []                       # 熔断 → 绝不发请求


def test_mixed_model_ids_in_one_batch(tmp_path) -> None:
    """同批混合 ark + wan：两个族各自的客户端被命中。"""
    ark = FakeClient(headers={"Authorization": "Bearer x"}, posts=[ark_post_ok()])
    wan = FakeClient(
        headers={"Authorization": "Bearer x"},
        posts=[wan_create()],
        gets=[wan_poll(["https://oss/z.jpg"])],
    )
    dl = FakeClient(headers={}, gets=[FakeResp(200, content=FAKE_JPEG),
                                      FakeResp(200, content=FAKE_JPEG)])
    p = StandardApiProvider(
        out_dir=tmp_path, ark_client=ark, wan_client=wan, download_client=dl,
        budget_guard=BudgetGuard(None), sleep_func=lambda *_: None,
    )
    outs = p.generate([make_item("m-ark"), make_item("m-wan")],
                      model_ids=["doubao-seedream-4-0", "qwen-image-3.0"])
    assert outs[0].ok is True and outs[1].ok is True
    assert len(ark.post_calls) == 1 and len(wan.post_calls) == 1


# ---------------- 并发跑批（P0-1） ----------------

def test_concurrent_batch_success_order_and_charges(tmp_path, monkeypatch) -> None:
    """8 个 item、AP_CONCURRENCY=4：全成功、返回顺序与 items 严格一致、扣费=8×单价。

    并发是"何时跑"的改变，不是"谁跑哪条"的改变：产物命名（key__http_api.jpg）、
    缓存写入（含 model_id）、顺序契约一条都不能松。
    """
    monkeypatch.setenv("AP_CONCURRENCY", "4")
    ark = FakeClient(headers={"Authorization": "Bearer ark"},
                     posts=[ark_post_ok() for _ in range(8)])
    dl = FakeClient(headers={}, gets=[FakeResp(200, content=FAKE_JPEG) for _ in range(8)])
    budget = BudgetGuard(None)
    p, _ = build_provider(tmp_path, ark=ark, download=dl, budget=budget)
    assert p.concurrency == 4

    items = [make_item(f"cc{i}") for i in range(8)]
    outs = p.generate(items, model_ids=[SEEDREAM] * 8)

    assert [o.task.idempotency_key for o in outs] == [f"cc{i}" for i in range(8)]
    assert all(o.ok is True for o in outs)
    assert [o.file_name for o in outs] == [f"cc{i}__http_api.jpg" for i in range(8)]
    assert len(ark.post_calls) == 8 and len(dl.get_calls) == 8
    assert budget.spent == pytest.approx(8 * SEEDREAM_UNIT)
    # 缓存全部落账且带 model_id（回放口径不因并发而漏键）
    for i in range(8):
        entry = json.loads((tmp_path / "cache" / "http_api" / f"cc{i}.json").read_text(encoding="utf-8"))
        assert entry["model_id"] == SEEDREAM


def test_concurrent_mixed_download_failure_refunds_only_failures(tmp_path, monkeypatch) -> None:
    """混合失败（偶数 index 下载 404）：失败项 refund，总额只算成功项（4×单价）。

    响应由请求内容决定（prompt 里的 FAILDOWN 标记 → 专属下载 URL → 404），
    这样"哪个 index 失败"与线程调度无关，断言才有确定性。
    """
    monkeypatch.setenv("AP_CONCURRENCY", "4")

    def posts_fn(url, json):
        tag = "fail" if "FAILDOWN" in json["prompt"] else "ok"
        return FakeResp(200, payload={"data": [{"url": f"https://ark/{tag}.jpg"}],
                                      "usage": {"generated_images": 1}},
                        headers={"x-request-id": f"req-{tag}"})

    def gets_fn(url):
        if "/fail" in url:
            return FakeResp(404, text="not found")
        return FakeResp(200, content=FAKE_JPEG)

    ark = FakeClient(headers={"Authorization": "Bearer ark"}, posts_fn=posts_fn)
    dl = FakeClient(headers={}, gets_fn=gets_fn)
    budget = BudgetGuard(None)
    p, _ = build_provider(tmp_path, ark=ark, download=dl, budget=budget)

    items = [
        make_item(f"mix{i}",
                  prompt=("FAILDOWN broken download " if i % 2 == 0 else "")
                         + "a bright clinical photo of a woman")
        for i in range(8)
    ]
    outs = p.generate(items, model_ids=[SEEDREAM] * 8)

    assert [o.task.idempotency_key for o in outs] == [f"mix{i}" for i in range(8)]
    for i, o in enumerate(outs):
        if i % 2 == 0:
            assert o.ok is False and "下载失败" in (o.error or "")
        else:
            assert o.ok is True and o.file_name == f"mix{i}__http_api.jpg"
    assert len(ark.post_calls) == 8                      # 8 张都发了请求（预扣通过）
    assert len(dl.get_calls) == 8                        # 含 4 次失败的下载
    # 失败零扣费：4 个失败项的预扣全部退还，总额只等于 4 个成功项
    assert budget.spent == pytest.approx(4 * SEEDREAM_UNIT)
    for i in range(8):
        assert (tmp_path / "cache" / "http_api" / f"mix{i}.json").exists() is (i % 2 == 1)


def test_concurrent_fuse_admits_exactly_two_of_eight(tmp_path) -> None:
    """熔断并发版：预算恰好容纳 2 张，8 并发放 8 个 item → 恰 2 张发出请求。

    这是 try_charge 原子性的现场验证：8 个线程同时问闸，若判定与扣减不同步，
    在途请求总额会越过 80% 熔断线。断言：熔断的 6 张**根本没打 API**
    （post 计数==2）、error 带"预算熔断"、spent 恰好 2×单价。
    """
    guard = BudgetGuard(2.5 * SEEDREAM_UNIT)             # 熔断线 = 2×单价：恰好容纳 2 张
    ark = FakeClient(headers={"Authorization": "Bearer ark"},
                     posts_fn=lambda url, json: ark_post_ok())
    dl = FakeClient(headers={}, gets_fn=lambda url: FakeResp(200, content=FAKE_JPEG))
    p, _ = build_provider(tmp_path, ark=ark, download=dl, budget=guard, concurrency=8)
    assert p.concurrency == 8

    items = [make_item(f"cf{i}") for i in range(8)]
    outs = p.generate(items, model_ids=[SEEDREAM] * 8)

    assert [o.task.idempotency_key for o in outs] == [f"cf{i}" for i in range(8)]
    passed = [o for o in outs if o.ok]
    blocked = [o for o in outs if not o.ok]
    assert len(passed) == 2
    assert len(blocked) == 6
    assert all("预算熔断" in (o.error or "") for o in blocked)
    assert all(o.model_id == SEEDREAM for o in blocked)  # 失败也带档位归因
    assert len(ark.post_calls) == 2                      # 被熔断的 6 张：请求根本没发
    assert len(dl.get_calls) == 2
    assert guard.spent == pytest.approx(2 * SEEDREAM_UNIT)


def test_concurrency_env_clamped_to_1_16(tmp_path, monkeypatch) -> None:
    """concurrency 钳位：env 越界/垃圾值都不失控；显式参数优先且同样钳位。"""
    monkeypatch.setenv("AP_CONCURRENCY", "999")
    p, _ = build_provider(tmp_path)
    assert p.concurrency == 16                           # 上限 16：防"越大越好"的想当然

    monkeypatch.setenv("AP_CONCURRENCY", "-3")
    p, _ = build_provider(tmp_path)
    assert p.concurrency == 1

    monkeypatch.setenv("AP_CONCURRENCY", "not-a-number")
    p, _ = build_provider(tmp_path)
    assert p.concurrency == 4                            # 垃圾值回落默认 4

    monkeypatch.setenv("AP_CONCURRENCY", "8")
    p, _ = build_provider(tmp_path, concurrency=3)       # 显式参数优先于 env
    assert p.concurrency == 3
    p, _ = build_provider(tmp_path, concurrency=200)     # 显式参数也要钳位
    assert p.concurrency == 16


def test_concurrency_one_takes_serial_path_without_pool(tmp_path, monkeypatch) -> None:
    """AP_CONCURRENCY=1 → 直接走原串行路径：不建 ThreadPoolExecutor，结果照常。"""
    monkeypatch.setenv("AP_CONCURRENCY", "1")

    def no_pool(*a, **k):
        raise AssertionError("并发=1 不该建线程池")

    monkeypatch.setattr(http_api_mod, "ThreadPoolExecutor", no_pool)

    ark = FakeClient(headers={"Authorization": "Bearer ark"},
                     posts=[ark_post_ok(), ark_post_ok()])
    dl = FakeClient(headers={}, gets=[FakeResp(200, content=FAKE_JPEG),
                                      FakeResp(200, content=FAKE_JPEG)])
    p, _ = build_provider(tmp_path, ark=ark, download=dl)
    assert p.concurrency == 1

    outs = p.generate([make_item("s0"), make_item("s1")], model_ids=[SEEDREAM, SEEDREAM])
    assert [o.task.idempotency_key for o in outs] == ["s0", "s1"]
    assert all(o.ok is True for o in outs)
