"""预-mortem 对抗演练：设想评审者会怎么攻击这套流水线，负向测试把攻击钉死。

视角不是"我的功能对不对"，是"如果我是挑剔的评审者，我要挑一个能让这套叙事崩掉的
攻击"。每一个测试是一段中文 docstring 讲清楚"这是谁的什么攻击、拦住它靠哪个机制"。

全部离线：注入假 http 客户端，绝不打真实网络、绝不需要 key。假客户端的注入手法
照抄 tests/test_http_api.py，台账断言照抄 tests/test_model_ledger.py。

五条攻击主线：
  1. 中文原始标签塞进画面 prompt（撞"原始中文标签不入画面 prompt"不变量）
  2. 同一幂等键重复提交（撞幂等跳过 + 撞"复合主键不双扣"）
  3. 预算打满仍请求（撞"在发请求之前"熔断）
  4. 断网（撞"失败带 model_id / 零扣费 / 不落盘" + replay 兜底交付）
  5. 自由发挥：model_ids 长度与 items 不一致、ark 族计费张数≠1
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from aesthetics_pipeline import ledger
from aesthetics_pipeline.cost_guard import BudgetGuard
from aesthetics_pipeline.models import GenItem, TaskSpec
from aesthetics_pipeline.provider.cache import write_cache
from aesthetics_pipeline.provider.http_api import StandardApiProvider
from aesthetics_pipeline.provider.models_spec import get as get_spec
from aesthetics_pipeline.provider.replay import ReplayProvider
from aesthetics_pipeline.router import Router

# 与 test_http_api 同风格：假 JPEG（FFD8 魔数）与 ark 单价档位。
FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"jpegbody" * 16
SEEDREAM = "doubao-seedream-4-0"          # ark 族
# 单价从价目表读，别硬编码——models_spec 明说实测价会随版本漂移。
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
    """记录 post/get 调用；按队列返回预设响应（照抄 test_http_api 的注入手法）。"""

    def __init__(self, headers=None, posts=None, gets=None):
        self.headers = headers or {}
        self._posts = list(posts or [])
        self._gets = list(gets or [])
        self.post_calls: list[dict] = []
        self.get_calls: list[dict] = []

    def post(self, url, json=None):
        self.post_calls.append({"url": url, "json": json, "headers": dict(self.headers)})
        return self._posts.pop(0)

    def get(self, url):
        self.get_calls.append({"url": url, "headers": dict(self.headers)})
        return self._gets.pop(0)


class BoomClient:
    """断网模拟：post/get 一被调用就抛传输层异常（ConnectError 之类）。"""

    def __init__(self, exc: BaseException):
        self.exc = exc
        self.post_calls: list[dict] = []
        self.get_calls: list[dict] = []

    def post(self, url, json=None):
        self.post_calls.append({"url": url, "json": json})
        raise self.exc

    def get(self, url):
        self.get_calls.append({"url": url})
        raise self.exc


def make_item(key: str, *, refs: list[str] | None = None) -> GenItem:
    spec = TaskSpec(
        city="武汉", project="光子嫩肤", theme="夏日透亮", channel="小红书",
        character=None, aspect_ratio="3:4", resolution="1536x2048",
        idempotency_key=key, path="t2i" if not refs else "i2i",
    )
    return GenItem(
        task=spec, prompt="a bright clinical photo of a woman",
        negative_prompt="text, watermark", aspect_ratio="3:4",
        resolution="1536x2048", input_urls=refs or [], output_file=f"{key}.png",
    )


def ark_ok(billed: int = 1) -> FakeResp:
    """ark 同步成功响应；billed 控制 usage.generated_images，用于计费张数攻击。"""
    return FakeResp(
        200,
        payload={"data": [{"url": "https://ark/x.jpg", "size": "1536x2048"}],
                 "usage": {"generated_images": billed}},
        headers={"x-request-id": "req-ark-1"},
    )


def ark_provider(tmp_path, *, ark, budget=None, download=None):
    dl = download or FakeClient(headers={}, gets=[FakeResp(200, content=FAKE_JPEG)])
    p = StandardApiProvider(
        out_dir=tmp_path,
        ark_client=ark,
        # 断网攻击里 wan 不会被命中（选的是 ark 档），给个空客户端即可。
        wan_client=FakeClient(headers={}, posts=[], gets=[]),
        download_client=dl,
        budget_guard=budget if budget is not None else BudgetGuard(None),
        sleep_func=lambda *_: None,
    )
    return p, dl


# ===========================================================================
# 攻击 1：中文原始标签塞进画面 prompt
# ===========================================================================

def test_chinese_label_injected_into_visual_mapping_breaks_invariant(tmp_path) -> None:
    """攻击：改坏 matrix 的 visual 映射，把中文原始标签（"光子嫩肤"）直接
    拼进画面描述——这正是产生乱码招牌的成因。

    拦它靠 router.compile 的**编译期不变量校验**（router.py）：原始标签一旦出现在
    visual 里直接 ValueError，把"会画乱码招牌的图"挡在生成之前，而不是静默产出一张
    带乱码汉字黑板的成品。断言：被拒 + 编译产物根本不会存在（没返回）。"""
    matrix = {
        "projects": {"光子嫩肤": {"visual": "a clinic advertising 光子嫩肤 on the wall"}},
        "themes": {"夏日透亮": {"visual": "bright dewy skin"}},
    }
    scene = {"city_english": {"武汉": "Wuhan"}, "photography": "documentary",
             "negative": "text, watermark"}
    r = Router(matrix, scene)
    spec = TaskSpec(
        city="武汉", project="光子嫩肤", theme="夏日透亮", channel="小红书",
        character=None, aspect_ratio="3:4", resolution="1536x2048",
        idempotency_key="atk1a",
    )
    with pytest.raises(ValueError, match="不变量"):
        r.compile(spec)


def test_visual_prompt_field_is_ignored_and_chinese_stays_out(tmp_path) -> None:
    """攻击：TaskSpec 上有个 visual_prompt 字段，有人往 visual_prompt 里塞满中文标签原文
    （"夏日透亮 直出招牌"），指望它被当成正向提示词直接下发。

    拦它靠**编译只吃 matrix 映射、根本不读 spec.visual_prompt**（router.compile
    从 projects/themes/scene 拼 visual）。断言：编译产物不含该中文原文；且缺映射时
    给告警、降级为中性描述——中文标签永远留在文案层，进不了画面 prompt。"""
    matrix = {
        "projects": {"光子嫩肤": {"visual": "a bright modern clinic interior"}},
        "themes": {},   # 故意缺 夏日透亮 的 visual 映射，走告警降级分支
    }
    scene = {"city_english": {"武汉": "Wuhan"}, "photography": "documentary",
             "negative": "text, watermark"}
    r = Router(matrix, scene)
    spec = TaskSpec(
        city="武汉", project="光子嫩肤", theme="夏日透亮", channel="小红书",
        character=None, aspect_ratio="3:4", resolution="1536x2048",
        idempotency_key="atk1b",
    )
    spec.visual_prompt = "夏日透亮 直出招牌 中文字"   # 试图从这个字段夹带
    visual, _neg, warnings = r.compile(spec)
    assert "夏日透亮" not in visual
    assert "直出招牌" not in visual
    assert warnings, "缺视觉映射必须降级并给出告警"


# ===========================================================================
# 攻击 2：同一幂等键重复提交
# ===========================================================================

def test_idempotent_key_recorded_is_skipped_by_default(tmp_path) -> None:
    """攻击：用同一个幂等键重复提交，指望流水线忘了自己产过、再花一次钱。

    拦它靠 ledger.successful_keys（只认 local_path 非空的成功行，按 provider 过滤）
    配合 cli generate 的默认 skip（skip_existing=True）。断言：成功产出一张并落账
    （local_path 非空）后，successful_keys 会把这个键列进来——默认跳过会把它挡在重发
    之外。"""
    budget = BudgetGuard(100.0)
    ark = FakeClient(headers={"Authorization": "Bearer x"}, posts=[ark_ok()])
    p, _ = ark_provider(tmp_path, ark=ark, budget=budget)
    out = p.generate([make_item("dup-key")], model_ids=[SEEDREAM])[0]
    assert out.ok is True and out.local_path
    db = tmp_path / "t.db"
    with ledger.connect(db) as conn:
        ledger.record_asset(conn, {
            "idempotency_key": "dup-key", "provider": "http_api",
            "local_path": out.local_path, "model_id": SEEDREAM,
            "est_cost": budget.spent,
        })
        # 默认 skip 用的就是这个集合：成功落过盘的键必须在里面。
        done = ledger.successful_keys(conn, provider="http_api")
    assert "dup-key" in done


def test_duplicate_billed_successes_charge_exactly_once_each(tmp_path) -> None:
    """攻击：即便绕过默认 skip 强行重发同一键（用户说"我就是要 --no-skip 再来一遍"），
    预算扣减必须是"单价 × 成功张数"，一次都不许多扣。

    拦它靠 http_api 的计费闸门：只有**成功且计费断言相符**才 self.budget.charge，
    且只扣一次；ledger 的复合主键 (idempotency_key, provider) 用 ON CONFLICT DO UPDATE
    局部更新，同键同后端不会插出第二行、也就不会被 _spent_from_ledger 重复累加。"""
    unit = SEEDREAM_UNIT
    budget = BudgetGuard(100.0)
    ark = FakeClient(headers={"Authorization": "Bearer x"}, posts=[ark_ok(), ark_ok()])
    dl = FakeClient(headers={}, gets=[FakeResp(200, content=FAKE_JPEG),
                                      FakeResp(200, content=FAKE_JPEG)])
    p, _ = ark_provider(tmp_path, ark=ark, budget=budget, download=dl)

    o1 = p.generate([make_item("rebill")], model_ids=[SEEDREAM])[0]
    assert o1.ok is True and budget.spent == pytest.approx(unit)   # 1 张成功 = 扣 1 次

    o2 = p.generate([make_item("rebill")], model_ids=[SEEDREAM])[0]
    assert o2.ok is True
    # 两次都是真成功 → 恰好扣 2 次单价，没有"双扣"（不是 3 次、不是 4 次）。
    assert budget.spent == pytest.approx(2 * unit)

    db = tmp_path / "t.db"
    with ledger.connect(db) as conn:
        for est in (unit, unit):
            ledger.record_asset(conn, {
                "idempotency_key": "rebill", "provider": "http_api",
                "local_path": o2.local_path, "model_id": SEEDREAM, "est_cost": est,
            })
        rows = ledger.assets(conn, "idempotency_key = 'rebill'")
        # 同键同后端只有一行（复合主键 upsert），台账跨批复算不会把它算成两笔钱。
        assert len(rows) == 1
        assert BudgetGuard(100.0, spent=BudgetGuard._spent_from_ledger(conn)).spent == pytest.approx(unit)


# ===========================================================================
# 攻击 3：预算打满仍请求
# ===========================================================================

def test_budget_full_blocks_request_before_api_and_no_growth(tmp_path) -> None:
    """攻击：把预算烧到 80% 熔断线以上还继续 generate，指望它照样把请求打出去
    （然后被上游 429 打死 / 把整月配额烧穿）。

    拦它靠 http_api 的第 2 道闸门：`if not self.budget.can_spend(...)` 就 return，
    **位置在令牌桶与 _ark_call 之前**。断言：假 ark 客户端 post 调用次数 == 0（请求
    根本没发）、outcome.ok=False、error 写"预算熔断"、budget.spent 纹丝不动（拒绝
    的请求不扣费，更不会再涨）。"""
    budget = BudgetGuard(100.0, spent=85.0)          # 85 > 80% 熔断线
    ark = FakeClient(headers={"Authorization": "Bearer x"}, posts=[])
    p, _ = ark_provider(tmp_path, ark=ark, budget=budget)
    out = p.generate([make_item("fusekey")], model_ids=[SEEDREAM])[0]
    assert out.ok is False
    assert "预算熔断" in (out.error or "")
    assert ark.post_calls == []                        # 在发请求之前就拦下
    assert budget.spent == 85.0                        # 被拒 → 预算不增


# ===========================================================================
# 攻击 4：断网
# ===========================================================================

def test_network_outage_fails_cleanly_then_replay_delivers(tmp_path) -> None:
    """攻击：运行环境断网。这里验证两件事——(a) 失败会不会被记成"生成成功"或扣钱；
    (b) 断网后整条流水线是不是直接崩掉、没法演示。

    拦 (a) 靠 http_api 的异常兜底：**不计费、不落账、不落盘**，且失败 outcome 带
    model_id（供台账归因"这个模型失败了几次"）。
    兜 (b) 靠 cache.py 预写的成功缓存 + replay 路线：ReplayProvider 从 provider 命名
    空间的缓存里把**真实生成过的那张**离线交付出来，不改归属（source_provider 仍是
    原后端），流程/质检/台账都不打折。"""
    boom = BoomClient(httpx.ConnectError("connection refused"))
    budget = BudgetGuard(100.0)
    p, _ = ark_provider(tmp_path, ark=boom, budget=budget)
    out = p.generate([make_item("netkey")], model_ids=[SEEDREAM])[0]

    # --- 失败面：带 model_id 归因 / 零扣费 / 不落盘 / 不写缓存 ---
    assert out.ok is False
    assert out.model_id == SEEDREAM                     # 台账能查出这个模型失败
    assert budget.spent == 0.0                          # 断网不计费
    assert out.local_path is None
    assert list((tmp_path / "images").glob("netkey__*")) == []   # 磁盘上没有半成品
    assert (tmp_path / "cache" / "http_api" / "netkey.json").exists() is False

    # --- 兜底面：预写成功缓存（模拟断网前已真实生成过），replay 离线交付 ---
    cached_img = tmp_path / "images" / "netkey__http_api.jpg"
    cached_img.parent.mkdir(parents=True, exist_ok=True)
    cached_img.write_bytes(FAKE_JPEG)
    write_cache(
        tmp_path / "cache", key="netkey", provider="http_api",
        node_id="req-ark-cached", file_name=cached_img.name,
        local_path=str(cached_img), prompt="a bright clinical photo",
        path="t2i", seconds=1.0, portable=False,
    )
    replay = ReplayProvider(out_dir=tmp_path, prefer_provider="http_api")
    delivered = replay.generate([make_item("netkey")])[0]
    assert delivered.ok is True
    assert delivered.local_path and Path(delivered.local_path).exists()
    assert delivered.source_provider == "http_api"       # 不改归属，不冒充别的路线


# ===========================================================================
# 攻击 5：自由发挥——契约级攻击
# ===========================================================================

def test_model_ids_length_mismatch_raises_before_any_request(tmp_path) -> None:
    """攻击：同批混合不同模型时，调用方传 model_ids 与 items 长度不一致（例如 2 items
    配 1 个 model_id），指望它错位发请求、把 A 任务的图用 B 模型的档位计费。

    拦它靠 StandardApiProvider.generate 的入口契约校验：长度不等直接 ValueError，
    在任何一次 _ark_call / _wan_call 之前抛出。断言：抛错 + 假 ark 客户端一次请求都没发。"""
    ark = FakeClient(headers={"Authorization": "Bearer x"}, posts=[])
    p, _ = ark_provider(tmp_path, ark=ark)
    with pytest.raises(ValueError, match="长度必须与 items 一致"):
        p.generate([make_item("a"), make_item("b")], model_ids=[SEEDREAM])
    assert ark.post_calls == []                          # 契约破了绝不下发请求


def test_ark_billing_count_mismatch_rejected_not_charged(tmp_path) -> None:
    """攻击：ark 响应谎报 usage.generated_images=2（服务端多计费 / 解析错位），
    而画面只返回 1 张 URL。验证会不会照单付 2 倍钱。

    拦它靠 http_api 的**计费断言**（wan 的 n=4 陷阱同一条防线，这里从 ark 侧攻）：
    billed != 1 就拒绝落账、return，**不扣预算、不写缓存**。注意这与 test_http_api 里
    wan 用 results 长度触发的错费是不同代码路径（那条走 usage 字段），互补不重复。"""
    budget = BudgetGuard(100.0)
    ark = FakeClient(headers={"Authorization": "Bearer x"}, posts=[ark_ok(billed=2)])
    p, _ = ark_provider(tmp_path, ark=ark, budget=budget)
    out = p.generate([make_item("arkbill")], model_ids=[SEEDREAM])[0]
    assert out.ok is False
    assert "计费断言" in (out.error or "")
    assert budget.spent == 0.0                           # 拒绝落账 → 不扣预算
    assert (tmp_path / "cache" / "http_api" / "arkbill.json").exists() is False
