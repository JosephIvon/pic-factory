"""成本护栏离线测试：RPM 令牌桶 + 预算熔断。不打真实 API、不睡真实时间。"""

from __future__ import annotations

import threading

import pytest

from aesthetics_pipeline import ledger
from aesthetics_pipeline.cost_guard import BudgetGuard, RateLimiter


class FakeClock:
    """假时钟：monotonic 读当前值，sleep 直接把指针往前推，验证限流窗口不靠真睡。"""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, dt: float) -> None:
        self.now += dt

    def advance(self, dt: float) -> None:
        self.now += dt


# ---------------- RateLimiter ----------------

def test_token_bucket_burst_then_throttle() -> None:
    """rpm=5：前 5 次瞬时通过，第 6 次必须等回补（约 60/5=12s）。"""
    clk = FakeClock()
    rl = RateLimiter(5, time_func=clk.monotonic, sleep_func=clk.sleep)
    for _ in range(5):
        rl.acquire()
    assert clk.now == 0.0          # 满桶突发不等待
    rl.acquire()                    # 第 6 次触发等待
    assert clk.now >= 10.0          # 至少跨过一个回补窗口


def test_token_bucket_window_total() -> None:
    """连续 10 次 acquire（rpm=5）：5 突发 + 5 排队，总耗时约 60s（一整分钟窗口）。"""
    clk = FakeClock()
    rl = RateLimiter(5, time_func=clk.monotonic, sleep_func=clk.sleep)
    for _ in range(10):
        rl.acquire()
    # 桶容量 5 免费，之后每次要等 12s 回补；总等待 ~= 5*12 = 60s。
    assert 55.0 <= clk.now <= 65.0


def test_token_bucket_rejects_bad_rpm() -> None:
    import pytest

    with pytest.raises(ValueError):
        RateLimiter(0)


# ---------------- BudgetGuard ----------------

def test_budget_unset_is_unlimited(monkeypatch) -> None:
    """未设 AP_BUDGET_CNY → 不限，兼容现有行为。"""
    monkeypatch.delenv("AP_BUDGET_CNY", raising=False)
    guard = BudgetGuard.from_env()
    assert guard.unlimited is True
    assert guard.can_spend(999.0) is True


def test_budget_fuse_at_80_percent() -> None:
    guard = BudgetGuard(100.0, spent=0.0)
    assert guard.can_spend(0.2) is True
    guard.charge(0.2)
    assert guard.spent == 0.2
    # 花到 80% 及以上 → 熔断
    guard2 = BudgetGuard(100.0, spent=80.0)
    assert guard2.can_spend(0.1) is False
    assert guard2.fuse_at == 80.0


def test_budget_from_env_reads_cap(monkeypatch) -> None:
    monkeypatch.setenv("AP_BUDGET_CNY", "50")
    guard = BudgetGuard.from_env(env={"AP_BUDGET_CNY": "50"})
    assert guard.budget == 50.0
    assert guard.fuse_at == 40.0
    assert guard.can_spend(1.0) is True


def test_budget_spent_seeded_from_ledger(tmp_path) -> None:
    """已花现金从台账 est_cost 累计（真实 sqlite，不 mock 数据库）。"""
    db = tmp_path / "t.db"
    with ledger.connect(db) as conn:
        ledger.record_asset(conn, {"idempotency_key": "a", "provider": "http_api", "est_cost": 10.0})
        ledger.record_asset(conn, {"idempotency_key": "b", "provider": "http_api", "est_cost": 25.0})
        guard = BudgetGuard.from_env(conn=conn, env={"AP_BUDGET_CNY": "50"})
    assert guard.spent == 35.0
    # 35 < 40（80% 线）→ 仍可花
    assert guard.can_spend(0.2) is True


# ---------------- try_charge / refund：并发预扣的原子性 ----------------

def test_try_charge_atomic_under_threads() -> None:
    """并发预扣的原子性：5 个线程同时 try_charge(0.49)，预算恰够 2 张 → 恰 2 个成功。

    这是并发跑批的地基：判定 + 扣减必须在同一把锁里一步完成，否则 5 个线程
    都看到 spent=0 全部放行，在途总额直接越过熔断线。
    （fuse_ratio 显式取 1.0 让 1.0 的预算恰好容纳 2×0.49；默认 0.8 下同构。）
    """
    guard = BudgetGuard(1.0, spent=0.0, fuse_ratio=1.0)   # 熔断线 = 1.0
    n_threads = 5
    barrier = threading.Barrier(n_threads)                 # 同时起跑，制造真并发
    results: list[bool] = []
    results_lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        ok = guard.try_charge(0.49)
        with results_lock:
            results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 2                        # 恰 2 个放行，不多不少
    assert guard.spent == pytest.approx(0.98)              # 只扣了放行的 2 笔
    # 被拒的 3 个没留任何扣费痕迹；再试仍是 False
    assert guard.try_charge(0.49) is False
    assert guard.spent == pytest.approx(0.98)
    # refund 后可再扣：预扣-退还闭环成立（失败零扣费的并发版）
    guard.refund(0.49)
    assert guard.spent == pytest.approx(0.49)
    assert guard.try_charge(0.49) is True
    assert guard.spent == pytest.approx(0.98)


def test_try_charge_allows_exact_fuse_line_only() -> None:
    """边界口径：允许恰好扣到熔断线（spent+cost == fuse_at 通过），越线一分就拒。"""
    guard = BudgetGuard(100.0, spent=79.0)                 # 熔断线 80.0
    assert guard.try_charge(1.0) is True                   # 80.0 == 80.0，恰好等于允许
    assert guard.spent == pytest.approx(80.0)
    assert guard.try_charge(0.01) is False                 # 80.01 > 80.0，拒
    assert guard.spent == pytest.approx(80.0)              # 被拒不涨基数


def test_refund_never_drops_below_zero() -> None:
    guard = BudgetGuard(100.0, spent=0.2)
    guard.refund(5.0)                                      # 退还会穿底的异常金额
    assert guard.spent == 0.0                              # 下限 0，不拖负


def test_try_charge_unlimited_behaves_like_old_charge() -> None:
    """unlimited（未设预算）行为不变：恒放行、spent 照记（供台账口径），refund 照退。"""
    monkeypatch_guard = BudgetGuard(None)
    assert monkeypatch_guard.unlimited is True
    assert monkeypatch_guard.try_charge(999.0) is True     # 不限 → 天文数字也放行
    assert monkeypatch_guard.spent == pytest.approx(999.0)
    assert monkeypatch_guard.try_charge(1.0) is True
    assert monkeypatch_guard.spent == pytest.approx(1000.0)
    monkeypatch_guard.refund(1000.0)
    assert monkeypatch_guard.spent == 0.0
