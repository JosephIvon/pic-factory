"""成本护栏：RPM 令牌桶 + 预算熔断。

批判性重构（不照抄 tools/ab）：A/B 实验台里没有这两样东西——它是"跑通就行"的
一次性脚本，本机手点 50 张不需要限速、不需要熔断。生产流水线 72 张 × 多档模型，
Qwen-3.0-Pro 实测 RPM 只有 5（MODEL-CHOICE §9），不打令牌桶会直接被 429 打死；
不设预算帽，一个失控的重试循环能把整月配额烧穿。

两个独立的闸：
  * RateLimiter —— 按模型族 RPM 节流（保护"别把上游打挂"）。
  * BudgetGuard —— 批次现金预算熔断（保护"别把自己烧穿"）。

都做成可注入时钟（`time_func`/`sleep_func`），这样离线测试能在假时钟里验证限流窗口，
而不是真睡 60 秒。
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Callable

from . import ledger

# 熔断阈值：花到预算的 80% 就停手，留 20% 余量给收尾/复核，绝不赌到 100%。
BUDGET_FUSE_RATIO = 0.8


class RateLimiter:
    """令牌桶：rpm 每分钟请求数，桶容量 = rpm，按 rpm/60 每秒回补。

    `acquire()` 阻塞到有余量。真实场景用 time.sleep；测试注入假 sleep 直接推时钟。
    """

    def __init__(
        self,
        rpm: int,
        *,
        time_func: Callable[[], float] = time.monotonic,
        sleep_func: Callable[[float], None] = time.sleep,
    ) -> None:
        if rpm <= 0:
            raise ValueError("rpm 必须为正")
        self.rpm = rpm
        self.capacity = float(rpm)
        self.tokens = float(rpm)          # 起步满桶：允许首分钟突发 rpm 次
        self.rate = rpm / 60.0            # 每秒回补
        self._time = time_func
        self._sleep = sleep_func
        self._last = time_func()
        # 并发跑批（http_api 线程池）下多线程共享同一桶：acquire 全程持锁，
        # 令"取令牌"串行化。测试注入的假 sleep 因此也不会和别人的回补窗口打架。
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._time()
        elapsed = max(now - self._last, 0.0)
        self._last = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

    def acquire(self, n: float = 1.0) -> None:
        with self._lock:
            while True:
                self._refill()
                if self.tokens >= n:
                    self.tokens -= n
                    return
                deficit = n - self.tokens
                wait = deficit / self.rate
                self._sleep(wait)


class BudgetGuard:
    """批次现金预算熔断。

    - 从 `AP_BUDGET_CNY` 读预算帽；未设 → 不限（兼容现有行为）。
    - 已花现金从台账 `est_cost` 累计（`ledger.assets()`），跨批复用。
    - `can_spend(next_cost)`：已花达到预算 80% 即返回 False，Provider 据此**不发请求**。
    - 批内成功后调 `charge(cost)` 累加，避免同一批把预算算漏。
    - 并发跑批专用：`try_charge(cost)` 原子预扣（锁内判定 + 扣减一步完成，
      多线程同时放行也不会让在途总额越过熔断线；允许恰好等于），
      失败分支用 `refund(cost)` 原样退回（下限 0，**失败零扣费**铁律不变）。
    """

    def __init__(
        self,
        budget_cny: float | None = None,
        *,
        spent: float = 0.0,
        fuse_ratio: float = BUDGET_FUSE_RATIO,
    ) -> None:
        self.budget = budget_cny
        self.spent = float(spent)
        self.fuse_ratio = fuse_ratio
        # spent 的一切读/写都在锁内：并发预扣-退还才有确定性。
        self._lock = threading.Lock()

    @classmethod
    def from_env(
        cls,
        conn: sqlite3.Connection | None = None,
        *,
        env: dict[str, str] | None = None,
    ) -> "BudgetGuard":
        """未设 AP_BUDGET_CNY → budget=None（不限，兼容旧行为）。"""
        env = env if env is not None else os.environ
        raw = env.get("AP_BUDGET_CNY")
        budget = float(raw) if raw not in (None, "") else None
        spent = cls._spent_from_ledger(conn) if conn is not None else 0.0
        return cls(budget, spent=spent)

    @staticmethod
    def _spent_from_ledger(conn: sqlite3.Connection) -> float:
        total = 0.0
        for row in ledger.assets(conn):
            cost = row.get("est_cost")
            if cost is not None:
                total += float(cost)
        return total

    @property
    def unlimited(self) -> bool:
        return self.budget is None

    @property
    def fuse_at(self) -> float | None:
        """熔断触发线（绝对金额）。"""
        return None if self.budget is None else self.budget * self.fuse_ratio

    def can_spend(self, next_cost: float = 0.0) -> bool:
        with self._lock:
            if self.budget is None:
                return True
            # 只看"已经花到的"是否越过 80%：越过就不再发新请求。
            return self.spent < self.budget * self.fuse_ratio

    def charge(self, cost: float) -> None:
        with self._lock:
            self.spent += float(cost)

    def try_charge(self, cost: float) -> bool:
        """原子预扣：锁内判定 + 扣减一步完成。

        通过条件（budget 为 None 时视为不限、恒通过）：spent + cost <= fuse_at。
        并发下 N 个线程同时预扣，放行的那批总额**不会越过熔断线**（恰好等于允许）。
        返回 False 时 spent 纹丝不动——被拒的请求既不扣费也不该发请求。
        """
        cost = float(cost)
        with self._lock:
            if self.budget is None:
                self.spent += cost
                return True
            if self.spent + cost <= self.budget * self.fuse_ratio:
                self.spent += cost
                return True
            return False

    def refund(self, cost: float) -> None:
        """失败退还：把预扣退回去（下限 0，防退回金额异常把 spent 拖负）。"""
        with self._lock:
            self.spent = max(0.0, self.spent - float(cost))
