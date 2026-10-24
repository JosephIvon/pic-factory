"""自动路由：按任务特征决定交给哪个后端。

核心要求是"由系统按任务自动调用模型"——所以"选哪个后端"这一步不能靠人敲
`--provider`。这一模块把选型做成**规则表**：有序、先命中先用，每条决策都带
`rule_id` + 理由，落台账，可逐条追溯"这张图为什么走了这条线"。

规则的依据全部是**实测**，不是偏好：

  * 强人物一致 → API：实测 API 图生图的身份保持高于本地栈（口径与数字见
    测试 `test_model_choice` 与 `docs/MODEL-CHOICE.md`；台账每行都记了
    `face_similarity`，可逐张核对）。
  * 泛化人物 / 铺量 → 本地栈：边际成本≈电费，数据不出网。
  * 不出网 → 本地栈：合规优先，压过质量优先（一致性由后置换脸兜底，
    实测两条路线换脸后都收敛到同一区间，差异被抹平）。
  * 需要中文排版 → API：**中文一律不在画面里渲染**（实测 API 与本地栈都会把
    中文画成黑板字/乱码招牌），由后处理合成；这类内容对"画面干净度"要求最高，
    所以走质量更高的一侧。

> 规则文本里**不写死数字**：数字会随模型/口径变化，写死就会与台账对不上。
> 理由只说"往哪边走、依据是什么口径"，数字去台账和测试/文档查。

优先级：**合规 > 排版 > 一致性 > 成本**。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Any, Callable

import os

from .config import MATRIX_CLI
from .models import TaskSpec

MATRIX = "matrix"
LOCAL = "local"
# 可交付的标准云后端（火山方舟 Seedream / 阿里百炼），对应工厂键 http_api。
# matrix 是 MiniMax 连接器，只作演示参照、不可交付；配了
# 标准 API key 后，云端决策默认改走 http_api。二者同属"云端"这一层，规则表
# 仍指向云端语义，不各写一份规则。值必须与 provider.build_provider 的键一致。
API = "http_api"
_CLOUD_TIER = {API, MATRIX}


@dataclass(frozen=True)
class RouteDecision:
    """一次路由决策。落台账的 route_rule_id / route_reason 就是它。"""

    provider: str
    rule_id: str
    reason: str
    est_cost: float | None = None


@dataclass(frozen=True)
class Rule:
    """规则表的一行。"""

    rule_id: str
    provider: str
    reason: str
    matches: Callable[[TaskSpec], bool]


def _no_cloud(spec: TaskSpec) -> bool:
    return spec.no_cloud


def _needs_text_overlay(spec: TaskSpec) -> bool:
    return spec.needs_text_overlay


def _strong_identity(spec: TaskSpec) -> bool:
    return bool(spec.character)


# 规则表：**顺序即优先级**，先命中先用。
RULES: tuple[Rule, ...] = (
    Rule(
        "R1_NO_CLOUD",
        LOCAL,
        "不出网优先：项目标记 no_cloud，交本地栈；一致性由后置换脸兜底",
        _no_cloud,
    ),
    Rule(
        "R2_TEXT_OVERLAY",
        MATRIX,
        "要带中文排版 → 画面干净度要求最高，走 API；中文由后处理合成、不进画面",
        _needs_text_overlay,
    ),
    Rule(
        "R3_STRONG_IDENTITY",
        MATRIX,
        "强人物一致：API 身份保持实测优于本地栈（数据见台账 face_similarity 列与模型选型文档）",
        _strong_identity,
    ),
    Rule(
        "R4_VOLUME_LOCAL",
        LOCAL,
        "泛化人物 / 铺量变体：本地栈边际成本≈电费、数据不出网",
        lambda spec: True,
    ),
)


def backend_available(name: str) -> tuple[bool, str]:
    """后端能不能用。返回 (可用, 原因)。缺失时给出可执行的提示。"""
    if name == LOCAL:
        from .config import local_python, local_model_dir

        try:
            local_python()
        except RuntimeError as exc:
            return False, str(exc)
        try:
            local_model_dir()
        except RuntimeError as exc:
            return False, str(exc)
        return True, "本地栈就绪"
    if name == MATRIX:
        if shutil.which(MATRIX_CLI) or shutil.which(f"{MATRIX_CLI}.cmd"):
            return True, "API 后端就绪"
        return False, (
            f"找不到生图 CLI {MATRIX_CLI!r}，请确认它在 PATH 上，或用 AP_MATRIX_CLI 指定绝对路径。"
        )
    if name == API:
        if os.environ.get("AP_ARK_KEY") or os.environ.get("AP_DASHSCOPE_KEY"):
            return True, "标准 API 后端就绪（Ark/Wan key 已配置）"
        return False, (
            "未配置 AP_ARK_KEY / AP_DASHSCOPE_KEY，标准 API 后端不可用；"
            "交付环境必须配标准 key（matrix 连接器不可交付）。"
        )
    return False, f"未知后端 {name!r}"


def _cloud_target(spec: TaskSpec, check: Callable[[str], tuple[bool, str]]) -> tuple[str, str]:
    """云端这一层内部：用可交付的 api（key 已配）还是参照的 matrix（连接器）。

    返回 (后端名, 附加说明)。顺序是**交付优先**：
      配了标准 key 且可用 → api（客户账号能跑的那条路）。
      否则 → matrix（本机演示参照；理由里写明它是连接器、不可交付）。
    """
    has_key = bool(os.environ.get("AP_ARK_KEY") or os.environ.get("AP_DASHSCOPE_KEY"))
    if has_key and os.environ.get("AP_PREFER_MATRIX") != "1":
        ok_api, why_api = check(API)
        if ok_api:
            return API, f"（{why_api}，matrix 降级为参照）"
    return MATRIX, ""


def decide_provider(
    spec: TaskSpec,
    *,
    no_cloud: bool = False,
    force: str | None = None,
    available: Callable[[str], tuple[bool, str]] | None = None,
    require_available: bool = True,
) -> RouteDecision:
    """决定这条任务交给哪个后端。

    force: "matrix" / "local" —— 人工覆盖（演示或排障用），决策理由会写明是覆盖。
    no_cloud: 全局不出网开关（合规要求），优先级高于规则表。
    require_available: False 时**不做可用性检查与回退**，只按规则表给决策。
        面板预览用这个：后端没装也要能看见"这张本该走哪条线、命中哪条规则"，
        而不是整页报错。
    """
    check = available or backend_available

    if no_cloud:
        decision = RouteDecision(
            provider=LOCAL,
            rule_id="R0_NO_CLOUD_GLOBAL",
            reason="全局 --no-cloud：本次运行不出网，全部交给本地栈",
        )
    elif force:
        decision = RouteDecision(
            provider=force,
            rule_id="R_MANUAL_OVERRIDE",
            reason=f"人工指定后端 {force}",
        )
    else:
        decision = None
        for rule in RULES:
            if rule.matches(spec):
                decision = RouteDecision(
                    provider=rule.provider,
                    rule_id=rule.rule_id,
                    reason=rule.reason,
                )
                break
        if decision is None:  # pragma: no cover —— R4 是兜底规则，永远命中
            decision = RouteDecision(LOCAL, "R_FALLBACK_LOCAL", "默认交给本地栈")

    # 云端层内部解析：命中"走云"的规则（matrix 是历史名，语义=云端）时，
    # 交付优先改走标准 api，理由里写明是"可交付后端"，不是静默替换。
    if decision.provider == MATRIX:
        target, note = _cloud_target(spec, check)
        if target != MATRIX:
            decision = RouteDecision(
                provider=target,
                rule_id=decision.rule_id,
                reason=decision.reason + f"；交付优先：{note}",
                est_cost=decision.est_cost,
            )

    if not require_available:
        return decision

    ok, why = check(decision.provider)
    if ok:
        return decision

    # 选中的后端不可用 → 回退到**另一层**（云端↔本地），并写明原因（不能静默换掉）
    other = LOCAL if decision.provider in _CLOUD_TIER else _cloud_target(spec, check)[0]
    ok_other, why_other = check(other)
    if not ok_other:
        raise RuntimeError(
            f"两个后端都不可用：{decision.provider}（{why}）；{other}（{why_other}）"
        )
    return RouteDecision(
        provider=other,
        rule_id="R_FALLBACK_UNAVAILABLE",
        reason=f"原定后端 {decision.provider} 不可用（{why}），回退到 {other}",
    )


def summarize(decisions: list[RouteDecision]) -> dict[str, Any]:
    """按后端汇总决策分布（报表与面板用）。"""
    summary: dict[str, Any] = {}
    for d in decisions:
        slot = summary.setdefault(d.provider, {"count": 0, "rules": {}})
        slot["count"] += 1
        slot["rules"][d.rule_id] = slot["rules"].get(d.rule_id, 0) + 1
    return summary
