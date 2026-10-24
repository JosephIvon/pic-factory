"""声明式模型注册表：把「接一个模型」从写一个类降为填一行配置。

批判性重构（不照抄 tools/ab 的三个 provider 类）：

  tools/ab 里 ArkProvider / WanProvider / QwenProvider 各写一套 http、轮询、
  落盘、计费——三个类是三份会各自变异的拷贝。实测（MODEL-CHOICE §9）要同时用
  5 个模型，照这个写法就是 5 份拷贝。

  这里改成两层：
    * ModelSpec —— 一个模型一行，纯数据（endpoint、单价、RPM、参考图上限、
      同步/异步、负向词字段、水印默认、解析方式键）。**同族加模型 = 加一行。**
    * ADAPTERS  —— 每「族」一个 build/poll 函数（ark / wan）。**换族才加代码。**

  为什么保留 ADAPTERS 而不追求"一个通用解析器吃掉所有"：Ark 是同步、阿里是
  异步任务轮询，响应结构不同构。硬塞成万能配置会得到一个全是 if 的怪物，
  不如如实承认「族」这一抽象层。这是可辩护的边界，不是偷懒。

数据来源：MODEL-CHOICE.md §1/§9（本机实跑 50 张的实测），不是官方文档转抄。
单价/一致性会随版本漂移，改本表即可，不用重新生成图片。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ModelSpec:
    """一行 = 一个可调用的生图模型。纯配置，无行为。"""

    model_id: str                      # 传给 API 的真实 model 名
    family: Literal["ark", "wan"]      # 决定用哪个 ADAPTER（build_request/parse）
    display: str                       # 报告/台账里给人看的名字
    unit_cost_cny: float               # 实测单价（元/张），MODEL-CHOICE §9
    rpm: int                           # 每分钟请求上限，令牌桶用
    ref_max: int                       # 参考图上限（0=纯文生图）
    mode: Literal["sync", "async"]     # ark=同步 HTTP；阿里=创建任务→轮询
    supports_negative: bool            # 有独立 negative_prompt 字段？（Seedream 无）
    watermark_default_on: bool         # 默认是否带水印（Seedream 默认有；阿里默认裸图，需显式开）
    supports_seed: bool = False        # 4.0+ / Qwen 实测不支持位级复现
    default_size: str = "1536x2048"    # 3:4 画幅（本项目主画幅）
    notes: str = ""

    def __post_init__(self) -> None:
        # 交付红线：默认必须出合规水印。裸图（watermark_default_on=False）
        # 在适配层会被强制显式开启，这里只是把风险写进数据本身。
        if not self.watermark_default_on:
            object.__setattr__(self, "notes", (self.notes + " [适配层强制显式开水印]").strip())


# ---------------------------------------------------------------------------
# 实测模型库（数值来自 MODEL-CHOICE §9，50 张本机实跑；一致性列见 model_choice）
# ---------------------------------------------------------------------------

REGISTRY: dict[str, ModelSpec] = {
    s.model_id: s
    for s in [
        ModelSpec(
            model_id="doubao-seedream-4-0",
            family="ark",
            display="Seedream 4.0",
            unit_cost_cny=0.20,
            rpm=300,           # 未在公开页查到，保守按大值，实测后收紧
            ref_max=10,
            mode="sync",
            supports_negative=False,
            watermark_default_on=True,
            supports_seed=False,
            notes="场景/铺量主力；盲测 0/4 被认出；P50 10.2s",
        ),
        ModelSpec(
            model_id="doubao-seedream-5-0-pro",
            family="ark",
            display="Seedream 5.0 Pro",
            unit_cost_cny=0.30,
            rpm=300,
            ref_max=14,
            mode="sync",
            supports_negative=False,
            watermark_default_on=True,
            notes="一致性方差大（非遮挡低分 0.4566/0.5639），禁入人物主线，见 model_choice",
        ),
        ModelSpec(
            model_id="doubao-seedream-5-0-flash",
            family="ark",
            display="Seedream 5.0 Flash",
            unit_cost_cny=0.10,
            rpm=300,
            ref_max=14,
            mode="sync",
            supports_negative=False,
            watermark_default_on=True,
            notes="最低价档；人物禁已平反（取脸伪影，订正后均值 0.815/方差 0.043，见 model-sim-recompute.json）",
        ),
        ModelSpec(
            model_id="qwen-image-3.0-pro",
            family="wan",
            display="Qwen-Image 3.0 Pro",
            unit_cost_cny=0.49,
            rpm=5,             # 实测限流，需令牌桶硬限速
            ref_max=3,
            mode="async",
            supports_negative=True,
            watermark_default_on=False,
            notes="人物封面主力；订正后剔遮挡均值 0.913/非遮挡最低 0.859/剔遮挡标准差 0.033，全档下限最高（见 model-sim-recompute.json）",
        ),
        ModelSpec(
            model_id="qwen-image-3.0",
            family="wan",
            display="Qwen-Image 3.0",
            unit_cost_cny=0.18,
            rpm=20,
            ref_max=3,
            mode="async",
            supports_negative=True,
            watermark_default_on=False,
            notes="预算敏感铺量档；订正后均值 0.863（见 model-sim-recompute.json）",
        ),
    ]
}


def get(model_id: str) -> ModelSpec:
    try:
        return REGISTRY[model_id]
    except KeyError:
        raise KeyError(
            f"未注册的模型 {model_id!r}。可用：{sorted(REGISTRY)}。"
            f"同族加模型只需在此表加一行 ModelSpec。"
        ) from None
