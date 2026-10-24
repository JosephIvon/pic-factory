"""模型档位选择：上游 routing 解决「API 还是本地栈」，这里解决「哪个模型」。

两个正交问题必须分开：
  * routing.py —— 后端维度（matrix/local），命中规则落 route_rule_id。
  * model_choice.py —— 模型维度（Seedream/Qwen 档位），命中落 model_id + 理由。
硬塞进一张表会让「后端规则」被「模型库扩张」拖着改。

优先级：**任务类型 > 一致性下限 > 成本**。

一致性数字的**口径**（2026-10-24 订正，重要）：
  早期引用的是实验台 YuNet 管线（按框面积取脸）的数字，外部审计发现该取脸规则
  在多脸图上会把分算到错误的人头上——Seedream 5.0-Flash 的"−0.005 换脸样本"
  就是伪影。本表数字已改用**生产度量**（insightface buffalo_l + BGR + gallery
  选脸=与参考照最相似的一对）在原始 50 张上一笔笔复算
  （docs/ab-evidence/model-sim-recompute.json）。结论变化：
    * 5.0-Flash 平反：均值 0.815、非遮挡最低 0.733、剔遮挡标准差 0.043 → 解除人物禁令。
    * 5.0-Pro 维持禁令：低分（0.4566/0.5639）与遮挡无关，剔遮挡标准差 0.171 仍是最大。
    * 排序（剔遮挡均值）：Qwen-Pro 0.913 > Qwen-3.0 0.863 > 5.0-Flash 0.815 >
      Seedream-4.0 0.792 > 5.0-Pro 0.751。人物主线的头名没变（Qwen-Pro 仍是唯一
      "最低分也过线"的档，0.913 / 非遮挡最低 0.859），但 5.0-Flash 从末位升到第三。
  仍是单次观测（每模型 10 张 ×1 轮，跨次方差可达 0.2）；要 3 次复核才能"排名"，
  本表只用来"排除"。
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import TaskSpec

# 注意：provider.models_spec 通过 provider 包的 __init__ 会拉起 http_api，
# 而 http_api 又依赖本模块的 choose_model——顶层直接 import 会成环。
# 故这里在函数内延迟导入（models_spec 只在选档时用到）。

# 人物一致性下限：低于此值的模型不进"需要同一人"的任务。
# 这是从最低分反推的工程红线，不是标定阈值（标定阈值在 qc/face 里，管放行）。
IDENTITY_FLOOR = 0.75

# 生产度量复算的"剔除遮挡均值"（model-sim-recompute.json，见模块 docstring）。
# 路由用它做一致性下限判断，单点、待 3 次复核——所以只用来"排除"，不用来"排名争第一"。
_MEAN_SIM: dict[str, float] = {
    "qwen-image-3.0-pro": 0.913,
    "qwen-image-3.0": 0.863,
    "doubao-seedream-5-0-flash": 0.815,
    "doubao-seedream-4-0": 0.792,
    "doubao-seedream-5-0-pro": 0.751,
}

# 人物主线黑名单：非遮挡低分与方差独立成立（即便均值过线也不给）。
# 5.0-Flash 2026-10-24 平反移出——它的"换脸证据 −0.005"被证实是实验台取脸伪影。
_PERSONA_BANNED = {"doubao-seedream-5-0-pro"}


@dataclass(frozen=True)
class ModelDecision:
    model_id: str
    rule_id: str
    reason: str


def choose_model(spec: TaskSpec, *, allow_persona_risk: bool = False) -> ModelDecision:
    """按任务特征选模型档位。只提议，不生成——真正的调用在 provider 层。"""
    from .provider.models_spec import get as get_spec

    is_character = bool(spec.character and spec.ref_asset)

    # M1 纯场景（无人物）：成本优先，Seedream 4.0 够且盲测 0/4 被认出。
    # 注意：这里**不降到 5.0-Flash(¥0.10)**——省一毛钱但盲测真实感 4.0=0/4 优于 Flash=1/4，
    # 医美对真实感敏感；也不引入 GLM-Image/火山通用(¥0.1) 低价档，因其 endpoint+真实感未实测。
    # 详见 docs/COST-MODEL.md §3。有实测低价档后再动此分支。
    if not is_character:
        if spec.path == "t2i" and not spec.character:
            return ModelDecision(
                "doubao-seedream-4-0", "M1_SCENE_CHEAP",
                "无人物纯场景：Seedream 4.0 最省且真实感盲测 0/4（不降 Flash，真实感优先）",
            )

    # M2 人物封面/主图（要强一致性）：只有均值过下限、且不在换脸黑名单的模型可选。
    if is_character:
        if not allow_persona_risk:
            eligible = [
                m for m, s in _MEAN_SIM.items()
                if s >= IDENTITY_FLOOR and m not in _PERSONA_BANNED
            ]
            # 候选里选最便宜的一档（订正后 5.0-Flash ¥0.10 已过下限且不在黑名单），
            # 预算敏感；封面若人审判漂，重试链会升档到更贵的合格档。
            eligible.sort(key=lambda m: (get_spec(m).unit_cost_cny, -_MEAN_SIM[m]))
            if eligible:
                chosen = eligible[0]
                return ModelDecision(
                    chosen, "M2_PERSONA_SAFE",
                    f"人物任务：只从均值≥{IDENTITY_FLOOR}且非换脸黑名单里选，"
                    f"取其中最低价 {get_spec(chosen).display}",
                )
        # 兜底：一致性下限卡死时，直接上实测最稳的 Pro（宁可贵，不可漂）。
        return ModelDecision(
            "qwen-image-3.0-pro", "M2_PERSONA_PRO_FALLBACK",
            "人物任务无合格低价档 / 允许高风险前未达标：升 Qwen-3.0-Pro（全档实测方差最低、下限最高）",
        )

    # M3 其余（有角色描述但无参考图的 t2i 铺量）：本就不该锁脸，走低成本。
    return ModelDecision(
        "doubao-seedream-4-0", "M3_VOLUME_CHEAP",
        "铺量变体、无需锁脸：Seedream 4.0 成本优先",
    )
