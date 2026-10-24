"""数据模型。

全部用 dataclass + 显式字段，避免 ORM 依赖；落库由 ledger 负责。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class CharacterProfile:
    """虚拟人档案。

    identity_block 是冻结的描述文本，在所有含该角色的任务里原样复用，
    避免"每次描述略有出入"导致的人脸漂移。
    """

    name: str
    identity_block: str
    ref_path: str | None = None
    negative_block: str = ""


@dataclass
class TaskSpec:
    """一个生成任务：城市 × 项目 × 主题 × 渠道 × 角色。"""

    city: str
    project: str
    theme: str
    channel: str
    character: str | None
    aspect_ratio: str
    resolution: str
    idempotency_key: str
    visual_prompt: str = ""
    negative_prompt: str = ""
    path: str = "t2i"  # t2i | i2i
    ref_asset: str | None = None
    warnings: list[str] = field(default_factory=list)
    # 路由特征：决定这条任务交给哪个后端（见 routing.py 的规则表）。
    # needs_text_overlay：成品要带中文排版（中文由后处理合成，绝不进画面）。
    # no_cloud：这条内容不允许出网（合规要求，压过质量优先）。
    needs_text_overlay: bool = False
    no_cloud: bool = False

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenItem:
    """一次生成请求（发往 provider 的最小单元）。"""

    task: TaskSpec
    prompt: str
    negative_prompt: str
    aspect_ratio: str
    resolution: str
    input_urls: list[str] = field(default_factory=list)
    output_file: str = ""


@dataclass
class GenOutcome:
    """单张生成的结果。"""

    task: TaskSpec
    ok: bool
    node_id: str | None = None
    file_name: str | None = None
    local_path: str | None = None
    error: str | None = None
    seconds: float = 0.0
    # 真正生成这张图的后端。回放时沿用原后端——回放是**传输方式**，
    # 不是新的生成后端，否则同一张图会在台账里被算两遍，污染成本/吞吐口径。
    source_provider: str | None = None
    # HTTP API 后端实际使用的模型档位（由 model_choice 选定、provider 解析后回带）。
    # 失败路径也要带上：这样"这个模型失败了 3 次"能从台账直接查出来。
    model_id: str | None = None


@dataclass
class QcVerdict:
    """质检判定。

    强制规则命中 -> rejected（出厂拦截）；VLM 分数只做 soft signal。
    """

    rejected: bool
    reasons: list[str] = field(default_factory=list)
    rules: dict[str, Any] = field(default_factory=dict)
    soft_scores: dict[str, float] = field(default_factory=dict)
    face_similarity: float | None = None
    text_found: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        return asdict(self)
