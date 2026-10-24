"""任务矩阵展开 + 幂等键。

城市 × 项目 × 主题 × 渠道 × 角色 → TaskSpec 列表。
幂等键 = 上述维度 + 提示词版本 的摘要；同键已成功产出的任务会被跳过，
所以同一条命令可以反复重跑而不会重复花钱。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from .config import DATA_DIR, PROMPT_DIR
from .models import TaskSpec
from .pack import try_load_pack

DEFAULT_MATRIX = DATA_DIR / "tasks.yaml"
DEFAULT_SCENE = PROMPT_DIR / "scene.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def pack_matrix() -> dict[str, Any]:
    """当前域包的矩阵；未启用域包（AP_PACK 未设）时退回 data/tasks.yaml。

    兼容优先：未设 AP_PACK 时读的文件与引入域包之前**完全一样**，
    现有部署与测试不需要任何改动。
    """
    pack = try_load_pack()
    return pack.matrix if pack is not None else load_yaml(DEFAULT_MATRIX)


def pack_scene() -> dict[str, Any]:
    pack = try_load_pack()
    return pack.scene if pack is not None else load_yaml(DEFAULT_SCENE)


def idempotency_key(parts: list[str]) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def expand(
    matrix_path: Path | None = None,
    scene_path: Path | None = None,
) -> list[TaskSpec]:
    """展开任务矩阵。返回顺序稳定，便于复现。

    显式传路径 → 按路径读（测试用）；都不传 → 域包优先，否则旧路径。
    """
    if matrix_path is None and scene_path is None:
        matrix, scene = pack_matrix(), pack_scene()
    else:
        matrix = load_yaml(matrix_path or DEFAULT_MATRIX)
        scene = load_yaml(scene_path or DEFAULT_SCENE)
    version = str(matrix.get("version", "0"))

    cities: list[str] = list(matrix.get("cities") or [])
    projects: dict[str, Any] = matrix.get("projects") or {}
    themes: dict[str, Any] = matrix.get("themes") or {}
    channels: dict[str, Any] = matrix.get("channels") or {}
    characters: dict[str, Any] = matrix.get("characters") or {}

    if not (cities and projects and themes and channels):
        raise ValueError("任务矩阵缺少 cities/projects/themes/channels 任一项")

    # 角色维度：项目可以显式声明要不要固定人物。
    #   use_character: true  -> 需要"同一个人"跨城市/跨主题复用 -> 图生图
    #   use_character: false -> 机构/价格类内容，人物泛化 -> 文生图
    #   不声明              -> 默认用全部角色
    def characters_for(project_cfg: dict[str, Any]) -> list[str | None]:
        if "use_character" in project_cfg:
            return list(characters.keys()) if project_cfg["use_character"] else [None]
        return list(characters.keys()) or [None]

    specs: list[TaskSpec] = []
    for city in cities:
        for project, project_cfg in projects.items():
            for theme, theme_cfg in themes.items():
                for channel, channel_cfg in channels.items():
                    for character in characters_for(project_cfg):
                        key = idempotency_key(
                            [city, project, theme, channel, str(character), version]
                        )
                        specs.append(
                            TaskSpec(
                                city=city,
                                project=project,
                                theme=theme,
                                channel=channel,
                                character=character,
                                aspect_ratio=str(channel_cfg.get("aspect_ratio", "3:4")),
                                resolution=str(channel_cfg.get("resolution", "2K")),
                                idempotency_key=key,
                                ref_asset=(characters.get(character) or {}).get("ref"),
                                # 路由特征：项目级与主题级都可以声明，项目级优先为"或"关系
                                needs_text_overlay=bool(
                                    theme_cfg.get("needs_text_overlay")
                                    or project_cfg.get("needs_text_overlay")
                                ),
                                no_cloud=bool(
                                    project_cfg.get("no_cloud") or theme_cfg.get("no_cloud")
                                ),
                            )
                        )
    return specs


def filter_specs(
    specs: list[TaskSpec],
    city: str | None = None,
    project: str | None = None,
    theme: str | None = None,
    limit: int | None = None,
) -> list[TaskSpec]:
    """按维度筛任务。CLI 与 Web 面板共用，避免两边的筛选口径不一致。"""
    out = specs
    if city:
        out = [s for s in out if s.city == city]
    if project:
        out = [s for s in out if s.project == project]
    if theme:
        out = [s for s in out if s.theme == theme]
    if limit:
        out = out[:limit]
    return out


def character_profiles(matrix_path: Path | None = None) -> dict[str, Any]:
    matrix = load_yaml(matrix_path) if matrix_path is not None else pack_matrix()
    return matrix.get("characters") or {}


def scene_config(scene_path: Path | None = None) -> dict[str, Any]:
    return load_yaml(scene_path) if scene_path is not None else pack_scene()
