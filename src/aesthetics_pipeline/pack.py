"""域包（Domain Pack）：把"行业知识"从代码里挪到配置目录。

**为什么要有这一层**：核心流水线（路由 / 闸门 / 台账 / 成本 / 采纳）是通用的；
医美只是第一个行业。没有域包时，换一个行业要改 `copy.py` 的话术、`tasks.yaml`
的矩阵、`qc` 的规则——改代码就意味着一轮回归测试和一次发布。

域包把这三样变成数据：

    packs/<name>/
    ├── pack.yaml          # 清单：名字、版本、matrix/scene/copy/compliance 的路径
    ├── matrix.yaml        # 城市 × 项目 × 主题 × 渠道 × 角色（可在包外，见 manifest）
    ├── scene.yaml         # 画面层提示词（摄影语汇 + negative）
    ├── copy.yaml          # 文案模板 + 禁用词表
    ├── compliance.yaml    # 合规要求（免责声明、AI 标识、禁止构图）
    └── assets/            # 角色卡 / 参考图（可选）

**兼容策略（重要）**：`AP_PACK` 未设置时，系统仍按旧路径读 `data/tasks.yaml` +
`prompts/scene.yaml`，行为与引入域包之前逐字节一致。这不是"以后再说"，是有意的
平滑迁移：现有测试与部署不需要任何改动就能跑；设了 `AP_PACK` 才切换。

用法：

    AP_PACK=medical-aesthetics aesthetics plan --limit 5
    aesthetics packs          # 列出可用域包
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import REPO_ROOT

DEFAULT_PACK_NAME = "medical-aesthetics"


def packs_root() -> Path:
    """域包根目录。AP_PACKS_DIR 可指向外部目录（便于客户自带行业包）。"""
    return Path(os.environ.get("AP_PACKS_DIR") or (REPO_ROOT / "packs"))


def active_pack_name() -> str | None:
    """当前生效的域包名。未设置 AP_PACK → None（走旧路径，保持兼容）。"""
    raw = os.environ.get("AP_PACK")
    if raw is None:
        return None
    name = raw.strip()
    return name or None


def available_packs() -> list[str]:
    root = packs_root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "pack.yaml").exists())


@dataclass(frozen=True)
class DomainPack:
    """一个已加载的域包。所有字段都来自磁盘，不改代码就能换行业。"""

    name: str
    root: Path
    manifest: dict[str, Any]
    matrix: dict[str, Any]
    scene: dict[str, Any]
    copy_cfg: dict[str, Any] = field(default_factory=dict)
    compliance: dict[str, Any] = field(default_factory=dict)
    # 角色授权登记表（licenses.yaml）：商用红线，发布前检查读取。
    licenses: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return str(self.matrix.get("version") or self.manifest.get("version") or "0")

    @property
    def assets_dir(self) -> Path | None:
        raw = self.manifest.get("assets")
        if not raw:
            return None
        p = (self.root / str(raw)).resolve()
        return p if p.is_dir() else None


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"域包文件不存在: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve(root: Path, raw: str | None) -> Path | None:
    """清单里的路径：相对包目录解析；`..` 可以引用仓库内旧文件（医美包就这么做）。"""
    if not raw:
        return None
    p = Path(raw)
    return (p if p.is_absolute() else (root / p)).resolve()


def load_pack(name: str | None = None) -> DomainPack:
    """加载域包。name 为空时读 AP_PACK；都没有则报错（调用方应先判 active_pack_name）。"""
    pack_name = name or active_pack_name()
    if not pack_name:
        raise RuntimeError(
            "没有指定域包：请设置 AP_PACK=<包名>，或显式传 name。"
            f"可用包：{available_packs() or '（无）'}"
        )
    root = packs_root() / pack_name
    manifest_path = root / "pack.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"找不到域包 {pack_name!r}：{manifest_path} 不存在。"
            f"可用包：{available_packs() or '（无）'}"
        )
    manifest = _read_yaml(manifest_path)

    matrix_path = _resolve(root, manifest.get("matrix"))
    scene_path = _resolve(root, manifest.get("scene"))
    if matrix_path is None or scene_path is None:
        raise ValueError(f"域包 {pack_name!r} 的 pack.yaml 必须声明 matrix 与 scene")

    matrix = _read_yaml(matrix_path)
    scene = _read_yaml(scene_path)
    _validate_matrix(pack_name, matrix)

    copy_path = _resolve(root, manifest.get("copy"))
    compliance_path = _resolve(root, manifest.get("compliance"))
    # 授权表：清单可显式指定，否则默认包内 assets/licenses.yaml（有就用，没有就空）
    licenses_path = _resolve(root, manifest.get("licenses") or "assets/licenses.yaml")

    copy_cfg = _read_yaml(copy_path) if copy_path else {}
    _validate_copy(pack_name, copy_cfg)

    return DomainPack(
        name=pack_name,
        root=root,
        manifest=manifest,
        matrix=matrix,
        scene=scene,
        copy_cfg=copy_cfg,
        compliance=_read_yaml(compliance_path) if compliance_path else {},
        licenses=_read_yaml(licenses_path) if licenses_path and licenses_path.exists() else {},
    )


def _validate_matrix(pack_name: str, matrix: dict[str, Any]) -> None:
    """早失败：缺维度时给出包名与缺失项，而不是等到 expand 里 KeyError。"""
    missing = [k for k in ("cities", "projects", "themes", "channels") if not matrix.get(k)]
    if missing:
        raise ValueError(f"域包 {pack_name!r} 的 matrix 缺少字段: {', '.join(missing)}")


_ALLOWED_PLACEHOLDERS = {"city", "project", "theme", "copy_angle", "channel"}


def _validate_copy(pack_name: str, cfg: dict[str, Any]) -> None:
    """文案模板的占位符白名单。配置期报错，胜过出文案时静默少字。"""
    import string

    for tpl_name, tpl in (cfg.get("templates") or {}).items():
        if not isinstance(tpl, dict):
            raise ValueError(f"域包 {pack_name!r} 的 copy.templates.{tpl_name} 必须是对象")
        for field_name in ("title", "body"):
            text = str(tpl.get(field_name) or "")
            for _literal, placeholder, _spec, _conv in string.Formatter().parse(text):
                if placeholder is None:
                    continue
                base = placeholder.split(".")[0].split("[")[0]
                if base not in _ALLOWED_PLACEHOLDERS:
                    raise ValueError(
                        f"域包 {pack_name!r} 的模板 {tpl_name}.{field_name} 用了未知占位符 "
                        f"{{{placeholder}}}；允许: {sorted(_ALLOWED_PLACEHOLDERS)}"
                    )


def try_load_pack(name: str | None = None) -> DomainPack | None:
    """未配置域包时返回 None（调用方回退到旧路径），配置了但加载失败则抛错。"""
    pack_name = name or active_pack_name()
    if not pack_name:
        return None
    return load_pack(pack_name)
