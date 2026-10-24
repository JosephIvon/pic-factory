"""路由层：路径决策 + 提示词编译。

两条铁律（都来自实测，不是偏好）：

1. **提示词双层分离**：画面层 prompt 只放英文视觉描述；主题/项目的**原始中文文本
   绝不进画面 prompt**。实测中文主题会被模型直接画成黑板字、屏幕字、桌面名牌并
   产生乱码汉字（见 evidence/batch/）。
2. **一致性是路由问题**：需要"同一个人"的任务走图生图（带参考图），
   不需要就走文生图。纯靠文字描述锁不住脸——实测同一段描述每次换人。

编译时会对不变量做防御性校验：原始中文标签一旦出现在画面 prompt 里直接报错，
而不是安静地生成一张带乱码招牌的图。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import EVIDENCE_DIR, REPO_ROOT
from .models import TaskSpec

NEUTRAL_VISUAL = "natural daylight, relaxed candid expression"


class Router:
    def __init__(
        self,
        matrix: dict[str, Any],
        scene: dict[str, Any],
        evidence_dir: Path | None = None,
    ) -> None:
        self.matrix = matrix
        self.scene = scene
        self.evidence_dir = evidence_dir or EVIDENCE_DIR

    # ---------- 路径决策 ----------

    def decide_path(self, spec: TaskSpec) -> tuple[str, str | None]:
        """返回 (path, ref_path)。需要同一个人 -> i2i，否则 t2i。

        参考图路径按"仓库相对"和"evidence 目录相对"两种写法都能解析，
        避免配置里写不写 evidence/ 前缀就悄悄降级成文生图。
        """
        if spec.character and spec.ref_asset:
            for candidate in (
                REPO_ROOT / spec.ref_asset,
                self.evidence_dir / spec.ref_asset,
            ):
                if candidate.exists():
                    return "i2i", str(candidate)
        return "t2i", None

    # ---------- 提示词编译 ----------

    def compile(self, spec: TaskSpec) -> tuple[str, str, list[str]]:
        """编译 (visual_prompt, negative_prompt, warnings)。"""
        warnings: list[str] = []

        city = (self.scene.get("city_english") or {}).get(spec.city)
        if not city:
            city = "a modern city"
            warnings.append(f"城市「{spec.city}」缺少英文映射，画面 prompt 降级为泛化城市")

        project_visual = ((self.matrix.get("projects") or {}).get(spec.project) or {}).get("visual")
        if not project_visual:
            project_visual = "a bright modern aesthetic clinic interior"
            warnings.append(f"项目「{spec.project}」缺少 visual 映射，使用中性场景描述")

        theme_visual = ((self.matrix.get("themes") or {}).get(spec.theme) or {}).get("visual")
        if not theme_visual:
            theme_visual = NEUTRAL_VISUAL
            warnings.append(
                f"主题「{spec.theme}」缺少 visual 映射，使用中性画面描述；"
                "主题原文不进入画面 prompt"
            )

        identity = ""
        if spec.character:
            profile = (self.matrix.get("characters") or {}).get(spec.character) or {}
            identity = profile.get("identity_block") or ""

        photography = (self.scene.get("photography") or "").strip()
        # 身份块放最前面：CLIP 上下文只有 77 token，超出部分从**尾部**截断。
        # 身份描述一旦落在尾部就会被静默丢掉，人脸随即漂移（本地栈实测
        # 身份相似度 0.2836/0.1469 → 前置后 0.4284/0.2403）。
        # 主体前置也是扩散模型的通用做法。
        parts = [identity, f"in {city}", project_visual, theme_visual, photography]
        visual = ", ".join(p.strip() for p in parts if p and p.strip())

        negative = (self.scene.get("negative") or "").strip()
        if spec.character:
            profile = (self.matrix.get("characters") or {}).get(spec.character) or {}
            extra = (profile.get("negative_block") or "").strip()
            if extra:
                negative = f"{negative}, {extra}"

        # 不变量：原始中文标签不得进入画面 prompt（会渲染成乱码招牌）
        for label in (spec.project, spec.theme):
            if label and label in visual:
                raise ValueError(
                    f"提示词不变量被破坏：原始标签「{label}」出现在画面 prompt 中，"
                    "会被模型渲染成乱码文字。请只改 visual 映射，不要改原始文本。"
                )

        return visual, negative, warnings

    # ---------- 文案层 ----------

    def copy_angle(self, theme: str) -> str:
        """主题的文案角度（中文原文只允许出现在文案层）。"""
        return ((self.matrix.get("themes") or {}).get(theme) or {}).get("copy_angle") or theme
