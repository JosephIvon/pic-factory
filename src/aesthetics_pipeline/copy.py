"""文案层：规则模板 + 禁用词表扫描。

按决策，文案先用规则模板占位（不接 LLM），但**模板必须过禁用词闸**——
医美内容的法规风险不来自模型写得好不好，而来自有没有绝对化用语和疗效承诺。

**模板与词表现在可以来自域包**（`packs/<name>/copy.yaml`）：
  - 启用了域包（`AP_PACK=...`）→ 用包里的 templates / banned_words；
  - 未启用 → 用本文件内置的医美默认值，行为与引入域包之前完全一致。

内置禁用词表来源（不自己发明规则）：
  - 《中华人民共和国广告法》第九条：禁止使用"国家级""最高级""最佳"等用语
  - 《医疗广告管理办法》：禁止含有治愈率、有效率等断言，以及患者/testimonial 式证明
  - 医美行业常见的疗效承诺类词（保证、永久、根治、零风险等）
表中每条都标注了归类，便于审计。**这是工作假设清单，正式上线前需法务确认。**
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .pack import try_load_pack

BANNED_WORDS: list[tuple[str, str]] = [
    # 绝对化用语（《广告法》第九条）
    ("国家级", "绝对化用语"),
    ("最高级", "绝对化用语"),
    ("最佳", "绝对化用语"),
    ("最好", "绝对化用语"),
    ("第一", "绝对化用语"),
    ("顶级", "绝对化用语"),
    ("唯一", "绝对化用语"),
    ("全网最", "绝对化用语"),
    ("史上最", "绝对化用语"),
    ("无与伦比", "绝对化用语"),
    # 疗效承诺
    ("保证", "疗效承诺"),
    ("承诺", "疗效承诺"),
    ("永久", "疗效承诺"),
    ("根治", "疗效承诺"),
    ("彻底治愈", "疗效承诺"),
    ("无效退款", "疗效承诺"),
    ("零风险", "疗效承诺"),
    ("无风险", "疗效承诺"),
    ("包满意", "疗效承诺"),
    ("包效果", "疗效承诺"),
    ("绝对安全", "疗效承诺"),
    # 效果断言（医疗广告禁止）
    ("治愈率", "效果断言"),
    ("有效率", "效果断言"),
    ("成功率", "效果断言"),
    ("疗效显著", "效果断言"),
    ("立刻见效", "效果断言"),
    ("立即见效", "效果断言"),
    ("立竿见影", "效果断言"),
    ("一次见效", "效果断言"),
    # 价格诱导
    ("全网最低", "价格诱导"),
    ("最低价", "价格诱导"),
    ("免费送", "价格诱导"),
    ("限时秒杀", "价格诱导"),
    ("最后一天", "价格诱导"),
    # 医疗术语外用（医疗广告限制）
    ("处方", "医疗用语"),
    ("药到病除", "医疗用语"),
    ("细胞再生", "医疗用语"),
]


@dataclass
class CopyDraft:
    title: str
    body: str
    tags: list[str]
    banned_hits: list[str]

    @property
    def ok(self) -> bool:
        return not self.banned_hits


NEGATION_MARKS = ("不", "未", "非", "无", "拒绝")
NEGATION_WINDOW = 4


def _copy_cfg(pack: Any = None) -> dict[str, Any]:
    """取域包里的文案配置；没有域包/没有配置时返回空 dict（走内置默认）。"""
    p = pack if pack is not None else try_load_pack()
    return (getattr(p, "copy_cfg", None) or {}) if p is not None else {}


def banned_words(pack: Any = None) -> list[tuple[str, str]]:
    raw = _copy_cfg(pack).get("banned_words") or []
    if not raw:
        return BANNED_WORDS
    return [(str(item[0]), str(item[1])) for item in raw]


def _is_negated(text: str, index: int) -> bool:
    """看命中位置前面有没有否定词。

    医美文案里"不保证疗效""无任何承诺"恰恰是**合规的正确写法**，
    不应该被同一个禁用词表拦下来。只看紧邻前缀（窗口 4 字），
    避免过度推断（例如"效果永久"不会被误判成否定）。
    """
    prefix = text[max(0, index - NEGATION_WINDOW) : index]
    return any(ch in prefix for ch in NEGATION_MARKS)


def scan(text: str, pack: Any = None) -> list[str]:
    """扫描禁用词。返回命中列表（去重）。否定语境不计为命中。"""
    hits: list[str] = []
    for word, category in banned_words(pack):
        start = text.find(word)
        while start != -1:
            if not _is_negated(text, start):
                hits.append(f"{word}({category})")
                break
            start = text.find(word, start + 1)
    return sorted(set(hits))


def _render(template: str, fields: dict[str, str]) -> str:
    """渲染模板；未知占位符直接报错（配置期问题不要留到出文案时才静默）。"""
    try:
        return template.format(**fields)
    except KeyError as exc:
        raise ValueError(f"文案模板含未知占位符 {exc}；可用占位符: {sorted(fields)}") from exc
    except (IndexError, ValueError) as exc:
        raise ValueError(f"文案模板格式错误: {exc}") from exc


def make_copy(
    city: str,
    project: str,
    theme: str,
    copy_angle: str,
    channel: str,
    character: str | None = None,
    pack: Any = None,
) -> CopyDraft:
    """规则模板出文案。

    刻意保守：不写疗效、不写价格承诺、不写绝对化用语；
    只描述"记录了什么画面"，把"项目怎么好"留给持证顾问与合规审核。

    有域包时用包里的模板；否则用下方内置模板（医美默认口径）。
    """
    cfg = _copy_cfg(pack)
    templates = cfg.get("templates") or {}
    tpl = templates.get(channel) or templates.get("default")

    if tpl:
        fields = {
            "city": city,
            "project": project,
            "theme": theme,
            "copy_angle": copy_angle,
            "channel": channel,
        }
        title = _render(str(tpl.get("title") or ""), fields)
        body = _render(str(tpl.get("body") or ""), fields)
        tags = [_render(str(t), fields) for t in (tpl.get("tags") or [])]
    else:
        subject = f"{city}{project}"
        if channel == "xiaohongshu":
            title = f"{subject}｜{copy_angle}"
            body = (
                f"📍{city}\n"
                f"项目：{project}\n"
                f"内容角度：{copy_angle}\n\n"
                f"本条记录的是一次真实就诊场景的画面，"
                f"画面为演示用生成素材，不代表任何机构的真实顾客。\n"
                f"具体方案、适应症与禁忌请以面诊结论为准。"
            )
            tags = [f"#{city}医美", f"#{project}", f"#{copy_angle}", "#医美记录"]
        else:
            title = f"{subject}：{copy_angle}"
            body = (
                f"在城市：{city}\n项目：{project}\n角度：{copy_angle}\n\n"
                f"本条为演示用生成素材，用于展示内容生产流程，不构成医疗建议。"
                f"实际适应症、禁忌与方案请以持证医师面诊为准。"
            )
            tags = [city, project, copy_angle]

    hits = scan(f"{title}\n{body}\n{' '.join(tags)}", pack=pack)
    return CopyDraft(title=title, body=body, tags=tags, banned_hits=hits)


def to_row(key: str, draft: CopyDraft) -> dict[str, Any]:
    return {
        "idempotency_key": key,
        "title": draft.title,
        "body": draft.body,
        "tags": draft.tags,
        "banned_hits": draft.banned_hits,
    }


def redact(text: str, pack: Any = None) -> str:
    """把禁用词替换掉，给"修复后重出"的降级链用。"""
    for word, _ in banned_words(pack):
        text = text.replace(word, "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()
