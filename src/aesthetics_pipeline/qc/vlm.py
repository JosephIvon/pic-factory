"""路3：VLM 语义评分（软信号）。

**已知缺陷，所以只做软信号**：实测同一张带乱码招牌的图，VLM 能读出乱码文字，
却判 "garbled: false"、"worst_issue: 无明显问题"。它的语义判断（场景对不对、
像不像真人照片）有价值，但"有没有文字"这种确定性问题不交给它。

输出统一解析成结构化分数，解析失败记 None，不猜。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from ..cli_runner import extract_json, run_cli
from ..config import MATRIX_CLI

TOOL = "connector__matrix__describe_images"

PROMPT = (
    "你是医美图文质检助手。回答下面的问题，只输出一个 JSON 对象，不要任何额外文字："
    '{"realism_1_5": 数字, "scene_match_1_5": 数字, "issues": ["问题描述"], "summary": "一句话结论"}。'
    "评分标准：realism 5=看不出是AI生成、1=明显AI感；scene_match 5=画面与描述的场景完全一致、1=完全不符。"
    "issues 只写你能明确看出的画面问题（例如手指畸形、脸部扭曲、文字乱码、过度磨皮）。"
    "画面描述：{desc}"
)


class VlmGate:
    def __init__(self, cli: str = MATRIX_CLI, timeout: int = 180) -> None:
        self.cli = cli
        self.timeout = timeout

    def _describe(self, url: str, desc: str) -> str | None:
        payload = {
            "image_info": [{"url": url, "prompt": PROMPT.format(desc=desc)}]
        }
        # 统一走 --args-file：内联 JSON 会被 Windows 的引号解析破坏（实测）
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
            args_file = fh.name
        try:
            proc = run_cli(self.cli, ["connector", "call", TOOL, "--args-file", args_file], timeout=self.timeout)
        except Exception:
            return None
        finally:
            Path(args_file).unlink(missing_ok=True)
        if proc.returncode != 0:
            return None
        try:
            return extract_json(proc.stdout)
        except Exception:
            return None

    def check(self, image_url: str, desc: str = "an aesthetic clinic photo") -> dict[str, Any]:
        payload = self._describe(image_url, desc)
        if not payload:
            return {"ok": True, "skipped": True, "scores": {}, "issues": [], "reason": "VLM 调用失败或返回无法解析"}
        results = payload.get("results") or []
        if not results:
            return {"ok": True, "skipped": True, "scores": {}, "issues": [], "reason": "VLM 无返回内容"}
        text = (results[0].get("description") or "").strip()
        parsed = _parse_json_block(text)
        if parsed is None:
            return {"ok": True, "skipped": True, "scores": {}, "issues": [], "reason": "VLM 返回非 JSON"}
        scores = {
            "realism": _as_float(parsed.get("realism_1_5")),
            "scene_match": _as_float(parsed.get("scene_match_1_5")),
        }
        issues = parsed.get("issues") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        return {"ok": True, "skipped": False, "scores": scores, "issues": issues, "reason": ""}


def _parse_json_block(text: str) -> dict | None:
    """从可能带 ``` 围栏的文本里抠出 JSON。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def local_file_to_url(path: str | Path, uploader) -> str:
    """本地图片要先上传成服务端可取的 URL 才能送 VLM。"""
    return uploader(path)
