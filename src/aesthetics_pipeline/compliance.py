"""合规校验与发布包：把域包里的合规要求变成**可执行的检查**，而不是文档里的口号。

三件事：
1. `audit()` —— 对着台账逐条查：文案有没有免责声明、图文是否成对、
   画面里有没有文字、文案有没有命中禁用词。返回结构化结果（可进 CI）。
2. `build_manifest()` —— 每张图一条记录：模型、prompt、质检分、采纳状态、
   AI 标识（显式/隐式）。这是《AI 生成合成内容标识办法》里"隐式标识"的落点。
3. `export_publish()` —— 打成可直接交给运营/客户的 zip：图 + 文案 + manifest +
   合规声明 + 校验结果。**校验不过默认拒绝打包**（fail-closed）。

边界（诚实说）：
- "显式 AI 标识"由平台水印提供，这里**不逐图像素级验证**，只在 manifest 里
  如实记录 `explicit: true` 与来源（platform_watermark）。
- `forbidden_motifs`（术前术后对比等）是**人工复核清单**，不自动判定——
  误判比漏判更贵。
- 免责声明做的是**子串检查**：它保证"该说的话没丢"，不保证措辞被法务认可。
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import ledger
from .pack import try_load_pack

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"


@dataclass(frozen=True)
class Issue:
    code: str
    severity: str
    key: str
    provider: str
    detail: str


def rules_from_pack(pack: Any = None) -> dict[str, Any]:
    """当前域包的合规要求；未启用域包时返回空 dict（只做通用检查）。"""
    p = pack if pack is not None else try_load_pack()
    return dict(getattr(p, "compliance", None) or {}) if p is not None else {}


def _copy_map(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        "SELECT idempotency_key, title, body, tags, banned_hits FROM copies"
    ).fetchall()
    return {r["idempotency_key"]: dict(r) for r in rows}


def _json_list(raw: Any) -> list[Any]:
    if raw in (None, "", "[]"):
        return []
    if isinstance(raw, list):
        return raw
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def check_row(row: dict[str, Any], copy_row: dict[str, Any] | None, rules: dict[str, Any]) -> list[Issue]:
    """单条资产的合规检查。只审"已放行"的（被拒的本来就不发布）。"""
    if row.get("qc_verdict") != "accepted":
        return []
    key = str(row.get("idempotency_key") or "")
    provider = str(row.get("provider") or "unknown")
    issues: list[Issue] = []

    if not (row.get("swapped_path") or row.get("local_path")):
        issues.append(Issue("NO_IMAGE", SEVERITY_ERROR, key, provider, "已放行但没有落盘产物"))

    texts = _json_list(row.get("text_found"))
    if texts:
        issues.append(
            Issue("TEXT_IN_IMAGE", SEVERITY_ERROR, key, provider, f"画面内检出文字: {texts[:3]}")
        )

    if copy_row is None:
        issues.append(Issue("NO_COPY", SEVERITY_ERROR, key, provider, "图文未成对：没有对应文案"))
        return issues

    text = "\n".join(
        [str(copy_row.get("title") or ""), str(copy_row.get("body") or "")]
        + [str(t) for t in _json_list(copy_row.get("tags"))]
    )
    disclaimer = str(rules.get("required_disclaimer") or "").strip()
    if disclaimer and disclaimer not in text:
        issues.append(
            Issue("MISSING_DISCLAIMER", SEVERITY_ERROR, key, provider, "文案缺少域包要求的免责声明")
        )

    hits = _json_list(copy_row.get("banned_hits"))
    if hits:
        issues.append(Issue("BANNED_HIT", SEVERITY_ERROR, key, provider, "文案命中禁用词: " + ", ".join(map(str, hits))))
    return issues


def audit(
    conn: sqlite3.Connection,
    *,
    pack: Any = None,
    adopted_only: bool = False,
) -> dict[str, Any]:
    """对着台账跑一遍合规检查。返回 {ok, checked, issues, by_code, rules}。

    pack 为 None 时按 AP_PACK 解析一次，**同一份** pack 同时用于规则和署名，
    避免"规则来自域包、标题却写未启用"的自相矛盾。
    """
    resolved = pack if pack is not None else try_load_pack()
    rules = rules_from_pack(resolved)
    copies = _copy_map(conn)
    rows = [r for r in ledger.assets(conn, "local_path IS NOT NULL") if r.get("qc_verdict") == "accepted"]
    if adopted_only:
        rows = [r for r in rows if r.get("adopted") == "adopted"]

    issues: list[Issue] = []
    for row in rows:
        issues.extend(check_row(row, copies.get(str(row.get("idempotency_key"))), rules))

    by_code: dict[str, int] = {}
    for issue in issues:
        by_code[issue.code] = by_code.get(issue.code, 0) + 1

    return {
        "ok": not issues,
        "checked": len(rows),
        "adopted_only": adopted_only,
        "pack": getattr(resolved, "name", None) or None,
        "rules": {
            "required_disclaimer": rules.get("required_disclaimer"),
            "ai_label": rules.get("ai_label") or {},
            "forbidden_motifs": rules.get("forbidden_motifs") or [],
        },
        "issues": [asdict(i) for i in issues],
        "by_code": by_code,
    }


def build_manifest(
    rows: list[dict[str, Any]],
    copies: dict[str, dict[str, Any]],
    *,
    pack: Any = None,
    batch: str = "",
) -> dict[str, Any]:
    """生成发布包清单。每张图一条，含 AI 标识与来源，供审计/追溯。"""
    rules = rules_from_pack(pack)
    ai_label = dict(rules.get("ai_label") or {})
    entries = []
    for row in rows:
        key = str(row.get("idempotency_key") or "")
        copy_row = copies.get(key) or {}
        entries.append(
            {
                "key": key,
                "provider": row.get("provider"),
                "model_id": row.get("model_id"),
                "route": row.get("route"),
                "city": row.get("city"),
                "project": row.get("project"),
                "theme": row.get("theme"),
                "channel": row.get("channel"),
                "prompt": row.get("prompt"),
                "qc": {
                    "verdict": row.get("qc_verdict"),
                    "face_similarity": row.get("face_similarity"),
                    "face_similarity_after": row.get("face_similarity_after"),
                },
                "adopted": row.get("adopted"),
                "created_at": row.get("created_at"),
                "ai_label": {
                    "explicit": bool(ai_label.get("explicit")),
                    "source": "platform_watermark",
                    "implicit_metadata": bool(ai_label.get("implicit_metadata", True)),
                },
                "copy": {
                    "title": copy_row.get("title"),
                    "body": copy_row.get("body"),
                    "tags": _json_list(copy_row.get("tags")),
                },
            }
        )
    return {
        "pack": getattr(pack, "name", None) or "legacy",
        "pack_version": getattr(pack, "version", None) or "0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "batch": batch,
        "disclaimer": rules.get("required_disclaimer"),
        "asset_count": len(entries),
        "assets": entries,
    }


def export_publish(
    conn: sqlite3.Connection,
    out_zip: Path,
    *,
    pack: Any = None,
    adopted_only: bool = True,
    allow_violations: bool = False,
    batch: str = "",
) -> Path:
    """打发布包。校验不过默认拒绝（fail-closed），除非显式 allow_violations。"""
    result = audit(conn, pack=pack, adopted_only=adopted_only)
    if not result["ok"] and not allow_violations:
        codes = ", ".join(f"{k}×{v}" for k, v in sorted(result["by_code"].items()))
        raise ValueError(f"合规校验未通过，拒绝打包（{codes}）。修完后重试，或显式 --allow-violations。")

    copies = _copy_map(conn)
    rows = [r for r in ledger.assets(conn, "local_path IS NOT NULL") if r.get("qc_verdict") == "accepted"]
    if adopted_only:
        rows = [r for r in rows if r.get("adopted") == "adopted"]

    manifest = build_manifest(rows, copies, pack=pack, batch=batch)
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.writestr("compliance.json", json.dumps(result, ensure_ascii=False, indent=2))
        zf.writestr("README.txt", _readme_text(manifest, result))
        for row in rows:
            path = ledger.asset_path(row, "swapped_path") or ledger.asset_path(row)
            if not path:
                continue
            arc = f"images/{row['idempotency_key']}__{row.get('provider') or 'unknown'}{path.suffix}"
            zf.write(path, arc)
            copy_row = copies.get(str(row.get("idempotency_key"))) or {}
            if copy_row:
                zf.writestr(
                    f"copy/{row['idempotency_key']}.txt",
                    f"{copy_row.get('title') or ''}\n\n{copy_row.get('body') or ''}\n\n"
                    f"{' '.join(map(str, _json_list(copy_row.get('tags'))))}\n",
                )
    return out_zip


def _readme_text(manifest: dict[str, Any], result: dict[str, Any]) -> str:
    lines = [
        f"发布包：{manifest['pack']} @ {manifest['pack_version']}",
        f"生成时间：{manifest['generated_at']}",
        f"素材数量：{manifest['asset_count']}",
        "",
        "【AI 生成声明】",
        str(manifest.get("disclaimer") or "（域包未配置免责声明）"),
        "",
        "每张图均带平台显式 AI 标识；本包 manifest.json 即隐式标识元数据，请勿删除。",
        "",
        "【人工复核清单】",
    ]
    rules = result.get("rules") or {}
    for motif in (rules.get("forbidden_motifs") or []):
        lines.append(f"- 不得出现：{motif}")
    if not (rules.get("forbidden_motifs") or []):
        lines.append("- （域包未配置禁止构图清单）")
    lines += ["", f"合规校验：{'通过' if result['ok'] else '未通过'}（检查 {result['checked']} 条）"]
    return "\n".join(lines) + "\n"
