"""文档数字防漂移门禁：文档里出现的每个"实测数字"都要能对着代码/证据文件实算。

为什么要有它：README / docs 里的数字全靠人肉同步，必错。本仓**已实测过**的漂移有四类：

  1. 测试数：README 一度写 62 / 124 / 129 / 155，实际已经 174；
  2. 阈值口径：README 三处写"当前未标定"，而代码早已标定 T=0.57（证据文件在仓内）；
  3. 离线回放：README 写"可 replay 三条路线（matrix / http_api / local）"，
     而 ``fixtures/cache/`` 里只有 matrix / local 两条，http_api 一条都没有；
  4. 夹具档位：README 写"缓存条目带档位…不丢 model_id"，而入库的 8 条 model_id 全是 null。

这类漂移的共同点是：**话本身没错，错的是它已经过期了**。所以做法不是"再加个人工核对清单"，
而是把数字变成**可复算的**——文档里那一行必须与代码/证据文件算出来的一致，不一致就红。

检查项分两种：
  * **实算值**：文档里必须出现"算出来的那个值"（如任务矩阵 150 条、T=0.57）；
  * **禁止出现**：文档里不得再出现某个已被推翻的说法（如"当前未标定"）。

用法：
  python tools/check_doc_numbers.py          # 只检查，不一致退出码 1
  python tools/check_doc_numbers.py --public # 公开树口径：目标文档不在场的检查项跳过（可见报告）
  python tools/check_doc_numbers.py --list   # 只列检查项，不跑

新增检查项：在 CHECKS 里加一条即可——``snippet`` 里的 ``{v}`` 会被实算值替换。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CALIB = ROOT / "docs" / "ab-evidence" / "threshold-calibration-production.json"


# ---------------------------------------------------------------- 实算函数
def _calib() -> dict:
    return json.loads(CALIB.read_text(encoding="utf-8"))


def task_total() -> str:
    """任务矩阵展开后的总条数。"""
    from aesthetics_pipeline.taskspec import expand

    return str(len(expand()))


def route_split() -> str:
    """路由分布：按 router 实际判出的 i2i / t2i 条数（不是猜"有角色就是 i2i"）。"""
    from aesthetics_pipeline.router import Router
    from aesthetics_pipeline.taskspec import expand, pack_matrix, scene_config

    router = Router(pack_matrix(), scene_config())
    specs = expand()
    i2i = sum(1 for s in specs if router.decide_path(s)[0] == "i2i")
    return f"I2I {i2i} / T2I {len(specs) - i2i}"


def fixtures_total() -> str:
    """入库夹具（缓存条目）条数。"""
    return str(len(list((ROOT / "fixtures" / "cache").glob("*/*.json"))))


def fixtures_routes() -> str:
    """实际有夹具的回放路线（按 PROVIDER_ORDER 顺序）——README 说什么就写什么，不许多不许少。"""
    from aesthetics_pipeline.provider.cache import PROVIDER_ORDER

    have = {p.name for p in (ROOT / "fixtures" / "cache").glob("*") if p.is_dir()}
    return " / ".join(p for p in PROVIDER_ORDER if p in have)


def face_threshold() -> str:
    """标定出来的阈值（证据文件为准，与 qc/face 的常量同源）。"""
    return str(_calib()["threshold"])


def calib_positives() -> str:
    c = _calib()
    return f"{c['positive_n']} / {c['positive_n_eligible']}"


def calib_negatives() -> str:
    return str(_calib()["indomain_negative_n"])


def budget_ratio() -> str:
    from aesthetics_pipeline.cost_guard import BUDGET_FUSE_RATIO

    return f"{int(round(BUDGET_FUSE_RATIO * 100))}%"


def concurrency_default() -> str:
    from aesthetics_pipeline.provider.http_api import (
        DEFAULT_CONCURRENCY,
        MAX_CONCURRENCY,
        MIN_CONCURRENCY,
    )

    return f"{DEFAULT_CONCURRENCY}（钳位 {MIN_CONCURRENCY}~{MAX_CONCURRENCY}"


def _env_default(name: str, fallback: str) -> str:
    import os

    return os.environ.get(name) or fallback


def batch_size() -> str:
    from aesthetics_pipeline.config import BATCH_SIZE

    return str(BATCH_SIZE)


def max_retry() -> str:
    from aesthetics_pipeline.config import MAX_RETRY

    return str(MAX_RETRY)


def db_path() -> str:
    from aesthetics_pipeline.config import DB_PATH

    try:
        return DB_PATH.relative_to(ROOT).as_posix()
    except ValueError:
        return str(DB_PATH)


def _repo_rel(path: Path) -> str:
    """文档里写的是**仓库相对路径**（机器换了也不能变），不是本机绝对路径。"""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def local_python_default() -> str:
    from aesthetics_pipeline.config import REPO_ROOT

    return _repo_rel(REPO_ROOT / ".venv-local" / "Scripts" / "python.exe")


def local_model_dir_default() -> str:
    from aesthetics_pipeline.config import REPO_ROOT

    return _repo_rel(REPO_ROOT / "models" / "sdxl-turbo")


def local_presets() -> str:
    from aesthetics_pipeline.config import LOCAL_PRESETS

    turbo = LOCAL_PRESETS["turbo"]
    photo = LOCAL_PRESETS["photo"]
    return (
        f"`turbo`（{int(turbo['steps'])} 步 / guidance {turbo['guidance']:.0f}）；"
        f"写实 checkpoint 用 `photo`（{int(photo['steps'])} 步 / guidance {photo['guidance']}）"
    )


def ocr_min_score() -> str:
    from aesthetics_pipeline.qc.rules import OCR_MIN_SCORE, MIN_SIDE

    return f"{MIN_SIDE}px / {OCR_MIN_SCORE}"


# ------------------------------------------------- 选型口径（生产度量复算值）
RECOMPUTE = ROOT / "docs" / "ab-evidence" / "model-sim-recompute.json"

MODEL_NAMES = {
    "ark-4.0": "Seedream 4.0",
    "ark-5.0-pro": "Seedream 5.0 Pro",
    "ark-5.0-flash": "Seedream 5.0 Flash",
    "qwen-3.0": "Qwen-Image 3.0",
    "qwen-3.0-pro": "Qwen-Image 3.0 Pro",
}


def _sims() -> dict:
    return json.loads(RECOMPUTE.read_text(encoding="utf-8"))["models"]


def model_row(key: str) -> str:
    """MODEL-CHOICE.md §9 里那一个模型行的前半段（生产度量口径）。

    列：模型 | 成功 | 一致性均值 | 剔遮挡均值 | 剔遮挡最低 | 剔遮挡标准差 | 最低 | 标准差
    """
    import statistics as st

    m = _sims()[key]
    per = m["per_prompt"]
    clean = [v for p, v in per.items() if v is not None and p != "p09_generic_aftercare"]
    every = [v for v in per.values() if v is not None]
    return (
        f"| {MODEL_NAMES[key]} | {m['n']}/10 | {m['sim_mean_all']:.3f} | "
        f"{m['sim_mean_clean_excl_p09']:.3f} | {min(clean):.3f} | {st.pstdev(clean):.3f} | "
        f"{m['sim_min_all']:.3f} | {st.pstdev(every):.3f} |"
    )


def model_row_check(key: str) -> Check:
    return Check(f"选型口径·{MODEL_NAMES[key]}", "docs/MODEL-CHOICE.md", model_row(key))


# ---------------------------------------------------------------- 检查项
@dataclass
class Check:
    name: str
    doc: str
    expect: str | None = None  # 必须出现的片段（可含 {v}）
    value: Callable[[], str] | None = None  # 实算值，填进 {v}
    forbid: str | None = None  # 不得出现的片段
    internal: bool = False  # 只在内仓口径跑：目标文档在公开树不存在或不是同一份


CHECKS: list[Check] = [
    Check("任务矩阵总条数", "README.md",
          "| 任务矩阵 | {v} 条（3 城 × 5 项目 × 5 主题 × 2 渠道 × 角色维度） |", task_total),
    Check("路由分布 i2i/t2i", "README.md",
          "| 路由分布 | {v} |", route_split),
    Check("入库夹具条数", "README.md",
          "`fixtures/` 随仓库提交（{v} 条，与台账同口径）", fixtures_total),
    Check("可回放路线（按实际夹具）", "README.md",
          "新 clone 机器可 replay {v} 路线", fixtures_routes),
    Check("人脸阈值（标定证据）", "README.md",
          "已标定 T={v}", face_threshold),
    Check("标定正样本 / 合格正样本", "README.md",
          "域内正样本 {v} 张", calib_positives),
    Check("标定同域难负样本", "README.md",
          "同域难负 {v} 张", calib_negatives),
    Check("预算熔断比例", "README.md",
          "累计花费达 **{v}** 熔断新调用", budget_ratio),
    Check("批内并发默认与钳位", "README.md",
          "| {v}；1=串行） |", concurrency_default),
    Check("AP_BATCH_SIZE 默认", "README.md",
          "| `AP_BATCH_SIZE` | 单批张数 | {v} |", batch_size),
    Check("AP_MAX_RETRY 默认", "README.md",
          "| `AP_MAX_RETRY` | 失败项重试次数 | {v} |", max_retry),
    Check("AP_DB 默认", "README.md",
          "| `AP_DB` | 台账路径 | `{v}` |", db_path),
    Check("AP_FACE_THRESHOLD 口径", "README.md",
          "| `AP_FACE_THRESHOLD` | 人脸阈值覆盖 | 见第 6 节（**已标定 T={v}**） |", face_threshold),
    Check("AP_LOCAL_PYTHON 默认", "README.md",
          "| `AP_LOCAL_PYTHON` | 本地栈解释器 | `{v}` |", local_python_default),
    Check("AP_LOCAL_MODEL_DIR 默认", "README.md",
          "| `AP_LOCAL_MODEL_DIR` | 本地栈模型目录 | `{v}` |", local_model_dir_default),
    Check("AP_LOCAL_PRESET 预设", "README.md",
          "| `AP_LOCAL_PRESET` | 本地栈采样预设 | {v} |", local_presets),
    # 已被推翻的说法：出现即红
    Check("README 不得再声称阈值未标定", "README.md", forbid="当前未标定"),
    Check("README 不得再声称阈值本身仍未标定", "README.md", forbid="阈值本身仍未标定"),
    Check("README 不得再声称 calibrate-face 尚未标定", "README.md", forbid="尚未标定"),
    Check("MODEL-CHOICE 不得再声称阈值未标定", "docs/MODEL-CHOICE.md", forbid="未标定"),
    Check("DOMAIN-PACKS 不得写死测试总数", "docs/DOMAIN-PACKS.md", forbid="155"),
    Check("DOMAIN-PACKS 不得写死测试总数", "docs/DOMAIN-PACKS.md", forbid="项全过"),
    Check("DOC 不得声称夹具带档位", "README.md", forbid="回放交付线图不丢 model_id"),
    # 选型口径：MODEL-CHOICE §9 的五行必须与复算 JSON 逐格一致
    *[model_row_check(k) for k in MODEL_NAMES],
    # 旧实验台尺的数字不得复活（0.888/0.706/0.759/0.724 是被推翻的那版）
    Check("MODEL-CHOICE 不得再出现实验台旧均值", "docs/MODEL-CHOICE.md", forbid="0.888"),
    Check("MODEL-CHOICE 不得再出现实验台旧均值", "docs/MODEL-CHOICE.md", forbid="0.706"),
    Check("MODEL-CHOICE 不得再出现实验台旧最低", "docs/MODEL-CHOICE.md", forbid="0.759"),
    Check("MODEL-CHOICE 不得再出现实验台旧均值", "docs/MODEL-CHOICE.md", forbid="0.724"),
    Check("MODEL-CHOICE 不得再出现实验台假换脸样本", "docs/MODEL-CHOICE.md", forbid="−0.005"),
    # README 的档位表与订正块
    Check("README 档位表剔遮挡均值", "README.md", "剔遮挡均值 0.913"),
    Check("README 订正块剔遮挡标准差", "README.md", "剔遮挡标准差 0.043"),
    Check("README 订正块剔遮挡标准差", "README.md", "剔遮挡标准差 0.171"),
    # 口径规则：闸门没跑 ≠ 合格（人脸闸 fail-closed）
    Check("README 必须写明闸门没跑≠合格", "README.md", "闸门没跑 ≠ 合格"),
    # 演示脚本不得用不存在的 CLI 参数（qc 没有 --limit）
    Check("DEMO 不得写不存在的 qc --limit", "DEMO.md", forbid="qc --limit"),
]

# 内部版口径：README.md 的数字表/环境变量表只对内仓 README 成立（公开树是另一份 README，
# 数量由 check_readme_test_count 单独守）；DEMO.md 不随公开树分发。
for _c in CHECKS:
    if _c.doc in ("README.md", "DEMO.md"):
        _c.internal = True

# 公开版 README 也携带的主张：两种口径都跑（它虽是另一份文档，但这些话它也必须说）
CHECKS += [
    Check("README 生产尺均值在两份 README 里都在场", "README.md", "剔遮挡均值 0.913"),
    Check("README 闸门口径在两份 README 里都在场", "README.md", "闸门没跑 ≠ 合格"),
]


def _docs() -> dict[str, str | None]:
    """读检查集涉及的所有文档；不在场的记 None（默认口径=缺失即红，防静默缩水）。"""
    out: dict[str, str | None] = {}
    for rel in {c.doc for c in CHECKS}:
        path = ROOT / rel
        out[rel] = path.read_text(encoding="utf-8") if path.is_file() else None
    return out


def missing_docs(docs: dict[str, str | None]) -> list[str]:
    return sorted(rel for rel, text in docs.items() if text is None)


def _norm(text: str) -> str:
    """比对前把强调标记与反引号去掉：文档里加粗/加代码不该让门禁变红。"""
    return text.replace("**", "").replace("`", "")


def run_checks(docs: dict[str, str | None], public: bool = False) -> list[str]:
    """跑一遍检查，返回失败描述列表（空 = 全过）。可传自定义 docs 便于自测。

    public=True（公开树口径）：internal 检查项跳过；目标文档不在场的其余检查项也跳过。
    默认（内仓口径）：internal 全跑；文档缺失按失败算，防止检查集被人悄悄删文档缩水。
    """
    failures: list[str] = []
    for c in CHECKS:
        if c.internal and public:
            continue
        raw = docs.get(c.doc)
        if raw is None:
            if not public:
                failures.append(f"{c.name}：{c.doc} 不在本树——确认该文档是否应存在")
            continue
        text = _norm(raw)
        if c.forbid is not None:
            if c.forbid in text:
                failures.append(
                    f"{c.name}：{c.doc} 里仍出现已作废的说法「{c.forbid}」"
                )
            continue
        if c.expect is None:
            continue
        want = _norm(c.expect.format(v=c.value() if c.value else ""))
        if want not in text:
            failures.append(f"{c.name}：{c.doc} 里找不到「{want}」")
    return failures


def main() -> int:
    args = sys.argv[1:]
    if "--list" in args:
        for c in CHECKS:
            print(f"{c.name}\t{c.doc}\t{c.expect or ('禁止：' + str(c.forbid))}")
        return 0

    public = "--public" in args
    docs = _docs()
    missing = missing_docs(docs)
    failures = run_checks(docs, public=public)
    if failures:
        print(f"文档数字漂移 {len(failures)} 处：")
        for f in failures:
            print(f"  - {f}")
        print(
            "\n这些数字都能对着代码/证据文件实算——修文档，或改代码后同步文档；"
            "别让它们各自漂着。"
        )
        if missing and not public:
            print("提示：有检查项目标文档不在本树；在没有内部文档的公开树上请用 --public。")
        return 1
    skipped = sum(1 for c in CHECKS if c.internal) if public else 0
    note = f"（公开树口径：跳过 {skipped} 项内部版检查）" if public else ""
    print(f"一致：{len(CHECKS) - skipped}/{len(CHECKS)} 项文档数字与代码/证据文件对得上 {note}".rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
