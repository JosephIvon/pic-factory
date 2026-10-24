"""运行时配置。

约定：不写死本机绝对路径。仓库内路径由 __file__ 反推；仓库外依赖走环境变量；
缺失时明确报错退出，不静默按当前目录找。
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("AP_REPO_ROOT") or Path(__file__).resolve().parents[2])
# ^ parents[2] 只在"从源码树跑"时成立；pip 安装进 site-packages 后它指向解释器目录。
# 容器镜像里显式设 AP_REPO_ROOT=/app，数据/提示词/证据/参考图才解析得到。

DATA_DIR = Path(os.environ.get("AP_DATA_DIR") or REPO_ROOT / "data")
PROMPT_DIR = Path(os.environ.get("AP_PROMPT_DIR") or REPO_ROOT / "prompts")
OUT_DIR = Path(os.environ.get("AP_OUT_DIR") or REPO_ROOT / "out")
DB_PATH = Path(os.environ.get("AP_DB") or DATA_DIR / "pipeline.db")
EVIDENCE_DIR = Path(os.environ.get("AP_EVIDENCE_DIR") or REPO_ROOT / "evidence")

# 生图后端 CLI（mcode-tools）。可换成任意同契约的 CLI。
MATRIX_CLI = os.environ.get("AP_MATRIX_CLI") or "mcode-tools"

# 一次调用塞多少张。实测 6 张稳定成功、约 7.5s/张。
BATCH_SIZE = int(os.environ.get("AP_BATCH_SIZE") or "6")

# 单张失败后的重试次数（只重试失败项，不整批重跑）。
MAX_RETRY = int(os.environ.get("AP_MAX_RETRY") or "2")

# 单张成本（元）。不设则台账里 cost 为 null 并标 unpriced —— 不编造单价。
UNIT_COST = os.environ.get("AP_UNIT_COST")


def unit_cost() -> float | None:
    """单张成本。None 表示未配置单价，台账只记调用次数与耗时。"""
    return None if UNIT_COST is None else float(UNIT_COST)


def api_unit_cost() -> float | None:
    """API 后端单张成本（元）。

    `AP_COST_API` 优先，`AP_UNIT_COST` 作为通用兜底。
    未配置返回 None —— **不编造单价**。
    """
    raw = os.environ.get("AP_COST_API") or os.environ.get("AP_UNIT_COST")
    return None if raw is None else float(raw)


def local_unit_cost(seconds: float) -> float | None:
    """本地栈单张成本（元）= 电费 + 设备折旧摊销。

    三个参数缺一个就返回 None（宁可没数字，也不编一个）：
      AP_LOCAL_WATT           GPU 满载功耗（W）
      AP_KWH_PRICE            电价（元/kWh）
      AP_LOCAL_DEPR_PER_HOUR  设备折旧摊销（元/小时）

    公式：秒数/3600 × (W/1000 × 电价 + 每小时折旧)
    """
    watt = os.environ.get("AP_LOCAL_WATT")
    kwh = os.environ.get("AP_KWH_PRICE")
    depr = os.environ.get("AP_LOCAL_DEPR_PER_HOUR")
    if not (watt and kwh and depr):
        return None
    per_hour = float(watt) / 1000.0 * float(kwh) + float(depr)
    return round(max(seconds, 0.0) / 3600.0 * per_hour, 6)


def unit_cost_for(provider: str, seconds: float) -> float | None:
    """按后端算单张成本。None = 未配置单价。"""
    if provider == "local":
        return local_unit_cost(seconds)
    if provider == "matrix":
        return api_unit_cost()
    return None


def local_model_dir() -> Path:
    """本地模型目录。环境变量优先，其次仓库内 models/sdxl-turbo。"""
    env = os.environ.get("AP_LOCAL_MODEL_DIR")
    candidate = Path(env) if env else REPO_ROOT / "models" / "sdxl-turbo"
    if not candidate.exists():
        raise RuntimeError(
            f"找不到本地模型目录 {candidate}。请设置 AP_LOCAL_MODEL_DIR 指向模型权重目录。"
        )
    return candidate


# 本地栈采样预设。turbo 系（4 步 / guidance 0）**不能**套到普通 SDXL checkpoint 上：
# 普通 SDXL 要 25~30 步、guidance 7~8，否则出图发灰、结构崩坏。
LOCAL_PRESETS: dict[str, dict[str, float]] = {
    "turbo": {"steps": 4, "guidance": 0.0},
    "photo": {"steps": 28, "guidance": 7.5},
}


def local_sampling() -> dict[str, float]:
    """本地栈采样参数。预设可用 AP_LOCAL_PRESET 选，单项可用 AP_LOCAL_STEPS / AP_LOCAL_GUIDANCE 覆盖。"""
    preset = LOCAL_PRESETS.get(os.environ.get("AP_LOCAL_PRESET", "turbo"), LOCAL_PRESETS["turbo"])
    steps_env = os.environ.get("AP_LOCAL_STEPS")
    guidance_env = os.environ.get("AP_LOCAL_GUIDANCE")
    return {
        "steps": int(steps_env) if steps_env else int(preset["steps"]),
        "guidance": float(guidance_env) if guidance_env else float(preset["guidance"]),
    }


def local_python() -> Path:
    """本地栈解释器路径（带 CUDA torch + diffusers 的 venv）。

    优先 AP_LOCAL_PYTHON 环境变量，其次仓库内 .venv-local。都没有就报错退出。
    """
    env = os.environ.get("AP_LOCAL_PYTHON")
    candidates = [Path(env)] if env else []
    candidates.append(REPO_ROOT / ".venv-local" / "Scripts" / "python.exe")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "找不到本地栈解释器。请设置 AP_LOCAL_PYTHON 指向装有 torch(CUDA)+diffusers 的虚拟环境，"
        "或按 README 的「本地栈」小节创建 .venv-local。"
    )


# 仓库内的子目录名：台账里存的是绝对路径，仓库被整体搬走后按这些前缀重新锚定。
REPO_SUBTREES = ("out", "evidence", "fixtures")


def resolve_repo_path(stored: str | Path | None) -> Path | None:
    """把台账里记的路径解析成当前可用的路径。

    1) 原路径还在 → 直接用；
    2) 原路径没了（仓库被搬到别处）→ 按 `out/ evidence/ fixtures/` 前缀
       重新锚定到当前仓库根；
    3) 都对不上 → 返回 None（**不猜、不静默按当前目录找**）。
    """
    if not stored:
        return None
    path = Path(stored)
    if path.exists():
        return path
    parts = path.parts
    for index, part in enumerate(parts):
        if part in REPO_SUBTREES:
            candidate = REPO_ROOT.joinpath(*parts[index:])
            if candidate.exists():
                return candidate
    return None
