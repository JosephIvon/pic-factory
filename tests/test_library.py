"""L1 素材资产化+复用层（library.py + generate --reuse-adopted）。

核心账目（docs/COST-MODEL.md 的最大降本杠杆）：发布量 ≠ 生成量——
同素材（角色×项目×主题）已有合格产出时本地复制顶上，est_cost=0.0 落台账，
后端 generate 根本不该被调用。四条回归钉死：
  1. 复用优先级：adopted+accepted > accepted > 其余落盘行，同档取最新；
  2. rejected（机器拦 / 人弃）永不被复用；
  3. 复用行 est_cost == 0.0 且假 backend 的 generate **零调用**；
  4. 默认 --no-reuse-adopted 时行为与旧路径完全一致（后端照常被调）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from typer.testing import CliRunner

from aesthetics_pipeline import config, ledger, taskspec
from aesthetics_pipeline.cli import app
from aesthetics_pipeline import library
from aesthetics_pipeline.models import GenItem, GenOutcome, TaskSpec

runner = CliRunner()

PROJECT = "光子嫩肤"
THEME = "术后修复"
OTHER_THEME = "价格透明拆解"

MATRIX = {
    "version": "test-lib",
    "cities": ["武汉", "成都"],
    "projects": {
        # 同一角色同一主题、只差城市：正是"发布量≠生成量"要复用的形态
        PROJECT: {"visual": "a female client seated in front of a light device",
                  "use_character": True},
    },
    "themes": {
        THEME: {"visual": "calm satisfied expression, healthy even skin",
                "copy_angle": "术后修复记录"},
    },
    "channels": {"xiaohongshu": {"aspect_ratio": "3:4", "resolution": "2K"}},
    "characters": {
        "lin": {
            "identity_block": "a 28 year old Chinese woman with straight black hair",
            "ref": "evidence/01_t2i_baseline.png",
        }
    },
}

SCENE = {
    "version": "test-lib",
    "city_english": {"武汉": "Wuhan", "成都": "Chengdu"},
    "photography": "documentary editorial photography, natural window daylight",
    "negative": "text, watermark",
}


class FakeBackend:
    """计数用的假后端：记录每次 generate 收到的 items，产出不落真图。"""

    name = "local"

    def __init__(self) -> None:
        self.calls: list[list[GenItem]] = []

    def generate(self, items: list[GenItem], model_ids: list[str | None] | None = None):
        self.calls.append(list(items))
        return [
            GenOutcome(
                task=i.task, ok=True, node_id="fake-node",
                file_name=f"{i.task.idempotency_key}.png",
                local_path=str(Path("out/images") / f"{i.task.idempotency_key}.png"),
                seconds=1.0,
            )
            for i in items
        ]


def _env(tmp_path: Path, monkeypatch) -> dict[str, Any]:
    """把任务矩阵/场景/产物目录都指到 tmp，隔离仓库真实数据。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "tasks.yaml").write_text(
        yaml.safe_dump(MATRIX, allow_unicode=True), encoding="utf-8"
    )
    (data_dir / "scene.yaml").write_text(
        yaml.safe_dump(SCENE, allow_unicode=True), encoding="utf-8"
    )
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "PROMPT_DIR", data_dir)
    monkeypatch.setattr(taskspec, "DEFAULT_MATRIX", data_dir / "tasks.yaml")
    monkeypatch.setattr(taskspec, "DEFAULT_SCENE", data_dir / "scene.yaml")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(config, "OUT_DIR", out_dir)
    return {"data_dir": data_dir, "out_dir": out_dir}


def _spec(key: str = "k-test", character: str | None = "lin", theme: str = THEME) -> TaskSpec:
    return TaskSpec(
        city="武汉", project=PROJECT, theme=theme, channel="xiaohongshu",
        character=character, aspect_ratio="3:4", resolution="2K",
        idempotency_key=key,
    )


def _seed(
    db: Path, key: str, provider: str = "matrix", tmp_src_dir: Path | None = None,
    file_exists: bool = True, **overrides: Any
) -> None:
    """造一条"已落盘"的源产出：local_path 指向真实存在的文件（复用要复制它）。"""
    src = tmp_src_dir / f"{key}.png"
    if file_exists:
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"fake-image-bytes")
    row: dict[str, Any] = {
        "idempotency_key": key,
        "provider": provider,
        "city": "武汉", "project": PROJECT, "theme": THEME,
        "channel": "xiaohongshu", "character": "lin",
        "route": "i2i",
        "local_path": str(src),
        "gen_seconds": 12.0,
        "est_cost": 0.2,
        "qc_verdict": "accepted",
    }
    row.update(overrides)
    if not row["local_path"]:
        row.pop("local_path")
    with ledger.connect(db) as conn:
        ledger.record_asset(conn, row)


# ---------------------------------------------------------------- find_reusable 优先级


def test_find_reusable_prefers_adopted_over_plain_accepted(tmp_path) -> None:
    """同素材三行：普通 accepted / accepted+人工采纳 / pending → 取采纳那条。"""
    db = tmp_path / "t.db"
    src = tmp_path / "src"
    _seed(db, "plain", tmp_src_dir=src)                       # accepted，人没处理
    _seed(db, "picked", tmp_src_dir=src, adopted="adopted")   # accepted + 人工采纳
    _seed(db, "fresh", tmp_src_dir=src, qc_verdict="pending")
    with ledger.connect(db) as conn:
        row = library.find_reusable(conn, _spec())
    assert row is not None
    assert row["idempotency_key"] == "picked"


def test_find_reusable_takes_latest_within_same_tier(tmp_path) -> None:
    """同为 accepted 时取 created_at 最新的那条。"""
    db = tmp_path / "t.db"
    src = tmp_path / "src"
    _seed(db, "old", tmp_src_dir=src, created_at="2026-10-01T00:00:00+00:00")
    _seed(db, "new", tmp_src_dir=src, created_at="2026-10-20T00:00:00+00:00")
    with ledger.connect(db) as conn:
        row = library.find_reusable(conn, _spec())
    assert row["idempotency_key"] == "new"


def test_find_reusable_never_returns_rejected(tmp_path) -> None:
    """rejected 永不被复用：机器拦（qc rejected）与人弃（adopted rejected）都排除，
    连"adopted 但 qc rejected"的矛盾行也整行排除——宁缺毋滥。"""
    db = tmp_path / "t.db"
    src = tmp_path / "src"
    _seed(db, "qc_rej", tmp_src_dir=src, qc_verdict="rejected")
    with ledger.connect(db) as conn:
        assert library.find_reusable(conn, _spec()) is None
    _seed(db, "adopt_rej", tmp_src_dir=src, adopted="rejected", adopt_note="手指畸形")
    with ledger.connect(db) as conn:
        assert library.find_reusable(conn, _spec()) is None
    # 人工点了"用"但质检拦着：两列都要认，不能被 adopted 顶出来
    _seed(db, "mixed", tmp_src_dir=src, qc_verdict="rejected", adopted="adopted",
          provider="local")
    with ledger.connect(db) as conn:
        assert library.find_reusable(conn, _spec()) is None


def test_find_reusable_requires_file_on_disk(tmp_path) -> None:
    """失败没落盘的行（local_path NULL）无源可拷，不参与匹配。"""
    db = tmp_path / "t.db"
    _seed(db, "ghost", tmp_src_dir=tmp_path / "src", local_path=None)
    with ledger.connect(db) as conn:
        assert library.find_reusable(conn, _spec()) is None


def test_find_reusable_match_key_is_character_project_theme(tmp_path) -> None:
    """语义键三要素各管一头：换角色/换主题都不该复用；
    任务是泛化人物（character 为空）时退化到 project+theme。"""
    db = tmp_path / "t.db"
    src = tmp_path / "src"
    _seed(db, "src1", tmp_src_dir=src)
    with ledger.connect(db) as conn:
        # 别的角色：身份不同一张素材都不是
        assert library.find_reusable(conn, _spec(character="wang")) is None
        # 别的主题：叙事不同
        assert library.find_reusable(conn, _spec(theme=OTHER_THEME)) is None
        # 泛化人物任务：不限角色，project+theme 命中即可
        row = library.find_reusable(conn, _spec(character=None))
        assert row is not None and row["idempotency_key"] == "src1"


# ---------------------------------------------------------------- CLI：generate --reuse-adopted


def test_reuse_flag_writes_zero_cost_row_without_calling_backend(tmp_path, monkeypatch) -> None:
    """L1 不变量：开启复用且素材已存在时——
    假 backend.generate **一次都没被调用**，每个复用行 est_cost == 0.0，
    源文件被复制成本后端产物命名，route_reason 写清复用了哪条源。"""
    db = tmp_path / "t.db"
    env = _env(tmp_path, monkeypatch)
    _seed(db, "src-adopted", tmp_src_dir=tmp_path / "src", adopted="adopted")

    fake = FakeBackend()
    import aesthetics_pipeline.cli as cli_mod
    monkeypatch.setattr(cli_mod, "build_provider", lambda name, **kw: fake)

    res = runner.invoke(app, ["generate", "--provider", "local", "--reuse-adopted",
                              "--db", str(db)])
    assert res.exit_code == 0, res.output

    # 生成量=0：两条城市任务全被复用顶掉，后端一次没碰
    assert fake.calls == []

    specs = taskspec.expand()
    keys = {s.idempotency_key for s in specs}
    assert len(keys) == 2  # 武汉/成都两城市 × 同素材
    with ledger.connect(db) as conn:
        rows = {r["idempotency_key"]: r for r in ledger.assets(
            conn, "provider = 'local'")}
    assert set(rows) == keys
    for key, row in rows.items():
        assert row["est_cost"] == 0.0  # 本地复制零边际成本——正是要测的不变量
        assert row["gen_seconds"] == 0.0
        # 复用行归本后端、产物按本后端命名，磁盘上真有这张复制出来的文件
        assert row["file_name"] == f"{key}__local.png"
        assert Path(row["local_path"]).exists()
        assert Path(row["local_path"]).read_bytes() == b"fake-image-bytes"
        assert Path(row["local_path"]).parent == env["out_dir"] / "images"
        # 源可追溯：key + provider 都写进 route_reason
        assert "src-adopted" in row["route_reason"]
        assert "matrix" in row["route_reason"]
        assert row["qc_verdict"] == "pending"  # 判定不白拿源行的
    # 源行原样不动
    with ledger.connect(db) as conn:
        src_row = ledger.assets(conn, "provider = 'matrix'")[0]
    assert src_row["adopted"] == "adopted" and src_row["est_cost"] == 0.2


def test_reuse_rerun_is_idempotent_with_skip_existing(tmp_path, monkeypatch) -> None:
    """复用行同样进跳过口径：第二次跑 skip_existing 直接跳过，不重复落行。"""
    db = tmp_path / "t.db"
    _env(tmp_path, monkeypatch)
    _seed(db, "src1", tmp_src_dir=tmp_path / "src")

    fake = FakeBackend()
    import aesthetics_pipeline.cli as cli_mod
    monkeypatch.setattr(cli_mod, "build_provider", lambda name, **kw: fake)

    for _ in range(2):
        res = runner.invoke(app, ["generate", "--provider", "local", "--reuse-adopted",
                                  "--db", str(db)])
        assert res.exit_code == 0, res.output
    assert fake.calls == []
    with ledger.connect(db) as conn:
        assert len(ledger.assets(conn, "provider = 'local'")) == 2  # 没插重复行


def test_default_no_reuse_keeps_old_behavior(tmp_path, monkeypatch) -> None:
    """默认关：不传 --reuse-adopted 时后端照常被调、台账里没有复用行。"""
    db = tmp_path / "t.db"
    _env(tmp_path, monkeypatch)
    _seed(db, "src1", tmp_src_dir=tmp_path / "src", adopted="adopted")

    fake = FakeBackend()
    import aesthetics_pipeline.cli as cli_mod
    monkeypatch.setattr(cli_mod, "build_provider", lambda name, **kw: fake)

    res = runner.invoke(app, ["generate", "--provider", "local", "--db", str(db)])
    assert res.exit_code == 0, res.output
    # 旧路径原样：两条任务都发给后端
    assert [len(c) for c in fake.calls] == [2]
    with ledger.connect(db) as conn:
        rows = ledger.assets(conn, "provider = 'local'")
    assert len(rows) == 2
    for row in rows:
        assert "复用" not in (row["route_reason"] or "")
        assert row["est_cost"] != 0.0  # 走的是真实生成口径（未配单价则为 None）


def test_no_reusable_asset_generates_normally(tmp_path, monkeypatch) -> None:
    """开了复用但库里没有同素材合格产出（主题不同）→ 照常生成，不静默放过。"""
    db = tmp_path / "t.db"
    _env(tmp_path, monkeypatch)
    _seed(db, "src-other-theme", tmp_src_dir=tmp_path / "src", theme=OTHER_THEME)

    fake = FakeBackend()
    import aesthetics_pipeline.cli as cli_mod
    monkeypatch.setattr(cli_mod, "build_provider", lambda name, **kw: fake)

    res = runner.invoke(app, ["generate", "--provider", "local", "--reuse-adopted",
                              "--db", str(db)])
    assert res.exit_code == 0, res.output
    assert [len(c) for c in fake.calls] == [2]
    with ledger.connect(db) as conn:
        rows = ledger.assets(conn, "provider = 'local'")
    assert all("复用" not in (r["route_reason"] or "") for r in rows)


# ---------------------------------------------------------------- library 盘点命令


def test_library_command_counts_reusable_by_group(tmp_path) -> None:
    """`aesthetics library` 按 角色×项目×主题 汇总落盘/放行/采纳，rejected 与未落盘不进账。"""
    db = tmp_path / "t.db"
    src = tmp_path / "src"
    _seed(db, "a1", tmp_src_dir=src, adopted="adopted")
    _seed(db, "a2", tmp_src_dir=src, provider="local")
    _seed(db, "rej", tmp_src_dir=src, qc_verdict="rejected")       # 拦掉的不算资产
    _seed(db, "gone", tmp_src_dir=src, local_path=None)            # 没落盘的不算
    _seed(db, "t2", tmp_src_dir=src, theme=OTHER_THEME)

    res = runner.invoke(app, ["library", "--db", str(db)])
    assert res.exit_code == 0, res.output
    assert PROJECT in res.output and THEME in res.output
    assert "lin" in res.output
    assert "合计 3 份可复用素材（其中已采纳 1）" in res.output
    # rejected / 未落盘的键不出现在盘点里
    assert "rej" not in res.output and "gone" not in res.output
