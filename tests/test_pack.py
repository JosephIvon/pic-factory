"""域包（Domain Pack）：加载、校验、与旧路径的等价性。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aesthetics_pipeline import copy as copy_mod
from aesthetics_pipeline import taskspec
from aesthetics_pipeline.pack import (
    active_pack_name,
    available_packs,
    load_pack,
    try_load_pack,
)


@pytest.fixture(autouse=True)
def _no_active_pack(monkeypatch):
    """默认让每个用例从"未启用域包"的干净状态开始。"""
    monkeypatch.delenv("AP_PACK", raising=False)
    monkeypatch.delenv("AP_PACKS_DIR", raising=False)


def test_available_packs_lists_medical_and_example() -> None:
    names = available_packs()
    assert "medical-aesthetics" in names
    assert "example-retail" in names


def test_no_env_means_no_active_pack() -> None:
    assert active_pack_name() is None
    assert try_load_pack() is None


def test_medical_pack_reuses_repo_matrix() -> None:
    """医美包引用仓库既有文件，不能是复制品——否则矩阵会漂移。"""
    pack = load_pack("medical-aesthetics")
    assert pack.matrix == taskspec.load_yaml(taskspec.DEFAULT_MATRIX)
    assert pack.scene == taskspec.load_yaml(taskspec.DEFAULT_SCENE)
    assert pack.compliance.get("required_disclaimer")


def test_expand_with_medical_pack_equals_legacy(monkeypatch) -> None:
    legacy = [s.idempotency_key for s in taskspec.expand()]
    monkeypatch.setenv("AP_PACK", "medical-aesthetics")
    via_pack = [s.idempotency_key for s in taskspec.expand()]
    assert via_pack == legacy


def test_expand_with_retail_pack_is_self_contained(monkeypatch) -> None:
    monkeypatch.setenv("AP_PACK", "example-retail")
    specs = taskspec.expand()
    assert {s.city for s in specs} == {"上海", "杭州"}
    assert {s.channel for s in specs} == {"wechat", "xiaohongshu"}
    assert all(s.character is None for s in specs)
    assert len(specs) == 2 * 2 * 2 * 2


def test_retail_pack_uses_its_own_copy_templates(monkeypatch) -> None:
    pack = load_pack("example-retail")
    draft = copy_mod.make_copy(
        city="上海",
        project="新品上市",
        theme="门店实拍",
        copy_angle="门店实拍记录",
        channel="wechat",
        pack=pack,
    )
    assert "上海" in draft.title
    assert "门店实拍记录" in draft.title
    assert draft.banned_hits == []


def test_retail_pack_has_narrower_banned_words(monkeypatch) -> None:
    """医美词表不该泄漏到零售包：零售包里没有"疗效承诺"这类条目。"""
    pack = load_pack("example-retail")
    assert copy_mod.banned_words(pack) != copy_mod.BANNED_WORDS
    assert copy_mod.scan("保证一次见效", pack=pack) == []
    assert copy_mod.scan("全网最低价", pack=pack) != []


def _write_pack(root: Path, *, matrix: dict, copy_cfg: dict | None = None) -> None:
    p = root / "tmp-pack"
    p.mkdir(parents=True)
    (p / "pack.yaml").write_text(
        yaml.safe_dump({"name": "tmp-pack", "matrix": "matrix.yaml", "scene": "scene.yaml",
                        "copy": "copy.yaml" if copy_cfg else None}),
        encoding="utf-8",
    )
    (p / "matrix.yaml").write_text(yaml.safe_dump(matrix, allow_unicode=True), encoding="utf-8")
    (p / "scene.yaml").write_text(yaml.safe_dump({"photography": "x", "negative": "y"}), encoding="utf-8")
    if copy_cfg:
        (p / "copy.yaml").write_text(yaml.safe_dump(copy_cfg, allow_unicode=True), encoding="utf-8")


VALID_MATRIX = {
    "version": "1",
    "cities": ["甲"],
    "projects": {"P": {"visual": "v", "use_character": False}},
    "themes": {"T": {"visual": "v", "copy_angle": "c"}},
    "channels": {"wechat": {"aspect_ratio": "16:9", "resolution": "2K"}},
    "characters": {},
}


def test_missing_dimension_fails_fast(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AP_PACKS_DIR", str(tmp_path))
    bad = dict(VALID_MATRIX)
    bad.pop("channels")
    _write_pack(tmp_path, matrix=bad)
    with pytest.raises(ValueError, match="channels"):
        load_pack("tmp-pack")


def test_unknown_placeholder_fails_at_load(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AP_PACKS_DIR", str(tmp_path))
    _write_pack(
        tmp_path,
        matrix=VALID_MATRIX,
        copy_cfg={"templates": {"default": {"title": "{cities}", "body": "{project}"}}},
    )
    with pytest.raises(ValueError, match="未知占位符"):
        load_pack("tmp-pack")


def test_missing_pack_gives_actionable_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AP_PACKS_DIR", str(tmp_path))
    monkeypatch.setenv("AP_PACK", "nope")
    with pytest.raises(FileNotFoundError, match="nope"):
        load_pack()
