"""台账里的绝对路径在仓库搬过家之后还能不能解析。

背景：台账记的是生成当时的**绝对路径**。仓库被整体拷到别的机器（或交付到内网环境
把文件夹拷过去）后，这些路径全部失效，画廊会变成一片空图——图还在，只是找不到。

修法：按仓库内子目录前缀（out/ evidence/ fixtures/）把路径重新锚定到当前仓库根。
"""

from __future__ import annotations

from pathlib import Path

from aesthetics_pipeline import config, ledger


def _repo(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    return tmp_path


def test_existing_path_is_returned_as_is(tmp_path, monkeypatch) -> None:
    _repo(tmp_path, monkeypatch)
    target = tmp_path / "out" / "images" / "a.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    assert config.resolve_repo_path(str(target)) == target


def test_path_is_reattached_to_current_repo_after_moving(tmp_path, monkeypatch) -> None:
    _repo(tmp_path, monkeypatch)
    target = tmp_path / "out" / "images" / "a.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    stale = "C:/old/place/workspace/out/images/a.png"
    assert config.resolve_repo_path(stale) == target


def test_unresolvable_path_returns_none(tmp_path, monkeypatch) -> None:
    """解析不出来就返回 None —— 不猜、不静默按当前目录找。"""
    _repo(tmp_path, monkeypatch)
    assert config.resolve_repo_path("C:/nowhere/x.png") is None
    assert config.resolve_repo_path(None) is None
    assert config.resolve_repo_path("") is None


def test_ledger_asset_path_uses_the_resolver(tmp_path, monkeypatch) -> None:
    _repo(tmp_path, monkeypatch)
    original = tmp_path / "out" / "images" / "k1.png"
    swapped = tmp_path / "out" / "swapped" / "matrix" / "k1.png"
    for p in (original, swapped):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    row = {
        # 仓库搬过家：local_path 指向旧位置
        "local_path": "C:/old/place/workspace/out/images/k1.png",
        "swapped_path": str(swapped),
    }
    assert ledger.asset_path(row) == original
    assert ledger.asset_path(row, "swapped_path") == swapped
