"""回放缓存：路径可移植（F3）+ 后端命名空间（F1 同类缺陷）。"""

from __future__ import annotations

import json
from pathlib import Path

from aesthetics_pipeline import config
from aesthetics_pipeline.provider.cache import (
    cache_path,
    read_cache,
    relative_to_repo,
    resolve_local_path,
    write_cache,
)


def _write(cache_dir: Path, provider: str, key: str = "k1") -> Path:
    target = config.REPO_ROOT / "out" / "images" / f"{provider}_{key}.jpg"
    return write_cache(
        cache_dir,
        key=key,
        provider=provider,
        node_id="n1",
        file_name=f"{provider}_{key}.jpg",
        prompt="p",
        path="i2i",
        seconds=1.0,
        local_path=str(target),
    )


def test_cache_path_is_repo_relative_and_resolves_back(tmp_path) -> None:
    """缓存里存仓库相对路径，随仓库搬到别的机器仍能读。"""
    cache_file = _write(tmp_path, "matrix")
    entry = json.loads(cache_file.read_text(encoding="utf-8"))
    assert entry["local_path"] == "out/images/matrix_k1.jpg"
    assert resolve_local_path(entry["local_path"]) == config.REPO_ROOT / "out" / "images" / "matrix_k1.jpg"


def test_absolute_cache_path_still_resolves(tmp_path) -> None:
    """旧缓存写的是绝对路径，也要能读（不能因为改了格式就废掉已有产物）。

    这里必须用**当前平台的绝对路径**：写死 `C:/tmp/x.png` 测的其实是本机路径语法
    ——它在 Windows 上是绝对路径，在 POSIX 上是相对路径（会被锚回仓库根），
    同一份断言在两种系统上含义相反，CI（Linux）会因此红。
    """
    raw = str(tmp_path / "legacy.png")
    assert Path(raw).is_absolute(), raw
    assert resolve_local_path(raw) == Path(raw)


def test_two_providers_same_key_do_not_overwrite(tmp_path) -> None:
    """同一幂等键被两个后端产出时，缓存文件必须分开（与换脸产物同一个坑）。"""
    a = _write(tmp_path, "matrix")
    b = _write(tmp_path, "local")
    assert a != b
    assert a.parent.name == "matrix" and b.parent.name == "local"
    assert json.loads(a.read_text(encoding="utf-8"))["provider"] == "matrix"
    assert json.loads(b.read_text(encoding="utf-8"))["provider"] == "local"


def test_read_cache_by_provider_does_not_cross_lines(tmp_path) -> None:
    """指定后端时只认该后端的条目，不拿别路线的图冒充。"""
    _write(tmp_path, "matrix")
    _write(tmp_path, "local")
    assert read_cache(tmp_path, "k1", "matrix")["provider"] == "matrix"
    assert read_cache(tmp_path, "k1", "local")["provider"] == "local"
    assert read_cache(tmp_path, "k1", "unknown") is None


def test_legacy_flat_cache_is_still_readable(tmp_path) -> None:
    """旧版扁平缓存（cache/<key>.json）不能因为改了布局就废掉。"""
    flat = tmp_path / "k1.json"
    flat.write_text(json.dumps({"provider": "matrix", "local_path": "out/x.jpg"}), encoding="utf-8")
    assert read_cache(tmp_path, "k1", "matrix")["local_path"] == "out/x.jpg"


def test_cache_path_layout() -> None:
    assert cache_path(Path("c"), "k", "matrix").as_posix() == "c/matrix/k.json"
    assert cache_path(Path("c"), "k").as_posix() == "c/k.json"


# ---------------- http_api 进回放口径 + 档位随行 ----------------


def test_replay_finds_http_api_entry_and_restores_model_id(tmp_path) -> None:
    """交付主线（http_api）的缓存必须能被默认回放口径找到，
    且 outcome 带上 model_id——否则断网兜底对交付线名不副实、台账丢档位。"""
    from aesthetics_pipeline.models import GenItem, TaskSpec
    from aesthetics_pipeline.provider.replay import ReplayProvider

    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake-bytes")
    cache_dir = tmp_path / "out" / "cache"
    write_cache(
        cache_dir, key="k9", provider="http_api", node_id="t-1",
        file_name="img.png", local_path=str(img), prompt="p", path="i2i",
        seconds=12.3, model_id="qwen-image-3.0-pro", portable=False,
    )
    spec = TaskSpec(
        city="武汉", project="光子嫩肤", theme="术后修复", channel="xiaohongshu",
        character="lin", aspect_ratio="3:4", resolution="2K", idempotency_key="k9",
    )
    item = GenItem(task=spec, prompt="p", negative_prompt="", aspect_ratio="3:4",
                   resolution="2K", input_urls=[])
    outcomes = ReplayProvider(out_dir=tmp_path / "out").generate([item])
    assert len(outcomes) == 1 and outcomes[0].ok, outcomes[0].error
    assert outcomes[0].source_provider == "http_api"
    assert outcomes[0].model_id == "qwen-image-3.0-pro"


def test_matrix_still_wins_over_http_api_in_provider_order(tmp_path) -> None:
    """补漏不改既有裁决：matrix 与 http_api 条目并存时回放仍优先 matrix，
    matrix/local 的相对优先级一字未动。"""
    _write(tmp_path, "matrix")
    _write(tmp_path, "http_api")
    assert read_cache(tmp_path, "k1")["provider"] == "matrix"
