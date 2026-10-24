"""合规校验与发布包：免责声明、图文成对、画面文字、禁用词、fail-closed 打包。"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from aesthetics_pipeline import compliance, ledger
from aesthetics_pipeline.pack import load_pack

DISCLAIMER = load_pack("medical-aesthetics").compliance["required_disclaimer"]


@pytest.fixture()
def env(tmp_path: Path):
    """隔离台账 + 一张真实存在的占位图。"""
    db = tmp_path / "pipeline.db"
    img = tmp_path / "out" / "k1__api.png"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    return {"db": db, "img": img, "tmp": tmp_path}


def _add_asset(conn, img: Path, *, key="k1", verdict="accepted", text_found=None, character=None):
    ledger.record_asset(
        conn,
        {
            "idempotency_key": key,
            "provider": "http_api",
            "model_id": "doubao-seedream-4-0",
            "city": "武汉",
            "project": "水光针补水",
            "theme": "术后第7天真实反馈",
            "channel": "xiaohongshu",
            "character": character,
            "route": "i2i",
            "local_path": str(img),
            "prompt": "a photo of a woman in a clinic",
            "gen_seconds": 10.0,
            "est_cost": 0.2,
            "qc_verdict": verdict,
            "qc_reasons": [],
            "text_found": text_found or [],
            "adopted": "adopted",
        },
    )


def _add_copy(conn, *, key="k1", body: str, banned=None):
    ledger.record_copy(
        conn,
        {
            "idempotency_key": key,
            "title": "武汉水光针补水｜术后记录",
            "body": body,
            "tags": ["#武汉医美"],
            "banned_hits": banned or [],
        },
    )


def _run(env, *, body=None, banned=None, text_found=None, verdict="accepted", with_copy=True):
    with ledger.connect(env["db"]) as conn:
        _add_asset(conn, env["img"], verdict=verdict, text_found=text_found)
        if with_copy:
            _add_copy(conn, body=body if body is not None else DISCLAIMER, banned=banned)
        return compliance.audit(conn, pack=load_pack("medical-aesthetics"))


def test_clean_asset_passes(env) -> None:
    result = _run(env)
    assert result["ok"] is True
    assert result["checked"] == 1
    assert result["issues"] == []


def test_missing_disclaimer_is_blocking(env) -> None:
    result = _run(env, body="武汉水光针补水｜术后第 7 天的真实状态记录")
    assert result["ok"] is False
    assert result["by_code"] == {"MISSING_DISCLAIMER": 1}


def test_missing_copy_is_blocking(env) -> None:
    result = _run(env, with_copy=False)
    assert result["by_code"] == {"NO_COPY": 1}


def test_text_in_image_is_blocking(env) -> None:
    result = _run(env, text_found=["10月24日"])
    assert result["by_code"] == {"TEXT_IN_IMAGE": 1}


def test_banned_hit_is_blocking(env) -> None:
    result = _run(env, banned=["保证(疗效承诺)"])
    assert result["by_code"] == {"BANNED_HIT": 1}


def test_rejected_assets_are_not_audited(env) -> None:
    result = _run(env, verdict="rejected", with_copy=False)
    assert result["checked"] == 0
    assert result["ok"] is True


def test_adopted_only_filter(env) -> None:
    with ledger.connect(env["db"]) as conn:
        _add_asset(conn, env["img"], key="k1")
        _add_copy(conn, key="k1", body=DISCLAIMER)
        ledger.record_asset(conn, {"idempotency_key": "k1", "provider": "http_api", "adopted": "rejected"})
        result = compliance.audit(conn, pack=load_pack("medical-aesthetics"), adopted_only=True)
    assert result["checked"] == 0


def test_export_refuses_when_violations(env) -> None:
    with ledger.connect(env["db"]) as conn:
        _add_asset(conn, env["img"])
        _add_copy(conn, body="没有免责声明")
        with pytest.raises(ValueError, match="合规校验未通过"):
            compliance.export_publish(
                conn, env["tmp"] / "publish.zip", pack=load_pack("medical-aesthetics")
            )


def test_export_package_contents(env) -> None:
    out = env["tmp"] / "publish.zip"
    with ledger.connect(env["db"]) as conn:
        _add_asset(conn, env["img"])
        _add_copy(conn, body=DISCLAIMER)
        path = compliance.export_publish(
            conn, out, pack=load_pack("medical-aesthetics"), batch="demo"
        )
    assert path == out and out.exists()
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names and "compliance.json" in names and "README.txt" in names
        assert any(n.startswith("images/") for n in names)
        assert any(n.startswith("copy/") for n in names)
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["asset_count"] == 1
        assert manifest["disclaimer"] == DISCLAIMER
        assert manifest["assets"][0]["ai_label"]["implicit_metadata"] is True
        assert manifest["assets"][0]["copy"]["title"].startswith("武汉")
        readme = zf.read("README.txt").decode("utf-8")
        assert "AI 生成声明" in readme


def test_export_allow_violations_writes_compliance(env) -> None:
    out = env["tmp"] / "publish-allow.zip"
    with ledger.connect(env["db"]) as conn:
        _add_asset(conn, env["img"])
        _add_copy(conn, body="没有免责声明")
        compliance.export_publish(
            conn,
            out,
            pack=load_pack("medical-aesthetics"),
            allow_violations=True,
        )
    with zipfile.ZipFile(out) as zf:
        result = json.loads(zf.read("compliance.json"))
    assert result["ok"] is False
    assert "MISSING_DISCLAIMER" in result["by_code"]


def test_pack_without_compliance_only_checks_generic_rules(env) -> None:
    """零售包的声明不同，不能拿医美的声明去要求它。"""
    retail = load_pack("example-retail")
    with ledger.connect(env["db"]) as conn:
        _add_asset(conn, env["img"])
        _add_copy(conn, body=retail.compliance["required_disclaimer"])
        result = compliance.audit(conn, pack=retail)
    assert result["ok"] is True
    assert result["rules"]["required_disclaimer"] == retail.compliance["required_disclaimer"]


def test_medical_pack_copy_passes_its_own_banned_word_gate() -> None:
    """合规声明本身不能命中禁用词表——否则每篇文案都会自相矛盾。

    实测踩过：早期 disclaimer 里写了"疗效承诺"，"承诺"在词表里，
    结果合规文案被自己的闸门拦下。
    """
    from aesthetics_pipeline import copy as copy_mod

    pack = load_pack("medical-aesthetics")
    assert copy_mod.scan(DISCLAIMER, pack=pack) == []
    for channel in ("xiaohongshu", "wechat"):
        draft = copy_mod.make_copy(
            city="武汉",
            project="水光针补水",
            theme="术后第7天真实反馈",
            copy_angle="术后第 7 天的真实状态记录",
            channel=channel,
            pack=pack,
        )
        assert draft.banned_hits == [], f"{channel} 命中 {draft.banned_hits}"
        assert DISCLAIMER in draft.body


# ---------------------------------------------------------------- 角色授权（商用红线）

def _fake_pack(licenses: dict) -> SimpleNamespace:
    return SimpleNamespace(
        name="test-pack",
        licenses=licenses,
        compliance=load_pack("medical-aesthetics").compliance,
    )


ROW = {"idempotency_key": "k1", "provider": "http_api", "character": "lin"}


def test_missing_license_is_blocking() -> None:
    issues = compliance.check_character_license(ROW, _fake_pack({}))
    assert [i.code for i in issues] == ["MISSING_LICENSE"]


def test_non_commercial_license_is_blocking() -> None:
    pack = _fake_pack({"characters": {"lin": {"commercial_use": False, "source": "real"}}})
    assert [i.code for i in compliance.check_character_license(ROW, pack)] == [
        "LICENSE_NOT_COMMERCIAL"
    ]


def test_expired_license_is_blocking() -> None:
    pack = _fake_pack(
        {"characters": {"lin": {"commercial_use": True, "valid_until": "2000-01-01"}}}
    )
    assert [i.code for i in compliance.check_character_license(ROW, pack)] == ["LICENSE_EXPIRED"]


def test_role_without_character_skips_license_check() -> None:
    row = {"idempotency_key": "k2", "provider": "http_api", "character": None}
    assert compliance.check_character_license(row, _fake_pack({})) == []


def test_medical_pack_has_license_for_every_character() -> None:
    """发布前的静态底线：矩阵用到的每个角色都必须出现在 licenses.yaml。"""
    pack = load_pack("medical-aesthetics")
    characters = set(pack.matrix.get("characters") or {})
    licensed = set(pack.licenses.get("characters") or {})
    assert characters and characters <= licensed


def test_audit_blocks_missing_license(env) -> None:
    with ledger.connect(env["db"]) as conn:
        _add_asset(conn, env["img"], character="lin")
        _add_copy(conn, body=DISCLAIMER)
        result = compliance.audit(conn, pack=_fake_pack({}))
    assert "MISSING_LICENSE" in result["by_code"]
    assert result["ok"] is False
