"""文档数字防漂移门禁的自测。

这个门禁本身是 CI 里唯一能挡住"文档数字悄悄过期"的东西，
所以它必须有这些回归：
  1. 仓库里的真实文档必须全过（否则门禁会被人手工改成放行）；
  2. 人为制造一处漂移，门禁必须抓出来（否则门禁可能是恒真的）；
  3. 文档缺失在内仓口径必须红、公开口径跳过（防检查集静默缩水）；
  4. internal 检查项只在内仓口径生效。

本仓有两棵树共用这套测试：内仓（README/DEMO 齐全）与公开树（另一份 README、无 DEMO）。
按 DEMO.md 是否在场自动判定树型、选对应口径——测试在两棵树上都必须真实跑过。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TREE = not (ROOT / "DEMO.md").is_file()


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_doc_numbers", ROOT / "tools" / "check_doc_numbers.py"
    )
    module = importlib.util.module_from_spec(spec)
    # 必须先登记进 sys.modules：模块里用了 @dataclass，
    # dataclasses 解析注解时会回查 sys.modules[cls.__module__]。
    sys.modules["check_doc_numbers"] = module
    spec.loader.exec_module(module)
    return module


def test_repo_docs_are_consistent() -> None:
    """仓库里的文档必须与当前树型的口径对得上——这就是门禁本身。"""
    mod = _load()
    failures = mod.run_checks(mod._docs(), public=PUBLIC_TREE)
    assert not failures, "文档数字漂移：\n" + "\n".join(failures)


def test_gate_catches_a_drifted_number() -> None:
    """把某个在跑检查的期望片段改没，门禁必须报出来——证明它不是恒真的。"""
    mod = _load()
    docs = mod._docs()
    in_scope = [c for c in mod.CHECKS if not (c.internal and PUBLIC_TREE)]
    check = next(c for c in in_scope if c.expect)
    want = mod._norm(check.expect.format(v=check.value() if check.value else ""))
    assert want in mod._norm(docs[check.doc]), f"门禁自己先失效了：{check.name}"
    docs[check.doc] = docs[check.doc].replace(want, "被改坏的数字")
    failures = mod.run_checks(docs, public=PUBLIC_TREE)
    assert failures, "门禁没抓到漂移——它可能是恒真的"


def test_gate_catches_a_revived_stale_claim() -> None:
    """被推翻的说法（forbid 项）一旦复活，门禁必须报出来。"""
    mod = _load()
    docs = mod._docs()
    in_scope = [c for c in mod.CHECKS if not (c.internal and PUBLIC_TREE)]
    check = next(c for c in in_scope if c.forbid)
    docs[check.doc] = docs[check.doc] + "\n" + check.forbid
    failures = mod.run_checks(docs, public=PUBLIC_TREE)
    assert failures, "门禁没抓到复活的旧说法——forbid 项可能没生效"


def test_missing_doc_fails_by_default_but_skips_in_public_mode() -> None:
    """公开树没有内部文档：默认口径缺失即红（防检查集静默缩水），
    --public 口径跳过内部版检查项且不误伤其余检查。"""
    mod = _load()
    docs = mod._docs()
    victim = next(c for c in mod.CHECKS if c.doc != "README.md")
    docs[victim.doc] = None
    assert mod.run_checks(docs), "默认口径下文档缺失必须红"
    assert not mod.run_checks(docs, public=True), (
        "--public 口径下内部/缺失检查应跳过，且不能牵连其余检查项"
    )
    assert victim.doc in mod.missing_docs(docs)


@pytest.mark.skipif(PUBLIC_TREE, reason="internal 检查项的标定文档只在内仓成立")
def test_internal_scope_only_fails_on_master_mode() -> None:
    """删掉一条只在内仓 README 成立的声明：内仓口径红，公开口径过（该检查是 internal）；
    删掉两树共有的主张：两种口径都红。"""
    mod = _load()
    docs = mod._docs()
    internal = next(c for c in mod.CHECKS if c.internal and c.expect and c.value)
    want = mod._norm(internal.expect.format(v=internal.value()))
    drifted = dict(docs)
    drifted[internal.doc] = mod._norm(docs[internal.doc]).replace(want, "已删除的内部声明")
    assert mod.run_checks(drifted), "内仓口径必须抓到内部声明缺失"
    assert not mod.run_checks(drifted, public=True), "公开口径应跳过 internal 检查"

    universal = next(c for c in mod.CHECKS
                     if c.doc == "README.md" and not c.internal and c.expect)
    drifted2 = dict(docs)
    drifted2[universal.doc] = drifted2[universal.doc].replace(
        mod._norm(universal.expect), "已删除的共有主张")
    assert mod.run_checks(drifted2, public=True), "两树共有的主张在公开口径也必须红"
