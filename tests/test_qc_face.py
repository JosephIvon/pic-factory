"""身份判定阶段化（F2）+ 换脸产物命名空间（F1）+ 选脸规则与通道顺序。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from aesthetics_pipeline import config
from aesthetics_pipeline.postprocess.faceswap import swapped_output_path
from aesthetics_pipeline.qc import face
from aesthetics_pipeline.qc.face import (
    face_gate_unavailable_reason,
    missing_face_reason,
    should_reject_on_identity,
)


def test_generate_phase_never_rejects_on_identity() -> None:
    """生成阶段只用阈值标记漂移，不拦图。

    实测（修正选脸与通道顺序后）：同一人的**生成图** 0.2022~0.7350、
    不同人的生成图 0.3182~0.4922，两组在「生成图 vs 定妆照」口径上重叠，
    拿这个阈值拦中间产物会误杀。
    """
    low = {"similarity": 0.2022, "threshold": 0.35, "skipped": False}
    assert should_reject_on_identity(low, "generate") is False


def test_post_swap_phase_rejects_below_threshold() -> None:
    """出厂判定放在换脸之后：仍低于阈值才拒收。"""
    assert should_reject_on_identity({"similarity": 0.2403, "threshold": 0.35}, "post_swap") is True
    assert should_reject_on_identity({"similarity": 0.8069, "threshold": 0.35}, "post_swap") is False


def test_identity_check_never_fabricates_a_score() -> None:
    """检不到人脸/模型不可用时**不伪造分数**；出厂阶段按"闸门没跑"拒收。

    口径（2026-10-24 复审后改）：出厂判定跑在**最终产物**上，
    "换脸换不出脸"或"人脸模型不可用"都意味着最终产物**没被校验过**——
    与 OCR 闸「文字闸未执行，不能记为合格」同口径，不能出厂放行。
    生成阶段仍然不拦（那里本来就不因身份拒收，只标记漂移）。
    """
    # 生成阶段：仍不因身份拒收
    assert should_reject_on_identity({"skipped": True, "similarity": None}, "generate") is False
    # 出厂阶段：闸门没跑 → 拒收（不能当成合格出厂）
    assert should_reject_on_identity({"skipped": True, "similarity": None}, "post_swap") is True
    # 没标 skipped 又没有分数 → 没有判定可依据，仍不拒收（不凭空造判定）
    assert should_reject_on_identity({"similarity": None, "threshold": 0.35}, "post_swap") is False


def test_face_gate_unavailable_reason_separates_three_cases() -> None:
    """人物任务"没法校验身份"要分三种说法，不能一律说成"没有人脸"。"""
    # 模型不可用（count=None）→ 闸门根本没跑
    reason = face_gate_unavailable_reason(True, {"skipped": True, "count": None})
    assert reason is not None and "人脸闸未执行" in reason
    # 画面里确实没有人脸 → missing_face_reason 的活，这里不重复说
    assert face_gate_unavailable_reason(True, {"skipped": True, "count": 0}) is None
    # 参考照检不到人脸
    reason = face_gate_unavailable_reason(True, {"skipped": True, "count": 2})
    assert reason is not None and "参考照" in reason
    # 非人物任务不适用：场景图不需要身份校验，不能因缺人脸模型被整批拒收
    assert face_gate_unavailable_reason(False, {"skipped": True, "count": None}) is None
    # 闸门正常跑出判定 → 没有拒收理由
    assert face_gate_unavailable_reason(True, {"skipped": False, "count": 1}) is None


def test_character_task_is_not_accepted_when_the_face_gate_cannot_run(
    tmp_path, monkeypatch
) -> None:
    """人物任务 + 人脸模型不可用 → **不能记为合格**（与 OCR 闸同口径）。

    锁的是"闸门没跑 ≠ 合格"：这里以前放行，等于把没验过身份的图标成合格出厂。
    """
    from typer.testing import CliRunner

    from aesthetics_pipeline import ledger
    from aesthetics_pipeline.cli import app

    monkeypatch.setattr(face, "_detect", lambda p: None)  # 模拟人脸模型不可用
    db = tmp_path / "t.db"
    with ledger.connect(db) as conn:
        ledger.record_asset(
            conn,
            {
                "idempotency_key": "k1",
                "provider": "matrix",
                "city": "武汉",
                "project": "热玛吉",
                "character": "lin",
                "route": "i2i",
                "local_path": str(config.EVIDENCE_DIR / "01_t2i_baseline.png"),
            },
        )
    result = CliRunner().invoke(app, ["qc", "--db", str(db)])
    assert result.exit_code == 0, result.output
    with ledger.connect(db) as conn:
        row = ledger.assets(conn, "idempotency_key = 'k1'")[0]
    reasons = json.loads(row["qc_reasons"] or "[]")
    assert row["qc_verdict"] == "rejected", reasons
    assert any("人脸闸未执行" in r for r in reasons), reasons


def test_factory_reject_reason_distinguishes_failure_modes() -> None:
    """出厂拒收理由要能区分"没检成"与"检了不合格"。"""
    from aesthetics_pipeline.cli import _factory_reject_reason

    assert "人脸闸未执行" in _factory_reject_reason({"skipped": True, "count": None}, 0.57)
    assert "换脸未生效" in _factory_reject_reason({"skipped": True, "count": 0}, 0.57)
    assert "参考照" in _factory_reject_reason({"skipped": True, "count": 3}, 0.57)
    assert "低于阈值" in _factory_reject_reason(
        {"skipped": False, "similarity": 0.4}, 0.57
    )


def test_character_task_without_a_face_is_rejected() -> None:
    """人物任务画面里没有人脸 = 没法校验身份，不能因为"检不到就跳过"而放行。

    实测教训：通道顺序错了（RGB 传进 insightface）时，一张大笑的人脸特写
    被判成"画面无人脸"，于是身份校验整段跳过 → 放行。
    """
    assert missing_face_reason(True, 0) is not None
    assert missing_face_reason(True, 1) is None
    # 非人物任务不适用这条规则
    assert missing_face_reason(False, 0) is None
    # count 为 None = 人脸模型不可用（没法判断），不当成"没有人脸"，不误杀
    assert missing_face_reason(True, None) is None


def test_similarity_picks_the_face_that_matches_the_reference(monkeypatch) -> None:
    """多脸图要取「与参考照最相似」的那张，不是「检测分最高」的那张。

    实测：evidence/batch/batch_03_chengdu_photon.jpg 里有两张脸，
    检测分最高的是背景里一张很小的脸（area 0.0054），用它比对得 0.2237，
    正确选脸得 0.4429 —— 旧规则把"不同人"压到了"像同一人"的区间。
    """
    faces = {
        "candidate": [
            {"emb": np.array([1.0, 0.0]), "det_score": 0.99, "area": 0.01},
            {"emb": np.array([0.0, 1.0]), "det_score": 0.80, "area": 0.20},
        ],
        "reference": [{"emb": np.array([0.0, 1.0]), "det_score": 0.5, "area": 0.5}],
    }
    monkeypatch.setattr(face, "_detect", lambda p: faces[p])
    assert face.similarity("candidate", "reference") == 1.0


def test_detect_converts_to_bgr_before_handing_to_insightface(monkeypatch, tmp_path) -> None:
    """insightface 的检测器按 OpenCV 惯例训练（输入是 BGR）。

    实测：传 RGB 会让整张脸都检不到（out/local_images/a004118539d12515.png：
    RGB 检 0 张 / BGR 检 1 张）。换脸那条链路用 cv2.imread（本来就是 BGR），
    两边必须一致，否则质检和换脸对同一张图的判断会打架。
    """
    from PIL import Image

    path = tmp_path / "x.png"
    array = np.dstack(
        [
            np.full((4, 4, 1), 10, np.uint8),  # R
            np.full((4, 4, 1), 20, np.uint8),  # G
            np.full((4, 4, 1), 30, np.uint8),  # B
        ]
    )
    Image.fromarray(array).save(path)

    class FakeApp:
        def __init__(self) -> None:
            self.captured = None

        def get(self, img):
            self.captured = img
            return []

    fake = FakeApp()
    monkeypatch.setattr(face, "_face_app", lambda: fake)
    face._detect(path)
    assert fake.captured is not None
    assert fake.captured[0, 0, 0] == 30  # BGR 第一通道是蓝
    assert fake.captured[0, 0, 2] == 10  # 最后一通道是红


def test_swapped_output_path_is_namespaced_by_provider() -> None:
    """matrix 与 local 共用同一幂等键，磁盘路径必须分开，否则互相覆盖。"""
    out = config.OUT_DIR
    a = swapped_output_path("matrix", "key123", out)
    b = swapped_output_path("local", "key123", out)
    assert a != b
    assert a.parent.name == "matrix" and b.parent.name == "local"
    assert a.name == b.name == "key123.png"


def test_swapped_output_path_is_namespaced_by_model() -> None:
    """同一后端换模型跑同一任务：换脸产物也不能互相覆盖（多档并存）。"""
    out = config.OUT_DIR
    a = swapped_output_path("http_api", "key123", out, "doubao-seedream-4-0")
    b = swapped_output_path("http_api", "key123", out, "qwen-image-3.0-pro")
    assert a != b
    assert "doubao-seedream-4-0" in a.name and "qwen-image-3.0-pro" in b.name


def test_default_threshold_is_calibrated_and_attributable(monkeypatch) -> None:
    """阈值默认值必须挂在本项目同域样本证据上。

    历史上默认 0.35 自认"未经标定"——经不起"这个数怎么来的"的追问。
    现在默认值来自 tools/recalibrate_face_threshold.py（生产尺、同域难负间隙
    中点 0.57），threshold_source 必须能报出证据文件；env 覆盖优先级不变。
    """
    monkeypatch.delenv("AP_FACE_THRESHOLD", raising=False)
    gate = face.FaceGate()
    assert gate.threshold == face.CALIBRATED_THRESHOLD == 0.57
    assert "标定" in gate.threshold_source
    assert "threshold-calibration-production.json" in gate.threshold_source

    monkeypatch.setenv("AP_FACE_THRESHOLD", "0.61")
    override = face.FaceGate()
    assert override.threshold == 0.61
    assert override.threshold_source == "env AP_FACE_THRESHOLD"
