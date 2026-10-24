"""后置人脸替换：人物一致性的工业解法。

为什么需要这一步（实测结论，不是想当然）：

  在**「生成图 vs 定妆照」**这个口径上，单一余弦阈值**不具判别力**：
  同一人的**生成图**落在 0.2022~0.7350，不同人的**生成图**落在 0.3182~0.4922，
  两组重叠 —— 拿它拦中间产物会误杀（复审纠正过这一点，早先的说法是
  "阈值根本没用"，那是把"生成方差"错说成了"模型分不出人"）。

  但模型**分得出人**：同一人的**真人照片**（人工 I2I，不是生成）能到 0.6551，
  明显高于不同人的真人照片 0.3463。
  真正的问题是**生成方差**：模型没把同一张脸稳定地画出来，分数里混进了
  姿态/光照/构图/风格/清晰度的差异。

  所以解法不是"调阈值"，而是把身份**确定性地施加**上去：

  层1 生成阶段：用参考图 + 冻结的 identity_block，把脸"引"到正确方向
  层2 后处理阶段：把定妆照的脸**确定性贴回**成品，身份从此不靠概率

  换脸后身份分实测收敛到 0.7250~0.9023（换脸前 0.2611~0.7350），
  且**与后端无关** —— API 线与本地栈线落在同一区间。

代价：inswapper_128.onnx 仅供非商业研究使用（GitHub release v0.7），
商业化前需换授权模型或取得许可 —— 这条必须写进交付说明，不能藏。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

MODEL_NAME = "inswapper_128.onnx"
MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx"


def swapped_output_path(
    provider: str, key: str, out_dir: Path, model_id: str | None = None
) -> Path:
    """换脸产物的落盘路径。

    **provider 与档位都必须进路径**：同一任务可能有多条资产（不同后端、
    同一后端的不同模型），不带它们会互相覆盖（台账主键不冲突，磁盘会冲突）。
    """
    name = f"{key}__{model_id}.png" if model_id else f"{key}.png"
    return Path(out_dir) / "swapped" / (provider or "unknown") / name


class FaceSwapper:
    """把参考图里的人脸贴到生成图上。"""

    def __init__(self, model_path: Path | None = None) -> None:
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model

        root = os.environ.get("AP_INSIGHTFACE_ROOT") or str(Path.home() / ".insightface")
        self.app = FaceAnalysis(name="buffalo_l", root=root)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        path = Path(model_path) if model_path else Path(root) / "models" / MODEL_NAME
        if not path.exists():
            # insightface 的 get_model 会自动从官方 release 下载
            self.swapper = get_model(MODEL_NAME, root=root)
        else:
            self.swapper = get_model(str(path), providers=["CPUExecutionProvider"])

    def swap(self, target: str | Path, reference: str | Path, out_path: str | Path) -> dict[str, Any]:
        import cv2
        import numpy as np
        from PIL import Image

        target_img = cv2.imread(str(target))
        ref_img = cv2.imread(str(reference))
        if target_img is None or ref_img is None:
            return {"ok": False, "reason": "图片读取失败"}

        target_faces = self.app.get(target_img)
        ref_faces = self.app.get(ref_img)
        if not target_faces:
            return {"ok": False, "reason": "生成图里检不到人脸"}
        if not ref_faces:
            return {"ok": False, "reason": "定妆照里检不到人脸"}

        # 定妆照的主体：画面里**面积最大**的那张脸（定妆照是单人肖像）。
        # 不能用"检测分最高"——那是检测器置信度，不是"哪张脸是主角"。
        def _area(face) -> float:
            x1, y1, x2, y2 = face.bbox
            return float(x2 - x1) * float(y2 - y1)

        source_face = max(ref_faces, key=_area)

        # 生成图里**与目标人物最像**的那张脸（不是检测分最高的那张）。
        # 实测画面里可能有医生/路人：按检测分选会**把医生的脸换掉**，
        # 换出来的图看着"脸很像"，但被换的根本不是主角。
        def _cos_to_source(face) -> float:
            return float(np.dot(face.normed_embedding, source_face.normed_embedding))

        target_face = max(target_faces, key=_cos_to_source)
        target_sim = _cos_to_source(target_face)

        result = self.swapper.get(target_img, target_face, source_face, paste_back=True)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).save(out_path)
        return {
            "ok": True,
            "path": str(out_path),
            "reason": "",
            "target_faces": len(target_faces),
            "reference_faces": len(ref_faces),
            # 被换的那张脸与定妆照的相似度（换之前）：用于证据链
            "target_face_similarity": round(target_sim, 4),
        }


def swap_available() -> tuple[bool, str]:
    """检查能不能做替换（缺依赖/缺模型时给出可执行的提示，而不是崩在流程中间）。"""
    try:
        import insightface  # noqa: F401
    except Exception as exc:
        return False, f"insightface 未安装：{exc}"
    root = Path(os.environ.get("AP_INSIGHTFACE_ROOT") or str(Path.home() / ".insightface"))
    model = root / "models" / MODEL_NAME
    if model.exists():
        return True, "模型已就绪"
    return True, "首次运行会自动下载 inswapper_128.onnx（约 530MB，需要网络）"
