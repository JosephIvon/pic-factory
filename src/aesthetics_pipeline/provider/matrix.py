"""真实后端：通过 mcode-tools CLI 调用矩阵生图服务。

同一条接口同时支持文生图与图生图（传 input_urls），所以"选模型"在这层
表现为：要不要挂参考图，而不是换一个后端。

Windows 注意事项（实测踩坑）：
  - JSON 参数必须用 `--args-file` 传；`--args '<json>'` 内联会被引号解析破坏。
  - `<<<` 是 bash 语法，PowerShell 会直接 ParserError。
  - `output_file` 只认 basename，**扩展名由服务端决定**（传 .png 可能返回 .jpg），
    落盘要用返回的 file_name。
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import httpx

from ..cli_runner import extract_json, run_cli
from ..config import BATCH_SIZE, MATRIX_CLI, OUT_DIR
from ..models import GenItem, GenOutcome
from .cache import write_cache

TOOL = "connector__matrix__generate_image"
UPLOAD_TOOL = "connector__matrix__upload_temp_url"


class MatrixProvider:
    name = "matrix"

    def __init__(
        self,
        cli: str = MATRIX_CLI,
        out_dir: Path | None = None,
        batch_size: int = BATCH_SIZE,
        timeout: int = 900,
    ) -> None:
        self.cli = cli
        self.out_dir = Path(out_dir or OUT_DIR)
        self.images_dir = self.out_dir / "images"
        self.cache_dir = self.out_dir / "cache"
        self.batch_size = batch_size
        self.timeout = timeout

    # ---------- CLI 调用 ----------

    def _run(self, args: list[str]) -> str:
        proc = run_cli(self.cli, args, timeout=self.timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"{self.cli} 退出码 {proc.returncode}: {(proc.stderr or proc.stdout)[:500]}")
        return proc.stdout

    @staticmethod
    def _extract_json(text: str) -> dict:
        return extract_json(text)

    def upload(self, local_path: str | Path) -> str:
        """把本地参考图换成服务端可取到的临时 URL（图生图需要）。"""
        out = self._run(["upload_temp_url", str(local_path)])
        return str(self._extract_json(out)["temp_url"])

    # ---------- 生成 ----------

    def generate(self, items: list[GenItem]) -> list[GenOutcome]:
        results: list[GenOutcome] = []
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            results.extend(self._generate_chunk(chunk))
        return results

    def _generate_chunk(self, chunk: list[GenItem]) -> list[GenOutcome]:
        requests: list[dict] = []
        for item in chunk:
            req: dict = {
                "prompt": item.prompt,
                "aspect_ratio": item.aspect_ratio,
                "resolution": item.resolution,
                "output_file": f"{item.task.idempotency_key}.png",
            }
            if item.input_urls:
                req["input_urls"] = item.input_urls
            requests.append(req)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump({"requests": requests}, fh, ensure_ascii=False)
            args_file = fh.name

        started = time.perf_counter()
        try:
            out = self._run(["connector", "call", TOOL, "--args-file", args_file])
        except Exception as exc:  # 整批失败：全部标失败，交给上层精确重试
            elapsed = time.perf_counter() - started
            return [
                GenOutcome(task=i.task, ok=False, error=str(exc), seconds=elapsed) for i in chunk
            ]
        finally:
            Path(args_file).unlink(missing_ok=True)

        payload = self._extract_json(out)
        elapsed = time.perf_counter() - started
        per_item = elapsed / max(len(chunk), 1)

        ok_items = list(payload.get("success_items", []))
        ok_names = {s.get("file_name"): s.get("node_id") for s in ok_items}
        failed_names = {f.get("file_name") for f in payload.get("failed_items", [])}

        # 按文件名（stem）对回请求；服务端改了名字时用位置兜底，
        # 避免"其实成功了却被判失败"——那等于白花钱还误导排查。
        unclaimed = list(ok_items)
        outcomes: list[GenOutcome] = []
        for item in chunk:
            base = f"{item.task.idempotency_key}.png"
            file_name = next(
                (n for n in ok_names if n and Path(n).stem == Path(base).stem),
                None,
            )
            if file_name is None:
                positional = next(
                    (
                        s
                        for s in unclaimed
                        if s.get("file_name")
                        and not any(
                            (f.get("file_name") or "") == s.get("file_name")
                            for f in payload.get("failed_items", [])
                        )
                    ),
                    None,
                )
                if positional is not None:
                    unclaimed.remove(positional)
                    file_name = positional.get("file_name")
            if file_name is None:
                reason = "接口未返回该请求的成功结果"
                if base in failed_names or any(
                    (f.get("file_name") or "").startswith(item.task.idempotency_key)
                    for f in payload.get("failed_items", [])
                ):
                    reason = "接口返回失败"
                outcomes.append(
                    GenOutcome(task=item.task, ok=False, error=reason, seconds=per_item)
                )
                continue

            node_id = ok_names[file_name]
            try:
                local_path = self._download(node_id, file_name, item.task.idempotency_key)
            except Exception as exc:
                outcomes.append(
                    GenOutcome(task=item.task, ok=False, error=f"下载失败: {exc}", seconds=per_item)
                )
                continue

            self._write_cache(item, node_id, file_name, local_path, per_item)
            outcomes.append(
                GenOutcome(
                    task=item.task,
                    ok=True,
                    node_id=node_id,
                    file_name=file_name,
                    local_path=str(local_path),
                    seconds=per_item,
                )
            )
        return outcomes

    def _download(self, node_id: str, file_name: str, key: str) -> Path:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        out = self._run(["get_asset_url", node_id])
        url = self._extract_json(out)["download_url"]
        ext = Path(file_name).suffix or ".png"
        target = self.images_dir / f"{key}{ext}"
        with httpx.stream("GET", url, timeout=180, follow_redirects=True) as resp:
            resp.raise_for_status()
            target.write_bytes(resp.read())
        # 服务端每次返回的扩展名会变（实测 .png / .jpg 来回），同一任务的旧扩展名
        # 文件会留在目录里 —— 现场打开目录会拿到上一轮的那张。清掉；
        # 清不掉也不该影响出图，所以只吞异常。
        for stale in self.images_dir.glob(f"{key}.*"):
            if stale != target and stale.is_file():
                try:
                    stale.unlink()
                except OSError:
                    pass
        return target

    def _write_cache(
        self, item: GenItem, node_id: str, file_name: str, local_path: Path, seconds: float
    ) -> None:
        write_cache(
            self.cache_dir,
            key=item.task.idempotency_key,
            provider=self.name,
            node_id=node_id,
            file_name=file_name,
            local_path=str(local_path),
            prompt=item.prompt,
            path=item.task.path,
            seconds=seconds,
        )
