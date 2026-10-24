"""标准 HTTP 生图后端：一个执行器 + 每族一个适配器。

批判性重构的落点（对齐 models_spec.py 的两层设计）：

  tools/ab 里 ArkProvider / WanProvider / QwenProvider 是**三份会各自变异的拷贝**
  ——各自写 http 客户端、各自轮询、各自落盘、各自计费。实测要同时用 5 个模型
  （MODEL-CHOICE §9），照那个写法就是 5 份拷贝，改一处坑要记得改五处。

  这里收敛成：
    * StandardApiProvider.generate()  —— 唯一执行器：解析 model_id → 取 ModelSpec
      → 预算/限速闸门 → 分派到族适配器 → 落盘/计费断言。同批可混合不同 model_id。
    * _ark_call / _wan_call            —— 每族一个函数（不是类）。换族才加代码，
      同族加模型只在 models_spec.REGISTRY 加一行。

实测踩过的坑（MODEL-CHOICE §2 / §9）都集中在这里处理，且**只处理一次**：
  - 阿里返回的是 OSS 签名 URL：下载**不能带 Authorization 头**，否则签名不匹配 403。
    → 所有下载走不带 auth 的 `download_client`。
  - 万相 `n` 默认 4 → 不显式设 n=1 就是付 4 倍钱。→ 强制 `parameters.n=1`，
    并做"计费张数==请求张数"断言。
  - 万相 `watermark` 默认 false（裸图）。人物交付红线要带 AI 生成水印 → 显式置 true。
  - Seedream 参数表没有独立 negative → 折进正向 prompt 尾部。
  - 参考图 base64 >~1MB 会被重置连接 → 本地路径先压到长边 1024 / JPEG q92。
  - 服务端返回的多是 JPEG，落盘扩展名按魔数定，不能信请求里的 .png。
  - 文件名带 provider（幂等键+provider），避免上游 F1 那种同键磁盘互相覆盖。

离线可测：http 客户端与预算/限速全部可注入，测试不打真实 API、不需要 key。
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import httpx

from ..config import OUT_DIR
from ..cost_guard import BudgetGuard, RateLimiter
from ..model_choice import choose_model
from ..models import GenItem, GenOutcome
from .cache import write_cache
from .models_spec import ModelSpec, get as get_spec

# 族适配器端点（与 tools/ab 一致的上游地址，但只在此处出现一次）。
ARK_EP = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
WAN_T2I_EP = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
WAN_I2I_EP = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis"
WAN_TASK_EP = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"

_REF_MAX_EDGE = 1024
_REF_JPEG_QUALITY = 92

# 并发跑批默认档位（P0-1）：RPM 300 的 Seedream 不该再被当 RPM 3 用——串行 50 张
# 要 10–25 分钟，4 路并发直接一个数量级。上限 16 是防"并发越大越好"的想当然：
# 太大并发会撞上上游连接池/网关限流，失败率反升，重试又把成本打回去。
DEFAULT_CONCURRENCY = 4
MIN_CONCURRENCY = 1
MAX_CONCURRENCY = 16


# ---------------------------------------------------------------------------
# 纯函数工具（供适配器与测试复用）
# ---------------------------------------------------------------------------

def sniff_ext(data: bytes) -> str:
    """按魔数定真实扩展名——服务端/请求里的 .png 都不可信（MODEL-CHOICE §9）。

    手工判魔数而非 imghdr：imghdr 在 Python 3.13 被移除，不赌运行环境版本。
    """
    if data[:3] == b"\xff\xd8\xff":           # JPEG FFD8
        return ".jpg"
    if data[:4] == b"\x89PNG" or data[:4] == b"\x89\x50\x4e\x47":  # PNG 89504E47
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    return ".bin"


def fold_negative(prompt: str, negative: str) -> str:
    """族不支持独立 negative 字段时（Seedream），把负向词折进正向 prompt 尾部。"""
    negative = (negative or "").strip()
    if not negative:
        return prompt
    return f"{prompt.rstrip('.')}. Avoid in image: {negative}."


def ref_data_uri(raw: str, *, max_edge: int = _REF_MAX_EDGE, quality: int = _REF_JPEG_QUALITY) -> str:
    """参考图转 data-uri。

    已是 URL（http/https/data:）原样透传；本地路径读图压到长边 max_edge、JPEG q=quality
    再 base64——>1MB 会被上游重置连接（MODEL-CHOICE §9 方法学修正）。
    """
    if raw.startswith(("http://", "https://", "data:")):
        return raw
    from PIL import Image  # 延迟导入：纯 t2i / 离线测试不必装 PIL

    img = Image.open(raw).convert("RGB")
    img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _size_for(spec: ModelSpec, item: GenItem) -> str:
    """WxH 画幅。ark 用 'x'，wan 用 '*'（两家 size 分隔符不同，实测踩过）。"""
    raw = item.resolution if "x" in (item.resolution or "") else spec.default_size
    return raw if spec.family == "ark" else raw.replace("x", "*")


# ---------------------------------------------------------------------------
# 执行器
# ---------------------------------------------------------------------------

class StandardApiProvider:
    """满足 Provider 协议：generate(items) -> 与入参等长、顺序一致的 outcomes。"""

    name = "http_api"

    def __init__(
        self,
        *,
        name: str | None = None,
        default_model_id: str | None = None,
        out_dir: Path | None = None,
        ark_client: Any | None = None,
        wan_client: Any | None = None,
        download_client: Any | None = None,
        budget_guard: BudgetGuard | None = None,
        poll_interval: float = 3.0,
        poll_timeout: float = 480.0,
        time_func: Callable[[], float] = time.monotonic,
        sleep_func: Callable[[float], None] = time.sleep,
        concurrency: int | None = None,
    ) -> None:
        if name:
            self.name = name
        self.default_model_id = default_model_id
        self.out_dir = Path(out_dir or OUT_DIR)
        self.images_dir = self.out_dir / "images"
        self.cache_dir = self.out_dir / "cache"

        # 三个客户端可注入（离线测试用假客户端）；真实运行按 env key 懒建。
        self._ark_client = ark_client
        self._wan_client = wan_client
        # 下载必须用**不带 Authorization** 的干净客户端——OSS 签名 URL 带 auth 会 403。
        self._download_client = download_client

        self.budget = budget_guard if budget_guard is not None else BudgetGuard.from_env()
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self._time = time_func
        self._sleep = sleep_func
        self._limiters: dict[int, RateLimiter] = {}
        # _limiters 是"按 rpm 分组懒建"的共享表：并发下建桶要串行化，
        # 否则两个线程各建一个同 rpm 的桶，令牌被对半劈开、限速名存实亡。
        self._limiters_lock = threading.Lock()
        self.concurrency = self._resolve_concurrency(concurrency)

    # ---------- 并发度解析 ----------

    @staticmethod
    def _resolve_concurrency(concurrency: int | None) -> int:
        """显式参数优先；缺省读 env `AP_CONCURRENCY`（默认 4）；一律钳到 1–16。"""
        if concurrency is None:
            raw = (os.environ.get("AP_CONCURRENCY") or "").strip()
            try:
                concurrency = int(raw) if raw else DEFAULT_CONCURRENCY
            except ValueError:
                concurrency = DEFAULT_CONCURRENCY
        return max(MIN_CONCURRENCY, min(int(concurrency), MAX_CONCURRENCY))

    # ---------- 客户端懒建（真实运行才碰 env key） ----------

    def _auth_header(self, key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _ark(self) -> Any:
        if self._ark_client is None:
            key = self._require_key("ARK_API_KEY")
            self._ark_client = httpx.Client(
                headers=self._auth_header(key), timeout=httpx.Timeout(300, connect=20)
            )
        return self._ark_client

    def _wan(self) -> Any:
        if self._wan_client is None:
            key = self._require_key("DASHSCOPE_API_KEY")
            headers = self._auth_header(key)
            headers["X-DashScope-Async"] = "enable"   # 万相异步任务必需头
            self._wan_client = httpx.Client(
                headers=headers, timeout=httpx.Timeout(120, connect=20)
            )
        return self._wan_client

    def _downloader(self) -> Any:
        if self._download_client is None:
            # 干净客户端：没有任何 Authorization。
            self._download_client = httpx.Client(
                timeout=httpx.Timeout(180, connect=20), follow_redirects=True
            )
        return self._download_client

    @staticmethod
    def _require_key(env_name: str) -> str:
        import os

        key = os.environ.get(env_name)
        if not key:
            raise RuntimeError(f"缺少 {env_name}（真实调用才需要；测试请注入客户端）")
        return key

    # ---------- 限速闸门 ----------

    def _limiter(self, rpm: int) -> RateLimiter:
        with self._limiters_lock:
            if rpm not in self._limiters:
                self._limiters[rpm] = RateLimiter(
                    rpm, time_func=self._time, sleep_func=self._sleep
                )
            return self._limiters[rpm]

    # ---------- 模型解析 ----------

    def _resolve_spec(self, item: GenItem, model_id: str | None) -> ModelSpec:
        mid = model_id or self.default_model_id or choose_model(item.task).model_id
        return get_spec(mid)

    # ---------- 主执行器 ----------

    def generate(
        self, items: list[GenItem], model_ids: list[str | None] | None = None
    ) -> list[GenOutcome]:
        if model_ids is not None and len(model_ids) != len(items):
            raise ValueError("model_ids 长度必须与 items 一致（同批可混合不同模型）")
        # 解析档位在主线程做完：model_ids 与 items 的对应关系在这里定死，
        # 之后并发只改"何时跑"，不改"谁跑哪条"。
        jobs = [
            (item, self._resolve_spec(item, model_ids[i] if model_ids else None))
            for i, item in enumerate(items)
        ]
        workers = min(self.concurrency, len(jobs))
        if workers <= 1:
            # 并发=1（或单条 item）：直接走原串行路径，不建线程池——
            # 少一条会引入线程语义的代码路径，回退口径与旧版逐字节一致。
            return [self._run_one(item, spec) for item, spec in jobs]
        # pool.map 的返回顺序 == 输入顺序（逐 future 阻塞收结果），
        # 契约"结果与 items 顺序严格一致"由 executor 本身保证。
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda job: self._run_one(*job), jobs))

    def _run_one(self, item: GenItem, spec: ModelSpec) -> GenOutcome:
        # 1) 参考图上限：超限直接在该 item 报错，绝不静默截断（截断会悄悄改变一致性）。
        if len(item.input_urls) > spec.ref_max:
            return GenOutcome(
                task=item.task, ok=False, source_provider=self.name, model_id=spec.model_id,
                error=f"参考图 {len(item.input_urls)} 张超过 {spec.display} 上限 {spec.ref_max}",
            )

        # 2) 预算熔断（预扣制）：**在真正调用 API 之前**原子预扣单价；扣不动
        #    （会越过 80% 熔断线）就绝不发请求。并发下 try_charge 的锁保证
        #    同时在途的请求总额不会把预算烧穿（允许恰好扣到熔断线）。
        if not self.budget.try_charge(spec.unit_cost_cny):
            return GenOutcome(
                task=item.task, ok=False, source_provider=self.name, model_id=spec.model_id,
                error="预算熔断：已达 80% 上限",
            )

        # 3) RPM 令牌桶：按族限速，保护上游。
        self._limiter(spec.rpm).acquire()

        started = self._time()
        try:
            if spec.family == "ark":
                urls, billed, request_id = self._ark_call(item, spec)
            elif spec.family == "wan":
                urls, billed, request_id = self._wan_call(item, spec)
            else:
                raise ValueError(f"未知族 {spec.family}")
        except Exception as exc:  # 网络/解析异常都归失败：**退还预扣、不计费、不落账**
            self.budget.refund(spec.unit_cost_cny)
            return GenOutcome(
                task=item.task, ok=False, source_provider=self.name, model_id=spec.model_id,
                error=f"{type(exc).__name__}: {exc}", seconds=self._time() - started,
            )

        elapsed = self._time() - started

        # 4) 计费断言：计费张数 == 请求张数（wan 的 n=1 陷阱就防这个）。不符拒绝落账。
        if billed != 1 or len(urls) != 1:
            self.budget.refund(spec.unit_cost_cny)
            return GenOutcome(
                task=item.task, ok=False, source_provider=self.name, model_id=spec.model_id,
                error=f"计费断言失败：请求 1 张却计费 {billed} 张/返回 {len(urls)} 张（拒绝落账）",
                seconds=elapsed,
            )

        # 5) 下载（不带 auth）+ 按魔数定扩展名 + 幂等键+provider 命名落盘。
        try:
            local_path = self._download(urls[0], item.task.idempotency_key)
        except Exception as exc:
            self.budget.refund(spec.unit_cost_cny)
            return GenOutcome(
                task=item.task, ok=False, source_provider=self.name, model_id=spec.model_id,
                error=f"下载失败: {exc}", seconds=elapsed,
            )

        file_name = local_path.name
        # 成功且计费相符：预扣即最终扣费（不再二次 charge），缓存落账。
        write_cache(
            self.cache_dir,
            key=item.task.idempotency_key,
            provider=self.name,
            node_id=request_id,
            file_name=file_name,
            local_path=str(local_path),
            prompt=item.prompt,
            path=item.task.path,
            seconds=elapsed,
            model_id=spec.model_id,
        )
        return GenOutcome(
            task=item.task, ok=True, node_id=request_id, file_name=file_name,
            local_path=str(local_path), seconds=elapsed, source_provider=self.name,
            model_id=spec.model_id,
        )

    # ---------- 族适配器：ark（同步） ----------

    def _ark_call(self, item: GenItem, spec: ModelSpec) -> tuple[list[str], int, str | None]:
        prompt = item.prompt
        # Seedream 无独立 negative 字段 → 折进正向 prompt。
        if not spec.supports_negative:
            prompt = fold_negative(prompt, item.negative_prompt)
        body: dict[str, Any] = {
            "model": spec.model_id,
            "prompt": prompt,
            "size": _size_for(spec, item),
            "response_format": "url",
        }
        if spec.supports_negative and item.negative_prompt:
            body["negative_prompt"] = item.negative_prompt
        if item.input_urls:
            body["image"] = [ref_data_uri(u) for u in item.input_urls]
        if spec.watermark_default_on:
            body["watermark"] = True   # Seedream 默认即 true，显式写出来更省心

        r = self._ark().post(ARK_EP, json=body)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {str(getattr(r, 'text', ''))[:300]}")
        data = r.json()
        items = data.get("data") or []
        urls = [it.get("url") for it in items if it.get("url")]
        # billed：接口 usage 若给了生成张数就用它，否则以返回列表长度为准。
        usage = data.get("usage") or {}
        billed = int(usage.get("generated_images", len(urls)))
        request_id = dict(getattr(r, "headers", {}) or {}).get("x-request-id")
        return urls, billed, request_id

    # ---------- 族适配器：wan（异步创建任务→轮询） ----------

    def _wan_call(self, item: GenItem, spec: ModelSpec) -> tuple[list[str], int, str | None]:
        input_block: dict[str, Any] = {"prompt": item.prompt}
        # 万相支持独立 negative_prompt（Qwen family=wan 也走这里）。
        if spec.supports_negative and item.negative_prompt:
            input_block["negative_prompt"] = item.negative_prompt
        if item.input_urls:
            input_block["images"] = [ref_data_uri(u) for u in item.input_urls]
            ep = WAN_I2I_EP
        else:
            ep = WAN_T2I_EP

        parameters: dict[str, Any] = {
            "n": 1,                              # 关键：默认 n=4 会付 4 倍钱，强制显式 1
            "size": _size_for(spec, item),
            "watermark": True,                   # 交付红线：阿里默认 false 裸图 → 显式开
        }
        body = {"model": spec.model_id, "input": input_block, "parameters": parameters}

        r = self._wan().post(ep, json=body)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {str(getattr(r, 'text', ''))[:300]}")
        task_id = (r.json().get("output") or {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"no task_id: {str(getattr(r, 'text', ''))[:300]}")

        deadline = self._time() + self.poll_timeout
        while self._time() < deadline:
            self._sleep(self.poll_interval)
            q = self._wan().get(WAN_TASK_EP.format(task_id=task_id))
            if q.status_code != 200:
                continue
            out = (q.json().get("output") or {})
            status = out.get("task_status")
            if status == "SUCCEEDED":
                results = out.get("results") or []
                urls = [x.get("url") for x in results if x.get("url")]
                if not urls:
                    raise RuntimeError("succeeded but no url")
                # billed：万相按 results 长度计（n=1 时应为 1）。
                return urls, len(results), task_id
            if status in ("FAILED", "CANCELED", "UNKNOWN"):
                raise RuntimeError(f"task {status}: {str(out)[:300]}")
        raise RuntimeError("poll timeout")

    # ---------- 下载：不带 auth，按魔数定扩展名 ----------

    def _download(self, url: str, key: str) -> Path:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        resp = self._downloader().get(url)
        if hasattr(resp, "raise_for_status"):
            resp.raise_for_status()
        data = resp.content
        ext = sniff_ext(data)
        # 文件名 = 幂等键 + provider（避免上游 F1 那种同键磁盘覆盖）。
        target = self.images_dir / f"{key}__{self.name}{ext}"
        target.write_bytes(data)
        for stale in self.images_dir.glob(f"{key}__{self.name}.*"):
            if stale != target and stale.is_file():
                try:
                    stale.unlink()
                except OSError:
                    pass
        return target
