# 交付用镜像：只装流水线核心 + 标准 API 后端 + Web 面板。
# 本地栈对比线（CUDA torch + SDXL）刻意不进这个镜像——它吃 GPU、属于实验线，
# 在宿主机独立 venv 跑（AP_LOCAL_PYTHON）。把重依赖塞进交付镜像是常见的"演示能跑、
# 交付跑不动"陷阱，这里用边界划清代替大包大揽。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# insightface→opencv 需要 GL 运行库（slim 镜像不带）。**闸门必须在交付镜像里
# 真能跑**：OCR 拦画面文字、人脸闸拦"画成别人"，缺一个四闸就是摆设。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖层（缓存友好）：只拷元数据相关的文件
COPY pyproject.toml README.md ./
COPY src ./src
# 逐目录显式映射：多源 COPY ./ 在本机构建上下文里会把各目录**内容**摊平进 /app
#（实测：tasks.yaml 落到 /app/tasks.yaml，/app/data 变空目录）。
COPY prompts /app/prompts
COPY packs /app/packs
COPY data /app/data
COPY fixtures /app/fixtures
COPY evidence /app/evidence
RUN pip install --no-cache-dir ".[qc,panel]"

# 台账/产物写目录（compose 里挂出来持久化）
# AP_REPO_ROOT：pip 安装后 __file__ 反推的仓库根会落到 site-packages，必须显式指回 /app
# AP_INSIGHTFACE_ROOT：buffalo_l 权重首次使用联网下载，落到挂载的 data 卷里，
# 容器重建不重下；离线部署把权重预置进该目录即可。
ENV AP_REPO_ROOT=/app \
    AP_PROMPT_DIR=/app/prompts \
    AP_EVIDENCE_DIR=/app/evidence \
    AP_DATA_DIR=/app/data \
    AP_OUT_DIR=/app/out \
    AP_DB=/app/data/pipeline.db \
    AP_INSIGHTFACE_ROOT=/app/data/insightface
RUN mkdir -p /app/data /app/out

EXPOSE 8000

# 默认起面板（只听容器内 0.0.0.0，由 compose 决定对外映射）。
# 一次性生成用 `docker compose run --rm app aesthetics generate --provider auto`。
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import os,urllib.request,sys; url='http://127.0.0.1:8000/api/meta?token='+os.environ.get('AP_PANEL_TOKEN',''); sys.exit(0 if urllib.request.urlopen(url,timeout=4).status==200 else 1)"

CMD ["aesthetics", "panel", "--host", "0.0.0.0", "--port", "8000"]
