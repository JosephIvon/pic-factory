# 部署 · 排错 · 维护（目标是能独立交付，不只是本地能跑）

> 一句话：`docker compose up` 起面板，`docker compose run --rm batch` 跑一批。
> 配置全走 `AP_*`，密钥只在 `.env`（已 gitignore）。没 key 也能跑——自动降级到离线 replay。

## 1. 三种运行形态（按"手上有什么"选，系统自己降级）

| 形态 | 前置 | 命令 | 用途 |
|---|---|---|---|
| **离线全链路** | 无网络、无 key | `aesthetics generate --provider replay` | 演示兜底：断网照常出图/质检/报表（读 fixtures） |
| **标准 API（交付）** | `AP_ARK_KEY` 或 `AP_DASHSCOPE_KEY` | `aesthetics generate --provider auto` | 真实交付：客户账号能跑的路径，路由自动选后端+模型档 |
| **连接器参照** | 本机有 mcode-tools | `AP_PREFER_MATRIX=1 … --provider auto` | 与 MiniMax 上游对齐的对比基线，**不可交付**，仅参照 |

路由是"云端层内部解析"（`routing._cloud_target`）：**配了标准 key 就默认走 `http_api`，matrix 自动降级为参照**；没 key 才回退 matrix。每次决策的 `rule_id + reason` 落台账，能逐条追"这张为什么走这条线"。

## 2. 起服务

```bash
cp .env.example .env          # 填 AP_ARK_KEY / AP_DASHSCOPE_KEY / AP_BUDGET_CNY / AP_PANEL_TOKEN
docker compose up --build     # 面板：http://127.0.0.1:8000/?token=<AP_PANEL_TOKEN>
```

**面板鉴权（P0-3）**：容器内面板绑 `0.0.0.0`，所以 `docker compose up` 现在**必须**在 `.env` 里设 `AP_PANEL_TOKEN`（随机串），否则 fail-closed 拒绝启动——这是预期行为，不是坏了。设了 token 后所有 `/api/*`、`/files/*` 都要凭据：浏览器用 `/?token=…` 打开一次（写入 cookie，页面 fetch/图片免逐请求带参），脚本用 `Authorization: Bearer <token>`。本机直接跑 `aesthetics panel`（默认 127.0.0.1）不设 token 仍可零门槛。

镜像装了完整闸门依赖（`.[qc,panel]`，含 insightface + rapidocr）。**首次跑人脸闸会联网下载 buffalo_l 权重到 `AP_INSIGHTFACE_ROOT`（默认 `/app/data/insightface`，compose 已挂出来）**，之后容器重建不重下；离线部署把权重预置进该目录即可。

跑一批（一次性容器，跑完即退）：

```bash
docker compose run --rm batch                      # 默认 auto 路由全量
docker compose run --rm batch aesthetics generate --provider auto --limit 6
```

## 3. 排错（对应故障库，现场能定位）

| 症状 | 大概率原因 | 定位/处置 |
|---|---|---|
| `RuntimeError: 两个后端都不可用` | 既没配标准 key，又没 matrix 连接器 | 用 `--provider replay` 走离线；或配 `AP_ARK_KEY` |
| 面板 `/api/meta` 显示 api 后端"不可用" | key 没进容器 | 确认 `.env` 有值且 compose 读到了（`docker compose config` 查注入） |
| 批量跑到一半 `预算熔断` | 累计现金达 `AP_BUDGET_CNY` 的 80% | 这是**设计行为不是 bug**：加预算或改走 replay；台账有每笔 est_cost |
| 阿里出图张数是请求的 4 倍 | 适配层 n 未强制（老坑） | `tests/test_http_api.py` 已钉 `n==1`；复跑测试确认没被改坏 |
| 下载 403 | 对 OSS 签名 URL 带了 Authorization 头 | 见 MODEL-CHOICE 方法学：下载签名 URL 不能加 auth 头 |
| 同一条任务重跑不扣费 | 正常——幂等键命中 `successful_keys` | 要强制重出用 `--no-skip-existing` |

## 4. 维护

- **换 provider 改哪三处**：`provider/models_spec.py` 加一行 ModelSpec（同族）；若新族，`provider/http_api.py` 加一个适配器函数；`config.py` 读对应 key。业务层、闸门、台账、面板零改动。
- **模型单价/一致性漂移**：只改 `models_spec.py` 的 `unit_cost_cny` 与 `model_choice._MEAN_SIM`，重跑 `aesthetics report` 即刷新成本账，**不必重新生成图片**。
- **回滚**：台账是 append+局部 UPDATE（`ON CONFLICT DO UPDATE`），批次可重跑；产物按 `幂等键+provider` 命名不互相覆盖；出问题 `git revert` 到上一 commit 或直接 `--provider replay` 出旧结果。
- **数据卫生**：`data/*.db` 与 `out/` 不入库；`.env` 不入库。50 张选型原图（85MB）留档在外部路径，仓库只带聚合证据 `docs/ab-evidence/`。

## 5. 边界（诚实声明，别当生产级结论）

- 交付镜像不含本地 GPU 栈（CUDA/SDXL 在宿主独立 venv，是"对比实验线"，非交付依赖）。
- 人脸阈值标定需 insightface + 标注集；离线演示走 replay 预置质检结果，不代表真机标定完成。
- 单人项目、样本量小；模型版本会漂移，跨版本不保证像素级复现（承诺的是"可追溯"：参数快照+模型版本+请求 ID）。
