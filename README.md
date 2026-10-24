# aesthetics-pipeline · AI 图文批量生产流水线

按任务矩阵（城市 × 项目 × 主题 × 渠道）自动路由模型、批量生图、闸门式质检、成本熔断与台账、合规审计与发布包的开源流水线。换行业不改代码——改一份 YAML。

[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> pytest 状态占位（不伪造徽章）：**174 项全过**——本地实跑口径（`pytest -q`，Python 3.11+）。
> CI 门禁定义已就绪（`.github/workflows/ci.yml`：pytest / docker build / 离线 replay 冒烟 / 密钥泄漏扫描），
> **推送到 GitHub 后徽章与自动化才生效**；在此之前请以 `tools/ci_local.sh` 的本地等价命令为准。

## 为什么做这个

行业内容（本地生活、零售、医美等）的批量图文生产卡在一个三角上：

- **真实感**：模型直出的图要么一眼 AI，要么在镜面、遮挡、无人空间这些结构性难点上露馅；真实感只能靠盲测和闸门逼近，不能靠提示词祈祷。
- **一致性**：同一个角色/模特跨城市、跨项目、跨渠道复用，"漂不漂"取决于选哪个模型档位——实测里最新档的一致性反而低于上一代（见 `docs/MODEL-CHOICE.md` §9）。
- **成本**：生图按张计费、没有包月经济（`docs/COST-MODEL.md`），而有的上游 API 默认一次请求出 4 张——不显式设 `n=1` 就是付 4 倍钱（`http_api` 强制显式并配计费断言）。

这个项目的回答是把三角拆成三个工程问题：选型做成**规则表 + 实测证据**，质量做成**出厂闸门**，钱做成**台账与熔断**。每一张图"为什么走这条线、花了几分钱、过没过闸"都可逐条回查。

## 核心思想

1. **模型是路由问题，不是提示词问题。** 选后端（云端 API / 本地栈 / 回放）由 `routing.py` 的有序规则表命中，选档位（哪个模型）由 `model_choice.py` 按任务类型 > 一致性下限 > 成本决定。每条决策带 `rule_id + reason` 落台账，可复算、可追责，不靠人肉敲参数。
2. **质量是闸门问题，不是模型问题。** 人脸一致性闸（生产尺阈值 T=0.57，同域标定）、OCR 文字闸（乱码文字出厂前拦截）、硬规则闸（尺寸/比例/dhash 重复图）、合规词闸（免责声明/禁用词）——VLM 语义分只作软信号不参与拦截，因为实测中 VLM 能读出画面乱码却判"无明显问题"。
3. **幂等台账让失败可回放。** 每条任务以幂等键落 SQLite 台账（append + `ON CONFLICT DO UPDATE`），同键重跑不重复扣费；失败批次可原样重放，产物按 `幂等键 + provider` 命名互不覆盖。"没跑成"和"跑成但没过闸"在账上是两回事。
4. **成本按档位实测价落账。** `provider/models_spec.py` 里每个档位单价来自本机实测与官方价目对照（见 `docs/ab-evidence/`）；单价未配置就报"未配置单价"，绝不编一个数进成本账（`test_cost_endpoint_does_not_invent_money` 钉住这条）。

## 架构

```
data/tasks.yaml / packs/<行业>/matrix.yaml
        │  任务矩阵展开（taskspec.py：城市×项目×主题×渠道 → TaskSpec + 幂等键）
        ▼
routing.py ──► model_choice.py
 后端维度规则表                模型档位规则表
 （云/本地/回放，优先级：      （Seedream/Qwen 按档，
   合规>排版>一致性>成本）      rule_id + reason 落台账）
        ▼
provider/  ── http_api（火山方舟/阿里百炼，交付路径）
             matrix（连接器参照，不可交付）
             local（SDXL 本地栈，边际成本≈电费）
             replay（离线回放真实缓存，无 key 兜底）
        ▼
质检闸门： 硬规则闸(尺寸/比例/dhash) │ OCR 文字闸 │ 人脸一致性闸(T=0.57) │ 合规词闸
           VLM 语义分 ── 只作软信号，不拦截
        ▼
ledger.py（SQLite：幂等键、route_reason、est_cost、qc_verdict、adopted）
        ▼
report.py（contact_sheet + CSV/JSON） ── panel（抽检台，人工采纳）
        ▼
compliance.py 审计 ──► publish 发布包（fail-closed：不过闸不打包）
```

## 快速开始

```bash
# 安装（Python 3.11+；panel 依赖可选，装不上不影响 CLI 主链路）
pip install -e ".[qc,panel]"

# 离线跑通全链路：不需要任何 API key，回放随仓库分发的真实生成缓存
aesthetics generate --provider replay
aesthetics qc
aesthetics report

# 本机面板（默认绑 127.0.0.1，零门槛）
aesthetics panel

# 真实交付路径（需 AP_ARK_KEY 或 AP_DASHSCOPE_KEY）
aesthetics plan            # 先看每条任务会路由到哪个后端/档位，不花钱
aesthetics generate --provider auto
```

Docker 交付形态：

```bash
cp .env.example .env   # 填 AP_ARK_KEY / AP_DASHSCOPE_KEY / AP_BUDGET_CNY / AP_PANEL_TOKEN
docker compose up --build
```

注意：容器内面板绑 `0.0.0.0`，**必须**设 `AP_PANEL_TOKEN`（随机串），否则 fail-closed 拒绝启动——这是预期行为。打开 `http://127.0.0.1:8000/?token=<AP_PANEL_TOKEN>`。首次跑人脸闸会联网下载 buffalo_l 权重（离线部署预置到 `AP_INSIGHTFACE_ROOT`）。详见 [DEPLOY.md](DEPLOY.md)。

## 关键机制

### 模型选型：证据在仓库里，不在口头承诺里

`docs/MODEL-CHOICE.md` 记录了本机实跑 50 张的 A/B：同角色卡、同 10 条 prompt、逐模型的一致性均值/最低/方差/P50/P95/有效单张成本。结论反直觉——一致性最好的不是最新档（Qwen-Image 3.0 Pro 0.888 > 3.0 0.844 > Seedream 4.0 0.757），"按版本号选新"在锁脸任务上不成立。原始证据（逐张分数、盲测标注、复算 JSON）在 `docs/ab-evidence/`。数字仍是单次观测口径，每模型 10 张×1 轮，用于"排除"而非"排名"。

### 人脸一致性闸：阈值 0.57 是同域标定出来的

`qc/face.py` 的出厂阈值 **T=0.57**，由 `tools/recalibrate_face_threshold.py` 在同域样本上取间隙中点标定：同域难负样本最高 0.4918，人物合格模型正样本最低（剔遮挡）0.6406（证据：`docs/ab-evidence/threshold-calibration-production.json`）。两点必须诚实：这把尺是生产口径（insightface buffalo_l + gallery 选脸），与实验台口径**不同尺、数字不可互相引用**；负样本当前 n=7，FAR 点估计无统计意义，扩到 150 张前阈值待复核（可用 `AP_FACE_THRESHOLD` 覆盖）。遮挡脸（如贴面膜）ArcFace 不可测，走人工复核而不是自动判死。

### 成本熔断 + 计费断言

`cost_guard.py` 两个独立闸：RPM 令牌桶（按模型族节流，Qwen-3.0-Pro 实测 RPM 仅 5，不打桶会被 429 打死）和预算熔断（累计现金达 `AP_BUDGET_CNY` 的 80% 即熔断新调用——这是设计行为不是 bug，台账里每笔有 `est_cost` 可对账）。`provider/http_api.py` 对每次调用断言"计费张数 == 请求张数"，专防上游 `n` 默认 4 的计费陷阱（`tests/test_http_api.py` 钉死）。

### 域包：换行业不改代码

`packs/` 把任务矩阵、画面提示词、文案模板、合规要求装进 YAML 包：`packs/medical-aesthetics/`（医美）与 `packs/example-retail/`（完全自包含的零售包，证明核心流水线与行业无关）。`AP_PACK=<名字> aesthetics plan` 即切换；客户行业包放仓库外用 `AP_PACKS_DIR` 指路径。未设 `AP_PACK` 时行为与引入域包前逐条一致（幂等键相等由测试断言防漂移）。新增行业 = 写 4 份 YAML + 跑回归，见 `docs/DOMAIN-PACKS.md`。

### 合规审计与发布包：fail-closed

`aesthetics compliance` 对台账逐条审计：免责声明是否在场、图文是否成对、画面内文字、文案禁用词，不过则退出码 1，可直接进 CI。`aesthetics publish` 打发布包（manifest + compliance.json + 人工复核 README + 图 + 文案），审计不过默认拒绝打包（退出码 2），显式 `--allow-violations` 放行的例外会写进 `compliance.json`，不存在"悄悄发出去"。禁用词闸会拦自己——合规声明文本也不能命中词表（有测试钉住）。

### replay：离线是第一条被验证的路线，不是装饰

`provider/replay.py` 只回放真实后端写下过的缓存（`fixtures/` 随仓库分发），不调任何外部服务、不伪造素材。断网、无 key、配额耗尽时，生成→质检→台账→报表全链路照常跑通，流程不打折。CI 的 `replay-smoke` job 就以此严格离线运行（故意不装 insightface，人脸闸走显式 skipped 路径——"跳过"和"通过"在台账上是两个值）。

## 测试

```bash
pip install -e ".[qc,panel,dev]"
pytest -q                    # 174 项
python tools/check_readme_test_count.py            # README 测试数防漂移门禁
python tools/check_readme_test_count.py --update   # 改完测试后同步本文件
```

- 数量以工具实测为准：本 README 写 174，是写作时 `python -m pytest --collect-only -q`（收集 174）与 `pytest -q`（174 passed）的实测值；CI 里 `check_readme_test_count.py` 会在数字漂移时报错。
- 全程离线可跑：人脸闸用例 monkeypatch 注入假检测器，不触发权重下载；`tests/test_adversarial.py` 是对抗演练（伪造计费张数、劫持幂等键等负向锚点）。
- 覆盖分布可看 `tests/`：路由/选型/台账/成本/HTTP 适配/四闸/域包/合规/面板/CLI 接线各有专档。

## 局限与诚实清单

- **换脸模型非商用许可**：兜底用的 `inswapper_128.onnx` 是明确非商业研究许可，商用必须替换为有授权的模型（见 `NOTICE`）。
- **buffalo_l 仅学术用途**：人脸一致性闸的权重许可为"仅学术研究"，商用需自行取得授权。
- **VLM 只作软信号**：实测同一张乱码招牌图，VLM 读得出乱码却判"无明显问题"，故其分数不参与出厂拦截。
- **合规词表未经法务**：免责声明是子串检查，只保证"该说的话没丢"，不保证措辞被法务认可；画面侧规则（forbidden_motifs）目前是人工复核清单，不做自动判定。
- **本地栈真实感有上限**：SDXL-Turbo 为 4 步蒸馏模型，偏"塑料感"；换写实 checkpoint 后更"像照片"但耗时 1.9 倍、图生图身份分反而更低（对比图见 `evidence/`，判定留给盲评）。
- **统计口径小**：A/B 每臂 10 张、盲测 1 人 20 张、人脸负样本 n=7，都是方向性结论；同配置跨次方差可达 0.2，"排名"需 3 次重复复核。
- **已知缺口**：域包 `assets/` 的授权字段未接入流程检查；模型版本会漂移，承诺的是可追溯（参数快照+模型版本+请求 ID）而非像素级复现。

## 许可

- 源代码：[MIT](LICENSE)。
- 第三方模型/权重/随库文件的许可边界：**逐条见 [NOTICE](NOTICE)**——代码是 MIT，不等于跑起来的东西都能商用；随库分发的演示图像 AI 生成，二次分发请保留"AI 生成"标注。
- 本项目不捆绑任何 API key 或服务额度，调用第三方生图 API 须以你自己的账号遵守各服务商条款。

## 目录索引

```
src/aesthetics_pipeline/   核心：taskspec/routing/model_choice/provider/qc/ledger/cost_guard/compliance/panel
packs/                     行业域包（医美 + 零售示例；AP_PACK 选用，未设走旧路径）
prompts/ data/             默认矩阵与画面提示词（域包引入前的旧路径）
tests/                     回归测试（见上节）
docs/                      MODEL-CHOICE / COST-MODEL / DOMAIN-PACKS / ab-evidence 实测证据
tools/                     A/B 实验台、阈值重标定、README 防漂移、本地 CI 等等价脚本
fixtures/ evidence/        离线回放缓存与实验证据图（随仓库分发）
.github/workflows/ci.yml   CI 门禁定义（推送 GitHub 后生效）
DEPLOY.md                  部署 · 排错 · 维护
```
