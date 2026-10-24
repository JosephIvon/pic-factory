# 域包（Domain Pack）：换行业不改代码

> 状态：已实现并通过全量回归（含域包、合规两组专项测试）。默认行为不变，`AP_PACK` 显式启用。

## 1. 为什么要有它

核心流水线（路由 → 生成 → 质检 → 换脸 · 采纳 · 成本 → 台账）是**通用**的；
"医美"只是第一个行业。没有域包时，换行业要改三处代码：

| 耦合点 | 原来在哪 | 现在在哪 |
|---|---|---|
| 任务矩阵（城市×项目×主题×渠道×角色） | `data/tasks.yaml` | `packs/<name>/matrix.yaml`（或清单引用旧路径） |
| 画面提示词（摄影语汇 + negative） | `prompts/scene.yaml` | `packs/<name>/scene.yaml` |
| 文案模板 + 禁用词 | `copy.py` 里的常量 | `packs/<name>/copy.yaml` |
| 合规要求（免责声明 / AI 标识 / 禁止构图） | 散在文档 | `packs/<name>/compliance.yaml` |

改数据的成本是"改一份 YAML + 跑测试"，改代码的成本是"改代码 + 全量回归 + 发布"。

## 2. 目录结构

```
packs/
├── medical-aesthetics/        # 医美：矩阵/画面复用仓库既有文件（不做第二份真相）
│   ├── pack.yaml
│   ├── copy.yaml
│   ├── compliance.yaml
│   └── assets/README.md
└── example-retail/            # 零售示例：完全自包含，证明核心与行业无关
    ├── pack.yaml
    ├── matrix.yaml
    ├── scene.yaml
    ├── copy.yaml
    └── compliance.yaml
```

`pack.yaml` 字段：

```yaml
name: medical-aesthetics
display: 医美内容生产        # CLI 列表里显示
version: "2026-10-24"
description: ...
matrix: ../../data/tasks.yaml   # 相对包目录；可以指向包外（医美复用旧文件）
scene: ../../prompts/scene.yaml
copy: copy.yaml                 # 可省略 → 用代码内置默认模板
compliance: compliance.yaml     # 可省略
assets: assets                  # 可省略
```

## 3. 兼容策略（重要）

**未设置 `AP_PACK` 时，行为与引入域包之前完全一致**：仍读 `data/tasks.yaml` +
`prompts/scene.yaml`，测试与部署零改动。

```
未设 AP_PACK          → 旧路径（默认，安全）
AP_PACK=medical-aesthetics → 医美包（矩阵/画面与旧路径同一份文件，输出等价）
AP_PACK=example-retail     → 零售包（完全不同的行业数据）
AP_PACKS_DIR=/path/to/packs → 客户自带行业包（不必放进本仓库）
```

测试 `test_expand_with_medical_pack_equals_legacy` 会断言"走包"与"走旧路径"的
幂等键**逐条相等**——防止两条路径悄悄漂移。

## 4. 新增一个行业包（5 步）

1. `mkdir packs/<industry>`；
2. 写 `matrix.yaml`（必须有 `cities / projects / themes / channels`，`characters` 可空）；
3. 写 `scene.yaml`（摄影语汇 + negative，**画面层只放英文**）；
4. 写 `copy.yaml`（模板占位符只允许 `{city} {project} {theme} {copy_angle} {channel}`）；
5. 写 `pack.yaml` 指过去，然后：

```bash
AP_PACK=<industry> aesthetics packs          # 检查清单是否加载成功
AP_PACK=<industry> aesthetics plan --limit 5 # 看路由决策（不花钱）
AP_PACK=<industry> pytest -q                 # 跑全量回归
```

## 5. 校验（早失败，不留到出图时）

| 情况 | 行为 |
|---|---|
| matrix 缺维度 | 加载时报 `ValueError: 域包 X 的 matrix 缺少字段: ...` |
| 模板用了未知占位符 | 加载时报错并列出允许的占位符 |
| `pack.yaml` 缺 matrix/scene | 加载时报错 |
| 包不存在 | `FileNotFoundError` + 列出可用包 |
| 单个包坏了 | `aesthetics packs` 标红该行，不影响其它包 |

## 6. 合规校验与发布包（已实现）

`compliance.yaml` 不再只是"文档里的字段"，已有消费方：

```bash
aesthetics compliance            # 审计：免责声明 / 图文成对 / 画面文字 / 禁用词，不过则退出码 1
aesthetics compliance --adopted-only
aesthetics publish --out out/publish/demo.zip   # 打发布包（默认 fail-closed）
aesthetics report                # 报表里同时写 report/compliance.json
```

发布包内容：

```
manifest.json     # 每张图一条：模型/prompt/质检分/采纳/AI 标识（显式+隐式）
compliance.json   # 本次审计结果（含规则快照）
README.txt        # 免责声明 + AI 标识声明 + 人工复核清单
images/…          # 最终产物（换过脸的用 swapped_path）
copy/…            # 标题/正文/标签
```

三条设计取舍：

1. **fail-closed**：审计不通过默认拒绝打包（退出码 2），必须显式 `--allow-violations`；
   被放行的例外会写进 `compliance.json`，不能"悄悄发出去"。
2. **隐式标识落在 manifest**：平台水印是"显式标识"，manifest.json 是"隐式元数据"，
   两者都在包里；不逐图像素级验证水印（做不到，也不该假装做到）。
3. **禁用词闸会拦自己**：合规声明文本不能包含词表里的词（实测："疗效承诺"里的
   "承诺"会命中）——`test_medical_pack_copy_passes_its_own_banned_word_gate` 钉住这条。

## 7. 现存缺口（诚实清单）

- **`panel.py` 的 `_router()` 仍直接读 `data/tasks.yaml`**：面板在非医美域包下会显示旧矩阵。
  改法是两行（`pack_matrix()/pack_scene()`），因该文件正被并行修改，未在本分支动手。
- **图像侧规则未接**：`compliance.yaml` 的 `forbidden_motifs`（术前术后对比等）目前是
  **人工复核清单**，会打印进发布包 README，但不自动判定——误判比漏判更贵。
- **`assets/` 的授权字段尚未接入流程**：角色卡 `license.yaml` 只有约定，交付前检查脚本未写。

## 8. 验收命令（本次交付）

```bash
pytest -q                                   # 全量通过（含域包 + 合规专项测试）
aesthetics packs                            # 列两个包
AP_PACK=example-retail aesthetics plan -n 3 # 零售矩阵走同一套路由
aesthetics compliance                       # 审计（退出码可进 CI）
aesthetics publish --out out/publish/demo.zip
```
