# 角色资产（医美域包）

当前角色卡的引用仍指向仓库既有文件 `evidence/01_t2i_baseline.png`（见
`data/tasks.yaml` 的 `characters.lin.ref`），**没有**复制进本目录——避免两份参考图漂移。

新增真实项目时，建议把角色卡放这里：

```
assets/
└── characters/
    └── <character_id>/
        ├── card.yaml     # 描述锁 / 禁止变化项
        ├── ref_*.png     # 6–8 张多角度定妆照
        └── license.yaml  # 来源、授权范围、可商用、有效期
```

`license.yaml` 是**交付红线**：没有授权信息的真人肖像不得进入生产包；
虚拟人资产要写明"AI 生成、可商用"。当前 `inswapper_128.onnx` 仅限非商业研究，
商用前必须替换（见 README §8）。
