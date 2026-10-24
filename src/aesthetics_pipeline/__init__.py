"""医美 AI 图文内容生产闭环 —— MVP 演示项目。

模块地图：
  taskspec  任务矩阵展开 + 幂等键
  router    路径决策(T2I/I2I) + 提示词编译(画面层/文案层分离)
  provider  生图后端适配(matrix 真实 / replay 离线 / local 本地栈)
  qc        三路质检(硬规则 + 人脸数值闸 + VLM 软信号)
  copy      规则模板文案 + 禁用词表扫描
  ledger    SQLite 台账(成本 / 耗时 / 通过率)
  report    拼图 + CSV + JSON 报表
"""

__version__ = "0.1.0"
