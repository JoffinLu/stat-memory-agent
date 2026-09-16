"""agent 包：任务规划、状态机与自纠错模块。

架构定位（对应系统架构图第一、二层）：
- planner.py          任务规划（意图 → 子任务 DAG）
- state_machine.py    长程任务状态机与轨迹管理
- self_correction.py  自纠错引擎（SPC 监控 + SPRT 决策）
"""
