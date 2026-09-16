"""evaluation 包：数据集模拟与统计评估模块。

架构定位：
- data_generator.py     多轮对话评估场景生成（偏好改变 / 长程任务 / 工具失败，
                        确定性骨架 + LLM 扩写 + 模板兜底，ground truth 代码推导）
- statistical_eval.py   检索质量与整体性能的统计评估（NDCG / MRR / 消融）
"""
