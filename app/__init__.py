"""app 包：基于统计优化的长程记忆与自纠错智能体系统。

模块结构：
    core/        配置与 LLM 客户端
    memory/      记忆生成、压缩、冲突消解
    retrieval/   混合检索与重排
    agent/       任务规划、状态机、自纠错
    evaluation/  数据集模拟与统计评估
    api/         FastAPI 路由
"""

__version__ = "0.1.0"
