"""memory 包：记忆生成、压缩与冲突消解模块。

架构定位（对应系统架构图第三层「记忆系统 · 统计优化」）：
- generator.py          记忆生成：从执行轨迹中抽取结构化事实
- compressor.py         记忆压缩：基于信息增益的上下文压缩
- conflict_resolver.py  冲突消解：贝叶斯信念更新
- models.py             核心数据模型（MemoryFact 等）
"""
