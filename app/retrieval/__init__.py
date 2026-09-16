"""retrieval 包：混合检索与重排模块。

架构定位（对应系统架构图第三层「检索排序 · 混合相关性模型」）：
- hybrid_retriever.py   BM25 + 向量召回 + 时间衰减融合
- reranker.py           候选重排（cross-encoder 或统计特征线性模型）
"""
