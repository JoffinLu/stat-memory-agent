"""统计评估模块（阶段五实现，ndcg_at_k 已就绪）。

规划：为阶段五的消融实验提供指标层，量化每个统计模块的收益：
- 检索质量：NDCG@K、MRR（本项目函数）+ scipy.stats 的显著性检验
- 自纠错收益：纠错触发后的成功率提升 + 配对 t 检验 / Wilcoxon 检验
- 成本维度：平均 LLM 调用次数、token 消耗

ndcg_at_k 为纯函数，可立即单测。
"""

from typing import List
import math


def ndcg_at_k(relevance: List[float], k: int) -> float:
    """计算 NDCG@K（Normalized Discounted Cumulative Gain）。

    DCG@k  = sum(rel_i / log2(i + 1)) for i in 1..k
    IDCG@k = DCG@k of the ideal (descending) ranking
    NDCG@k = DCG@k / IDCG@k

    Args:
        relevance: 按系统排序顺序给出的相关性分数列表。
        k: 截断位置，1 <= k <= len(relevance)。

    Returns:
        NDCG@K，范围 [0, 1]。理想排序时为 1。

    Raises:
        ValueError: k 越界或 relevance 含负值。

    Note:
        选 NDCG 而非 Precision@K 的原因：记忆相关性是分级而非二元的，
        NDCG 对分级标签且带有位置折扣，更贴合检索评估。
    """
    if k < 1 or k > len(relevance):
        raise ValueError(f"k 必须在 [1, {len(relevance)}]，当前: {k}")
    if any(r < 0 for r in relevance):
        raise ValueError("相关性分数不允许为负")

    def dcg(scores: List[float]) -> float:
        return sum(
            score / math.log2(rank + 1)
            for rank, score in enumerate(scores[:k], start=1)
        )

    actual = dcg(relevance)
    ideal = dcg(sorted(relevance, reverse=True))

    if ideal == 0.0:
        return 0.0
    return actual / ideal


def mrr(relevance_binary: List[int]) -> float:
    """计算 MRR（Mean Reciprocal Rank，单查询版本）。

    Args:
        relevance_binary: 按系统排序顺序给出的 0/1 相关列表（1 = 相关）。

    Returns:
        第一个相关文档的 1/rank；若无相关文档返回 0.0。

    Raises:
        ValueError: 列表为空或含非法值。
    """
    if not relevance_binary:
        raise ValueError("相关性列表不能为空")
    if any(r not in (0, 1) for r in relevance_binary):
        raise ValueError("MRR 要求二值相关性（0 或 1）")

    for rank, relevant in enumerate(relevance_binary, start=1):
        if relevant == 1:
            return 1.0 / rank
    return 0.0
