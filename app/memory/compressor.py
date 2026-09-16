"""记忆压缩模块：对同主题冗余记忆做 TF-IDF 聚类 + LLM 摘要合并。

触发条件：同一主题（本函数的输入列表）记忆数超过 5 条。
流程：
    memories > 5 条
        -> TfidfVectorizer（char_wb 字符 n-gram，兼容中文）
        -> 余弦相似度矩阵
        -> 相似度 > 0.85 的记忆经 Union-Find 传递闭包聚类
        -> 每个簇调用 LLM 摘要合并为一条 MemoryFact
        -> 簇内置换（摘要落在簇首位置），Singleton 原样保留

统计决策说明（面试可讲的点）：
- 合并后的 confidence 取簇内均值而非 noisy-OR：高相似意味着这些记忆
  是「同一事实的重复陈述」而非「独立证据」，noisy-OR 的独立性假设
  在此不成立，会虚假膨胀置信度；均值是对底层事实置信度的保守估计。
- timestamp 取簇内最大值：合并记忆的「最后已知时间」应为最新证据时间。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from app.core.llm_client import get_llm_client
from app.memory.models import MemoryFact

logger = logging.getLogger(__name__)


class LLMNotConfiguredError(RuntimeError):
    """需要 LLM 摘要合并但客户端未配置（离线模式）。"""


# 同主题记忆数超过此值才触发压缩
COMPRESSION_TRIGGER: int = 5
# 余弦相似度超过此值视为冗余
SIMILARITY_THRESHOLD: float = 0.85

MERGE_PROMPT_TEMPLATE = (
    "以下是与同一主题相关的多条记忆，它们内容高度相似（互为冗余）。\n"
    "请将它们合并为一条摘要记忆，要求保留所有条目中的信息，"
    "只输出摘要文本本身，不要任何解释、前缀或 Markdown 标记。\n\n"
    "记忆条目：\n{memories}"
)


def _build_merge_messages(numbered_contents: str) -> List[Any]:
    """构造簇合并的 LLM 消息序列（与 extractor 相同的双路径策略）。"""
    human_text = MERGE_PROMPT_TEMPLATE.replace("{memories}", numbered_contents)
    try:
        from langchain_core.messages import HumanMessage  # noqa: PLC0415

        return [HumanMessage(content=human_text)]
    except ImportError:
        return [{"role": "user", "content": human_text}]


def _cosine_matrix(corpus: Sequence[str]):
    """计算文本语料的 TF-IDF 余弦相似度矩阵。

    主动决策（偏离字面需求的点，已标注）：
    TfidfVectorizer 默认按词切分（token_pattern 匹配 2+ 词字符），
    对中文会把整段连续汉字当成一个 token，导致 TF-IDF 失效。
    因此改用 char_wb 字符 2~4 元组分析器——无需 jieba 分词即可
    让中文文本获得有意义的相似度度量；对英文同样有效。

    Args:
        corpus: 文本列表，长度须 >= 2。

    Returns:
        n x n numpy 矩阵，对角线为 1，对称。
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415
    from sklearn.metrics.pairwise import cosine_similarity  # noqa: PLC0415

    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(corpus)
    return cosine_similarity(tfidf)


def _cluster_by_similarity(sim, threshold: float) -> Dict[int, List[int]]:
    """按相似度阈值聚类：相似对经 Union-Find 求传递闭包。

    为什么用传递闭包而非直接按对分组：A~B、B~C 但 A~C 只有 0.84 时，
    逐对判断会把它们拆散；传递闭包将 A/B/C 归为一簇——冗余关系
    本质上是图连通性问题。

    Args:
        sim: n x n 相似度矩阵。
        threshold: 冗余判定阈值（不含，即严格大于）。

    Returns:
        {簇代表根索引 -> [成员索引]}，仅含成员数 >= 2 的簇。
    """
    n = len(sim)
    parent = list(range(n))

    def find(x: int) -> int:
        """路径压缩的查找。"""
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 路径减半
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i][j] > threshold:
                parent[find(i)] = find(j)

    clusters: Dict[int, List[int]] = {}
    for idx in range(n):
        clusters.setdefault(find(idx), []).append(idx)

    return {root: members for root, members in clusters.items() if len(members) >= 2}


def _merge_cluster(cluster: List[MemoryFact], llm_client: Any) -> MemoryFact:
    """调用 LLM 将一个冗余簇摘要合并为单条 MemoryFact。

    Args:
        cluster: 高相似记忆列表（长度 >= 2）。
        llm_client: LLM 客户端（invoke(messages) 返回 .content 或 str）。

    Returns:
        合并后的记忆：content 为 LLM 摘要，confidence 为簇内均值，
        timestamp 为簇内最新时间，id 为新生成 UUID。

    Raises:
        LLMNotConfiguredError: 客户端不可用。
    """
    client = llm_client if llm_client is not None else get_llm_client()
    if client is None:
        raise LLMNotConfiguredError("LLM 客户端未配置（SMA_LLM_API_KEY 未设置），无法进行摘要合并")

    numbered = "\n".join(f"{k}. {m.content}" for k, m in enumerate(cluster, start=1))
    response = client.invoke(_build_merge_messages(numbered))
    raw_text = getattr(response, "content", response)

    if isinstance(raw_text, str) and raw_text.strip():
        summary = raw_text.strip()
    else:
        # LLM 返回空/非文本：降级为代表元策略（保留信息，流水线不断）
        representative = max(cluster, key=lambda m: m.confidence)
        logger.warning("LLM 摘要为空，降级为簇内最高置信度代表元")
        return representative.model_copy()

    mean_confidence = sum(m.confidence for m in cluster) / len(cluster)
    latest = max(m.timestamp for m in cluster)
    return MemoryFact(content=summary, confidence=mean_confidence, timestamp=latest)


def compress_memories(
    memories: List[MemoryFact],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    trigger_size: int = COMPRESSION_TRIGGER,
    llm_client: Optional[Any] = None,
) -> List[MemoryFact]:
    """压缩同主题冗余记忆。

    Args:
        memories: 同一主题的记忆列表（由调用方按主题分组后传入）。
        similarity_threshold: 冗余相似度阈值，默认 0.85。
        trigger_size: 触发压缩的数量门槛，默认 5（超过才压缩）。
        llm_client: 可注入的 LLM 客户端；None 时用全局单例。

    Returns:
        压缩后的记忆列表：冗余簇被替换为单条摘要（置于簇首位置），
        Singleton 保持原顺序。未触发门槛时返回原列表的浅拷贝。

    Raises:
        TypeError: 输入不是 MemoryFact 列表。
        LLMNotConfiguredError: 触发压缩但存在冗余簇且 LLM 不可用。
    """
    if not isinstance(memories, (list, tuple)):
        raise TypeError(f"memories 必须是列表，收到: {type(memories).__name__}")
    if any(not isinstance(m, MemoryFact) for m in memories):
        raise TypeError("memories 中包含非 MemoryFact 元素")

    # 未达门槛：不压缩，返回浅拷贝（避免调用方误改内部状态）
    if len(memories) <= trigger_size:
        return list(memories)

    sim = _cosine_matrix([m.content for m in memories])
    clusters = _cluster_by_similarity(sim, similarity_threshold)

    if not clusters:
        logger.info("共 %d 条记忆，无冗余簇，跳过 LLM 合并", len(memories))
        return list(memories)

    consumed: set = set()
    result: List[MemoryFact] = []
    for idx, fact in enumerate(memories):
        if idx in consumed:
            continue
        root_members = None
        for members in clusters.values():
            if idx in members:
                root_members = members
                break
        if root_members is None:
            result.append(fact)
            continue
        cluster = [memories[j] for j in root_members]
        merged = _merge_cluster(cluster, llm_client)
        result.append(merged)
        consumed.update(root_members)
        logger.info("簇 %s (%d 条) 已合并为 1 条", root_members, len(root_members))

    return result
