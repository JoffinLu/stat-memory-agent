"""混合检索模块：向量检索 + BM25 词法检索 + RRF 融合。

架构：
    query -> [向量路径: ChromaDB 余弦相似度]  \
              -> Reciprocal Rank Fusion -> Top-K 记忆
            [词法路径: BM25Okapi 字符二元组] /

RRF（Reciprocal Rank Fusion, Cormack et al. 2009）：
    RRF_score(d) = sum_i 1 / (k + rank_i(d))
    其中 rank_i 为文档在第 i 路检索结果中的名次（1 起），
    k 为平滑常数（原论文 60），抑制头部名次的过大差距。

设计要点：
1. 向量库与 LLM 同样走依赖注入 —— 单测用内存假向量库，零重依赖；
2. BM25 分词用字符二元组（与压缩模块同款决策）：中文免分词，
   英文同样有效；
3. 两路结果的 id 交集文档天然获得两份 RRF 贡献 —— 这正是 RRF
   「两路都认为相关才真正相关」的设计意图。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.memory.models import MemoryFact

logger = logging.getLogger(__name__)

# RRF 平滑常数默认值（Cormack et al. 2009 原论文取值）
DEFAULT_K_RRF: int = 60


@dataclass
class RetrievalResult:
    """单条检索结果。

    Attributes:
        fact: 命中的记忆。
        score: RRF 融合得分（越大越相关）。
    """

    fact: MemoryFact
    score: float


def _tokenize(text: str) -> List[str]:
    """字符二元组分词器（中文免分词方案）。

    jieba 等中文分词器是额外重依赖且引入分词歧义；
    字符二元组对中文检索是业界验证过的轻量替代，
    对英文连续字母串同样产生二元组，两可适用。
    """
    cleaned = "".join(text.split())
    if len(cleaned) < 2:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]


def _rrf_fuse(rankings: Dict[str, List[str]], k: int = DEFAULT_K_RRF) -> Dict[str, float]:
    """Reciprocal Rank Fusion 融合多路排名。

    Args:
        rankings: {检索路名称 -> 按Related性降序的文档 id 列表}。
        k: 平滑常数，k 越大各名次差距越平缓。

    Returns:
        {文档 id -> RRF 得分}，仅含至少出现在一路结果中的文档。
    """
    scores: Dict[str, float] = {}
    for ranked_ids in rankings.values():
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class ChromaVectorStore:
    """ChromaDB 本地持久化向量库封装（生产路径，懒加载）。

    chromadb 依赖较重（含 onnxruntime 等），与 langchain 同策略：
    仅在实际构造本类时才 import，测试用注入的假向量库替代。
    """

    def __init__(
        self,
        persist_dir: str,
        embedding_fn: Callable[[str], List[float]],
        collection_name: str = "memories",
    ) -> None:
        """连接（或创建）本地 ChromaDB 持久化库。

        Args:
            persist_dir: 持久化目录（对应 Settings.chroma_persist_dir）。
            embedding_fn: 文本 -> 向量函数（由调用方注入，
                如 OpenAI text-embedding-3-small）。
            collection_name: 集合名。
        """
        import chromadb  # noqa: PLC0415  懒加载重依赖

        self._embed = embedding_fn
        self._client = chromadb.PersistentClient(path=persist_dir)
        # cosine 空间：Chroma 返回余弦距离，1 - distance 即余弦相似度
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, fact: MemoryFact, embedding: Optional[List[float]] = None) -> None:
        """写入/更新一条记忆（upsert 语义）。"""
        vector = embedding if embedding is not None else self._embed(fact.content)
        self._collection.upsert(
            ids=[fact.id],
            embeddings=[vector],
            documents=[fact.content],
            metadatas=[
                {
                    "confidence": fact.confidence,
                    "timestamp": fact.timestamp.isoformat(),
                }
            ],
        )

    def delete(self, fact_id: str) -> None:
        """从 ChromaDB 集合中删除一条文档。"""
        self._collection.delete(ids=[fact_id])

    def query(self, query_text: str, top_k: int) -> List[Tuple[MemoryFact, float]]:
        """按余弦相似度检索 Top-K。"""
        count = self._collection.count()
        if count == 0:
            return []
        vector = self._embed(query_text)
        response = self._collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        results: List[Tuple[MemoryFact, float]] = []
        for doc_id, doc, meta, dist in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            fact = MemoryFact(
                id=doc_id,
                content=doc,
                confidence=float(meta["confidence"]),
                timestamp=datetime.fromisoformat(meta["timestamp"]),
            )
            results.append((fact, 1.0 - float(dist)))  # 余弦距离 -> 相似度
        return results


def hash_embedding(text: str, dim: int = 256) -> List[float]:
    """零依赖的确定性嵌入：字符二元组 md5 哈希计数向量（L2 归一化）。

    共享二元组越多 -> 余弦相似度越高，是真实语义嵌入的离线近似 ——
    足够支撑小语料的混合检索与离线基准，且完全确定、可复现。
    生产环境换成真实 embedding_fn 即可，接口不变。
    """
    import hashlib

    vec = [0.0] * dim
    for gram in _tokenize(text):
        digest = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
        vec[digest % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class InMemoryVectorStore:
    """内置内存向量库（与 ChromaVectorStore 同接口，暴力余弦检索）。

    适用：小语料（万条以内）、离线评估、无 chromadb 依赖的部署环境。
    超出规模或需要持久化时换 ChromaVectorStore，接口完全兼容。
    """

    def __init__(self, embedding_fn: Optional[Callable[[str], List[float]]] = None) -> None:
        self._embed = embedding_fn or hash_embedding
        self._items: Dict[str, Tuple[MemoryFact, List[float]]] = {}

    def add(self, fact: MemoryFact, embedding: Optional[List[float]] = None) -> None:
        """写入/更新一条记忆（upsert 语义）。"""
        vector = embedding if embedding is not None else self._embed(fact.content)
        self._items[fact.id] = (fact, vector)

    def delete(self, fact_id: str) -> None:
        """删除一条记忆；id 不存在时静默（与 ChromaDB 语义一致）。"""
        self._items.pop(fact_id, None)

    def query(self, query_text: str, top_k: int) -> List[Tuple[MemoryFact, float]]:
        """按余弦相似度检索 Top-K（相似度 <= 0 的正交命中被过滤）。"""
        if not self._items:
            return []
        q = self._embed(query_text)
        scored = []
        for fact, emb in self._items.values():
            sim = sum(a * b for a, b in zip(q, emb))
            if sim > 0.0:  # 正交/负相关文档不参与排名（与 Chroma 路策略一致）
                scored.append((fact, sim))
        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._items)


class HybridSearch:
    """混合检索器（BM25 + 向量 + RRF 融合）。

    Args:
        vector_store: 向量库实例（如 ChromaVectorStore 或测试假库）。
            为 None 时仅启用 BM25 词法路径。
        k_rrf: RRF 平滑常数，默认 60。
    """

    def __init__(
        self,
        vector_store: Optional[Any] = None,
        k_rrf: int = DEFAULT_K_RRF,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        self._vector_store = vector_store
        self._k_rrf = k_rrf
        self._embedding_fn = embedding_fn
        self._corpus: List[MemoryFact] = []
        self._bm25: Optional[Any] = None

    # ---------- 索引维护 ----------

    def add_fact(self, fact: MemoryFact, embedding: Optional[List[float]] = None) -> None:
        """索引一条记忆：同时写入向量库与 BM25 语料（保持两路同步）。"""
        if self._vector_store is not None:
            self._vector_store.add(fact, embedding)
        self._corpus.append(fact)
        self._rebuild_bm25()

    def add_facts(self, facts: List[MemoryFact]) -> None:
        """批量索引（逐条复用 add_fact，保持实现简单；规模大再优化）。"""
        for fact in facts:
            self.add_fact(fact)

    def _rebuild_bm25(self) -> None:
        """重建 BM25 索引（全量重建，O(n)；当前规模下足够，注意。）。"""
        from rank_bm25 import BM25Okapi  # noqa: PLC0415

        if not self._corpus:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi([_tokenize(f.content) for f in self._corpus])

    # ---------- 索引维护（供编排层使用的增删查接口） ----------

    @property
    def facts(self) -> List[MemoryFact]:
        """当前语料的浅拷贝列表（防止外部直接改内部状态）。"""
        return list(self._corpus)

    def __len__(self) -> int:
        return len(self._corpus)

    def remove_fact(self, fact_id: str) -> bool:
        """按 id 移除一条记忆（同步删除向量库文档）并重建 BM25 索引。

        Returns:
            移除了返回 True；id 不存在返回 False。
        """
        remaining = [f for f in self._corpus if f.id != fact_id]
        if len(remaining) == len(self._corpus):
            return False
        self._corpus = remaining
        if self._vector_store is not None:
            self._vector_store.delete(fact_id)  # 防止向量库"幽灵记忆"复活
        self._rebuild_bm25()
        return True

    # ---------- 检索 ----------

    def search(self, query: str, top_k: int = 50) -> List[RetrievalResult]:
        """混合检索：两路召回 -> RRF 融合 -> Top-K。

        Args:
            query: 查询文本，非空。
            top_k: 返回条数上限，默认 50。

        Returns:
            按 RRF 得分降序的检索结果（同分按 id 字典序稳定排序）。

        Raises:
            ValueError: query 为空或 top_k < 1。
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        if top_k < 1:
            raise ValueError(f"top_k 必须 >= 1，收到: {top_k}")

        rankings: Dict[str, List[str]] = {}

        # 路径一：BM25 词法检索（仅保留得分 > 0 的候选参与排名）
        # 注意：BM25Okapi 的 idf = log((N-n+0.5)/(n+0.5))，小语料上
        # （n 接近 N/2）idf 可为 0 —— 这是小语料 BM25 的固有弱点，
        # 生产语料足够大时自然消解，且 RRF 的另一路可补偿。
        bm25_hits: Dict[str, MemoryFact] = {}
        if self._bm25 is not None:
            scores = self._bm25.get_scores(_tokenize(query))
            scored = sorted(
                (
                    (self._corpus[i], s)
                    for i, s in enumerate(scores)
                    if s > 0
                ),
                key=lambda pair: (-pair[1], pair[0].id),
            )
            ranked = scored[:top_k]
            rankings["bm25"] = [fact.id for fact, _ in ranked]
            bm25_hits = {fact.id: fact for fact, _ in ranked}

        # 路径二：向量余弦检索
        # 过滤相似度 <= 0 的命中：ChromaDB 类向量库按最近邻返回，
        # 不看绝对相似度 —— 正交（无关）文档不应参与 RRF 排名。
        vec_hits: Dict[str, MemoryFact] = {}
        if self._vector_store is not None:
            vec_ranked = [
                (fact, sim)
                for fact, sim in self._vector_store.query(query, top_k)
                if sim > 0
            ]
            rankings["vector"] = [fact.id for fact, _ in vec_ranked]
            vec_hits = {fact.id: fact for fact, _ in vec_ranked}

        if not rankings:
            return []

        # RRF 融合：出现在两路中的文档自动叠加两份贡献
        fused = _rrf_fuse(rankings, self._k_rrf)
        id_to_fact = {**bm25_hits, **vec_hits}
        ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))

        return [
            RetrievalResult(fact=id_to_fact[doc_id], score=score)
            for doc_id, score in ordered[:top_k]
        ]
