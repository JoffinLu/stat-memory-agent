"""混合检索模块的单元测试（内存假向量库 + 确定性玩具嵌入，全离线）。

覆盖：RRF 融合公式（精确值）、字符二元组分词、两路召回与融合、
索引同步、top_k 截断、输入校验、（可选）真实 ChromaDB 封装。
"""

import math
from typing import List, Optional, Tuple

import pytest
from conftest import FakeVectorStore, toy_embedding

from app.memory.models import MemoryFact
from app.retrieval.hybrid_search import (
    DEFAULT_K_RRF,
    HybridSearch,
    RetrievalResult,
    _rrf_fuse,
    _tokenize,
)


def make_fact(content: str) -> MemoryFact:
    return MemoryFact(content=content)


class TestTokenizer:
    """字符二元组分词测试。"""

    def test_basic_bigrams(self) -> None:
        assert _tokenize("偏好") == ["偏好"]
        assert _tokenize("用户偏好") == ["用户", "户偏", "偏好"]

    def test_whitespace_removed(self) -> None:
        """空白在分词前被剔除（英文按字符流处理）。"""
        assert _tokenize("a b") == ["ab"]

    def test_single_char_fallback(self) -> None:
        assert _tokenize("好") == ["好"]
        assert _tokenize("") == []


class TestRRFFusion:
    """RRF 公式精确值测试。"""

    def test_exact_scores(self) -> None:
        """两路排名已知时逐文档精确验证 1/(k+rank) 求和。"""
        rankings = {"bm25": ["A", "B"], "vector": ["B", "C"]}
        fused = _rrf_fuse(rankings, k=60)
        assert fused["A"] == pytest.approx(1 / 61)
        assert fused["B"] == pytest.approx(1 / 61 + 1 / 62)  # 两路都命中，叠加
        assert fused["C"] == pytest.approx(1 / 62)
        # 两路共识者 B 排第一 —— RRF 的设计意图
        assert fused["B"] > fused["A"] > fused["C"]

    def test_k_smaller_sharpens_head(self) -> None:
        """k 越小，头部名次优势越大（平滑减弱）。"""
        rankings = {"a": ["X", "Y"]}
        top_gap_k1 = _rrf_fuse(rankings, k=1)["X"] - _rrf_fuse(rankings, k=1)["Y"]
        top_gap_k60 = _rrf_fuse(rankings, k=60)["X"] - _rrf_fuse(rankings, k=60)["Y"]
        assert top_gap_k1 > top_gap_k60

    def test_default_k_is_60(self) -> None:
        assert DEFAULT_K_RRF == 60

    def test_empty_rankings(self) -> None:
        assert _rrf_fuse({}) == {}


class TestHybridSearch:
    """混合检索主流程测试。"""

    def test_relevant_document_ranks_first(self) -> None:
        """查询与某条记忆共享大量二元组 -> 该记忆排首位。"""
        searcher = HybridSearch(vector_store=FakeVectorStore())
        searcher.add_facts(
            [
                make_fact("用户偏好使用 pytest 编写单元测试"),
                make_fact("项目部署在 Ubuntu 服务器"),
                make_fact("数据库使用 PostgreSQL"),
            ]
        )
        results = searcher.search("编写单元测试的工具偏好", top_k=3)
        assert results[0].fact.content == "用户偏好使用 pytest 编写单元测试"

    def test_dual_path_hit_scores_higher(self) -> None:
        """同一条记忆在两路都命中 -> 得分高于仅单路命中的记忆。"""
        searcher = HybridSearch(vector_store=FakeVectorStore())
        both = make_fact("用户偏好使用 pytest 编写单元测试")   # 词法+语义都相关
        only_vec = make_fact("pytest 是测试框架之选")           # 部分二元组重叠
        searcher.add_facts([both, only_vec])
        results = searcher.search("用户偏好 pytest 编写单元测试", top_k=2)
        assert results[0].fact.id == both.id
        assert results[0].score > results[1].score

    def test_top_k_truncates(self) -> None:
        """top_k 截断返回数量。"""
        searcher = HybridSearch(vector_store=FakeVectorStore())
        searcher.add_facts([make_fact(f"记忆条目编号{i}内容各异") for i in range(10)])
        results = searcher.search("记忆条目", top_k=3)
        assert len(results) == 3

    def test_top_k_larger_than_corpus(self) -> None:
        """top_k 大于语料规模时返回全部有分文档。"""
        searcher = HybridSearch(vector_store=FakeVectorStore())
        searcher.add_facts([make_fact("唯一记忆")])
        results = searcher.search("唯一记忆", top_k=50)
        assert len(results) == 1

    def test_result_type_and_order(self) -> None:
        """返回 RetrievalResult 列表，得分降序。"""
        searcher = HybridSearch(vector_store=FakeVectorStore())
        searcher.add_facts([make_fact("苹果手机评测"), make_fact("苹果公司财报")])
        results = searcher.search("苹果", top_k=2)
        assert all(isinstance(r, RetrievalResult) for r in results)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_match_returns_empty(self) -> None:
        """查询与语料毫无重叠 -> 两路都无候选 -> 空列表。"""
        searcher = HybridSearch(vector_store=FakeVectorStore())
        searcher.add_facts([make_fact("数据库使用 PostgreSQL")])
        results = searcher.search("量子纠缠", top_k=5)
        assert results == []

    def test_empty_index_returns_empty(self) -> None:
        """空索引（未 add 过任何记忆）不报错，返回空。"""
        searcher = HybridSearch(vector_store=FakeVectorStore())
        assert searcher.search("任意查询", top_k=5) == []


class TestVectorStoreOptional:
    """单路降级测试。"""

    def test_bm25_only_mode(self) -> None:
        """vector_store=None 时仅 BM25 路径工作。

        语料用 4 条：BM25Okapi 的 idf 在极小语料上（n 接近 N/2）
        会衰减为 0，这是 BM25 的固有特性，测试语料需足够大。
        """
        searcher = HybridSearch(vector_store=None)
        searcher.add_facts(
            [
                make_fact("用户偏好使用 pytest 编写测试"),
                make_fact("数据库使用 PostgreSQL 存储"),
                make_fact("部署环境是 Ubuntu 服务器"),
                make_fact("项目使用 Python 语言开发"),
            ]
        )
        results = searcher.search("用户偏好", top_k=4)
        assert len(results) == 1
        assert results[0].fact.content == "用户偏好使用 pytest 编写测试"

    def test_add_fact_syncs_both_paths(self) -> None:
        """add_fact 必须同时写入向量库与 BM25 语料。"""
        store = FakeVectorStore()
        searcher = HybridSearch(vector_store=store)
        searcher.add_fact(make_fact("新增记忆条目"))
        assert len(store._items) == 1          # 向量路已同步
        assert len(searcher.search("新增记忆条目", top_k=5)) == 1  # 词法路可检回


class TestInputValidation:
    """输入校验测试。"""

    @pytest.mark.parametrize("bad_query", ["", "   ", None, 42])
    def test_invalid_query_rejected(self, bad_query) -> None:
        searcher = HybridSearch(vector_store=FakeVectorStore())
        with pytest.raises(ValueError):
            searcher.search(bad_query, top_k=5)

    @pytest.mark.parametrize("bad_k", [0, -3])
    def test_invalid_top_k_rejected(self, bad_k) -> None:
        searcher = HybridSearch(vector_store=FakeVectorStore())
        with pytest.raises(ValueError):
            searcher.search("查询", top_k=bad_k)


class TestChromaWrapper:
    """真实 ChromaDB 封装测试（未安装 chromadb 时自动跳过）。"""

    def test_chroma_roundtrip(self, tmp_path) -> None:
        pytest.importorskip("chromadb")
        from app.retrieval.hybrid_search import ChromaVectorStore

        store = ChromaVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            embedding_fn=toy_embedding,
        )
        fact = make_fact("用户偏好使用 pytest 编写单元测试")
        store.add(fact)
        hits = store.query("编写单元测试", top_k=1)
        assert len(hits) == 1
        assert hits[0][0].id == fact.id
        assert hits[0][1] > 0.5  # 余弦相似度合理
