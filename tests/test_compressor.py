"""记忆压缩模块的单元测试（FakeLLM 注入，全离线）。

覆盖：触发门槛、TF-IDF 相似度矩阵性质、传递闭包聚类、
LLM 摘要合并（含降级路径）、输出顺序与统计聚合。
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.memory.compressor import (
    COMPRESSION_TRIGGER,
    SIMILARITY_THRESHOLD,
    LLMNotConfiguredError,
    _cluster_by_similarity,
    _cosine_matrix,
    _merge_cluster,
    compress_memories,
)
from app.memory.models import MemoryFact

T0 = datetime(2026, 9, 15, 8, 0, 0, tzinfo=timezone.utc)

NEAR_DUP_A = [
    "用户偏好使用 pytest 框架编写单元测试",
    "用户偏好使用 pytest 框架编写单元测试",   # 与第 1 条完全相同
    "用户偏好使用 pytest 框架编写单元测试呀",  # 高度相似（尾部微调，cos 仍 > 0.85）
]

DISTINCT = ["数据库使用 PostgreSQL", "部署环境是 Ubuntu 服务器", "项目截止日期为十月"]

MERGE_SUMMARY = "摘要：用户偏好使用 pytest 框架编写单元测试并关注覆盖率"


class FakeLLM:
    """测试替身：记录调用次数与消息，返回固定摘要。"""

    def __init__(self, response: str = MERGE_SUMMARY) -> None:
        self.response = response
        self.calls = 0
        self.last_messages = None

    def invoke(self, messages):
        self.calls += 1
        self.last_messages = messages
        return SimpleNamespace(content=self.response)


def make_fact(content: str, confidence: float = 0.5, offset_minutes: int = 0) -> MemoryFact:
    """构造带可控时间戳的测试记忆。"""
    return MemoryFact(
        content=content,
        confidence=confidence,
        timestamp=T0 + timedelta(minutes=offset_minutes),
    )


class TestTriggerCondition:
    """触发门槛测试。"""

    def test_five_or_fewer_never_compressed(self) -> None:
        """5 条（含）以内即使完全重复也不触发压缩。"""
        dup5 = [make_fact("完全相同的记忆") for _ in range(COMPRESSION_TRIGGER)]
        fake = FakeLLM()
        result = compress_memories(dup5, llm_client=fake)
        assert len(result) == COMPRESSION_TRIGGER
        assert fake.calls == 0

    def test_six_identical_trigger_compression(self) -> None:
        """6 条高冗余记忆触发压缩：3 条簇 -> 1 条摘要。"""
        memories = [make_fact(c, 0.8, i) for i, c in enumerate(NEAR_DUP_A)] + [
            make_fact(c, 0.5, i + 10) for i, c in enumerate(DISTINCT)
        ]
        fake = FakeLLM()
        result = compress_memories(memories, llm_client=fake)
        assert len(result) == 4  # 3 -> 1 + 3 个 singleton
        assert fake.calls == 1

    def test_empty_list_returns_empty(self) -> None:
        """空输入返回空列表。"""
        assert compress_memories([], llm_client=FakeLLM()) == []

    def test_no_redundancy_llm_never_called(self) -> None:
        """超过门槛但无冗余簇时，不调用 LLM，逐条返回。"""
        memories = [make_fact(f"独特记忆编号{i}关于主题{i}") for i in range(6)]
        fake = FakeLLM()
        result = compress_memories(memories, llm_client=fake)
        assert len(result) == 6
        assert fake.calls == 0
        assert [r.content for r in result] == [m.content for m in memories]


class TestCosineMatrix:
    """TF-IDF 余弦相似度矩阵测试。"""

    def test_matrix_properties(self) -> None:
        """矩阵对称、对角线为 1、值域 [0, 1]。"""
        corpus = NEAR_DUP_A + DISTINCT
        sim = _cosine_matrix(corpus)
        n = len(corpus)
        assert sim.shape == (n, n)
        for i in range(n):
            assert sim[i][i] == pytest.approx(1.0)
            for j in range(n):
                assert sim[i][j] == pytest.approx(sim[j][i])
                assert 0.0 <= sim[i][j] <= 1.0 + 1e-9

    def test_duplicates_similarity_near_one(self) -> None:
        """完全相同的文本相似度应为 1（冗余判定的正例）。"""
        sim = _cosine_matrix([NEAR_DUP_A[0], NEAR_DUP_A[1], DISTINCT[0]])
        assert sim[0][1] == pytest.approx(1.0)

    def test_distinct_similarity_below_threshold(self) -> None:
        """不相关文本的相似度应低于阈值（冗余判定的反例）。"""
        sim = _cosine_matrix([NEAR_DUP_A[0], DISTINCT[0], DISTINCT[1]])
        assert sim[0][1] < SIMILARITY_THRESHOLD
        assert sim[0][2] < SIMILARITY_THRESHOLD


class TestClusterBySimilarity:
    """传递闭包聚类测试。"""

    def test_transitive_closure_merges_chain(self) -> None:
        """A~B、B~C 相似但 A~C 未达阈值时，三者仍归为一簇（传递闭包）。"""
        # 构造相似度矩阵：0~1 与 1~2 超阈值，0~2 不超
        sim = [
            [1.0, 0.9, 0.5],
            [0.9, 1.0, 0.9],
            [0.5, 0.9, 1.0],
        ]
        clusters = _cluster_by_similarity(sim, SIMILARITY_THRESHOLD)
        assert list(clusters.values()) == [[0, 1, 2]]

    def test_singletons_excluded(self) -> None:
        """孤立点不出现在聚类结果中。"""
        sim = [
            [1.0, 0.95, 0.1],
            [0.95, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ]
        clusters = _cluster_by_similarity(sim, SIMILARITY_THRESHOLD)
        assert list(clusters.values()) == [[0, 1]]

    def test_strictly_greater_than_threshold(self) -> None:
        """相似度恰好等于阈值不算冗余（严格大于）。"""
        sim = [[1.0, 0.85], [0.85, 1.0]]
        assert _cluster_by_similarity(sim, 0.85) == {}


class TestMergeCluster:
    """簇合并（LLM 摘要）测试。"""

    def test_summary_fact_statistical_aggregation(self) -> None:
        """合并记忆：content 为 LLM 摘要、confidence 为均值、timestamp 取最新。"""
        cluster = [
            make_fact("甲", confidence=0.9, offset_minutes=0),
            make_fact("乙", confidence=0.7, offset_minutes=30),
            make_fact("丙", confidence=0.5, offset_minutes=60),
        ]
        merged = _merge_cluster(cluster, FakeLLM())
        assert merged.content == MERGE_SUMMARY
        assert merged.confidence == pytest.approx((0.9 + 0.7 + 0.5) / 3)
        assert merged.timestamp == T0 + timedelta(minutes=60)  # 簇内最新
        assert merged.id not in {m.id for m in cluster}  # 新 UUID

    def test_prompt_contains_all_contents(self) -> None:
        """合并 Prompt 必须携带簇内全部记忆文本。"""
        cluster = [make_fact(c) for c in NEAR_DUP_A]
        fake = FakeLLM()
        _merge_cluster(cluster, fake)
        text = fake.last_messages[0]["content"]
        for content in NEAR_DUP_A:
            assert content in text

    def test_empty_llm_response_falls_back_to_representative(self) -> None:
        """LLM 返回空摘要 -> 降级为簇内最高置信度代表元（流水线不断）。"""
        cluster = [
            make_fact("低置信条目", confidence=0.3),
            make_fact("高置信条目", confidence=0.95),
        ]
        fake = FakeLLM(response="   ")  # 空白响应
        merged = _merge_cluster(cluster, fake)
        assert merged.content == "高置信条目"
        assert merged.confidence == pytest.approx(0.95)

    def test_non_text_response_falls_back(self) -> None:
        """LLM 返回非文本同样触发代表元降级。"""
        cluster = [make_fact("条目", confidence=0.8)]
        weird = FakeLLM()
        weird.invoke = lambda messages: {"oops": 1}
        merged = _merge_cluster(cluster, weird)
        assert merged.content == "条目"

    def test_no_client_raises(self) -> None:
        """显式传 None 且全局未配置时抛异常。"""
        with pytest.raises(LLMNotConfiguredError):
            _merge_cluster([make_fact("甲"), make_fact("乙")], None)


class TestOutputAssembly:
    """压缩输出的结构与顺序测试。"""

    def test_merged_fact_replaces_cluster_at_first_position(self) -> None:
        """摘要记忆应出现在簇首原来的位置，singleton 顺序保持。"""
        memories = [make_fact(c, 0.8, i) for i, c in enumerate(NEAR_DUP_A)] + [
            make_fact(DISTINCT[0]),
            make_fact(DISTINCT[1]),
            make_fact(DISTINCT[2]),
        ]
        result = compress_memories(memories, llm_client=FakeLLM())
        assert len(result) == 4
        assert result[0].content == MERGE_SUMMARY       # 簇首位置
        assert result[1].content == DISTINCT[0]          # singleton 顺序不变
        assert result[2].content == DISTINCT[1]
        assert result[3].content == DISTINCT[2]

    def test_two_clusters_both_merged(self) -> None:
        """两个独立冗余簇各自合并（6 条 -> 2 条）。"""
        group_a = NEAR_DUP_A
        group_b = ["部署环境是 Ubuntu 服务器版本", "部署环境是 Ubuntu 服务器版本", "部署环境是 Ubuntu 服务器版本"]
        memories = [make_fact(c, 0.6) for c in group_a] + [make_fact(c, 0.9) for c in group_b]
        fake = FakeLLM()
        result = compress_memories(memories, llm_client=fake)
        assert len(result) == 2
        assert fake.calls == 2
        confidences = sorted(r.confidence for r in result)
        assert confidences == pytest.approx([0.6, 0.9])

    def test_singleton_confidences_preserved(self) -> None:
        """未参与合并的记忆，confidence 必须原样保留。"""
        memories = [make_fact(c, 0.8, i) for i, c in enumerate(NEAR_DUP_A)] + [
            make_fact(DISTINCT[0], 0.33),
            make_fact(DISTINCT[1], 0.44),
            make_fact(DISTINCT[2], 0.55),
        ]
        result = compress_memories(memories, llm_client=FakeLLM())
        kept = [r.confidence for r in result[1:]]
        assert kept == [pytest.approx(0.33), pytest.approx(0.44), pytest.approx(0.55)]


class TestInputValidation:
    """输入校验测试。"""

    def test_non_list_rejected(self) -> None:
        with pytest.raises(TypeError):
            compress_memories("不是列表", llm_client=FakeLLM())

    def test_non_memory_fact_elements_rejected(self) -> None:
        with pytest.raises(TypeError):
            compress_memories([make_fact("甲"), "字符串", None], llm_client=FakeLLM())

    def test_not_mutating_input(self) -> None:
        """压缩不得修改输入列表（返回新列表）。"""
        memories = [make_fact(c) for c in NEAR_DUP_A] + [make_fact(c) for c in DISTINCT]
        snapshot = [m.id for m in memories]
        compress_memories(memories, llm_client=FakeLLM())
        assert [m.id for m in memories] == snapshot
        assert len(memories) == 6
