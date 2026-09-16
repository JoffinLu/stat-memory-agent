"""记忆编排门面 MemoryManager 的单元测试（FakeLLM + 假向量库，全离线）。

覆盖：写入 pipeline（存储/覆盖/保留/融合）、档案沉淀、自动压缩触发、
读取 pipeline、状态统计。
"""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from conftest import FakeVectorStore

from app.memory.extractor import LLMNotConfiguredError
from app.memory.manager import MemoryManager
from app.memory.models import MemoryFact

T_NOW = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)

DUP_CONTENT = "用户偏好使用 pytest 编写单元测试"
EXTRACT_ONE = [{"content": DUP_CONTENT, "confidence": 0.9}]
EXTRACT_LOW = [{"content": DUP_CONTENT, "confidence": 0.4}]
EXTRACT_TWO = [
    {"content": DUP_CONTENT, "confidence": 0.8},
    {"content": "项目部署在 Ubuntu 服务器", "confidence": 0.7},
]
# 覆盖场景：0.95 vs 0.8 差 0.15 > 0.1（0.9 与 0.8 的浮点差是 0.0999，会误入融合路径）
EXTRACT_REPLACE = [
    {"content": DUP_CONTENT, "confidence": 0.95},
    {"content": "项目部署在 Ubuntu 服务器", "confidence": 0.7},
]
FUSED_TEXT = "融合记忆：用户偏好使用 pytest 框架编写单元测试"


class SmartFakeLLM:
    """按 Prompt 类型分发的测试替身：抽取请求回 JSON，融合请求回文本。"""

    def __init__(self, extract_items=None, fused_text: str = FUSED_TEXT) -> None:
        self.extract_items = extract_items if extract_items is not None else EXTRACT_ONE
        self.fused_text = fused_text
        self.extract_calls = 0
        self.fuse_calls = 0

    def invoke(self, messages):
        combined = " ".join(
            getattr(m, "content", m.get("content", "")) for m in messages
        )
        if "JSON 数组" in combined:  # 抽取模板特征
            self.extract_calls += 1
            return SimpleNamespace(content=json.dumps(self.extract_items, ensure_ascii=False))
        self.fuse_calls += 1  # 压缩/融合模板
        return SimpleNamespace(content=self.fused_text)


def make_fact(content: str, confidence: float = 0.5) -> MemoryFact:
    return MemoryFact(content=content, confidence=confidence, timestamp=T_NOW)


def make_manager(**kwargs) -> MemoryManager:
    """构造双路全开的被测实例（向量库注入，贴近生产形态）。"""
    kwargs.setdefault("vector_store", FakeVectorStore())
    return MemoryManager(**kwargs)


class TestIngestPipeline:
    """写入 pipeline 测试。"""

    def test_fresh_fact_stored(self) -> None:
        """库中无相关记忆 -> 直接存储。"""
        manager = make_manager(llm_client=SmartFakeLLM())
        report = manager.ingest("老板说他偏好 pytest", now=T_NOW)
        assert report.extracted == 1
        assert report.actions[0]["action"] == "stored"
        assert manager.size == 1

    def test_two_facts_both_stored(self) -> None:
        """一次对话抽出两条互不相关的事实 -> 都入库。"""
        manager = make_manager(llm_client=SmartFakeLLM(extract_items=EXTRACT_TWO))
        report = manager.ingest("一段包含两个事实的对话", now=T_NOW)
        assert report.extracted == 2
        assert all(a["action"] == "stored" for a in report.actions)
        assert manager.size == 2

    def test_higher_confidence_replaces_existing(self) -> None:
        """新事实有效置信度显著更高 -> 覆盖，旧记忆入档案。"""
        manager = make_manager(llm_client=SmartFakeLLM(extract_items=EXTRACT_ONE))
        manager.add_fact(make_fact("用户偏好使用 pytest 编写单元测试", confidence=0.4))
        report = manager.ingest("再次确认偏好", now=T_NOW)
        assert report.actions[0]["action"] == "replaced"
        assert manager.size == 1                       # 旧的去、新的来
        assert manager.archive_size == 1               # 旧记忆沉淀
        contents = [r.fact.content for r in manager.recall("pytest 单元测试")]
        assert contents == ["用户偏好使用 pytest 编写单元测试"]  # 新内容同文，但 id 已换

    def test_lower_confidence_goes_to_archive(self) -> None:
        """新事实置信度显著更低 -> 主库不动，新记忆入档案。"""
        manager = make_manager(llm_client=SmartFakeLLM(extract_items=EXTRACT_LOW))
        manager.add_fact(make_fact("用户偏好使用 pytest 编写单元测试", confidence=0.9))
        report = manager.ingest("低置信重复陈述", now=T_NOW)
        assert report.actions[0]["action"] == "kept_with_history"
        assert manager.size == 1
        assert manager.archive_size == 1

    def test_close_confidence_fuses(self) -> None:
        """置信度接近 -> LLM 融合，两条原始记忆都入档案。"""
        manager = make_manager(llm_client=SmartFakeLLM(extract_items=EXTRACT_ONE))
        manager.add_fact(make_fact("用户偏好使用 pytest 编写单元测试", confidence=0.85))
        report = manager.ingest("近似重复陈述", now=T_NOW)
        assert report.actions[0]["action"] == "fused"
        assert manager.size == 1
        results = manager.recall("pytest", top_k=1)
        assert results[0].fact.content == FUSED_TEXT
        assert manager.archive_size == 2

    def test_no_llm_raises(self) -> None:
        """LLM 未配置 -> ingest 明确报错。"""
        manager = make_manager(llm_client=None)
        with pytest.raises(LLMNotConfiguredError):
            manager.ingest("对话", now=T_NOW)

    def test_empty_dialogue_propagates_error(self) -> None:
        """空对话 -> extractor 的 ValueError 透传。"""
        manager = make_manager(llm_client=SmartFakeLLM())
        with pytest.raises(ValueError):
            manager.ingest("   ", now=T_NOW)


class TestCompressionTrigger:
    """自动压缩触发测试。"""

    def test_auto_compress_when_over_threshold(self) -> None:
        """库总量超门槛后再次 ingest -> 触发全库冗余压缩。

        场景：预置 6 条同文冗余（conf 0.8），ingest 抽出同文 conf 0.95
        （差 0.15 -> replaced，库内 6 条同文）+ 1 条无关 Ubuntu。
        压缩把 6 条同文合并为 1 条：compressed == 6，最终 2 条。
        """
        manager = make_manager(
            llm_client=SmartFakeLLM(extract_items=EXTRACT_REPLACE, fused_text=DUP_CONTENT)
        )
        for _ in range(6):
            manager.add_fact(make_fact(DUP_CONTENT, confidence=0.8))
        assert manager.size == 6

        report = manager.ingest("无关新对话", now=T_NOW)
        assert report.actions[0]["action"] == "replaced"
        # 6 条同文参与聚类合并为 1 条（替换后仍是 6 条同文），Ubuntu 保留
        assert report.compressed == 5
        assert manager.size == 2  # 1 条压缩记忆 + 1 条 Ubuntu

    def test_no_compress_below_threshold(self) -> None:
        """库总量不超门槛 -> 不触发压缩，compressed=0。"""
        manager = make_manager(llm_client=SmartFakeLLM())
        report = manager.ingest("老板说他偏好 pytest", now=T_NOW)
        assert report.compressed == 0

    def test_compress_skipped_when_no_redundancy(self) -> None:
        """超门槛但全是互不相关记忆 -> 无冗余簇，compressed=0。"""
        manager = make_manager(llm_client=SmartFakeLLM(extract_items=EXTRACT_TWO))
        for i in range(5):
            manager.add_fact(make_fact(f"完全不同的记忆主题编号{i}"))
        report = manager.ingest("无关新对话", now=T_NOW)
        assert report.compressed == 0
        assert manager.size == 7  # 5 条预置 + 2 条新抽取，均互不相关


class TestRecallPipeline:
    """读取 pipeline 测试。"""

    def test_recall_returns_relevant_first(self) -> None:
        """召回结果首位应是与查询最相关的记忆。"""
        manager = make_manager(llm_client=SmartFakeLLM(extract_items=EXTRACT_TWO))
        manager.ingest("包含两条事实的对话", now=T_NOW)
        results = manager.recall("单元测试工具", top_k=2)
        assert results[0].fact.content == "用户偏好使用 pytest 编写单元测试"

    def test_recall_empty_store(self) -> None:
        """空库召回返回空列表。"""
        manager = make_manager(llm_client=SmartFakeLLM())
        assert manager.recall("任意查询", top_k=5) == []

    def test_recall_no_llm_involved(self) -> None:
        """recall 是纯检索路径，即使 LLM 未配置也应正常工作。"""
        manager = make_manager(llm_client=SmartFakeLLM())
        manager.add_fact(make_fact("部署环境是 Ubuntu 服务器"))
        assert len(manager.recall("部署环境", top_k=5)) == 1


class TestStateAccessors:
    """状态统计测试。"""

    def test_size_and_archive_counters(self) -> None:
        manager = make_manager(llm_client=SmartFakeLLM(extract_items=EXTRACT_ONE))
        assert manager.size == 0
        assert manager.archive_size == 0
        # 预置低置信旧记忆（内容必须与新抽取事实同文，才能构成冲突场景）
        manager.add_fact(make_fact(DUP_CONTENT, confidence=0.4))
        manager.ingest("触发覆盖", now=T_NOW)
        assert manager.size == 1
        assert manager.archive_size == 1
