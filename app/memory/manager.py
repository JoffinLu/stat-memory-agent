"""记忆系统编排门面：把抽取、消解、压缩、检索四件套串成 pipeline。

这是阶段二的收官件，也是阶段四 Agent 核心循环的接入点：
    ingest(dialogue)  写入路径：抽取 -> 冲突消解 -> 落库 -> 按需压缩
    recall(query)     读取路径：混合检索（BM25 + 向量 + RRF）

落库规则（对每条新抽取的事实）：
    1. 混合检索库内 Top-1 相关记忆；
    2. 无命中 -> 直接入库；
    3. 有命中 -> resolve_conflict 裁决，按 action 应用：
        - replaced: 移除旧记忆，写入新记忆（旧入档案）；
        - kept_with_history: 主记忆不动，新记忆入档案；
        - fused: 移除旧记忆，写入融合记忆（两条原始记忆入档案）。

简化声明（阶段五改进项）：
- 冲突检测用「检索 Top-1 命中即相关」，语义级冲突识别（NLI/LLM 蕴含
  判断）留待阶段四接入；
- 压缩触发用全库计数而非按主题分组 —— 冗余簇按 0.85 相似度聚类，
  无关记忆是 singleton 不受影响，全库压缩是安全的粗粒度近似。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.llm_client import get_llm_client
from app.memory.compressor import COMPRESSION_TRIGGER, compress_memories
from app.memory.conflict_resolver import ConflictResolution, resolve_conflict
from app.memory.extractor import LLMNotConfiguredError, extract_facts
from app.memory.models import MemoryFact
from app.retrieval.hybrid_search import HybridSearch, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    """单次 ingest 的处理报告。

    Attributes:
        extracted: 抽取出的记忆条数。
        actions: 每条记忆的处理动作明细
            （{"content": ..., "action": stored/replaced/kept_with_history/fused}）。
        compressed: 自动压缩合并掉的记忆条数（0 表示未触发压缩）。
    """

    extracted: int
    actions: List[Dict[str, str]] = field(default_factory=list)
    compressed: int = 0


class MemoryManager:
    """记忆系统门面：统一写入与读取路径。

    Args:
        llm_client: 可注入的 LLM 客户端（抽取与融合共用）；None 时
            延迟到首次使用时取全局单例。
        vector_store: 注入的向量库（如 ChromaVectorStore / 测试假库）；
            None 时仅 BM25 词法路径。
        embedding_fn: 向量库生产路径的嵌入函数。
        k_rrf: RRF 平滑常数，透传给 HybridSearch。
        statistical: 冲突消解策略开关（A/B 消融用）。
            True  -> 统计消解：时间衰减有效置信度 + 三路裁决（默认）；
            False -> 朴素基线：新记忆无条件覆盖旧记忆（对照组）。
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        embedding_fn: Optional[Any] = None,
        k_rrf: int = 60,
        statistical: bool = True,
    ) -> None:
        self._llm_client = llm_client
        self._has_vector = vector_store is not None
        self._statistical = statistical
        self._searcher = HybridSearch(
            vector_store=vector_store, k_rrf=k_rrf, embedding_fn=embedding_fn
        )
        self._archive: List[MemoryFact] = []

    # ---------- 写入路径 ----------

    def ingest(
        self,
        dialogue_history: str,
        now: Optional[datetime] = None,
    ) -> IngestReport:
        """从对话历史抽取事实并写入记忆库（完整写入 pipeline）。

        Args:
            dialogue_history: 对话文本，非空。
            now: 冲突消解的时间基准（测试注入固定时钟用）。

        Returns:
            IngestReport：抽取条数、逐条动作、压缩统计。

        Raises:
            ValueError: 对话为空。
            LLMNotConfiguredError: LLM 客户端不可用。
        """
        client = self._llm_client if self._llm_client is not None else get_llm_client()
        if client is None:
            raise LLMNotConfiguredError("LLM 客户端未配置，无法执行记忆写入 pipeline")

        facts = extract_facts(dialogue_history, llm_client=client)
        actions: List[Dict[str, str]] = []

        for fact in facts:
            action = self.add_fact_resolved(fact, now=now, client=client)
            actions.append({"content": fact.content, "action": action})

        compressed = self._compress_if_needed(client)
        logger.info(
            "ingest 完成: 抽取 %d 条, 压缩合并 %d 条", len(facts), compressed
        )
        return IngestReport(extracted=len(facts), actions=actions, compressed=compressed)

    def add_fact_resolved(
        self,
        fact: MemoryFact,
        now: Optional[datetime] = None,
        client: Optional[Any] = None,
    ) -> str:
        """写入单条记忆并做冲突消解（ingest 循环体 / runtime 经验回写共用）。

        Args:
            fact: 待写入记忆。
            now: 消解时间基准（测试注入固定时钟）。
            client: LLM 客户端（融合用）；None 时取全局单例。

        Returns:
            动作名：stored / replaced / kept_with_history / fused。

        Note:
            冲突检测仅在向量路可用时启用：BM25 没有绝对相关性标尺
            （全库同词时 idf 被 eps 托底，无关查询也会产生微弱命中），
            而"是否同主题"的判断依赖语义相似度。
        """
        hits = self._searcher.search(fact.content, top_k=1)
        if not (self._has_vector and hits):
            self._searcher.add_fact(fact)
            return "stored"

        existing = hits[0].fact
        if not self._statistical:
            # 朴素基线（对照组）：无条件覆盖 —— 有损且不做置信度裁决
            self._searcher.remove_fact(existing.id)
            self._archive.append(existing.model_copy())
            self._searcher.add_fact(fact)
            logger.info("朴素覆盖: %s -> %s", existing.id, fact.id)
            return "replaced"

        if client is None:
            client = self._llm_client if self._llm_client is not None else get_llm_client()
        resolution = resolve_conflict(existing, fact, llm_client=client, now=now)
        self._apply_resolution(resolution)
        return resolution.action

    def add_fact(self, fact: MemoryFact) -> None:
        """绕过 LLM 直接入库（程序化写入/测试用）。"""
        self._searcher.add_fact(fact)

    def _apply_resolution(self, resolution: ConflictResolution) -> None:
        """按消解结果更新主存储与档案（对三种 action 统一处理）。

        规则：历史列表中凡是已存在于主库的记忆先移除；
        主记忆若不在库中则写入。该规则对 replaced / kept_with_history /
        fused 三种情形均正确，无需按 action 分支。
        """
        current_ids = {f.id for f in self._searcher.facts}
        for history_fact in resolution.history:
            if history_fact.id in current_ids:
                self._searcher.remove_fact(history_fact.id)
            self._archive.append(history_fact.model_copy())

        if resolution.primary.id not in current_ids:
            self._searcher.add_fact(resolution.primary)
        else:
            # primary 就是库内既有记忆（kept_with_history 情形），无操作
            logger.debug("主记忆 %s 已在库中，无需写入", resolution.primary.id)

    def _compress_if_needed(self, client: Any) -> int:
        """库总量超过压缩门槛时执行全库压缩。

        Returns:
            被合并掉的记忆条数（0 = 未触发或无冗余）。
        """
        facts = self._searcher.facts
        if len(facts) <= COMPRESSION_TRIGGER:
            return 0

        compressed = compress_memories(facts, llm_client=client)
        merged = len(facts) - len(compressed)
        if merged <= 0:
            return 0

        # 全量替换主库
        for fact in facts:
            self._searcher.remove_fact(fact.id)
        self._searcher.add_facts(compressed)
        logger.info("自动压缩: %d -> %d 条", len(facts), len(compressed))
        return merged

    def maintain(self) -> int:
        """记忆库自检：无视触发门槛强制执行一次全库冗余压缩。

        由 AgentRuntime 的 SPC 失控信号调用 —— 任务成功率跌破控制限时，
        怀疑记忆库被噪声污染，主动清理冗余/陈旧记忆后观察是否恢复。
        """
        client = self._llm_client if self._llm_client is not None else get_llm_client()
        if client is None:
            logger.warning("记忆库自检跳过：LLM 客户端不可用（无法执行融合）")
            return 0
        facts = self._searcher.facts
        compressed = compress_memories(facts, llm_client=client)
        merged = len(facts) - len(compressed)
        if merged <= 0:
            return 0
        for fact in facts:
            self._searcher.remove_fact(fact.id)
        self._searcher.add_facts(compressed)
        logger.warning("记忆库自检压缩: %d -> %d 条", len(facts), len(compressed))
        return merged

    # ---------- 读取路径 ----------

    def recall(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """混合检索召回记忆（读取 pipeline，无 LLM 参与，零成本）。"""
        return self._searcher.search(query, top_k=top_k)

    # ---------- 状态查询 ----------

    @property
    def size(self) -> int:
        """主库记忆条数。"""
        return len(self._searcher)

    @property
    def archive_size(self) -> int:
        """历史档案条数（被覆盖/融合/降级的记忆）。"""
        return len(self._archive)
