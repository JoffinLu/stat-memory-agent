"""Agent 运行时：把记忆系统与执行循环焊成真正的长程智能体闭环。

此前 MemoryManager 与 AgentExecutor 是两个孤岛；本模块是双向胶水：

    run(task) 的完整闭环：

        1. recall   : 混合检索与任务相关的记忆（纯统计路径，零 LLM 成本）；
        2. inject   : 记忆上下文注入规划 / ReAct 执行 / Critic 验证三层 Prompt；
        3. execute  : AgentExecutor 走完整状态机循环；
        4. writeback: 执行经验（成功/失败 + 原因）结构化回写记忆库 ——
                      下次同类任务自动受益（自进化的"进化"发生在 Recall）；
        5. monitor  : 任务成功率进 SPC Shewhart 控制图，失控（过程退化）
                      触发记忆库自检（强制压缩清理），怀疑记忆噪声拖累执行。

设计决策（标注）：
- 经验回写跳过 LLM 抽取：ExecutionReport 本身就是结构化事实，
  再让 LLM 从结构化数据里"抽取结构化数据"纯属浪费 —— 直接构造
  MemoryFact 走 add_fact_resolved（保留冲突消解保护）；
- SPC 监控对象用任务成功率（二元）而非检索命中率：后者需要
  ground truth，线上没有；前者免费可得。二元信号的 Shewhart 图
  在零方差期（连续全成功/全失败）后对任何偏离都敏感 —— 这是有意的
  保守触发，长时间平稳后的异常值得一次自检。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, List, Optional

from app.agent.planner import AgentExecutor, ExecutionReport
from app.agent.self_correction import is_out_of_control
from app.core.config import settings
from app.memory.extractor import LLMNotConfiguredError
from app.memory.manager import MemoryManager
from app.memory.models import MemoryFact

logger = logging.getLogger(__name__)

# 经验记忆的置信度：执行结果是被验证过的结构化事实，高于无信息先验
_EXPERIENCE_CONFIDENCE_SUCCESS = 0.7
_EXPERIENCE_CONFIDENCE_FAILURE = 0.6


@dataclass
class RuntimeReport:
    """一次 runtime.run 的完整报告。"""

    execution: ExecutionReport                 # 执行循环报告（状态/子任务/转移）
    context_used: List[str] = field(default_factory=list)  # 注入的记忆内容
    experiences_recorded: int = 0              # 回写的经验记忆条数
    self_check_triggered: bool = False         # SPC 失控是否触发了记忆自检


class AgentRuntime:
    """长程智能体运行时：记忆 ⇄ 执行的双向闭环 + SPC 健康监控。

    Args:
        memory: 记忆系统门面（MemoryManager）。
        executor: ReAct 执行器（AgentExecutor）。
        llm_client: 预留（经验回写走结构化路径，暂不消费）。
        max_context_memories: 注入 Prompt 的记忆条数上限（上下文预算）。
        spc_window_size: SPC 滑动窗口大小；None 用 settings。
    """

    def __init__(
        self,
        memory: MemoryManager,
        executor: AgentExecutor,
        llm_client: Optional[Any] = None,
        max_context_memories: int = 5,
        spc_window_size: Optional[int] = None,
    ) -> None:
        if max_context_memories < 1:
            raise ValueError("max_context_memories 至少为 1")
        self._memory = memory
        self._executor = executor
        self._llm_client = llm_client
        self._max_context = max_context_memories
        window_size = spc_window_size or settings.spc_window_size
        if window_size < 3:
            raise ValueError("SPC 窗口至少为 3（标准差才能估计）")
        self._window: Deque[float] = deque(maxlen=window_size)

    # ---------- 主入口 ----------

    def run(self, task: str, now: Optional[datetime] = None) -> RuntimeReport:
        """记忆增强的任务执行闭环（召回 -> 注入 -> 执行 -> 回写 -> 监控）。

        Args:
            task: 总体任务描述。
            now: 经验记忆的时间基准（测试注入固定时钟用）。

        Returns:
            RuntimeReport：执行报告 + 记忆交互明细。

        Raises:
            ValueError: task 非法。
        """
        # 1. 召回：与任务相关的历史记忆（零 LLM 成本）
        hits = self._memory.recall(task, top_k=self._max_context)
        context_used = [hit.fact.content for hit in hits]

        # 2. 注入 + 执行
        context_text = "\n".join(f"- {content}" for content in context_used)
        execution = self._executor.run(task, memory_context=context_text)

        # 3. 经验回写（结构化直写，见模块 docstring 决策）
        recorded = self._record_experience(task, execution, now)

        # 4. SPC 健康监控
        self_check = self._monitor_spc(execution)

        return RuntimeReport(
            execution=execution,
            context_used=context_used,
            experiences_recorded=recorded,
            self_check_triggered=self_check,
        )

    # ---------- 经验回写 ----------

    def _record_experience(
        self,
        task: str,
        execution: ExecutionReport,
        now: Optional[datetime],
    ) -> int:
        """把执行经验写回记忆库，返回写入条数。

        每个子任务一条经验（保留粒度）；任务级终态并入最后一条，
        避免重复记录。失败经验的置信度略低于成功经验 —— 失败
        归因可能不完整，权重应反映这一点。
        """
        if not execution.results:
            return 0

        count = 0
        total = len(execution.results)
        for idx, result in enumerate(execution.results):
            if result.passed:
                outcome = "成功"
                confidence = _EXPERIENCE_CONFIDENCE_SUCCESS
                detail = result.answer or "（无输出）"
            else:
                outcome = "失败"
                confidence = _EXPERIENCE_CONFIDENCE_FAILURE
                detail = result.critic_reason or "（未通过验证）"

            if idx == total - 1:
                outcome += f"（任务终态: {execution.status}）"

            fact = MemoryFact(
                content=(
                    f"执行经验：子任务「{result.description}」{outcome}，"
                    f"结果：{detail}"
                ),
                confidence=confidence,
            )
            if now is not None:
                fact.timestamp = now
            try:
                self._memory.add_fact_resolved(fact, now=now)
            except LLMNotConfiguredError:
                # 融合路径需要 LLM 但未配置 -> 降级为直接入库（fail-open）。
                # 长程循环里同类任务重复执行是常态，相似经验必然触发
                # 融合分支 —— 纠错层故障不能中断执行闭环；
                # 重复条目由压缩器的冗余聚类兜底清理。
                logger.debug("融合不可用，经验直接入库: %s", fact.content[:50])
                self._memory.add_fact(fact)
            count += 1
        return count

    # ---------- SPC 健康监控 ----------

    def _monitor_spc(self, execution: ExecutionReport) -> bool:
        """任务成功率进 Shewhart 控制图；失控触发记忆库自检。

        Returns:
            是否触发了自检。
        """
        value = 1.0 if execution.status == "success" else 0.0

        if len(self._window) < self._window.maxlen:
            self._window.append(value)
            return False  # 窗口未满，控制限不可估计

        samples = list(self._window)
        self._window.append(value)

        try:
            degraded = is_out_of_control(value, samples)
        except ValueError as exc:  # 样本含非有限值等退化情形
            logger.warning("SPC 监控跳过: %s", exc)
            return False

        if not degraded:
            return False

        # 失控：执行过程异常，怀疑记忆库被噪声/陈旧记忆污染 ——
        # 先清理再观察（自检收益与代价都低：压缩是幂等的）
        logger.warning(
            "任务成功率失控（观测 %.0f，窗口均值偏离控制限），触发记忆库自检",
            value,
        )
        try:
            self._memory.maintain()
        except Exception as exc:  # noqa: BLE001 —— 自检失败不阻断主流程
            logger.error("记忆库自检失败（不影响任务结果）: %s", exc)
        return True
