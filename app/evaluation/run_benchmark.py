"""端到端基准 Runner：真实运行 A/B 消融，产出可写进简历的指标。

对照设计（离线可复现，ground truth 全部代码推导）：

    A 组（基础 Agent）：
        - 冲突消解 = 朴素覆盖（新记忆无条件替换旧记忆）
        - 重试策略 = 固定 max_retries（不启用 SPRT）
    B 组（统计优化 Agent）：
        - 冲突消解 = 时间衰减有效置信度 + 三路裁决
        - 重试策略 = SPRT 序贯止损（连续失败推过 Wald 上界即停）

任务集（golden_set 生成，两类）：
    - conflict_weak_new  : 弱新证据不得推翻强旧偏好 -> A 必败 B 应胜；
    - conflict_strong_new: 强新证据应胜出 -> 两组基线（防"总能赢"假象）；
    - retry_hopeless     : 无望工具 -> 两组行为符合预期（正确终止），
      但 B 以显著更少的尝试次数达成（效率差异进 n_steps）；
    - retry_transient    : 瞬态故障 -> 两组都应恢复（SPRT 不伤害正常恢复）。

诚实性声明（写进报告）：检索层两组共用同一混合检索配置 —— 字符二元组
哈希嵌入下向量路与词法路高度相关，离线无法区分真实语义嵌入的增益；
该维度需接入 vLLM/真实 embedding 后评估（evaluation TODO）。

用法:
    python -m app.evaluation.run_benchmark                  # 默认 30+30
    python -m app.evaluation.run_benchmark --n-conflict 50 --n-retry 50
"""

from __future__ import annotations

import argparse
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

from app.agent.self_correction import FailureLogger, with_retry
from app.core.config import settings
from app.evaluation.golden_set import (
    BENCHMARK_NOW,
    TASK_CONFLICT_STRONG,
    TASK_CONFLICT_WEAK,
    TASK_RETRY_HOPELESS,
    TASK_RETRY_TRANSIENT,
    GoldenTask,
    generate_golden_tasks,
)
from app.evaluation.metrics import TaskResult, ab_test
from app.memory.manager import MemoryManager
from app.retrieval.hybrid_search import InMemoryVectorStore

logger = logging.getLogger(__name__)


@dataclass
class GroupConfig:
    """实验组配置。"""

    name: str                # "A" / "B"
    statistical: bool        # MemoryManager 冲突消解策略
    sprt_enabled: bool       # with_retry SPRT 止损开关
    sprt_p0: float = 0.5     # B 组 SPRT 参数（单次重试成功率基线）
    sprt_p1: float = 0.05    # B 组 SPRT 参数（退化水平）


GROUP_A = GroupConfig(name="A", statistical=False, sprt_enabled=False)
GROUP_B = GroupConfig(name="B", statistical=True, sprt_enabled=True)


@contextmanager
def _sprt_override(cfg: GroupConfig) -> Iterator[None]:
    """按组配置临时覆盖 SPRT 设置（基准是单线程受控实验）。"""
    saved = (settings.sprt_retry_enabled, settings.sprt_retry_p0, settings.sprt_retry_p1)
    settings.sprt_retry_enabled = cfg.sprt_enabled
    settings.sprt_retry_p0 = cfg.sprt_p0
    settings.sprt_retry_p1 = cfg.sprt_p1
    try:
        yield
    finally:
        (settings.sprt_retry_enabled, settings.sprt_retry_p0, settings.sprt_retry_p1) = saved


# ---------------- conflict 任务执行 ----------------


def _run_conflict_task(task: GoldenTask, cfg: GroupConfig) -> TaskResult:
    """预置记忆 -> 新记忆触发消解 -> 检索验证 Top-1 是否为期望记忆。"""
    store = InMemoryVectorStore()
    manager = MemoryManager(vector_store=store, statistical=cfg.statistical)
    for fact in task.setup_facts:
        manager.add_fact(fact)
    manager.add_fact_resolved(task.incoming_fact, now=BENCHMARK_NOW)

    hits = manager.recall(task.query, top_k=2)
    retrieved = [hit.fact.content for hit in hits]
    success = bool(retrieved) and retrieved[0] == task.expected_content

    return TaskResult(
        task_id=task.task_id,
        relevant_ids=[task.expected_content],
        retrieved_ids=retrieved,
        success=success,
        n_steps=1,
        tokens=0,
    )


# ---------------- retry 任务执行 ----------------


def _make_tool(mode: str, counter: dict):
    """按模式构造工具：hopeless 每次必败；transient 首败后恢复。"""

    def tool(tool_input: str) -> str:
        counter["calls"] += 1
        if mode == "hopeless":
            raise RuntimeError(f"dependency missing (call #{counter['calls']})")
        if counter["calls"] == 1:
            raise ValueError("transient network glitch")
        return f"ok:{tool_input}"

    return tool


def _run_retry_task(task: GoldenTask, cfg: GroupConfig, work_dir: Path) -> TaskResult:
    """with_retry 执行工具，观测实际尝试次数（SPRT 收益的载体）。

    Note:
        重试任务没有检索 ground truth，relevant_ids 用哨兵值占位 ——
        metrics 层的记忆指标（Recall/MRR/NDCG）对重试任务无意义，
        基准报告只解读 success / n_steps 两个维度。
    """
    counter = {"calls": 0}
    tool = _make_tool(task.tool_mode, counter)
    log_path = work_dir / f"failure_log_{cfg.name}.json"
    wrapped = with_retry(
        max_retries=task.max_retries,
        task_id=task.task_id,
        replan_llm=None,   # 离线：无 Replan -> 原样重试（公平比较重试策略）
        failure_logger=FailureLogger(str(log_path)),
    )(tool)

    aborted = False
    try:
        wrapped("payload")
        success = task.expect_success
    except Exception:  # noqa: BLE001 —— hopeless 工具被正确终止（上抛）
        success = not task.expect_success
        aborted = True
    if aborted and task.tool_mode == "hopeless":
        logger.debug("任务 %s 无望工具已终止，尝试 %d 次", task.task_id, counter["calls"])

    return TaskResult(
        task_id=task.task_id,
        relevant_ids=["__no_retrieval__"],   # 哨兵：重试任务无检索基准
        retrieved_ids=[],
        success=success,
        n_steps=counter["calls"],
        tokens=0,
    )


# ---------------- 分组执行与聚合 ----------------


def run_group(tasks: List[GoldenTask], cfg: GroupConfig, work_dir: Path) -> List[TaskResult]:
    """在一组黄金任务上按组配置执行，返回 TaskResult 列表。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    results: List[TaskResult] = []

    with _sprt_override(cfg):
        for task in tasks:
            if task.task_type in (TASK_CONFLICT_WEAK, TASK_CONFLICT_STRONG):
                results.append(_run_conflict_task(task, cfg))
            elif task.task_type in (TASK_RETRY_HOPELESS, TASK_RETRY_TRANSIENT):
                result = _run_retry_task(task, cfg, work_dir)
                results.append(result)
            else:
                raise ValueError(f"未知任务类型: {task.task_type}")
    return results


def run_benchmark(
    n_conflict: int = 30,
    n_retry: int = 30,
    seed: int = 42,
    output_dir: str = "evaluation/benchmark",
) -> dict:
    """完整基准：生成黄金任务 -> A/B 两组分层执行 -> Welch t 检验 + 报告 + 图表。

    分层检验（而非混在一张表）：
        - conflict 层：记忆指标（Recall/MRR/NDCG）与成功率有意义；
        - retry 层：只解读成功率与平均步数（记忆指标为哨兵占位）。

    Returns:
        摘要 dict（任务数、两份 ab_test 报告、产物路径）。
    """
    tasks = generate_golden_tasks(n_conflict=n_conflict, n_retry=n_retry, seed=seed)
    out_dir = Path(output_dir)
    work_dir = out_dir / "work"

    conflict_tasks = [t for t in tasks if t.task_type in (TASK_CONFLICT_WEAK, TASK_CONFLICT_STRONG)]
    retry_tasks = [t for t in tasks if t.task_type in (TASK_RETRY_HOPELESS, TASK_RETRY_TRANSIENT)]

    a_conflict = run_group(conflict_tasks, GROUP_A, work_dir)
    b_conflict = run_group(conflict_tasks, GROUP_B, work_dir)
    a_retry = run_group(retry_tasks, GROUP_A, work_dir)
    b_retry = run_group(retry_tasks, GROUP_B, work_dir)

    report_conflict = ab_test(a_conflict, b_conflict, output_dir=str(out_dir / "conflict"))
    report_retry = ab_test(a_retry, b_retry, output_dir=str(out_dir / "retry"))

    summary = {
        "n_tasks": len(tasks),
        "report_conflict": report_conflict,
        "report_retry": report_retry,
        "output_dir": str(out_dir),
    }
    logger.info(
        "基准完成: %d 任务（冲突 %d + 重试 %d），产物见 %s",
        len(tasks), len(conflict_tasks), len(retry_tasks), out_dir,
    )
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="统计优化 A/B 端到端基准")
    parser.add_argument("--n-conflict", type=int, default=30)
    parser.add_argument("--n-retry", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="evaluation/benchmark")
    args = parser.parse_args()

    run_benchmark(
        n_conflict=args.n_conflict,
        n_retry=args.n_retry,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
