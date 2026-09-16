"""黄金评估集：确定性生成的被测任务（ground truth 代码推导，不经 LLM）。

三类任务覆盖两个统计优化点：
1. conflict_weak_new  —— 旧偏好(高置信、陈旧) vs 新信息(低置信、新鲜)：
   正确行为 = 保留旧偏好（新证据太弱，不足以推翻）。朴素覆盖组必败；
2. conflict_strong_new —— 旧信息(低置信、陈旧) vs 新信息(高置信、新鲜)：
   正确行为 = 采纳新信息。两组都应通过（平衡基线，防"总能赢"的假象）；
3. retry_hopeless / retry_transient —— 无望工具（每次必败）与瞬态故障
   （失败一次后恢复）：SPRT 止损组应在无望工具上提前止损，
   同时不影响瞬态故障的正常恢复。

与 data_generator 同一立场：骨架参数由 random.Random(seed) 确定性生成，
expected_* 由代码推导 —— ground truth 永远不经 LLM。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from app.memory.models import MemoryFact

# 基准时间（所有时间戳相对它构造，衰减数学完全确定）
BENCHMARK_NOW = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)

# 冲突场景的记忆年龄（天）—— 30 天在 lambda=0.01 下衰减系数 0.741
AGE_DAYS = 30

TASK_CONFLICT_WEAK = "conflict_weak_new"
TASK_CONFLICT_STRONG = "conflict_strong_new"
TASK_RETRY_HOPELESS = "retry_hopeless"
TASK_RETRY_TRANSIENT = "retry_transient"


@dataclass
class GoldenTask:
    """单个黄金任务。

    conflict 任务: setup_facts 预置 -> incoming_fact 触发消解 ->
        recall(query) 的 Top-1 应为 expected_content；
    retry 任务: 按 tool_mode 构造工具，max_retries 上限内执行，
        观测实际尝试次数（n_steps）与是否最终成功。
    """

    task_id: str
    task_type: str
    # ---- conflict 任务字段 ----
    setup_facts: List[MemoryFact] = field(default_factory=list)
    incoming_fact: Optional[MemoryFact] = None
    query: str = ""
    expected_content: str = ""
    # ---- retry 任务字段 ----
    tool_mode: str = ""           # "hopeless" / "transient"
    max_retries: int = 0
    # 期望最终成功（transient=True, hopeless=False）
    expect_success: bool = False


def _fact(content: str, confidence: float, age_days: float) -> MemoryFact:
    """构造带固定时间戳的记忆（相对 BENCHMARK_NOW）。"""
    fact = MemoryFact(content=content, confidence=confidence)
    fact.timestamp = BENCHMARK_NOW - timedelta(days=age_days)
    return fact


def _distractors(rng: random.Random, index: int) -> List[MemoryFact]:
    """与目标主题无关的干扰记忆（保证语料规模、防检索退化）。"""
    pool = [
        "团队每周五下午进行代码评审会议",
        "数据管道使用 Airflow 做任务调度",
        "公司报销系统每月十五号开放提单",
        "会议室投影仪需要提前一天预约",
    ]
    picked = rng.sample(pool, k=2)
    return [_fact(c, 0.5, age_days=rng.uniform(1, 10)) for c in picked]


def generate_golden_tasks(n_conflict: int = 10, n_retry: int = 10, seed: int = 42) -> List[GoldenTask]:
    """确定性生成黄金任务集。

    Args:
        n_conflict: 冲突消解任务数（强弱新证据各半）。
        n_retry: 重试任务数（无望/瞬态约 6:4）。
        seed: 随机种子（干扰记忆抽样）。

    Returns:
        GoldenTask 列表（洗牌混排）。
    """
    rng = random.Random(seed)
    tasks: List[GoldenTask] = []

    # ---- 冲突消解任务 ----
    weak_old = "用户偏好使用 pytest 框架编写单元测试"
    weak_new = "用户提到或许想改用 unittest"
    strong_old = "项目部署在 Windows 服务器上"
    strong_new = "项目已迁移部署到 Linux 服务器上"

    for i in range(n_conflict):
        distractors = _distractors(rng, i)
        if i % 2 == 0:  # 弱新证据：旧偏好应存活
            tasks.append(GoldenTask(
                task_id=f"GC-weak-{i:03d}",
                task_type=TASK_CONFLICT_WEAK,
                setup_facts=[_fact(weak_old, 0.9, AGE_DAYS)] + distractors,
                incoming_fact=_fact(weak_new, 0.4, 0.0),
                query="pytest 单元测试框架偏好",
                expected_content=weak_old,
            ))
        else:  # 强新证据：新信息应胜出
            tasks.append(GoldenTask(
                task_id=f"GC-strong-{i:03d}",
                task_type=TASK_CONFLICT_STRONG,
                setup_facts=[_fact(strong_old, 0.4, AGE_DAYS)] + distractors,
                incoming_fact=_fact(strong_new, 0.9, 0.0),
                query="项目部署在什么服务器上",
                expected_content=strong_new,
            ))

    # ---- 重试任务 ----
    n_hopeless = max(1, round(n_retry * 0.6))
    for i in range(n_retry):
        mode = TASK_RETRY_HOPELESS if i < n_hopeless else TASK_RETRY_TRANSIENT
        tasks.append(GoldenTask(
            task_id=f"GR-{mode.split('_')[1]}-{i:03d}",
            task_type=mode,
            tool_mode=mode.split("_")[1],       # "hopeless" / "transient"
            max_retries=8,
            expect_success=(mode == TASK_RETRY_TRANSIENT),
        ))

    rng.shuffle(tasks)
    return tasks
