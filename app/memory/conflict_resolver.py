"""记忆冲突消解模块。

核心公式（用户提供）：
    有效置信度 = 初始置信度 * exp(-lambda * (当前时间 - 记忆时间))

即指数时间衰减模型：记忆的可信度随年龄呈指数衰减，lambda 为日衰减系数。
（lambda=0.01 意味着约 69 天后有效置信度衰减到一半，ln(2)/0.01 ~ 69.3，
即该参数下记忆的"半衰期"约为 69 天。）

消解策略（三路决策）：
    1. 新记忆有效置信度显著更高 -> 覆盖旧记忆（旧记忆降级入历史）；
    2. 旧记忆显著更高 -> 保留旧记忆，新记忆作为补充信息入历史；
    3. 两者接近（差值 < 0.1）-> 交给 LLM 融合为一条新记忆。

显式决策（偏离字面需求的点，已标注）：
- 「覆盖」时旧记忆同样进入历史 —— 信息不灭原则：覆盖是有损操作，
  历史留档让上层（或阶段五的消融实验）能回溯覆盖是否正确；
- 融合记忆的 confidence 取两条**初始置信度的均值**（而非衰减后的有效值）：
  融合是一次新的信息综合，发生在当下，不应继承旧记忆的年龄惩罚。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional

from app.core.llm_client import get_llm_client
from app.memory.extractor import LLMNotConfiguredError  # noqa: F401 —— 统一异常类型（原为本地重复定义）
from app.memory.models import MemoryFact

logger = logging.getLogger(__name__)


# 日衰减系数默认值：记忆半衰期 ln(2)/0.01 ~ 69 天
DEFAULT_LAMBDA: float = 0.01
# 有效置信度差值低于此阈值 -> 触发 LLM 融合
FUSE_THRESHOLD: float = 0.1

FUSE_PROMPT_TEMPLATE = (
    "以下是关于同一主题的两条相互冲突的记忆：\n"
    "记忆A（较早）：{old_content}\n"
    "记忆B（较新）：{new_content}\n\n"
    "请将它们融合为一条新记忆，要求不丢失任一条中的信息并化解冲突，"
    "只输出融合后的记忆文本本身，不要任何解释或前缀。"
)


@dataclass
class ConflictResolution:
    """冲突消解结果。

    Attributes:
        action: 消解动作 —— "replaced"（新覆盖旧）/ "kept_with_history"
            （旧保留，新入历史）/ "fused"（LLM 融合为新记忆）。
        primary: 消解后的主记忆（调用方以此替换存储中的旧记忆）。
        history: 被降级为补充信息的记忆列表（调用方追加到历史库）。
    """

    action: str
    primary: MemoryFact
    history: List[MemoryFact] = field(default_factory=list)


def _ensure_utc(value: datetime) -> datetime:
    """统一时区为 UTC：naive 按 UTC 补齐，aware 转换到 UTC。

    所有时间差计算都在 UTC 基准上进行，避免混用时区导致
    衰减时间出现正负 8 小时的系统性偏差（北京时间场景下的典型坑）。
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def effective_confidence(
    memory: MemoryFact,
    lambda_val: float = DEFAULT_LAMBDA,
    now: Optional[datetime] = None,
) -> float:
    """计算记忆的当前有效置信度（指数时间衰减）。

    eff = confidence * exp(-lambda * delta_days)，delta_days < 0（未来时间）
    按 0 处理 —— 未来记忆不享受"负衰减"的置信度增益。

    Args:
        memory: 目标记忆。
        lambda_val: 日衰减系数，必须 >= 0。
        now: 计算基准时间，默认当前 UTC 时间（测试可注入固定时钟）。

    Returns:
        有效置信度，范围 [0, 1]（confidence 已被模型约束在 [0, 1]，
        exp(-x) in (0, 1]，乘积不会超出）。

    Raises:
        TypeError: memory 不是 MemoryFact。
        ValueError: lambda_val 为负。
    """
    if not isinstance(memory, MemoryFact):
        raise TypeError(f"memory 必须是 MemoryFact，收到: {type(memory).__name__}")
    if lambda_val < 0:
        raise ValueError(f"lambda_val 不能为负，收到: {lambda_val}")

    reference = _ensure_utc(now) if now is not None else datetime.now(timezone.utc)
    elapsed = (_ensure_utc(reference) - _ensure_utc(memory.timestamp)).total_seconds()
    delta_days = max(0.0, elapsed / 86400.0)

    return memory.confidence * math.exp(-lambda_val * delta_days)


def _build_fuse_messages(old_memory: MemoryFact, new_memory: MemoryFact) -> List[Any]:
    """构造融合 Prompt（与 extractor/compressor 相同的双路径策略）。"""
    human_text = FUSE_PROMPT_TEMPLATE.replace(
        "{old_content}", old_memory.content
    ).replace("{new_content}", new_memory.content)
    try:
        from langchain_core.messages import HumanMessage  # noqa: PLC0415

        return [HumanMessage(content=human_text)]
    except ImportError:
        return [{"role": "user", "content": human_text}]


def _fuse_memories(
    old_memory: MemoryFact,
    new_memory: MemoryFact,
    eff_old: float,
    eff_new: float,
    llm_client: Optional[Any],
    now: Optional[datetime],
) -> ConflictResolution:
    """调用 LLM 融合两条接近的记忆。

    融合记忆属性（统计决策）：
        - confidence = 两条初始置信度的均值（保守估计，不继承年龄惩罚）；
        - timestamp = 融合发生时间（融合是新信息事件）。

    LLM 不可用/返回无效时降级：保留有效置信度较高一方的原内容。
    """
    client = llm_client if llm_client is not None else get_llm_client()
    if client is None:
        raise LLMNotConfiguredError("LLM 客户端未配置（SMA_LLM_API_KEY 未设置），无法融合记忆")

    response = client.invoke(_build_fuse_messages(old_memory, new_memory))
    raw_text = getattr(response, "content", response)

    fused_time = _ensure_utc(now) if now is not None else datetime.now(timezone.utc)
    fused_confidence = (old_memory.confidence + new_memory.confidence) / 2.0

    if isinstance(raw_text, str) and raw_text.strip():
        primary = MemoryFact(
            content=raw_text.strip(),
            confidence=fused_confidence,
            timestamp=fused_time,
        )
        return ConflictResolution("fused", primary, [old_memory.model_copy(), new_memory.model_copy()])

    # 降级路径：LLM 输出无效 -> 保留有效置信度较高的原记忆
    logger.warning("LLM 融合输出无效，降级为保留有效置信度较高的一方")
    if eff_new >= eff_old:
        return ConflictResolution(
            "fused", new_memory.model_copy(), [old_memory.model_copy()]
        )
    return ConflictResolution(
        "fused", old_memory.model_copy(), [new_memory.model_copy()]
    )


def resolve_conflict(
    old_memory: MemoryFact,
    new_memory: MemoryFact,
    lambda_val: float = DEFAULT_LAMBDA,
    llm_client: Optional[Any] = None,
    now: Optional[datetime] = None,
) -> ConflictResolution:
    """消解两条同主题冲突记忆。

    决策逻辑（有效置信度差值 d = |eff_old - eff_new|）：
        - d < 0.1        -> LLM 融合（两条证据强度接近，无法程序裁决）；
        - eff_new > eff_old -> 新记忆覆盖旧记忆，旧记忆入历史；
        - 否则           -> 保留旧记忆，新记忆作为补充信息入历史。

    Args:
        old_memory: 存量记忆。
        new_memory: 新到达的记忆。
        lambda_val: 日衰减系数，默认 0.01。
        llm_client: 可注入的 LLM 客户端；None 时用全局单例。
        now: 计算基准时间，默认当前 UTC（测试可注入固定时钟）。

    Returns:
        ConflictResolution：主记忆 + 历史记忆列表。

    Raises:
        TypeError: 输入不是 MemoryFact。
        ValueError: lambda_val 为负。
        LLMNotConfiguredError: 需要融合但 LLM 不可用。
    """
    if not isinstance(old_memory, MemoryFact) or not isinstance(new_memory, MemoryFact):
        raise TypeError("old_memory 与 new_memory 都必须是 MemoryFact")
    if lambda_val < 0:
        raise ValueError(f"lambda_val 不能为负，收到: {lambda_val}")

    eff_old = effective_confidence(old_memory, lambda_val, now)
    eff_new = effective_confidence(new_memory, lambda_val, now)
    logger.debug(
        "冲突消解: eff_old=%.4f, eff_new=%.4f, lambda=%.4f", eff_old, eff_new, lambda_val
    )

    if abs(eff_old - eff_new) < FUSE_THRESHOLD:
        return _fuse_memories(old_memory, new_memory, eff_old, eff_new, llm_client, now)

    if eff_new > eff_old:
        return ConflictResolution(
            "replaced", new_memory.model_copy(), [old_memory.model_copy()]
        )
    return ConflictResolution(
        "kept_with_history", old_memory.model_copy(), [new_memory.model_copy()]
    )


def bayesian_update(prior: float, likelihood_ratio: float) -> float:
    """单步贝叶斯更新（对数域，数值稳定）。

    使用 log-odds 形式避免小概率数值下溢：
        logit(posterior) = logit(prior) + log(likelihood_ratio)

    Args:
        prior: 先验置信度，(0, 1) 开区间。
        likelihood_ratio: 似然比 P(E|H) / P(E|~H)，> 1 支持假设，< 1 反对。

    Returns:
        更新后的后验置信度，(0, 1) 开区间。

    Raises:
        ValueError: prior 越界或 likelihood_ratio 非正。

    Note:
        与 resolve_conflict 的分工：本函数处理「同向证据的信念强化」，
        resolve_conflict 处理「互斥证据的裁决」。阶段二收尾时二者
        将被统一编排进 ConflictResolver 门面。
    """
    if not 0.0 < prior < 1.0:
        raise ValueError(f"prior 必须在开区间 (0, 1)，当前值: {prior}")
    if likelihood_ratio <= 0:
        raise ValueError(f"likelihood_ratio 必须为正数，当前值: {likelihood_ratio}")

    logit_prior = math.log(prior / (1.0 - prior))
    logit_posterior = logit_prior + math.log(likelihood_ratio)
    odds = math.exp(logit_posterior)
    return odds / (1.0 + odds)
