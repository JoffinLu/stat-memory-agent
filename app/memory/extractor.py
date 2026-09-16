"""记忆抽取模块：从对话历史中提取结构化事实（MemoryFact）。

流程：
    dialogue_history --(Prompt Engineering)--> LLM --(容错 JSON 解析)--> List[MemoryFact]

设计原则：
1. 双层 Prompt：system 层做 JSON 输出硬约束，human 层为用户指定的抽取模板原文；
2. 不信任 LLM 的格式输出：解析层容错（代码围栏 / 前后缀文字）；
3. 逐条防御校验：单条畸形不拖垮整批，置信度越界截断而非丢弃；
4. 依赖注入：LLM 客户端可注入，统计逻辑全程可离线单测。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from app.core.llm_client import get_llm_client
from app.memory.models import MemoryFact

logger = logging.getLogger(__name__)


class LLMNotConfiguredError(RuntimeError):
    """LLM 客户端未配置（离线模式）时调用抽取功能。"""


class FactExtractionError(RuntimeError):
    """LLM 返回内容无法解析出 JSON 数组。"""


# 用户指定的抽取模板（原文保留，{dialogue_history} 用 replace 填充，
# 避免 .format() 被对话内容中的花括号干扰）
PROMPT_TEMPLATE = (
    "从以下对话中提取用户偏好、事实和约束，以 JSON 数组返回，"
    "包含 content 和 confidence (0-1) 字段。对话内容：{dialogue_history}"
)

# system 层：强制 JSON 输出格式
SYSTEM_PROMPT = (
    "你是一个信息抽取引擎。你只输出一个 JSON 数组，"
    "不得包含任何解释性文字、前缀、后缀或 Markdown 代码块标记。"
    "数组中每个元素仅包含两个字段："
    "content（字符串，抽取的事实）、confidence（0 到 1 之间的浮点数，该事实的可信度）。"
    "若对话中没有可提取的事实，返回空数组 []。"
)

# 匹配 Markdown 代码围栏（```json ... ``` 或 ``` ... ```）
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _build_messages(dialogue_history: str) -> List[Any]:
    """构造发给 LLM 的消息序列。

    优先使用 LangChain 消息类型（生产路径）；langchain 未安装时
    降级为 OpenAI 风格 dict（离线测试路径），保证统计逻辑可脱离重依赖运行。

    Args:
        dialogue_history: 原始对话文本。

    Returns:
        [system 消息, human 消息] 列表。
    """
    human_text = PROMPT_TEMPLATE.replace("{dialogue_history}", dialogue_history)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_text),
        ]
    except ImportError:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": human_text},
        ]


def _extract_json_array(raw_text: str) -> List[Any]:
    """从 LLM 原始输出中容错地提取 JSON 数组。

    解析降级链：
        1. 直接 json.loads；
        2. 剥离 Markdown 代码围栏后解析；
        3. 截取文本中最外层的 [ ... ] 子串后解析（容忍前后缀文字）。

    Args:
        raw_text: LLM 返回的原始字符串。

    Returns:
        解析出的 list（JSON 数组）。

    Raises:
        FactExtractionError: 三层策略均失败。
    """
    text = raw_text.strip()

    # 策略 1：输出本身即合法 JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # 策略 2：剥离代码围栏
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # 策略 3：截取最外层方括号子串（容忍 "结果如下：[...] 以上" 之类的包裹）
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    raise FactExtractionError(f"无法从 LLM 输出中解析 JSON 数组，原始输出前 200 字符: {raw_text[:200]!r}")


def _clamp_confidence(value: Any) -> Optional[float]:
    """将 LLM 给出的 confidence 归位到 [0, 1]，非法则返回 None。

    统计依据：confidence 是贝叶斯更新的先验输入，[0, 1] 是其概率语义。
    LLM 输出 1.2 / -0.3 属于数值噪声 —— 截断的偏差远小于丢弃整条事实
    造成的信息损失；非数值则无法修复，返回 None 交由调用方跳过。

    Args:
        value: 任意 LLM 输出的 confidence 字段。

    Returns:
        截断后的浮点数；不可修复时为 None。
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num:  # NaN 检查（NaN != NaN）
        return None
    return min(1.0, max(0.0, num))


def _parse_fact_items(items: List[Any]) -> List[MemoryFact]:
    """将 JSON 数组元素逐条解析为 MemoryFact（畸形条目跳过并告警）。

    Args:
        items: _extract_json_array 的解析结果。

    Returns:
        合法条目构成的 MemoryFact 列表。
    """
    facts: List[MemoryFact] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            logger.warning("第 %d 条非对象，已跳过: %r", idx, item)
            continue

        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            logger.warning("第 %d 条 content 缺失或为空，已跳过: %r", idx, item)
            continue

        raw_confidence = item.get("confidence", 0.5)  # 缺失 -> 无信息先验
        confidence = _clamp_confidence(raw_confidence)
        if confidence is None:
            logger.warning("第 %d 条 confidence 非数值，改用默认 0.5: %r", idx, raw_confidence)
            confidence = 0.5

        # 只取契约字段：LLM 可能多输出 type/source 等，一律忽略
        facts.append(MemoryFact(content=content.strip(), confidence=confidence))

    return facts


def extract_facts(
    dialogue_history: str,
    llm_client: Optional[Any] = None,
) -> List[MemoryFact]:
    """从对话历史中提取结构化事实。

    Args:
        dialogue_history: 对话文本。要求非空字符串。
        llm_client: 可注入的 LLM 客户端（需实现 invoke(messages) 且返回
            带 .content 属性的消息对象）；为 None 时使用全局单例。

    Returns:
        提取出的 MemoryFact 列表（可能为空，表示对话中无事实）。

    Raises:
        ValueError: dialogue_history 不是非空字符串。
        LLMNotConfiguredError: LLM 客户端不可用（离线模式）。
        FactExtractionError: LLM 输出无法解析为 JSON 数组。
    """
    if not isinstance(dialogue_history, str) or not dialogue_history.strip():
        raise ValueError("dialogue_history 必须是非空字符串")

    client = llm_client if llm_client is not None else get_llm_client()
    if client is None:
        raise LLMNotConfiguredError(
            "LLM 客户端未配置（SMA_LLM_API_KEY 未设置），无法进行事实抽取"
        )

    messages = _build_messages(dialogue_history)
    response = client.invoke(messages)

    # 兼容 LangChain AIMMessage（.content）与裸字符串返回
    raw_text = getattr(response, "content", response)
    if not isinstance(raw_text, str):
        raise FactExtractionError(f"LLM 返回了非文本内容: {type(raw_text).__name__}")

    items = _extract_json_array(raw_text)
    facts = _parse_fact_items(items)
    logger.info("事实抽取完成: %d/%d 条有效", len(facts), len(items))
    return facts
