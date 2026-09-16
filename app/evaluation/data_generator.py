"""评估数据集生成器：多轮对话测试场景（阶段五评估的数据基础）。

生成 100 个（可配）多轮对话场景，覆盖三大被测能力：
1. preference_shift —— 用户偏好改变（测试冲突消解：时间衰减 + 融合/覆盖决策）；
2. long_horizon    —— 复杂多步任务（测试长程规划：DAG 分解 + 顺序执行）；
3. tool_failure    —— 工具调用失败（测试自纠错：with_retry + Replan）。

统计设计（与 dataset_simulator 骨架一脉相承的「已知 ground truth」原则）：
- 场景骨架（类型、注入参数）由代码确定性生成（random.Random(seed)，可复现）；
- LLM 只负责把骨架扩写成自然对话（生成内容，不定义答案）；
- expected_outcome 由代码从骨架参数推导 —— 评估集的 ground truth
  必须可控，否则「同源 LLM 自己出题自己判」会污染整个评估；
- LLM 不可用或单条失败时以确定性模板兜底，场景数量恒定、CI 离线可跑。

输出 JSON（每条记录）：
    scenario_id      场景唯一标识（类型前缀 + 序号）
    scenario_type    场景类型（preference_shift / long_horizon / tool_failure）
    dialogue_history 多轮对话文本（"用户: ...\\n助手: ..."，对接 extractor 接口）
    expected_outcome 结构化 dict：类型 + 描述 + 类型特定 ground truth

CLI：
    python -m app.evaluation.data_generator --n 100 --seed 42 \
        --output data/eval/scenarios.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.llm_client import get_llm_client

logger = logging.getLogger(__name__)


# ============================ 类型定义 ============================


class ScenarioType(str, Enum):
    """场景类型（与被测能力一一对应）。"""

    PREFERENCE_SHIFT = "preference_shift"
    LONG_HORIZON = "long_horizon"
    TOOL_FAILURE = "tool_failure"


# 场景 ID 前缀（文件里一眼可辨类型）
_ID_PREFIX = {
    ScenarioType.PREFERENCE_SHIFT: "PREF",
    ScenarioType.LONG_HORIZON: "LONG",
    ScenarioType.TOOL_FAILURE: "TOOL",
}

# 三类场景的默认配比：冲突消解最难调，样本最多；余数归工具失败
DEFAULT_RATIO = {
    ScenarioType.PREFERENCE_SHIFT: 0.4,
    ScenarioType.LONG_HORIZON: 0.3,
    ScenarioType.TOOL_FAILURE: 0.3,
}

# 单条对话的最短有效长度（低于此视为 LLM 输出劣化，走兜底）
_MIN_DIALOGUE_LEN = 20


@dataclass
class ScenarioSpec:
    """场景骨架：代码生成、参数确定（LLM 扩写前的「题干」）。"""

    scenario_id: str
    scenario_type: ScenarioType
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetReport:
    """生成报告：来源构成透明可审计。"""

    total: int
    llm_count: int
    fallback_count: int
    output_path: str


# ============================ 参数池 ============================

# 偏好主题池：每个主题一对（旧偏好 -> 新偏好）
_PREFERENCE_PAIRS = [
    ("pytest 写单元测试", "unittest 写单元测试"),
    ("VS Code 编辑器", "PyCharm 编辑器"),
    ("邮件沟通", "即时消息沟通"),
    ("黑咖啡", "拿铁"),
    ("每日站会同步进度", "周报文档同步进度"),
    ("Jira 管理任务", "看板便签管理任务"),
    ("浅色主题", "深色主题"),
    ("Windows 办公", "macOS 办公"),
    ("Tab 缩进", "空格缩进"),
    ("正餐吃米饭", "正餐吃面食"),
]

# 偏好转变理由池
_SHIFT_REASONS = [
    "换了新团队，团队统一用另一套",
    "旧的用久了想换换环境",
    "发现新的效率更高",
    "项目要求必须换",
    "朋友强烈推荐试了之后觉得更好",
]

# 长程任务池：（任务描述, 典型子任务领域词）
_LONG_TASKS = [
    ("搭建一条从数据清洗到可视化的数据分析流水线", "数据"),
    ("策划一场 30 人规模的技术分享会", "筹备"),
    ("制定并执行一个三个月的健身增肌计划", "训练"),
    ("把一个 Flask 老项目迁移到 FastAPI", "迁移"),
    ("组织一次团队外的客户需求调研", "调研"),
    ("准备研究生复试的完整复习方案", "复习"),
    ("为新宿舍挑选并配置一套性价比电脑设备", "采购"),
]

# 工具失败场景池：（工具名, 错误类型, 用户意图描述）
_TOOLS = [
    ("search_web", "TimeoutError", "查证一个统计概念的定义"),
    ("query_database", "ConnectionError", "查询上月的项目进度数据"),
    ("send_email", "PermissionError", "给导师发送周报邮件"),
    ("run_sql", "TimeoutError", "跑一个聚合分析查询"),
    ("deploy_service", "ConnectionError", "发布一个演示服务"),
    ("file_read", "PermissionError", "读取一份受限的报告文件"),
]


# ============================ 骨架生成（确定性） ============================


def generate_specs(n: int = 100, seed: int = 42) -> List[ScenarioSpec]:
    """确定性生成 n 个场景骨架（同 seed 恒复现）。

    Args:
        n: 场景总数。
        seed: 随机种子（参数池抽样、步骤数、洗牌顺序全部由它驱动）。

    Returns:
        场景骨架列表（三类混合排列，id 按类型独立编号）。
    """
    if n < 1:
        raise ValueError(f"n 至少为 1，当前: {n}")

    rng = random.Random(seed)

    # 按配比分配数量，余数归 tool_failure，保证恰好 n 条
    n_pref = round(n * DEFAULT_RATIO[ScenarioType.PREFERENCE_SHIFT])
    n_long = round(n * DEFAULT_RATIO[ScenarioType.LONG_HORIZON])
    n_tool = n - n_pref - n_long

    specs: List[ScenarioSpec] = []

    for _ in range(n_pref):
        old, new = rng.choice(_PREFERENCE_PAIRS)
        specs.append(ScenarioSpec(
            scenario_id="",  # 洗牌后统一编号
            scenario_type=ScenarioType.PREFERENCE_SHIFT,
            params={
                "old_preference": old,
                "new_preference": new,
                "reason": rng.choice(_SHIFT_REASONS),
                "turns": rng.randint(4, 6),
            },
        ))

    for _ in range(n_long):
        task, domain = rng.choice(_LONG_TASKS)
        specs.append(ScenarioSpec(
            scenario_id="",
            scenario_type=ScenarioType.LONG_HORIZON,
            params={
                "task": task,
                "domain": domain,
                "min_subtasks": rng.randint(3, 5),
                "turns": rng.randint(4, 6),
            },
        ))

    for _ in range(n_tool):
        tool, error, intent = rng.choice(_TOOLS)
        specs.append(ScenarioSpec(
            scenario_id="",
            scenario_type=ScenarioType.TOOL_FAILURE,
            params={
                "tool": tool,
                "error": error,
                "intent": intent,
                "failures": rng.randint(1, 3),
                "turns": rng.randint(3, 5),
            },
        ))

    # 洗牌混排后按类型独立编号（同 seed 下编号恒定）
    rng.shuffle(specs)
    counters: Dict[ScenarioType, int] = {}
    for spec in specs:
        counters[spec.scenario_type] = counters.get(spec.scenario_type, 0) + 1
        spec.scenario_id = f"{_ID_PREFIX[spec.scenario_type]}-{counters[spec.scenario_type]:04d}"

    return specs


# ============================ ground truth 构造（纯代码） ============================


def build_expected_outcome(spec: ScenarioSpec) -> Dict[str, Any]:
    """由骨架参数推导标准答案（不经过 LLM，ground truth 可控）。

    Args:
        spec: 场景骨架。

    Returns:
        结构化 expected_outcome 字典。

    Raises:
        ValueError: 未知场景类型（编程防御）。
    """
    p = spec.params

    if spec.scenario_type is ScenarioType.PREFERENCE_SHIFT:
        return {
            "type": "preference_shift",
            "description": (
                "记忆系统应检测到新旧偏好冲突，并依据时间衰减有效置信度"
                "采纳新偏好（覆盖或融合），而非保留过时旧偏好"
            ),
            "old_preference": p["old_preference"],
            "new_preference": p["new_preference"],
            "expected_action": "replaced or fused",
        }

    if spec.scenario_type is ScenarioType.LONG_HORIZON:
        return {
            "type": "long_horizon",
            "description": (
                "规划器应将任务分解为带依赖关系的子任务 DAG，"
                "并按拓扑顺序逐个执行、逐个验证"
            ),
            "task": p["task"],
            "expected_subtasks_min": p["min_subtasks"],
        }

    if spec.scenario_type is ScenarioType.TOOL_FAILURE:
        return {
            "type": "tool_failure",
            "description": (
                "执行器应在工具失败后捕获异常并触发自纠错"
                "（with_retry + Replan），最终完成任务或明确报告失败原因"
            ),
            "tool": p["tool"],
            "injected_error": p["error"],
            "expected_retries_min": p["failures"],
        }

    raise ValueError(f"未知场景类型: {spec.scenario_type}")


# ============================ 对话生成（LLM 扩写 + 模板兜底） ============================

_GENERATION_SYSTEM_PROMPT = (
    "你是对话数据生成器。根据给定的场景要求，生成一段真实自然的多轮中文对话。"
    "只输出对话本身，格式为每行「用户: ...」或「助手: ...」，"
    "不得包含任何场景说明、编号、标题或 Markdown 标记。"
)


def _build_generation_prompt(spec: ScenarioSpec) -> str:
    """按场景类型拼装 LLM 扩写指令（含骨架参数）。"""
    p = spec.params
    if spec.scenario_type is ScenarioType.PREFERENCE_SHIFT:
        return (
            f"生成一段 {p['turns']} 轮左右的对话。情节：用户在对话前半段明确表达"
            f"自己习惯「{p['old_preference']}」，助手确认记录；对话后半段用户因"
            f"「{p['reason']}」明确表示改用「{p['new_preference']}」，语气自然坚定。"
        )
    if spec.scenario_type is ScenarioType.LONG_HORIZON:
        return (
            f"生成一段 {p['turns']} 轮左右的对话。情节：用户提出一个复杂任务"
            f"「{p['task']}」，助手确认任务并承诺分步骤推进；用户补充了至少一个"
            f"约束条件。任务应明显需要至少 {p['min_subtasks']} 个步骤才能完成。"
        )
    # TOOL_FAILURE
    return (
        f"生成一段 {p['turns']} 轮左右的对话。情节：用户要求助手{p['intent']}，"
        f"助手调用工具 {p['tool']} 时遇到了 {p['error']} 问题并如实告知，"
        f"该情况在对话中出现了 {p['failures']} 次，用户随后追问或调整了要求。"
    )


def _fallback_dialogue(spec: ScenarioSpec) -> str:
    """确定性模板兜底对话（LLM 不可用时保证数据集数量恒定）。

    模板平淡但骨架参数完整 —— 评估管道（抽取/消解/规划/自纠错）
    所需的信号全部保留，仅牺牲语言自然度。
    """
    p = spec.params
    if spec.scenario_type is ScenarioType.PREFERENCE_SHIFT:
        return (
            f"用户: 我平时习惯{p['old_preference']}。\n"
            f"助手: 好的，已记录你的偏好：{p['old_preference']}。\n"
            f"用户: 前半段先聊这些，顺便说下我最近{p['reason']}，"
            f"现在改用{p['new_preference']}了。\n"
            f"助手: 明白，以后以{p['new_preference']}为准。\n"
            "用户: 对，就按这个来。"
        )
    if spec.scenario_type is ScenarioType.LONG_HORIZON:
        return (
            f"用户: 我想{p['task']}，帮我规划一下。\n"
            f"助手: 这个任务需要分至少 {p['min_subtasks']} 个步骤推进，"
            f"我会按依赖顺序逐个执行并验证。\n"
            f"用户: 补充一个约束：{p['domain']}相关的每一步都要留档。\n"
            "助手: 收到，每步完成后我会汇报结果再进入下一步。"
        )
    return (
        f"用户: 帮我{p['intent']}。\n"
        f"助手: 好的，我调用 {p['tool']} 处理——失败报错了：{p['error']}。\n"
        f"用户: 再试一次看看。\n"
        f"助手: 又出现了 {p['error']}，我再分析原因调整参数重试。\n"
        "用户: 好，麻烦最终给我一个明确结果。"
    )


def _generate_one(
    spec: ScenarioSpec,
    llm_client: Optional[Any],
    allow_fallback: bool,
) -> tuple[str, bool]:
    """生成单条对话并报告来源：返回 (对话文本, 是否使用了模板兜底)。"""
    if llm_client is None:
        return _fallback_dialogue(spec), True

    prompt = _build_generation_prompt(spec)
    try:
        from app.agent.planner import _build_messages  # noqa: PLC0415

        response = llm_client.invoke(_build_messages(_GENERATION_SYSTEM_PROMPT, prompt))
        raw = getattr(response, "content", response)
        text = raw.strip() if isinstance(raw, str) else ""
    except Exception as exc:  # noqa: BLE001 —— 单条失败不拖垮整批
        logger.warning("场景 %s LLM 生成失败: %s", spec.scenario_id, exc)
        text = ""

    if len(text) >= _MIN_DIALOGUE_LEN and "用户:" in text:
        return text, False

    logger.warning("场景 %s LLM 输出劣化（过短或缺对话格式），启用兜底", spec.scenario_id)
    if not allow_fallback:
        raise RuntimeError(f"场景 {spec.scenario_id} LLM 生成失败且未启用兜底")
    return _fallback_dialogue(spec), True


def generate_dialogue(
    spec: ScenarioSpec,
    llm_client: Optional[Any] = None,
    allow_fallback: bool = True,
) -> str:
    """为单个场景生成对话文本。

    Args:
        spec: 场景骨架。
        llm_client: 可注入 LLM；None 时直接走模板兜底。
        allow_fallback: LLM 失败时是否允许模板兜底；False 则抛异常
            （严格模式，用于评估「纯 LLM 数据集」的生成质量）。

    Returns:
        多轮对话文本。

    Raises:
        RuntimeError: allow_fallback=False 且 LLM 生成失败。
    """
    return _generate_one(spec, llm_client, allow_fallback)[0]


# ============================ 数据集组装 ============================


def generate_dataset(
    n: int = 100,
    seed: int = 42,
    llm_client: Optional[Any] = None,
    output_path: str = "data/eval/scenarios.json",
    allow_fallback: bool = True,
) -> DatasetReport:
    """生成完整评估数据集并写入 JSON 文件。

    Args:
        n: 场景总数（默认 100）。
        seed: 随机种子。
        llm_client: 可注入 LLM；None 时全量模板兜底（离线模式）。
        output_path: 输出 JSON 路径（目录自动创建）。
        allow_fallback: 单条 LLM 失败时是否模板兜底。

    Returns:
        DatasetReport（总数 / LLM 条数 / 兜底条数 / 输出路径）。
    """
    specs = generate_specs(n=n, seed=seed)
    records: List[Dict[str, Any]] = []
    llm_count = 0

    for spec in specs:
        dialogue, fallback_used = _generate_one(spec, llm_client, allow_fallback)
        if not fallback_used:
            llm_count += 1
        records.append({
            "scenario_id": spec.scenario_id,
            "scenario_type": spec.scenario_type.value,
            "dialogue_history": dialogue,
            "expected_outcome": build_expected_outcome(spec),
        })

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = DatasetReport(
        total=len(records),
        llm_count=llm_count,
        fallback_count=len(records) - llm_count,
        output_path=str(out),
    )
    logger.info(
        "数据集生成完成: 共 %d 条（LLM %d / 兜底 %d）-> %s",
        report.total, report.llm_count, report.fallback_count, out,
    )
    return report


def main() -> None:
    """CLI 入口：python -m app.evaluation.data_generator [选项]。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="生成多轮对话评估数据集")
    parser.add_argument("--n", type=int, default=100, help="场景总数（默认 100）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument(
        "--output", type=str, default="data/eval/scenarios.json",
        help="输出 JSON 路径（默认 data/eval/scenarios.json）",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="离线模式：全部使用确定性模板，不调用 LLM",
    )
    args = parser.parse_args()

    client = None if args.offline else get_llm_client()
    report = generate_dataset(
        n=args.n,
        seed=args.seed,
        llm_client=client,
        output_path=args.output,
    )
    print(
        f"生成完成: {report.total} 条"
        f"（LLM {report.llm_count} / 模板兜底 {report.fallback_count}）"
        f" -> {report.output_path}"
    )


if __name__ == "__main__":
    main()
