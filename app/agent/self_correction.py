"""自纠错引擎：SPC/SPRT 统计工具 + LLM Replan 重试装饰器。

统计层（纯函数，可脱离 LLM 独立单测）：
1. 结果评估：SPC 控制图监控执行指标（成功率、耗时、工具报错率），
   超出 3-sigma 控制限即触发纠错流程；
2. 偏差检测：SPRT 序贯假设检验决定「继续执行 / 停下反思」，
   在 I 类错误率 alpha、II 类错误率 beta 约束下最早做出决策；
3. 失败归因：反思推理链（LLM 驱动，见下方 Replan 机制）；
4. 策略修正：修正结论沉淀到失败日志与记忆系统。

自纠错层（本文件后半部分）：
    @with_retry(max_retries=3) —— 工具函数抛异常时，将「错误信息 +
    历史执行轨迹」拼入 Prompt 调用 LLM 重新规划（Replan），按修正参数
    重试；每次失败记录到 logs/failure_log.json（任务ID、失败步骤、
    错误原因、重试次数、是否最终成功）。

设计原则：
- fail-open 于纠错层：Replan LLM 不可用 / 输出非法时退化为原样重试，
  装饰器绝不因纠错层自身故障而失效；
- fail-closed 于结果层：重试耗尽后重新抛出原异常，不吞错误；
- 依赖注入：LLM 与日志器均可注入，全链路离线可测。
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from app.core.config import settings
from app.agent.planner import (
    _build_messages,
    _extract_json_object,
    _strip_response_text,
)

logger = logging.getLogger(__name__)


def spc_control_limits(samples: List[float]):
    """计算 Shewhart 控制图的中心线与上下控制限。

    CL  = mean(samples)
    UCL = CL + sigma_width * std(samples)
    LCL = CL - sigma_width * std(samples)

    Args:
        samples: 滑动窗口内的历史指标样本（如每次任务的成功率）。

    Returns:
        (LCL, CL, UCL) 三元组。

    Raises:
        ValueError: 样本数少于 2（标准差无法估计）或包含非有限值。

    Note:
        sigma_width 默认 3.0（经典 3-sigma 控制图），
        从 Settings.spc_sigma 读取，便于调参。
    """
    if len(samples) < 2:
        raise ValueError(f"SPC 至少需要 2 个样本，当前: {len(samples)}")
    if not all(math.isfinite(s) for s in samples):
        raise ValueError("样本中包含非有限值（NaN 或 Inf）")

    n = len(samples)
    mean = sum(samples) / n
    variance = sum((s - mean) ** 2 for s in samples) / (n - 1)
    std = math.sqrt(variance)

    width = settings.spc_sigma * std
    return mean - width, mean, mean + width


def is_out_of_control(value: float, samples: List[float]) -> bool:
    """判断新观测值是否超出控制限（过程失控）。

    Args:
        value: 新观测的指标值。
        samples: 历史样本窗口（不含 value）。

    Returns:
        失控（超出 [LCL, UCL]）返回 True。

    Raises:
        ValueError: 样本不足或 value 非有限。
    """
    if not math.isfinite(value):
        raise ValueError(f"观测值非有限: {value}")
    lcl, _, ucl = spc_control_limits(samples)
    return value < lcl or value > ucl


def sprt_decision(successes: int, trials: int, p0: float, p1: float) -> str:
    """SPRT 序贯概率比检验（伯努利成功率的单侧检验）。

    H0: p = p0（过程正常）  vs  H1: p = p1（过程退化，p1 < p0）

    对数似然比：
        log LR = x * log(p1/p0) + (n - x) * log((1-p1)/(1-p0))

    决策边界（Wald 边界）：
        log(B) = log((1-beta)/alpha)   接受 H1
        log(A) = log(beta/(1-alpha))   接受 H0
        介于两者之间 -> 继续观察

    Args:
        successes: 截至目前的成功次数 x。
        trials: 截至目前的总试验次数 n。
        p0: H0 下的成功率（历史基线）。
        p1: H1 下的成功率（怀疑退化到的水平），要求 p1 < p0。
        alpha/beta: 从 Settings 读取（I / II 类错误率上限）。

    Returns:
        "accept_h1"（判定退化，触发纠错）、"accept_h0"（过程正常）
        或 "continue"（证据不足，继续执行）。

    Raises:
        ValueError: 参数不满足检验前提。

    Note:
        SPRT 的价值：平均样本量显著小于固定样本量检验（Wald, 1945），
        让智能体「用最少的执行次数」做出继续/纠错决策。
    """
    alpha = settings.sprt_alpha
    beta = settings.sprt_beta

    if not 0.0 < p1 < p0 < 1.0:
        raise ValueError(f"要求 0 < p1 < p0 < 1，当前 p0={p0}, p1={p1}")
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError(f"非法计数: successes={successes}, trials={trials}")

    log_lr = (
        successes * math.log(p1 / p0)
        + (trials - successes) * math.log((1.0 - p1) / (1.0 - p0))
    )

    upper = math.log((1.0 - beta) / alpha)   # 越过此线 -> 接受 H1
    lower = math.log(beta / (1.0 - alpha))   # 低于此线 -> 接受 H0

    if log_lr >= upper:
        return "accept_h1"
    if log_lr <= lower:
        return "accept_h0"
    return "continue"


# ============================ Replan 自纠错 ============================

# Replan system 层：JSON 裁决契约
REPLAN_SYSTEM_PROMPT = (
    "你是自纠错引擎的重新规划器（Replan）。根据工具函数的调用参数、"
    "错误信息与历史执行轨迹，分析失败原因并给出修正后的调用参数。"
    "只输出一个 JSON 对象："
    '{"analysis": "失败归因", "revised_params": {"参数名": "新值"} 或 null, '
    '"reason": "修正理由"}。'
    "若认为原参数无需修改，revised_params 输出 null。"
    "不得包含任何解释性文字或 Markdown 代码块标记。"
)

# Replan human 层：错误信息 + 历史执行轨迹（用户契约原文要求）
REPLAN_HUMAN_TEMPLATE = (
    "工具函数：{func_name}\n"
    "函数签名：{signature}\n"
    "当前调用参数：{call_args}\n"
    "最新错误信息：{error}\n"
    "历史执行轨迹：\n{history}\n"
    "请分析失败原因并输出修正方案。"
)

# 单条字段在日志/Prompt 中的最大长度（防异常消息失控膨胀）
_MAX_TEXT_LEN = 500


def _truncate(text: str, limit: int = _MAX_TEXT_LEN) -> str:
    """截断过长文本（日志与 Prompt 都不应被单条错误撑爆）。"""
    return text if len(text) <= limit else text[:limit] + "...(截断)"


class FailureLogger:
    """失败日志器：JSON 数组追加写入，线程安全 + 损坏容错。

    文件格式：顶层 JSON 数组，每元素一条失败记录。
    写入采用「临时文件 + 原子替换」，避免写一半崩溃留下残缺文件；
    读取遇损坏文件时告警并将旧文件改名备份后重置（数据可追溯优先于丢弃）。
    """

    def __init__(self, log_path: str) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record_many(self, records: List[Dict[str, Any]]) -> None:
        """批量追加失败记录（一次重试链的所有失败一次性落盘）。"""
        if not records:
            return
        with self._lock:
            existing = self._read_unlocked()
            existing.extend(records)
            tmp_path = self._path.with_suffix(".json.tmp")
            tmp_path.write_text(
                json.dumps(existing, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(self._path)

    @property
    def path(self) -> Path:
        """日志文件路径（只读语义）。"""
        return self._path

    def read_all(self) -> List[Dict[str, Any]]:
        """读取全部记录（测试与审计用）。"""
        with self._lock:
            return self._read_unlocked()

    def _read_unlocked(self) -> List[Dict[str, Any]]:
        """读取现有记录；文件损坏时备份重置，绝不抛异常中断主流程。"""
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            logger.warning("失败日志顶层结构异常（非数组），已备份重置")
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("失败日志损坏（%s），已备份重置", exc)
        backup = self._path.with_suffix(".json.corrupt")
        self._path.replace(backup)
        return []


_failure_logger: Optional[FailureLogger] = None
_failure_logger_lock = threading.Lock()


def get_failure_logger() -> FailureLogger:
    """模块级默认失败日志器单例（路径来自 Settings.failure_log_path）。"""
    global _failure_logger
    with _failure_logger_lock:
        if _failure_logger is None:
            _failure_logger = FailureLogger(settings.failure_log_path)
        return _failure_logger


def _replan(
    func: Callable,
    args: tuple,
    kwargs: Dict[str, Any],
    error_text: str,
    history: List[str],
    llm_client: Any,
) -> Optional[Dict[str, Any]]:
    """调用 LLM 重新规划，返回修正参数字典；失败时返回 None。

    Replan 是纠错层而非业务层：任何解析/调用失败都只降级为
    「原样重试」（fail-open），绝不向上抛异常中断工具执行。
    """
    history_text = "\n".join(history) if history else "（无，本次为首次失败）"
    human_text = REPLAN_HUMAN_TEMPLATE.format(
        func_name=func.__qualname__,
        signature=str(inspect.signature(func)),
        call_args=_truncate(f"args={args!r}, kwargs={kwargs!r}"),
        error=_truncate(error_text),
        history=_truncate(history_text),
    )
    try:
        response = llm_client.invoke(_build_messages(REPLAN_SYSTEM_PROMPT, human_text))
        payload = _extract_json_object(_strip_response_text(response))
    except Exception as exc:  # noqa: BLE001 —— fail-open，见函数 docstring
        logger.warning("Replan 调用/解析失败，退化为原样重试: %s", exc)
        return None

    revised = payload.get("revised_params")
    if not isinstance(revised, dict) or not revised:
        return None
    return revised


def with_retry(
    max_retries: int = 3,
    *,
    task_id: Optional[str] = None,
    replan_llm: Optional[Any] = None,
    failure_logger: Optional[FailureLogger] = None,
) -> Callable:
    """自纠错重试装饰器（装饰器工厂，配合 functools.wraps）。

    语义：总尝试次数 = 1 次初始调用 + max_retries 次重试。
    每次失败将「错误信息 + 历史执行轨迹」拼入 Prompt 调用 LLM 重新规划，
    按 revised_params 修正调用参数后重试；修正不可得则原样重试。
    重试耗尽后重新抛出最后一次异常（不吞错误）。

    失败日志：每次失败一条记录（task_id / failed_step / action /
    error_reason / retry_count / final_success / revised_params /
    timestamp；链成功时回填 resolved_action），在重试链结束时批量写入 ——
    final_success 需链终态才能确定，内存累积后统一回填落盘。
    action / resolved_action 两字段供 finetune 模块构建 DPO 偏好对。

    Args:
        max_retries: 最大重试次数（不含初始调用）。
        task_id: 任务标识；缺省为「模块.函数名#短UUID」。
        replan_llm: 可注入的 Replan LLM；None 时尝试全局单例，
            全局也不可用则退化为无 Replan 的原样重试。
        failure_logger: 可注入日志器；None 时用模块级默认单例。

    Returns:
        装饰器函数。

    Raises:
        ValueError: max_retries 非正。
    """
    if max_retries < 1:
        raise ValueError(f"max_retries 至少为 1，当前: {max_retries}")

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log_writer = failure_logger if failure_logger is not None else get_failure_logger()
            client = replan_llm
            if client is None:
                # 延迟导入全局客户端：保持装饰器在离线环境可用
                from app.core.llm_client import get_llm_client  # noqa: PLC0415

                client = get_llm_client()

            tid = task_id or f"{func.__module__}.{func.__qualname__}#{uuid4().hex[:8]}"
            history: List[str] = []
            call_kwargs = dict(kwargs)
            failures: List[Dict[str, Any]] = []
            last_exc: Optional[BaseException] = None
            sprt_broken = False  # SPRT 配置非法时本次链内禁用（fail-open）

            for attempt in range(1, max_retries + 2):
                # 本次尝试的调用快照（try 前固定：except 内 call_kwargs
                # 已被 Replan 合并覆盖，事后取会把"修正后"误记成"失败时"）
                attempt_repr = _truncate(f"args={args!r}, kwargs={call_kwargs!r}")
                try:
                    result = func(*args, **call_kwargs)
                    # 成功：回填最终结局（含最终成功动作，供微调偏好对构造）
                    for rec in failures:
                        rec["final_success"] = True
                        rec["resolved_action"] = attempt_repr
                    if failures:
                        log_writer.record_many(failures)
                    return result
                except Exception as exc:  # noqa: BLE001 —— 自纠错需捕获任意工具异常
                    last_exc = exc
                    error_text = f"{type(exc).__name__}: {exc}"
                    action = "原样重试"

                    revised = None
                    if client is not None:
                        revised = _replan(func, args, call_kwargs, error_text, history, client)
                    if revised is not None:
                        # 修正参数覆盖原值（保留未提及的原始参数）
                        call_kwargs = {**call_kwargs, **revised}
                        action = f"按 Replan 修正参数重试: {sorted(revised)}"

                    failures.append({
                        "task_id": tid,
                        "failed_step": attempt,      # 第几次尝试（1-based）
                        "action": attempt_repr,      # 失败当时的调用（偏好对 Rejected 侧）
                        "error_reason": _truncate(error_text),
                        "retry_count": attempt - 1,  # 已重试次数（首次失败为 0）
                        "final_success": False,      # 链成功时统一回填 True
                        "revised_params": revised,   # 本次采纳的 Replan 修正（可 None）
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    history.append(f"第{attempt}次尝试失败: {_truncate(error_text)} | 动作: {action}")
                    logger.warning(
                        "任务 %s 第 %d 次尝试失败: %s | %s", tid, attempt, error_text, action
                    )

                    # ---- SPRT 提前止损（统计优化进入运行时路径）----
                    # 证据：到目前为止 trials 次尝试全部失败（successes=0）。
                    # H0: 单次重试成功率 = p0（值得继续）
                    # H1: 退化到 p1（继续只是烧钱）—— 连续失败把对数似然比
                    # 推过 Wald 上界时提前止损，不再烧满 max_retries。
                    # 上限仍是 max_retries：SPRT 只会提前停，不会越权重试。
                    if settings.sprt_retry_enabled and not sprt_broken:
                        try:
                            verdict = sprt_decision(
                                successes=0,
                                trials=len(failures),
                                p0=settings.sprt_retry_p0,
                                p1=settings.sprt_retry_p1,
                            )
                        except ValueError as exc:
                            logger.warning("SPRT 重试参数非法，本次链禁用止损: %s", exc)
                            sprt_broken = True
                        else:
                            if verdict == "accept_h1":
                                logger.warning(
                                    "任务 %s SPRT 判定过程退化，提前止损（连续 %d 次失败，"
                                    "上限 %d 次）",
                                    tid, len(failures), max_retries + 1,
                                )
                                break

            # 重试耗尽（或 SPRT 提前止损）：落盘失败记录（final_success=False）
            # 并抛出原异常 —— 止损只是不再烧重试，错误照常上抛（fail-closed 不变）
            log_writer.record_many(failures)
            assert last_exc is not None  # 循环至少失败一次才会到达此处
            raise last_exc

        return wrapper

    return decorator
