"""评估指标与 A/B 消融检验（阶段五核心模块）。

三大指标域（calculate_metrics）：
1. 记忆模块：Recall@K、MRR、NDCG@K（复用 statistical_eval 的纯函数实现）；
2. 任务模块：成功率、平均步数；
3. 效率模块：Token 总消耗与均值。

A/B 消融（ab_test）——项目灵魂所在：
    对照组 A：无统计优化的基础 Agent（固定阈值 / 新记忆覆盖旧记忆 / 向量 Top-K）；
    实验组 B：统计优化 Agent（SPC+SPRT 自纠错 / 贝叶斯冲突消解 / 混合检索+重排）。
    对每个指标做 Welch's t 检验（独立样本），给出 P 值与均值差的 95% 置信区间，
    产出对比柱状图与 Markdown 评估报告 —— 「统计方法有收益」从此是可检验的
    命题，而不是一句口号。

统计决策（显式声明）：
- 用 Welch's t 检验（equal_var=False）而非经典 Student t：两组方差未必齐性，
  Welch 对方差不齐稳健，且是 scipy 官方推荐的默认选择。成功率是 0/1 二元
  变量，严格最优应 Fisher 精确检验或两比例 z 检验；此处按需求统一用 t 检验，
  大样本下近似成立，Welch 版本对二元的方差失真也最不敏感；
- 95% CI 用 Welch–Satterthwaite 近似自由度 + t 分位数手动构造，不依赖
  scipy 版本相关的 CI API；
- 差值方向统一为 mean_b - mean_a（B 相对 A 的收益），配合
  higher_better 标志解释改善/退化。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from scipy import stats

from app.evaluation.statistical_eval import ndcg_at_k as _ndcg_single
from app.evaluation.statistical_eval import mrr as _rr_single

logger = logging.getLogger(__name__)


# ============================ 数据契约 ============================


@dataclass(frozen=True)
class TaskResult:
    """单次任务执行的评估记录（一条 = 一次 run 的完整可测输出）。

    Attributes:
        task_id: 任务唯一标识。
        relevant_ids: ground truth 相关记忆 id 集合（记忆模块评估基准）。
        retrieved_ids: 系统实际召回的记忆 id（按系统排序）。
        success: 任务是否成功完成。
        n_steps: 实际执行的步数（子任务数 / 工具调用数）。
        tokens: 本次任务的 Token 总消耗。
    """

    task_id: str
    relevant_ids: List[str] = field(default_factory=list)
    retrieved_ids: List[str] = field(default_factory=list)
    success: bool = False
    n_steps: int = 1
    tokens: int = 0

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id 不能为空")
        if self.n_steps < 0:
            raise ValueError(f"n_steps 不可为负: {self.n_steps}")
        if self.tokens < 0:
            raise ValueError(f"tokens 不可为负: {self.tokens}")


def _coerce_result(item: Union[TaskResult, Dict[str, Any]]) -> TaskResult:
    """宽容输入：TaskResult 直接用，dict 转换（键需匹配字段名）。"""
    if isinstance(item, TaskResult):
        return item
    if isinstance(item, dict):
        return TaskResult(**item)
    raise TypeError(f"无法解析的结果类型: {type(item).__name__}")


# ============================ 指标聚合 ============================


@dataclass
class MetricsSummary:
    """一组结果的指标汇总（聚合值用于展示，per-task 数组用于检验）。

    Attributes:
        n_tasks: 任务数。
        k: Recall@K / NDCG@K 的截断位置。
        recall_values / mrr_values / ndcg_values: 每任务的记忆指标数组。
        success_values: 每任务成功与否的 0/1 数组。
        steps_values / tokens_values: 每任务步数与 Token 数组。
    """

    n_tasks: int
    k: int
    recall_values: List[float]
    mrr_values: List[float]
    ndcg_values: List[float]
    success_values: List[float]
    steps_values: List[float]
    tokens_values: List[float]

    # ---------- 聚合展示值 ----------

    @property
    def recall_at_k(self) -> float:
        """Recall@K 均值。"""
        return sum(self.recall_values) / self.n_tasks

    @property
    def mrr(self) -> float:
        """MRR 均值。"""
        return sum(self.mrr_values) / self.n_tasks

    @property
    def ndcg_at_k(self) -> float:
        """NDCG@K 均值。"""
        return sum(self.ndcg_values) / self.n_tasks

    @property
    def success_rate(self) -> float:
        """任务成功率。"""
        return sum(self.success_values) / self.n_tasks

    @property
    def avg_steps(self) -> float:
        """平均步数。"""
        return sum(self.steps_values) / self.n_tasks

    @property
    def total_tokens(self) -> int:
        """Token 总消耗。"""
        return int(sum(self.tokens_values))

    @property
    def avg_tokens(self) -> float:
        """平均 Token 消耗。"""
        return sum(self.tokens_values) / self.n_tasks


def calculate_metrics(results: Sequence[Union[TaskResult, Dict[str, Any]]], k: int = 5) -> MetricsSummary:
    """计算记忆 / 任务 / 效率三大模块的指标。

    Args:
        results: TaskResult（或等价 dict）序列。
        k: Recall@K 与 NDCG@K 的截断位置（默认 5）。

    Returns:
        MetricsSummary，含聚合值与 per-task 数组。

    Raises:
        ValueError: results 为空，或 k < 1。
    """
    if not results:
        raise ValueError("results 不能为空")
    if k < 1:
        raise ValueError(f"k 至少为 1，当前: {k}")

    recall_arr: List[float] = []
    mrr_arr: List[float] = []
    ndcg_arr: List[float] = []
    success_arr: List[float] = []
    steps_arr: List[float] = []
    tokens_arr: List[float] = []

    for item in results:
        r = _coerce_result(item)
        relevant = set(r.relevant_ids)

        if not relevant:
            # 契约要求：无 ground truth 的任务不能进记忆评估（会稀释指标）
            raise ValueError(f"任务 {r.task_id} 的 relevant_ids 为空")

        if not r.retrieved_ids:
            # 召回为空：三个记忆指标全 0（比跳过更诚实——空召回是系统性失败）
            recall_arr.append(0.0)
            mrr_arr.append(0.0)
            ndcg_arr.append(0.0)
        else:
            hit_topk = sum(1 for tid in r.retrieved_ids[:k] if tid in relevant)
            recall_arr.append(hit_topk / len(relevant))

            binary = [1 if tid in relevant else 0 for tid in r.retrieved_ids]
            mrr_arr.append(_rr_single(binary))
            # 二值相关性代入 NDCG（ground truth 只有集合，无分级标签）
            ndcg_arr.append(_ndcg_single([float(b) for b in binary], min(k, len(binary))))

        success_arr.append(1.0 if r.success else 0.0)
        steps_arr.append(float(r.n_steps))
        tokens_arr.append(float(r.tokens))

    return MetricsSummary(
        n_tasks=len(results),
        k=k,
        recall_values=recall_arr,
        mrr_values=mrr_arr,
        ndcg_values=ndcg_arr,
        success_values=success_arr,
        steps_values=steps_arr,
        tokens_values=tokens_arr,
    )


# ============================ A/B 消融检验 ============================


@dataclass
class MetricComparison:
    """单指标的 A/B 对比与检验结论。"""

    metric: str            # 指标键名
    label: str             # 展示名（中文）
    higher_better: bool    # True=越大越好（如成功率），False=越小越好（如 Token）
    mean_a: float
    mean_b: float
    diff: float            # mean_b - mean_a（B 相对 A 的变化）
    t_stat: float
    p_value: float
    ci_low: float          # diff 的 95% CI 下界
    ci_high: float         # diff 的 95% CI 上界
    significant: bool      # p < alpha

    @property
    def b_improved(self) -> bool:
        """B 是否显著改善（结合方向判定）。"""
        return self.significant and (
            (self.diff > 0 and self.higher_better) or (self.diff < 0 and not self.higher_better)
        )


@dataclass
class ABTestReport:
    """完整 A/B 消融报告。

    Attributes:
        arrays_a / arrays_b: 各指标的 per-task 原始数组（键 = 指标键名），
            供图表误差棒与后续效应量分析使用。
    """

    n_a: int
    n_b: int
    k: int
    alpha: float
    comparisons: List[MetricComparison]
    arrays_a: Dict[str, List[float]] = field(default_factory=dict)
    arrays_b: Dict[str, List[float]] = field(default_factory=dict)
    chart_path: Optional[str] = None
    report_path: Optional[str] = None
    markdown: str = ""

    def significant_improvements(self) -> List[MetricComparison]:
        """B 显著优于 A 的指标列表。"""
        return [c for c in self.comparisons if c.b_improved]

    def significant_degradations(self) -> List[MetricComparison]:
        """B 显著差于 A 的指标列表。"""
        return [
            c for c in self.comparisons
            if c.significant and not c.b_improved and c.diff != 0.0
        ]


def _welch_ttest(arr_a: List[float], arr_b: List[float]) -> tuple[float, float, float, float]:
    """Welch t 检验 + 均值差 95% CI（Welch–Satterthwaite 自由度）。

    Returns:
        (t_stat, p_value, ci_low, ci_high)，diff = mean_b - mean_a。

    Raises:
        ValueError: 任一组样本数 < 2（方差无法估计）。
    """
    n_a, n_b = len(arr_a), len(arr_b)
    if n_a < 2 or n_b < 2:
        raise ValueError(f"t 检验要求每组至少 2 个样本，当前 A={n_a}, B={n_b}")

    result = stats.ttest_ind(arr_b, arr_a, equal_var=False)  # diff 方向 = B - A
    mean_a = sum(arr_a) / n_a
    mean_b = sum(arr_b) / n_b
    diff = mean_b - mean_a

    var_a = sum((x - mean_a) ** 2 for x in arr_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in arr_b) / (n_b - 1)
    se_sq_a, se_sq_b = var_a / n_a, var_b / n_b
    se = math.sqrt(se_sq_a + se_sq_b)

    if se == 0.0:
        # 两组均为零方差：无统计噪声。均值相等 -> 无差异（p=1）；
        # 均值不等 -> 确定性差异（p=0），CI 退化为点区间。
        if diff == 0.0:
            return 0.0, 1.0, 0.0, 0.0
        return math.inf, 0.0, diff, diff

    # Welch–Satterthwaite 近似自由度
    df = (se_sq_a + se_sq_b) ** 2 / (
        se_sq_a**2 / (n_a - 1) + se_sq_b**2 / (n_b - 1)
    )
    t_crit = stats.t.ppf(0.975, df)
    return float(result.statistic), float(result.pvalue), diff - t_crit * se, diff + t_crit * se


# 指标注册表：(键名, 展示名, 越大越好?, 取 per-task 数组的属性名)
_METRIC_DEFS = [
    ("recall_at_k", "Recall@K", True, "recall_values"),
    ("mrr", "MRR", True, "mrr_values"),
    ("ndcg_at_k", "NDCG@K", True, "ndcg_values"),
    ("success_rate", "成功率", True, "success_values"),
    ("avg_steps", "平均步数", False, "steps_values"),
    ("avg_tokens", "Token 消耗", False, "tokens_values"),
]


def ab_test(
    group_a_results: Sequence[Union[TaskResult, Dict[str, Any]]],
    group_b_results: Sequence[Union[TaskResult, Dict[str, Any]]],
    k: int = 5,
    alpha: float = 0.05,
    output_dir: str = "evaluation",
    make_chart: bool = True,
    make_report: bool = True,
) -> ABTestReport:
    """A/B 消融检验：对照 A（基础 Agent）vs 实验 B（统计优化 Agent）。

    对六个指标逐一做 Welch's t 检验（P 值 + 95% CI），默认同时产出：
    - 对比柱状图 -> {output_dir}/results.png
    - Markdown 报告 -> {output_dir}/report.md

    Args:
        group_a_results: 对照组结果集。
        group_b_results: 实验组结果集。
        k: 记忆指标截断位置。
        alpha: 显著性水平。
        output_dir: 图表与报告的输出目录。
        make_chart: 是否绘制对比柱状图。
        make_report: 是否生成 Markdown 报告。

    Returns:
        ABTestReport。

    Raises:
        ValueError: 任一组为空或样本数 < 2。
    """
    summary_a = calculate_metrics(group_a_results, k=k)
    summary_b = calculate_metrics(group_b_results, k=k)

    comparisons: List[MetricComparison] = []
    for key, label, higher_better, attr in _METRIC_DEFS:
        arr_a: List[float] = getattr(summary_a, attr)
        arr_b: List[float] = getattr(summary_b, attr)
        t_stat, p_value, ci_low, ci_high = _welch_ttest(arr_a, arr_b)
        comparisons.append(MetricComparison(
            metric=key,
            label=label,
            higher_better=higher_better,
            mean_a=sum(arr_a) / len(arr_a),
            mean_b=sum(arr_b) / len(arr_b),
            diff=sum(arr_b) / len(arr_b) - sum(arr_a) / len(arr_a),
            t_stat=t_stat,
            p_value=p_value,
            ci_low=ci_low,
            ci_high=ci_high,
            significant=p_value < alpha,
        ))

    report = ABTestReport(
        n_a=summary_a.n_tasks,
        n_b=summary_b.n_tasks,
        k=k,
        alpha=alpha,
        comparisons=comparisons,
        arrays_a={key: list(getattr(summary_a, attr)) for key, _, _, attr in _METRIC_DEFS},
        arrays_b={key: list(getattr(summary_b, attr)) for key, _, _, attr in _METRIC_DEFS},
    )

    if make_chart:
        chart_path = Path(output_dir) / "results.png"
        render_chart(report, str(chart_path))
        report.chart_path = str(chart_path)

    if make_report:
        md = generate_markdown_report(report)
        report_path = Path(output_dir) / "report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(md, encoding="utf-8")
        report.report_path = str(report_path)
        report.markdown = md

    logger.info(
        "A/B 检验完成: A=%d, B=%d, 显著改善 %d 项 / 显著退化 %d 项",
        report.n_a, report.n_b,
        len(report.significant_improvements()), len(report.significant_degradations()),
    )
    return report


# ============================ 图表 ============================


def render_chart(report: ABTestReport, output_path: str) -> str:
    """绘制 A/B 对比柱状图（2×3 子图，误差棒为均值 95% CI）。

    工程细节：matplotlib.use("Agg") 强制无头后端（无显示器环境可跑）；
    rcParams 挂中文字体（Windows: 微软雅黑），否则中文全部渲染为方框。

    Args:
        report: ab_test 的检验报告。
        output_path: PNG 输出路径（目录自动创建）。

    Returns:
        实际写入的路径。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    color_a, color_b = "#9e9e9e", "#2e7d32"  # 对照组灰 / 实验组绿

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    fig.suptitle(
        f"A/B 消融实验：基础 Agent (A, n={report.n_a}) vs 统计优化 Agent (B, n={report.n_b})"
        f"   [Welch's t-test, α={report.alpha}]",
        fontsize=13, fontweight="bold",
    )

    for ax, comp in zip(axes.flat, report.comparisons):
        # 误差棒 = 各组均值的 95% CI 半宽（t 临界 × 标准误，从 per-task 原始数组计算）
        yerr = [
            _group_ci_halfwidth(report.arrays_a[comp.metric]),
            _group_ci_halfwidth(report.arrays_b[comp.metric]),
        ]
        bars = ax.bar(["A 基础", "B 统计优化"], [comp.mean_a, comp.mean_b],
                      color=[color_a, color_b], width=0.55,
                      yerr=yerr, capsize=5,
                      error_kw={"ecolor": "#444444", "elinewidth": 1.2})
        # 数值标签放在误差棒上沿之上，避免与 cap 重叠
        for idx, (bar, value) in enumerate(zip(bars, (comp.mean_a, comp.mean_b))):
            ax.text(bar.get_x() + bar.get_width() / 2, value + yerr[idx],
                    f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        star = " *" if comp.significant else ""
        ax.set_title(f"{comp.label}{'（越低越好）' if not comp.higher_better else ''}"
                     f"  p={comp.p_value:.4f}{star}", fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        # 给误差棒上留出空间
        ax.margins(y=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def _group_ci_halfwidth(values: List[float]) -> float:
    """单组均值的 95% CI 半宽（t 临界 × 标准误）；零方差返回 0。"""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    sem = math.sqrt(var / n)
    if sem == 0.0:
        return 0.0
    return float(stats.t.ppf(0.975, n - 1) * sem)


# ============================ Markdown 报告 ============================


def generate_markdown_report(report: ABTestReport) -> str:
    """生成 Markdown 评估报告（元信息 + 对比表 + 自动结论 + 图表引用）。

    Args:
        report: ab_test 的检验报告。

    Returns:
        Markdown 字符串。
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = [
        "# A/B 消融实验评估报告",
        "",
        f"- **生成时间**: {now}",
        f"- **对照组 A**（无统计优化的基础 Agent）: {report.n_a} 个任务",
        f"- **实验组 B**（统计优化 Agent：SPC/SPRT 自纠错 + 贝叶斯冲突消解 + 混合检索）: {report.n_b} 个任务",
        f"- **检验方法**: Welch's t 检验（独立样本，双侧），显著性水平 α = {report.alpha}",
        f"- **置信区间**: 均值差 (B − A) 的 95% CI（Welch–Satterthwaite 自由度）",
        "",
        "## 指标对比",
        "",
        "| 指标 | A 均值 | B 均值 | Δ (B−A) | t 统计量 | P 值 | 95% CI | 显著 | 方向 |",
        "|------|--------|--------|---------|----------|------|--------|------|------|",
    ]

    for c in report.comparisons:
        direction = "越高越好" if c.higher_better else "越低越好"
        lines.append(
            f"| {c.label} | {c.mean_a:.4f} | {c.mean_b:.4f} | {c.diff:+.4f} "
            f"| {c.t_stat:.3f} | {c.p_value:.4f} | [{c.ci_low:+.4f}, {c.ci_high:+.4f}] "
            f"| {'✓' if c.significant else '✗'} | {direction} |"
        )

    lines += ["", "## 结论", ""]

    improvements = report.significant_improvements()
    degradations = report.significant_degradations()

    if improvements:
        lines.append("**统计优化（B 组）带来显著改善的指标：**")
        lines.append("")
        for c in improvements:
            lines.append(
                f"- **{c.label}**: {c.mean_a:.4f} → {c.mean_b:.4f}"
                f"（Δ = {c.diff:+.4f}，95% CI [{c.ci_low:+.4f}, {c.ci_high:+.4f}]，"
                f"p = {c.p_value:.4f}）"
            )
        lines.append("")

    if degradations:
        lines.append("**显著退化的指标（需关注成本/步数代价）：**")
        lines.append("")
        for c in degradations:
            lines.append(
                f"- **{c.label}**: {c.mean_a:.4f} → {c.mean_b:.4f}"
                f"（Δ = {c.diff:+.4f}，p = {c.p_value:.4f}）"
            )
        lines.append("")

    if not improvements and not degradations:
        lines.append(
            f"在 α = {report.alpha} 水平下，未观察到任何指标的显著差异。"
            "可能原因：样本量不足（检验功效低）、或统计优化的真实效应量小于组内噪声。"
            "建议：增大样本量后重跑，或按指标做效应量（Cohen's d）分析。"
        )
        lines.append("")

    lines += [
        "## 实验设计说明",
        "",
        "- **记忆模块指标**（Recall@K / MRR / NDCG@K）：基于 ground truth 相关记忆集合"
        "与系统召回排序的对照评估，逐任务计算后取均值；",
        "- **任务模块指标**（成功率 / 平均步数）：成功率按 0/1 二元变量处理，"
        "步数为实际执行的子任务数；",
        "- **效率模块指标**（Token 消耗）：含规划、执行、Critic 验证、Replan 的全部消耗；",
        "- 成功率为二元变量，t 检验为其近似方案（严格应 Fisher 精确检验）；"
        "其余连续指标在独立样本假设下 Welch's t 检验是稳健选择。",
        "",
    ]

    if report.chart_path:
        lines += ["![A/B 对比柱状图](results.png)", ""]

    return "\n".join(lines)
