"""evaluation/metrics.py 单元测试：指标计算 + A/B 检验 + 图表 + 报告。

统计值均为手工计算的对拍基准，全程离线（matplotlib Agg 后端）。
"""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.metrics import (  # noqa: E402
    ABTestReport,
    TaskResult,
    ab_test,
    calculate_metrics,
    generate_markdown_report,
    render_chart,
)
from app.evaluation.metrics import _welch_ttest  # noqa: E402


# ============================ 构造工具 ============================


def make_result(
    task_id: str,
    relevant: list,
    retrieved: list,
    success: bool = True,
    steps: int = 3,
    tokens: int = 100,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        relevant_ids=relevant,
        retrieved_ids=retrieved,
        success=success,
        n_steps=steps,
        tokens=tokens,
    )


# ============================ 记忆模块指标 ============================


class TestMemoryMetrics:

    def test_recall_at_k_exact(self) -> None:
        """Recall@K 手工对拍：top-2 命中 1/2 个相关。"""
        r = make_result("t1", relevant=["a", "b"], retrieved=["a", "c", "b"])
        summary = calculate_metrics([r], k=2)
        assert summary.recall_at_k == pytest.approx(0.5)

    def test_recall_full_hit(self) -> None:
        r = make_result("t1", relevant=["a", "b"], retrieved=["a", "b", "c"])
        assert calculate_metrics([r], k=2).recall_at_k == pytest.approx(1.0)

    def test_mrr_first_position(self) -> None:
        """第一个相关结果在第 1 位 -> RR = 1.0。"""
        r = make_result("t1", relevant=["a"], retrieved=["a", "x", "y"])
        assert calculate_metrics([r]).mrr == pytest.approx(1.0)

    def test_mrr_third_position(self) -> None:
        r = make_result("t1", relevant=["a"], retrieved=["x", "y", "a"])
        assert calculate_metrics([r]).mrr == pytest.approx(1.0 / 3)

    def test_mrr_no_hit(self) -> None:
        r = make_result("t1", relevant=["a"], retrieved=["x", "y"])
        assert calculate_metrics([r]).mrr == pytest.approx(0.0)

    def test_ndcg_ideal_ordering_is_one(self) -> None:
        """完美排序 -> NDCG = 1.0。"""
        r = make_result("t1", relevant=["a", "b"], retrieved=["a", "b", "x"])
        assert calculate_metrics([r], k=3).ndcg_at_k == pytest.approx(1.0)

    def test_ndcg_hand_computed(self) -> None:
        """NDCG 手工对拍：binary [1,0,1]，k=3。

        DCG  = 1/log2(2) + 0 + 1/log2(4) = 1.5
        IDCG = 1/log2(2) + 1/log2(3)     = 1.63093
        NDCG = 1.5 / 1.63093             = 0.91972
        """
        r = make_result("t1", relevant=["a", "b"], retrieved=["a", "c", "b"])
        assert calculate_metrics([r], k=3).ndcg_at_k == pytest.approx(0.919721, rel=1e-5)

    def test_empty_retrieval_scores_zero(self) -> None:
        """空召回 = 系统性失败，三指标全 0（不跳过、不豁免）。"""
        r = make_result("t1", relevant=["a"], retrieved=[])
        summary = calculate_metrics([r])
        assert summary.recall_at_k == 0.0
        assert summary.mrr == 0.0
        assert summary.ndcg_at_k == 0.0


# ============================ 任务 / 效率模块指标 ============================


class TestTaskAndEfficiencyMetrics:

    def test_success_rate_and_steps(self) -> None:
        results = [
            make_result("t1", ["a"], ["a"], success=True, steps=2),
            make_result("t2", ["a"], ["a"], success=True, steps=4),
            make_result("t3", ["a"], ["a"], success=False, steps=6),
        ]
        summary = calculate_metrics(results)
        assert summary.success_rate == pytest.approx(2 / 3)
        assert summary.avg_steps == pytest.approx(4.0)

    def test_tokens_total_and_avg(self) -> None:
        results = [
            make_result("t1", ["a"], ["a"], tokens=100),
            make_result("t2", ["a"], ["a"], tokens=300),
        ]
        summary = calculate_metrics(results)
        assert summary.total_tokens == 400
        assert summary.avg_tokens == pytest.approx(200.0)

    def test_dict_input_coerced(self) -> None:
        """dict 形式的结果可透明转换。"""
        summary = calculate_metrics([{
            "task_id": "t1", "relevant_ids": ["a"], "retrieved_ids": ["a"],
            "success": True, "n_steps": 2, "tokens": 50,
        }])
        assert summary.success_rate == 1.0
        assert summary.avg_tokens == 50.0

    def test_aggregation_across_tasks(self) -> None:
        """多任务聚合 = 每任务指标的无权均值。"""
        results = [
            make_result("t1", ["a", "b"], ["a", "x"]),   # recall@1 = 0.5
            make_result("t2", ["a"], ["a"]),             # recall@1 = 1.0
        ]
        summary = calculate_metrics(results, k=1)
        assert summary.recall_at_k == pytest.approx(0.75)

    def test_empty_results_rejected(self) -> None:
        with pytest.raises(ValueError):
            calculate_metrics([])

    def test_missing_ground_truth_rejected(self) -> None:
        with pytest.raises(ValueError, match="relevant_ids"):
            calculate_metrics([make_result("t1", [], ["a"])])

    def test_invalid_steps_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_result("t1", ["a"], ["a"], steps=-1)


# ============================ Welch t 检验 ============================


class TestWelchTtest:

    def test_known_difference_significant(self) -> None:
        """确定性构造：A=[0,0,0,0,1] vs B=[1,1,1,1,1] -> t=4.0, df=4, p≈0.016。"""
        t_stat, p_value, ci_low, ci_high = _welch_ttest(
            [0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]
        )
        assert t_stat == pytest.approx(4.0, rel=1e-3)
        assert p_value == pytest.approx(0.01613, rel=1e-2)
        assert ci_low > 0  # 差值 CI 不跨 0（显著）
        assert ci_low < 0.8 < ci_high

    def test_identical_groups_not_significant(self) -> None:
        arr = [0.5, 0.6, 0.4, 0.7, 0.5]
        _, p_value, ci_low, ci_high = _welch_ttest(arr, list(arr))
        assert p_value == pytest.approx(1.0)
        assert ci_low < 0 < ci_high  # CI 跨 0

    def test_zero_variance_both_groups(self) -> None:
        """两组零方差：均值相等 -> p=1；不等 -> 确定性差异 p=0。"""
        assert _welch_ttest([1.0, 1.0], [1.0, 1.0]) == (0.0, 1.0, 0.0, 0.0)
        t_stat, p_value, ci_low, ci_high = _welch_ttest([0.0, 0.0], [1.0, 1.0])
        assert p_value == 0.0
        assert ci_low == ci_high == 1.0

    def test_single_sample_rejected(self) -> None:
        with pytest.raises(ValueError, match="至少 2 个样本"):
            _welch_ttest([1.0], [1.0, 2.0])


# ============================ A/B 消融 ============================


class TestABTest:

    def _base_group(self, success_rate: float, n: int, prefix: str) -> list:
        """构造可控结果集：成功率为 success_rate，其余字段有微噪声。"""
        results = []
        n_success = round(n * success_rate)
        for i in range(n):
            results.append(make_result(
                f"{prefix}-{i:03d}",
                relevant=["mem-1", "mem-2"],
                retrieved=["mem-1", "mem-3", "mem-2"],  # 排序恒定，记忆指标恒定
                success=i < n_success,
                steps=3 if i < n_success else 6,
                tokens=500 if i < n_success else 800,
            ))
        return results

    def test_full_pipeline_artifacts(self, tmp_path) -> None:
        """端到端：检验 + 图表 + Markdown 报告一次产出。"""
        group_a = self._base_group(0.5, 20, "A")
        group_b = self._base_group(0.9, 20, "B")

        report = ab_test(
            group_a, group_b,
            output_dir=str(tmp_path),
        )

        assert isinstance(report, ABTestReport)
        assert report.n_a == 20 and report.n_b == 20
        assert len(report.comparisons) == 6

        # 图表与报告落盘
        chart = Path(report.chart_path)
        md = Path(report.report_path)
        assert chart.exists() and chart.stat().st_size > 0
        assert md.exists()
        assert chart.name == "results.png"
        assert md.name == "report.md"
        assert report.markdown  # markdown 内容同时在内存

    def test_b_better_significant_improvement(self, tmp_path) -> None:
        """B 明显更优 -> 成功率显著改善且方向正确。"""
        report = ab_test(
            self._base_group(0.4, 30, "A"),
            self._base_group(0.9, 30, "B"),
            output_dir=str(tmp_path), make_chart=False, make_report=False,
        )
        succ = next(c for c in report.comparisons if c.metric == "success_rate")
        assert succ.significant
        assert succ.b_improved
        assert succ.diff > 0
        assert succ.ci_low > 0  # 差值 CI 不跨 0

        # B 组步数/Token 更低（低越好）-> 改善
        tokens = next(c for c in report.comparisons if c.metric == "avg_tokens")
        assert tokens.b_improved

    def test_identical_groups_no_significance(self, tmp_path) -> None:
        """两组相同 -> 无显著项，报告含样本量建议。"""
        group = self._base_group(0.6, 15, "S")
        report = ab_test(group, list(group), output_dir=str(tmp_path),
                         make_chart=False, make_report=False)
        assert all(not c.significant for c in report.comparisons)

    def test_markdown_report_contents(self, tmp_path) -> None:
        """报告包含元信息、对比表、结论与图表引用。"""
        report = ab_test(
            self._base_group(0.4, 25, "A"),
            self._base_group(0.9, 25, "B"),
            output_dir=str(tmp_path),
        )
        md = report.markdown
        assert "# A/B 消融实验评估报告" in md
        assert "Welch's t 检验" in md
        assert "| 指标 | A 均值 | B 均值 |" in md
        for label in ("Recall@K", "MRR", "NDCG@K", "成功率", "平均步数", "Token 消耗"):
            assert label in md
        assert "p =" in md
        assert "显著改善" in md
        assert "results.png" in md  # 图表引用

    def test_memory_metrics_direction_not_forced(self, tmp_path) -> None:
        """记忆指标在两组合并的构造中恒定 -> 不应误报显著。"""
        results_a = self._base_group(0.5, 10, "A")
        results_b = self._base_group(0.5, 10, "B")  # 检索行为相同，仅成功率不同
        report = ab_test(results_a, results_b, output_dir=str(tmp_path),
                         make_chart=False, make_report=False)
        recall = next(c for c in report.comparisons if c.metric == "recall_at_k")
        assert not recall.significant
        assert recall.diff == pytest.approx(0.0)

    def test_empty_group_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            ab_test([], self._base_group(0.5, 5, "B"), output_dir=str(tmp_path))

    def test_chart_with_zero_variance_group(self, tmp_path) -> None:
        """零方差组（全部相同）也能画图不炸。"""
        flat = [make_result(f"t{i}", ["a"], ["a"], success=True, steps=3, tokens=100)
                for i in range(4)]
        report = ab_test(flat, flat, output_dir=str(tmp_path), make_report=False)
        assert Path(report.chart_path).exists()
