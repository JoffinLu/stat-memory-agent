"""基准 runner 与黄金集单元测试（离线、确定性）。"""

from pathlib import Path

from app.evaluation.golden_set import (
    BENCHMARK_NOW,
    TASK_CONFLICT_STRONG,
    TASK_CONFLICT_WEAK,
    TASK_RETRY_HOPELESS,
    TASK_RETRY_TRANSIENT,
    generate_golden_tasks,
)
from app.evaluation.run_benchmark import GROUP_A, GROUP_B, run_benchmark, run_group
from app.evaluation.metrics import ab_test


class TestGoldenSet:
    def test_deterministic(self) -> None:
        a = generate_golden_tasks(n_conflict=6, n_retry=6, seed=7)
        b = generate_golden_tasks(n_conflict=6, n_retry=6, seed=7)
        assert [(t.task_id, t.task_type) for t in a] == [(t.task_id, t.task_type) for t in b]

    def test_task_mix(self) -> None:
        tasks = generate_golden_tasks(n_conflict=10, n_retry=10, seed=42)
        types = [t.task_type for t in tasks]
        assert types.count(TASK_CONFLICT_WEAK) == 5
        assert types.count(TASK_CONFLICT_STRONG) == 5
        assert types.count(TASK_RETRY_HOPELESS) == 6
        assert types.count(TASK_RETRY_TRANSIENT) == 4

    def test_conflict_math(self) -> None:
        """弱新证据任务：旧记忆衰减后仍显著强于新证据（|diff| > 0.1）。"""
        import math as _m

        old_eff = 0.9 * _m.exp(-0.01 * 30)   # 0.667
        new_eff = 0.4
        assert abs(old_eff - new_eff) > 0.1  # 走 kept_with_history，不走融合
        old_eff = 0.4 * _m.exp(-0.01 * 30)   # 0.296
        new_eff = 0.9
        assert new_eff - old_eff > 0.1       # 走 replaced


class TestRunBenchmark:
    def test_statistical_group_wins_conflict(self, tmp_path: Path) -> None:
        """统计组应全对（弱新证据保留 + 强新证据采纳），朴素组弱证据必败。"""
        tasks = generate_golden_tasks(n_conflict=6, n_retry=4, seed=1)
        result_b = run_group(tasks, GROUP_B, tmp_path / "b")
        result_a = run_group(tasks, GROUP_A, tmp_path / "a")

        conflict_idx = [
            i for i, t in enumerate(tasks)
            if t.task_type in (TASK_CONFLICT_WEAK, TASK_CONFLICT_STRONG)
        ]
        b_conflict = [result_b[i] for i in conflict_idx]
        a_conflict = [result_a[i] for i in conflict_idx]
        assert all(r.success for r in b_conflict)          # B 全对
        assert not all(r.success for r in a_conflict)      # A 弱证据任务必败

    def test_sprt_saves_attempts_on_hopeless_tools(self, tmp_path: Path) -> None:
        """无望工具：B 组（SPRT）平均尝试次数显著少于 A 组（固定重试）。"""
        tasks = generate_golden_tasks(n_conflict=2, n_retry=6, seed=3)
        result_a = run_group(tasks, GROUP_A, tmp_path / "a")
        result_b = run_group(tasks, GROUP_B, tmp_path / "b")

        hopeless_idx = [i for i, t in enumerate(tasks) if t.task_type == TASK_RETRY_HOPELESS]
        a_steps = [result_a[i].n_steps for i in hopeless_idx]
        b_steps = [result_b[i].n_steps for i in hopeless_idx]
        assert all(s == 9 for s in a_steps)   # 1 + max_retries(8)
        assert all(s == 5 for s in b_steps)   # SPRT Wald 上界：5 次失败止损

    def test_transient_recovery_unharmed(self, tmp_path: Path) -> None:
        """瞬态故障：SPRT 不影响正常恢复（两组都 2 次尝试成功）。"""
        tasks = generate_golden_tasks(n_conflict=0, n_retry=4, seed=5)
        for cfg in (GROUP_A, GROUP_B):
            results = run_group(tasks, cfg, tmp_path / cfg.name)
            transient = [
                r for r, t in zip(results, tasks) if t.task_type == TASK_RETRY_TRANSIENT
            ]
            assert all(r.success and r.n_steps == 2 for r in transient)

    def test_full_benchmark_produces_artifacts(self, tmp_path: Path) -> None:
        """端到端：小规模基准跑通，分层产出图表与 Markdown 报告。"""
        summary = run_benchmark(
            n_conflict=6, n_retry=6, seed=42,
            output_dir=str(tmp_path / "benchmark"),
        )
        out = Path(summary["output_dir"])
        for layer in ("conflict", "retry"):
            assert (out / layer / "results.png").exists()
            assert (out / layer / "report.md").exists()

        # 冲突层：成功率 B 组显著优于 A 组（弱证据任务 A 必败 vs B 全对）
        conflict = summary["report_conflict"]
        success_comp = next(c for c in conflict.comparisons if "成功率" in c.label)
        assert success_comp.mean_b > success_comp.mean_a

        # 重试层：平均步数 B 组显著更少（SPRT 止损），成功率两组相同
        retry = summary["report_retry"]
        steps_comp = next(c for c in retry.comparisons if "步数" in c.label)
        assert steps_comp.mean_b < steps_comp.mean_a
        success_comp = next(c for c in retry.comparisons if "成功率" in c.label)
        assert success_comp.mean_b >= success_comp.mean_a  # 止损不以牺牲结果为代价
