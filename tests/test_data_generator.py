"""evaluation/data_generator.py 单元测试。

全程离线：LLM 用 FakeLLM 注入（成功 / 失败 / 劣化输出三种脚本），
输出文件写入 tmp_path。
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.data_generator import (  # noqa: E402
    ScenarioType,
    build_expected_outcome,
    generate_dataset,
    generate_dialogue,
    generate_specs,
)
from app.evaluation.data_generator import ScenarioSpec  # noqa: E402


class GenFakeLLM:
    """生成用 FakeLLM：按脚本返回 / 抛异常 / 输出劣化。"""

    def __init__(self, response="用户: 你好，请帮我处理这件事。\n助手: 好的，我已经记录下来了。") -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, messages):
        combined = " ".join(
            m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            for m in messages
        )
        self.prompts.append(combined)
        if isinstance(self.response, Exception):
            raise self.response
        return SimpleNamespace(content=self.response)


# ============================ 骨架生成 ============================


class TestGenerateSpecs:

    def test_count_and_ratio(self) -> None:
        """100 条场景，三类比例 40/30/30。"""
        specs = generate_specs(n=100, seed=42)
        assert len(specs) == 100
        by_type = {}
        for spec in specs:
            by_type[spec.scenario_type] = by_type.get(spec.scenario_type, 0) + 1
        assert by_type[ScenarioType.PREFERENCE_SHIFT] == 40
        assert by_type[ScenarioType.LONG_HORIZON] == 30
        assert by_type[ScenarioType.TOOL_FAILURE] == 30

    def test_seed_reproducibility(self) -> None:
        """同 seed 两次生成完全一致（含参数）。"""
        a = generate_specs(n=50, seed=7)
        b = generate_specs(n=50, seed=7)
        assert [s.params for s in a] == [s.params for s in b]
        assert [s.scenario_type for s in a] == [s.scenario_type for s in b]

    def test_different_seed_differs(self) -> None:
        a = generate_specs(n=50, seed=1)
        b = generate_specs(n=50, seed=2)
        assert [s.params for s in a] != [s.params for s in b]

    def test_unique_ids_with_type_prefix(self) -> None:
        """id 全局唯一且带类型前缀 + 类型内独立编号。"""
        specs = generate_specs(n=100, seed=42)
        ids = [s.scenario_id for s in specs]
        assert len(ids) == len(set(ids))
        pref_ids = [s.scenario_id for s in specs if s.scenario_type is ScenarioType.PREFERENCE_SHIFT]
        assert pref_ids[0].startswith("PREF-")
        assert len(pref_ids) == 40

    def test_preference_params_shape(self) -> None:
        """偏好场景骨架含 old/new/reason/turns，且新旧偏好不同。"""
        specs = generate_specs(n=100, seed=42)
        pref = [s for s in specs if s.scenario_type is ScenarioType.PREFERENCE_SHIFT]
        for spec in pref[:10]:
            assert spec.params["old_preference"] != spec.params["new_preference"]
            assert {"old_preference", "new_preference", "reason", "turns"} <= set(spec.params)

    def test_tool_params_shape(self) -> None:
        """工具失败场景骨架含 tool/error/failures。"""
        specs = generate_specs(n=100, seed=42)
        tools = [s for s in specs if s.scenario_type is ScenarioType.TOOL_FAILURE]
        for spec in tools[:10]:
            assert {"tool", "error", "failures"} <= set(spec.params)
            assert 1 <= spec.params["failures"] <= 3

    def test_invalid_n_rejected(self) -> None:
        with pytest.raises(ValueError):
            generate_specs(n=0)


# ============================ ground truth 构造 ============================


class TestBuildExpectedOutcome:

    def test_preference_shift_ground_truth(self) -> None:
        spec = ScenarioSpec("PREF-0001", ScenarioType.PREFERENCE_SHIFT, {
            "old_preference": "黑咖啡", "new_preference": "拿铁",
            "reason": "换口味", "turns": 4,
        })
        outcome = build_expected_outcome(spec)
        assert outcome["type"] == "preference_shift"
        assert outcome["old_preference"] == "黑咖啡"
        assert outcome["new_preference"] == "拿铁"
        assert outcome["expected_action"] == "replaced or fused"

    def test_long_horizon_ground_truth(self) -> None:
        spec = ScenarioSpec("LONG-0001", ScenarioType.LONG_HORIZON, {
            "task": "搭建流水线", "domain": "数据", "min_subtasks": 4, "turns": 5,
        })
        outcome = build_expected_outcome(spec)
        assert outcome["type"] == "long_horizon"
        assert outcome["expected_subtasks_min"] == 4

    def test_tool_failure_ground_truth(self) -> None:
        spec = ScenarioSpec("TOOL-0001", ScenarioType.TOOL_FAILURE, {
            "tool": "search_web", "error": "TimeoutError",
            "intent": "查资料", "failures": 2, "turns": 4,
        })
        outcome = build_expected_outcome(spec)
        assert outcome["type"] == "tool_failure"
        assert outcome["expected_retries_min"] == 2
        assert outcome["injected_error"] == "TimeoutError"


# ============================ 对话生成 ============================


class TestGenerateDialogue:

    def _spec(self, stype: ScenarioType) -> ScenarioSpec:
        specs = generate_specs(n=100, seed=42)
        return next(s for s in specs if s.scenario_type is stype)

    def test_llm_generation_uses_skeleton_params(self) -> None:
        """LLM 收到的 Prompt 包含骨架参数（扩写而非自由发挥）。"""
        llm = GenFakeLLM()
        spec = self._spec(ScenarioType.PREFERENCE_SHIFT)
        generate_dialogue(spec, llm)
        prompt = llm.prompts[0]
        assert spec.params["old_preference"] in prompt
        assert spec.params["new_preference"] in prompt
        assert spec.params["reason"] in prompt

    def test_llm_output_returned_verbatim(self) -> None:
        dialogue = "用户: 自定义对话内容。\n助手: 收到。"
        assert generate_dialogue(self._spec(ScenarioType.LONG_HORIZON), GenFakeLLM(dialogue)) == dialogue

    def test_fallback_on_llm_exception(self) -> None:
        """LLM 抛异常 -> 模板兜底，骨架参数仍在对话中。"""
        spec = self._spec(ScenarioType.TOOL_FAILURE)
        dialogue = generate_dialogue(spec, GenFakeLLM(ConnectionError("网络炸了")))
        assert "用户:" in dialogue
        assert spec.params["tool"] in dialogue

    def test_fallback_on_degraded_output(self) -> None:
        """LLM 输出过短/无对话格式 -> 视为劣化，走兜底。"""
        for bad in ("嗯。", "这是一段说明文字，没有用户: 前缀，也不该被采用。"):
            dialogue = generate_dialogue(self._spec(ScenarioType.TOOL_FAILURE), GenFakeLLM(bad))
            assert "用户:" in dialogue

    def test_offline_returns_template(self) -> None:
        """llm_client=None -> 直接模板兜底。"""
        spec = self._spec(ScenarioType.PREFERENCE_SHIFT)
        dialogue = generate_dialogue(spec, None)
        assert spec.params["old_preference"] in dialogue
        assert spec.params["new_preference"] in dialogue

    def test_strict_mode_raises_without_fallback(self) -> None:
        """allow_fallback=False 且 LLM 失败 -> 抛异常。"""
        with pytest.raises(RuntimeError):
            generate_dialogue(self._spec(ScenarioType.TOOL_FAILURE),
                              GenFakeLLM(ValueError("炸了")), allow_fallback=False)


# ============================ 数据集组装 ============================


class TestGenerateDataset:

    def test_dataset_file_structure(self, tmp_path) -> None:
        """输出 JSON：字段齐全、总数正确、三类齐备。"""
        out = tmp_path / "scenarios.json"
        report = generate_dataset(n=100, seed=42, llm_client=None, output_path=str(out))

        assert report.total == 100
        assert report.fallback_count == 100  # 离线全兜底
        assert Path(report.output_path).exists()

        records = json.loads(out.read_text(encoding="utf-8"))
        assert len(records) == 100
        ids = [r["scenario_id"] for r in records]
        assert len(ids) == len(set(ids))
        for r in records:
            assert {"scenario_id", "scenario_type", "dialogue_history", "expected_outcome"} <= set(r)
            assert "用户:" in r["dialogue_history"]
        types = {r["scenario_type"] for r in records}
        assert types == {"preference_shift", "long_horizon", "tool_failure"}

    def test_expected_outcome_matches_dialogue_params(self, tmp_path) -> None:
        """ground truth 与对话内容同源：偏好出现在对话里，答案由参数推导。"""
        out = tmp_path / "scenarios.json"
        generate_dataset(n=20, seed=3, llm_client=None, output_path=str(out))
        records = json.loads(out.read_text(encoding="utf-8"))
        pref = next(r for r in records if r["scenario_type"] == "preference_shift")
        assert pref["expected_outcome"]["old_preference"] in pref["dialogue_history"]
        assert pref["expected_outcome"]["new_preference"] in pref["dialogue_history"]

    def test_llm_count_reported(self, tmp_path) -> None:
        """LLM 正常时报告 LLM 条数；失败时报告兜底条数。"""
        out = tmp_path / "scenarios.json"
        report_ok = generate_dataset(n=6, seed=42, llm_client=GenFakeLLM(), output_path=str(out))
        assert report_ok.llm_count == 6 and report_ok.fallback_count == 0

        report_bad = generate_dataset(
            n=6, seed=42, llm_client=GenFakeLLM(ValueError("全炸")), output_path=str(out)
        )
        assert report_bad.llm_count == 0 and report_bad.fallback_count == 6

    def test_output_dir_auto_created(self, tmp_path) -> None:
        out = tmp_path / "a" / "b" / "scenarios.json"
        generate_dataset(n=5, seed=42, llm_client=None, output_path=str(out))
        assert out.exists()

    def test_custom_n_exact(self, tmp_path) -> None:
        """任意 n 都精确产出 n 条（余数归 tool_failure）。"""
        out = tmp_path / "s.json"
        for n in (1, 2, 7, 13):
            report = generate_dataset(n=n, seed=1, llm_client=None, output_path=str(out))
            assert report.total == n
