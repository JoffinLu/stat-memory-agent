"""finetune 数据管道单元测试（全离线，不依赖训练库）。"""

import json
from pathlib import Path

import pytest

from app.agent.finetune import (
    QLoRADPOConfig,
    build_preference_pairs,
    export_sharegpt,
    group_by_task,
    load_records,
    run_data_pipeline,
    to_sharegpt,
)

# ---------------- 测试数据 ----------------

SUCCESS_CHAIN = [
    {
        "task_id": "app.tools.fetch#abc",
        "failed_step": 1,
        "action": "args=(), kwargs={'url': 'http://bad'}",
        "error_reason": "ConnectionError: timeout",
        "retry_count": 0,
        "final_success": True,
        "revised_params": {"url": "http://good"},
        "timestamp": "2026-09-15T08:00:00+00:00",
    },
    {
        "task_id": "app.tools.fetch#abc",
        "failed_step": 2,
        "action": "args=(), kwargs={'url': 'http://good'}",
        "error_reason": "ValueError: parse failed",
        "retry_count": 1,
        "final_success": True,
        "revised_params": None,
        "resolved_action": "args=(), kwargs={'url': 'http://good', 'timeout': 30}",
        "timestamp": "2026-09-15T08:00:01+00:00",
    },
]

PURE_FAILURE = {
    "task_id": "app.tools.div#def",
    "failed_step": 1,
    "action": "args=(8,), kwargs={'denominator': 0}",
    "error_reason": "ZeroDivisionError: division by zero",
    "retry_count": 0,
    "final_success": False,
    "revised_params": None,
    "timestamp": "2026-09-15T08:01:00+00:00",
}

LEGACY_SUCCESS = {
    "task_id": "app.tools.legacy#ghi",
    "failed_step": 1,
    "error_reason": "TypeError: bad arg",
    "retry_count": 0,
    "final_success": True,   # 老格式：无 action / resolved_action
    "timestamp": "2026-09-15T08:02:00+00:00",
}


def make_records() -> list:
    """两条成功链（含一条乱序）+ 一条纯失败 + 一条老格式成功链。"""
    second = dict(SUCCESS_CHAIN[1])
    second["failed_step"] = 0  # 故意乱序，验证排序
    return [
        second,
        dict(SUCCESS_CHAIN[0]),
        dict(PURE_FAILURE),
        dict(LEGACY_SUCCESS),
    ]
# ---------------- load_records ----------------

class TestLoadRecords:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_records(str(tmp_path / "nope.json")) == []

    def test_corrupt_file_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "broken.json"
        p.write_text("{not json", encoding="utf-8")
        assert load_records(str(p)) == []

    def test_non_list_top_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "obj.json"
        p.write_text('{"a": 1}', encoding="utf-8")
        assert load_records(str(p)) == []

    def test_filters_invalid_entries(self, tmp_path: Path) -> None:
        p = tmp_path / "mixed.json"
        p.write_text(json.dumps([{"no_task_id": 1}, "junk", SUCCESS_CHAIN[0]]), encoding="utf-8")
        records = load_records(str(p))
        assert len(records) == 1
        assert records[0]["task_id"] == SUCCESS_CHAIN[0]["task_id"]


# ---------------- group_by_task ----------------

class TestGroupByTask:
    def test_groups_and_sorts_by_step(self) -> None:
        grouped = group_by_task(make_records())
        chain = grouped["app.tools.fetch#abc"]
        # make_records 故意把第二条乱序为 failed_step=0
        assert [r["failed_step"] for r in chain] == [0, 1]


# ---------------- build_preference_pairs ----------------

class TestBuildPreferencePairs:
    def test_success_chain_produces_pair(self) -> None:
        pairs, report = build_preference_pairs([dict(r) for r in SUCCESS_CHAIN])
        assert len(pairs) == 1
        pair = pairs[0]
        assert pair.chosen == SUCCESS_CHAIN[1]["resolved_action"]
        assert pair.rejected == SUCCESS_CHAIN[0]["action"]
        assert pair.task_id == "app.tools.fetch#abc"
        assert report.n_pairs == 1
        assert report.n_success == 1
        assert report.n_skipped == 0

    def test_rejected_is_first_failed_attempt(self) -> None:
        """Rejected 取首次失败（failed_step 最小），非最后一次。"""
        chain = [
            {**SUCCESS_CHAIN[0], "failed_step": 1, "action": "FIRST_ATTEMPT"},
            {**SUCCESS_CHAIN[1], "failed_step": 2},
        ]
        pairs, _ = build_preference_pairs(chain)
        assert pairs[0].rejected == "FIRST_ATTEMPT"

    def test_pure_failure_skipped_not_fabricated(self) -> None:
        """纯失败链无验证过的 Chosen，必须跳过而非伪造。"""
        pairs, report = build_preference_pairs([dict(PURE_FAILURE)])
        assert pairs == []
        assert report.n_failure == 1
        assert report.n_skipped == 1

    def test_legacy_success_without_action_skipped(self) -> None:
        """老格式记录缺 action/resolved_action -> 跳过。"""
        pairs, report = build_preference_pairs([dict(LEGACY_SUCCESS)])
        assert pairs == []
        assert report.n_skipped == 1

    def test_empty_records(self) -> None:
        pairs, report = build_preference_pairs([])
        assert pairs == []
        assert report.n_records == 0 and report.n_tasks == 0

    def test_prompt_contains_context_sections(self) -> None:
        pairs, _ = build_preference_pairs([dict(r) for r in SUCCESS_CHAIN])
        prompt = pairs[0].prompt
        assert "函数/任务" in prompt
        assert SUCCESS_CHAIN[0]["action"] in prompt        # 失败调用
        assert "ConnectionError" in prompt                  # 错误信息
        assert "历史执行轨迹" in prompt

    def test_mixed_tasks_counts(self) -> None:
        records = make_records()
        pairs, report = build_preference_pairs(records)
        assert report.n_records == 4
        assert report.n_tasks == 3
        assert report.n_success == 2
        assert report.n_failure == 1
        assert report.n_pairs == 1          # 两条成功链中一条可产出
        assert report.n_skipped == 2        # 纯失败链 + legacy 各跳过一条


# ---------------- ShareGPT 导出 ----------------

class TestShareGPT:
    def test_format_shape(self) -> None:
        pairs, _ = build_preference_pairs([dict(r) for r in SUCCESS_CHAIN])
        data = to_sharegpt(pairs)
        assert len(data) == 1
        item = data[0]
        assert item["conversations"] == [
            {"from": "human", "value": pairs[0].prompt}
        ]
        assert item["chosen"] == {"from": "gpt", "value": pairs[0].chosen}
        assert item["rejected"] == {"from": "gpt", "value": pairs[0].rejected}

    def test_export_roundtrip(self, tmp_path: Path) -> None:
        pairs, _ = build_preference_pairs([dict(r) for r in SUCCESS_CHAIN])
        out = export_sharegpt(pairs, str(tmp_path / "sub" / "prefs.json"))
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == to_sharegpt(pairs)


# ---------------- 端到端管道 ----------------

class TestPipeline:
    def test_run_data_pipeline(self, tmp_path: Path) -> None:
        log = tmp_path / "failure_log.json"
        log.write_text(json.dumps(make_records(), ensure_ascii=False), encoding="utf-8")
        out = tmp_path / "prefs.json"

        pairs, report = run_data_pipeline(str(log), str(out))
        assert report.n_pairs == 1
        assert report.output_path == str(out)
        assert out.exists()
        assert len(json.loads(out.read_text(encoding="utf-8"))) == 1

    def test_pipeline_with_empty_log(self, tmp_path: Path) -> None:
        pairs, report = run_data_pipeline(str(tmp_path / "none.json"), str(tmp_path / "o.json"))
        assert pairs == []
        assert report.output_path == ""  # 无产出时不写输出路径


# ---------------- 训练配置 ----------------

class TestConfig:
    def test_defaults_sanity(self) -> None:
        cfg = QLoRADPOConfig()
        assert cfg.model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert cfg.load_in_4bit and cfg.bnb_quant_type == "nf4"
        assert cfg.lora_r == 32 and cfg.lora_alpha == 64
        assert cfg.dpo_beta == 0.1
        assert cfg.learning_rate == 5e-6
        assert cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps == 16
        assert cfg.bf16 and cfg.gradient_checkpointing

    def test_target_modules_cover_all_linear(self) -> None:
        targets = set(QLoRADPOConfig().lora_target_modules)
        assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= targets
        assert {"gate_proj", "up_proj", "down_proj"} <= targets
