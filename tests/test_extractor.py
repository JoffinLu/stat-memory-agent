"""记忆抽取模块的单元测试（全部离线，注入 FakeLLM）。

覆盖：Prompt 构造、容错 JSON 解析（干净 / 围栏 / 前后缀）、
逐条防御校验（缺字段 / 越界置信度 / 畸形条目）、异常路径。
"""

import math
from types import SimpleNamespace

import pytest

from app.memory.extractor import (
    PROMPT_TEMPLATE,
    FactExtractionError,
    LLMNotConfiguredError,
    SYSTEM_PROMPT,
    _clamp_confidence,
    _extract_json_array,
    extract_facts,
)
from app.memory.models import MemoryFact

VALID_JSON = (
    '[{"content": "用户偏好 pytest", "confidence": 0.9},'
    ' {"content": "项目截止 10 月", "confidence": 0.7}]'
)


class FakeLLM:
    """测试替身：模拟 LangChain 客户端的 invoke 接口。

    记录最近一次收到的消息，便于断言 Prompt 构造是否正确。
    """

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return SimpleNamespace(content=self.response)


class TestClampConfidence:
    """置信度截断的数学行为测试。"""

    @pytest.mark.parametrize("raw,expected", [(1.7, 1.0), (-0.3, 0.0), ("0.8", 0.8), (2, 1.0)])
    def test_out_of_range_or_numeric_string_clamped(self, raw, expected: float) -> None:
        """越界值截断到 [0,1]；数值字符串可转换。"""
        assert _clamp_confidence(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", [None, "abc", float("nan"), [0.5], {"c": 1}])
    def test_unfixable_values_return_none(self, raw) -> None:
        """非数值 / NaN / 容器类型不可修复，返回 None。"""
        assert _clamp_confidence(raw) is None


class TestExtractJsonArray:
    """容错 JSON 解析的三层降级测试。"""

    def test_clean_array(self) -> None:
        """干净的 JSON 数组直接解析。"""
        assert _extract_json_array(VALID_JSON) == [
            {"content": "用户偏好 pytest", "confidence": 0.9},
            {"content": "项目截止 10 月", "confidence": 0.7},
        ]

    def test_markdown_fence(self) -> None:
        """带 ```json 围栏的输出可解析（LLM 最常见违规形态）。"""
        wrapped = f"```json\n{VALID_JSON}\n```"
        assert len(_extract_json_array(wrapped)) == 2

    def test_prose_around_json(self) -> None:
        """前后缀说明文字包裹时可截取最外层数组。"""
        wrapped = f"好的，以下是提取结果：\n{VALID_JSON}\n以上就是全部事实。"
        assert len(_extract_json_array(wrapped)) == 2

    def test_object_instead_of_array_raises(self) -> None:
        """返回 JSON 对象（非数组）应报错。"""
        with pytest.raises(FactExtractionError):
            _extract_json_array('{"content": "x", "confidence": 0.5}')

    def test_no_json_at_all_raises(self) -> None:
        """完全没有 JSON 结构应报错。"""
        with pytest.raises(FactExtractionError):
            _extract_json_array("抱歉，我无法处理这个请求。")

    def test_unterminated_array_raises(self) -> None:
        """被截断的 JSON 应报错而不是静默返回部分结果。"""
        with pytest.raises(FactExtractionError):
            _extract_json_array('[{"content": "截断')


class TestExtractFacts:
    """主流程测试（注入 FakeLLM）。"""

    def test_returns_memory_fact_list(self) -> None:
        """返回类型必须是 MemoryFact 列表，字段值正确。"""
        facts = extract_facts("对话内容", llm_client=FakeLLM(VALID_JSON))
        assert len(facts) == 2
        assert all(isinstance(f, MemoryFact) for f in facts)
        assert facts[0].content == "用户偏好 pytest"
        assert facts[0].confidence == pytest.approx(0.9)

    def test_prompt_carries_system_and_dialogue(self) -> None:
        """system 消息含 JSON 硬约束，human 消息含用户模板与对话原文。"""
        fake = FakeLLM("[]")
        extract_facts("老板喜欢深绿色主题", llm_client=fake)

        assert fake.last_messages is not None
        contents = [getattr(m, "content", m.get("content", "")) for m in fake.last_messages]
        assert SYSTEM_PROMPT in contents[0]
        assert PROMPT_TEMPLATE.replace("{dialogue_history}", "老板喜欢深绿色主题") == contents[1]

    def test_missing_confidence_defaults_to_prior(self) -> None:
        """LLM 未给 confidence 时取 0.5（无信息先验）。"""
        fake = FakeLLM('[{"content": "只带 content 的条目"}]')
        facts = extract_facts("对话", llm_client=fake)
        assert facts[0].confidence == pytest.approx(0.5)

    def test_out_of_range_confidence_clamped(self) -> None:
        """LLM 输出越界置信度应被截断而不是拒绝整条。"""
        fake = FakeLLM('[{"content": "甲", "confidence": 1.7}, {"content": "乙", "confidence": -0.4}]')
        facts = extract_facts("对话", llm_client=fake)
        assert facts[0].confidence == 1.0
        assert facts[1].confidence == 0.0

    def test_extra_keys_ignored(self) -> None:
        """LLM 多输出的字段（type 等）不进入数据模型。"""
        fake = FakeLLM('[{"content": "事实", "confidence": 0.6, "type": "preference", "source": "user"}]')
        facts = extract_facts("对话", llm_client=fake)
        assert facts[0].model_dump() == {
            "id": facts[0].id,
            "content": "事实",
            "timestamp": facts[0].timestamp,
            "confidence": pytest.approx(0.6),
            "embedding": None,
        }

    def test_malformed_items_skipped_not_fatal(self) -> None:
        """单条畸形（非对象 / content 缺失 / 空内容）跳过，其余保留。"""
        fake = FakeLLM(
            '["纯字符串", {"confidence": 0.5}, {"content": ""},'
            ' {"content": "   "}, {"content": "合法条目", "confidence": "abc"},'
            ' {"content": "好条目", "confidence": 0.8}]'
        )
        facts = extract_facts("对话", llm_client=fake)
        assert len(facts) == 2
        assert facts[0].confidence == pytest.approx(0.5)  # 非数值 -> 默认先验
        assert facts[1].content == "好条目"

    def test_empty_array_returns_empty_list(self) -> None:
        """对话中无事实 -> 返回空列表而非报错。"""
        facts = extract_facts("今天天气不错", llm_client=FakeLLM("[]"))
        assert facts == []

    def test_nan_confidence_falls_back_to_prior(self) -> None:
        """NaN 置信度不可修复 -> 回退 0.5（NaN 会污染贝叶斯更新）。"""
        fake = FakeLLM('[{"content": "事实", "confidence": NaN}]')
        facts = extract_facts("对话", llm_client=fake)
        assert facts[0].confidence == pytest.approx(0.5)

    def test_object_content_attribute_accepted(self) -> None:
        """返回 .content 属性对象（LangChain 风格）与裸字符串均可处理。"""
        bare = FakeLLM(VALID_JSON)
        bare.invoke = lambda messages: VALID_JSON  # 直接返回字符串
        facts = extract_facts("对话", llm_client=bare)
        assert len(facts) == 2

    def test_non_text_response_raises(self) -> None:
        """LLM 返回非文本（如 dict）应报错。"""
        weird = FakeLLM("")
        weird.invoke = lambda messages: {"unexpected": True}
        with pytest.raises(FactExtractionError):
            extract_facts("对话", llm_client=weird)


class TestErrorPaths:
    """异常路径测试。"""

    def test_no_client_anywhere_raises(self) -> None:
        """显式传入 None 且全局未配置时必须抛异常（不可静默丢数据）。"""
        with pytest.raises(LLMNotConfiguredError):
            extract_facts("对话", llm_client=None)

    @pytest.mark.parametrize("bad", ["", "   ", None, 123])
    def test_invalid_dialogue_rejected(self, bad) -> None:
        """空 / 空白 / 非字符串对话必须拒绝。"""
        with pytest.raises(ValueError):
            extract_facts(bad, llm_client=FakeLLM("[]"))

    def test_unparseable_llm_output_raises(self) -> None:
        """LLM 输出无法解析时抛 FactExtractionError（调用方可重试）。"""
        with pytest.raises(FactExtractionError):
            extract_facts("对话", llm_client=FakeLLM("我觉得这个对话没什么好提取的"))
