"""FastAPI 接口层与统计工具函数的单元测试。

覆盖：
- /health 健康检查
- POST /api/v1/memory/facts 的校验与默认值补全
- bayesian_update（对数域贝叶斯更新）
- spc_control_limits / is_out_of_control（Shewhart 控制图）
- sprt_decision（序贯概率比检验）
- ndcg_at_k / mrr（检索评估指标）
"""

import math
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import main
from app.agent.self_correction import (
    is_out_of_control,
    spc_control_limits,
    sprt_decision,
)
from app.evaluation.statistical_eval import mrr, ndcg_at_k
from app.memory.conflict_resolver import bayesian_update


@pytest.fixture()
def client() -> TestClient:
    """构造测试客户端。"""
    return TestClient(main.app)


class TestHealthEndpoint:
    """健康检查测试。"""

    def test_health_returns_ok(self, client: TestClient) -> None:
        """/health 应返回 200 且 status=ok。"""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"


class TestMemoryFactEndpoint:
    """MemoryFact API 端点测试。"""

    def test_create_fact_fills_defaults(self, client: TestClient) -> None:
        """只传 content 时，服务端应补全 id/timestamp/confidence 默认值。"""
        resp = client.post("/api/v1/memory/facts", json={"content": "用户偏好 pytest"})
        assert resp.status_code == 201
        body = resp.json()
        assert body["content"] == "用户偏好 pytest"
        assert body["confidence"] == 0.5
        assert body["embedding"] is None
        assert len(body["id"]) == 36  # UUID4 字符串长度
        assert body["timestamp"].endswith("+00:00") or body["timestamp"].endswith("Z")

    def test_create_fact_invalid_confidence_rejected(self, client: TestClient) -> None:
        """confidence=1.5 应被 Pydantic 校验层拒绝，返回 422。"""
        resp = client.post(
            "/api/v1/memory/facts",
            json={"content": "测试", "confidence": 1.5},
        )
        assert resp.status_code == 422

    def test_create_fact_empty_content_rejected(self, client: TestClient) -> None:
        """空内容应返回 422。"""
        resp = client.post("/api/v1/memory/facts", json={"content": ""})
        assert resp.status_code == 422


class TestBayesianUpdate:
    """贝叶斯信念更新的数学性质测试。"""

    def test_supporting_evidence_increases_belief(self) -> None:
        """似然比 > 1 时后验必须高于先验。"""
        posterior = bayesian_update(prior=0.5, likelihood_ratio=4.0)
        assert posterior > 0.5

    def test_opposing_evidence_decreases_belief(self) -> None:
        """似然比 < 1 时后验必须低于先验。"""
        posterior = bayesian_update(prior=0.5, likelihood_ratio=0.25)
        assert posterior < 0.5

    def test_neutral_evidence_keeps_prior(self) -> None:
        """似然比 = 1（无信息证据）时后验等于先验。"""
        prior = 0.3
        assert bayesian_update(prior=prior, likelihood_ratio=1.0) == pytest.approx(prior)

    def test_prior_odds_multiplied_by_lr(self) -> None:
        """贝叶斯定理核心恒等式：posterior_odds = prior_odds * LR。"""
        prior, lr = 0.2, 9.0
        posterior = bayesian_update(prior=prior, likelihood_ratio=lr)
        expected_odds = (prior / (1 - prior)) * lr
        assert posterior / (1 - posterior) == pytest.approx(expected_odds)

    def test_extreme_prior_rejected(self) -> None:
        """先验 0 或 1（闭合信念）应被拒绝——永远保持可修正性。"""
        with pytest.raises(ValueError):
            bayesian_update(prior=0.0, likelihood_ratio=2.0)
        with pytest.raises(ValueError):
            bayesian_update(prior=1.0, likelihood_ratio=2.0)

    def test_negative_lr_rejected(self) -> None:
        """负似然比在概率上无意义。"""
        with pytest.raises(ValueError):
            bayesian_update(prior=0.5, likelihood_ratio=-1.0)


class TestSPCControlLimits:
    """Shewhart 控制图测试。"""

    def test_symmetric_limits_around_mean(self) -> None:
        """控制限应关于中心线对称。"""
        samples = [10.0, 10.2, 9.8, 10.1, 9.9]
        lcl, cl, ucl = spc_control_limits(samples)
        assert cl == pytest.approx(10.0)
        assert (ucl - cl) == pytest.approx(cl - lcl)

    def test_stable_process_within_limits(self) -> None:
        """来自同一分布的样本不应被判失控（3-sigma 内）。"""
        samples = [10.0, 10.2, 9.8, 10.1, 9.9, 10.05, 9.95]
        assert not is_out_of_control(10.02, samples)

    def test_extreme_value_is_out_of_control(self) -> None:
        """极端离群值应被判失控。"""
        samples = [10.0, 10.1, 9.9, 10.05, 9.95]  # std 极小
        assert is_out_of_control(11.0, samples)

    def test_insufficient_samples_rejected(self) -> None:
        """单样本无法估计标准差，必须拒绝。"""
        with pytest.raises(ValueError):
            spc_control_limits([1.0])

    def test_nan_sample_rejected(self) -> None:
        """NaN 样本必须被拒绝。"""
        with pytest.raises(ValueError):
            spc_control_limits([1.0, float("nan")])


class TestSPRTDecision:
    """SPRT 序贯检验测试。"""

    def test_clear_degradation_accepts_h1(self) -> None:
        """成功率显著下滑（40 次里只成功 8 次，p0=0.9, p1=0.5）应判定退化。"""
        # log LR = 8*log(0.5/0.9) + 32*log(0.5/0.1) >> 上界
        assert sprt_decision(successes=8, trials=40, p0=0.9, p1=0.5) == "accept_h1"

    def test_healthy_process_accepts_h0(self) -> None:
        """成功率维持高位（50 次成功 45 次）应判定正常。"""
        # log LR = 45*log(0.5/0.9) + 5*log(0.5/0.1) << 下界
        assert sprt_decision(successes=45, trials=50, p0=0.9, p1=0.5) == "accept_h0"

    def test_borderline_evidence_continues(self) -> None:
        """证据不足时应返回 continue（SPRT 的序贯性价值所在）。"""
        # 少量试验，似然比接近 0
        result = sprt_decision(successes=1, trials=2, p0=0.9, p1=0.5)
        assert result in ("accept_h1", "continue")

    def test_invalid_parameter_order_rejected(self) -> None:
        """p1 >= p0 违反单侧检验前提。"""
        with pytest.raises(ValueError):
            sprt_decision(successes=1, trials=2, p0=0.5, p1=0.9)

    def test_success_count_exceeding_trials_rejected(self) -> None:
        """successes > trials 在计数上不可能。"""
        with pytest.raises(ValueError):
            sprt_decision(successes=5, trials=3, p0=0.9, p1=0.5)


class TestRetrievalMetrics:
    """检索评估指标测试。"""

    def test_ndcg_perfect_ranking_is_one(self) -> None:
        """理想排序的 NDCG 应为 1。"""
        assert ndcg_at_k([3.0, 2.0, 1.0, 0.0], k=4) == pytest.approx(1.0)

    def test_ndcg_inverted_ranking_less_than_one(self) -> None:
        """倒序排列的 NDCG 应显著小于 1。"""
        score = ndcg_at_k([0.0, 1.0, 2.0, 3.0], k=4)
        assert 0.0 < score < 1.0

    def test_ndcg_zero_when_no_relevant_within_k(self) -> None:
        """前 k 名全不相关时 NDCG 为 0。"""
        assert ndcg_at_k([0.0, 0.0], k=2) == 0.0

    def test_ndcg_rejects_invalid_k(self) -> None:
        """k 越界必须拒绝。"""
        with pytest.raises(ValueError):
            ndcg_at_k([1.0, 2.0], k=3)

    def test_mrr_first_relevant_document(self) -> None:
        """第 3 位首个相关 -> MRR = 1/3。"""
        assert mrr([0, 0, 1, 1]) == pytest.approx(1.0 / 3.0)

    def test_mrr_no_relevant_document(self) -> None:
        """无相关文档 -> MRR = 0。"""
        assert mrr([0, 0, 0]) == 0.0

    def test_mrr_rejects_non_binary(self) -> None:
        """非二值标签必须拒绝（MRR 的定义前提）。"""
        with pytest.raises(ValueError):
            mrr([0.5, 1])
