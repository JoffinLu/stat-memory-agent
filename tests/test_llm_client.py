"""LLM 客户端（vLLM 接入）单元测试。全程使用本地假 vLLM 服务器，零外部依赖。"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.core import llm_client as llm_mod
from app.core.config import Settings, settings


# ---------------- 假 vLLM 服务器 ----------------

class _FakeVLLMHandler(BaseHTTPRequestHandler):
    """/health 返回 200；/v1/chat/completions 回显最后一条消息。"""

    def do_GET(self) -> None:  # noqa: N802（http.server 命名约定）
        if self.path == "/health":
            self.send_response(200)
        else:
            self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        last_content = body.get("messages", [{}])[-1].get("content", "")
        payload = {
            "choices": [{"message": {"role": "assistant", "content": f"echo:{last_content}"}}],
            "model": body.get("model", "unknown"),
        }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:  # 静默请求日志
        pass


@pytest.fixture(scope="module")
def fake_vllm():
    """模块级假 vLLM 服务器（随机端口）。"""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeVLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


# ---------------- 配置默认值 ----------------

class TestConfigDefaults:
    def test_defaults_point_to_local_vllm(self, monkeypatch) -> None:
        """清掉环境变量后，默认配置必须指向本地 vLLM。"""
        for key in list("SMA_LLM_API_KEY SMA_LLM_BASE_URL SMA_LLM_MODEL_NAME".split()):
            monkeypatch.delenv(key, raising=False)
        fresh = Settings()
        assert fresh.llm_base_url == "http://localhost:8000/v1"
        assert fresh.llm_model_name == "Qwen2.5-7B-Instruct"
        assert fresh.llm_api_key == ""  # 空 -> 离线模式

    def test_env_override_switches_to_cloud(self, monkeypatch) -> None:
        """SMA_LLM_* 环境变量可无缝切换到云 API。"""
        monkeypatch.setenv("SMA_LLM_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("SMA_LLM_MODEL_NAME", "gpt-4o-mini")
        fresh = Settings()
        assert fresh.llm_base_url == "https://api.openai.com/v1"
        assert fresh.llm_model_name == "gpt-4o-mini"


# ---------------- 离线判断 ----------------

class TestOfflineMode:
    def test_no_api_key_returns_none(self, monkeypatch) -> None:
        """key 为空 -> 离线模式（返回 None，不抛异常）。"""
        monkeypatch.setattr(settings, "llm_api_key", "")
        monkeypatch.setattr(llm_mod, "_llm_client", None)
        assert llm_mod.get_llm_client() is None

    def test_langchain_missing_degrades_to_none(self, monkeypatch) -> None:
        """key 非空但 langchain-openai 未安装 -> 降级为 None。"""
        monkeypatch.setattr(settings, "llm_api_key", "EMPTY")
        monkeypatch.setattr(llm_mod, "_llm_client", None)
        try:
            import langchain_openai  # noqa: F401
            pytest.skip("langchain-openai 已安装，跳过降级测试")
        except ImportError:
            pass
        assert llm_mod.get_llm_client() is None


# ---------------- 健康探测 ----------------

class TestPingVllm:
    def test_healthy_server(self, fake_vllm, monkeypatch) -> None:
        monkeypatch.setattr(settings, "llm_healthcheck_timeout", 2.0)
        assert llm_mod.ping_vllm(fake_vllm) is True

    def test_unreachable_server(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "llm_healthcheck_timeout", 0.5)
        # 端口 1 几乎必然拒绝连接
        assert llm_mod.ping_vllm("http://127.0.0.1:1/v1") is False

    def test_timeout_zero_skips(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "llm_healthcheck_timeout", 0.0)
        assert llm_mod.ping_vllm("http://127.0.0.1:1/v1") is False


# ---------------- HTTP 直连补全 ----------------

class TestChatCompletionViaVllm:
    def test_basic_roundtrip(self, fake_vllm, monkeypatch) -> None:
        monkeypatch.setattr(settings, "llm_base_url", fake_vllm)
        monkeypatch.setattr(settings, "llm_model_name", "Qwen2.5-7B-Instruct")
        messages = [{"role": "system", "content": "你是测试"}, {"role": "user", "content": "你好"}]
        out = llm_mod.chat_completion_via_vllm(messages)
        assert out == "echo:你好"  # 假服务器回显最后一条

    def test_langchain_message_role_mapping(self, fake_vllm, monkeypatch) -> None:
        """LangChain 消息对象（type=human/ai）映射为 user/assistant。"""
        from types import SimpleNamespace

        monkeypatch.setattr(settings, "llm_base_url", fake_vllm)
        messages = [
            SimpleNamespace(type="human", content="左边"),
            SimpleNamespace(type="ai", content="右边"),
        ]
        out = llm_mod.chat_completion_via_vllm(messages)
        assert out == "echo:右边"

    def test_connection_error_raises_runtime_error(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "llm_base_url", "http://127.0.0.1:1/v1")
        monkeypatch.setattr(settings, "llm_healthcheck_timeout", 0.2)
        with pytest.raises(RuntimeError, match="vLLM 调用失败"):
            llm_mod.chat_completion_via_vllm([{"role": "user", "content": "x"}])

    def test_malformed_response_raises(self, fake_vllm, monkeypatch) -> None:
        """响应缺 choices -> 结构异常报错（fail-closed，不返回空串）。"""
        monkeypatch.setattr(settings, "llm_base_url", fake_vllm)

        class EmptyHandler(_FakeVLLMHandler):
            def do_POST(self) -> None:  # noqa: N802
                data = json.dumps({"object": "chat.completion"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", 0), EmptyHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            monkeypatch.setattr(
                settings, "llm_base_url",
                f"http://127.0.0.1:{server.server_address[1]}/v1",
            )
            with pytest.raises(RuntimeError, match="结构异常"):
                llm_mod.chat_completion_via_vllm([{"role": "user", "content": "x"}])
        finally:
            server.shutdown()
