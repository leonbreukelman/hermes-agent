import sys
import types

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent
from agent.transports.types import NormalizedResponse


def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


def test_claude_code_agent_initializes_without_openai_or_anthropic_client(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.setenv("HERMES_CLAUDE_CODE_COMMAND", "/tmp/claude")

    agent = run_agent.AIAgent(
        model="opus",
        provider="claude-code",
        api_mode="claude_code",
        base_url="claude-code://local",
        api_key="",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )

    assert agent.api_mode == "claude_code"
    assert agent.provider == "claude-code"
    assert agent.client is None
    assert agent._anthropic_client is None

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Say hello."},
    ])

    assert kwargs["command"] == [
        "/tmp/claude",
        "-p",
        "Say hello.",
        "--model",
        "opus",
        "--output-format",
        "json",
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--system-prompt",
        "Be concise.",
        "--tools",
        "",
    ]


def test_claude_code_build_api_kwargs_raises_when_transport_missing(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="opus",
        provider="claude-code",
        api_mode="claude_code",
        base_url="claude-code://local",
        api_key="",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    monkeypatch.setattr(agent, "_get_transport", lambda api_mode=None: None)

    with pytest.raises(RuntimeError, match="Claude Code transport is not registered"):
        agent._build_api_kwargs([{"role": "user", "content": "hello"}])


def test_claude_code_init_ignores_acp_args_but_preserves_command(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="opus",
        provider="claude-code",
        api_mode="claude_code",
        base_url="claude-code://local",
        api_key="",
        acp_command="/tmp/claude",
        acp_args=["--dangerous-extra-arg"],
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )

    assert agent.acp_command == "/tmp/claude"
    assert agent.acp_args == []
    assert agent._client_kwargs["command"] == "/tmp/claude"
    assert agent._client_kwargs["args"] == []
    kwargs = agent._build_api_kwargs([{"role": "user", "content": "hello"}])
    assert "--dangerous-extra-arg" not in kwargs["command"]


def test_claude_code_agent_calls_transport_not_sdk_clients(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)

    agent = run_agent.AIAgent(
        model="sonnet",
        provider="claude-code",
        api_mode="claude_code",
        base_url="claude-code://local",
        api_key="ignored-api-key",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )

    calls = []

    class FakeClaudeCodeTransport:
        def run(self, api_kwargs, *, cancel_check=None):
            assert callable(cancel_check)
            assert cancel_check() is False
            calls.append(api_kwargs)
            return NormalizedResponse(
                content="OK-HERMES-CLAUDE-CODE",
                tool_calls=None,
                finish_reason="stop",
            )

    monkeypatch.setattr(agent, "_get_transport", lambda api_mode=None: FakeClaudeCodeTransport())

    response = agent._interruptible_api_call({"command": ["/tmp/claude"], "timeout": 1})

    assert calls == [{"command": ["/tmp/claude"], "timeout": 1}]
    assert response.content == "OK-HERMES-CLAUDE-CODE"
    assert agent.client is None
    assert agent._anthropic_client is None
    assert agent.api_key == ""


def test_claude_code_streaming_entrypoint_uses_transport_stream_json(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)

    agent = run_agent.AIAgent(
        model="sonnet",
        provider="claude-code",
        api_mode="claude_code",
        base_url="claude-code://local",
        api_key="",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    calls = []
    first_delta_calls = []
    streamed = []

    class FakeClaudeCodeTransport:
        def run_stream(self, api_kwargs, *, cancel_check=None, on_text_delta=None):
            assert callable(cancel_check)
            assert cancel_check() is False
            calls.append(api_kwargs)
            if on_text_delta:
                on_text_delta("OK-")
                on_text_delta("HERMES-CLAUDE-CODE")
            return NormalizedResponse(
                content="OK-HERMES-CLAUDE-CODE",
                tool_calls=None,
                finish_reason="stop",
            )

    def fail_openai_client(*args, **kwargs):
        raise AssertionError("claude-code streaming path must not create OpenAI clients")

    monkeypatch.setattr(agent, "_get_transport", lambda api_mode=None: FakeClaudeCodeTransport())
    monkeypatch.setattr(agent, "_create_request_openai_client", fail_openai_client)
    agent.stream_delta_callback = streamed.append

    response = agent._interruptible_streaming_api_call(
        {"command": ["/tmp/claude"], "timeout": 1},
        on_first_delta=lambda: first_delta_calls.append(True),
    )

    assert calls == [{"command": ["/tmp/claude"], "timeout": 1}]
    assert first_delta_calls == [True]
    assert streamed == ["OK-", "HERMES-CLAUDE-CODE"]
    assert response.content == "OK-HERMES-CLAUDE-CODE"


def test_claude_code_run_conversation_uses_stream_json_subprocess_path(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.setenv("HERMES_CLAUDE_CODE_COMMAND", "/tmp/claude")

    agent = run_agent.AIAgent(
        model="haiku",
        provider="claude-code",
        api_mode="claude_code",
        base_url="claude-code://local",
        api_key="",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    streaming_calls = []

    def fail_non_streaming(*args, **kwargs):
        raise AssertionError("claude-code should use stream-json path after Phase 5")

    def fake_streaming(api_kwargs, **kwargs):
        streaming_calls.append(api_kwargs)
        return NormalizedResponse(
            content="OK-HERMES-CLAUDE-CODE",
            tool_calls=None,
            finish_reason="stop",
        )

    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", fake_streaming)
    monkeypatch.setattr(agent, "_interruptible_api_call", fail_non_streaming)

    result = agent.run_conversation("Reply with OK")

    assert streaming_calls
    assert streaming_calls[0]["command"][:2] == ["/tmp/claude", "-p"]
    assert result["final_response"] == "OK-HERMES-CLAUDE-CODE"


def test_aiagent_interrupt_passes_cancel_check_to_claude_code_transport(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)

    agent = run_agent.AIAgent(
        model="sonnet",
        provider="claude-code",
        api_mode="claude_code",
        base_url="claude-code://local",
        api_key="",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._interrupt_requested = True
    seen = {"cancel_check": None}

    class FakeClaudeCodeTransport:
        def run(self, api_kwargs, *, cancel_check=None):
            seen["cancel_check"] = cancel_check
            assert callable(cancel_check)
            assert cancel_check() is True
            return NormalizedResponse(
                content="cancel check observed",
                tool_calls=None,
                finish_reason="stop",
            )

    monkeypatch.setattr(agent, "_get_transport", lambda api_mode=None: FakeClaudeCodeTransport())

    response = agent._interruptible_api_call({"command": ["/tmp/claude"], "timeout": 1})

    assert seen["cancel_check"] is not None
    assert response.content == "cancel check observed"


def test_claude_code_does_not_use_anthropic_credential_pool(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "g" * 30)

    def fail_anthropic_client(*args, **kwargs):
        raise AssertionError("claude-code must not initialize Anthropic SDK clients")

    monkeypatch.setattr(run_agent, "Anthropic", fail_anthropic_client, raising=False)
    monkeypatch.setattr(run_agent, "AsyncAnthropic", fail_anthropic_client, raising=False)

    agent = run_agent.AIAgent(
        model="opus",
        provider="claude-code",
        api_mode="claude_code",
        base_url="claude-code://local",
        api_key="sk-ant-" + "h" * 30,
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )

    assert agent.provider == "claude-code"
    assert agent.api_mode == "claude_code"
    assert agent.api_key == ""
    assert agent.client is None
    assert agent._anthropic_client is None


def test_switch_model_to_claude_code_preserves_command_metadata(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)

    agent = run_agent.AIAgent(
        model="gpt-5.5",
        provider="claude-code",
        api_mode="claude_code",
        base_url="claude-code://local",
        api_key="",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )

    agent.switch_model(
        new_model="sonnet",
        new_provider="claude-code",
        api_key="ignored-api-key",
        base_url="claude-code://local",
        api_mode="claude_code",
        acp_command="/tmp/claude",
        acp_args=["--dangerous-extra-arg"],
    )

    assert agent.provider == "claude-code"
    assert agent.api_mode == "claude_code"
    assert agent.api_key == ""
    assert agent.acp_command == "/tmp/claude"
    assert agent.acp_args == []
    assert agent._client_kwargs["command"] == "/tmp/claude"
    assert agent._client_kwargs["args"] == []
