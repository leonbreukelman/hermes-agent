import json
import sys
import types

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent
from agent.transports.claude_code import (
    CLAUDE_CODE_BUILTIN_TOOLS_TO_DISABLE,
    HERMES_MCP_SERVER_NAME,
)
from agent.transports.types import NormalizedResponse

_IGNORED_KEY_VALUE = "ignored" + "-api-key"
_FAKE_ANTHROPIC_PREFIX = "sk" + "-ant-"


def _fake_anthropic_token(fill: str, length: int = 30) -> str:
    return _FAKE_ANTHROPIC_PREFIX + (fill * length)


def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


def _fake_tool_definition(name: str, parameters: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Fake {name} tool",
            "parameters": parameters or {"type": "object", "properties": {}},
        },
    }


def _disallowed_tools_csv() -> str:
    return ",".join(CLAUDE_CODE_BUILTIN_TOOLS_TO_DISABLE)


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

    command = kwargs["command"]
    assert command == [
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
        "--disallowedTools",
        _disallowed_tools_csv(),
    ]
    assert "--tools" not in command


def test_claude_code_build_api_kwargs_exposes_hermes_mcp_without_tools_mask(monkeypatch):
    web_search_params = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    monkeypatch.setattr(
        run_agent,
        "get_tool_definitions",
        lambda **kwargs: [_fake_tool_definition("web_search", web_search_params)],
    )
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    monkeypatch.setenv("HERMES_CLAUDE_CODE_COMMAND", "/tmp/claude")

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

    kwargs = agent._build_api_kwargs([{"role": "user", "content": "Search the web."}])
    command = kwargs["command"]

    assert "--tools" not in command
    assert command[command.index("--disallowedTools") + 1] == _disallowed_tools_csv()
    for builtin in (
        "Bash",
        "Read",
        "Write",
        "Edit",
        "Grep",
        "LS",
        "WebFetch",
        "LSP",
        "Skill",
        "Workflow",
        "AskUserQuestion",
        "ScheduleWakeup",
        "TaskCreate",
        "CronCreate",
    ):
        assert builtin in CLAUDE_CODE_BUILTIN_TOOLS_TO_DISABLE
    assert "ToolSearch" not in CLAUDE_CODE_BUILTIN_TOOLS_TO_DISABLE

    allowed = command[command.index("--allowedTools") + 1].split(",")
    assert allowed == [f"mcp__{HERMES_MCP_SERVER_NAME}__web_search"]

    raw_mcp_config = command[command.index("--mcp-config") + 1]
    mcp_config = json.loads(raw_mcp_config)
    assert set(mcp_config["mcpServers"]) == {HERMES_MCP_SERVER_NAME}
    assert f"mcp__{next(iter(mcp_config['mcpServers']))}__web_search" in allowed


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
        api_key=_IGNORED_KEY_VALUE,
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
    monkeypatch.setenv("ANTHROPIC_API_KEY", _fake_anthropic_token("g"))

    def fail_anthropic_client(*args, **kwargs):
        raise AssertionError("claude-code must not initialize Anthropic SDK clients")

    monkeypatch.setattr(run_agent, "Anthropic", fail_anthropic_client, raising=False)
    monkeypatch.setattr(run_agent, "AsyncAnthropic", fail_anthropic_client, raising=False)

    agent = run_agent.AIAgent(
        model="opus",
        provider="claude-code",
        api_mode="claude_code",
        base_url="claude-code://local",
        api_key=_fake_anthropic_token("h"),
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
        api_key=_IGNORED_KEY_VALUE,
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
