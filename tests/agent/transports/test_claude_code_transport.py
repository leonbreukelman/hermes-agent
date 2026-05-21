"""Tests for the Claude Code CLI transport.

The transport is intentionally subscription-backed and local-process based; it
must not route through Anthropic API clients or require an API key.
"""

import builtins
import json
import os
import signal
import subprocess
from types import SimpleNamespace

import pytest

import agent.transports.claude_code as claude_code_module
from agent.transports import get_transport
from agent.transports.types import NormalizedResponse


class _FakePopen:
    def __init__(
        self,
        *,
        stdout='{ "type": "result", "result": "done", "usage": {"input_tokens": 1, "output_tokens": 2} }',
        stderr="",
        returncode=0,
        timeout_count=0,
        wait_times_out=False,
    ):
        self.pid = 4242
        self.stdout_text = stdout
        self.stderr_text = stderr
        self.returncode = returncode
        self.timeout_count = timeout_count
        self.wait_times_out = wait_times_out
        self.communicate_calls = []
        self.wait_calls = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def communicate(self, timeout=None):
        self.communicate_calls.append(timeout)
        if self.timeout_count > 0:
            self.timeout_count -= 1
            raise subprocess.TimeoutExpired(
                ["/tmp/claude"],
                timeout=timeout,
                output=self.stdout_text,
                stderr=self.stderr_text,
            )
        return self.stdout_text, self.stderr_text

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.wait_times_out:
            raise subprocess.TimeoutExpired(["/tmp/claude"], timeout=timeout)
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1


class _FakeStreamPipe:
    def __init__(self, lines):
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return "".join(self._lines)


class _FakeStreamPopen:
    def __init__(self, *, stdout_lines, stderr_lines=None, returncode=0):
        self.pid = 4343
        self.stdout = _FakeStreamPipe(stdout_lines)
        self.stderr = _FakeStreamPipe(stderr_lines or [])
        self.returncode = returncode
        self.wait_calls = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1


def test_claude_code_transport_is_registered():
    transport = get_transport("claude_code")

    assert transport is not None
    assert transport.api_mode == "claude_code"


def test_build_kwargs_creates_safe_noninteractive_cli_command():
    transport = get_transport("claude_code")
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Say hello."},
    ]

    kwargs = transport.build_kwargs(
        model="opus",
        messages=messages,
        tools=[{"type": "function", "function": {"name": "unsafe"}}],
        cli_path="/tmp/claude",
        timeout=12,
    )

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
        "You are concise.",
        "--tools",
        "",
    ]
    assert kwargs["timeout"] == 12
    assert kwargs["prompt"] == "Say hello."
    assert kwargs["system_prompt"] == "You are concise."


def test_build_kwargs_includes_provider_mode_isolation_flags():
    transport = get_transport("claude_code")

    kwargs = transport.build_kwargs(
        model="sonnet",
        messages=[{"role": "user", "content": "Say hello."}],
        tools=[{"type": "function", "function": {"name": "unsafe"}}],
        cli_path="/tmp/claude",
        timeout=12,
    )

    command = kwargs["command"]
    assert command.count("--tools") == 1
    assert command[command.index("--tools") + 1] == ""
    assert command.count("--disable-slash-commands") == 1
    assert command.count("--strict-mcp-config") == 1
    assert command.count("--mcp-config") == 1
    mcp_config = command[command.index("--mcp-config") + 1]
    assert json.loads(mcp_config) == {"mcpServers": {}}
    assert "--bare" not in command


def test_no_env_or_config_can_inject_bare_or_user_mcp_config(monkeypatch):
    transport = get_transport("claude_code")
    monkeypatch.setenv(
        "HERMES_CLAUDE_CODE_ARGS",
        "--bare --mcp-config /tmp/user.json --settings /tmp/settings.json --setting-sources user --tools default",
    )

    kwargs = transport.build_kwargs(
        model="haiku",
        messages=[{"role": "user", "content": "Say hello."}],
        cli_path="/tmp/claude",
        timeout=12,
    )

    command = kwargs["command"]
    assert "--bare" not in command
    assert "/tmp/user.json" not in command
    assert "--settings" not in command
    assert "--setting-sources" not in command
    assert command.count("--mcp-config") == 1
    assert json.loads(command[command.index("--mcp-config") + 1]) == {"mcpServers": {}}
    assert command.count("--tools") == 1
    assert command[command.index("--tools") + 1] == ""


def test_cli_path_resolution_order(monkeypatch):
    transport = get_transport("claude_code")
    default_path = claude_code_module._DEFAULT_CLAUDE_CLI_PATH

    monkeypatch.delenv("HERMES_CLAUDE_CODE_COMMAND", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_CLI_PATH", raising=False)
    monkeypatch.setattr(claude_code_module.os.path, "exists", lambda path: path == default_path)
    monkeypatch.setattr(
        claude_code_module.shutil,
        "which",
        lambda command: "/usr/bin/claude" if command == "claude" else None,
    )

    assert transport._resolve_cli_path("/explicit/claude") == "/explicit/claude"

    monkeypatch.setenv("HERMES_CLAUDE_CODE_COMMAND", "/env/hermes-claude")
    monkeypatch.setenv("CLAUDE_CODE_CLI_PATH", "/env/legacy-claude")
    assert transport._resolve_cli_path() == "/env/hermes-claude"

    monkeypatch.delenv("HERMES_CLAUDE_CODE_COMMAND", raising=False)
    assert transport._resolve_cli_path() == "/env/legacy-claude"

    monkeypatch.delenv("CLAUDE_CODE_CLI_PATH", raising=False)
    assert transport._resolve_cli_path() == "/usr/bin/claude"

    monkeypatch.setattr(claude_code_module.shutil, "which", lambda command: None)
    assert transport._resolve_cli_path() == default_path

    monkeypatch.setattr(claude_code_module.os.path, "exists", lambda path: False)
    assert transport._resolve_cli_path() == "claude"


def test_build_kwargs_flattens_recent_conversation_for_cli_prompt():
    transport = get_transport("claude_code")
    messages = [
        {"role": "system", "content": "System rules."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow up"},
    ]

    kwargs = transport.build_kwargs(model="sonnet", messages=messages)

    assert kwargs["system_prompt"] == "System rules."
    assert "User: First question" in kwargs["prompt"]
    assert "Assistant: First answer" in kwargs["prompt"]
    assert kwargs["prompt"].endswith("User: Follow up")


def test_normalize_response_parses_json_result_and_usage():
    transport = get_transport("claude_code")

    response = transport.normalize_response(
        {
            "type": "result",
            "subtype": "success",
            "result": "Hello from Claude Code.",
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }
    )

    assert isinstance(response, NormalizedResponse)
    assert response.content == "Hello from Claude Code."
    assert response.tool_calls is None
    assert response.finish_reason == "stop"
    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 4
    assert response.usage.total_tokens == 7


def test_run_raises_clear_error_on_nonzero_exit():
    transport = get_transport("claude_code")

    def fake_runner(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="Claude Code quota exhausted")

    with pytest.raises(RuntimeError, match="Claude Code CLI failed.*quota exhausted"):
        transport.run(
            {
                "command": ["/tmp/claude", "-p", "hi", "--output-format", "json"],
                "timeout": 5,
            },
            runner=fake_runner,
        )


def test_run_raises_actionable_redacted_error_when_cli_binary_is_missing():
    transport = get_transport("claude_code")
    secret = "sk-ant-" + "a" * 30

    def fake_runner(*args, **kwargs):
        raise FileNotFoundError(f"No such file or directory: '/tmp/{secret}/claude'")

    with pytest.raises(RuntimeError) as exc_info:
        transport.run(
            {
                "command": [f"/tmp/{secret}/claude", "-p", "hi", "--output-format", "json"],
                "timeout": 5,
            },
            runner=fake_runner,
        )

    message = str(exc_info.value)
    assert "Claude Code CLI failed to start" in message
    assert "Install Claude Code" in message
    assert "HERMES_CLAUDE_CODE_COMMAND" in message
    assert secret not in message
    assert "[REDACTED]" in message


def test_run_redacts_credential_like_values_from_nonzero_exit():
    transport = get_transport("claude_code")
    secret = "sk-ant-" + "b" * 30

    def fake_runner(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr=f"Authentication failed for token {secret}")

    with pytest.raises(RuntimeError) as exc_info:
        transport.run(
            {
                "command": ["/tmp/claude", "-p", "hi", "--output-format", "json"],
                "timeout": 5,
            },
            runner=fake_runner,
        )

    message = str(exc_info.value)
    assert "Claude Code CLI failed" in message
    assert secret not in message
    assert "[REDACTED]" in message


def test_run_raises_clear_redacted_error_on_invalid_json_stdout():
    transport = get_transport("claude_code")
    secret = "sk-ant-" + "c" * 30

    def fake_runner(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=f"not json {secret}", stderr="")

    with pytest.raises(RuntimeError) as exc_info:
        transport.run(
            {
                "command": ["/tmp/claude", "-p", "hi", "--output-format", "json"],
                "timeout": 5,
            },
            runner=fake_runner,
        )

    message = str(exc_info.value)
    assert "Claude Code CLI returned invalid JSON" in message
    assert secret not in message
    assert "[REDACTED]" in message


def test_run_raises_when_valid_json_lacks_usable_assistant_content():
    transport = get_transport("claude_code")

    def fake_runner(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout='{ "type": "result", "subtype": "success", "usage": {"input_tokens": 1, "output_tokens": 0} }',
            stderr="",
        )

    with pytest.raises(RuntimeError, match="did not include usable assistant content"):
        transport.run(
            {
                "command": ["/tmp/claude", "-p", "hi", "--output-format", "json"],
                "timeout": 5,
            },
            runner=fake_runner,
        )


def test_run_raises_clear_error_on_timeout():
    transport = get_transport("claude_code")

    def fake_runner(command, **kwargs):
        raise subprocess.TimeoutExpired(command, timeout=kwargs.get("timeout"))

    with pytest.raises(RuntimeError, match="Claude Code CLI timed out after 5s"):
        transport.run(
            {
                "command": ["/tmp/claude", "-p", "hi", "--output-format", "json"],
                "timeout": 5,
            },
            runner=fake_runner,
        )


def test_run_invokes_runner_and_parses_json_stdout():
    transport = get_transport("claude_code")
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout='{ "type": "result", "result": "done", "usage": {"input_tokens": 1, "output_tokens": 2} }',
            stderr="",
        )

    response = transport.run(
        {
            "command": ["/tmp/claude", "-p", "hi", "--output-format", "json"],
            "timeout": 5,
        },
        runner=fake_runner,
    )

    assert calls == [
        (
            ["/tmp/claude", "-p", "hi", "--output-format", "json"],
            {"capture_output": True, "text": True, "timeout": 5},
        )
    ]
    assert response.content == "done"
    assert response.usage.total_tokens == 3


def test_timeout_terminates_claude_child_process(monkeypatch):
    transport = get_transport("claude_code")
    process = _FakePopen(
        stdout="token sk-ant-" + "d" * 30,
        stderr="api_key=" + "e" * 30,
        timeout_count=2,
        wait_times_out=True,
    )
    popen_calls = []
    killpg_calls = []

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return process

    monkeypatch.setattr(claude_code_module.os, "getpgid", lambda pid: 9001)
    monkeypatch.setattr(
        claude_code_module.os,
        "killpg",
        lambda pgid, sig: killpg_calls.append((pgid, sig)),
    )

    with pytest.raises(RuntimeError) as exc_info:
        transport.run(
            {
                "command": ["/tmp/claude", "-p", "hi", "--output-format", "json"],
                "timeout": 0.01,
            },
            popen_factory=fake_popen,
            sleep=lambda _: None,
        )

    command, kwargs = popen_calls[0]
    assert command == ["/tmp/claude", "-p", "hi", "--output-format", "json"]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.PIPE
    assert kwargs["text"] is True
    if os.name == "posix":
        assert kwargs["start_new_session"] is True
        assert killpg_calls == [(9001, signal.SIGTERM), (9001, signal.SIGKILL)]
    else:
        assert process.terminate_calls == 1
        assert process.kill_calls == 1
    assert process.wait_calls
    message = str(exc_info.value)
    assert "Claude Code CLI timed out" in message
    assert "child process terminated" in message
    assert "sk-ant-" not in message
    assert "eeee" not in message
    assert "[REDACTED]" in message
    assert len(message) < 1200


def test_force_kill_falls_back_when_sigkill_is_unavailable(monkeypatch):
    transport = get_transport("claude_code")
    assert transport is not None
    process = _FakePopen(wait_times_out=True)
    killpg_calls = []

    monkeypatch.setattr(claude_code_module.os, "getpgid", lambda pid: 9003)
    monkeypatch.setattr(
        claude_code_module.os,
        "killpg",
        lambda pgid, sig: killpg_calls.append((pgid, sig)),
    )
    monkeypatch.delattr(claude_code_module.signal, "SIGKILL", raising=False)

    transport._terminate_child_process(process, force_kill=True)

    assert killpg_calls == [(9003, signal.SIGTERM)]
    assert process.kill_calls == 1


def test_cancellation_terminates_claude_child_process(monkeypatch):
    transport = get_transport("claude_code")
    process = _FakePopen(timeout_count=3, wait_times_out=False)
    killpg_calls = []
    checks = []

    def fake_popen(command, **kwargs):
        return process

    def cancel_check():
        checks.append(True)
        return len(checks) >= 1

    monkeypatch.setattr(claude_code_module.os, "getpgid", lambda pid: 9002)
    monkeypatch.setattr(
        claude_code_module.os,
        "killpg",
        lambda pgid, sig: killpg_calls.append((pgid, sig)),
    )

    with pytest.raises(InterruptedError, match="Claude Code CLI call cancelled"):
        transport.run(
            {
                "command": ["/tmp/claude", "-p", "hi", "--output-format", "json"],
                "timeout": 30,
            },
            popen_factory=fake_popen,
            cancel_check=cancel_check,
            sleep=lambda _: None,
        )

    assert checks
    assert process.wait_calls
    if os.name == "posix":
        assert killpg_calls == [(9002, signal.SIGTERM)]
    else:
        assert process.terminate_calls == 1


def test_subprocess_runs_with_stdin_devnull():
    transport = get_transport("claude_code")
    popen_calls = []

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return _FakePopen()

    response = transport.run(
        {
            "command": ["/tmp/claude", "-p", "hi", "--output-format", "json"],
            "timeout": 5,
        },
        popen_factory=fake_popen,
    )

    assert response.content == "done"
    _, kwargs = popen_calls[0]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.PIPE
    assert kwargs["text"] is True


def test_subprocess_does_not_inherit_anthropic_credentials(monkeypatch):
    transport = get_transport("claude_code")
    secret = "sk-ant-" + "f" * 30
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "auth-token-should-not-pass")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.invalid")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-token-should-not-pass")
    popen_calls = []

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return _FakePopen()

    transport.run(
        {
            "command": ["/tmp/claude", "-p", "hi", "--output-format", "json"],
            "timeout": 5,
        },
        popen_factory=fake_popen,
    )

    child_env = popen_calls[0][1]["env"]
    assert child_env.get("HOME")
    assert child_env.get("PATH")
    assert "ANTHROPIC_API_KEY" not in child_env
    assert "ANTHROPIC_AUTH_TOKEN" not in child_env
    assert "ANTHROPIC_BASE_URL" not in child_env
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in child_env
    assert secret not in child_env.values()


def test_child_env_preserves_windows_bootstrap_vars_without_anthropic_credentials(monkeypatch):
    monkeypatch.setenv("SystemRoot", r"C:\\Windows")
    monkeypatch.setenv("SystemDrive", "C:")
    monkeypatch.setenv("WINDIR", r"C:\\Windows")
    monkeypatch.setenv("COMSPEC", r"C:\\Windows\\System32\\cmd.exe")
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    monkeypatch.setenv("TEMP", r"C:\\Temp")
    monkeypatch.setenv("TMP", r"C:\\Tmp")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "w" * 30)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-token-should-not-pass")

    child_env = claude_code_module.ClaudeCodeTransport._build_child_env()

    assert child_env["SystemRoot"] == r"C:\\Windows"
    assert child_env["SystemDrive"] == "C:"
    assert child_env["WINDIR"] == r"C:\\Windows"
    assert child_env["COMSPEC"].endswith("cmd.exe")
    assert child_env["PATHEXT"] == ".COM;.EXE;.BAT;.CMD"
    assert child_env["TEMP"] == r"C:\\Temp"
    assert child_env["TMP"] == r"C:\\Tmp"
    assert "ANTHROPIC_API_KEY" not in child_env
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in child_env


def test_transport_opens_no_credential_files(monkeypatch):
    transport = get_transport("claude_code")
    opened_paths = []
    real_open = builtins.open

    def tracking_open(path, *args, **kwargs):
        opened_paths.append(str(path))
        lowered = str(path).lower()
        assert "/.claude/" not in lowered
        assert "/.config/claude/" not in lowered
        assert "credentials" not in lowered
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    kwargs = transport.build_kwargs(
        model="sonnet",
        messages=[{"role": "user", "content": "hi"}],
        cli_path="/tmp/claude",
        timeout=5,
    )
    response = transport.run(kwargs, popen_factory=lambda command, **kw: _FakePopen())

    assert response.content == "done"
    assert opened_paths == []


def test_error_json_subtype_raises_redacted_transport_error():
    transport = get_transport("claude_code")
    secret = "sk-ant-" + "g" * 30
    opaque_key = "opaque-provider-key-value-123456789"

    def fake_runner(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '{'
                '"type":"result",'
                '"subtype":"error_max_turns",'
                f'"result":"stopped after max turns with token {secret}",'
                f'"api_key":"{opaque_key}"'
                '}'
            ),
            stderr="",
        )

    with pytest.raises(RuntimeError) as exc_info:
        transport.run(
            {
                "command": ["/tmp/claude", "-p", "hi", "--output-format", "json"],
                "timeout": 5,
            },
            runner=fake_runner,
        )

    message = str(exc_info.value)
    assert "Claude Code CLI returned error result" in message
    assert "error_max_turns" in message
    assert secret not in message
    assert opaque_key not in message
    assert "[REDACTED]" in message
    assert len(message) < 1000


def test_is_error_json_result_raises_redacted_transport_error():
    transport = get_transport("claude_code")
    secret = "ghp_" + "h" * 30

    def fake_runner(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '{'
                '"type":"result",'
                '"subtype":"error_during_execution",'
                '"is_error":true,'
                f'"result":"provider failed with authorization {secret}"'
                '}'
            ),
            stderr="",
        )

    with pytest.raises(RuntimeError) as exc_info:
        transport.run(
            {
                "command": ["/tmp/claude", "-p", "hi", "--output-format", "json"],
                "timeout": 5,
            },
            runner=fake_runner,
        )

    message = str(exc_info.value)
    assert "Claude Code CLI returned error result" in message
    assert "error_during_execution" in message
    assert secret not in message
    assert "[REDACTED]" in message


def test_invalid_timeout_env_var_produces_clear_error(monkeypatch):
    transport = get_transport("claude_code")
    secret = "sk-ant-" + "i" * 30
    monkeypatch.setenv("HERMES_CLAUDE_CODE_TIMEOUT_SECONDS", f"oops {secret}")

    with pytest.raises(RuntimeError) as exc_info:
        transport.build_kwargs(
            model="sonnet",
            messages=[{"role": "user", "content": "hi"}],
            cli_path="/tmp/claude",
        )

    message = str(exc_info.value)
    assert "Invalid HERMES_CLAUDE_CODE_TIMEOUT_SECONDS" in message
    assert "seconds" in message
    assert secret not in message
    assert "[REDACTED]" in message


def test_error_previews_are_bounded_and_redacted():
    transport = get_transport("claude_code")
    secret = "sk-ant-" + "j" * 30
    huge_stderr = f"Authentication failed for token {secret} " + ("x" * 5000)

    def fake_runner(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr=huge_stderr)

    with pytest.raises(RuntimeError) as exc_info:
        transport.run(
            {
                "command": ["/tmp/claude", "-p", "hi", "--output-format", "json"],
                "timeout": 5,
            },
            runner=fake_runner,
        )

    message = str(exc_info.value)
    assert "Claude Code CLI failed" in message
    assert secret not in message
    assert "[REDACTED]" in message
    assert "[truncated]" in message
    assert len(message) < 700


def test_provider_data_in_successful_response_is_redacted_and_bounded():
    transport = get_transport("claude_code")
    secret = "sk-ant-" + "k" * 30
    opaque_key = "opaque-provider-key-value-987654321"
    camel_token = "opaque-access-token-value-123456789"
    huge_payload = "z" * 5000

    def fake_runner(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '{'
                '"type":"result",'
                '"subtype":"success",'
                '"result":"ok",'
                '"session_id":"session-safe",'
                f'"api_key":"{opaque_key}",'
                f'"accessToken":"{camel_token}",'
                f'"modelUsage":{{"claude-sonnet":{{"credential":"{secret}","details":"{huge_payload}"}}}}'
                '}'
            ),
            stderr="",
        )

    response = transport.run(
        {
            "command": ["/tmp/claude", "-p", "hi", "--output-format", "json"],
            "timeout": 5,
        },
        runner=fake_runner,
    )

    assert response.content == "ok"
    provider_data_text = str(response.provider_data)
    assert response.provider_data["session_id"] == "session-safe"
    assert secret not in provider_data_text
    assert opaque_key not in provider_data_text
    assert camel_token not in provider_data_text
    assert "[REDACTED]" in provider_data_text
    assert len(provider_data_text) < 2500


def test_stream_json_parser_emits_assistant_deltas_only():
    secret = "sk-ant-" + "m" * 30
    lines = [
        json.dumps({"type": "system", "message": f"startup token {secret}"}) + "\n",
        json.dumps(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hel"},
                },
            }
        ) + "\n",
        json.dumps(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": f"lo {secret}"},
                },
            }
        ) + "\n",
        json.dumps(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "input_json_delta", "partial_json": "{\"cmd\":"},
                },
            }
        ) + "\n",
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "result": "Hello final",
                "session_id": "safe-session",
                "accessToken": secret,
            }
        ) + "\n",
    ]

    events = list(claude_code_module.parse_stream_events(lines))

    assert [(event.kind, event.text) for event in events] == [
        ("delta", "Hel"),
        ("delta", "lo [REDACTED]"),
        ("result", "Hello final"),
    ]
    result_event = events[-1]
    assert result_event.data["session_id"] == "safe-session"
    assert result_event.data["accessToken"] == "[REDACTED]"
    assert secret not in str(events)


def test_stream_json_parser_rejects_malformed_lines_redacted():
    secret = "sk-ant-" + "n" * 30

    with pytest.raises(RuntimeError) as exc_info:
        list(claude_code_module.parse_stream_events([f'{{"type":"stream_event","token":"{secret}"\n']))

    message = str(exc_info.value)
    assert "malformed stream-json" in message
    assert secret not in message
    assert "[REDACTED]" in message
    assert len(message) < 800


def test_run_stream_invokes_stream_json_and_emits_deltas():
    transport = get_transport("claude_code")
    secret = "ghp_" + "p" * 30
    stdout_lines = [
        json.dumps(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "OK-"},
                },
            }
        ) + "\n",
        json.dumps(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": f"STREAM {secret}"},
                },
            }
        ) + "\n",
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "result": "OK-STREAM final",
                "session_id": "safe-stream-session",
                "api_key": secret,
                "usage": {"input_tokens": 2, "output_tokens": 3},
            }
        ) + "\n",
    ]
    process = _FakeStreamPopen(stdout_lines=stdout_lines)
    popen_calls = []
    deltas = []

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return process

    response = transport.run_stream(
        {
            "command": [
                "/tmp/claude",
                "-p",
                "hi",
                "--output-format",
                "json",
                "--tools",
                "",
            ],
            "timeout": 5,
        },
        popen_factory=fake_popen,
        on_text_delta=deltas.append,
        sleep=lambda _: None,
    )

    command, kwargs = popen_calls[0]
    assert command[command.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in command
    assert "--include-partial-messages" in command
    assert command[command.index("--tools") + 1] == ""
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.PIPE
    assert kwargs["text"] is True
    assert deltas == ["OK-", "STREAM [REDACTED]"]
    assert response.content == "OK-STREAM final"
    assert response.usage.total_tokens == 5
    assert response.provider_data["session_id"] == "safe-stream-session"
    assert secret not in str(response.provider_data)


def test_run_stream_requires_final_result_event():
    transport = get_transport("claude_code")
    process = _FakeStreamPopen(
        stdout_lines=[
            json.dumps(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "partial"},
                    },
                }
            ) + "\n"
        ]
    )

    with pytest.raises(RuntimeError, match="did not emit a final result"):
        transport.run_stream(
            {
                "command": ["/tmp/claude", "-p", "hi", "--output-format", "json"],
                "timeout": 5,
            },
            popen_factory=lambda command, **kwargs: process,
            sleep=lambda _: None,
        )
