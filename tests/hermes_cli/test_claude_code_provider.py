"""Tests for Claude Code subscription-backed provider selection.

The provider named ``claude-code`` must stay distinct from ``anthropic`` so
``/model opus --provider claude-code`` uses the local Claude Code CLI rather
than the Anthropic Messages API.
"""

import builtins

import pytest

import agent.transports.claude_code as claude_code_module
import hermes_cli.auth as auth_module

_FAKE_ANTHROPIC_PREFIX = "sk" + "-ant-"


def _fake_anthropic_token(fill: str, length: int = 30) -> str:
    return _FAKE_ANTHROPIC_PREFIX + (fill * length)
from hermes_cli.auth import (
    get_external_process_provider_status,
    resolve_external_process_provider_credentials,
)
from hermes_cli.model_normalize import normalize_model_for_provider
from hermes_cli.model_switch import list_authenticated_providers, switch_model
from hermes_cli.models import CANONICAL_PROVIDERS, normalize_provider, provider_model_ids
from hermes_cli.providers import determine_api_mode, resolve_provider_full
from hermes_cli.runtime_provider import resolve_runtime_provider


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def test_claude_code_is_canonical_provider_not_anthropic_alias():
    assert normalize_provider("claude-code") == "claude-code"
    assert any(provider.slug == "claude-code" for provider in CANONICAL_PROVIDERS)

    pdef = resolve_provider_full("claude-code")

    assert pdef is not None
    assert pdef.id == "claude-code"
    assert pdef.transport == "claude_code"
    assert pdef.auth_type == "external_process"


def test_claude_code_shows_in_authenticated_provider_picker_when_cli_available(monkeypatch):
    default_path = claude_code_module._DEFAULT_CLAUDE_CLI_PATH

    monkeypatch.delenv("HERMES_CLAUDE_CODE_COMMAND", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_CLI_PATH", raising=False)
    monkeypatch.delenv("HERMES_CLAUDE_CODE_ARGS", raising=False)
    monkeypatch.setattr(claude_code_module.shutil, "which", lambda command: None)
    monkeypatch.setattr(auth_module.shutil, "which", lambda command: None)
    monkeypatch.setattr(
        claude_code_module.os.path,
        "exists",
        lambda path: path == default_path,
    )
    monkeypatch.setattr(
        auth_module.os.path,
        "exists",
        lambda path: path == default_path,
    )

    providers = list_authenticated_providers(max_models=8)
    claude_code = next((p for p in providers if p["slug"] == "claude-code"), None)

    assert claude_code is not None
    assert claude_code["name"] == "Claude Code"
    assert claude_code["models"] == ["opus", "sonnet", "haiku"]
    assert claude_code["source"] == "hermes"


def test_claude_code_api_mode_is_external_cli_mode():
    assert determine_api_mode("claude-code", "claude-code://local") == "claude_code"


def test_external_process_error_redaction_does_not_import_transport(monkeypatch):
    import_attempts = []
    real_import = builtins.__import__

    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "agent.transports.claude_code":
            import_attempts.append(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    text = auth_module._safe_external_process_error_text(
        "token " + _fake_anthropic_token("z"),
        limit=180,
    )

    assert import_attempts == []
    assert _FAKE_ANTHROPIC_PREFIX not in text
    assert "[REDACTED]" in text


def test_claude_code_runtime_provider_uses_external_process_without_api_key(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CODE_COMMAND", "/tmp/claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-used-by-claude-code")
    monkeypatch.setattr(
        "hermes_cli.auth.shutil.which",
        lambda command: command if command == "/tmp/claude" else None,
    )

    runtime = resolve_runtime_provider(requested="claude-code", target_model="opus")

    assert runtime["provider"] == "claude-code"
    assert runtime["api_mode"] == "claude_code"
    assert runtime["base_url"] == "claude-code://local"
    assert runtime["api_key"] == ""
    assert runtime["command"] == "/tmp/claude"
    assert runtime["requested_provider"] == "claude-code"


def test_auth_resolver_and_transport_resolver_return_same_command(monkeypatch):
    default_path = claude_code_module._DEFAULT_CLAUDE_CLI_PATH

    monkeypatch.delenv("HERMES_CLAUDE_CODE_COMMAND", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_CLI_PATH", raising=False)
    monkeypatch.delenv("HERMES_CLAUDE_CODE_ARGS", raising=False)

    def fake_which(command):
        if command in {"claude", "/usr/bin/claude"}:
            return "/usr/bin/claude"
        return None

    def fake_exists(path):
        return path in {default_path, "/usr/bin/claude"}

    monkeypatch.setattr(claude_code_module.shutil, "which", fake_which)
    monkeypatch.setattr(claude_code_module.os.path, "exists", fake_exists)
    monkeypatch.setattr(auth_module.shutil, "which", fake_which)
    monkeypatch.setattr(auth_module.os.path, "exists", fake_exists)

    transport_command = claude_code_module.ClaudeCodeTransport()._resolve_cli_path()
    creds = resolve_external_process_provider_credentials("claude-code")
    status = get_external_process_provider_status("claude-code")

    assert transport_command == "/usr/bin/claude"
    assert creds["command"] == transport_command
    assert status["command"] == transport_command
    assert status["resolved_command"] == transport_command
    assert creds["args"] == []
    assert status["args"] == []


def test_claude_code_resolvers_do_not_read_credential_files(monkeypatch):
    real_open = builtins.open
    default_path = claude_code_module._DEFAULT_CLAUDE_CLI_PATH

    def forbidden_credential_open(file, *args, **kwargs):
        path = str(file)
        if ".claude" in path or "credential" in path.lower():
            raise AssertionError(f"unexpected credential file read: {path}")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", forbidden_credential_open)
    monkeypatch.delenv("HERMES_CLAUDE_CODE_COMMAND", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_CLI_PATH", raising=False)
    monkeypatch.setattr(claude_code_module.shutil, "which", lambda command: "/usr/bin/claude")
    monkeypatch.setattr(auth_module.shutil, "which", lambda command: "/usr/bin/claude")
    monkeypatch.setattr(
        claude_code_module.os.path,
        "exists",
        lambda path: path in {default_path, "/usr/bin/claude"},
    )
    monkeypatch.setattr(
        auth_module.os.path,
        "exists",
        lambda path: path in {default_path, "/usr/bin/claude"},
    )

    assert claude_code_module.ClaudeCodeTransport()._resolve_cli_path() == "/usr/bin/claude"
    assert resolve_external_process_provider_credentials("claude-code")["command"] == "/usr/bin/claude"
    assert get_external_process_provider_status("claude-code")["command"] == "/usr/bin/claude"


def test_missing_claude_code_command_error_is_redacted_and_bounded(monkeypatch):
    secret_suffix = "sk-" + ("a" * 120)
    secret_command = f"/tmp/claude-{secret_suffix}"
    monkeypatch.setenv("HERMES_CLAUDE_CODE_COMMAND", secret_command)
    monkeypatch.setattr(auth_module.shutil, "which", lambda command: None)
    monkeypatch.setattr(auth_module.os.path, "exists", lambda path: False)

    with pytest.raises(Exception) as exc_info:
        resolve_external_process_provider_credentials("claude-code")

    message = str(exc_info.value)
    assert "HERMES_CLAUDE_CODE_COMMAND" in message
    assert "CLAUDE_CODE_CLI_PATH" in message
    assert "[REDACTED]" in message
    assert secret_suffix not in message
    assert len(message) < 700


def test_claude_code_runtime_provider_does_not_require_anthropic_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.setenv("HERMES_CLAUDE_CODE_COMMAND", "/tmp/claude")
    monkeypatch.setattr(
        "hermes_cli.auth.shutil.which",
        lambda command: command if command == "/tmp/claude" else None,
    )

    runtime = resolve_runtime_provider(requested="claude-code", target_model="sonnet")

    assert runtime["provider"] == "claude-code"
    assert runtime["api_mode"] == "claude_code"
    assert runtime["api_key"] == ""
    assert runtime["command"] == "/tmp/claude"


def test_anthropic_runtime_provider_still_uses_anthropic_messages_path(monkeypatch):
    token = _fake_anthropic_token("d")
    monkeypatch.setattr("agent.anthropic_adapter.resolve_anthropic_token", lambda: token)

    runtime = resolve_runtime_provider(requested="anthropic", target_model="claude-sonnet-4-6")

    assert runtime["provider"] == "anthropic"
    assert runtime["api_mode"] == "anthropic_messages"
    assert runtime["base_url"] == "https://api.anthropic.com"
    assert runtime["api_key"] == token


def test_claude_code_model_normalization_keeps_short_cli_selectors():
    for selector in ("opus", "sonnet", "haiku"):
        assert normalize_model_for_provider(selector, "claude-code") == selector


@pytest.mark.parametrize("selector", ["opus", "sonnet", "haiku"])
def test_model_switch_short_aliases_with_claude_code_use_cli_mode(monkeypatch, selector):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "",
            "base_url": "claude-code://local",
            "api_mode": "claude_code",
        },
    )
    monkeypatch.setattr("hermes_cli.models.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)

    result = switch_model(
        raw_input=selector,
        current_provider="openai-codex",
        current_model="gpt-5.5",
        current_base_url="https://chatgpt.com/backend-api/codex",
        current_api_key="",
        explicit_provider="claude-code",
    )

    assert result.success is True
    assert result.target_provider == "claude-code"
    assert result.new_model == selector
    assert result.base_url == "claude-code://local"
    assert result.api_mode == "claude_code"


def test_claude_code_catalog_includes_cli_aliases():
    models = provider_model_ids("claude-code")

    assert "opus" in models
    assert "sonnet" in models
    assert "haiku" in models


def test_model_switch_opus_with_claude_code_uses_cli_mode(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "",
            "base_url": "claude-code://local",
            "api_mode": "claude_code",
        },
    )
    monkeypatch.setattr("hermes_cli.models.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)

    result = switch_model(
        raw_input="opus",
        current_provider="openai-codex",
        current_model="gpt-5.5",
        current_base_url="https://chatgpt.com/backend-api/codex",
        current_api_key="",
        explicit_provider="claude-code",
    )

    assert result.success is True
    assert result.target_provider == "claude-code"
    assert result.new_model == "opus"
    assert result.base_url == "claude-code://local"
    assert result.api_mode == "claude_code"


def test_model_switch_result_preserves_claude_code_command_metadata(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "",
            "base_url": "claude-code://local",
            "api_mode": "claude_code",
            "command": "/tmp/claude",
            "args": [],
        },
    )
    monkeypatch.setattr("hermes_cli.models.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)

    result = switch_model(
        raw_input="sonnet",
        current_provider="anthropic",
        current_model="claude-sonnet-4-6",
        current_base_url="https://api.anthropic.com",
        current_api_key="***",
        explicit_provider="claude-code",
    )

    assert result.success is True
    assert result.target_provider == "claude-code"
    assert result.api_mode == "claude_code"
    assert result.api_key == ""
    assert result.acp_command == "/tmp/claude"
    assert result.acp_args == []


def test_model_switch_result_clears_command_metadata_when_leaving_claude_code(monkeypatch):
    def fake_resolve_runtime_provider(**kwargs):
        assert kwargs["requested"] == "anthropic"
        return {
            "api_key": "sk-ant-test-key",
            "base_url": "https://api.anthropic.com",
            "api_mode": "anthropic_messages",
            "command": None,
            "args": [],
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve_runtime_provider,
    )
    monkeypatch.setattr("hermes_cli.models.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)

    result = switch_model(
        raw_input="claude-sonnet-4-6",
        current_provider="claude-code",
        current_model="sonnet",
        current_base_url="claude-code://local",
        current_api_key="",
        explicit_provider="anthropic",
    )

    assert result.success is True
    assert result.target_provider == "anthropic"
    assert result.api_mode == "anthropic_messages"
    assert result.acp_command is None
    assert result.acp_args == []
