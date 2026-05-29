"""Claude Code CLI transport.

This transport backs provider='claude-code' / api_mode='claude_code' by
invoking the local Claude Code CLI in non-interactive print mode. It is
intentionally subscription/session backed: no Anthropic API key is read or sent.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from agent.redact import redact_sensitive_text
from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, Usage


_DEFAULT_CLAUDE_CLI_PATH = "claude"
_EMPTY_MCP_CONFIG_JSON = '{"mcpServers":{}}'
_REDACTED = "[REDACTED]"

# This transport is an external-process boundary. Claude Code stdout/stderr can
# include CLI/auth diagnostics; never surface credential-looking values verbatim
# in RuntimeError text even if global log redaction has been disabled.
_CREDENTIAL_VALUE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"sk-[A-Za-z0-9_-]{10,}|"
    r"ghp_[A-Za-z0-9]{10,}|"
    r"github_pat_[A-Za-z0-9_]{10,}|"
    r"gh[ousr]_[A-Za-z0-9]{10,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"AIza[A-Za-z0-9_-]{30,}|"
    r"hf_[A-Za-z0-9]{10,}|"
    r"pplx-[A-Za-z0-9]{10,}|"
    r"gsk_[A-Za-z0-9]{10,}|"
    r"eyJ[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_=-]{4,}){0,2}"
    r")(?![A-Za-z0-9])"
)
_SENSITIVE_LABEL_RE = re.compile(
    r"(?i)\b(token|api[_ -]?key|secret|password|credential|authorization)"
    r"(\s*[:=]\s*|\s+)"
    r"(['\"]?)"
    r"([^\s,'\"]{10,})"
    r"(['\"]?)"
)
_SENSITIVE_JSON_FIELD_RE = re.compile(
    r"(?i)((?:\"|')?(?:token|api[_ -]?key|apikey|secret|password|credential|authorization|bearer|auth[_ -]?token|access[_ -]?token|refresh[_ -]?token|id[_ -]?token)(?:\"|')?\s*:\s*)"
    r"(['\"]?)"
    r"([^,}\]\s'\"]{4,})"
    r"(['\"]?)"
)
_SENSITIVE_PROVIDER_KEYS = {
    "access_token",
    "refresh_token",
    "id_token",
    "token",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "bearer",
    "secret",
    "password",
    "credential",
    "credentials",
    "private_key",
    "client_secret",
    "key",
}
_PROVIDER_DATA_STRING_LIMIT = 500
_PROVIDER_DATA_TOTAL_LIMIT = 1800
_PROVIDER_DATA_MAX_DEPTH = 5
_PROVIDER_DATA_MAX_ITEMS = 30
_STREAM_RESULT_CONTENT_KEYS = {"result", "content", "message", "text"}
_STREAM_LINE_LIMIT = 256_000
_STREAM_ERROR_PREVIEW_LIMIT = 800


@dataclass
class ClaudeCodeStreamEvent:
    """Sanitized event emitted by the Claude Code stream-json parser."""

    kind: str
    text: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


def resolve_claude_code_cli_path(cli_path: Any = None) -> str:
    """Resolve the Claude Code CLI command without reading credential files.

    Order is explicit caller path, Hermes env override, legacy env override,
    PATH lookup, then a final bare command so downstream start/validation code
    can raise an actionable missing-binary error.
    """
    explicit = str(cli_path or "").strip()
    if explicit:
        return explicit

    env_path = os.getenv("HERMES_CLAUDE_CODE_COMMAND", "").strip()
    if env_path:
        return env_path

    legacy_env_path = os.getenv("CLAUDE_CODE_CLI_PATH", "").strip()
    if legacy_env_path:
        return legacy_env_path

    path_command = shutil.which("claude")
    if path_command:
        return path_command

    return "claude"


class ClaudeCodeTransport(ProviderTransport):
    """Subprocess transport for the local Claude Code CLI."""

    @property
    def api_mode(self) -> str:
        return "claude_code"

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, str]:
        """Flatten OpenAI-style messages into Claude Code prompt fields."""
        system_parts: list[str] = []
        conversation_parts: list[str] = []

        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user").strip().lower()
            text = self._content_to_text(msg.get("content"))
            if not text:
                continue
            if role == "system":
                system_parts.append(text)
                continue
            if role == "assistant":
                label = "Assistant"
            elif role == "tool":
                label = "Tool"
            else:
                label = "User"
            conversation_parts.append(f"{label}: {text}")

        # For the common single-turn case, pass only the user's content as the
        # prompt. This keeps `claude -p` invocations compact and matches CLI UX.
        if len(conversation_parts) == 1 and conversation_parts[0].startswith("User: "):
            prompt = conversation_parts[0][len("User: "):]
        else:
            prompt = "\n\n".join(conversation_parts)

        return {
            "prompt": prompt,
            "system_prompt": "\n\n".join(system_parts),
        }

    def convert_tools(self, tools: List[Dict[str, Any]]) -> str:
        """Disable Claude Code's built-in tools for this bridge.

        Hermes exposes a curated subset of its native tools through a strict
        stdio MCP server.  Claude Code's own filesystem/shell tools stay
        disabled so the Hermes permission/tool boundary remains authoritative.
        """
        return ""

    def _extract_supported_hermes_tool_names(self, tools: List[Dict[str, Any]]) -> list[str]:
        if not tools:
            return []
        try:
            from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        except Exception:
            return []
        exposed = set(EXPOSED_TOOLS)
        names: list[str] = []
        seen: set[str] = set()
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") if tool.get("type") == "function" else None
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name") or "").strip()
            if name and name in exposed and name not in seen:
                names.append(name)
                seen.add(name)
        return names

    def _hermes_tools_mcp_config(self) -> str:
        env: dict[str, str] = {
            "HERMES_QUIET": "1",
            "HERMES_REDACT_SECRETS": os.environ.get("HERMES_REDACT_SECRETS", "true"),
        }
        hermes_home = os.environ.get("HERMES_HOME")
        if hermes_home:
            env["HERMES_HOME"] = hermes_home
        pythonpath = os.environ.get("PYTHONPATH")
        if pythonpath:
            env["PYTHONPATH"] = pythonpath
        return json.dumps(
            {
                "mcpServers": {
                    "hermes-tools": {
                        "command": sys.executable,
                        "args": ["-m", "agent.transports.hermes_tools_mcp_server"],
                        "env": env,
                        "startup_timeout_sec": 30.0,
                        "tool_timeout_sec": 600.0,
                    }
                }
            },
            separators=(",", ":"),
        )

    def _allowed_hermes_mcp_tools(self, tools: List[Dict[str, Any]]) -> list[str]:
        return [
            f"mcp__hermes-tools__{name}"
            for name in self._extract_supported_hermes_tool_names(tools)
        ]

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params,
    ) -> Dict[str, Any]:
        extra_args = params.get("acp_args") or params.get("cli_args") or []
        if extra_args:
            raise RuntimeError(
                "Claude Code transport does not accept extra CLI/acp_args; "
                "set HERMES_CLAUDE_CODE_COMMAND or CLAUDE_CODE_CLI_PATH to choose the executable."
            )

        fields = self.convert_messages(messages)
        prompt = fields["prompt"]
        system_prompt = fields["system_prompt"]
        cli_path = self._resolve_cli_path(params.get("cli_path"))

        command = [
            cli_path,
            "-p",
            prompt,
        ]
        if model:
            command.extend(["--model", str(model)])
        command.extend(["--output-format", "json"])
        allowed_hermes_tools = self._allowed_hermes_mcp_tools(tools or [])
        mcp_config = (
            self._hermes_tools_mcp_config()
            if allowed_hermes_tools
            else _EMPTY_MCP_CONFIG_JSON
        )
        command.extend([
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--mcp-config",
            mcp_config,
        ])
        if allowed_hermes_tools:
            command.extend(["--allowedTools", ",".join(allowed_hermes_tools)])
        if system_prompt:
            command.extend(["--system-prompt", system_prompt])
        command.extend(["--tools", self.convert_tools(tools or [])])

        timeout = params.get("timeout")
        if timeout is None:
            timeout = self._default_timeout_seconds()

        return {
            "command": command,
            "timeout": timeout,
            "prompt": prompt,
            "system_prompt": system_prompt,
        }

    def run(
        self,
        api_kwargs: Dict[str, Any],
        *,
        runner: Any = None,
        popen_factory: Any = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> NormalizedResponse:
        """Invoke Claude Code and normalize its JSON stdout.

        Production uses a ``Popen`` child so Hermes can cancel and reap the
        Claude CLI process. ``runner`` is retained as a completed-process test
        seam for older unit tests; the default path never uses ``subprocess.run``.
        """
        command = api_kwargs.get("command")
        if not isinstance(command, list) or not command:
            raise RuntimeError("Claude Code CLI command is missing or invalid")

        timeout = api_kwargs.get("timeout")
        if runner is not None and popen_factory is None:
            completed = self._run_completed_process_runner(command, timeout, runner)
            stdout = getattr(completed, "stdout", "") or ""
            stderr = getattr(completed, "stderr", "") or ""
            returncode = getattr(completed, "returncode", 0)
        else:
            stdout, stderr, returncode = self._run_popen_child(
                command,
                timeout=timeout,
                popen_factory=popen_factory or subprocess.Popen,
                cancel_check=cancel_check,
                sleep=sleep,
            )

        if returncode != 0:
            detail = (stderr or stdout or f"exit code {returncode}").strip()
            detail = self._redacted_preview(detail)
            raise RuntimeError(f"Claude Code CLI failed: {detail}")

        try:
            raw = json.loads(stdout)
        except json.JSONDecodeError as exc:
            preview = self._redacted_preview(stdout.strip())
            raise RuntimeError(
                f"Claude Code CLI returned invalid JSON: {preview or '<empty stdout>'}"
            ) from exc

        self._raise_if_error_result(raw)
        normalized = self.normalize_response(raw)
        if not self.validate_response(normalized):
            raise RuntimeError("Claude Code CLI response did not include usable assistant content")
        return normalized

    def run_stream(
        self,
        api_kwargs: Dict[str, Any],
        *,
        popen_factory: Any = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        on_text_delta: Optional[Callable[[str], None]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> NormalizedResponse:
        """Invoke Claude Code in ``stream-json`` mode and normalize the final result.

        Streaming stays inside the same external-process safety boundary as
        ``run()``: stdin is closed, Claude tools remain disabled via
        ``--tools ""``, Anthropic-shaped env vars are scrubbed, and stdout
        events are parsed into sanitized Hermes deltas before callbacks fire.
        """
        command = api_kwargs.get("command")
        if not isinstance(command, list) or not command:
            raise RuntimeError("Claude Code CLI command is missing or invalid")

        stream_command = self._stream_json_command(command)
        timeout = api_kwargs.get("timeout")
        process, stderr_chunks = self._start_stream_process(
            stream_command,
            popen_factory=popen_factory or subprocess.Popen,
        )

        final_event: Optional[ClaudeCodeStreamEvent] = None
        delta_chunks: list[str] = []
        try:
            stdout_iter = self._iter_stream_stdout(
                process,
                timeout=timeout,
                cancel_check=cancel_check,
                sleep=sleep,
                stderr_chunks=stderr_chunks,
            )
            for event in parse_stream_events(stdout_iter):
                if event.kind == "delta":
                    if event.text:
                        delta_chunks.append(event.text)
                        if on_text_delta:
                            on_text_delta(event.text)
                elif event.kind == "result":
                    final_event = event
        except InterruptedError:
            raise
        except RuntimeError as exc:
            returncode = self._wait_for_stream_exit(process)
            if returncode != 0:
                detail = self._stream_error_detail(stderr_chunks, fallback=str(exc))
                raise RuntimeError(f"Claude Code CLI failed: {detail}") from exc
            raise

        returncode = self._wait_for_stream_exit(process)
        if returncode != 0:
            detail = self._stream_error_detail(stderr_chunks, fallback=f"exit code {returncode}")
            raise RuntimeError(f"Claude Code CLI failed: {detail}")

        if final_event is None:
            raise RuntimeError("Claude Code stream-json did not emit a final result")

        normalized = self.normalize_response(final_event.data)
        if not self.validate_response(normalized):
            if (
                isinstance(normalized, NormalizedResponse)
                and not normalized.content
                and not normalized.tool_calls
                and delta_chunks
            ):
                normalized.content = "".join(delta_chunks)
            if not self.validate_response(normalized):
                raise RuntimeError("Claude Code CLI response did not include usable assistant content")
        return normalized

    def _run_completed_process_runner(self, command: list[str], timeout: Any, runner: Any) -> Any:
        """Compatibility seam for tests that inject subprocess.run-like fakes."""
        try:
            return runner(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Claude Code CLI timed out after {timeout}s"
            ) from exc
        except OSError as exc:
            detail = self._redacted_preview(str(exc)).strip() or exc.__class__.__name__
            raise RuntimeError(
                "Claude Code CLI failed to start: "
                f"{detail}. Install Claude Code and set HERMES_CLAUDE_CODE_COMMAND "
                "to the Claude CLI binary if it is not on PATH."
            ) from exc

    def _run_popen_child(
        self,
        command: list[str],
        *,
        timeout: Any,
        popen_factory: Any,
        cancel_check: Optional[Callable[[], bool]],
        sleep: Callable[[float], None],
    ) -> tuple[str, str, int]:
        popen_kwargs = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "env": self._build_child_env(),
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True

        try:
            process = popen_factory(command, **popen_kwargs)
        except OSError as exc:
            detail = self._redacted_preview(str(exc)).strip() or exc.__class__.__name__
            raise RuntimeError(
                "Claude Code CLI failed to start: "
                f"{detail}. Install Claude Code and set HERMES_CLAUDE_CODE_COMMAND "
                "to the Claude CLI binary if it is not on PATH."
            ) from exc

        deadline = None
        if timeout is not None:
            try:
                deadline = time.monotonic() + float(timeout)
            except (TypeError, ValueError):
                deadline = None

        stdout = ""
        stderr = ""
        while True:
            if cancel_check is not None and cancel_check():
                self._terminate_child_process(process, force_kill=False)
                raise InterruptedError("Claude Code CLI call cancelled; child process terminated")

            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                self._terminate_child_process(process, force_kill=True)
                raise RuntimeError(
                    self._timeout_error_message(timeout, stdout=stdout, stderr=stderr)
                )

            communicate_timeout = 0.2 if remaining is None else min(0.2, max(remaining, 0.001))
            try:
                stdout, stderr = process.communicate(timeout=communicate_timeout)
                break
            except subprocess.TimeoutExpired as exc:
                stdout = self._timeout_text(getattr(exc, "output", None), stdout)
                stderr = self._timeout_text(getattr(exc, "stderr", None), stderr)
                if remaining is not None and communicate_timeout >= remaining:
                    self._terminate_child_process(process, force_kill=True)
                    raise RuntimeError(
                        self._timeout_error_message(timeout, stdout=stdout, stderr=stderr)
                    ) from exc
                if sleep is not None:
                    sleep(0.05)

        return stdout or "", stderr or "", int(getattr(process, "returncode", 0) or 0)

    @classmethod
    def _stream_json_command(cls, command: list[str]) -> list[str]:
        stream_command = list(command)
        if "--output-format" in stream_command:
            idx = stream_command.index("--output-format")
            if idx + 1 < len(stream_command):
                stream_command[idx + 1] = "stream-json"
            else:
                stream_command.append("stream-json")
        else:
            stream_command.extend(["--output-format", "stream-json"])

        if "--verbose" not in stream_command:
            stream_command.append("--verbose")
        if "--include-partial-messages" not in stream_command:
            stream_command.append("--include-partial-messages")

        if "--tools" in stream_command:
            idx = stream_command.index("--tools")
            if idx + 1 < len(stream_command):
                stream_command[idx + 1] = ""
            else:
                stream_command.append("")
        else:
            stream_command.extend(["--tools", ""])
        return stream_command

    def _start_stream_process(self, command: list[str], *, popen_factory: Any) -> tuple[Any, list[str]]:
        popen_kwargs = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "env": self._build_child_env(),
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True

        try:
            process = popen_factory(command, **popen_kwargs)
        except OSError as exc:
            detail = self._redacted_preview(str(exc)).strip() or exc.__class__.__name__
            raise RuntimeError(
                "Claude Code CLI failed to start: "
                f"{detail}. Install Claude Code and set HERMES_CLAUDE_CODE_COMMAND "
                "to the Claude CLI binary if it is not on PATH."
            ) from exc

        stderr_chunks: list[str] = []
        self._start_stream_stderr_reader(process, stderr_chunks)
        return process, stderr_chunks

    def _start_stream_stderr_reader(self, process: Any, stderr_chunks: list[str]) -> None:
        stderr_pipe = getattr(process, "stderr", None)
        if stderr_pipe is None:
            return

        def _reader() -> None:
            try:
                for chunk in stderr_pipe:
                    if chunk:
                        self._append_bounded_chunk(stderr_chunks, chunk)
            except Exception:
                try:
                    chunk = stderr_pipe.read()
                except Exception:
                    chunk = ""
                if chunk:
                    self._append_bounded_chunk(stderr_chunks, chunk)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()

    def _iter_stream_stdout(
        self,
        process: Any,
        *,
        timeout: Any,
        cancel_check: Optional[Callable[[], bool]],
        sleep: Callable[[float], None],
        stderr_chunks: list[str],
    ) -> Iterator[str]:
        stdout_pipe = getattr(process, "stdout", None)
        if stdout_pipe is None:
            return

        stdout_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        sentinel = object()
        stdout_preview_chunks: list[str] = []

        def _reader() -> None:
            try:
                for chunk in stdout_pipe:
                    stdout_queue.put(("chunk", chunk))
            except Exception as exc:
                stdout_queue.put(("error", exc))
            finally:
                stdout_queue.put(("done", sentinel))

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()

        deadline = None
        if timeout is not None:
            try:
                deadline = time.monotonic() + float(timeout)
            except (TypeError, ValueError):
                deadline = None

        while True:
            if cancel_check is not None and cancel_check():
                self._terminate_child_process(process, force_kill=False)
                raise InterruptedError("Claude Code CLI call cancelled; child process terminated")

            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                self._terminate_child_process(process, force_kill=True)
                raise RuntimeError(
                    self._timeout_error_message(
                        timeout,
                        stdout="".join(stdout_preview_chunks),
                        stderr="".join(stderr_chunks),
                    )
                )

            poll_timeout = 0.1 if remaining is None else min(0.1, max(remaining, 0.001))
            try:
                kind, payload = stdout_queue.get(timeout=poll_timeout)
            except queue.Empty:
                if sleep is not None:
                    sleep(0.01)
                continue

            if kind == "done":
                break
            if kind == "error":
                raise RuntimeError(
                    "Claude Code stream-json stdout read failed: "
                    f"{self._redacted_preview(payload, limit=300)}"
                )

            chunk = payload or ""
            self._append_bounded_chunk(stdout_preview_chunks, chunk)
            yield chunk

    def _wait_for_stream_exit(self, process: Any) -> int:
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._terminate_child_process(process, force_kill=True)
        except Exception:
            pass
        return int(getattr(process, "returncode", 0) or 0)

    @classmethod
    def _stream_error_detail(cls, stderr_chunks: list[str], *, fallback: str) -> str:
        detail = "".join(stderr_chunks).strip() or fallback
        return cls._redacted_preview(detail, limit=_STREAM_ERROR_PREVIEW_LIMIT) or "<no details>"

    @staticmethod
    def _append_bounded_chunk(chunks: list[str], chunk: Any, *, limit: int = _STREAM_ERROR_PREVIEW_LIMIT) -> None:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        text = str(chunk)
        current_len = sum(len(part) for part in chunks)
        if current_len >= limit:
            return
        remaining = limit - current_len
        if len(text) <= remaining:
            chunks.append(text)
        else:
            chunks.append(text[:remaining] + "...[truncated]")

    def _terminate_child_process(self, process: Any, *, force_kill: bool) -> None:
        """Terminate a Claude Code child, using process groups on POSIX."""
        def _send(sig: signal.Signals) -> None:
            if os.name == "posix":
                try:
                    os.killpg(os.getpgid(process.pid), sig)  # windows-footgun: ok - guarded by os.name == "posix"
                    return
                except Exception:
                    pass
            try:
                if sig == signal.SIGTERM:
                    process.terminate()
                else:
                    process.kill()
            except Exception:
                pass

        _send(signal.SIGTERM)
        try:
            process.wait(timeout=2.0)
            return
        except subprocess.TimeoutExpired:
            if not force_kill:
                return
        except Exception:
            if not force_kill:
                return

        kill_signal = getattr(signal, "SIGKILL", None)
        if kill_signal is not None:
            _send(kill_signal)
        else:
            # Windows does not expose SIGKILL.  For a forced shutdown still use
            # the subprocess API's hard-kill primitive instead of issuing a
            # second terminate/SIGTERM that may leave the CLI running.
            try:
                process.kill()
            except Exception:
                pass
        try:
            process.wait(timeout=1.0)
        except Exception:
            pass

    @classmethod
    def _timeout_error_message(cls, timeout: Any, *, stdout: Any, stderr: Any) -> str:
        stdout_preview = cls._redacted_preview(stdout)
        stderr_preview = cls._redacted_preview(stderr)
        details = []
        if stdout_preview:
            details.append(f"stdout: {stdout_preview}")
        if stderr_preview:
            details.append(f"stderr: {stderr_preview}")
        suffix = f" ({'; '.join(details)})" if details else ""
        return f"Claude Code CLI timed out after {timeout}s; child process terminated{suffix}"

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        if isinstance(response, NormalizedResponse):
            return response

        data = self._object_to_dict(response)
        content = data.get("result")
        if content is None:
            content = data.get("content") or data.get("message") or data.get("text")
        if content is not None and not isinstance(content, str):
            content = json.dumps(content)

        usage_data = data.get("usage") or {}
        usage = None
        if isinstance(usage_data, dict):
            prompt_tokens = self._int_or_zero(
                usage_data.get("input_tokens", usage_data.get("prompt_tokens", 0))
            )
            completion_tokens = self._int_or_zero(
                usage_data.get("output_tokens", usage_data.get("completion_tokens", 0))
            )
            total_tokens = self._int_or_zero(
                usage_data.get("total_tokens", prompt_tokens + completion_tokens)
            )
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cached_tokens=self._int_or_zero(usage_data.get("cache_read_input_tokens", 0)),
            )

        subtype = str(data.get("subtype") or data.get("stop_reason") or "").lower()
        finish_reason = "length" if subtype in {"max_tokens", "length"} else "stop"

        provider_data = self._sanitize_provider_data(
            {k: v for k, v in data.items() if k not in {"result", "content", "message", "text", "usage"}}
        )
        return NormalizedResponse(
            content=content or "",
            tool_calls=None,
            finish_reason=finish_reason,
            usage=usage,
            provider_data=provider_data or None,
        )

    def validate_response(self, response: Any) -> bool:
        if isinstance(response, NormalizedResponse):
            return bool(response.content or response.tool_calls is not None)
        if isinstance(response, dict):
            return any(response.get(k) for k in ("result", "content", "message", "text"))
        return response is not None

    @classmethod
    def _default_timeout_seconds(cls) -> int:
        raw_value = os.getenv("HERMES_CLAUDE_CODE_TIMEOUT_SECONDS", "1800")
        try:
            return int(raw_value)
        except (TypeError, ValueError) as exc:
            safe_value = cls._redacted_preview(raw_value, limit=120)
            raise RuntimeError(
                "Invalid HERMES_CLAUDE_CODE_TIMEOUT_SECONDS value "
                f"{safe_value!r}; expected an integer number of seconds"
            ) from exc

    @classmethod
    def _raise_if_error_result(cls, data: Any) -> None:
        if not isinstance(data, dict):
            return
        subtype = str(data.get("subtype") or "").strip().lower()
        is_error = data.get("is_error") is True or str(data.get("is_error") or "").lower() == "true"
        if not (is_error or subtype.startswith("error")):
            return

        detail = data.get("result") or data.get("error") or data.get("message") or data.get("text")
        if detail is None:
            detail = data
        safe_subtype = cls._redacted_preview(subtype or "unknown", limit=80)
        safe_detail = cls._redacted_preview(detail, limit=700) or "<no details>"
        raise RuntimeError(
            f"Claude Code CLI returned error result (subtype={safe_subtype}): {safe_detail}"
        )

    @classmethod
    def _sanitize_stream_result_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize a final stream result without truncating assistant text.

        Final ``stream-json`` result events contain both assistant content and
        provider metadata.  Provider metadata must stay redacted and bounded,
        but content fields need to reach ``normalize_response`` intact or long
        Claude Code answers are clipped to provider-data preview limits.
        """
        if not isinstance(data, dict):
            return {}

        metadata = {
            key: value
            for key, value in data.items()
            if key not in _STREAM_RESULT_CONTENT_KEYS and key != "usage"
        }
        sanitized = cls._sanitize_provider_data(metadata) or {}

        for key in _STREAM_RESULT_CONTENT_KEYS:
            if key in data:
                sanitized[key] = data[key]

        if "usage" in data:
            sanitized["usage"] = cls._sanitize_provider_value(data["usage"], key="usage")

        return sanitized

    @classmethod
    def _sanitize_provider_data(cls, value: Dict[str, Any]) -> Dict[str, Any] | None:
        if not value:
            return None
        sanitized = cls._sanitize_provider_value(value)
        if not isinstance(sanitized, dict):
            sanitized = {"value": sanitized}
        return cls._bound_provider_data(sanitized)

    @classmethod
    def _sanitize_provider_value(cls, value: Any, *, key: Any = None, depth: int = 0) -> Any:
        if key is not None and cls._is_sensitive_provider_key(key):
            return _REDACTED
        if depth >= _PROVIDER_DATA_MAX_DEPTH:
            return cls._redacted_preview(value, limit=200)
        if isinstance(value, dict):
            sanitized: Dict[str, Any] = {}
            items = list(value.items())
            for child_key, child_value in items[:_PROVIDER_DATA_MAX_ITEMS]:
                safe_key = cls._redacted_preview(child_key, limit=120)
                sanitized[safe_key] = cls._sanitize_provider_value(
                    child_value,
                    key=child_key,
                    depth=depth + 1,
                )
            if len(items) > _PROVIDER_DATA_MAX_ITEMS:
                sanitized["_truncated_keys"] = len(items) - _PROVIDER_DATA_MAX_ITEMS
            return sanitized
        if isinstance(value, (list, tuple, set)):
            items = list(value)
            sanitized_items = [
                cls._sanitize_provider_value(item, depth=depth + 1)
                for item in items[:_PROVIDER_DATA_MAX_ITEMS]
            ]
            if len(items) > _PROVIDER_DATA_MAX_ITEMS:
                sanitized_items.append({"_truncated_items": len(items) - _PROVIDER_DATA_MAX_ITEMS})
            return sanitized_items
        if isinstance(value, (str, bytes)):
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            return cls._redacted_preview(value, limit=_PROVIDER_DATA_STRING_LIMIT)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return cls._redacted_preview(value, limit=_PROVIDER_DATA_STRING_LIMIT)

    @classmethod
    def _bound_provider_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            text = json.dumps(data, default=str, sort_keys=True)
        except Exception:
            return {"_provider_data_preview": cls._redacted_preview(data, limit=_PROVIDER_DATA_TOTAL_LIMIT)}
        if len(text) <= _PROVIDER_DATA_TOTAL_LIMIT:
            return data

        keep_keys = [
            "type",
            "subtype",
            "session_id",
            "num_turns",
            "duration_ms",
            "total_cost_usd",
            "stop_reason",
            "terminal_reason",
        ]
        bounded = {key: data[key] for key in keep_keys if key in data}
        bounded["_truncated_provider_data"] = True
        bounded["_provider_data_preview"] = cls._redacted_preview(
            text,
            limit=_PROVIDER_DATA_TOTAL_LIMIT,
        )
        return bounded

    @staticmethod
    def _is_sensitive_provider_key(key: Any) -> bool:
        raw = str(key).strip()
        camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", raw)
        normalized = re.sub(r"[^a-z0-9]+", "_", camel_split.lower()).strip("_")
        compact = normalized.replace("_", "")
        if normalized in _SENSITIVE_PROVIDER_KEYS or compact in _SENSITIVE_PROVIDER_KEYS:
            return True
        if normalized.endswith(("_api_key", "_apikey", "_auth_token", "_access_token", "_refresh_token")):
            return True
        if normalized.endswith(("_secret", "_password", "_credential", "_credentials", "_private_key")):
            return True
        return False

    @staticmethod
    def _redact_error_text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value)
        if not text:
            return text
        text = _CREDENTIAL_VALUE_RE.sub(_REDACTED, text)

        def _redact_label(match: re.Match[str]) -> str:
            return f"{match.group(1)}{match.group(2)}{match.group(3)}{_REDACTED}{match.group(5)}"

        def _redact_json_field(match: re.Match[str]) -> str:
            return f"{match.group(1)}{match.group(2)}{_REDACTED}{match.group(4)}"

        text = _SENSITIVE_JSON_FIELD_RE.sub(_redact_json_field, text)
        text = _SENSITIVE_LABEL_RE.sub(_redact_label, text)
        text = redact_sensitive_text(text, force=True)
        # Global log redaction uses "***" for historical CLI/log output. This
        # provider boundary surfaces errors to users, so normalize any remaining
        # redaction placeholders to the explicit provider marker expected by the
        # Claude Code hardening contract.
        text = text.replace("***", _REDACTED)
        return text

    @classmethod
    def _redacted_preview(cls, value: Any, *, limit: int = 500) -> str:
        if value is None:
            return ""
        text = str(value)
        if len(text) > limit:
            text = f"{text[:limit]}...[truncated]"
        return cls._redact_error_text(text)

    @staticmethod
    def _timeout_text(value: Any, fallback: str) -> str:
        if value is None:
            return fallback
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="replace")
            except Exception:
                return fallback
        return str(value)

    @staticmethod
    def _build_child_env() -> Dict[str, str]:
        """Return a minimal Claude CLI environment without Anthropic credentials."""
        safe_names = {
            "HOME",
            "PATH",
            "USER",
            "LOGNAME",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TERM",
            "SHELL",
            "TMPDIR",
            "SSH_AUTH_SOCK",
            # Windows process bootstrap variables.  Without these, subprocesses
            # launched under a restricted allowlist can fail before Claude Code
            # starts (for example cmd/PATH/PATHEXT resolution and temp dirs).
            "APPDATA",
            "COMSPEC",
            "HOMEDRIVE",
            "HOMEPATH",
            "LOCALAPPDATA",
            "PATHEXT",
            "PROGRAMDATA",
            "PROGRAMFILES",
            "PROGRAMFILES(X86)",
            "SystemDrive",
            "SYSTEMDRIVE",
            "SystemRoot",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "USERPROFILE",
            "WINDIR",
            "windir",
        }
        blocked_names = {
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_TOKEN",
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_MODEL",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_CODE_API_KEY",
        }
        env: Dict[str, str] = {}
        for key, value in os.environ.items():
            if key in blocked_names or key.startswith("ANTHROPIC_"):
                continue
            if key in safe_names or key.startswith("XDG_"):
                env[key] = value
        env.setdefault("HOME", os.path.expanduser("~"))
        env.setdefault("PATH", os.environ.get("PATH") or os.defpath)
        return env

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    if part.get("type") in {"text", "input_text"} and part.get("text") is not None:
                        parts.append(str(part.get("text")))
                    elif part.get("content") is not None:
                        parts.append(str(part.get("content")))
            return "\n".join(p for p in parts if p)
        if isinstance(content, dict):
            if content.get("text") is not None:
                return str(content.get("text"))
            if content.get("content") is not None:
                return str(content.get("content"))
            return json.dumps(content)
        return str(content)

    @staticmethod
    def _resolve_cli_path(cli_path: Any = None) -> str:
        return resolve_claude_code_cli_path(cli_path)

    @staticmethod
    def _object_to_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if is_dataclass(value):
            return asdict(value)
        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump()
                if isinstance(dumped, dict):
                    return dumped
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)
        return {"result": str(value)}

    @staticmethod
    def _int_or_zero(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0


def parse_stream_events(
    lines: Iterable[Any],
    *,
    require_result: bool = True,
) -> Iterator[ClaudeCodeStreamEvent]:
    """Parse Claude Code ``stream-json`` NDJSON into sanitized events.

    The parser is intentionally allowlist-based: only assistant text deltas and
    final result objects become user-visible events. System/status/hook/MCP and
    unknown events are ignored after line-level validation/redaction.
    """
    buffer = ""
    saw_result = False
    for chunk in lines:
        if chunk is None:
            continue
        if isinstance(chunk, bytes):
            text = chunk.decode("utf-8", errors="replace")
        else:
            text = str(chunk)
        buffer += text
        if len(buffer) > _STREAM_LINE_LIMIT:
            preview = ClaudeCodeTransport._redacted_preview(buffer, limit=300)
            raise RuntimeError(f"Claude Code stream-json line exceeded size limit: {preview}")

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            event = _parse_stream_json_line(line)
            if event is None:
                continue
            if event.kind == "result":
                saw_result = True
            yield event

    if buffer.strip():
        event = _parse_stream_json_line(buffer)
        if event is not None:
            if event.kind == "result":
                saw_result = True
            yield event

    if require_result and not saw_result:
        raise RuntimeError("Claude Code stream-json did not emit a final result")


def _parse_stream_json_line(line: str) -> Optional[ClaudeCodeStreamEvent]:
    stripped = line.strip()
    if not stripped:
        return None
    if len(stripped) > _STREAM_LINE_LIMIT:
        preview = ClaudeCodeTransport._redacted_preview(stripped, limit=300)
        raise RuntimeError(f"Claude Code stream-json line exceeded size limit: {preview}")

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        preview = ClaudeCodeTransport._redacted_preview(stripped, limit=500)
        raise RuntimeError(f"Claude Code malformed stream-json line: {preview}") from exc

    if not isinstance(data, dict):
        return None

    ClaudeCodeTransport._raise_if_error_result(data)
    event_type = str(data.get("type") or "").strip()

    if event_type == "result":
        sanitized = ClaudeCodeTransport._sanitize_stream_result_data(data)
        content = _stream_event_text(data)
        return ClaudeCodeStreamEvent(
            kind="result",
            text=ClaudeCodeTransport._redacted_preview(content, limit=_PROVIDER_DATA_STRING_LIMIT),
            data=sanitized,
        )

    text_delta = _extract_stream_text_delta(data)
    if text_delta:
        return ClaudeCodeStreamEvent(
            kind="delta",
            text=ClaudeCodeTransport._redacted_preview(text_delta, limit=_PROVIDER_DATA_STRING_LIMIT),
            data={},
        )

    return None


def _extract_stream_text_delta(data: Dict[str, Any]) -> str:
    event = data.get("event") if isinstance(data.get("event"), dict) else data
    if not isinstance(event, dict):
        return ""

    event_type = str(event.get("type") or "")
    delta = event.get("delta") if isinstance(event.get("delta"), dict) else {}
    if event_type == "content_block_delta" and isinstance(delta, dict):
        if str(delta.get("type") or "") == "text_delta" and delta.get("text") is not None:
            return str(delta.get("text"))
    if event_type in {"text_delta", "response.output_text.delta"} and event.get("text") is not None:
        return str(event.get("text"))
    if event_type == "response.output_text.delta" and event.get("delta") is not None:
        return str(event.get("delta"))
    return ""


def _stream_event_text(data: Dict[str, Any]) -> str:
    for key in ("result", "content", "message", "text"):
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value)
        except Exception:
            return str(value)
    return ""


# Auto-register on import
from agent.transports import register_transport  # noqa: E402

register_transport("claude_code", ClaudeCodeTransport)
