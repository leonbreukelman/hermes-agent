"""Hermes-tools-as-MCP server for the codex_app_server runtime.

When the user runs `openai/*` turns through the codex app-server, codex
owns the loop and builds its own tool list. By default, that means
Hermes' richer tool surface — web search, browser automation,
delegate_task subagents, vision analysis, persistent memory, skills,
cross-session search, image generation, TTS — is unreachable.

This module exposes a curated subset of those Hermes tools to the
spawned codex subprocess via stdio MCP. Codex registers it as a normal
MCP server (per `~/.codex/config.toml [mcp_servers.hermes-tools]`) and
the user gets full Hermes capability inside a Codex turn.

Scope (what we expose):
  - web_search, web_extract              — Firecrawl, no codex equivalent
  - browser_navigate / _click / _type /  — Camofox/Browserbase automation
    _snapshot / _scroll / _back / _press /
    _get_images / _console / _vision
  - vision_analyze                       — image inspection by vision model
  - image_generate                       — image generation
  - skill_view, skills_list              — Hermes' skill library
  - text_to_speech                       — TTS
  - kanban_* (complete/block/comment/    — kanban worker + orchestrator
    heartbeat/show/list/create/            handoff (stateless: read env var,
    unblock/link)                          write ~/.hermes/kanban.db)

What we DO NOT expose:
  - terminal / shell                     — codex's own shell tool
  - read_file / write_file / patch       — codex's apply_patch + shell
  - search_files / process               — codex's shell
  - clarify                              — codex's own UX
  - delegate_task / memory /             — `_AGENT_LOOP_TOOLS` in Hermes
    session_search / todo                  (model_tools.py). They require
                                           the running AIAgent context to
                                           dispatch (mid-loop state), so a
                                           stateless MCP callback can't
                                           drive them. See the inline
                                           comment on EXPOSED_TOOLS below.

Run with: python -m agent.transports.hermes_tools_mcp_server
Spawned by: CodexAppServerSession.ensure_started() when the runtime is
            active and config opts in.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

HERMES_MCP_SERVER_NAME = "hermes-tools"


# Tools we expose. Each name MUST match a registered Hermes tool that
# `model_tools.handle_function_call()` can dispatch.
#
# What we deliberately DO NOT expose:
#   - terminal / shell / read_file / write_file / patch / search_files /
#     process — codex's built-ins cover these and approval routes through
#     codex's own UI.
#   - delegate_task / memory / session_search / todo — these are
#     `_AGENT_LOOP_TOOLS` in Hermes (model_tools.py:493). They require
#     the running AIAgent context to dispatch (mid-loop state), so a
#     stateless MCP callback can't drive them. Hermes' default runtime
#     keeps these working; the codex_app_server runtime cannot.
EXPOSED_TOOLS: tuple[str, ...] = (
    "web_search",
    "web_extract",
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_press",
    "browser_snapshot",
    "browser_scroll",
    "browser_back",
    "browser_get_images",
    "browser_console",
    "browser_vision",
    "vision_analyze",
    "image_generate",
    "skill_view",
    "skills_list",
    "text_to_speech",
    # Kanban worker handoff tools — gated on HERMES_KANBAN_TASK env var
    # (set by the kanban dispatcher when spawning a worker). Without these
    # in the callback, a worker spawned with openai_runtime=codex_app_server
    # could do the work but couldn't report completion back to the kernel,
    # making it hang until timeout. Stateless dispatch — they just read
    # the env var and write to ~/.hermes/kanban.db.
    "kanban_complete",
    "kanban_block",
    "kanban_comment",
    "kanban_heartbeat",
    "kanban_show",
    "kanban_list",
    # NOTE: kanban_create / kanban_unblock / kanban_link are orchestrator-
    # only — the kanban tool gates them on HERMES_KANBAN_TASK being unset.
    # They're exposed here for orchestrator agents running on the codex
    # runtime that need to dispatch new tasks.
    "kanban_create",
    "kanban_unblock",
    "kanban_link",
)


def _build_server() -> Any:
    """Create the low-level MCP server with Hermes tools attached.

    The low-level server API lets us return Hermes' authoritative JSON schemas
    verbatim. FastMCP inspects Python callable signatures, and a generic
    ``**kwargs`` dispatch wrapper exposes an incorrect top-level ``kwargs``
    schema to Claude Code.
    """
    try:
        from mcp import types as mcp_types
        from mcp.server.lowlevel import Server
    except ImportError as exc:  # pragma: no cover - install hint
        raise ImportError(
            f"hermes-tools MCP server requires the 'mcp' package: {exc}"
        ) from exc

    # Discover Hermes tools so dispatch works.
    from model_tools import (
        get_tool_definitions,
        handle_function_call,
    )

    server = Server(
        HERMES_MCP_SERVER_NAME,
        instructions=(
            "Hermes Agent's tool surface, exposed for use inside a Codex "
            "or Claude Code session. Use these for capabilities the host "
            "runtime's built-in toolset doesn't cover: web search/extract, "
            "browser automation, vision, image generation, skills, TTS, and "
            "kanban handoff."
        ),
    )

    # Pull authoritative Hermes tool schemas for the ones we expose, so
    # MCP clients see the same parameter docs Hermes gives the model.
    all_defs = {
        td["function"]["name"]: td["function"]
        for td in (get_tool_definitions(quiet_mode=True) or [])
        if isinstance(td, dict) and td.get("type") == "function"
    }

    tool_specs: dict[str, dict[str, Any]] = {}
    for name in EXPOSED_TOOLS:
        spec = all_defs.get(name)
        if spec is None:
            logger.debug(
                "skipping %s — not registered in this Hermes process", name
            )
            continue

        params_schema = spec.get("parameters") or {"type": "object", "properties": {}}
        if not isinstance(params_schema, dict):
            params_schema = {"type": "object", "properties": {}}
        tool_specs[name] = {
            "description": spec.get("description") or f"Hermes {name} tool",
            "input_schema": params_schema,
        }

    @server.list_tools()
    async def _list_tools() -> list[Any]:
        return [
            mcp_types.Tool(
                name=name,
                description=tool_specs[name]["description"],
                inputSchema=tool_specs[name]["input_schema"],
            )
            for name in tool_specs
        ]

    @server.call_tool(validate_input=True)
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> Any:
        if name not in tool_specs:
            return mcp_types.CallToolResult(
                content=[
                    mcp_types.TextContent(
                        type="text",
                        text=json.dumps({"error": f"unknown Hermes tool: {name}", "tool": name}),
                    )
                ],
                isError=True,
            )
        try:
            result = handle_function_call(name, arguments or {})
        except Exception as exc:
            logger.exception("tool %s raised", name)
            return mcp_types.CallToolResult(
                content=[
                    mcp_types.TextContent(
                        type="text",
                        text=json.dumps({"error": str(exc), "tool": name}),
                    )
                ],
                isError=True,
            )

        if isinstance(result, str):
            text = result
        else:
            try:
                text = json.dumps(result)
            except TypeError:
                text = str(result)
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=text)],
            isError=False,
        )

    logger.info(
        "hermes-tools MCP server registered %d/%d tools",
        len(tool_specs),
        len(EXPOSED_TOOLS),
    )
    return server


async def _run_stdio_server(server: Any) -> None:
    """Run a low-level MCP server over stdio without writing logs to stdout."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for `python -m agent.transports.hermes_tools_mcp_server`."""
    argv = argv or sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv

    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        stream=sys.stderr,  # MCP uses stdio for protocol — logs MUST go to stderr
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Quiet mode: keep Hermes' own banners off stdout (which is the MCP wire).
    os.environ.setdefault("HERMES_QUIET", "1")
    os.environ.setdefault("HERMES_REDACT_SECRETS", "true")

    try:
        server = _build_server()
    except ImportError as exc:
        sys.stderr.write(f"hermes-tools MCP server cannot start: {exc}\n")
        return 2

    # Low-level MCP uses explicit streams. Retain the legacy zero-arg run path
    # for tests that monkeypatch _build_server with a tiny fake.
    try:
        if hasattr(server, "create_initialization_options"):
            asyncio.run(_run_stdio_server(server))
        else:  # pragma: no cover - compatibility seam for simple fakes
            server.run()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logger.exception("hermes-tools MCP server crashed")
        sys.stderr.write(f"hermes-tools MCP server error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
