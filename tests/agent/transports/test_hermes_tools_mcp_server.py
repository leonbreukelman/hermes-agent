"""Tests for the hermes-tools-as-MCP server module surface.

We don't run a live MCP session in unit tests — that requires the codex
subprocess + client + an event loop. These tests pin the static
contract: the module imports, the EXPOSED_TOOLS list is sane, and the
build helper assembles a server when the SDK is present.
"""

from __future__ import annotations

import asyncio


def _fake_tool_definition(name: str, params: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Fake {name} description",
            "parameters": params,
        },
    }


def _build_test_server(monkeypatch, *, calls=None, raise_on_call: Exception | None = None):
    import model_tools
    import agent.transports.hermes_tools_mcp_server as m

    definitions = [
        _fake_tool_definition(
            "web_search",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        _fake_tool_definition(
            "skill_view",
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        ),
    ]

    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda quiet_mode=True: definitions)

    def fake_handle_function_call(name, arguments, *args, **kwargs):
        if calls is not None:
            calls.append((name, arguments))
        if raise_on_call is not None:
            raise raise_on_call
        return f"handled {name}"

    monkeypatch.setattr(model_tools, "handle_function_call", fake_handle_function_call)
    return m._build_server()


async def _listed_tools(server):
    from mcp import types as mcp_types

    result = await server.request_handlers[mcp_types.ListToolsRequest](None)
    return result.root.tools


async def _call_tool(server, name: str, arguments: dict | None):
    from mcp import types as mcp_types

    request = mcp_types.CallToolRequest(
        params=mcp_types.CallToolRequestParams(name=name, arguments=arguments)
    )
    return await server.request_handlers[mcp_types.CallToolRequest](request)




class TestModuleSurface:
    def test_module_imports_clean(self):
        from agent.transports import hermes_tools_mcp_server as m
        assert callable(m.main)
        assert callable(m._build_server)
        assert isinstance(m.EXPOSED_TOOLS, tuple)
        assert len(m.EXPOSED_TOOLS) > 0

    def test_exposed_tools_are_safe_subset(self):
        """We MUST NOT expose tools codex already has, because codex'
        own builtins are better-integrated with its sandbox + approvals.
        Specifically: no terminal/shell, no read_file/write_file, no
        patch — those are codex's built-in tools."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        forbidden = {
            "terminal", "shell", "read_file", "write_file", "patch",
            "search_files", "process",
        }
        leaked = forbidden & set(EXPOSED_TOOLS)
        assert not leaked, (
            f"these tools must NOT be exposed via the codex callback "
            f"because codex has built-in equivalents: {leaked}"
        )

    def test_expected_hermes_specific_tools_listed(self):
        """The Hermes-specific tools should be present so users on the
        codex runtime keep access to them."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        for required in (
            "web_search",
            "web_extract",
            "browser_navigate",
            "vision_analyze",
            "image_generate",
            "skill_view",
        ):
            assert required in EXPOSED_TOOLS, f"missing {required!r}"

    def test_agent_loop_tools_not_exposed(self):
        """delegate_task / memory / session_search / todo require the
        running AIAgent context to dispatch, so a stateless MCP callback
        can't drive them. They must NOT be in EXPOSED_TOOLS."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        for agent_loop_tool in ("delegate_task", "memory", "session_search", "todo"):
            assert agent_loop_tool not in EXPOSED_TOOLS, (
                f"{agent_loop_tool!r} requires the agent loop context "
                "and can't be reached through a stateless MCP callback"
            )

    def test_kanban_worker_tools_exposed(self):
        """Kanban workers run as `hermes chat -q` subprocesses; if they
        come up on the codex_app_server runtime, the worker can do the
        actual work via codex's shell but needs the kanban tools through
        the MCP callback to report back to the kernel. Without these
        tools available, the worker would hang at completion time."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        # Worker handoff tools — every dispatched worker uses at least
        # one of {complete, block, comment} to close out its task.
        for worker_tool in (
            "kanban_complete",
            "kanban_block",
            "kanban_comment",
            "kanban_heartbeat",
        ):
            assert worker_tool in EXPOSED_TOOLS, (
                f"{worker_tool!r} missing from codex callback — kanban "
                "workers on codex_app_server runtime would hang"
            )

    def test_kanban_orchestrator_tools_exposed(self):
        """Orchestrator agents need to dispatch new tasks, query the
        board, and unblock/link tasks. Exposed so an orchestrator on
        codex_app_server can do its job."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        for orch_tool in (
            "kanban_create",
            "kanban_show",
            "kanban_list",
            "kanban_unblock",
            "kanban_link",
        ):
            assert orch_tool in EXPOSED_TOOLS, (
                f"{orch_tool!r} missing from codex callback"
            )

    def test_mcp_list_tools_uses_hermes_json_schema_not_kwargs(self, monkeypatch):
        server = _build_test_server(monkeypatch)

        tools = {tool.name: tool for tool in asyncio.run(_listed_tools(server))}

        web_schema = tools["web_search"].inputSchema
        assert "query" in web_schema["properties"]
        assert web_schema["required"] == ["query"]
        assert "kwargs" not in web_schema["properties"]

        skill_schema = tools["skill_view"].inputSchema
        assert "name" in skill_schema["properties"]
        assert skill_schema["required"] == ["name"]
        assert "kwargs" not in skill_schema["properties"]

    def test_mcp_call_tool_dispatches_top_level_arguments(self, monkeypatch):
        calls = []
        server = _build_test_server(monkeypatch, calls=calls)

        result = asyncio.run(_call_tool(server, "web_search", {"query": "hermes agent"}))

        assert calls == [("web_search", {"query": "hermes agent"})]
        assert result.root.isError is False
        assert result.root.content[0].text == "handled web_search"

    def test_mcp_call_tool_missing_required_field_returns_error(self, monkeypatch):
        calls = []
        server = _build_test_server(monkeypatch, calls=calls)

        result = asyncio.run(_call_tool(server, "web_search", {}))

        assert calls == []
        assert result.root.isError is True
        assert "Input validation error" in result.root.content[0].text
        assert "query" in result.root.content[0].text

    def test_mcp_call_tool_exception_returns_error_result(self, monkeypatch):
        server = _build_test_server(monkeypatch, raise_on_call=RuntimeError("boom"))

        result = asyncio.run(_call_tool(server, "web_search", {"query": "hermes"}))

        assert result.root.isError is True
        assert "boom" in result.root.content[0].text


class TestMain:
    def test_main_returns_2_when_mcp_unavailable(self, monkeypatch):
        """When the mcp package isn't installed, main() should exit
        cleanly with code 2 and an install hint, not crash."""
        import agent.transports.hermes_tools_mcp_server as m

        def boom_build(*a, **kw):
            raise ImportError("mcp not installed")

        monkeypatch.setattr(m, "_build_server", boom_build)
        rc = m.main(["--verbose"])
        assert rc == 2

    def test_main_handles_keyboard_interrupt(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        class FakeServer:
            def run(self):
                raise KeyboardInterrupt()

        monkeypatch.setattr(m, "_build_server", lambda: FakeServer())
        rc = m.main([])
        assert rc == 0

    def test_main_returns_1_on_runtime_error(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        class CrashingServer:
            def run(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(m, "_build_server", lambda: CrashingServer())
        rc = m.main([])
        assert rc == 1
