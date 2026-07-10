from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# --- import guard ---

def test_import_does_not_raise():
    """Module must load regardless of whether mcp is installed."""
    from agent_poc.tools.mcp_adapter import MCPAdapter, MCP_AVAILABLE  # noqa: F401
    assert isinstance(MCP_AVAILABLE, bool)


def test_mcp_unavailable_connect_is_noop():
    """When MCP_AVAILABLE=False, connect() silently does nothing."""
    import agent_poc.tools.mcp_adapter as mod
    from agent_poc.tools.mcp_adapter import MCPAdapter

    original = mod.MCP_AVAILABLE
    try:
        mod.MCP_AVAILABLE = False
        adapter = MCPAdapter("srv", "cmd", [])
        adapter.connect()  # must not raise
        assert adapter.list_tools() == []
    finally:
        mod.MCP_AVAILABLE = original


def test_mcp_unavailable_call_tool_returns_error_string():
    """When MCP_AVAILABLE=False, call_tool() returns an error string, not an exception."""
    import agent_poc.tools.mcp_adapter as mod
    from agent_poc.tools.mcp_adapter import MCPAdapter

    original = mod.MCP_AVAILABLE
    try:
        mod.MCP_AVAILABLE = False
        adapter = MCPAdapter("srv", "cmd", [])
        result = adapter.call_tool("any_tool", {})
        assert isinstance(result, str)
        assert "not installed" in result.lower() or "error" in result.lower()
    finally:
        mod.MCP_AVAILABLE = original


# --- lambda capture ---

def _make_fake_mcp_tool(name: str) -> MagicMock:
    t = MagicMock()
    t.name = name
    t.description = f"Tool {name}"
    t.inputSchema = {"type": "object", "properties": {}}
    return t


def _connect_with_fake_tools(names: list[str]):
    """Run connect() with asyncio.run mocked to return fake MCP tool objects."""
    import asyncio
    from agent_poc.tools.mcp_adapter import MCPAdapter

    fake_tools = [_make_fake_mcp_tool(n) for n in names]

    def _fake_run(coro):
        # Close the coroutine to suppress "never awaited" warnings.
        coro.close()
        return fake_tools

    with patch("asyncio.run", side_effect=_fake_run):
        adapter = MCPAdapter("test_srv", "fake_cmd", [])
        adapter.connect()
    return adapter


def test_tool_registration_count():
    adapter = _connect_with_fake_tools(["alpha", "beta", "gamma"])
    assert len(adapter.list_tools()) == 3


def test_tool_registration_names():
    adapter = _connect_with_fake_tools(["alpha", "beta", "gamma"])
    names = {t.name for t in adapter.list_tools()}
    assert names == {"alpha", "beta", "gamma"}


def test_tool_registration_source_is_mcp():
    from agent_poc.agent.types import ToolSource
    adapter = _connect_with_fake_tools(["tool_x"])
    assert adapter.list_tools()[0].source == ToolSource.MCP


def test_tool_registration_lambda_capture():
    """Each callable must be bound to its own tool name — not the last one in the loop."""
    adapter = _connect_with_fake_tools(["alpha", "beta", "gamma"])
    tools = adapter.list_tools()

    called = []
    with patch.object(adapter, "call_tool", side_effect=lambda name, args: called.append(name) or "ok"):
        for tool in tools:
            tool.callable({})

    # Every registered tool must have called with its own name
    assert sorted(called) == ["alpha", "beta", "gamma"]
    # No name should appear more than once (confirms no shared closure variable)
    assert len(set(called)) == 3


def test_tool_registration_lambda_capture_single_name():
    """Regression: a single tool must still call with the correct name."""
    adapter = _connect_with_fake_tools(["only_tool"])
    tools = adapter.list_tools()

    called = []
    with patch.object(adapter, "call_tool", side_effect=lambda name, args: called.append(name) or "ok"):
        tools[0].callable({})

    assert called == ["only_tool"]


# --- disconnect ---

def test_disconnect_does_not_raise():
    from agent_poc.tools.mcp_adapter import MCPAdapter
    adapter = MCPAdapter("srv", "cmd", [])
    adapter.disconnect()  # reconnect-per-call; must be a no-op
