from __future__ import annotations

import time

import pytest

from agent_poc.agent.types import RegisteredTool, ToolCall, ToolSource
from agent_poc.tools.registry import ToolRegistry


def _tool(name: str, fn, timeout: float = 30.0) -> RegisteredTool:
    return RegisteredTool(
        name=name,
        description="",
        input_schema={"type": "object", "properties": {}},
        callable=fn,
        source=ToolSource.STATIC,
        timeout_seconds=timeout,
    )


def _call(name: str, args: dict = {}, call_id: str = "tc1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=args)


# --- unknown tool ---

def test_unknown_tool():
    r = ToolRegistry()
    result = r.execute(_call("ghost"))
    assert result.error is True
    assert "ghost" in result.output


def test_unknown_tool_lists_available():
    r = ToolRegistry()
    r.register(_tool("real_tool", lambda args: "ok"))
    result = r.execute(_call("ghost"))
    assert "real_tool" in result.output


# --- successful execution ---

def test_successful_execution():
    r = ToolRegistry()
    r.register(_tool("greet", lambda args: "hello"))
    result = r.execute(_call("greet"))
    assert result.output == "hello"
    assert result.error is False
    assert result.tool_call_id == "tc1"
    assert result.name == "greet"


def test_successful_execution_passes_args():
    r = ToolRegistry()
    r.register(_tool("echo", lambda args: args.get("text", "")))
    result = r.execute(_call("echo", {"text": "world"}))
    assert result.output == "world"


# --- timeout ---

def test_timeout():
    r = ToolRegistry()
    r.register(_tool("slow", lambda args: time.sleep(60), timeout=0.1))
    result = r.execute(_call("slow"))
    assert result.error is True
    assert "timeout" in result.output.lower() or "timed out" in result.output.lower()


def test_timeout_does_not_raise():
    r = ToolRegistry()
    r.register(_tool("slow", lambda args: time.sleep(60), timeout=0.1))
    # Must not raise — execute() always returns ToolResult
    result = r.execute(_call("slow"))
    assert isinstance(result.output, str)


# --- exception in callable ---

def test_exception_in_callable():
    r = ToolRegistry()
    r.register(_tool("bad", lambda args: (_ for _ in ()).throw(RuntimeError("boom"))))
    result = r.execute(_call("bad"))
    assert result.error is True
    assert "boom" in result.output


def test_exception_in_callable_does_not_raise():
    r = ToolRegistry()
    r.register(_tool("bad", lambda args: 1 / 0))
    result = r.execute(_call("bad"))
    assert result.error is True


# --- timeout_override ---

def test_timeout_override_shortens_timeout():
    r = ToolRegistry()
    r.register(_tool("slow", lambda args: time.sleep(60), timeout=30.0))
    result = r.execute(_call("slow"), timeout_override=0.1)
    assert result.error is True
    assert "timeout" in result.output.lower() or "timed out" in result.output.lower()
