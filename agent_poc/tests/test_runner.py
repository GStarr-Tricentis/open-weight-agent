from __future__ import annotations

import json

import pytest

from agent_poc.agent.runner import AgentRunner
from agent_poc.agent.types import RegisteredTool, ToolSource
from agent_poc.config.loader import (
    AgentCoreConfig,
    AgentPocConfig,
    MCPConfig,
    ModelConfig,
    SandboxConfig,
    ToolsConfig,
)
from agent_poc.tools.registry import ToolRegistry
from agent_poc.tests.conftest import MockBackend, make_stop_response, make_tool_call_response


def _config(max_iterations: int = 5, window: int = 3) -> AgentPocConfig:
    return AgentPocConfig(
        model=ModelConfig(
            provider="test",
            base_url="http://localhost",
            api_key="x",
            model_name="x",
            temperature=0.0,
        ),
        agent=AgentCoreConfig(
            max_iterations=max_iterations,
            tool_timeout_seconds=30,
            repeated_call_window=window,
        ),
        tools=ToolsConfig(static=[]),
        mcp=MCPConfig(servers=[]),
        sandbox=SandboxConfig(),
    )


def _registry_with_echo() -> ToolRegistry:
    r = ToolRegistry()
    r.register(
        RegisteredTool(
            name="echo",
            description="",
            input_schema={},
            callable=lambda args: args.get("text", ""),
            source=ToolSource.STATIC,
        )
    )
    return r


# --- happy path ---

def test_stop_immediately():
    backend = MockBackend([make_stop_response("Done.")])
    runner = AgentRunner(backend, _registry_with_echo(), _config(), system_prompt="")
    state = runner.run("hello")
    assert state.finished is True
    assert state.iteration == 1


def test_stop_response_content_in_messages():
    backend = MockBackend([make_stop_response("The answer is 42.")])
    runner = AgentRunner(backend, _registry_with_echo(), _config(), system_prompt="")
    state = runner.run("what is the answer?")
    assistant_msgs = [m for m in state.messages if m.get("role") == "assistant"]
    assert any("42" in (m.get("content") or "") for m in assistant_msgs)


def test_one_tool_call_then_stop():
    backend = MockBackend([
        make_tool_call_response("echo", {"text": "hi"}, call_id="tc1"),
        make_stop_response("Done."),
    ])
    runner = AgentRunner(backend, _registry_with_echo(), _config(), system_prompt="")
    state = runner.run("hello")
    assert state.finished is True
    tool_msgs = [m for m in state.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "tc1"


def test_system_prompt_prepended():
    backend = MockBackend([make_stop_response()])
    runner = AgentRunner(backend, _registry_with_echo(), _config(), system_prompt="Be helpful.")
    state = runner.run("hi")
    assert state.messages[0]["role"] == "system"
    assert state.messages[0]["content"] == "Be helpful."


def test_no_system_prompt_skips_system_message():
    backend = MockBackend([make_stop_response()])
    runner = AgentRunner(backend, _registry_with_echo(), _config(), system_prompt="")
    state = runner.run("hi")
    assert state.messages[0]["role"] == "user"


def test_user_message_appended():
    backend = MockBackend([make_stop_response()])
    runner = AgentRunner(backend, _registry_with_echo(), _config(), system_prompt="")
    state = runner.run("hello world")
    user_msgs = [m for m in state.messages if m.get("role") == "user"]
    assert user_msgs[0]["content"] == "hello world"


# --- iteration limit ---

def test_max_iterations_stops_loop():
    responses = [make_tool_call_response("echo", {"text": str(i)}, call_id=f"tc{i}") for i in range(20)]
    backend = MockBackend(responses)
    runner = AgentRunner(backend, _registry_with_echo(), _config(max_iterations=3), system_prompt="")
    state = runner.run("go")
    assert state.iteration <= 3
    assert state.finished is False


# --- repeated-call detection ---

def test_repeated_call_detection_injects_error():
    window = 3
    responses = [
        make_tool_call_response("echo", {"text": "x"}, call_id=f"tc{i}") for i in range(window + 2)
    ] + [make_stop_response()]
    backend = MockBackend(responses)
    runner = AgentRunner(backend, _registry_with_echo(), _config(max_iterations=10, window=window), system_prompt="")
    state = runner.run("go")
    tool_msgs = [m for m in state.messages if m.get("role") == "tool"]
    error_msgs = [m for m in tool_msgs if "Repeated" in m.get("content", "")]
    assert len(error_msgs) >= 1


def test_repeated_call_detection_window_size():
    """Deque maxlen must equal repeated_call_window, not a hardcoded 10."""
    window = 2
    # 3 identical calls → detection fires on call 3
    responses = [
        make_tool_call_response("echo", {"text": "x"}, call_id=f"tc{i}") for i in range(window + 1)
    ] + [make_stop_response()]
    backend = MockBackend(responses)
    runner = AgentRunner(backend, _registry_with_echo(), _config(max_iterations=10, window=window), system_prompt="")
    state = runner.run("go")
    tool_msgs = [m for m in state.messages if m.get("role") == "tool"]
    error_msgs = [m for m in tool_msgs if "Repeated" in m.get("content", "")]
    assert len(error_msgs) >= 1


# --- plan acceptance-checklist aliases (phase 1) ---

test_one_tool_call = test_one_tool_call_then_stop
test_max_iterations = test_max_iterations_stops_loop
test_repeated_call_detection = test_repeated_call_detection_injects_error


def test_one_tool_call_iteration_count():
    """After one tool call + stop, iteration should be 2 and execution_history has 1 entry."""
    backend = MockBackend([
        make_tool_call_response("echo", {"text": "hi"}, call_id="tc1"),
        make_stop_response("Done."),
    ])
    runner = AgentRunner(backend, _registry_with_echo(), _config(), system_prompt="")
    state = runner.run("hello")
    assert state.iteration == 2
    assert len(state.execution_history) == 1


def test_max_iterations_finish_reason():
    """When loop exits via max_iterations, finish_reason must be 'max_iterations'."""
    responses = [make_tool_call_response("echo", {"text": str(i)}, call_id=f"tc{i}") for i in range(20)]
    backend = MockBackend(responses)
    runner = AgentRunner(backend, _registry_with_echo(), _config(max_iterations=3), system_prompt="")
    state = runner.run("go")
    assert state.finished is False
    assert state.finish_reason == "max_iterations"


def test_unknown_tool_self_correction():
    """A call to an unknown tool produces an error ToolResult but does not crash the runner."""
    backend = MockBackend([
        make_tool_call_response("no_such_tool", {"x": 1}, call_id="tc1"),
        make_stop_response("I give up."),
    ])
    runner = AgentRunner(backend, _registry_with_echo(), _config(), system_prompt="")
    state = runner.run("go")
    assert state.finished is True
    error_results = [r for r in state.execution_history if r.error]
    assert len(error_results) == 1
    assert "no_such_tool" in error_results[0].output


def test_nonidentical_calls_not_flagged():
    """Alternating different calls must never trigger repeat detection."""
    responses = [
        make_tool_call_response("echo", {"text": "a"}, call_id="tc0"),
        make_tool_call_response("echo", {"text": "b"}, call_id="tc1"),
        make_tool_call_response("echo", {"text": "a"}, call_id="tc2"),
        make_stop_response(),
    ]
    backend = MockBackend(responses)
    runner = AgentRunner(backend, _registry_with_echo(), _config(max_iterations=10, window=3), system_prompt="")
    state = runner.run("go")
    tool_msgs = [m for m in state.messages if m.get("role") == "tool"]
    error_msgs = [m for m in tool_msgs if "Repeated" in m.get("content", "")]
    assert len(error_msgs) == 0
