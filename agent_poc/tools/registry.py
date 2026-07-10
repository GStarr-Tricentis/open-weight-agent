from __future__ import annotations

import concurrent.futures
import logging

from agent_poc.agent.types import RegisteredTool, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s (source=%s)", tool.name, tool.source)

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[RegisteredTool]:
        return list(self._tools.values())

    def execute(self, call: ToolCall, timeout_override: float | None = None) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                error=True,
                output=(
                    f"Unknown tool '{call.name}'. "
                    f"Available tools: {list(self._tools.keys())}"
                ),
            )

        timeout = timeout_override if timeout_override is not None else tool.timeout_seconds

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(tool.callable, call.arguments)
                output = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                error=True,
                output=f"Tool '{call.name}' timed out after {timeout}s.",
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                error=True,
                output=f"Tool '{call.name}' raised an exception: {exc}",
            )

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            output=str(output),
        )
