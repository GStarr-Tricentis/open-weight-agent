from __future__ import annotations

import json
import logging
from collections import deque

from agent_poc.agent.state import RunState
from agent_poc.agent.types import ModelBackend, ToolResult
from agent_poc.config.loader import AgentPocConfig
from agent_poc.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentRunner:
    def __init__(
        self,
        backend: ModelBackend,
        registry: ToolRegistry,
        config: AgentPocConfig,
        system_prompt: str = "",
    ) -> None:
        self._backend = backend
        self._registry = registry
        self._config = config
        self._system_prompt = system_prompt

    def run(self, user_input: str) -> RunState:
        window = self._config.agent.repeated_call_window
        state = RunState()
        state.recent_calls = deque(maxlen=window)

        if self._system_prompt:
            state.messages.append({"role": "system", "content": self._system_prompt})
        state.messages.append({"role": "user", "content": user_input})

        while state.iteration < self._config.agent.max_iterations:
            logger.debug("Iteration %d", state.iteration)

            response = self._backend.complete(state.messages, self._registry.list_tools())

            state.messages.append(
                response.raw.choices[0].message.model_dump(exclude_none=False)
            )

            if response.finish_reason == "stop" and not response.tool_calls:
                state.finished = True
                state.finish_reason = "stop"
                state.iteration += 1
                break

            if response.finish_reason == "length":
                state.finish_reason = "length"
                state.iteration += 1
                break

            for tool_call in response.tool_calls:
                call_key = (tool_call.name, json.dumps(tool_call.arguments, sort_keys=True))

                recent = list(state.recent_calls)
                if len(recent) >= window and len(set(recent[-(window - 1):])) == 1 and recent[-(window - 1)] == call_key:
                    result = ToolResult(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        output=(
                            f"Repeated identical tool call detected for '{tool_call.name}'. "
                            "Try a different approach."
                        ),
                        error=True,
                    )
                else:
                    result = self._registry.execute(tool_call)

                state.recent_calls.append(call_key)
                state.execution_history.append(result)
                state.messages.append({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.output,
                })

            state.iteration += 1

        if not state.finished and not state.finish_reason:
            state.finish_reason = "max_iterations"

        return state
