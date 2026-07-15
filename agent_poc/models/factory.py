from __future__ import annotations

from agent_poc.agent.types import ModelBackend
from agent_poc.config.loader import AgentPocConfig
from agent_poc.models.openai_compatible import OpenAICompatibleBackend


def make_backend(
    config: AgentPocConfig,
    provider: str | None = None,
    model_override: str | None = None,
) -> ModelBackend:
    provider = provider or config.model.provider
    if model_override:
        config.model.model_name = model_override
    if provider == "local":
        return OpenAICompatibleBackend(config.model)
    raise ValueError(f"Unknown provider: {provider!r}. Valid options: local")
