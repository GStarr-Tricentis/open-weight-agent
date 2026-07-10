from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    provider: str
    base_url: str
    api_key: str
    model_name: str
    temperature: float = 0.0


class AgentCoreConfig(BaseModel):
    max_iterations: int = 20
    tool_timeout_seconds: float = 30.0
    repeated_call_window: int = 3


class MCPServerConfig(BaseModel):
    name: str
    command: str
    args: list[str] = []


class MCPConfig(BaseModel):
    servers: list[MCPServerConfig] = []


class ToolsConfig(BaseModel):
    static: list[str] = []


class SandboxConfig(BaseModel):
    timeout_seconds: float = 15.0
    max_output_bytes: int = 65536
    allow_network: bool = False


class AgentPocConfig(BaseModel):
    model: ModelConfig
    agent: AgentCoreConfig
    tools: ToolsConfig
    mcp: MCPConfig = MCPConfig()
    sandbox: SandboxConfig = SandboxConfig()


def load_config(path: str | Path) -> AgentPocConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AgentPocConfig.model_validate(raw)
