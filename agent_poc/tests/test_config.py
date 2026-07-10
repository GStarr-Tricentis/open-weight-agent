from __future__ import annotations

import pytest

from agent_poc.config.loader import load_config


CONFIG_PATH = "agent_poc/config/config.yaml"


def test_load_config_model_name():
    c = load_config(CONFIG_PATH)
    assert c.model.model_name == "qwen2.5:7b"


def test_load_config_agent_defaults():
    c = load_config(CONFIG_PATH)
    assert c.agent.max_iterations == 20
    assert c.agent.repeated_call_window == 3


def test_load_config_sandbox():
    c = load_config(CONFIG_PATH)
    assert c.sandbox.timeout_seconds == 15
    assert c.sandbox.max_output_bytes == 65536
    assert c.sandbox.allow_network is False


def test_load_config_mcp_empty():
    c = load_config(CONFIG_PATH)
    assert c.mcp.servers == []
