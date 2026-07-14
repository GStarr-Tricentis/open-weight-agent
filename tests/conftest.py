import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--llm",
        action="store_true",
        default=False,
        help="Run tests that call a real LLM (requires live Ollama instance)",
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests that require live external services (Neo4j, etc.)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "llm: mark test as requiring a live LLM (deselected unless --llm is passed)",
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring live external services (deselected unless --integration is passed)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--llm"):
        skip_llm = pytest.mark.skip(reason="pass --llm to run LLM integration tests")
        for item in items:
            if item.get_closest_marker("llm"):
                item.add_marker(skip_llm)

    if not config.getoption("--integration"):
        skip_integration = pytest.mark.skip(reason="pass --integration to run integration tests")
        for item in items:
            if item.get_closest_marker("integration"):
                item.add_marker(skip_integration)
