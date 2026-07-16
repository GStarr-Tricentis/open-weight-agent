# Fix TricentisBackend: Anthropic and OpenAI Routing

## Root Cause

The backend uses a single OpenAI client pointed at `/hub-service/openai/` for all
deployments. The TAIS hub routes differently by model type:

- **Anthropic models** (`anthropic.*`) → `/hub-service` via Anthropic SDK +
  `AIHubAnthropicSyncTransport` (custom httpx transport that handles auth per-request)
- **OpenAI models** (everything else) → `/hub-service/openai/deployments/{model}` via
  OpenAI SDK (our current URL is missing the `/deployments/{model}` suffix)

## File Changed

Only `agent_poc/models/tricentis_backend.py`. No other files change — the `ModelBackend`
protocol, runner, and all entry points stay the same. All conversion is contained inside
the backend.

---

## `__init__`

Add a flag that controls branching everywhere:

```python
self._is_anthropic = deployment.lower().startswith("anthropic.")
```

---

## `_setup()`

Branch on `self._is_anthropic`.

**Anthropic path:**

```python
from tricentis_ai_client.hub.anthropic_transport import AIHubAnthropicSyncTransport
import anthropic, httpx

transport = AIHubAnthropicSyncTransport(
    model_id=self._deployment,
    bearer_token_provider=lambda: self._tais_client.token_provider.get_valid_token(),
)
self._anthropic_client = anthropic.Anthropic(
    api_key="placeholder",
    base_url=f"{config.gateway_url}/api/v1/hub-service",
    http_client=httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(120.0, connect=10.0),
    ),
)
```

`bearer_token_provider` is a lambda so `get_valid_token()` is called fresh on every
request — the transport handles injection. No separate `_fresh_token()` call needed on
the Anthropic path.

**OpenAI path:**

Fix the URL by appending `/deployments/{model}`:

```python
return OpenAI(
    base_url=f"{config.gateway_url}/api/v1/hub-service/openai/deployments/{self._deployment}",
    api_key=token,
    default_headers={
        "x-product-name": config.product_name,
        "x-tenant-name": config.tenant_name,
    },
)
```

---

## `complete()`

Branch on `self._is_anthropic`.

### Anthropic path

**Message conversion (OpenAI → Anthropic):**

The runner maintains messages in OpenAI format. Convert before calling the SDK:

| OpenAI format | Anthropic format |
|---|---|
| `{"role": "system", "content": "..."}` | Pull out → `system=` param |
| `{"role": "user"/"assistant", "content": "..."}` | Pass through |
| `{"role": "assistant", "tool_calls": [...]}` | `{"role": "assistant", "content": [{"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": json.loads(tc.function.arguments)}]}` |
| `{"role": "tool", "tool_call_id": "...", "content": "..."}` | `{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]}` |

**Tool schema conversion:**

```python
anthropic_tools = [
    {"name": t.name, "description": t.description, "input_schema": t.input_schema}
    for t in tools
]
```

**Call:**

```python
response = self._anthropic_client.messages.create(
    model=self._deployment,
    messages=anthropic_messages,
    tools=anthropic_tools or anthropic.NOT_GIVEN,
    system=system_text or anthropic.NOT_GIVEN,
    max_tokens=4096,
    temperature=self._temperature,
)
```

**Response conversion (Anthropic → ModelResponse):**

```python
text_content = next(
    (b.text for b in response.content if b.type == "text"), None
)
tool_calls = [
    ToolCall(id=b.id, name=b.name, arguments=b.input)
    for b in response.content if b.type == "tool_use"
]
finish_reason = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
}.get(response.stop_reason, response.stop_reason)

# Reconstruct OpenAI-format assistant_message so the runner can append it normally
assistant_message = {
    "role": "assistant",
    "content": text_content,
    "tool_calls": [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
        }
        for tc in tool_calls
    ] or None,
}
```

### OpenAI path

Unchanged from today. The only fix is the URL in `_setup()`.

---

## Testing Checklist

After implementation, verify in order:

1. `python main.py --provider tricentis --prompt "say hello"` — Anthropic path, no tools
2. `python main.py --provider tricentis --prompt "list files in the current directory"` — tool call round-trip
3. Re-run the structured output smoke test (`test_structured_output.py`) to answer whether `response_format` is honored — this unblocks the schema discovery PR decision
