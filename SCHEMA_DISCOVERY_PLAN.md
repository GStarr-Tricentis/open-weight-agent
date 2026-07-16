# Robust Schema Discovery for Open-Weight Models

## Overview

Two targeted improvements to make the graph pipeline more reliable with smaller models:

1. **Structured outputs** — constrain LLM responses to a JSON schema so parsing never fails on malformed output
2. **Conversational retry** — pass parse errors back to the model so retries are informed, not identical

Both changes are confined to `graph_pipeline/schema_discovery.py` and the `ModelBackend` protocol.
Nothing in the extractor, ingest script, or entry points changes.

---

## Part 1: Structured Outputs

### 1a. Extend `ModelBackend.complete()`

**File: `agent_poc/agent/types.py`**

Add an optional `response_format` parameter to the protocol:

```python
class ModelBackend(Protocol):
    def complete(
        self,
        messages: list[dict],
        tools: list[RegisteredTool],
        response_format: dict | None = None,
    ) -> ModelResponse: ...
```

`response_format` follows the OpenAI spec:
```python
{"type": "json_schema", "json_schema": {"name": "...", "schema": {...}, "strict": True}}
```

When `None` (the default), behaviour is identical to today — no existing code breaks.

---

### 1b. Pass `response_format` through both backends

**File: `agent_poc/models/openai_compatible.py`**

Add `response_format: dict | None = None` to `complete()`. Pass it to the SDK call:

```python
response = self._client.chat.completions.create(
    model=self._model,
    messages=messages,
    tools=tools_param,
    temperature=self._temperature,
    response_format=response_format or openai.NOT_GIVEN,
)
```

**File: `agent_poc/models/tricentis_backend.py`**

Add `response_format: dict | None = None` to `complete()`, `_complete_anthropic()`, and
`_complete_openai()`. Do not pass it to either the Anthropic or OpenAI SDK calls — the
Anthropic API does not support `json_schema` mode, and the parameter is accepted for
interface compatibility only.

---

### 1c. Define JSON schemas

**New directory: `graph_pipeline/schemas/`**

Three schema files, derived directly from the existing Pydantic models in `context_store.py`.

**`node_types.json`** — covers the full Call 1 output. Merges the two previously separate JSON
values (array + structural config) into a single object, which is required for `json_schema` mode:

```json
{
  "name": "node_types_response",
  "strict": true,
  "schema": {
    "type": "object",
    "required": ["node_types", "id_field", "type_field"],
    "additionalProperties": false,
    "properties": {
      "node_types": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["name", "maps_to", "identity_key"],
          "additionalProperties": false,
          "properties": {
            "name": {"type": "string"},
            "maps_to": {"type": "string"},
            "identity_key": {"type": "string"}
          }
        }
      },
      "id_field": {"type": "string"},
      "type_field": {"type": ["string", "null"]},
      "hierarchy_config": {
        "oneOf": [{"type": "null"}, {
          "type": "object",
          "required": ["field", "separator", "phantom_label", "edge_type"],
          "properties": {
            "field": {"type": "string"},
            "separator": {"type": "string"},
            "phantom_label": {"type": "string"},
            "edge_type": {"type": "string"}
          }
        }]
      },
      "nested_collections": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["field", "child_label", "edge_type", "id_field"],
          "properties": {
            "field": {"type": "string"},
            "child_label": {"type": "string"},
            "edge_type": {"type": "string"},
            "id_field": {"type": "string"}
          }
        }
      }
    }
  }
}
```

**`relationship_types.json`** — covers Call 2 output exactly as structured today.

**`ambiguous_fields.json`** — Call 3 output wrapped in an object (bare arrays are not valid
root values in `json_schema` mode):

```json
{
  "name": "ambiguous_fields_response",
  "strict": true,
  "schema": {
    "type": "object",
    "required": ["fields"],
    "additionalProperties": false,
    "properties": {
      "fields": {
        "type": "array",
        "items": {"type": "string"}
      }
    }
  }
}
```

---

### 1d. Update prompts to match new schemas

**Files: `graph_pipeline/prompts/schema_proposal_nodes.txt`**

Remove the instruction to output two separate JSON values. Replace with a single-object
instruction. The embedded example JSON is updated to match the new merged structure.

**File: `graph_pipeline/prompts/schema_proposal_relationships.txt`**

Minor update only — already returns a single JSON object, no structural change needed.

**File: `graph_pipeline/prompts/schema_proposal_ambiguous.txt`**

Change:
> "Return ONLY a JSON array of field name strings"

To:
> "Return a JSON object with a single key `fields` containing an array of field name strings"

---

### 1e. Update `_llm_call` and callers in `schema_discovery.py`

**File: `graph_pipeline/schema_discovery.py`**

Add `response_format: dict | None = None` to `_llm_call`. Pass it through to `backend.complete()`.

Load schemas at module level (once, not per-call):

```python
_SCHEMAS_DIR = Path(__file__).parent / "schemas"
_NODE_TYPES_FORMAT = json.loads((_SCHEMAS_DIR / "node_types.json").read_text())
_REL_TYPES_FORMAT = json.loads((_SCHEMAS_DIR / "relationship_types.json").read_text())
_AMBIGUOUS_FORMAT = json.loads((_SCHEMAS_DIR / "ambiguous_fields.json").read_text())
```

Each `_propose_*` function passes its schema to `_llm_call`.

Update parsing in `_propose_node_types`: instead of extracting two separate JSON values with
`_extract_json_values()`, parse a single object and read `data["node_types"]` and the structural
config keys directly.

Update parsing in `_propose_ambiguous_fields`: read `data["fields"]` instead of treating
`data` as the list directly.

---

---

## Part 2: Conversational Retry

### 2a. Refactor `_llm_call` to accept a messages list

**File: `graph_pipeline/schema_discovery.py`**

Change `_llm_call` to accept a `messages: list[dict]` instead of a `prompt: str`. The callers
own and extend the messages list across retries. `_llm_call` is now responsible only for the
LLM call and empty-response retry:

```python
def _llm_call(
    backend: ModelBackend,
    messages: list[dict],
    max_retries: int,
    label: str,
    response_format: dict | None = None,
) -> str:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = backend.complete(
                messages=messages,
                tools=[],
                response_format=response_format,
            )
            raw = response.content or ""
            if raw.strip():
                return raw
            last_error = ValueError("empty response")
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                "Your response was empty. Please return the required JSON."
            })
        except Exception as exc:
            logger.warning("%s attempt %d error: %s", label, attempt, exc)
            last_error = exc

    raise RuntimeError(
        f"{label} failed after {max_retries} attempts. Last error: {last_error}"
    )
```

---

### 2b. Add parse-failure feedback in each `_propose_*` function

Each function owns a messages list and extends it with the model's bad response + a correction
signal before retrying:

```python
messages = [{"role": "user", "content": prompt}]
last_error: Exception | None = None

for attempt in range(1, max_retries + 1):
    raw = _llm_call(backend, messages, max_retries=1,
                    label=f"node_types attempt {attempt}",
                    response_format=_NODE_TYPES_FORMAT)
    try:
        data = parse(raw)
        return data
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("node_types attempt %d failed: %s", attempt, exc)
        last_error = exc
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content":
            f"Your response could not be parsed (attempt {attempt}): {exc}. "
            f"Please return only valid JSON matching the required format. "
            f"Do not include any explanation or markdown."
        })
```

The correction message includes the actual parse error (`exc`) so the model knows specifically
what was wrong — missing key, unexpected type, extra text, etc.

This pattern applies identically to `_propose_node_types`, `_propose_relationship_types`,
and `_propose_ambiguous_fields`.

---

## File Summary

| File | Change |
|------|--------|
| `agent_poc/agent/types.py` | Add `response_format: dict \| None = None` to `ModelBackend.complete()` |
| `agent_poc/models/openai_compatible.py` | Pass `response_format` through to SDK |
| `agent_poc/models/tricentis_backend.py` | Same |
| `graph_pipeline/schemas/node_types.json` | **New** — JSON schema for Call 1 |
| `graph_pipeline/schemas/relationship_types.json` | **New** — JSON schema for Call 2 |
| `graph_pipeline/schemas/ambiguous_fields.json` | **New** — JSON schema for Call 3 |
| `graph_pipeline/prompts/schema_proposal_nodes.txt` | Update to single-object output |
| `graph_pipeline/prompts/schema_proposal_relationships.txt` | Minor update |
| `graph_pipeline/prompts/schema_proposal_ambiguous.txt` | Wrap output in `{"fields": [...]}` |
| `graph_pipeline/schema_discovery.py` | Refactor `_llm_call`, conversational retry, load schemas, update parsers |

**No changes to:** `extractor.py`, `scripts/ingest.py`, `main.py`, `scripts/query.py`,
`scripts/benchmark.py`, `context_store.py`, or any tests beyond updating fixtures.

---

## Key Design Decisions

**Why merge the two Call 1 JSON values into one object?**
Because `json_schema` mode requires a single root object. The current two-value output was
a workaround for free-form generation — with structured outputs it is unnecessary and simpler
to parse.

**Why not add `json_schema` to Call 4 (per-record extraction)?**
Call 4's output schema depends on what nodes were extracted for a given dataset — it is not
statically definable. Conversational retry is sufficient there.

**Why keep the fallback?**
The pipeline should work against any `ModelBackend`, including ones backed by models or
endpoints that do not support `response_format`. The fallback ensures the pipeline degrades
gracefully rather than breaking entirely.
