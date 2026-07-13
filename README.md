# open-weight-agent

A synchronous tool-using agent that runs against any OpenAI-compatible local model server (Ollama, llama.cpp, vLLM, etc.).

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running (`ollama serve`)
- A pulled model — `ollama pull qwen2.5:7b` is the default

## Install

```bash
# Core only
pip install -e .

# With Streamlit UI
pip install -e ".[ui]"

# With MCP server support
pip install -e ".[mcp]"

# Development (includes pytest)
pip install -e ".[dev,mcp]"
```

## Streamlit UI

A web UI that shows tool calls, per-call latency, and token counts after each run.

```bash
pip install -e ".[ui]"
streamlit run ui/app.py
```

- **Left panel (70%)**: chat history and prompt input
- **Right panel (30%)**: tool call cards with args, output, and elapsed time, plus a run summary (iterations, finish reason, wall time, token counts)
- Model dropdown populated from `ollama list`

## Quick start

```bash
ollama pull qwen2.5:7b
python -m agent_poc.main --prompt "list the files in the current directory"
```

Interactive mode (no `--prompt`):
```bash
python -m agent_poc.main
> what is 2 + 2?
```

Override model without editing config:
```bash
python -m agent_poc.main --model llama3.1:8b --prompt "hello"
```

## Tested models

| Model | Tool use | Notes |
|---|---|---|
| qwen2.5:14b | Excellent | Best overall |
| qwen2.5:7b | Good | Default; use when RAM-limited |
| llama3.1:8b | Good | Minor formatting quirks |
| mistral:7b | Fair | Sometimes malforms tool call JSON |
| codellama:* | Poor | Not recommended |

## The four demos

### Demo 1 — File Q&A
```bash
echo -e "Paris\nTokyo\nNairobi\nSydney" > cities.txt
python -m agent_poc.main --prompt "Read cities.txt and tell me how many cities are listed"
```
Agent calls `read_file`, then answers from the content.

### Demo 2 — Python computation
```bash
python -m agent_poc.main --prompt "Use python_exec to compute the sum of the first 100 natural numbers"
```
Agent writes and runs Python code, reads the output (`5050`), reports the answer.

### Demo 3 — MCP server tools
Add a server to `agent_poc/config/config.yaml`:
```yaml
mcp:
  servers:
    - name: fs
      command: uvx
      args: ["mcp-server-filesystem", "/tmp"]
```
```bash
python -m agent_poc.main --prompt "list tools available"
```
MCP tools appear alongside static tools; the agent can invoke them.

### Demo 4 — Custom format parser
```bash
cat > sample.dat << 'EOF'
##name=Alice;age=30;city=NYC
##name=Bob;age=25;city=LA
EOF
python -m agent_poc.main --prompt "Parse sample.dat — each line starts with ## and fields are separated by ; in key=value format. How many records are there and what are the names?"
```
Agent reads the file, recognises the format, writes a parser with `python_exec`, iterates on errors, and reports the result.

## Config reference (`agent_poc/config/config.yaml`)

```yaml
model:
  provider: ollama          # informational only
  base_url: http://localhost:11434/v1
  api_key: ollama           # required by OpenAI SDK; value ignored by Ollama
  model_name: qwen2.5:7b
  temperature: 0.0

agent:
  max_iterations: 20        # hard cap on tool-call rounds
  tool_timeout_seconds: 30  # per-tool execution timeout
  repeated_call_window: 3   # identical consecutive calls before error injection

tools:
  static:                   # which static tool groups to register
    - filesystem            # read_file, write_file, list_dir
    - shell                 # shell (subprocess, no pipes)
    - python_exec           # sandboxed Python subprocess

mcp:
  servers: []               # list of {name, command, args}

sandbox:
  timeout_seconds: 15       # python_exec subprocess timeout
  max_output_bytes: 65536   # truncate output above this size
  allow_network: false      # informational (not enforced on macOS)
```

## Running tests

```bash
# Unit tests (no Ollama required)
pytest agent_poc/tests/ -v --ignore=agent_poc/tests/integration

# Integration tests (Ollama must be running)
pytest agent_poc/tests/integration/ -v -m integration

# Override model for integration tests
AGENT_MODEL=llama3.1:8b pytest agent_poc/tests/integration/ -v -m integration
```

## Known limitations

- **No true network isolation on macOS** — the sandbox subprocess runs with the same network access as the parent. Blocking outbound connections requires a firewall rule or container.
- **MCP reconnects per call** — each tool invocation opens a new MCP subprocess, performs the handshake, calls the tool, then closes. This avoids persistent session management but adds ~100–500 ms of latency per MCP tool call.
- **Shell tool has no safelist** — `shell` runs arbitrary commands as the current user. Intended for local/trusted use only.
- **Generated tool code runs in sandbox** — only stdlib is available; third-party packages installed in the venv are not accessible from inside `python_exec`.
