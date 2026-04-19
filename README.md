# AtriNexus

[English](README.md) | [简体中文](README.zh-CN.md)

AtriNexus is a WeCom-based personal AI assistant focused on long-running conversation, multi-layer memory, knowledge retrieval, and safe workspace actions.

It is designed for real, long-running personal usage rather than a generic chatbot demo. A production deployment runs on:

- WeCom as the chat entry point
- FastAPI service layer
- PostgreSQL for conversation, memory, and diary data
- Qdrant for vector memory
- `atrinexus-rag-sdk` for knowledge-base retrieval

## Highlights

- Long-lived WeCom conversations with persistent memory and daily diary generation
- Knowledge-base upload and on-demand retrieval through agent tools
- Safe workspace actions: paged file reading, glob and content search, preview-confirmed edits
- Multi-step task support via an explicit todo tool and a clarification tool for ambiguous requests
- Per-session capability profiles and a layered prompt assembly pipeline
- Operational observability: `/health`, `/health/simple`, `/metrics`, and optional trajectory logging

## What It Does

### 1. Personal assistant chat in WeCom

- Natural text conversation
- Image recognition
- Scheduled messages
- Memory-aware replies
- Knowledge-base lookup via agent tools

### 2. Multi-layer memory

- Short-term conversation history
- Core memory of persistent facts
- Vector memory for semantic recall
- Daily diary generation from conversation history

Storage split:

- PostgreSQL — conversation history, short-term memory, core memory, diaries
- Qdrant — vector memory

### 3. Knowledge-base retrieval

- `SdkRAGService` backed by `atrinexus-rag-sdk`
- SDK-managed namespace per user
- Dedicated Qdrant instance for RAG
- Retrieval is agent-driven: normal messages skip front-loaded retrieval; the agent calls KB tools only when needed

### 4. Workspace actions

The assistant can perform the following actions from chat:

- **Read** — `read_file` with 1-indexed line-numbered output and `offset` / `limit` paging for long files
- **Explore** — `list_directory`, `search_files` for text search, `glob` for pattern-based path lookup
- **Edit** — `preview_edit_file` for precise replace, `preview_write_file` for full rewrite, `preview_append_file` for head/tail append, `rename_path` for rename or move
- **Execute** — read-only command pipelines (e.g. `find`, `du`, `wc`, `stat`, `tree`) run directly; other commands require explicit user confirmation

All file modifications follow a preview-first flow: a diff is generated first and written only after the user replies with approval (`通过` / `确认`).

### 5. Task orchestration

- **Todo tool** — per-session todo list the agent maintains across turns; state rides with the system prompt so it survives context compaction
- **Clarify tool** — when a request is ambiguous, the agent asks a clarifying question and yields back to the user mid-run; the user's next message re-enters the loop with the answer in context

### 6. Operational visibility

- `/health` — full health check (DB, Qdrant, RAG SDK, WeCom credentials)
- `/health/simple` — lightweight liveness probe
- `/metrics` — Prometheus metrics for request counts, token usage, rate-limit state
- Trajectory logging (optional) — per-turn JSONL records for offline review and evaluation

## Architecture Overview

### Service entry

- `run.py`
- `src/app/server.py`

### Conversation pipeline

- `src/conversation/message_handler.py` — ingress orchestration (dedupe, pending confirmation, fast-path, agent loop)
- `src/conversation/context_builder.py` — per-turn memory and mode assembly
- `src/conversation/fast_path_router.py` — deterministic short-path for state-machine replies
- `src/prompting/prompt_manager.py` — layered prompt assembly (static shell + runtime capability snapshot + persona + memory)

### Agent runtime

- `src/agent_runtime/agent_service.py` — run lifecycle, cancellation, follow-up queue
- `src/agent_runtime/agent_loop.py` — tool-calling loop with streaming responses
- `src/agent_runtime/tool_catalog.py` — declarative tool registry with profile-driven exposure
- `src/agent_runtime/tool_profiles.py` — capability profiles (`chat`, `workspace_read`, `workspace_edit`, `workspace_exec`, `full`)
- `src/agent_runtime/agent_tool_guard.py` — tool-call validation, path repair, loop guard, result shaping
- `src/agent_runtime/hooks.py` — four extension hooks (`before_tool_call`, `after_tool_call`, `transform_context`, `on_response`)
- `src/agent_runtime/context_engine.py` — pluggable context-window compression
- `src/agent_runtime/user_runtime.py` — per-user run claim, abort signal, follow-up queue
- `src/agent_runtime/todo_store.py` — per-session todo state
- `src/agent_runtime/clarify_store.py` — mid-run clarify signal

### Provider layer

- `src/ai/providers/openai_compat.py` — OpenAI-compatible streaming client
- `src/ai/stream.py` — SSE parsing and tool-call accumulation
- `src/ai/llm_service.py`, `src/ai/embedding_service.py`, `src/ai/model_manager.py` — model coordination

### Memory and diary

- `src/memory/memory_manager.py`
- `src/memory/memory_store.py`
- `src/features/diary_service.py`
- `src/platform_core/database.py`

### Knowledge base

- `src/knowledge/rag_service.py`
- `src/knowledge/kb_tools.py`

### Workspace runtime

- `src/agent_runtime/runtime.py` — file I/O, search, command execution policy, preview change tracking

### Vector storage

- `src/platform_core/vector_store/qdrant.py`

## Tech Stack

- Python 3.12
- FastAPI
- PostgreSQL
- Qdrant
- `atrinexus-rag-sdk`
- WeCom / `wechatpy`
- APScheduler
- Prometheus client
- httpx

## Project Structure

Source code is organized by capability domain rather than by framework.

```text
AtriNexus/
├── run.py
├── pyproject.toml
├── requirements.txt
├── src/
│   ├── app/              # application assembly and startup
│   ├── ingress/          # WeCom callback, HTTP routers, middleware
│   ├── conversation/     # message orchestration, fast-path, reply cleaner
│   ├── agent_runtime/    # agent loop, tool catalog, hooks, context engine
│   ├── prompting/        # prompt assembly and prompt markdown resources
│   ├── memory/           # three-layer memory + context + update orchestration
│   ├── knowledge/        # RAG service + KB agent tools
│   ├── ai/               # provider adapters and streaming
│   ├── workspace/        # workspace capabilities
│   ├── platform_core/    # database, session, token monitor, vector store, utilities
│   ├── features/         # diary service, web templates
│   └── tests/
├── data/
│   ├── config/
│   ├── database/
│   ├── vectordb_qdrant/
│   └── tasks.json
├── deployment/
└── docs/
```

See `docs/PROJECT_STRUCTURE.md` for module responsibilities.

## Configuration

Runtime configuration lives in `data/config/config.json`. Only a sanitized template (`data/config/config.json.template`) is kept in the repository.

Sensitive values are provided through environment variables:

- `ATRINEXUS_DATABASE_URL`
- `ATRINEXUS_LLM_API_KEY`
- `ATRINEXUS_VISION_API_KEY`
- `ATRINEXUS_NETWORK_SEARCH_API_KEY`
- `ATRINEXUS_INTENT_API_KEY`
- `ATRINEXUS_EMBEDDING_API_KEY`
- `ATRINEXUS_WECOM_SECRET`
- `ATRINEXUS_WECOM_TOKEN`
- `ATRINEXUS_WECOM_ENCODING_AES_KEY`
- `ATRINEXUS_ADMIN_PASSWORD`

Optional runtime toggles:

- `ATRINEXUS_FAST_PATH_INTENT` — `full` (default) or `disabled`. When `disabled`, deterministic fast-path routing is skipped; messages are handled end-to-end by the agent loop. Useful for A/B comparison.
- `ATRINEXUS_TRAJECTORY_PATH` — absolute path to a JSONL file. When set, every turn is appended with user message, assistant reply, tool events, and routing metadata (`fast_path_hit`, `intent`).
- `ATRINEXUS_AGENT_CONTEXT_LENGTH` — context window size used by the compressor (default `32000`).
- `ATRINEXUS_AGENT_MAX_ITERATIONS` — maximum tool-call iterations per turn (default `12`).

For local development, copy `.env.example` to `.env`. Production should prefer systemd `Environment=` / `EnvironmentFile=` over a checked-in `.env`.

## Running Locally

### 1. Install dependencies

Using `uv`:

```bash
uv sync
```

Or with `pip`:

```bash
python -m pip install -r requirements.txt
```

### 2. Prepare config

```bash
cp data/config/config.json.template data/config/config.json
```

Then fill in runtime settings. Secrets can also be placed in `.env`.

### 3. Start the service

```bash
python run.py
```

### 4. Verify

```bash
curl http://127.0.0.1:8080/health/simple
curl http://127.0.0.1:8080/health
```

## Deployment Notes

The project runs behind nginx with a public WeCom callback endpoint. Typical production layout:

- systemd-managed `run.py`
- systemd unit file: `deployment/atrinexus.service`
- nginx reverse proxy
- `/health` and `/metrics` exposed to the monitoring stack
- WeCom callback routed to the service

VPS bootstrap:

```bash
sudo bash deployment/setup_vps.sh
sudo systemctl start atrinexus
sudo systemctl status atrinexus
sudo journalctl -u atrinexus -f
```

## Who This Is For

AtriNexus suits:

- personal AI assistant deployments
- WeCom-based assistant use cases
- long-running memory-centric assistants
- practical RAG + memory + tool integration

It is not designed to be a generic agent platform or a universal SaaS framework.

## License

MIT. See [LICENSE](LICENSE).
