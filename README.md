# AtriNexus

[English](README.md) | [简体中文](README.zh-CN.md)

AtriNexus is a WeCom-based personal AI assistant for persistent conversation, memory, knowledge-base tooling, and lightweight workspace actions.

It is designed for a real long-running personal usage scenario rather than a generic chatbot demo. The current runtime path is:

- WeCom as the chat entrypoint
- FastAPI as the service layer
- PostgreSQL for conversation, memory, and diary data
- Qdrant for vector memory storage
- `atrinexus-rag-sdk` for knowledge-base retrieval
- LangChain for the lightweight agent/tool layer

The current runtime is no longer a simple "chat + attached tools" stack. It now centers on:

- `PromptManager` for layered prompt assembly
- `ToolProfile` for session-stable tool exposure
- `FastPathRouter` for deterministic file/tool requests
- middleware-based agent control for model/tool governance

## What It Does

- WeCom conversation handling
- Persistent short-term and core memory
- Daily diary generation from conversation history
- Knowledge-base upload and retrieval
- Lightweight agent actions from chat
- System health and Prometheus metrics
- Web pages for memory, settings, and knowledge upload

## Current Capabilities

### 1. Personal assistant chat in WeCom

The assistant is built around WeCom callback handling and long-running personal conversation.

It supports:

- normal text conversation
- image recognition integration
- scheduled messages
- memory-aware replies
- knowledge-base lookup through agent tools

### 2. Memory system

The project keeps multiple memory layers:

- short-term memory
- core memory
- vector memory
- diary generation from conversation history

Current storage split:

- PostgreSQL stores conversation history, short-term memory, core memory, and diaries
- Qdrant stores vector memory

### 3. RAG with SDK-first design

The default RAG path now uses `atrinexus-rag-sdk` instead of continuing to expand a custom in-project RAG stack.

The current codebase now uses:

- `SdkRAGService`
- SDK-managed namespace per user
- separate Qdrant-backed RAG storage

### 4. Lightweight execution abilities

The assistant can perform basic workspace actions from chat:

- run commands
- read files
- search files
- write files
- replace text in files

Workspace modifications follow a preview-first flow:

- generate a preview / diff first
- require explicit confirmation before applying changes

This is intentionally lightweight. The goal is not to be a full AI IDE or a general-purpose autonomous agent platform.

### 5. Operational visibility

The service exposes:

- `/health`
- `/health/simple`
- `/metrics`

It is designed to sit behind nginx and works with external monitoring such as Prometheus and Grafana.

## Architecture Overview

### Service entry

- `run.py`
- `src/app/server.py`

### Main runtime path

- `src/conversation/message_handler.py`
- `src/conversation/context_builder.py`
- `src/agent_runtime/langchain_agent_service.py`
- `src/prompting/prompt_manager.py`
- `src/conversation/fast_path_router.py`

### Memory and diary

- `src/memory/memory_manager.py`
- `src/memory/memory_store.py`
- `src/features/diary_service.py`
- `src/platform_core/database.py`

### RAG

- `src/knowledge/rag_service.py`
- `src/knowledge/kb_tools.py`

Knowledge-base retrieval is now agent-driven:

- normal messages no longer go through front-loaded KB retrieval
- the agent uses KB tools on demand

### Vector storage

- `src/platform_core/vector_store/qdrant.py`

### AI services

- `src/ai/llm_service.py`
- `src/ai/embedding_service.py`
- `src/ai/model_manager.py`

## Tech Stack

- Python 3.12
- FastAPI
- LangChain 1.x
- PostgreSQL
- Qdrant
- `atrinexus-rag-sdk`
- WeCom / `wechatpy`
- APScheduler
- Prometheus client

## Project Structure

Source code is organized by capability domain, not by framework. `src/wecom/`
no longer occupies a top-level slot because WeCom is just one current
ingress — future Discord / Slack / CLI adapters would sit alongside it
under `ingress/`.

```text
AtriNexus/
├── run.py
├── pyproject.toml
├── requirements.txt
├── src/
│   ├── app/              # application assembly, startup
│   ├── ingress/          # WeCom callback, HTTP routers, middleware
│   ├── conversation/     # message orchestration, fast-path, reply cleaner
│   ├── agent_runtime/    # langchain adapter, runtime, middleware, tool guard
│   ├── prompting/        # prompt assembly + all prompt markdown resources
│   ├── memory/           # three-layer memory + context + updates
│   ├── knowledge/        # RAG service + KB agent tools
│   ├── ai/               # LLM, embedding, vision, web search, model mgr
│   ├── workspace/        # workspace capabilities
│   ├── platform_core/    # DB, session, token monitor, vector store, utils
│   ├── features/         # diary, web templates
│   └── tests/
├── data/
│   ├── config/
│   ├── database/
│   ├── vectordb_qdrant/
│   └── tasks.json
├── deployment/
└── docs/
```

See `docs/PROJECT_STRUCTURE.md` for the full rationale and module mapping.

## Configuration

The main runtime configuration lives in:

- `data/config/config.json`

The repository only keeps a sanitized template:

- `data/config/config.json.template`

The project expects non-sensitive runtime settings to remain in `config.json`.

Sensitive values now support environment-variable overrides:

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

For local development, copy:

- `.env.example -> .env`

Production should prefer systemd `Environment=` / `EnvironmentFile=` instead of relying on a checked-out `.env`.

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

Create your local runtime config from the template:

```bash
cp data/config/config.json.template data/config/config.json
```

Then fill in your real settings.

Optional: copy `.env.example` to `.env` and place secrets there instead of writing them back into `config.json`.

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

The project is intended to run behind nginx with a public WeCom callback endpoint.

Typical production layout includes:

- systemd-managed `run.py`
- systemd service file: `deployment/atrinexus.service`
- nginx reverse proxy
- `/health` and `/metrics` exposure
- WeCom callback routing

The VPS bootstrap script installs that same service name:

```bash
sudo bash deployment/setup_vps.sh
sudo systemctl start atrinexus
sudo systemctl status atrinexus
sudo journalctl -u atrinexus -f
```

## Important Notes

- The runtime path is now centered on Qdrant, `atrinexus-rag-sdk`, and LangChain.
- PostgreSQL is now the source of truth for conversation history, short-term memory, core memory, and diaries.
- Qdrant local state under `data/vectordb_qdrant/` is runtime data and is not tracked in git.
- KB lookup is now exposed as agent tools instead of a front-routed retrieval step on every normal message.
- The project intentionally stays lightweight instead of growing into a general-purpose agent platform.
- When the runtime structure changes, update `README.md` and `README.zh-CN.md` in the same change to keep architecture notes in sync.

## Who This Is For

AtriNexus is a better fit for:

- personal AI assistant experiments
- WeCom-based assistant deployment
- long-running memory-centric assistant use
- practical RAG + memory + tool integration

It is not intended to be:

- a generic agent platform
- a polished SaaS product
- a framework for every use case

## License

MIT. See [LICENSE](LICENSE).
