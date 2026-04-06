# AtriNexus

[English](README.md) | [简体中文](README.zh-CN.md)

AtriNexus is a WeCom-based personal AI assistant for persistent conversation, memory, knowledge-base tooling, and lightweight workspace actions.

It is designed for a real long-running personal usage scenario rather than a generic chatbot demo. The current runtime path is:

- WeCom as the chat entrypoint
- FastAPI as the service layer
- SQLite for conversation, memory, and diary data
- Qdrant for vector memory storage
- `atrinexus-rag-sdk` for knowledge-base retrieval
- LangChain for the lightweight agent/tool layer

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

- SQLite stores conversation history, short-term memory, core memory, and diaries
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
- `src/wecom/server.py`

### Main runtime path

- `src/wecom/handlers/message_handler.py`
- `src/wecom/processors/context_builder.py`
- `src/services/agent/langchain_agent_service.py`

### Memory and diary

- `src/services/memory_manager.py`
- `src/services/memory_store.py`
- `src/services/diary_service.py`
- `src/services/database.py`

### RAG

- `src/services/rag_service.py`

### Vector storage

- `src/services/vector_store/qdrant.py`

### AI services

- `src/services/ai/llm_service.py`
- `src/services/ai/embedding_service.py`
- `src/services/ai/model_manager.py`

## Tech Stack

- Python 3.12
- FastAPI
- LangChain 1.x
- SQLite
- Qdrant
- `atrinexus-rag-sdk`
- WeCom / `wechatpy`
- APScheduler
- Prometheus client

## Project Structure

```text
AtriNexus/
├── run.py
├── pyproject.toml
├── requirements.txt
├── src/
│   ├── services/
│   │   ├── agent/
│   │   ├── ai/
│   │   ├── vector_store/
│   │   ├── database.py
│   │   ├── diary_service.py
│   │   ├── llm_service.py
│   │   ├── memory_manager.py
│   │   ├── memory_store.py
│   │   ├── rag_service.py
│   │   └── session_service.py
│   ├── utils/
│   └── wecom/
├── data/
│   ├── config/
│   ├── database/
│   └── tasks.json
├── deployment/
└── docs/
```

## Configuration

The main runtime configuration lives in:

- `data/config/config.json`

The repository only keeps a sanitized template:

- `data/config/config.json.template`

The project expects WeCom, model, embedding, and other runtime settings to be configured there.

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
- nginx reverse proxy
- `/health` and `/metrics` exposure
- WeCom callback routing

## Important Notes

- The runtime path is now centered on Qdrant, `atrinexus-rag-sdk`, and LangChain.
- SQLite remains the source of truth for conversation history, short-term memory, core memory, and diaries.
- The project intentionally stays lightweight instead of growing into a general-purpose agent platform.

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
