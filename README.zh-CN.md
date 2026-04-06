# AtriNexus

[English](README.md) | [简体中文](README.zh-CN.md)

AtriNexus 是一个基于企业微信的个人 AI 助手，重点面向长期对话、记忆、知识库工具调用，以及轻量级工作区执行能力。

它不是一个泛化聊天机器人 Demo，而是一个服务于真实长期个人使用场景的项目。当前默认运行路径是：

- 企业微信作为聊天入口
- FastAPI 作为服务层
- SQLite 存储对话、记忆和日记数据
- Qdrant 存储向量记忆
- `atrinexus-rag-sdk` 负责知识库检索
- LangChain 负责轻量级 agent 与工具调用层

当前运行时已经不再是简单的“聊天 + 挂几个工具”，而是围绕以下骨架组织：

- `PromptManager` 负责分层 prompt 组装
- `ToolProfile` 负责会话级稳定工具暴露
- `FastPathRouter` 负责确定性文件/工具请求直达
- middleware 负责模型与工具调用治理

## 项目能做什么

- 企业微信对话处理
- 持久化短期记忆与核心记忆
- 基于对话历史的每日日记生成
- 知识库上传与检索
- 聊天中触发轻量命令与文件操作
- 健康检查与 Prometheus 指标
- 记忆、设置、知识库上传等 Web 页面

## 当前能力

### 1. 企业微信个人助手对话

项目围绕企业微信回调和长期个人对话场景构建。

支持：

- 普通文本对话
- 图像识别接入
- 定时消息
- 基于记忆的回复
- 基于知识库工具的按需查询

### 2. 记忆系统

项目目前维护多层记忆：

- 短期记忆
- 核心记忆
- 向量记忆
- 由对话沉淀生成的日记

当前存储拆分：

- SQLite 存对话历史、短期记忆、核心记忆和日记
- Qdrant 存向量记忆

### 3. SDK 优先的 RAG 路径

默认 RAG 路径已经切到 `atrinexus-rag-sdk`，不再继续把主项目堆成一套完整自实现 RAG 内核。

当前代码基线直接使用：

- `SdkRAGService`
- 每个用户一个 SDK namespace
- 独立的 Qdrant 支撑 RAG

### 4. 轻量执行能力

助手可以在聊天中执行基础工作区操作：

- 执行命令
- 读取文件
- 搜索文件
- 写文件
- 替换文件中的文本

文件修改默认走 preview-first 流程：

- 先生成预览 / diff
- 再等待用户明确确认后落盘

这部分是刻意保持轻量的。目标不是把项目做成 AI IDE 或通用 autonomous agent 平台。

### 5. 运行可观测性

服务暴露：

- `/health`
- `/health/simple`
- `/metrics`

适合放在 nginx 后面运行，并配合 Prometheus / Grafana 使用。

## 架构概览

### 服务入口

- `run.py`
- `src/wecom/server.py`

### 主运行链路

- `src/wecom/handlers/message_handler.py`
- `src/wecom/processors/context_builder.py`
- `src/services/agent/langchain_agent_service.py`
- `src/services/prompt_manager.py`
- `src/wecom/processors/fast_path_router.py`

### 记忆与日记

- `src/services/memory_manager.py`
- `src/services/memory_store.py`
- `src/services/diary_service.py`
- `src/services/database.py`

### RAG

- `src/services/rag_service.py`

知识库路径现在已经改成 agent 按需调用：

- 普通消息不再前置跑 KB 检索
- 由 agent 在需要时调用知识库工具

### 向量存储

- `src/services/vector_store/qdrant.py`

### AI 服务

- `src/services/ai/llm_service.py`
- `src/services/ai/embedding_service.py`
- `src/services/ai/model_manager.py`

## 技术栈

- Python 3.12
- FastAPI
- LangChain 1.x
- SQLite
- Qdrant
- `atrinexus-rag-sdk`
- 企业微信 / `wechatpy`
- APScheduler
- Prometheus client

## 项目结构

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
│   ├── vectordb_qdrant/
│   └── tasks.json
├── deployment/
└── docs/
```

## 配置

主运行配置位于：

- `data/config/config.json`

仓库中只保留脱敏模板：

- `data/config/config.json.template`

项目期望在这里配置企业微信、模型、embedding 及其他运行参数。

## 本地运行

### 1. 安装依赖

使用 `uv`：

```bash
uv sync
```

或使用 `pip`：

```bash
python -m pip install -r requirements.txt
```

### 2. 准备配置

从模板创建本地配置：

```bash
cp data/config/config.json.template data/config/config.json
```

然后填入真实参数。

### 3. 启动服务

```bash
python run.py
```

### 4. 验证

```bash
curl http://127.0.0.1:8080/health/simple
curl http://127.0.0.1:8080/health
```

## 部署说明

项目默认适合运行在 nginx 后面，并通过公网企业微信回调地址接收消息。

典型生产结构包括：

- 由 systemd 管理 `run.py`
- nginx 反向代理
- 暴露 `/health` 与 `/metrics`
- 企业微信回调转发

## 重要说明

- 当前运行主路径已经统一到 Qdrant、`atrinexus-rag-sdk` 和 LangChain。
- SQLite 仍然是对话历史、短期记忆、核心记忆和日记的事实来源。
- `data/vectordb_qdrant/` 属于本地运行状态数据，不纳入 Git 跟踪。
- 知识库查询已经从“每条普通消息前置检索”收口为 agent 工具按需调用。
- 项目刻意保持轻量，不朝通用 Agent 平台方向膨胀。

## 适合谁

AtriNexus 更适合：

- 个人 AI 助手实验
- 基于企业微信的助手部署
- 强调长期记忆的助手使用场景
- 实用型 RAG + memory + tool integration

它并不打算成为：

- 通用 Agent 平台
- 打磨完善的 SaaS 产品
- 面向所有场景的通用框架

## 协议

MIT。见 [LICENSE](LICENSE)。
