# AtriNexus

[English](README.md) | [简体中文](README.zh-CN.md)

AtriNexus 是一个基于企业微信的个人 AI 助手，重点面向长期对话、多层记忆、知识库工具调用和安全的工作区操作。

它面向真实的长期个人使用场景，不是一个通用聊天机器人 Demo。生产部署主链路基于：

- 企业微信作为聊天入口
- FastAPI 作为服务层
- PostgreSQL 存储对话、记忆与日记
- Qdrant 存储向量记忆
- `atrinexus-rag-sdk` 负责知识库检索

## 核心能力

- 企业微信长期对话，带持久化记忆与每日日记
- 知识库上传与按需检索（agent 工具按需触发）
- 安全的工作区操作：分页读文件、glob 与全文搜索、预览式文件修改
- 显式的待办工具和澄清工具支持多步任务
- 会话级能力档位 + 分层 prompt 组装
- 运行可观测：`/health` / `/health/simple` / `/metrics`，以及可选的轨迹落盘

## 能力详情

### 1. 企业微信个人助手对话

- 自然文本对话
- 图像识别接入
- 定时消息
- 基于记忆的回复
- 基于知识库工具的按需查询

### 2. 多层记忆系统

- 短期对话历史
- 核心记忆（长期事实）
- 向量记忆（语义检索）
- 基于对话沉淀的每日日记

存储拆分：

- PostgreSQL —— 对话历史、短期记忆、核心记忆、日记
- Qdrant —— 向量记忆

### 3. 知识库检索

- `SdkRAGService` 基于 `atrinexus-rag-sdk`
- 每个用户一个 SDK namespace
- 独立的 Qdrant 实例承载 RAG
- 检索由 agent 驱动：普通消息不再前置跑 KB 检索，agent 需要时再调 KB 工具

### 4. 工作区操作

助手在聊天中可执行以下操作：

- **读取** —— `read_file` 输出带行号，`offset` / `limit` 分页读长文件
- **浏览** —— `list_directory`、`search_files` 按文本搜、`glob` 按模式搜路径
- **修改** —— `preview_edit_file` 精确替换、`preview_write_file` 整文件重写、`preview_append_file` 头尾追加、`rename_path` 重命名 / 移动
- **执行** —— 只读命令管道（`find`、`du`、`wc`、`stat`、`tree` 等）直接放行；其他命令一律要用户确认

所有文件修改走 preview-first 流程：先生成 diff，用户回复"通过" / "确认"后再落盘。

### 5. 任务编排

- **Todo 工具** —— 会话级待办清单，由 agent 跨轮维护；状态随 system prompt 下发，因此不会被上下文压缩清除
- **Clarify 工具** —— 请求歧义时，agent 向用户发出澄清问题并结束本轮 run；用户的下一条消息作为答案再次进入 agent loop

### 6. 运行可观测性

- `/health` —— 完整健康检查（数据库 / Qdrant / RAG SDK / 企微凭证）
- `/health/simple` —— 轻量存活探针
- `/metrics` —— Prometheus 指标（请求数、token 用量、速率限制状态）
- 轨迹落盘（可选）—— 每轮 JSONL 记录用户消息、助手回复、工具事件、路由元数据（`fast_path_hit`、`intent`）

## 架构概览

### 服务入口

- `run.py`
- `src/app/server.py`

### 对话主链路

- `src/conversation/message_handler.py` —— 入口编排（去重、待审批、fast-path、agent loop）
- `src/conversation/context_builder.py` —— 每轮记忆 / 模式组装
- `src/conversation/fast_path_router.py` —— 确定性回复的短路径
- `src/prompting/prompt_manager.py` —— 分层 prompt 组装（静态壳 + 运行时能力快照 + 风格 + 记忆）

### Agent 运行时

- `src/agent_runtime/agent_service.py` —— run 生命周期、取消、follow-up 队列
- `src/agent_runtime/agent_loop.py` —— 工具调用循环与流式响应
- `src/agent_runtime/tool_catalog.py` —— 声明式工具注册，按档位暴露
- `src/agent_runtime/tool_profiles.py` —— 能力档位（`chat` / `workspace_read` / `workspace_edit` / `workspace_exec` / `full`）
- `src/agent_runtime/agent_tool_guard.py` —— 工具调用校验、路径修复、loop guard、结果整形
- `src/agent_runtime/hooks.py` —— 四个扩展 hook（`before_tool_call` / `after_tool_call` / `transform_context` / `on_response`）
- `src/agent_runtime/context_engine.py` —— 可插拔的上下文窗口压缩
- `src/agent_runtime/user_runtime.py` —— per-user run 认领、取消信号、follow-up 队列
- `src/agent_runtime/todo_store.py` —— 会话级 todo 状态
- `src/agent_runtime/clarify_store.py` —— Run 中 clarify 信号

### Provider 层

- `src/ai/providers/openai_compat.py` —— OpenAI 兼容流式客户端
- `src/ai/stream.py` —— SSE 解析与 tool-call 累积
- `src/ai/llm_service.py`、`src/ai/embedding_service.py`、`src/ai/model_manager.py` —— 模型协调

### 记忆与日记

- `src/memory/memory_manager.py`
- `src/memory/memory_store.py`
- `src/features/diary_service.py`
- `src/platform_core/database.py`

### 知识库

- `src/knowledge/rag_service.py`
- `src/knowledge/kb_tools.py`

### 工作区运行时

- `src/agent_runtime/runtime.py` —— 文件 I/O、搜索、命令执行策略、预览变更跟踪

### 向量存储

- `src/platform_core/vector_store/qdrant.py`

## 技术栈

- Python 3.12
- FastAPI
- PostgreSQL
- Qdrant
- `atrinexus-rag-sdk`
- 企业微信 / `wechatpy`
- APScheduler
- Prometheus client
- httpx

## 项目结构

源码按能力域分层，不按框架分层。

```text
AtriNexus/
├── run.py
├── pyproject.toml
├── requirements.txt
├── src/
│   ├── app/              # 应用装配与启动
│   ├── ingress/          # 企微回调、HTTP 路由、中间件
│   ├── conversation/     # 消息编排、快速通道、回复清理
│   ├── agent_runtime/    # agent loop、工具目录、hook、context engine
│   ├── prompting/        # prompt 组装与 prompt markdown 资源
│   ├── memory/           # 三层记忆 + 上下文 + 更新编排
│   ├── knowledge/        # RAG 服务 + KB agent 工具
│   ├── ai/               # Provider 适配与流式解析
│   ├── workspace/        # 工作区能力
│   ├── platform_core/    # 数据库、会话、token 监控、向量存储、工具函数
│   ├── features/         # 日记、web 模板
│   └── tests/
├── data/
│   ├── config/
│   ├── database/
│   ├── vectordb_qdrant/
│   └── tasks.json
├── deployment/
└── docs/
```

详见 `docs/PROJECT_STRUCTURE.md`。

## 配置

运行配置位于 `data/config/config.json`，仓库中只保留脱敏模板 `data/config/config.json.template`。

敏感值通过环境变量注入：

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

可选运行开关：

- `ATRINEXUS_FAST_PATH_INTENT` —— `full`（默认）或 `disabled`。设为 `disabled` 时跳过确定性 fast-path，消息全部交 agent loop 处理，方便 A/B 对比。
- `ATRINEXUS_TRAJECTORY_PATH` —— JSONL 文件绝对路径。设定后每轮会追加一条记录（用户消息、助手回复、工具事件，以及路由元数据 `fast_path_hit` / `intent`）。
- `ATRINEXUS_AGENT_CONTEXT_LENGTH` —— 压缩器使用的上下文窗口大小（默认 `32000`）。
- `ATRINEXUS_AGENT_MAX_ITERATIONS` —— 每轮最大工具调用迭代数（默认 `12`）。

本地开发可 `cp .env.example .env`。生产部署优先使用 systemd 的 `Environment=` / `EnvironmentFile=`，不依赖仓库里的 `.env`。

## 本地运行

### 1. 安装依赖

使用 `uv`：

```bash
uv sync
```

或 `pip`：

```bash
python -m pip install -r requirements.txt
```

### 2. 准备配置

```bash
cp data/config/config.json.template data/config/config.json
```

填入运行参数。密钥可放在 `.env` 里。

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

项目默认运行在 nginx 后面，接收公网企微回调。典型生产布局：

- 由 systemd 管理 `run.py`
- systemd unit 文件：`deployment/atrinexus.service`
- nginx 反向代理
- 向监控栈暴露 `/health` 与 `/metrics`
- 企业微信回调转发到服务

VPS 一键引导：

```bash
sudo bash deployment/setup_vps.sh
sudo systemctl start atrinexus
sudo systemctl status atrinexus
sudo journalctl -u atrinexus -f
```

## 适合谁

AtriNexus 适合：

- 个人 AI 助手部署
- 基于企业微信的助手场景
- 强调长期记忆的助手使用
- 实用型 RAG + memory + tool integration

它不打算成为通用 Agent 平台或面向所有场景的 SaaS 框架。

## 协议

MIT。详见 [LICENSE](LICENSE)。
