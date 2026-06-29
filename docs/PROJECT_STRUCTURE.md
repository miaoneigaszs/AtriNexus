# AtriNexus 项目架构

AtriNexus 是面向企业微信的智能伙伴服务。当前代码按能力域组织，而不是按框架或历史迁移阶段组织：接入层负责把外部消息带进来，conversation 层编排对话，agent_runtime 执行工具调用和模型循环，memory / knowledge / ai / platform_core 提供底层能力。

## 运行链路

1. `run.py` 读取 `WECOM_HOST` / `WECOM_PORT`，确保运行目录存在，然后调用 `src.app.server.start_server()`。
2. `src/app/server.py` 创建 FastAPI 应用，配置日志、鉴权、指标、健康检查，并挂载 `src.ingress.routers` 中的路由。
3. 企业微信回调进入 `src.ingress.routers.callback`，解析消息后交给 `MessageHandler`。
4. `src.conversation.message_handler.MessageHandler` 处理命令、待确认动作、FastPath 状态和普通聊天，并负责落库与记忆更新触发。
5. 普通聊天进入 `src.agent_runtime.agent_service.AgentService`，由 agent loop 组装上下文、调用 provider、执行工具、处理取消和 follow-up 队列。
6. 模型访问通过 `src.ai.providers.openai_compat.OpenAICompatProvider` 和 `src.ai.stream` 完成；后台摘要、日记和定时任务仍使用 `src.ai.llm_service.LLMService`。
7. 长期记忆由 `src.memory` 管理，知识库检索由 `src.knowledge` 管理，数据库、HTTP 池、指标、向量库等基础设施在 `src.platform_core`。

## 顶层文件结构

```text
AtriNexus/
├── run.py                  # 服务启动入口
├── pyproject.toml          # Python 项目依赖与开发依赖
├── requirements.txt        # VPS 部署使用的 pip 依赖清单
├── README.md               # 英文说明
├── README.zh-CN.md          # 中文说明
├── deployment/             # systemd 与 VPS 初始化脚本
├── data/                   # 运行数据与配置模板；敏感/生成数据被 .gitignore 排除
├── src/                    # 运行源码
├── tests/                  # pytest 测试集
└── docs/PROJECT_STRUCTURE.md
```

## `src/` 能力域

| 目录 | 职责 |
|---|---|
| `src/app/` | FastAPI 应用装配、日志、中间件、健康检查、服务启动。 |
| `src/ingress/` | 外部入口层，包括企业微信客户端、回调路由、HTTP 管理接口、去重中间件和调度器。 |
| `src/conversation/` | 对话编排层，包括消息处理、命令处理、FastPath 状态、上下文构建、图片处理和回复清理。 |
| `src/agent_runtime/` | Agent 运行时，包括流式 agent loop、工具目录、工具护栏、hooks、上下文压缩、用户 run 状态、todo/clarify/trajectory。 |
| `src/prompting/` | 系统 prompt、模式 prompt、工具说明和 prompt 组装。 |
| `src/memory/` | 用户短期记忆、核心记忆、向量记忆、上下文拼装和记忆更新。 |
| `src/knowledge/` | 知识库 RAG 服务和可被 agent 调用的知识库工具。 |
| `src/ai/` | 模型能力登记、LLM 后台服务、OpenAI 兼容 provider、SSE 流解析、embedding、图像识别和联网搜索。 |
| `src/platform_core/` | 共享基础设施：同步数据库 session、SQLAlchemy 模型、会话状态、HTTP client 池、指标、限流状态、token 监控、时间工具和 Qdrant 向量存储。 |
| `src/workspace/` | 工作区搜索能力。 |
| `src/features/` | 独立业务功能，目前包含日记服务和简单 Web 模板。 |

## 数据与配置

运行配置来自 `data/config/config.json`，仓库只提交 `data/config/config.json.template`。敏感值和部署环境相关值通过环境变量提供，核心变量包括：

- `ATRINEXUS_DATABASE_URL`
- `ATRINEXUS_LLM_API_KEY`
- `WECOM_CORP_ID`
- `WECOM_AGENT_ID`
- `WECOM_SECRET`
- `WECOM_TOKEN`
- `WECOM_ENCODING_AES_KEY`

数据库当前使用同步 SQLAlchemy + PostgreSQL。`postgresql://` 和 `postgresql+asyncpg://` 输入都会转换为同步 `postgresql+psycopg://` URL。

## 测试与验证

测试位于顶层 `tests/`，不再混在运行包 `src/` 中。`tests/conftest.py` 负责把项目根目录加入 import path。

常用验证命令：

```powershell
.venv\Scripts\python.exe -m pytest -q
uv lock --check
python -m compileall -q run.py src tests
git diff --check
```

## 部署同步

服务器通过 git 同步代码。推荐流程：

1. 本地完成修改并通过测试。
2. 提交到当前分支并推送远端。
3. 通过 `ssh root@47.252.40.197` 进入服务器，在 `/opt` 下的项目目录执行 `git pull`。
4. 如依赖或服务入口变更，运行部署脚本或重启 systemd 服务。

部署目录和服务名以服务器上的 `/opt/*` 实际目录与 `deployment/atrinexus.service` 为准。