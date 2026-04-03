# AtriNexus 项目整改主计划

## 1. 背景与目标

AtriNexus 当前处于“生产可用、工程失控”的阶段。

已知事实：

- 服务器 `47.252.40.197` 上的版本可运行，且已积累接近 30 天个人记忆数据
- 本地代码与服务器代码、GitHub 仓库状态明显脱节
- 当前代码具备聊天、记忆、工具调用、RAG、监测等核心能力
- 项目维护流程不健康，服务器长期充当“唯一真实运行环境”
- 下一阶段目标不只是修补，而是完成架构升级与能力扩展

本计划的总体目标：

1. 收回生产代码与配置的控制权
2. 建立可持续维护的开发、测试、发布流程
3. 先整理现有代码，再实施框架与存储迁移
4. 将项目从“聊天机器人”演进为“可扩展 Agent 平台”

## 2. 当前问题分组

### 2.1 工程治理问题

- 本地、GitHub、服务器三者失配
- 生产版本未形成可审计资产
- 部署与发布流程缺乏标准化
- 依赖声明、入口、文档与实际运行形态不一致

### 2.2 代码质量问题

- 模块边界不稳定，部分编排逻辑仍然耦合
- 配置模型与实际使用存在不一致
- 异常处理过宽，存在静默降级
- 测试体系与当前实现脱节
- 日志、调试输出、安全卫生不统一

### 2.3 架构问题

- 当前流程编排高度自定义，长期演进成本偏高
- RAG 流程与底层向量存储耦合
- ChromaDB 不适合作为后续主线存储
- 业务层未与具体存储、RAG 实现有效隔离

### 2.4 产品演进问题

- 当前能力集中于聊天
- 缺乏统一的 Agent Runtime
- 未形成工具权限模型、沙箱模型、审计模型
- 新能力扩展容易污染既有业务链路

## 3. 整改原则

1. 生产优先
2. 数据优先
3. 先收口，再迁移
4. 先抽象，再替换
5. 小步切换，保留回滚

解释：

- 生产优先：任何重构不得先破坏现网
- 数据优先：30 天记忆和知识数据视为核心资产
- 先收口，再迁移：先解决失控维护流程，再做架构升级
- 先抽象，再替换：先定义接口，再换 LangGraph、Qdrant、RAG SDK
- 小步切换，保留回滚：避免一次性重写

## 4. 分阶段执行计划

## Phase 0：生产现状冻结

### 目标

在任何重构前，完整盘点并备份服务器当前状态。

### 任务

1. 记录服务器当前代码路径
2. 记录服务器当前 git 分支、commit、未提交修改
3. 备份配置文件
4. 备份 SQLite 数据库
5. 备份向量数据库目录
6. 记录 systemd 服务文件
7. 记录反向代理、开放端口、环境变量、启动命令
8. 记录 Python 版本与依赖版本

### 交付物

- `docs/ops/production-inventory.md`
- `docs/ops/production-runtime.md`
- `docs/ops/production-deploy.md`
- `backups/production-YYYY-MM-DD/`

### 验收标准

- 可以在不依赖记忆的情况下说清生产如何运行
- 具备独立的数据回滚基础
- 知道服务器代码与 GitHub 的精确差异起点

## Phase 1：代码与部署收口

### 目标

让 GitHub 重新成为源码事实源，服务器只作为部署目标。

### 任务

1. 从服务器回流当前真实代码
2. 建立 `reconcile/production-state` 分支
3. 比较服务器代码、本地代码、GitHub 主线差异
4. 将服务器上的真实修改以提交形式纳入版本管理
5. 建立明确分支策略
6. 明确开发、测试、发布、回滚流程
7. 统一依赖声明与项目入口

### 交付物

- `docs/ops/reconciliation-report.md`
- `docs/ops/branching-and-release.md`
- `docs/ops/local-dev-setup.md`

### 验收标准

- 服务器不再存在“只在线上”的代码
- 本地可以基于某个 commit 重建环境
- 发布流程可重复执行

## Phase 2：代码质量整治与架构收边

### 目标

在不大规模换栈的前提下，把现有系统整理成可迁移状态。

### 任务

1. 清理硬编码密钥、测试残留、调试输出
2. 修复高风险代码缺陷与配置不一致
3. 收敛异常处理与日志规范
4. 统一配置模型
5. 拆分业务编排与基础设施调用
6. 建立最小测试体系
7. 建立 CI 基础检查

### 推荐结构

- `src/app/entry`
- `src/app/domain`
- `src/app/orchestration`
- `src/app/infrastructure`
- `src/app/interfaces`
- `src/app/config`

### 先抽象的接口

- `ChatService`
- `MemoryStore`
- `VectorStore`
- `RAGService`
- `ToolExecutor`
- `AgentRuntime`

### 交付物

- 模块重组后的目录结构
- 冒烟测试
- 关键单元测试
- 基础静态检查脚本

### 验收标准

- 主消息链路可在本地稳定运行
- 存储与 RAG 不再直接写死在业务流程里
- 新旧实现切换具备接口承载能力

## Phase 3：RAG 与向量层迁移

### 目标

从 ChromaDB 迁移到 Qdrant，并接入自定义 RAG SDK。

### 任务

1. 定义 `VectorStore` 统一接口
2. 保留 `ChromaVectorStore` 作为旧实现
3. 新增 `QdrantVectorStore` 实现
4. 引入 `atrinexus-rag-sdk` 适配层
5. 实现知识库与中期记忆的迁移脚本
6. 做双写或可切换验证
7. 对比新旧召回结果与性能

### 集成原则

- 业务层只依赖 `RAGService` 接口
- 不在业务流程中直接依赖 SDK 内部模块
- 迁移期允许 `legacy` 与 `sdk` 双后端共存

### 交付物

- `src/app/infrastructure/vectorstores/qdrant_store.py`
- `src/app/infrastructure/rag/rag_sdk_adapter.py`
- `scripts/migrate_chroma_to_qdrant.py`
- `docs/migration/rag-and-vector-migration.md`

### 验收标准

- 新检索链路可独立运行
- 旧数据可迁
- 检索质量不明显退化
- 可以切换回旧链路

## Phase 4：LangGraph 编排迁移

### 目标

用 LangGraph 替代当前核心消息流程中的自定义编排层。

### 迁移范围

优先迁移：

1. 对话主链路
2. 记忆加载与更新
3. RAG 检索决策
4. 工具调用流程
5. 错误恢复 / fallback

### 典型图节点

- receive_message
- load_session
- load_memory
- route_intent
- retrieve_knowledge
- decide_tools
- run_tools
- generate_response
- persist_conversation
- update_memory
- emit_metrics

### 迁移策略

1. 先以现有流程图谱化
2. 搭建最小可运行 graph
3. 通过 feature flag 灰度切换
4. 保留旧流程作为回滚路径

### 交付物

- `src/app/orchestration/graph_runtime.py`
- `src/app/orchestration/graphs/chat_graph.py`
- `docs/architecture/langgraph-migration.md`

### 验收标准

- 单用户主链路由 graph 驱动
- 新旧链路可切换
- 监控与日志支持链路对比

## Phase 5：Agent Runtime 扩展

### 目标

将项目演进为具备命令执行、文件系统操作、计划执行等能力的智能体平台。

### 新能力范围

1. 命令行执行
2. 文件系统读写
3. 多工具编排
4. 任务计划与执行
5. 工作区上下文记忆
6. 人工确认机制
7. 审计日志

### 必须先有的安全约束

1. 工作目录白名单
2. 命令权限边界
3. 高风险操作确认
4. 输出截断与超时控制
5. 操作审计

### 交付物

- `src/app/tools/`
- `src/app/runtime/`
- `src/app/security/`
- `docs/architecture/agent-runtime.md`

### 验收标准

- 工具调用具备权限模型
- 关键操作具备人工确认
- Agent 能力扩展不污染聊天主链路

## 5. 技术路线建议

### 保留

- FastAPI 作为接口层
- SQLite 作为轻量会话/记录存储的过渡方案
- 企业微信接入层
- 监控与健康检查能力

### 替换

- 自定义主流程编排 -> LangGraph
- ChromaDB -> Qdrant
- 现有 RAG 流程实现 -> 自定义 RAG SDK 适配层

### 暂缓

- 不立即重构 Web UI
- 不先做多用户复杂协作
- 不在存储和编排未稳定前引入高风险 agent 工具

## 6. 风险与控制

### 风险 1：生产数据丢失

控制措施：

- 先备份
- 迁移脚本只读验证后再执行写入
- 每次迁移保留回滚快照

### 风险 2：服务器版本继续漂移

控制措施：

- 服务器回流 Git
- 冻结线上直接改代码的行为
- 统一发布入口

### 风险 3：迁移范围过大导致项目长期不可用

控制措施：

- 分阶段 feature flag
- 保留旧链路
- 单模块替换

### 风险 4：Agent 能力引入安全问题

控制措施：

- 最后做
- 先定义权限模型
- 默认拒绝高风险操作

## 7. 里程碑建议

### M1：收口与回流

周期：1-2 周

完成标准：

- 生产状态备份完成
- 服务器代码回流完成
- 本地环境可重建

### M2：代码整治

周期：2-3 周

完成标准：

- 高风险代码问题修复
- 接口抽象完成
- 测试和 CI 起步

### M3：向量与 RAG 迁移

周期：2-4 周

完成标准：

- Qdrant 与 RAG SDK 可切换
- 旧数据可迁

### M4：LangGraph 化

周期：2-4 周

完成标准：

- 主链路 graph 化
- 新旧链路灰度切换

### M5：Agent 扩展

周期：持续演进

完成标准：

- 工具系统、运行时、安全控制成型

## 8. 当前建议的执行顺序

严格按下面顺序执行：

1. 生产盘点与备份
2. 服务器代码回流 Git
3. 本地开发环境重建
4. 代码质量整治与接口抽象
5. Qdrant 与 RAG SDK 迁移
6. LangGraph 迁移
7. Agent Runtime 扩展

不要跳步。

## 9. 第一批立即执行项

1. 从服务器拉取真实代码状态
2. 备份生产数据库与向量数据
3. 形成生产运行清单
4. 输出 reconciliation 报告
5. 以服务器版本为基线修复本地环境

## 10. 结论

这个项目当前最需要的不是“马上重写”，而是“把真实运行状态收回来并纳入工程管理”。

整改成功的标志不是换成了 LangGraph 或 Qdrant，而是：

- 生产与仓库重新一致
- 本地可复现
- 数据可迁移
- 架构可扩展
- 未来新增 agent 能力不会再次把项目拖回失控状态
