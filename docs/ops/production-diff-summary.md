# 生产快照与当前仓库差异摘要

更新时间：2026-04-03

## 1. 比对对象

### 当前仓库

- 仓库：`https://github.com/miaoneigaszs/AtriNexus.git`
- 当前分支：`reconcile/production-state`
- 当前基线 commit：`e01ef2db4aa6323bb43fde6b4286e76d16207e5f`

### 生产快照

- 来源：`/opt/AtriNexus`
- 快照目录：`backups/production/20260403-163132/extracted`
- 生产 commit：`26d40b28739e4b07cc68b2231693e63c03d65eca`
- 生产分支：`main`
- 生产 remote：`https://github.com/KouriChat/KouriChat.git`

## 2. 总体结论

生产快照不是当前仓库上的少量热修版本，而是一套从旧历史演进出来的“大规模改造后状态”。

已知证据：

- 生产 `git diff --stat`：
  - `138 files changed`
  - `1150 insertions(+)`
  - `30967 deletions(-)`

这说明：

1. 生产代码与旧 KouriChat 历史之间发生了大规模裁剪和重组
2. 当前 GitHub 仓库与生产版本之间不能简单用“补几个文件”解决
3. 后续必须以“回流生产状态”为前提建立新基线

## 3. 关键文件确认结果

以下文件在当前仓库与生产快照中均不相同：

- `README.md`
- `run.py`
- `requirements.txt`
- `pyproject.toml`
- `data/config/__init__.py`
- `src/services/ai/llm_service.py`
- `src/services/memory_manager.py`
- `src/wecom/server.py`

结论：

- 不只是边缘文件不同
- 主入口、依赖、配置、核心服务、服务入口都不同
- 生产快照应视为另一条待合流的基线

## 4. 差异类型分组

### 4.1 生产快照额外包含的内容

主要包括：

- 旧文档与辅助脚本
- 更完整的人设与资源目录
- 一些历史遗留文件
- 部分运行时数据与素材文件

说明：

- 其中一部分属于应回流的源码/文档资产
- 另一部分属于数据或资源，不应直接进主线源码提交

### 4.2 当前仓库额外包含的内容

当前仓库额外内容主要是：

- 本地备份目录
- 本次整改文档
- 已下载的生产快照副本

这部分不构成生产逻辑差异。

## 5. 回流建议

不要直接把 `backups/production/.../extracted` 整体覆盖到当前仓库。

建议分三层处理：

### 第一层：必须回流的源码

- `run.py`
- `requirements.txt`
- `pyproject.toml`
- `data/config/*`
- `src/services/*`
- `src/wecom/*`
- `src/utils/*`
- `src/web/*`
- `deployment/*`

### 第二层：待判断的资源和文档

- `data/avatars/*`
- `data/mode/*`
- `archive/*`
- 根目录零散文档

### 第三层：绝不进主线的运行数据

- `data/database/*`
- `data/vectordb/*`
- `logs/*`
- 任何生产配置实值文件

## 6. 推荐回流策略

采用两步法：

### Step A：建立“生产源码镜像提交”

目标：

- 把生产源码状态完整纳入 Git 视野
- 但不把数据库、向量库、日志带进仓库

### Step B：在该分支上做“规范化整理”

目标：

- 把生产快照中的历史残留与当前仓库结构统一
- 建立后续可维护基线

## 7. 当前判断

现阶段不应直接开始：

1. LangGraph 重构
2. Qdrant 替换
3. RAG SDK 接入
4. Agent 工具扩展

原因不是技术上不能做，而是当前源码基线还没有完成回流和统一。

## 8. 下一步

下一步应进入“生产源码镜像提交”准备：

1. 生成待导入路径白名单
2. 排除运行数据目录
3. 将生产源码导入 `reconcile/production-state`
4. 提交第一版回流快照
