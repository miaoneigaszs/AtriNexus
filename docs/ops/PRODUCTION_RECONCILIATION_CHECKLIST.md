# 生产收口与对齐清单

## 目的

本清单用于把以下三者重新对齐：

1. 本地工作区
2. GitHub 仓库
3. 服务器 `47.252.40.197`

在完成本清单前，不执行大规模重构。

## A. 服务器基础盘点

需要确认并记录：

1. 项目实际部署目录
2. 当前服务名
3. 当前运行用户
4. Python 版本
5. 虚拟环境路径
6. 启动命令
7. 端口监听情况
8. 反向代理配置
9. systemd 配置
10. 环境变量来源

建议命令：

```bash
pwd
whoami
python3 --version
which python3
systemctl status atrinexus --no-pager
journalctl -u atrinexus -n 100 --no-pager
ss -ltnp
```

## B. 服务器代码状态盘点

需要确认并记录：

1. 当前 git 分支
2. 当前 commit hash
3. 是否存在未提交修改
4. 是否存在未跟踪文件
5. 是否存在服务器本地直接改动

建议命令：

```bash
git rev-parse --show-toplevel
git rev-parse HEAD
git branch --show-current
git status --short
git remote -v
git log --oneline -n 20
```

## C. 服务器数据盘点

需要确认并记录：

1. SQLite 路径
2. 向量数据库路径
3. 配置文件路径
4. 日志目录
5. 备份目录

需要备份的核心对象：

1. `data/config/config.json`
2. `data/database/`
3. `data/vectordb/`
4. `logs/`

## D. 回流原则

### 规则 1

服务器代码如果与 GitHub 不一致，必须先回流，再谈重构。

### 规则 2

服务器上的真实运行代码优先级高于本地猜测。

### 规则 3

回流时不覆盖生产数据目录，只回流源码与部署配置。

## E. 推荐回流流程

1. 在服务器项目目录执行 `git status --short`
2. 如有未提交修改，创建回流分支
3. 提交服务器上的真实改动
4. 推送到 GitHub 新分支，例如 `reconcile/production-state`
5. 本地拉取该分支
6. 与当前本地代码做 diff
7. 输出 reconciliation 报告
8. 以服务器真实版本为整改基线

## F. Reconciliation 报告应包含

1. 服务器与本地的文件差异
2. 服务器独有修改
3. 本地独有修改
4. 文档与实际运行不一致项
5. 配置模型与实际配置不一致项
6. 依赖声明与实际环境不一致项
7. 高风险问题列表

## G. 在完成回流前禁止执行的事情

1. 禁止直接做 LangGraph 重构
2. 禁止直接替换 Chroma 为 Qdrant
3. 禁止直接迁移记忆数据
4. 禁止新增高风险 Agent 工具
5. 禁止依据当前本地代码做“全量重写”

## H. 第一阶段完成标准

满足以下条件才算第一阶段完成：

1. 服务器代码已纳入版本控制
2. 本地可基于服务器真实代码重建环境
3. 生产配置、数据库、向量数据已备份
4. 生产服务运行方式已有文档
5. 后续整改以同一基线推进
