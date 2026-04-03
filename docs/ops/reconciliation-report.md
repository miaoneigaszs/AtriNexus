# 本地 / GitHub / 生产对齐报告

更新时间：2026-04-03

## 1. 三方基线

### 本地工作区

- 路径：`D:\learnsomething\AtriNexus`
- 当前已切换到回流分支：`reconcile/production-state`
- commit：`e01ef2db4aa6323bb43fde6b4286e76d16207e5f`
- 远端：`https://github.com/miaoneigaszs/AtriNexus.git`
- 生产快照本地路径：`backups/production/20260403-163132`

### 生产工作区

- 路径：`/opt/AtriNexus`
- 分支：`main`
- commit：`26d40b28739e4b07cc68b2231693e63c03d65eca`
- 远端：`https://github.com/KouriChat/KouriChat.git`

### 结论

本地与生产不是同一基线，且生产还挂错远端。

## 2. 当前已确认的不一致

### 仓库层面

1. 本地远端是 `miaoneigaszs/AtriNexus`
2. 生产远端是 `KouriChat/KouriChat`
3. 本地分支为 `master`
4. 生产分支为 `main`
5. 两边 commit 完全不同

### 工作树层面

生产存在以下状态：

- 大量 `M`
- 大量 `D`
- 大量 `??`
- `git diff --stat` 显示：`138 files changed, 1150 insertions(+), 30967 deletions(-)`

这说明生产并非标准部署产物，而是长期演化中的工作树。

### 数据层面

生产有真实配置、数据库、Chroma 向量数据，且仍在活跃使用。

### 部署层面

已确认：

- AtriNexus 服务监听 `127.0.0.1:8080`
- nginx 在主域名 `cln-nagisa.xyz` 下将以下路径反代到该服务：
  - `/api/wechat/callback`
  - `/memory`
  - `/kb-upload`
  - `/setting`
  - `/api/`

说明：

- 后续迁移部署方式时，不能只看 systemd，还必须同步纳入 nginx 配置变更

## 3. 当前风险判断

### 高风险

1. 任何直接从本地覆盖生产的行为，都可能破坏生产真实代码与数据兼容性
2. 任何基于当前本地代码的直接重构，都可能偏离现网真实运行逻辑
3. 任何未备份的数据迁移，都可能破坏近 30 天记忆资产

### 中风险

1. 生产继续在线直接改代码，会让对齐成本继续上升
2. 生产远端错误，可能导致后续误操作推送到错误仓库

## 4. 整改建议

### 第一阶段必须完成

1. 从生产读取完整 `git status --short`
2. 从生产读取 `git diff --stat`
3. 备份配置、数据库、向量数据
4. 将生产真实源码单独回流到 `reconcile/production-state`
5. 在本地基于该回流版本重建开发环境

当前状态：

- `git status --short` 已获取
- `git diff --stat` 已获取
- 服务器正式备份已执行
- 生产快照已下载到本地
- 本地回流分支已创建
- 代码回流提交尚未执行

### 第二阶段再做

1. 代码质量整治
2. 接口抽象
3. Qdrant 与 RAG SDK 迁移
4. LangGraph 迁移

## 5. 当前推荐基线

整改基线应以：

- `/opt/AtriNexus` 当前真实运行版本

作为事实起点，而不是当前本地仓库状态。

## 6. 下一步需要补齐的信息

1. 生产 `git diff --stat`
2. 生产 systemd 文件实物
3. 生产 nginx 配置
4. 生产环境变量来源
5. 生产配置与本地模板差异

当前已补齐：

- 生产 `git diff --stat`
- 生产 systemd 文件实物
- 生产 nginx 反代配置
- 生产配置结构检查结果
