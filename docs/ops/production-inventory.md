# 生产环境盘点

更新时间：2026-04-03

## 1. 主机与服务

- 服务器：`47.252.40.197`
- 登录用户：`root`
- 当前工作目录：`/root`
- Python 版本：`Python 3.12.3`
- systemd 服务：`atrinexus.service`
- 服务状态：`active (running)`
- 已连续运行：约 3 周
- 进程启动命令：`/opt/AtriNexus/.venv/bin/python run.py`

## 2. 网络监听

- 应用监听：`127.0.0.1:8080`
- 另有 `127.0.0.1:8000` 上的 `uvicorn` 进程
- nginx 对外监听：
  - `0.0.0.0:80`
  - `0.0.0.0:443`

当前判断：

- AtriNexus 主服务位于 nginx 反代后
- `8080` 是当前企微应用服务实际监听端口
- `8000` 属于另一套服务，需后续确认是否与当前项目相关

## 2.1 nginx 反代关系

已确认 `/etc/nginx/sites-enabled/wechat` 中与 AtriNexus 相关的路由：

- `https://cln-nagisa.xyz/api/wechat/callback` -> `http://127.0.0.1:8080/api/wechat/callback`
- `https://cln-nagisa.xyz/memory` -> `http://127.0.0.1:8080/memory`
- `https://cln-nagisa.xyz/kb-upload` -> `http://127.0.0.1:8080/kb-upload`
- `https://cln-nagisa.xyz/setting` -> `http://127.0.0.1:8080/setting`
- `https://cln-nagisa.xyz/api/` -> `http://127.0.0.1:8080/api/`

说明：

- AtriNexus 当前不是独立子域名，而是挂在主域名 `cln-nagisa.xyz` 下
- 这意味着后续切换服务端口或部署方式时，必须同步修改 nginx 路由配置

## 3. 部署目录

服务器 `/opt` 下已确认存在：

- `/opt/AtriNexus`
- `/opt/sofce`
- `/opt/LangBot`

当前 AtriNexus 服务工作目录为：

- `/opt/AtriNexus`

## 4. AtriNexus 仓库状态

在 `/opt/AtriNexus` 下确认：

- 当前分支：`main`
- 当前 commit：`26d40b28739e4b07cc68b2231693e63c03d65eca`
- 工作树：非干净，存在大量已修改、已删除、未跟踪文件
- 远端仓库：`https://github.com/KouriChat/KouriChat.git`
- `git diff --stat` 规模：`138 files changed, 1150 insertions(+), 30967 deletions(-)`

这是当前最关键的治理问题：

- 服务器部署目录名是 `AtriNexus`
- 但 git 远端仍指向 `KouriChat/KouriChat`
- 且服务器存在长期未回流的源码改动

结论：

- 服务器不是“可复现部署产物”
- 而是“长期在线修改后的运行工作树”

## 5. 数据目录盘点

### `data/config`

已确认存在：

- `config.json`
- `config.json.template`
- `config.json.template.bak`
- `backups/`

当前生产配置结构检查结果：

- 顶层键：`categories`
- 当前包含的 category：
  - `auth_settings`
  - `behavior_settings`
  - `embedding_settings`
  - `intent_recognition_settings`
  - `knowledge_base_settings`
  - `llm_settings`
  - `media_settings`
  - `network_search_settings`
  - `schedule_settings`
  - `system_performance_settings`
  - `user_settings`
  - `wecom_settings`

说明：

- 生产配置结构与当前本地模板在 category 层级上基本一致
- 这有利于后续以生产版本为基线重建本地环境

### `data/database`

已确认存在：

- `chat_history.db`
- `chat_history.db-shm`
- `chat_history.db-wal`

文件体量：

- `chat_history.db` 约 `479232` bytes
- `chat_history.db-wal` 约 `4124152` bytes

说明：

- 当前数据库处于活跃 WAL 模式
- 备份时必须按 SQLite WAL 场景处理

### `data/vectordb`

已确认存在：

- `chroma.sqlite3`
- 至少两个 collection 目录

文件体量：

- `chroma.sqlite3` 约 `6.0 MB`

说明：

- 生产环境当前仍在使用 ChromaDB
- 后续迁移到 Qdrant 时必须先做只读盘点和导出验证

## 6. 初步结论

当前生产环境具备以下特征：

1. 服务稳定运行
2. 生产数据持续积累
3. 代码运行基线未纳入当前 GitHub 仓库主线
4. 服务器源码存在大量未提交演化
5. 当前整改必须以 `/opt/AtriNexus` 的真实状态为起点
6. 当前生产代码与历史 KouriChat 基线之间存在大规模重构残留

## 7. 下一步

1. 读取 `/opt/AtriNexus` 的 `git diff --stat`
2. 读取服务器 systemd 服务文件内容
3. 读取 nginx 站点配置
4. 对生产配置与本地配置模板做结构比较
5. 输出正式 reconciliation 报告

## 8. 当前 systemd 实物

生产机 `/etc/systemd/system/atrinexus.service` 当前配置要点：

- `WorkingDirectory=/opt/AtriNexus`
- `ExecStart=/opt/AtriNexus/.venv/bin/python run.py`
- `Environment=WECOM_PORT=8080`
- `Environment=PYTHONUNBUFFERED=1`
- `Environment=PYTHONIOENCODING=utf-8`

当前未看到显式的 `WECOM_HOST` 设置。
