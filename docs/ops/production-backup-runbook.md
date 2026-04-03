# 生产备份执行手册

适用对象：`47.252.40.197` 上的 `/opt/AtriNexus`

更新时间：2026-04-03

## 1. 目标

在整改前完成以下对象的可回滚备份：

1. 生产配置
2. SQLite 会话与记忆数据库
3. Chroma 向量数据库
4. 当前源码工作树快照
5. systemd 与 nginx 配置

## 2. 已确认的核心路径

- 代码目录：`/opt/AtriNexus`
- 配置文件：`/opt/AtriNexus/data/config/config.json`
- SQLite 数据库：`/opt/AtriNexus/data/database/chat_history.db`
- SQLite WAL 文件：`/opt/AtriNexus/data/database/chat_history.db-wal`
- SQLite SHM 文件：`/opt/AtriNexus/data/database/chat_history.db-shm`
- Chroma 主文件：`/opt/AtriNexus/data/vectordb/chroma.sqlite3`
- 服务文件：`/etc/systemd/system/atrinexus.service`
- nginx 站点文件：`/etc/nginx/sites-enabled/wechat`

## 3. 备份原则

1. 先备份，再回流，再重构
2. 数据备份与源码快照分开保存
3. 保留时间戳目录
4. 先做文件级快照，不做任何迁移写入

## 4. 推荐备份目录

```bash
/root/backups/atrinexus/YYYYMMDD-HHMMSS/
```

## 5. 执行步骤

### 5.1 创建备份目录

```bash
TS=$(date +%Y%m%d-%H%M%S)
BASE=/root/backups/atrinexus/$TS
mkdir -p "$BASE"
mkdir -p "$BASE/runtime" "$BASE/data" "$BASE/code"
```

### 5.2 备份运行时配置

```bash
cp /etc/systemd/system/atrinexus.service "$BASE/runtime/"
cp /etc/nginx/sites-enabled/wechat "$BASE/runtime/"
systemctl status atrinexus --no-pager > "$BASE/runtime/systemctl-status.txt" || true
journalctl -u atrinexus -n 200 --no-pager > "$BASE/runtime/journal-tail.txt" || true
ss -ltnp > "$BASE/runtime/listen-ports.txt" || true
```

### 5.3 备份源码快照

```bash
cd /opt/AtriNexus
git rev-parse HEAD > "$BASE/code/git-head.txt"
git branch --show-current > "$BASE/code/git-branch.txt"
git remote -v > "$BASE/code/git-remote.txt"
git status --short > "$BASE/code/git-status-short.txt"
git diff --stat > "$BASE/code/git-diff-stat.txt"
tar --exclude='.venv' --exclude='__pycache__' -czf "$BASE/code/working-tree.tar.gz" .
```

### 5.4 备份生产数据

```bash
cp /opt/AtriNexus/data/config/config.json "$BASE/data/"
cp -a /opt/AtriNexus/data/database "$BASE/data/"
cp -a /opt/AtriNexus/data/vectordb "$BASE/data/"
```

说明：

- 当前 SQLite 使用 WAL 模式，因此备份整个 `data/database/` 目录，不单拷主库文件
- Chroma 备份整个 `data/vectordb/`，不要只拷 `chroma.sqlite3`

### 5.5 生成摘要文件

```bash
{
  echo "timestamp=$TS"
  echo "host=$(hostname)"
  echo "python=$(python3 --version 2>&1)"
  echo "service=atrinexus"
  echo "code_dir=/opt/AtriNexus"
} > "$BASE/backup-summary.txt"
```

## 6. 备份完成后的校验

检查以下文件是否存在：

```bash
find "$BASE" -maxdepth 3 -type f | sort
```

最少应看到：

- `runtime/atrinexus.service`
- `runtime/wechat`
- `code/working-tree.tar.gz`
- `code/git-status-short.txt`
- `data/config.json`
- `data/database/chat_history.db`
- `data/vectordb/chroma.sqlite3`

## 7. 风险说明

1. 该备份是文件级快照，不是数据库逻辑导出
2. 对当前整改阶段已足够，但后续迁移到 Qdrant 前，仍建议再做一次只读迁移前备份
3. 不建议在未保存备份目录前执行任何代码回流和数据迁移
