# 生产源码回流执行手册

适用对象：`47.252.40.197:/opt/AtriNexus`

更新时间：2026-04-03

## 1. 目标

将生产服务器上的真实运行源码纳入当前 GitHub 项目治理范围。

当前已知问题：

- 生产目录名为 `AtriNexus`
- 但 `origin` 指向 `KouriChat/KouriChat.git`
- 工作树存在大量未提交修改

因此不能直接 `git pull` 或强行覆盖。

## 2. 回流原则

1. 先备份，再回流
2. 先保留生产状态，不做清理式操作
3. 回流阶段不处理数据目录
4. 回流的目标是“保存真实源码状态”，不是立即整理代码

## 3. 推荐方案

推荐使用“源码压缩包 + 本地新分支重建”的方式回流，而不是直接在生产机上改 remote 并强推。

原因：

1. 当前生产 git 历史不可信
2. remote 明显错误
3. 直接在生产机上改 Git 关系，误操作风险高

## 4. 执行路径

### 方案 A：推荐

1. 在生产机生成源码快照压缩包
2. 下载到本地
3. 在本地创建 `reconcile/production-state`
4. 将快照覆盖到本地新分支工作区
5. 提交为一次“生产状态回流提交”
6. 推送到当前 GitHub 正确仓库

### 方案 B：不推荐但可选

1. 在生产机修正 remote
2. 新建回流分支
3. 提交当前工作树
4. 直接 push 到当前 GitHub

不推荐原因：

- 生产机直接写 GitHub 风险更高
- 容易把错误 remote、错误历史、错误内容一起推上去

## 5. 方案 A 详细步骤

### 5.1 生产机生成源码快照

```bash
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p /root/backups/atrinexus/$TS/code
cd /opt/AtriNexus
tar --exclude='.venv' --exclude='__pycache__' -czf /root/backups/atrinexus/$TS/code/working-tree.tar.gz .
git status --short > /root/backups/atrinexus/$TS/code/git-status-short.txt
git diff --stat > /root/backups/atrinexus/$TS/code/git-diff-stat.txt
git rev-parse HEAD > /root/backups/atrinexus/$TS/code/git-head.txt
git remote -v > /root/backups/atrinexus/$TS/code/git-remote.txt
```

### 5.2 下载快照到本地

建议方式：

```bash
scp -r root@47.252.40.197:/root/backups/atrinexus/YYYYMMDD-HHMMSS D:/learnsomething/AtriNexus-backups/
```

### 5.3 本地创建回流分支

```bash
git checkout -b reconcile/production-state
```

### 5.4 解压工作树快照到临时目录

不要直接覆盖现有工作区。先解压到单独目录，对比后再导入。

### 5.5 生成差异报告

对比：

1. 当前本地工作区
2. 生产快照工作区
3. 当前 GitHub 主线

### 5.6 导入到回流分支并提交

提交信息建议：

```text
Reconcile production state from /opt/AtriNexus on 2026-04-03
```

### 5.7 推送回 GitHub

```bash
git push origin reconcile/production-state
```

## 6. 回流后的输出物

1. `reconcile/production-state` 分支
2. 本地与生产 diff 报告
3. 后续整改统一基线

## 7. 禁止事项

1. 禁止 `git reset --hard`
2. 禁止直接删除生产未跟踪文件
3. 禁止未备份即切换 remote
4. 禁止直接用当前本地代码覆盖 `/opt/AtriNexus`

## 8. 完成标准

以下条件全部满足，才算“源码回流完成”：

1. 生产真实源码已保存为本地可审计资产
2. 回流状态已提交到当前 GitHub 仓库新分支
3. 后续整改可以明确基于哪个分支推进
