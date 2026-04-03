# 生产源码镜像提交清单 v1

更新时间：2026-04-03

基于文档：

- `docs/ops/production-import-boundary.md`

本清单定义第一版镜像提交时的导入与排除范围。

## 1. 目标

建立第一版“生产源码镜像提交”，以恢复真实源码基线。

本次目标不是整理代码，而是把生产真实源码状态纳入当前仓库。

## 2. 导入路径

### 根目录文件

- `README.md`
- `LICENSE`
- `version.json`
- `run.py`
- `requirements.txt`
- `pyproject.toml`

### 目录

- `deployment/`
- `dev_tools/`
- `scripts/`
- `src/`

### 配置与提示

- `data/__init__.py`
- `data/config/__init__.py`
- `data/config/config.json.template`
- `data/config/config.json.template.bak`
- `data/prompts/`

### 需要单独评估后可加入的文件

- `deployment_guide.md`

## 3. 暂缓导入

这部分先不进入第一版镜像提交。

### 文档

- `archive/`
- `IMPROVEMENT_SUMMARY.md`
- `PROJECT_FEATURES.md`
- `WECOM_DESIGN_DOC.md`
- `项目技术文档.md`
- `rename_project_guide.md`
- `Thanks.md`

### 资源

- `data/avatars/MONO/`
- `data/avatars/Nijiko/`
- `data/images/`

### 零散脚本

- `check_time.py`
- `test.py`
- `test_benchmark.py`
- `【RDP远程必用】断联脚本.bat`

## 4. 明确排除

### 运行数据

- `data/database/`
- `data/vectordb/`
- `logs/`
- `data/token_stats/`
- `data/knowledge/`

### 私有配置

- `data/config/config.json`
- `data/config/backups/`

### 高体积素材

- `data/mode/`
- `data/voices/`

### 其他不应参与源码镜像的内容

- `backups/`
- 生产快照中的 `.git/`

## 5. 提交策略

建议拆成两个提交，不要一次提交所有内容。

### 提交 1

导入主链路源码：

- `run.py`
- `requirements.txt`
- `pyproject.toml`
- `src/`
- `deployment/`
- `dev_tools/`
- `scripts/`

### 提交 2

导入模板与辅助配置：

- `data/__init__.py`
- `data/config/__init__.py`
- `data/config/config.json.template`
- `data/config/config.json.template.bak`
- `data/prompts/`

## 6. 提交前检查

在实际导入前必须确认：

1. 未把 `data/config/config.json` 带入
2. 未把 `data/database/` 带入
3. 未把 `data/vectordb/` 带入
4. 未把 `logs/` 带入
5. 未把生产快照里的 `.git/` 带入

## 7. 提交后检查

提交后应立即检查：

1. `git status --short`
2. 导入文件总量是否符合预期
3. 是否出现大量二进制资源误入仓库
4. `.gitignore` 是否仍然能阻止运行数据进入 Git
