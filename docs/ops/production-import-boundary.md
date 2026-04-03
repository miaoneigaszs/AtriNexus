# 生产源码回流边界定义

更新时间：2026-04-03

## 1. 目的

定义从生产快照 `backups/production/20260403-163132/extracted` 回流到当前仓库时：

1. 哪些内容必须导入
2. 哪些内容可以延后判断
3. 哪些内容绝对不能进入 Git 主线

该文档用于指导第一版“生产源码镜像提交”。

## 2. 总原则

### 原则 1

源码回流优先于资源回流。

### 原则 2

运行数据绝不进入 Git。

### 原则 3

生产实值配置不进入 Git。

### 原则 4

第一版回流以“恢复真实源码基线”为目标，不追求一次整理干净。

## 3. 第一层：必须导入

这些内容构成生产代码的主链路，应纳入第一版镜像提交。

### 根目录

- `README.md`
- `LICENSE`
- `version.json`
- `run.py`
- `requirements.txt`
- `pyproject.toml`

### 配置与提示

- `data/__init__.py`
- `data/config/__init__.py`
- `data/config/config.json.template`
- `data/config/config.json.template.bak`
- `data/prompts/`
- `src/base/`

注意：

- `data/config/config.json` 不导入

### 部署与辅助脚本

- `deployment/`
- `deployment_guide.md`
- `dev_tools/`
- `scripts/`

### 核心源码

- `src/__init__.py`
- `src/services/`
- `src/utils/`
- `src/web/`
- `src/wecom/`

### 测试

- `src/tests/`

说明：

- 测试当前未必可信，但它们属于源码资产，应先回流，再逐步重建

## 4. 第二层：待判断导入

这些内容不是主链路源码，但可能有业务价值。

### 人设与资源

- `data/avatars/MONO/`
- `data/avatars/Nijiko/`
- `data/images/`

### 历史文档

- `archive/`
- `IMPROVEMENT_SUMMARY.md`
- `PROJECT_FEATURES.md`
- `WECOM_DESIGN_DOC.md`
- `项目技术文档.md`
- `rename_project_guide.md`
- `Thanks.md`

### 零散脚本

- `check_time.py`
- `test.py`
- `test_benchmark.py`
- `【RDP远程必用】断联脚本.bat`

处理建议：

- 先不放进第一版镜像提交
- 单独列入候选资产清单
- 等源码基线稳定后再决定是否纳入

## 5. 第三层：绝不导入 Git 主线

这些内容属于运行时数据、私有数据或高噪音资产。

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

说明：

- 这部分可保留在备份目录
- 可作为迁移参考
- 但不应进入源码回流提交

## 6. 现有 `.gitignore` 相关结论

当前 `.gitignore` 已经排除了大部分运行数据：

- `data/config/config.json`
- `data/database/`
- `data/vectordb/`
- `data/mode/`
- `data/tasks.json`
- `docs/`

但也有两个需要注意的点：

### 问题 1

`docs/` 被整体忽略，导致新写的整改文档默认不会进入 Git。

### 问题 2

`*.md`、`*.txt` 被整体忽略，只靠少量例外保留，这会让回流历史文档变得不透明。

结论：

- 在真正提交整改文档前，需要单独调整 `.gitignore`
- 但该动作应与源码镜像提交分开处理，避免把边界问题和回流问题混在一起

## 7. 第一版镜像提交建议范围

### 建议纳入

- 根目录核心启动与依赖文件
- `deployment/`
- `dev_tools/`
- `scripts/`
- `src/`
- `data/config/` 中的源码和模板
- `src/base/`

### 建议暂缓

- `archive/`
- 历史总结类文档
- 人设扩展资源
- 表情与素材目录

### 明确排除

- `data/database/`
- `data/vectordb/`
- `logs/`
- `data/config/config.json`
- `data/mode/`
- `data/voices/`

## 8. 下一步

基于本边界文档，下一步应生成：

1. 第一版导入清单
2. 第一版排除清单
3. 实际导入操作步骤
4. 第一版镜像提交
