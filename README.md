# AtriNexus

<div align="center">

**在虚拟与现实交织处，给予永恒的温柔羁绊**

# AtriNexus

[![GitHub Stars](https://img.shields.io/github/stars/miaoneigaszs/AtriNexus?color=ff69b4&style=flat-square)](https://github.com/miaoneigaszs/AtriNexus/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/miaoneigaszs/AtriNexus?color=9c27b0&style=flat-square)](https://github.com/miaoneigaszs/AtriNexus/network/members)
[![License](https://img.shields.io/github/license/miaoneigaszs/AtriNexus?color=03a9f4&style=flat-square)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.4.3.2-ff69b4?style=flat-square)](version.json)

</div>

---

## 📖 项目简介

AtriNexus 是一个基于企业微信的智能 AI 助手，集成了多模态对话能力。通过 DeepSeek、阿里云等大语言模型，实现了文本对话、图像识别、语音交互等功能。项目支持沉浸式角色扮演、持久记忆存储、定时任务等特性，为用户提供温暖而智能的陪伴体验。

### ✨ 核心特性

- 🤖 **多模态对话** - 支持文本、图像、语音多种交互方式
- 💭 **智能记忆** - 持久化记忆存储，支持记忆衰减与权重管理
- 🎭 **角色扮演** - 沉浸式角色扮演，支持自定义人设
- 🔄 **意图识别** - 智能识别用户意图，精准响应
- 📚 **知识库** - 支持文档上传与向量化检索
- 🌐 **网络搜索** - 集成 Tavily 搜索，实时获取信息
- ⏰ **定时任务** - 支持定时消息推送
- 🎨 **WebUI 管理** - 可视化配置界面，易于管理
- 👥 **多用户支持** - 完善的多用户与群聊支持

---

## 🚀 快速开始

### 📋 环境要求

- Python 3.10 或更高版本
- 企业微信账号（管理员权限）
- Windows/Linux/macOS

### 🔑 获取 API 密钥

项目需要以下 API 密钥：

| 服务 | 用途 | 获取地址 |
|------|------|----------|
| DeepSeek | 对话模型 | [DeepSeek](https://platform.deepseek.com/) |
| 阿里云百炼 | 图像识别 | [阿里云百炼](https://bailian.console.aliyun.com/) |
| 硅基流动 | 意图识别 & Embedding | [硅基流动](https://cloud.siliconflow.cn/) |
| Tavily | 网络搜索 | [Tavily](https://tavily.com/) |

### 📥 安装步骤

#### 方法一：快速部署

```bash
# 1. 克隆仓库
git clone https://github.com/AtriNexus/AtriNexus.git
cd AtriNexus

# 2. 运行启动脚本
python run.py
```

#### 方法二：手动部署

```bash
# 1. 克隆仓库
git clone https://github.com/AtriNexus/AtriNexus.git
cd AtriNexus

# 2. 升级 pip
python -m pip install --upgrade pip

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置文件（见下方说明）

# 5. 启动程序
python run.py
```

### ⚙️ 配置说明

**⚠️ 重要：配置文件包含敏感信息，请勿提交到 Git！**

1. **复制配置模板**

```bash
cp data/config/config.json.template data/config/config.json
```

2. **编辑配置文件**

编辑 `data/config/config.json`，填入你的 API 密钥：

```json
{
  "categories": {
    "wecom_settings": {
      "settings": {
        "corp_id": { "value": "你的企业ID" },
        "agent_id": { "value": "你的应用AgentId" },
        "secret": { "value": "你的应用Secret" },
        "token": { "value": "你的Token" },
        "encoding_aes_key": { "value": "你的EncodingAESKey" }
      }
    },
    "llm_settings": {
      "settings": {
        "api_key": { "value": "你的API密钥" },
        "base_url": { "value": "https://api.deepseek.com/v1" },
        "model": { "value": "deepseek-chat" }
      }
    }
  }
}
```

3. **使用 WebUI 配置（可选）**

```bash
python run_config_web.py
```

### 🎯 启动服务

```bash
# 前台运行
python run.py

# 后台运行（Linux）
nohup python run.py &

# Windows 服务部署
# 参考 deployment/ 目录下的配置文件
```

---

## 📁 项目结构

```
AtriNexus/
├── run.py                      # 主程序入口
├── pyproject.toml              # 项目配置(uv包管理)
├── requirements.txt            # 依赖清单
├── version.json                # 版本信息
│
├── src/                        # 源代码目录
│   ├── base/                   # 基础配置和提示词
│   │   ├── base.md             # 基础行为指南
│   │   ├── memory.md           # 记忆摘要提示词
│   │   ├── worldview.md        # 世界观设定
│   │   └── group.md            # 群聊配置
│   │
│   ├── services/               # 核心服务层
│   │   ├── ai/                 # AI服务
│   │   │   ├── llm_service.py          # LLM对话服务
│   │   │   ├── embedding_service.py    # Embedding服务
│   │   │   ├── image_recognition_service.py  # 图片识别
│   │   │   ├── model_manager.py        # 模型管理
│   │   │   └── network_search_service.py # 网络搜索
│   │   │
│   │   ├── database.py         # 数据库模型
│   │   ├── memory_manager.py   # 记忆管理器
│   │   ├── rag_engine.py       # RAG检索引擎
│   │   ├── intent_service.py   # 意图识别服务
│   │   ├── session_service.py  # 会话管理
│   │   ├── diary_service.py    # 日记生成服务
│   │   └── token_monitor.py    # Token监测
│   │
│   ├── wecom/                  # 企业微信相关
│   │   ├── server.py           # FastAPI服务入口(精简版)
│   │   ├── deps.py             # 共享依赖和全局实例
│   │   ├── scheduler.py        # 定时任务调度器
│   │   ├── client.py           # 企微API客户端
│   │   ├── handlers.py         # 消息处理器
│   │   └── routers/            # API路由模块
│   │       ├── callback.py     # 企微回调路由
│   │       ├── knowledge.py    # 知识库路由
│   │       ├── memory.py       # 记忆管理路由
│   │       ├── config.py       # 系统配置路由
│   │       ├── tasks.py        # 定时任务路由
│   │       ├── diary.py        # 日记路由
│   │       └── token.py        # Token监控路由
│   │
│   ├── utils/                  # 工具类
│   │   ├── metrics.py          # Prometheus指标
│   │   ├── health_check.py     # 健康检查
│   │   ├── async_utils.py      # 异步工具
│   │   ├── http_pool.py        # HTTP连接池
│   │   └── version.py          # 版本管理
│   │
│   └── web/                    # Web界面
│       └── templates/
│           └── upload.html     # 知识库上传页面
│
├── data/                       # 数据目录
│   ├── config/                 # 配置文件
│   │   ├── config.json         # 主配置文件
│   │   └── config.json.template # 配置模板
│   │
│   ├── avatars/                # 人设目录
│   │   ├── ATRI/               # 人设示例
│   │   │   └── avatar.md       # 人设提示词
│   │   └── MONO/
│   │
│   ├── database/               # SQLite数据库
│   │   └── chat_history.db     # 聊天记录+记忆
│   │
│   ├── vectordb/               # ChromaDB向量存储
│   ├── mode/                   # 表情包资源(可选)
│   └── tasks.json              # 定时任务配置
│
├── scripts/                    # 脚本工具
│   ├── check_time.py           # 时间检查
│   ├── describe_emojis.py      # 表情包描述生成
│   └── test_single.py          # 单元测试
│
├── deployment/                 # 部署相关
│   ├── atrinexus.service       # Systemd服务配置
│   └── install.sh              # 安装脚本
│
|── README.md                   # 项目说明
```

---

详细架构说明请参考：[项目技术文档.md](项目技术文档.md)

---

## 🔒 安全须知

### ⚠️ 重要提醒

- **🔐 永远不要提交 `config.json` 到 Git**
- **🔐 定期更换 API 密钥**
- **🔐 使用强密码保护管理员账户**
- **🔐 不要在公开场合分享配置文件**

### 📝 配置文件管理

项目已通过 `.gitignore` 自动忽略以下敏感文件：

```gitignore
data/config/config.json          # 主配置文件
data/database/                   # 用户数据库
data/vectordb/                   # 向量数据库
data/knowledge/                  # 知识库文件
logs/                            # 日志文件
*.db, *.sqlite, *.sqlite3        # 数据库文件
.env                             # 环境变量
```

### 🔍 开源前检查

在推送代码到 GitHub 前，请运行检查脚本：

```bash
python scripts/check_sensitive_files.py
```

---

## 🛠️ 高级配置

### 自定义人设

人设文件位于 `data/avatars/` 目录：

```
data/avatars/ATRI/
├── avatar.md      # 角色设定
└── emojis/        # 表情包
```

### 群聊配置

在 `config.json` 的 `user_settings.group_chat_config` 中配置不同群聊的专属设定。

### 定时任务

通过 WebUI 或配置文件设置定时消息推送。

---

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 如何贡献

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

### 贡献领域

- 🐛 修复 Bug
- ✨ 添加新功能
- 📝 改进文档
- 🎨 优化代码结构
- 🌐 多语言支持

---

## 📜 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。

### 使用须知

- ✅ 允许商业使用
- ✅ 允许修改和分发
- ⚠️ 需保留版权声明
- ⚠️ 开发者不承担任何责任

### 法律与伦理准则

- 本项目仅供技术研究与学习交流
- 禁止用于任何违法或违反道德的场景
- 生成内容不代表开发者立场
- 角色版权归属原始创作者
- 使用者需对自身行为负全责
- 未成年人应在监护下使用

---

## 🙏 致谢

本项目离不开以下支持和贡献，详见 [Thanks.md](Thanks.md)

---

<div align="center">

**Made with ❤️ by AtriNexus Team**

[⬆ 返回顶部](#atriNexus)

</div>
