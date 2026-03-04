"""
数据库服务模块
提供数据库相关功能，包括:
- 定义数据库模型（ChatMessage, SessionState, MemorySnapshot）
- 创建数据库连接（SQLite + WAL模式）
- 管理会话
- 存储聊天记录、会话状态、记忆快照
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, event, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 创建基类
Base = declarative_base()

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
db_path = os.path.join(project_root, 'data', 'database', 'chat_history.db')

# 确保数据库目录存在
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# 创建数据库连接
engine = create_engine(f'sqlite:///{db_path}')

# 启用 WAL 模式以提升并发性能
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()

# 创建会话工厂
Session = sessionmaker(bind=engine) # bind参数指定了这个Session类将使用哪个数据库引擎进行连接和操作。在这里，我们将之前创建的engine传递给Session，使得通过这个Session创建的会话对象能够与指定的SQLite数据库进行交互。


class ChatMessage(Base):
    """聊天记录模型"""
    __tablename__ = 'chat_messages'

    id = Column(Integer, primary_key=True)
    sender_id = Column(String(100))       # 发送者ID（企微UserID或旧微信ID）
    sender_name = Column(String(100))     # 发送者昵称
    message = Column(Text)               # 发送的消息
    reply = Column(Text)                 # 机器人的回复
    wecom_msg_id = Column(String(64), unique=True, nullable=True)  # 企微消息ID（用于去重）
    created_at = Column(DateTime, default=datetime.now)


class SessionState(Base):
    """用户会话状态模型"""
    __tablename__ = 'session_states'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), unique=True, nullable=False)  # 企微UserID
    mode = Column(String(20), default='work')                   # 当前模式: work / companion
    avatar_name = Column(String(100), default='ATRI')           # 当前使用的人设名称
    last_active = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    variables = Column(Text, default='{}')                      # 扩展变量（JSON）


class MemorySnapshot(Base):
    """记忆快照模型 — 替代JSON文件存储"""
    __tablename__ = 'memory_snapshots'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False)               # 企微UserID
    avatar_name = Column(String(100), nullable=False)           # 人设名称
    memory_type = Column(String(20), nullable=False)            # short / core
    content = Column(Text, default='[]')                        # JSON Blob
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class ConversationCounter(Base):
    """对话计数器 - 用于记忆更新触发"""
    __tablename__ = 'conversation_counters'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False)               # 企微UserID
    avatar_name = Column(String(100), nullable=False)           # 人设名称
    count = Column(Integer, default=0)                          # 核心记忆计数（每15轮重置）
    vector_count = Column(Integer, default=0)                   # 向量记忆计数（每10轮重置）
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # 联合唯一约束：确保每个用户+人设组合只有一条记录
    __table_args__ = (
        UniqueConstraint('user_id', 'avatar_name', name='uix_user_avatar_counter'),
        {'sqlite_autoincrement': True},
    )


class KBSearchSession(Base):
    """知识库检索会话 - 支持多轮检索对话"""
    __tablename__ = 'kb_search_sessions'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False)               # 企微UserID
    original_query = Column(String(500))                        # 原始查询
    current_filter = Column(String(200))                        # 当前过滤条件
    waiting_for = Column(String(50))                            # 等待类型: category/header/clarify
    candidates = Column(Text, default='[]')                     # 候选选项 (JSON)
    created_at = Column(DateTime, default=datetime.now)
    expires_at = Column(DateTime)                               # 过期时间


class Diary(Base):
    """日记数据模型 - 每日对话回忆"""
    __tablename__ = 'diaries'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False)               # 企微UserID
    avatar_name = Column(String(100), nullable=False)           # 人设名称
    date = Column(String(10), nullable=False)                   # 日期 (YYYY-MM-DD)
    content = Column(Text, nullable=False)                      # 日记内容 (第一人称)
    conversation_count = Column(Integer, default=0)             # 当日对话轮次
    created_at = Column(DateTime, default=datetime.now)
    
    # 联合唯一约束：确保每个用户+人设+日期只有一条日记
    __table_args__ = (
        UniqueConstraint('user_id', 'avatar_name', 'date', name='uix_user_avatar_date'),
        {'sqlite_autoincrement': True},
    )


# 创建数据库表
Base.metadata.create_all(engine)