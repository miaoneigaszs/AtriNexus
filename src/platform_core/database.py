"""同步数据库入口。当前主运行数据库为 PostgreSQL。"""

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker

from src.platform_core.database_config import build_sync_database_url, build_sync_engine_kwargs

# 创建基类
Base = declarative_base()

DATABASE_URL = build_sync_database_url()
engine = create_engine(DATABASE_URL, **build_sync_engine_kwargs())

Session = sessionmaker(bind=engine, future=True)


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
    )


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
    )


# 创建数据库表
Base.metadata.create_all(engine)
