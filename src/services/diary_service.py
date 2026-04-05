"""
日记回溯服务
功能：
- 每日自动生成助手日记（以助手第一人称视角）
- 结合人设与核心记忆，生成第一人称日记
- 支持按用户、日期查询日记
- 定时任务触发机制
"""

import logging
import os
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from sqlalchemy import and_
from sqlalchemy.orm import Session as DBSession

from src.services.database import Session, Diary, ChatMessage
from src.services.ai.llm_service import LLMService
from src.services.memory_manager import MemoryManager
from data.config import config

logger = logging.getLogger('wecom')


@dataclass
class DiaryEntry:
    """日记条目"""
    id: int
    user_id: str
    avatar_name: str
    date: str
    content: str
    conversation_count: int
    created_at: datetime


class DiaryService:
    """日记生成服务"""
    
    # 日记生成提示词模板（以助手第一人称视角）
    DIARY_PROMPT_TEMPLATE = """你是一个日记生成助手。请根据今天的对话记录，以AI助手（{avatar_name}）的第一人称视角写一篇日记。

## 助手人设
- 名称：{avatar_name}
- 核心记忆（关于用户的关键信息）：
{core_memory}

## 今日与用户的对话记录
{conversations}

## 要求
1. 以AI助手（{avatar_name}）的第一人称视角写日记（"我"）
2. 记录今天与用户的互动，表达助手的感受和想法
3. 自然融入核心记忆中关于用户的信息
4. 语言风格要贴合{avatar_name}的人设特点
5. 控制在200-400字之间
6. 体现助手对用户的关心和陪伴
7. 不要出现"根据对话记录"等元信息，直接写日记内容

请直接输出日记内容，不要添加标题或其他格式。"""

    def __init__(self, llm_service: LLMService = None, memory_manager: MemoryManager = None):
        """
        初始化日记服务
        
        Args:
            llm_service: LLM服务实例
            memory_manager: 记忆管理器实例
        """
        self.llm_service = llm_service
        self.memory_manager = memory_manager
        
    def get_diary(self, user_id: str, avatar_name: str, date_str: str) -> Optional[DiaryEntry]:
        """
        获取指定日期的日记
        
        Args:
            user_id: 用户ID
            avatar_name: 人设名称
            date_str: 日期字符串 (YYYY-MM-DD)
        
        Returns:
            DiaryEntry 或 None
        """
        with Session() as session:
            diary = session.query(Diary).filter(
                and_(
                    Diary.user_id == user_id,
                    Diary.avatar_name == avatar_name,
                    Diary.date == date_str
                )
            ).first()
            
            if diary:
                return DiaryEntry(
                    id=diary.id,
                    user_id=diary.user_id,
                    avatar_name=diary.avatar_name,
                    date=diary.date,
                    content=diary.content,
                    conversation_count=diary.conversation_count,
                    created_at=diary.created_at
                )
            return None
    
    def get_diaries_by_range(
        self, 
        user_id: str, 
        avatar_name: str, 
        start_date: str, 
        end_date: str,
        limit: int = 30
    ) -> List[DiaryEntry]:
        """
        获取日期范围内的日记列表
        
        Args:
            user_id: 用户ID
            avatar_name: 人设名称
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            limit: 最大返回数量
        
        Returns:
            日记列表（按日期倒序）
        """
        with Session() as session:
            diaries = session.query(Diary).filter(
                and_(
                    Diary.user_id == user_id,
                    Diary.avatar_name == avatar_name,
                    Diary.date >= start_date,
                    Diary.date <= end_date
                )
            ).order_by(Diary.date.desc()).limit(limit).all()
            
            return [
                DiaryEntry(
                    id=d.id,
                    user_id=d.user_id,
                    avatar_name=d.avatar_name,
                    date=d.date,
                    content=d.content,
                    conversation_count=d.conversation_count,
                    created_at=d.created_at
                )
                for d in diaries
            ]
    
    def get_diary_dates_by_month(
        self, 
        user_id: str, 
        avatar_name: str, 
        year: int, 
        month: int
    ) -> List[str]:
        """
        获取某月有日记的日期列表
        
        Args:
            user_id: 用户ID
            avatar_name: 人设名称
            year: 年份
            month: 月份
        
        Returns:
            日期字符串列表 ['2024-01-01', '2024-01-02', ...]
        """
        with Session() as session:
            start_date = f"{year:04d}-{month:02d}-01"
            if month == 12:
                end_date = f"{year + 1:04d}-01-01"
            else:
                end_date = f"{year:04d}-{month + 1:02d}-01"
            
            diaries = session.query(Diary.date).filter(
                and_(
                    Diary.user_id == user_id,
                    Diary.avatar_name == avatar_name,
                    Diary.date >= start_date,
                    Diary.date < end_date
                )
            ).all()
            
            return [d.date for d in diaries]
    
    def _get_conversations_by_date(
        self, 
        user_id: str, 
        date_str: str
    ) -> List[Dict[str, Any]]:
        """
        获取指定日期的对话记录
        
        Args:
            user_id: 用户ID
            date_str: 日期字符串 (YYYY-MM-DD)
        
        Returns:
            对话记录列表
        """
        with Session() as session:
            # 解析日期
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            next_date = target_date + timedelta(days=1)
            
            messages = session.query(ChatMessage).filter(
                and_(
                    ChatMessage.sender_id == user_id,
                    ChatMessage.created_at >= target_date,
                    ChatMessage.created_at < next_date
                )
            ).order_by(ChatMessage.created_at).all()
            
            conversations = []
            for msg in messages:
                conversations.append({
                    "user": msg.message,
                    "bot": msg.reply,
                    "timestamp": msg.created_at.strftime("%H:%M")
                })
            
            return conversations
    
    def generate_diary(
        self, 
        user_id: str, 
        avatar_name: str, 
        date_str: str = None,
        force_regenerate: bool = False
    ) -> Optional[DiaryEntry]:
        """
        生成指定日期的日记
        
        Args:
            user_id: 用户ID
            avatar_name: 人设名称
            date_str: 日期字符串 (YYYY-MM-DD)，默认为昨天
            force_regenerate: 是否强制重新生成
        
        Returns:
            生成的日记条目，如果无对话记录则返回 None
        """
        # 默认生成昨天的日记
        if not date_str:
            date_str = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # 检查是否已存在
        if not force_regenerate:
            existing = self.get_diary(user_id, avatar_name, date_str)
            if existing:
                logger.info(f"日记已存在: user={user_id}, date={date_str}")
                return existing
        
        # 获取对话记录
        conversations = self._get_conversations_by_date(user_id, date_str)
        if not conversations:
            logger.info(f"无对话记录，跳过日记生成: user={user_id}, date={date_str}")
            return None
        
        # 获取核心记忆
        core_memory = ""
        if self.memory_manager:
            core_memory = self.memory_manager.get_core_memory(user_id, avatar_name) or "暂无核心记忆"
        
        # 格式化对话记录
        conv_text = ""
        for i, conv in enumerate(conversations, 1):
            conv_text += f"\n[对话 {i}] {conv['timestamp']}\n"
            conv_text += f"用户：{conv['user']}\n"
            conv_text += f"助手：{conv['bot']}\n"
        
        # 构建提示词
        prompt = self.DIARY_PROMPT_TEMPLATE.format(
            avatar_name=avatar_name,
            core_memory=core_memory,
            conversations=conv_text
        )
        
        # 调用LLM生成日记
        if not self.llm_service:
            logger.error("LLM服务未初始化，无法生成日记")
            return None
        
        try:
            diary_content = self.llm_service.chat(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "请根据上述对话记录，以AI助手的第一人称视角写一篇日记。"},
                ]
            )
            
            diary_content = diary_content.strip()
            
            # 保存日记
            with Session() as session:
                # 先删除旧的（如果强制重新生成）
                if force_regenerate:
                    session.query(Diary).filter(
                        and_(
                            Diary.user_id == user_id,
                            Diary.avatar_name == avatar_name,
                            Diary.date == date_str
                        )
                    ).delete()
                
                diary = Diary(
                    user_id=user_id,
                    avatar_name=avatar_name,
                    date=date_str,
                    content=diary_content,
                    conversation_count=len(conversations)
                )
                session.add(diary)
                session.commit()
                
                logger.info(f"日记生成成功: user={user_id}, date={date_str}, conversations={len(conversations)}")
                
                return DiaryEntry(
                    id=diary.id,
                    user_id=diary.user_id,
                    avatar_name=diary.avatar_name,
                    date=diary.date,
                    content=diary.content,
                    conversation_count=diary.conversation_count,
                    created_at=diary.created_at
                )
                
        except Exception as e:
            logger.error(f"生成日记失败: {e}")
            return None
    
    def generate_diaries_for_active_users(
        self, 
        date_str: str = None,
        avatar_name: str = "ATRI"
    ) -> Dict[str, Any]:
        """
        为当天有对话的用户生成日记
        
        Args:
            date_str: 日期字符串 (YYYY-MM-DD)，默认为昨天
            avatar_name: 人设名称
        
        Returns:
            生成结果统计
        """
        if not date_str:
            date_str = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # 查找当天有对话的用户
        with Session() as session:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            next_date = target_date + timedelta(days=1)
            
            active_users = session.query(ChatMessage.sender_id).filter(
                and_(
                    ChatMessage.created_at >= target_date,
                    ChatMessage.created_at < next_date
                )
            ).distinct().all()
            
            user_ids = [u.sender_id for u in active_users]
        
        results = {
            "date": date_str,
            "total_users": len(user_ids),
            "success": 0,
            "skipped": 0,
            "failed": 0,
            "details": []
        }
        
        for user_id in user_ids:
            try:
                diary = self.generate_diary(user_id, avatar_name, date_str)
                if diary:
                    results["success"] += 1
                    results["details"].append({
                        "user_id": user_id,
                        "status": "created",
                        "conversations": diary.conversation_count
                    })
                else:
                    results["skipped"] += 1
                    results["details"].append({
                        "user_id": user_id,
                        "status": "no_conversations"
                    })
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "user_id": user_id,
                    "status": "error",
                    "error": str(e)
                })
                logger.error(f"生成日记失败: user={user_id}, error={e}")
        
        logger.info(f"日记批量生成完成: date={date_str}, success={results['success']}, skipped={results['skipped']}, failed={results['failed']}")
        return results
    
    def delete_diary(self, user_id: str, avatar_name: str, date_str: str) -> bool:
        """删除指定日记"""
        with Session() as session:
            deleted = session.query(Diary).filter(
                and_(
                    Diary.user_id == user_id,
                    Diary.avatar_name == avatar_name,
                    Diary.date == date_str
                )
            ).delete()
            session.commit()
            return deleted > 0
    
    def get_stats(self, user_id: str = None) -> Dict[str, Any]:
        """获取日记统计信息"""
        with Session() as session:
            query = session.query(Diary)
            if user_id:
                query = query.filter(Diary.user_id == user_id)
            
            total = query.count()
            
            # 获取日期范围
            oldest = session.query(Diary.date).filter(
                Diary.user_id == user_id if user_id else True
            ).order_by(Diary.date).first()
            
            newest = session.query(Diary.date).filter(
                Diary.user_id == user_id if user_id else True
            ).order_by(Diary.date.desc()).first()
            
            return {
                "total_diaries": total,
                "oldest_date": oldest[0] if oldest else None,
                "newest_date": newest[0] if newest else None
            }
