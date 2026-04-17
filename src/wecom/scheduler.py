"""
定时任务调度器模块
负责定时任务的加载、执行、管理
"""

import os
import json
import logging
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.services.agent.tool_profiles import default_tool_profile_for_mode
from src.wecom.deps import ROOT_DIR, message_handler, get_diary_service

logger = logging.getLogger('wecom')

# 全局调度器实例
scheduler = None

# tasks.json 路径
TASKS_FILE = os.path.join(ROOT_DIR, 'data', 'tasks.json')


def load_tasks_file() -> list:
    """读取 tasks.json"""
    try:
        if os.path.exists(TASKS_FILE):
            with open(TASKS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"[定时任务] 读取 tasks.json 失败: {e}")
        return []


def save_tasks_file(tasks: list) -> bool:
    """保存 tasks.json"""
    try:
        with open(TASKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        logger.error(f"[定时任务] 保存 tasks.json 失败: {e}")
        return False


def execute_scheduled_task(task: dict):
    """
    执行定时任务：加载模式提示+记忆 -> 走主 agent 回复链 -> 发送给用户
    
    Args:
        task: 任务配置字典，包含 task_id, chat_id, content 等
    """
    from src.wecom.deps import wecom_client
    
    task_id = task.get('task_id', 'unknown')
    chat_id = task.get('chat_id', '')
    content = task.get('content', '')
    
    if not chat_id or not content:
        logger.warning(f"[定时任务] 任务 {task_id} 缺少 chat_id 或 content，跳过")
        return
    
    logger.info(f"[定时任务] 开始执行: {task_id} -> 用户 {chat_id}")
    
    try:
        # 1. 构建模式提示词与会话上下文
        state = message_handler.session_service.get_session(chat_id)
        avatar_name = state.avatar_name or 'ATRI'
        current_mode = 'companion'
        system_prompt = message_handler.context_builder.build_system_prompt(
            avatar_name=avatar_name,
            current_mode=current_mode,
        )
        
        # 2. 加载核心记忆和近期上下文
        core_memory = ""
        previous_context = []
        try:
            core_memory = message_handler.memory.get_core_memory(chat_id, avatar_name)
            
            # 取最近10轮对话作为上下文
            short_memory = message_handler.memory.get_short_memory(chat_id, avatar_name)
            recent_10 = short_memory[-10:] if short_memory else []
            previous_context = message_handler.memory.build_context_from_memory(recent_10)
        except Exception as mem_err:
            logger.warning(f"[定时任务] 加载记忆/上下文失败: {mem_err}")
        
        # 3. 复用主 Agent 运行时骨架生成回复，避免定时任务链路与主聊天链路漂移
        reply = message_handler.reply_service.generate_reply(
            message=content,
            user_id=chat_id,
            system_prompt=system_prompt,
            tool_profile=default_tool_profile_for_mode(current_mode),
            previous_context=previous_context,
            core_memory=core_memory,
        )
        
        # 4. 清理回复
        reply = reply.strip() if reply else ""
        if not reply:
            logger.warning(f"[定时任务] 任务 {task_id} LLM 返回空回复")
            return
        
        # 5. 发送消息
        success = wecom_client.send_text(chat_id, reply)
        
        if success:
            logger.info(f"[定时任务] 任务 {task_id} 执行成功: {reply[:80]}...")
        else:
            logger.error(f"[定时任务] 任务 {task_id} 发送失败")
        
    except Exception as e:
        logger.error(f"[定时任务] 任务 {task_id} 执行异常: {e}", exc_info=True)


def load_scheduled_tasks(sched):
    """
    从 tasks.json 加载定时任务并注册到调度器
    
    Args:
        sched: APScheduler BackgroundScheduler 实例
    """
    tasks = load_tasks_file()
    loaded_count = 0
    
    for task in tasks:
        task_id = task.get('task_id', '')
        schedule_time = task.get('schedule_time', '')
        is_active = task.get('is_active', True)
        
        if not task_id or not schedule_time:
            logger.warning(f"[定时任务] 跳过无效任务: {task}")
            continue
        
        if not is_active:
            logger.info(f"[定时任务] 任务 {task_id} 已禁用，跳过")
            continue
        
        try:
            # 解析 cron 表达式: "分 时 日 月 周"
            parts = schedule_time.split()
            if len(parts) != 5:
                logger.error(f"[定时任务] 任务 {task_id} 的 cron 表达式无效: {schedule_time}")
                continue
            
            trigger = CronTrigger(
                minute=parts[0],
                hour=parts[1],
                day=parts[2],
                month=parts[3],
                day_of_week=parts[4],
                timezone=pytz.timezone('Asia/Shanghai')
            )
            
            sched.add_job(
                execute_scheduled_task,
                trigger,
                args=[task],
                id=f'task_{task_id}',
                name=f'定时任务: {task_id}',
                replace_existing=True
            )
            loaded_count += 1
            logger.info(f"[定时任务] 已注册: {task_id} | cron: {schedule_time} | 目标: {task.get('chat_id', 'N/A')}")
            
        except Exception as e:
            logger.error(f"[定时任务] 注册任务 {task_id} 失败: {e}")
    
    logger.info(f"[定时任务] 共加载 {loaded_count}/{len(tasks)} 个定时任务")


def reload_scheduled_tasks():
    """重新加载所有定时任务（用于设置页面修改后刷新）"""
    global scheduler
    if not scheduler:
        logger.warning("[定时任务] 调度器未初始化")
        return
    
    # 移除所有 task_ 开头的任务（保留 diary_generation）
    for job in scheduler.get_jobs():
        if job.id.startswith('task_'):
            scheduler.remove_job(job.id)
    
    # 重新加载
    load_scheduled_tasks(scheduler)
    logger.info("[定时任务] 定时任务已重新加载")


def scheduled_diary_generation():
    """
    定时任务：每日 0:00 自动生成昨天的日记
    以助手（亚托莉）的第一人称视角生成
    """
    logger.info("[定时任务] 开始自动生成日记...")
    
    try:
        diary_service = get_diary_service()
        result = diary_service.generate_diaries_for_active_users(avatar_name="ATRI")
        
        logger.info(f"[定时任务] 日记生成完成: success={result['success']}, skipped={result['skipped']}, failed={result['failed']}")
        
    except Exception as e:
        logger.error(f"[定时任务] 日记生成失败: {e}", exc_info=True)


def init_scheduler():
    """初始化定时任务调度器"""
    global scheduler
    scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Shanghai'))
    
    # 1. 每日 0:05 生成昨天的日记
    scheduler.add_job(
        scheduled_diary_generation,
        CronTrigger(hour=0, minute=5),
        id='diary_generation',
        name='每日日记生成',
        replace_existing=True
    )
    
    # 2. 加载用户定时任务
    load_scheduled_tasks(scheduler)
    
    scheduler.start()
    logger.info("[定时任务] 调度器已启动")
    
    return scheduler
