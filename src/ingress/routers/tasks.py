"""
定时任务 API 路由
处理定时任务的 CRUD 操作
"""

import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.platform_core.async_utils import run_sync
from src.ingress.scheduler import (
    load_tasks_file, save_tasks_file, reload_scheduled_tasks
)
from src.ingress.deps import logger

router = APIRouter(tags=["定时任务"])


@router.get("/api/tasks")
async def api_tasks_list():
    """获取所有定时任务"""
    try:
        tasks = await run_sync(load_tasks_file)
        return JSONResponse(content={"success": True, "data": tasks})
    except Exception as e:
        logger.error(f"[Tasks API] 获取任务列表失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


@router.post("/api/tasks")
async def api_tasks_create(request: Request):
    """创建新定时任务"""
    try:
        body = await request.json()
        task_id = body.get('task_id', '').strip()
        chat_id = body.get('chat_id', '').strip()
        content = body.get('content', '').strip()
        schedule_time = body.get('schedule_time', '').strip()
        is_active = body.get('is_active', True)
        
        if not all([task_id, chat_id, content, schedule_time]):
            return JSONResponse(status_code=400, content={"success": False, "message": "task_id, chat_id, content, schedule_time 均为必填"})
        
        tasks = await run_sync(load_tasks_file)
        
        # 检查 ID 重复
        if any(t.get('task_id') == task_id for t in tasks):
            return JSONResponse(status_code=400, content={"success": False, "message": f"task_id '{task_id}' 已存在"})
        
        new_task = {
            "task_id": task_id,
            "chat_id": chat_id,
            "content": content,
            "schedule_type": "cron",
            "schedule_time": schedule_time,
            "is_active": is_active
        }
        tasks.append(new_task)
        
        if await run_sync(save_tasks_file, tasks):
            await run_sync(reload_scheduled_tasks)
            logger.info(f"[Tasks API] 创建任务: {task_id}")
            return JSONResponse(content={"success": True, "message": "任务已创建", "data": new_task})
        else:
            return JSONResponse(status_code=500, content={"success": False, "message": "保存失败"})
    except Exception as e:
        logger.error(f"[Tasks API] 创建任务失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


@router.put("/api/tasks/{task_id}")
async def api_tasks_update(task_id: str, request: Request):
    """更新定时任务"""
    try:
        body = await request.json()
        tasks = await run_sync(load_tasks_file)
        
        found = False
        for i, t in enumerate(tasks):
            if t.get('task_id') == task_id:
                # 更新可修改字段
                if 'chat_id' in body:
                    tasks[i]['chat_id'] = body['chat_id']
                if 'content' in body:
                    tasks[i]['content'] = body['content']
                if 'schedule_time' in body:
                    tasks[i]['schedule_time'] = body['schedule_time']
                if 'is_active' in body:
                    tasks[i]['is_active'] = body['is_active']
                found = True
                break
        
        if not found:
            return JSONResponse(status_code=404, content={"success": False, "message": f"任务 '{task_id}' 不存在"})
        
        if await run_sync(save_tasks_file, tasks):
            await run_sync(reload_scheduled_tasks)
            logger.info(f"[Tasks API] 更新任务: {task_id}")
            return JSONResponse(content={"success": True, "message": "任务已更新"})
        else:
            return JSONResponse(status_code=500, content={"success": False, "message": "保存失败"})
    except Exception as e:
        logger.error(f"[Tasks API] 更新任务失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


@router.delete("/api/tasks/{task_id}")
async def api_tasks_delete(task_id: str):
    """删除定时任务"""
    try:
        tasks = await run_sync(load_tasks_file)
        original_len = len(tasks)
        tasks = [t for t in tasks if t.get('task_id') != task_id]
        
        if len(tasks) == original_len:
            return JSONResponse(status_code=404, content={"success": False, "message": f"任务 '{task_id}' 不存在"})
        
        if await run_sync(save_tasks_file, tasks):
            await run_sync(reload_scheduled_tasks)
            logger.info(f"[Tasks API] 删除任务: {task_id}")
            return JSONResponse(content={"success": True, "message": "任务已删除"})
        else:
            return JSONResponse(status_code=500, content={"success": False, "message": "保存失败"})
    except Exception as e:
        logger.error(f"[Tasks API] 删除任务失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


@router.post("/api/tasks/reload")
async def api_tasks_reload():
    """重新加载定时任务到调度器"""
    try:
        await run_sync(reload_scheduled_tasks)
        return JSONResponse(content={"success": True, "message": "定时任务已重新加载"})
    except Exception as e:
        logger.error(f"[Tasks API] 重新加载失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})
