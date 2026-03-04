"""
Token 监控路由
处理 Token 使用统计、费用估算等操作
"""

import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.services.token_monitor import token_monitor
from src.wecom.deps import logger

router = APIRouter(tags=["Token 监控"])


@router.get("/api/token/stats")
async def api_token_stats():
    """获取 token 使用统计"""
    try:
        stats = token_monitor.get_all_stats()
        return JSONResponse(content={"success": True, "data": stats})
    except Exception as e:
        logger.error(f"获取 token 统计异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": "获取统计失败"})


@router.get("/api/token/usage")
async def api_token_usage(period_hours: int = 24):
    """获取指定时间段内的 token 使用摘要"""
    try:
        summary = token_monitor.get_usage_summary(period_hours)
        return JSONResponse(content={"success": True, "data": summary})
    except Exception as e:
        logger.error(f"获取 token 使用摘要异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": "获取使用摘要失败"})


@router.get("/api/token/cost")
async def api_token_cost(period_hours: int = 24):
    """获取指定时间段内的 token 费用估算"""
    try:
        cost_summary = token_monitor.get_cost_summary(period_hours)
        return JSONResponse(content={"success": True, "data": cost_summary})
    except Exception as e:
        logger.error(f"获取 token 费用摘要异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": "获取费用摘要失败"})


@router.get("/api/token/models")
async def api_token_models():
    """获取按模型统计的 token 使用情况"""
    try:
        model_stats = token_monitor.get_model_stats()
        return JSONResponse(content={"success": True, "data": model_stats})
    except Exception as e:
        logger.error(f"获取模型统计异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": "获取模型统计失败"})


@router.get("/api/token/users")
async def api_token_users():
    """获取按用户统计的 token 使用情况"""
    try:
        user_stats = token_monitor.get_user_stats()
        return JSONResponse(content={"success": True, "data": user_stats})
    except Exception as e:
        logger.error(f"获取用户统计异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": "获取用户统计失败"})


@router.get("/api/token/recent")
async def api_token_recent(limit: int = 100):
    limit = min(max(1, limit), 500)  # 限制范围 1-500
    """获取最近的 token 使用记录"""
    try:
        records = token_monitor.get_recent_records(limit)
        return JSONResponse(content={"success": True, "data": records, "count": len(records)})
    except Exception as e:
        logger.error(f"获取最近记录异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": "获取最近记录失败"})
