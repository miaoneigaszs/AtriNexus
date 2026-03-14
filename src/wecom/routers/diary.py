"""
日记路由
处理日记的查看、生成、删除等操作
"""

import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.wecom.deps import logger, validate_user_id, get_diary_service

router = APIRouter(tags=["日记"])


@router.get("/api/diary")
async def api_diary_get(user_id: str, avatar_name: str = "ATRI", date: str = None):
    """
    获取指定日期的日记
    
    Args:
        user_id: 用户ID
        avatar_name: 人设名称
        date: 日期 (YYYY-MM-DD)，默认为昨天
    """
    if not validate_user_id(user_id):
        return JSONResponse(status_code=400, content={"success": False, "message": "无效的 UserID"})
    
    try:
        diary_service = get_diary_service()
        
        if not date:
            date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        diary = diary_service.get_diary(user_id, avatar_name, date)
        
        if diary:
            return JSONResponse(content={
                "success": True,
                "data": {
                    "id": diary.id,
                    "user_id": diary.user_id,
                    "avatar_name": diary.avatar_name,
                    "date": diary.date,
                    "content": diary.content,
                    "conversation_count": diary.conversation_count,
                    "created_at": diary.created_at.isoformat()
                }
            })
        else:
            return JSONResponse(content={
                "success": True,
                "data": None,
                "message": "该日期暂无日记"
            })
    except Exception as e:
        logger.error(f"获取日记异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"获取失败: {str(e)}"})


@router.get("/api/diary/list")
async def api_diary_list(
    user_id: str, 
    avatar_name: str = "ATRI",
    start_date: str = None,
    end_date: str = None,
    limit: int = 30
):
    """
    获取日记列表
    
    Args:
        user_id: 用户ID
        avatar_name: 人设名称
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        limit: 最大返回数量
    """
    if not validate_user_id(user_id):
        return JSONResponse(status_code=400, content={"success": False, "message": "无效的 UserID"})
    
    try:
        diary_service = get_diary_service()
        
        # 默认查询最近30天
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        diaries = diary_service.get_diaries_by_range(user_id, avatar_name, start_date, end_date, limit)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "user_id": user_id,
                "avatar_name": avatar_name,
                "start_date": start_date,
                "end_date": end_date,
                "count": len(diaries),
                "diaries": [
                    {
                        "id": d.id,
                        "date": d.date,
                        "content": d.content,
                        "conversation_count": d.conversation_count,
                        "created_at": d.created_at.isoformat()
                    }
                    for d in diaries
                ]
            }
        })
    except Exception as e:
        logger.error(f"获取日记列表异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"获取失败: {str(e)}"})


@router.get("/api/diary/dates")
async def api_diary_dates(user_id: str, avatar_name: str = "ATRI", year: int = None, month: int = None):
    """
    获取某月有日记的日期列表
    
    Args:
        user_id: 用户ID
        avatar_name: 人设名称
        year: 年份
        month: 月份
    """
    if not validate_user_id(user_id):
        return JSONResponse(status_code=400, content={"success": False, "message": "无效的 UserID"})
    
    try:
        diary_service = get_diary_service()
        
        # 默认当前月份
        now = datetime.now()
        if not year:
            year = now.year
        if not month:
            month = now.month
        
        dates = diary_service.get_diary_dates_by_month(user_id, avatar_name, year, month)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "user_id": user_id,
                "avatar_name": avatar_name,
                "year": year,
                "month": month,
                "dates": dates,
                "count": len(dates)
            }
        })
    except Exception as e:
        logger.error(f"获取日记日期异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"获取失败: {str(e)}"})


@router.post("/api/diary/generate")
async def api_diary_generate(request: Request):
    """
    手动触发生成日记
    
    Body:
        user_id: 用户ID
        avatar_name: 人设名称
        date: 日期 (YYYY-MM-DD)，可选
        force_regenerate: 是否强制重新生成
    """
    try:
        body = await request.json()
        user_id = body.get('user_id')
        avatar_name = body.get('avatar_name', 'ATRI')
        date_str = body.get('date')
        force_regenerate = body.get('force_regenerate', False)
        
        if not validate_user_id(user_id):
            return JSONResponse(status_code=400, content={"success": False, "message": "无效的 UserID"})
        
        diary_service = get_diary_service()
        diary = await diary_service.generate_diary(user_id, avatar_name, date_str, force_regenerate)
        
        if diary:
            return JSONResponse(content={
                "success": True,
                "message": "日记生成成功",
                "data": {
                    "id": diary.id,
                    "date": diary.date,
                    "content": diary.content,
                    "conversation_count": diary.conversation_count
                }
            })
        else:
            return JSONResponse(content={
                "success": True,
                "message": "该日期暂无对话记录，无法生成日记",
                "data": None
            })
    except Exception as e:
        logger.error(f"生成日记异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"生成失败: {str(e)}"})


@router.delete("/api/diary")
async def api_diary_delete(user_id: str, avatar_name: str = "ATRI", date: str = None):
    """删除指定日记"""
    if not validate_user_id(user_id):
        return JSONResponse(status_code=400, content={"success": False, "message": "无效的 UserID"})
    
    if not date:
        return JSONResponse(status_code=400, content={"success": False, "message": "请指定日期"})
    
    try:
        diary_service = get_diary_service()
        deleted = diary_service.delete_diary(user_id, avatar_name, date)
        
        if deleted:
            return JSONResponse(content={"success": True, "message": f"已删除 {date} 的日记"})
        else:
            return JSONResponse(content={"success": True, "message": "日记不存在"})
    except Exception as e:
        logger.error(f"删除日记异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"删除失败: {str(e)}"})


@router.get("/api/diary/stats")
async def api_diary_stats(user_id: str = None):
    """获取日记统计信息"""
    try:
        diary_service = get_diary_service()
        stats = diary_service.get_stats(user_id)
        
        return JSONResponse(content={"success": True, "data": stats})
    except Exception as e:
        logger.error(f"获取日记统计异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"获取统计失败: {str(e)}"})
