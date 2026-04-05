"""
记忆管理路由
处理记忆的查看、更新、删除等操作
"""

import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, HTMLResponse

from src.services.database import Session, MemorySnapshot, ConversationCounter
from src.utils.async_utils import run_sync
from src.wecom.deps import (
    message_handler, logger, validate_user_id, load_template
)

router = APIRouter(tags=["记忆管理"])

# ========== 页面路由 ==========

@router.get("/memory", response_class=HTMLResponse)
async def get_memory_page():
    """返回记忆管理网页"""
    return load_template('memory.html')


# ========== 记忆查看 API ==========

@router.get("/api/memory/short")
async def api_memory_short(user_id: str, avatar_name: str = "default"):
    """
    获取用户的短期记忆（最近对话记录）
    
    Args:
        user_id: 用户ID
        avatar_name: 人设名称，默认为 default
    """
    if not validate_user_id(user_id):
        return JSONResponse(status_code=400, content={"success": False, "message": "无效的 UserID"})
    
    try:
        short_memory = await run_sync(message_handler.memory.get_short_memory, user_id, avatar_name)
        return JSONResponse(content={
            "success": True, 
            "data": {
                "user_id": user_id,
                "avatar_name": avatar_name,
                "memory_type": "short",
                "count": len(short_memory),
                "memories": short_memory
            }
        })
    except Exception as e:
        logger.error(f"获取短期记忆异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"获取失败: {str(e)}"})


@router.get("/api/memory/core")
async def api_memory_core(user_id: str, avatar_name: str = "default"):
    """
    获取用户的核心记忆（关键信息摘要）
    
    Args:
        user_id: 用户ID
        avatar_name: 人设名称，默认为 default
    """
    if not validate_user_id(user_id):
        return JSONResponse(status_code=400, content={"success": False, "message": "无效的 UserID"})
    
    try:
        core_memory = await run_sync(message_handler.memory.get_core_memory, user_id, avatar_name)
        return JSONResponse(content={
            "success": True,
            "data": {
                "user_id": user_id,
                "avatar_name": avatar_name,
                "memory_type": "core",
                "content": core_memory
            }
        })
    except Exception as e:
        logger.error(f"获取核心记忆异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"获取失败: {str(e)}"})


@router.get("/api/memory/vector")
async def api_memory_vector(user_id: str, avatar_name: str = "default", limit: int = 20):
    limit = min(max(1, limit), 200)  # 限制范围 1-200
    """
    获取用户的中期记忆（向量库中的对话摘要）
    
    Args:
        user_id: 用户ID
        avatar_name: 人设名称，默认为 default
        limit: 返回条数，默认20条
    """
    if not validate_user_id(user_id):
        return JSONResponse(status_code=400, content={"success": False, "message": "无效的 UserID"})
    
    try:
        # 使用公共方法获取向量记忆
        result = await run_sync(message_handler.memory.get_vector_memories, user_id, avatar_name, limit)
        
        if not result['collection']:
            return JSONResponse(content={
                "success": True,
                "data": {
                    "user_id": user_id,
                    "avatar_name": avatar_name,
                    "memory_type": "vector",
                    "count": 0,
                    "memories": [],
                    "message": "向量存储未启用或初始化失败"
                }
            })
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "user_id": user_id,
                "avatar_name": avatar_name,
                "memory_type": "vector",
                "count": len(result['memories']),
                "total_in_store": result['total'],
                "memories": result['memories']
            }
        })
    except Exception as e:
        logger.error(f"获取向量记忆异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"获取失败: {str(e)}"})


@router.get("/api/memory/search")
async def api_memory_search(user_id: str, query: str, avatar_name: str = "default", top_k: int = 5):
    """
    语义搜索用户的中期记忆
    
    Args:
        user_id: 用户ID
        query: 搜索关键词
        avatar_name: 人设名称，默认为 default
        top_k: 返回最相关的N条，默认5条
    """
    if not validate_user_id(user_id):
        return JSONResponse(status_code=400, content={"success": False, "message": "无效的 UserID"})
    
    if not query:
        return JSONResponse(status_code=400, content={"success": False, "message": "搜索关键词不能为空"})
    
    try:
        results = await run_sync(
            message_handler.memory.search_relevant_memories,
            user_id,
            avatar_name,
            query,
            top_k,
        )
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "user_id": user_id,
                "avatar_name": avatar_name,
                "query": query,
                "count": len(results),
                "memories": results
            }
        })
    except Exception as e:
        logger.error(f"搜索记忆异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"搜索失败: {str(e)}"})


@router.get("/api/memory/stats")
async def api_memory_stats(user_id: str = None, avatar_name: str = None):
    """
    获取记忆统计信息
    
    Args:
        user_id: 可选，指定用户则返回该用户的统计
        avatar_name: 可选，指定人设
    """
    try:
        stats = await run_sync(_build_memory_stats, user_id, avatar_name)
        return JSONResponse(content={"success": True, "data": stats})
            
    except Exception as e:
        logger.error(f"获取记忆统计异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"获取统计失败: {str(e)}"})


# ========== 记忆管理 API (增删改) ==========

@router.post("/api/memory/core")
async def api_memory_core_update(request: Request):
    """
    更新核心记忆
    
    Body:
        user_id: 用户ID
        avatar_name: 人设名称
        content: 核心记忆内容
    """
    try:
        body = await request.json()
        user_id = body.get('user_id')
        avatar_name = body.get('avatar_name', 'default')
        content = body.get('content', '')
        
        if not validate_user_id(user_id):
            return JSONResponse(status_code=400, content={"success": False, "message": "无效的 UserID"})
        
        await run_sync(message_handler.memory.save_core_memory, user_id, avatar_name, content)
        logger.info(f"核心记忆已更新: user={user_id}, avatar={avatar_name}")
        return JSONResponse(content={"success": True, "message": "核心记忆已保存"})
        
    except Exception as e:
        logger.error(f"更新核心记忆异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"保存失败: {str(e)}"})


@router.delete("/api/memory/core")
async def api_memory_core_delete(user_id: str, avatar_name: str = "default"):
    """删除核心记忆"""
    if not validate_user_id(user_id):
        return JSONResponse(status_code=400, content={"success": False, "message": "无效的 UserID"})
    
    try:
        deleted = await run_sync(_delete_memory_snapshot, user_id, avatar_name, "core")
        if deleted:
            logger.info(f"核心记忆已删除: user={user_id}, avatar={avatar_name}")
            return JSONResponse(content={"success": True, "message": "核心记忆已清空"})
        return JSONResponse(content={"success": True, "message": "核心记忆本就为空"})
    except Exception as e:
        logger.error(f"删除核心记忆异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"删除失败: {str(e)}"})


@router.delete("/api/memory/short")
async def api_memory_short_delete(user_id: str, avatar_name: str = "default"):
    """清空短期记忆"""
    if not validate_user_id(user_id):
        return JSONResponse(status_code=400, content={"success": False, "message": "无效的 UserID"})
    
    try:
        await run_sync(_delete_memory_snapshot, user_id, avatar_name, "short")
        logger.info(f"短期记忆已清空: user={user_id}, avatar={avatar_name}")
        return JSONResponse(content={"success": True, "message": "短期记忆已清空"})
    except Exception as e:
        logger.error(f"清空短期记忆异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"清空失败: {str(e)}"})


@router.delete("/api/memory/vector")
async def api_memory_vector_delete(user_id: str, avatar_name: str = "default", memory_id: str = None):
    """
    删除向量记忆
    
    Args:
        user_id: 用户ID
        avatar_name: 人设名称
        memory_id: 记忆ID（可选，不提供则清空全部）
    """
    if not validate_user_id(user_id):
        return JSONResponse(status_code=400, content={"success": False, "message": "无效的 UserID"})
    
    try:
        # 使用公共方法删除向量记忆
        success = await run_sync(message_handler.memory.delete_vector_memory, user_id, avatar_name, memory_id)
        
        if success:
            if memory_id:
                return JSONResponse(content={"success": True, "message": "记忆已删除"})
            else:
                return JSONResponse(content={"success": True, "message": "向量记忆已清空"})
        else:
            return JSONResponse(status_code=500, content={"success": False, "message": "删除失败"})
            
    except Exception as e:
        logger.error(f"删除向量记忆异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"删除失败: {str(e)}"})


def _build_memory_stats(user_id: str = None, avatar_name: str = None):
    with Session() as session:
        stats = {
            "total_short_memories": 0,
            "total_core_memories": 0,
            "total_conversation_counters": 0,
            "vector_collections": []
        }

        if user_id:
            stats["total_short_memories"] = session.query(MemorySnapshot).filter_by(
                user_id=user_id, avatar_name=avatar_name or "default", memory_type='short'
            ).count()
            stats["total_core_memories"] = session.query(MemorySnapshot).filter_by(
                user_id=user_id, avatar_name=avatar_name or "default", memory_type='core'
            ).count()
        else:
            stats["total_short_memories"] = session.query(MemorySnapshot).filter_by(
                memory_type='short'
            ).count()
            stats["total_core_memories"] = session.query(MemorySnapshot).filter_by(
                memory_type='core'
            ).count()

        stats["total_conversation_counters"] = session.query(ConversationCounter).count()
        stats["vector_collections"] = message_handler.memory.get_vector_store_stats()
        return stats


def _delete_memory_snapshot(user_id: str, avatar_name: str, memory_type: str) -> bool:
    with Session() as session:
        snapshot = session.query(MemorySnapshot).filter_by(
            user_id=user_id,
            avatar_name=avatar_name,
            memory_type=memory_type,
        ).first()
        if not snapshot:
            return False
        session.delete(snapshot)
        session.commit()
        return True
