"""
知识库路由
处理知识库的上传、检索、删除等操作
"""

import os
import time
import logging
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse

from src.platform_core.async_utils import run_sync
from src.ingress.deps import (
    ROOT_DIR, config, message_handler, logger,
    validate_user_id, sanitize_filename, load_template
)

router = APIRouter(tags=["知识库"])


def _build_result_metadata(result: dict) -> dict:
    metadata = dict(result.get("metadata") or {})
    if metadata:
        return metadata

    heading_path = result.get("heading_path") or []
    normalized = {
        "file_name": result.get("source_file", "") or result.get("file_name", ""),
        "category": result.get("category", ""),
    }
    for idx, title in enumerate(heading_path[:3], start=1):
        normalized[f"H{idx}"] = title
    return normalized

# ========== 页面路由 ==========

@router.get("/kb-upload", response_class=HTMLResponse)
async def get_kb_upload_page():
    """返回独立的知识库上传网页"""
    return load_template('upload.html')


# ========== API 路由 ==========

@router.post("/api/kb/upload")
async def api_kb_upload(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    category: str = Form("默认分类")
):
    """处理上传的知识库文件，支持格式校验、大小校验、清洗与向量化入库"""
    start_time = time.time()
    
    # UserID 合法性校验
    if not validate_user_id(user_id):
        return JSONResponse(status_code=400, content={"success": False, "message": "无效的 UserID 格式"})

    # 安全过滤文件名
    safe_filename = sanitize_filename(file.filename)

    # 文件类型校验 - 检查文件扩展名
    allowed_extensions = ['.pdf', '.doc', '.docx', '.txt', '.md']
    file_ext = os.path.splitext(safe_filename)[1].lower()
    if file_ext not in allowed_extensions:
        return JSONResponse(
            status_code=400, 
            content={
                "success": False, 
                "message": f"不支持的文件类型: {file_ext}。只允许: {', '.join(allowed_extensions)}"
            }
        )

    temp_dir = os.path.join(ROOT_DIR, 'data', 'knowledge', 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{int(start_time)}_{safe_filename}")
    
    try:
        content = await file.read()
        file_size = len(content)
        # 大小校验
        max_size_bytes = config.kb.file_upload.max_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            return JSONResponse(
                status_code=400, 
                content={
                    "success": False, 
                    "message": f"文件过大，最大限制为 {config.kb.file_upload.max_size_mb}MB"
                }
            )
            
        await run_sync(_write_temp_file, temp_file_path, content)
            
        logger.info(f"开始入库: {file.filename}, size={file_size}B, user_id={user_id}, category={category}")
        
        # 调用 RAG 引擎入库
        success, msg = await run_sync(
            message_handler.rag.index_document,
            user_id=user_id,
            file_name=file.filename,
            file_path=temp_file_path,
            category=category,
        )
        
        elapsed = time.time() - start_time
        if success:
            logger.info(f"入库成功: {file.filename} 耗时: {elapsed:.2f}s")
            return JSONResponse(content={"success": True, "message": msg})
        else:
            logger.error(f"入库失败: {file.filename} 耗时: {elapsed:.2f}s, msg: {msg}")
            return JSONResponse(status_code=500, content={"success": False, "message": msg})
            
    except Exception as e:
        logger.error(f"处理上传文件异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"服务器内部错误: {str(e)}"})
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass


@router.get("/api/kb/list")
async def api_kb_list(user_id: str):
    """获取指定用户的知识库文件列表 (按 category 归类)"""
    if not validate_user_id(user_id):
        return JSONResponse(status_code=400, content={"success": False, "message": "无效的 UserID"})
        
    try:
        kb_data = await run_sync(message_handler.rag.list_documents, user_id)
        return JSONResponse(content={"success": True, "data": kb_data})
    except Exception as e:
        logger.error(f"获取知识库列表异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": "获取列表失败"})


@router.delete("/api/kb/delete")
async def api_kb_delete(user_id: str, file_name: str):
    """根据文件名物理删除指定的知识库文档切片"""
    if not user_id or not file_name:
        return JSONResponse(status_code=400, content={"success": False, "message": "缺少必要参数"})
        
    try:
        deleted = await run_sync(message_handler.rag.delete_document, user_id, file_name)
        if deleted:
            logger.info(f"成功从向量库删除文档: User={user_id}, File={file_name}")
            return JSONResponse(content={"success": True, "message": f"已成功删除《{file_name}》"})
        else:
            return JSONResponse(status_code=500, content={"success": False, "message": "删除失败，文档可能不存在"})
    except Exception as e:
        logger.error(f"删除知识库文档异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": "删除异常"})


@router.get("/api/kb/outline")
async def api_kb_outline(user_id: str, file_name: str = None):
    """
    获取文档目录结构大纲

    Args:
        user_id: 用户ID
        file_name: 指定文件名，为空则返回所有文档的大纲
    """
    if not validate_user_id(user_id):
        return JSONResponse(status_code=400, content={"success": False, "message": "无效的 UserID"})

    try:
        outline = await run_sync(message_handler.rag.get_document_outline, user_id, file_name)
        return JSONResponse(content={"success": True, "data": outline})
    except Exception as e:
        logger.error(f"获取文档大纲异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": "获取大纲失败"})


@router.get("/api/kb/search")
async def api_kb_search(user_id: str, query: str, top_k: int = 10):
    top_k = min(max(1, top_k), 50)  # 限制范围 1-50
    """
    在知识库中语义检索相关内容
    
    Args:
        user_id: 用户ID
        query: 检索关键词
        top_k: 返回结果数量，默认10
    """
    if not user_id or not query:
        return JSONResponse(status_code=400, content={"success": False, "message": "缺少必要参数"})
    
    try:
        # 使用 RAG 引擎的检索功能
        retrieval = await run_sync(
            message_handler.rag.retrieve,
            user_id,
            query,
            top_k=top_k,
        )
        results = retrieval["results"]
        
        # 格式化返回结果
        chunks = []
        for r in results:
            chunks.append({
                "content": r.get("content", ""),
                "metadata": _build_result_metadata(r),
                "score": r.get("score", 0)
            })
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "user_id": user_id,
                "query": query,
                "count": len(chunks),
                "chunks": chunks
            }
        })
    except Exception as e:
        logger.error(f"知识库检索异常: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"检索失败: {str(e)}"})


def _write_temp_file(path: str, content: bytes) -> None:
    with open(path, "wb") as buffer:
        buffer.write(content)
