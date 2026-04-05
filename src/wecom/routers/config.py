"""
系统配置路由
处理系统配置的读取、保存等操作
"""

import json
import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, HTMLResponse

from data.config import config
from src.utils.async_utils import run_sync
from src.wecom.deps import message_handler, logger, load_template

router = APIRouter(tags=["系统配置"])

# ========== 页面路由 ==========

@router.get("/setting", response_class=HTMLResponse)
async def get_setting_page():
    """返回系统配置网页"""
    return load_template('setting.html')


# ========== API 路由 ==========

@router.get("/api/config")
async def api_config_get():
    """
    获取完整配置（敏感字段掩码处理）
    返回 config.json 的 categories 对象
    """
    try:
        config_data = await run_sync(_load_config_data)
        
        categories = config_data.get('categories', {})
        
        logger.info("[Config API] 配置已读取并返回给前端")
        return JSONResponse(content={"success": True, "data": categories})
    except Exception as e:
        logger.error(f"[Config API] 读取配置失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"读取配置失败: {str(e)}"})


@router.put("/api/config")
async def api_config_update(request: Request):
    """
    保存修改的配置
    接收前端提交的完整 categories 对象，合并到 config.json
    """
    try:
        body = await request.json()
        categories = body.get('categories')
        
        if not categories:
            return JSONResponse(status_code=400, content={"success": False, "message": "缺少 categories 数据"})
        
        # 使用 config 的 save_config 方法（自动备份）
        success = await run_sync(config.save_config, {"categories": categories})
        
        if success:
            logger.info("[Config API] 配置已保存")
            # 重新加载配置到内存
            try:
                await run_sync(config.load_config)
                logger.info("[Config API] 配置已重新加载到内存")
            except Exception as reload_err:
                logger.warning(f"[Config API] 配置已保存到文件，但重新加载到内存失败（需重启服务生效）: {reload_err}")
            
            return JSONResponse(content={"success": True, "message": "配置已保存"})
        else:
            return JSONResponse(status_code=500, content={"success": False, "message": "保存配置失败"})
    except Exception as e:
        logger.error(f"[Config API] 保存配置失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"保存失败: {str(e)}"})


@router.get("/api/models/refresh")
async def api_models_refresh():
    """
    刷新可用模型列表
    调用 ModelManager 的 refresh_models 重新查询 API
    """
    try:
        models = await run_sync(message_handler.llm_service.model_manager.refresh_models)
        logger.info(f"[Config API] 模型列表已刷新，共 {len(models)} 个模型")
        return JSONResponse(content={
            "success": True,
            "data": models,
            "count": len(models)
        })
    except Exception as e:
        logger.error(f"[Config API] 刷新模型列表失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"刷新失败: {str(e)}"})


def _load_config_data():
    with open(config.config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
