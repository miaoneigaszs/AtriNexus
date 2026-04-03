"""
Agent 审批路由
处理待审批变更的查看、应用和丢弃。
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.services.agent import LangChainAgentService
from src.wecom.deps import message_handler, logger

router = APIRouter(tags=["Agent"])


def _get_agent_service() -> LangChainAgentService | None:
    reply_service = getattr(message_handler, "reply_service", None)
    if isinstance(reply_service, LangChainAgentService):
        return reply_service
    return None


@router.get("/api/agent/pending-changes")
async def list_pending_changes():
    service = _get_agent_service()
    if service is None:
        return JSONResponse(status_code=400, content={"success": False, "message": "当前未启用 LangChain agent"})

    return JSONResponse(content={
        "success": True,
        "data": {
            "pending_changes": service.list_pending_changes()
        }
    })


@router.post("/api/agent/pending-changes/apply")
async def apply_pending_change(request: Request):
    service = _get_agent_service()
    if service is None:
        return JSONResponse(status_code=400, content={"success": False, "message": "当前未启用 LangChain agent"})

    try:
        body = await request.json()
        change_id = str(body.get("change_id", "")).strip()
        if not change_id:
            return JSONResponse(status_code=400, content={"success": False, "message": "缺少 change_id"})

        result = service.apply_pending_change(change_id)
        return JSONResponse(content={"success": True, "message": result})
    except Exception as e:
        logger.error(f"应用待审批变更失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"应用失败: {str(e)}"})


@router.post("/api/agent/pending-changes/discard")
async def discard_pending_change(request: Request):
    service = _get_agent_service()
    if service is None:
        return JSONResponse(status_code=400, content={"success": False, "message": "当前未启用 LangChain agent"})

    try:
        body = await request.json()
        change_id = str(body.get("change_id", "")).strip()
        if not change_id:
            return JSONResponse(status_code=400, content={"success": False, "message": "缺少 change_id"})

        result = service.discard_pending_change(change_id)
        return JSONResponse(content={"success": True, "message": result})
    except Exception as e:
        logger.error(f"丢弃待审批变更失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"丢弃失败: {str(e)}"})
