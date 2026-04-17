"""
企微回调路由
处理企业微信的消息回调
"""

import logging
from fastapi import APIRouter, Request, Response, BackgroundTasks
from fastapi.responses import JSONResponse
from wechatpy.enterprise import parse_message
from wechatpy.enterprise.crypto import WeChatCrypto

from src.ingress.deps import (
    CORP_ID, TOKEN, ENCODING_AES_KEY,
    wecom_client, message_handler, logger
)

router = APIRouter(tags=["企业微信回调"])

# 初始化加密实例
crypto = WeChatCrypto(TOKEN, ENCODING_AES_KEY, CORP_ID)


@router.get("/api/wechat/callback")
async def wechat_verify(msg_signature: str, timestamp: str, nonce: str, echostr: str):
    """企业微信服务器验证URL有效性"""
    try:
        decrypted_echostr = crypto.check_signature(msg_signature, timestamp, nonce, echostr)
        return Response(content=decrypted_echostr, status_code=200)
    except Exception as e:
        logger.error(f"URL验证失败: {e}")
        return Response(content=f"Error: {e}", status_code=403)


@router.post("/api/wechat/callback")
async def wechat_callback(request: Request, background_tasks: BackgroundTasks):
    """
    接收并处理企业微信发送的消息

    关键设计：立即返回200，消息异步处理，防止企微5秒超时重试
    """
    msg_signature = request.query_params.get('msg_signature')
    timestamp = request.query_params.get('timestamp')
    nonce = request.query_params.get('nonce')

    body = await request.body()
    
    try:
        decrypted_message = crypto.decrypt_message(body, msg_signature, timestamp, nonce)
        msg = parse_message(decrypted_message)

        if msg.type == 'text':
            logger.info(f"收到文本消息: '{msg.content[:50]}' from {msg.source}")
            # 异步处理 — 立即返回200，后台执行
            background_tasks.add_task(
                message_handler.process_message,
                user_id=msg.source,
                content=msg.content,
                msg_id=str(msg.id)
            )
        elif msg.type == 'image':
            logger.info(f"收到图片消息 from {msg.source} (media_id={msg.media_id})")
            background_tasks.add_task(
                message_handler.process_image_message,
                user_id=msg.source,
                media_id=msg.media_id,
                msg_id=str(msg.id),
                pic_url=getattr(msg, 'image', None)
            )
        else:
            logger.info(f"收到非文本消息 (type={msg.type}) from {getattr(msg, 'source', 'unknown')}")
            background_tasks.add_task(
                wecom_client.send_text,
                user_id=getattr(msg, 'source', 'unknown'),
                content="暂时只支持处理文本和图片消息哦 😊，若是上传资料请访问系统提示的网页上传入口。"
            )

        # 立即返回200 OK，防止企微5秒超时重试
        return Response(status_code=200)

    except Exception as e:
        logger.error(f"消息接收处理失败: {e}", exc_info=True)
        return Response(status_code=200)  # 即使出错也返回200，避免企微重试
