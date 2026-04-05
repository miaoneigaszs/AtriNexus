"""
WeCom API 客户端封装
负责:
- 封装 wechatpy 消息发送接口
- Access Token 缓存/刷新（由 wechatpy 内部管理）
- 简单速率限制
"""

import asyncio
import logging
import time
import threading
from wechatpy.enterprise import WeChatClient
from src.utils.http_pool import get_async_client

logger = logging.getLogger('wecom')


class WeComClient:
    """企业微信 API 客户端"""
    API_BASE_URL = "https://qyapi.weixin.qq.com/cgi-bin"

    def __init__(self, corp_id: str, secret: str, agent_id: str):
        """
        初始化企微客户端

        Args:
            corp_id: 企业ID
            secret: 应用Secret
            agent_id: 应用AgentID
        """
        self.agent_id = agent_id
        self.client = WeChatClient(corp_id, secret)

        # 简单速率限制：记录最近发送时间
        self._last_send_time = 0.0
        self._min_interval = 0.5  # 最小发送间隔（秒）
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()

        logger.info(f"WeComClient 初始化完成 (agent_id={agent_id})")

    def _rate_limit(self):
        """简单速率限制"""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_send_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_send_time = time.time()

    async def _rate_limit_async(self):
        """异步速率限制，避免阻塞事件循环。"""
        async with self._async_lock:
            now = time.time()
            elapsed = now - self._last_send_time
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_send_time = time.time()

    def send_text(self, user_id: str, content: str) -> bool:
        """
        发送文本消息给指定用户

        Args:
            user_id: 企微 UserID
            content: 文本内容

        Returns:
            bool: 是否发送成功
        """
        try:
            self._rate_limit()
            self.client.message.send_text(
                self.agent_id,
                user_ids=[user_id],
                content=content
            )
            logger.info(f"消息已发送给 {user_id}: {content[:50]}...")
            return True
        except Exception as e:
            logger.error(f"发送消息失败 (user={user_id}): {e}")
            return False

    async def send_text_async(self, user_id: str, content: str) -> bool:
        """异步发送文本消息给指定用户。"""
        try:
            await self._rate_limit_async()
            access_token = await asyncio.to_thread(lambda: self.client.access_token)
            response = await get_async_client().post(
                f"{self.API_BASE_URL}/message/send",
                params={"access_token": access_token},
                json={
                    "touser": user_id,
                    "toparty": "",
                    "totag": "",
                    "agentid": self.agent_id,
                    "msgtype": "text",
                    "text": {"content": content},
                    "safe": 0,
                },
            )
            response.raise_for_status()
            result = response.json()
            if result.get("errcode", 0) != 0:
                logger.error(
                    "异步发送消息失败 (user=%s): errcode=%s errmsg=%s",
                    user_id,
                    result.get("errcode"),
                    result.get("errmsg"),
                )
                return False

            logger.info(f"消息已异步发送给 {user_id}: {content[:50]}...")
            return True
        except Exception as e:
            logger.error(f"异步发送消息失败 (user={user_id}): {e}")
            return False

    def download_media(self, media_id: str) -> bytes:
        """
        下载企微临时素材（图片/语音/视频/文件）

        Args:
            media_id: 媒体文件ID

        Returns:
            bytes: 文件二进制内容，失败返回 None
        """
        try:
            response = self.client.media.download(media_id)
            if response.status_code == 200:
                logger.info(f"媒体文件下载成功 (media_id={media_id}, size={len(response.content)} bytes)")
                return response.content
            else:
                logger.error(f"媒体文件下载失败 (status={response.status_code})")
                return None
        except Exception as e:
            logger.error(f"媒体文件下载异常: {e}")
            return None

