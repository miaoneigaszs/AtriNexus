"""
图片消息处理器
负责处理用户发送的图片消息
"""

import logging
from typing import Optional

from src.services.ai.image_recognition_service import ImageRecognitionService
from src.wecom.client import WeComClient
from data.config import config

logger = logging.getLogger('wecom')


class ImageHandler:
    """图片消息处理器"""

    def __init__(self, wecom_client: WeComClient):
        """
        初始化图片处理器

        Args:
            wecom_client: 企业微信客户端
        """
        self.client = wecom_client

    async def process_image(self, user_id: str, media_id: str) -> Optional[str]:
        """
        处理图片消息（异步版本）

        Args:
            user_id: 用户ID
            media_id: 媒体文件ID

        Returns:
            Optional[str]: 图片描述文本，失败返回 None
        """
        # 下载图片
        image_data = self.client.download_media(media_id)
        if not image_data:
            self.client.send_text(user_id, "抱歉，图片下载失败了，请重新发送试试 😊")
            return None

        # 识别图片内容（异步）
        image_description = await self._recognize_image(image_data)
        logger.info(f"图片描述完成: {image_description[:50]}...")

        return image_description

    async def _recognize_image(self, image_data: bytes) -> str:
        """
        调用 VL 模型识别图片内容（异步版本）

        Args:
            image_data: 图片二进制数据

        Returns:
            str: 图片描述
        """
        vl_config = config.media.image_recognition
        api_key = vl_config.api_key or config.llm.api_key
        base_url = vl_config.base_url or config.llm.base_url
        model = vl_config.model

        if not model:
            logger.warning("[Handler] 未配置图片识别模型，跳过图片识别")
            return "无法识别的图片"

        service = ImageRecognitionService(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=getattr(vl_config, 'temperature', 0.3)
        )

        description = await service.describe(image_data)
        return description
