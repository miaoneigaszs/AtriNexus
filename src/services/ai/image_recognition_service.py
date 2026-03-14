"""
图像识别 AI 服务模块

接收图片二进制数据，调用 VL（Vision Language）模型识别内容，返回文字描述。

主要特性：
- 接收 bytes，无需落盘保存文件
- 自动检测图片 MIME 类型（JPEG / PNG / GIF / WebP），不再硬编码 image/jpeg
- 超过 500KB 时自动压缩到 1024x1024（压缩后固定为 JPEG）
- 使用共享 HTTP 连接池，异步调用，不阻塞事件循环
"""

import base64
import logging
from io import BytesIO
from typing import Optional

from src.utils.http_pool import get_async_client, build_headers

logger = logging.getLogger('wecom')

# 文件头 magic bytes → MIME
_MAGIC = [
    (b'\xff\xd8\xff',        'image/jpeg'),
    (b'\x89PNG\r\n\x1a\n',  'image/png'),
    (b'GIF87a',              'image/gif'),
    (b'GIF89a',              'image/gif'),
    (b'RIFF',                'image/webp'),   # RIFF....WEBP
]

_DEFAULT_MIME = 'image/jpeg'


def _detect_mime(data: bytes) -> str:
    """
    根据文件头 magic bytes 检测图片 MIME 类型。
    无法识别时返回 'image/jpeg' 作为保底。
    """
    for magic, mime in _MAGIC:
        if data[:len(magic)] == magic:
            # WebP：RIFF 开头还需要校验第 8-12 字节是 WEBP
            if mime == 'image/webp':
                if data[8:12] == b'WEBP':
                    return mime
                continue
            return mime
    return _DEFAULT_MIME


class ImageRecognitionService:
    """
    VL 模型图片识别服务（异步版本）。

    典型调用方式（在 handlers.py 中）：
        image_data: bytes = wecom_client.download_image(media_id)
        description = await image_service.describe(image_data)
    """

    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.3):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        # temperature 对纯描述任务意义不大，建议设 0.1~0.5
        self.temperature = min(max(0.0, temperature), 2.0)

        if not model:
            logger.warning("[ImageRecognition] 未配置 VL 模型（model 为空）")

    async def describe(self, image_data: bytes, prompt: Optional[str] = None) -> str:
        """
        识别图片内容，返回文字描述（异步版本）。

        Args:
            image_data: 图片二进制数据
            prompt:     自定义提示词；为 None 时使用默认描述 prompt

        Returns:
            描述文本。失败时返回错误提示字符串。
        """
        if not self.model:
            logger.warning("[ImageRecognition] 跳过图片识别：未配置 VL 模型")
            return "（图片识别未配置）"

        if not image_data:
            return "（收到空图片数据）"

        # 1. 压缩大图
        image_data, mime = self._maybe_compress(image_data)

        # 2. Base64 编码
        b64 = base64.b64encode(image_data).decode('utf-8')
        data_url = f"data:{mime};base64,{b64}"

        # 3. 组装 prompt
        if prompt is None:
            prompt = (
                "请用中文简洁客观地描述这张图片的内容，"
                "包括画面中的人物、场景、物品、文字等关键信息。"
                "不超过150字。"
            )

        # 4. 构建请求
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature": self.temperature
        }

        # 5. 发送请求（异步）
        try:
            client = get_async_client()
            headers = build_headers(self.api_key)
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0
            )

            if response.status_code != 200:
                logger.error(
                    f"[ImageRecognition] API 请求失败 "
                    f"status={response.status_code} body={response.text[:200]}"
                )
                return "（图片识别服务暂时不可用）"

            result = response.json()
            choices = result.get('choices') or []
            if not choices:
                logger.error(f"[ImageRecognition] API 返回空 choices: {result}")
                return "（无法解析图片内容）"

            description = choices[0]['message']['content'].strip()
            logger.info(f"[ImageRecognition] 识别完成: {description[:100]}{'...' if len(description) > 100 else ''}")
            return description

        except TimeoutError:
            logger.error("[ImageRecognition] API 请求超时")
            return "（图片识别超时）"
        except Exception as e:
            logger.error(f"[ImageRecognition] 请求异常: {e}", exc_info=True)
            return "（图片识别出错）"

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _maybe_compress(self, image_data: bytes) -> tuple[bytes, str]:
        """
        超过 500KB 时用 Pillow 缩放到 1024x1024 并转为 JPEG。
        压缩失败时返回原始数据 + 原始 MIME。

        Returns:
            (处理后的 bytes, MIME 类型字符串)
        """
        mime = _detect_mime(image_data)
        size_kb = len(image_data) // 1024

        if len(image_data) <= 500 * 1024:
            logger.debug(f"[ImageRecognition] 图片 {size_kb}KB，无需压缩，MIME={mime}")
            return image_data, mime

        try:
            from PIL import Image

            img = Image.open(BytesIO(image_data))
            img.thumbnail((1024, 1024), Image.LANCZOS)

            buf = BytesIO()
            img.save(buf, format='JPEG', quality=82)
            compressed = buf.getvalue()

            logger.info(
                f"[ImageRecognition] 压缩: {size_kb}KB → {len(compressed)//1024}KB"
            )
            return compressed, 'image/jpeg'

        except Exception as e:
            logger.warning(f"[ImageRecognition] 压缩失败，使用原图: {e}")
            return image_data, mime
