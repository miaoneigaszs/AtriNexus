"""
共享依赖模块
提供全局实例、工具函数、配置等
"""

import os
import re
import logging
import unicodedata
from datetime import datetime, timedelta

from data.config import config
from src.ingress.client import WeComClient
from src.conversation.message_handler import MessageHandler

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 模板目录
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'features', 'web', 'templates')

# 日志器
logger = logging.getLogger('wecom')

# ========== 配置加载 ==========
CORP_ID = config.wecom.corp_id
AGENT_ID = config.wecom.agent_id
SECRET = config.wecom.secret
TOKEN = config.wecom.token
ENCODING_AES_KEY = config.wecom.encoding_aes_key

# ========== 全局实例 ==========
wecom_client = WeComClient(corp_id=CORP_ID, secret=SECRET, agent_id=AGENT_ID)
message_handler = MessageHandler(wecom_client=wecom_client)

# DiaryService 单例（延迟初始化）
_diary_service = None

def get_diary_service():
    """获取 DiaryService 单例"""
    global _diary_service
    if _diary_service is None:
        from src.features.diary_service import DiaryService
        _diary_service = DiaryService(
            llm_service=message_handler.llm_service,
            memory_manager=message_handler.memory
        )
    return _diary_service


# ========== 工具函数 ==========

# UserID 校验正则（编译一次，全局复用）
USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\@\.]+$")

def validate_user_id(user_id: str) -> bool:
    """校验 UserID 合法性"""
    return bool(user_id and USER_ID_PATTERN.match(user_id))


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """安全过滤上传文件名，防御路径穿越、特殊字符、超长文件名等攻击"""
    # 1. 去除路径分隔符，只保留文件名
    name = os.path.basename(filename)
    # 2. 去除 null 字节和控制字符
    name = name.replace('\x00', '')
    name = ''.join(c for c in name if unicodedata.category(c)[0] != 'C')
    # 3. 只保留安全字符（字母、数字、中文、下划线、短横线、点）
    name = re.sub(r'[^\w\u4e00-\u9fff.\-]', '_', name)
    # 4. 防止隐藏文件（.开头）
    name = name.lstrip('.')
    # 5. 截断长度
    stem, ext = os.path.splitext(name)
    if len(stem) > max_length:
        stem = stem[:max_length]
    name = stem + ext
    # 6. 兜底
    return name or "unnamed_file"


def load_template(name: str) -> str:
    """加载 HTML 模板文件"""
    path = os.path.join(TEMPLATE_DIR, name)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Template not found."
