"""
WeCom 消息处理器模块（入口）
重构后的代码已拆分到 handlers/ 目录下
"""

# 为了保持向后兼容，从新的位置导入
from src.wecom.handlers.message_handler import MessageHandler

__all__ = ['MessageHandler']
