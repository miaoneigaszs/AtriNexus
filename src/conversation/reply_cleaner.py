"""
回复清理工具
负责清理 LLM 回复中的格式问题
"""

import re


class ReplyCleaner:
    """回复清理工具"""

    @staticmethod
    def clean_reply(text: str) -> str:
        """
        清理回复中的格式问题

        Args:
            text: 原始回复文本

        Returns:
            str: 清理后的文本
        """
        # 移除 LaTeX 行内公式
        text = re.sub(r'\$([^$]+)\$', r'\1', text)
        # 移除 LaTeX 块公式
        text = re.sub(r'\$\$([^$]+)\$\$', r'\1', text)
        # 移除残留的 $ 符号
        text = text.replace('$', '')
        # 移除 markdown 加粗
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        # 移除 markdown 斜体
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        return text.strip()
