"""
命令处理器
负责处理知识库管理相关的命令
"""

import logging
from typing import Optional

from src.knowledge.rag_service import RAGService

logger = logging.getLogger('wecom')


class CommandHandler:
    """知识库管理命令处理器"""

    def __init__(self, rag_service: RAGService):
        """
        初始化命令处理器

        Args:
            rag_service: RAG 服务实例
        """
        self.rag = rag_service

    def is_command(self, content: str) -> bool:
        """
        判断消息是否是命令

        Args:
            content: 消息内容

        Returns:
            bool: 是否是命令
        """
        content_trim = content.strip()
        return any([
            content_trim == '知识库列表',
            content_trim == '知识库结构',
            content_trim.startswith('知识库结构 '),
            content_trim.startswith('删除知识库')
        ])

    def handle_command(self, user_id: str, content: str) -> Optional[str]:
        """
        处理命令

        Args:
            user_id: 用户ID
            content: 消息内容

        Returns:
            Optional[str]: 命令回复，如果不是命令返回 None
        """
        content_trim = content.strip()

        # 1. 知识库列表
        if content_trim == '知识库列表':
            return self._handle_kb_list(user_id)

        # 2. 知识库结构
        if content_trim == '知识库结构' or content_trim.startswith('知识库结构 '):
            file_name = content_trim.replace('知识库结构', '').strip() or None
            return self._handle_kb_outline(user_id, file_name)

        # 3. 删除知识库
        if content_trim.startswith('删除知识库'):
            file_name = content_trim.replace('删除知识库', '').strip()
            return self._handle_kb_delete(user_id, file_name)

        return None

    def _handle_kb_list(self, user_id: str) -> str:
        """处理: 知识库列表"""
        kb_list = self.rag.list_documents(user_id)

        if not kb_list:
            return "您的知识库当前是空的哦~"

        reply = "📚 **您的专属知识库一览**:\n\n"
        for cat, files in kb_list.items():
            reply += f"📂 【{cat}】\n"
            for i, f in enumerate(files):
                reply += f"  {i+1}. {f}\n"

        reply += "\n发送【删除知识库 文件名】可移除特定资料。"
        reply += "\n发送【知识库结构】查看文档章节大纲。"

        return reply

    def _handle_kb_outline(self, user_id: str, file_name: Optional[str] = None) -> str:
        """处理: 知识库结构"""
        outline = self.rag.get_document_outline(user_id, file_name)

        if not outline or not outline.get("documents"):
            return "📚 您的知识库中没有文档，请先上传文档。"

        reply_parts = ["📚 知识库文档结构：\n"]

        for doc_name, structure in outline["documents"].items():
            reply_parts.append(f"\n📄 《{doc_name}》【{structure['category']}】")

            h1_list = structure.get("H1", [])
            h2_list = structure.get("H2", [])

            # 显示 H1 章节（最多10个）
            if h1_list:
                reply_parts.append("  📖 一级章节：")
                for h1 in h1_list[:10]:
                    reply_parts.append(f"    ├─ {h1}")
                if len(h1_list) > 10:
                    reply_parts.append(f"    └─ ... 还有 {len(h1_list) - 10} 个")

            # 显示 H2 子章节（最多10个）
            if h2_list:
                reply_parts.append("  📑 二级章节：")
                for h2 in h2_list[:10]:
                    reply_parts.append(f"    ├─ {h2}")
                if len(h2_list) > 10:
                    reply_parts.append(f"    └─ ... 还有 {len(h2_list) - 10} 个")

        reply_parts.append("\n💡 提示：提问时指定章节可以获得更精准的答案，如：")
        reply_parts.append('   "查一下财务制度中第一章的报销标准"')

        return "\n".join(reply_parts)

    def _handle_kb_delete(self, user_id: str, file_name: str) -> str:
        """处理: 删除知识库"""
        if not file_name:
            return "请附加要删除的文件名，例如：删除知识库 考勤规定.pdf"

        success = self.rag.delete_document(user_id, file_name)

        if success:
            return f"🗑️ 已成功从知识库中删除资料《{file_name}》"
        else:
            return f"❌ 删除失败，可能未找到《{file_name}》相关的块。"
