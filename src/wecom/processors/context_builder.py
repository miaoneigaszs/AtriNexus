"""
上下文构建器
负责构建各种上下文（记忆、知识库、系统提示词等）
"""

import logging
import os
import re
from typing import Dict, Any, List, Tuple

from src.services.memory_manager import MemoryManager
from src.services.session_service import SessionService
from data.config import config

logger = logging.getLogger('wecom')


class ContextBuilder:
    """上下文构建器"""

    @staticmethod
    def _normalize_kb_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
        """兼容不同来源的检索结果结构。"""
        metadata = dict(result.get("metadata") or {})
        if metadata:
            return metadata

        heading_path = result.get("heading_path") or []
        normalized = {
            "file_name": result.get("source_file", "") or result.get("file_name", ""),
            "category": result.get("category", ""),
        }
        for idx, title in enumerate(heading_path[:3], start=1):
            normalized[f"H{idx}"] = title
        return normalized

    def __init__(self, memory_manager: MemoryManager, session_service: SessionService, root_dir: str):
        """
        初始化上下文构建器

        Args:
            memory_manager: 记忆管理器
            session_service: 会话服务
            root_dir: 项目根目录
        """
        self.memory = memory_manager
        self.session_service = session_service
        self.root_dir = root_dir
        self.avatar_dir = config.behavior.context.avatar_dir

    def build_search_context(self, user_id: str, content: str) -> Dict[str, Any]:
        """
        构建检索上下文

        Args:
            user_id: 用户ID
            content: 消息内容

        Returns:
            Dict: 包含会话状态、人设、模式、记忆等上下文信息
        """
        state = self.session_service.get_session(user_id)
        avatar_name = state.avatar_name or 'ATRI'
        current_mode = self._resolve_mode(user_id, content, state.mode or 'work')
        mem_ctx = self._build_memory_context(user_id, avatar_name, content)

        return {
            "state": state,
            "avatar_name": avatar_name,
            "current_mode": current_mode,
            "mem_ctx": mem_ctx,
            "previous_context": mem_ctx["previous_context"],
        }

    def _resolve_mode(self, user_id: str, content: str, current_mode: str) -> str:
        if not self._check_companion_trigger(content):
            return current_mode

        self.session_service.update_session_mode(user_id, 'companion')
        logger.info(f"用户 {user_id} 切换至陪伴模式")
        return 'companion'

    def _build_memory_context(self, user_id: str, avatar_name: str, content: str) -> Dict[str, Any]:
        return self.memory.build_full_context(user_id, avatar_name, content)

    def build_merged_memory_context(self, mem_ctx: Dict[str, Any]) -> str:
        """
        构建合并后的记忆上下文

        Args:
            mem_ctx: 记忆上下文

        Returns:
            str: 合并后的记忆文本
        """
        core_memory = mem_ctx["core_memory"]

        if mem_ctx["relevant_memories"]:
            relevant_text = "\n".join(f"- {m}" for m in mem_ctx["relevant_memories"])
            if core_memory:
                core_memory = f"{core_memory}\n\n【相关历史对话记忆】\n{relevant_text}"
            else:
                core_memory = f"【相关历史对话记忆】\n{relevant_text}"

        return core_memory

    def build_system_prompt(self, avatar_name: str, current_mode: str) -> str:
        """
        构建系统提示词

        Args:
            avatar_name: 人设名称
            current_mode: 当前模式

        Returns:
            str: 系统提示词
        """
        # 加载人设提示词
        system_prompt = self._load_avatar_prompt(avatar_name)

        # 添加模式提示
        if current_mode == 'work' and system_prompt:
            system_prompt += "\n\n【当前模式：工作模式】请以专业、简洁、高效的方式回应。"

        return system_prompt

    def build_kb_context(self, kb_results: List[Dict], include_headers: bool = True) -> str:
        """
        构建知识库上下文，为每个片段分配编号供LLM引用

        Args:
            kb_results: 检索结果列表
            include_headers: 是否包含标题结构

        Returns:
            str: 知识库上下文文本
        """
        if not kb_results:
            return ""

        kb_context = self._build_kb_context_header()

        for i, res in enumerate(kb_results):
            kb_context += self._format_kb_item(i + 1, res, include_headers)

        return kb_context

    def _build_kb_context_header(self) -> str:
        return (
            "[从知识库中检索到的参考资料]：\n"
            "【引用说明】如果你在回复中使用了某个参考片段，请在相关句子末尾用 [1][2][3] 等标注来源编号。\n"
            "如果所有片段都与问题无关，请完全忽略它们，自然回复即可。\n\n"
        )

    def _format_kb_item(self, ref_num: int, result: Dict[str, Any], include_headers: bool) -> str:
        meta = self._normalize_kb_metadata(result)
        lines = [f"[{ref_num}] 来源：《{meta.get('file_name', '未知文件')}》"]
        category = meta.get('category', '')
        if category:
            lines[0] += f" ({category})"

        if include_headers:
            titles = [meta.get(f"H{j}") for j in range(1, 4) if meta.get(f"H{j}")]
            if titles:
                lines.append(f"    章节：{' > '.join(titles)}")

        lines.append(f"    内容：{result['content']}")
        return "\n".join(lines) + "\n\n"

    def format_kb_references(self, kb_results: List[Dict], used_indices: List[int] = None) -> str:
        """
        格式化参考来源标注，只展示被使用的片段

        Args:
            kb_results: 检索结果列表
            used_indices: 被使用的片段索引列表（从1开始），None则展示所有

        Returns:
            str: 参考来源文本
        """
        if not kb_results:
            return ""

        # 如果指定了使用的片段，只展示这些
        if used_indices is not None:
            kb_results = [r for i, r in enumerate(kb_results, 1) if i in used_indices]

        if not kb_results:
            return ""

        references = []
        for res in kb_results:
            meta = self._normalize_kb_metadata(res)
            file_name = meta.get('file_name', '')
            h1 = meta.get('H1', '')
            if file_name:
                ref = f"《{file_name}》"
                if h1:
                    ref += f"({h1})"
                if ref not in references:
                    references.append(ref)

        if references:
            return f"\n\n🔍 **参考依据**: " + ", ".join(references)
        return ""

    def _load_avatar_prompt(self, avatar_name: str) -> str:
        """加载人设提示词（纯粹读取用户定义角色，不干扰额外世界设定）"""
        prompt = ""

        # 加载角色专属人设
        avatar_path = os.path.join(self.root_dir, 'data', 'avatars', avatar_name, 'avatar.md')
        try:
            with open(avatar_path, 'r', encoding='utf-8') as f:
                prompt += f.read()
        except FileNotFoundError:
            logger.warning(f"人设文件不存在: {avatar_path}，当前无预设人设")

        return prompt

    def _check_companion_trigger(self, content: str) -> bool:
        """检查是否触发陪伴模式"""
        triggers = config.companion_mode.triggers
        for trigger in triggers:
            if trigger.lower() in content.lower():
                return True
        return False

    def extract_and_clean_references(self, reply: str) -> Tuple[str, List[int]]:
        """
        从回复中提取引用标记并清理

        Args:
            reply: LLM 回复文本

        Returns:
            Tuple[str, List[int]]: (清理后的文本, 引用的片段编号列表)
        """
        # 匹配 [1], [2], [3] 等引用标记（可能是连续的如 [1][2]）
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, reply)

        # 提取唯一的引用编号
        used_indices = list(set(int(m) for m in matches))

        cleaned_reply = re.sub(pattern, '', reply)
        cleaned_reply = self._normalize_reply_spacing(cleaned_reply)

        if used_indices:
            logger.info(f"[RAG引用] LLM使用了知识片段: {sorted(used_indices)}")

        return cleaned_reply, sorted(used_indices)

    def _normalize_reply_spacing(self, text: str) -> str:
        normalized_lines = [
            re.sub(r'[ \t]{2,}', ' ', line).rstrip()
            for line in text.splitlines()
        ]
        normalized = "\n".join(normalized_lines)
        normalized = re.sub(r'\n{3,}', '\n\n', normalized)
        return normalized.strip()

