from __future__ import annotations

import re
from typing import Any, Optional, Tuple


class FastPathRewriteHelper:
    """处理块级文档改写，避免这部分逻辑挤在主路由里。"""

    def __init__(self, runtime: Any, llm_service: Any, prompt_manager: Any) -> None:
        self.runtime = runtime
        self.llm_service = llm_service
        self.prompt_manager = prompt_manager

    def handle_block_rewrite(self, user_id: str, path: str, target: str, instruction: str) -> str:
        if not self.llm_service:
            return "当前未接入段落改写能力，请改用精确替换或整文件预览修改。"

        target_file = self.runtime._resolve_path(path)
        if not target_file.exists():
            return f"文件不存在: {path}"
        if not target_file.is_file():
            return f"目标不是文件: {path}"

        text = target_file.read_text(encoding="utf-8", errors="ignore")
        span = self.locate_rewrite_block(text, target)
        if not span:
            return f"没有在 {path} 中定位到可改写的{target}。"

        start, end = span
        original_block = text[start:end].strip()
        if not original_block:
            return f"{path} 中的{target}内容为空，无法改写。"

        rewritten = self.rewrite_block_with_llm(path, target, instruction, original_block)
        if not rewritten:
            return "改写模型没有返回有效内容，请稍后再试。"

        return self.runtime.preview_replace_span(
            path,
            start,
            end,
            rewritten.strip(),
            owner_user_id=user_id,
        )

    def locate_rewrite_block(self, text: str, target: str) -> Optional[Tuple[int, int]]:
        if target == "标题":
            return self.find_first_heading_span(text)
        return self.find_first_paragraph_span(text)

    def find_first_heading_span(self, text: str) -> Optional[Tuple[int, int]]:
        match = re.search(r"^(#.+)$", text, re.MULTILINE)
        if not match:
            return None
        return match.start(1), match.end(1)

    def find_first_paragraph_span(self, text: str) -> Optional[Tuple[int, int]]:
        match = re.search(r"\S(?:[\s\S]*?)(?:\n\s*\n|$)", text)
        if not match:
            return None
        return match.start(), match.end()

    def rewrite_block_with_llm(self, path: str, target: str, instruction: str, original_block: str) -> str:
        messages = [
            {
                "role": "system",
                "content": self.prompt_manager.build_fast_path_rewrite_prompt(),
            },
            {
                "role": "user",
                "content": (
                    f"文件路径：{path}\n"
                    f"目标块：{target}\n"
                    f"改写要求：{instruction}\n\n"
                    "请只改写下面这个已经定位好的块，不要扩写其它部分。\n\n"
                    f"原文如下：\n{original_block}"
                ),
            },
        ]
        return self.llm_service.chat(messages).strip()
