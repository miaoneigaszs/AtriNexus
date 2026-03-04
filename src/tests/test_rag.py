"""
RAG 模块测试
测试增强的知识库检索功能
"""

import pytest
from src.wecom.handlers import MessageHandler


class MockMemoryManager:
    def build_full_context(self, *args, **kwargs):
        return {
            "previous_context": "用户之前询问过关于报销和打车的规定。",
            "core_memory": "用户的名字叫张三。",
            "relevant_memories": ["打车实报实销"]
        }


class MockRAG:
    def __init__(self):
        self.knowledge_list = {
            "规章制度": ["报销规定.md", "考勤制度.md"],
            "产品文档": ["API文档.md", "使用指南.md"]
        }
    
    def get_knowledge_list(self, user_id):
        return self.knowledge_list
    
    def get_document_outline(self, user_id, file_name=None):
        return {
            "documents": {
                "报销规定.md": {
                    "category": "规章制度",
                    "H1": ["第一章 总则", "第二章 报销流程", "第三章 费用标准"],
                    "H2": ["2.1 差旅报销", "2.2 日常费用", "3.1 住宿标准"]
                }
            },
            "categories": ["规章制度", "产品文档"]
        }
    
    def retrieve_knowledge(self, user_id, query, top_k=3, category_filter=None, h1_filter=None, h2_filter=None):
        results = [
            {
                "metadata": {
                    "file_name": "报销规定.md",
                    "category": "规章制度",
                    "H1": "第二章 报销流程",
                    "H2": "2.1 差旅报销"
                },
                "content": "差旅费实报实销，需提供发票和出差申请表。",
                "score": 0.95
            }
        ]
        if category_filter and category_filter != "规章制度":
            return []
        return results


class MockLLM:
    def get_response(self, *args, **kwargs):
        return "好的"
    
    def chat(self, messages, **kwargs):
        return "模拟的 LLM 回复"


class MockWeComClient:
    def send_text(self, user_id, text):
        print(f"[发送到 {user_id}]: {text[:100]}...")


@pytest.fixture
def handler():
    h = MessageHandler(MockWeComClient())
    h.memory = MockMemoryManager()
    h.rag = MockRAG()
    h.llm_service = MockLLM()
    return h


def test_intent_router_chitchat(handler):
    """测试闲聊意图识别"""
    result = handler._check_kb_intent_v2("test_user", "你好啊", previous_context="")
    # 由于没有配置真实的 API，应该返回默认的 CHITCHAT
    assert result["intent"] == "TYPE_CHITCHAT"


def test_kb_outline_command(handler):
    """测试文档大纲命令"""
    reply = handler._handle_kb_outline_command("test_user")
    assert "报销规定.md" in reply
    assert "规章制度" in reply
    assert "第一章 总则" in reply




def test_build_kb_context(handler):
    """测试知识库上下文构建"""
    kb_results = [
        {
            "metadata": {
                "file_name": "测试.md",
                "category": "测试分类",
                "H1": "第一章",
                "H2": "1.1 节"
            },
            "content": "测试内容",
            "score": 0.9
        }
    ]
    context = handler._build_kb_context(kb_results)
    assert "测试.md" in context
    assert "测试分类" in context
    assert "第一章" in context
    assert "0.90" in context


def test_format_kb_references(handler):
    """测试参考来源格式化"""
    kb_results = [
        {
            "metadata": {
                "file_name": "报销规定.md",
                "H1": "第二章"
            }
        }
    ]
    refs = handler._format_kb_references(kb_results)
    assert "报销规定.md" in refs
    assert "第二章" in refs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
