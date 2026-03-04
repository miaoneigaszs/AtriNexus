import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.wecom.handlers import MessageHandler

class MockMemoryManager:
    def build_full_context(self, *args, **kwargs):
        return {
            "previous_context": "用户之前询问过关于报销和打车的规定。",
            "core_memory": "用户的名字叫张三。",
            "relevant_memories": ["打车实报实销"]
        }

class MockRAG:
    def retrieve_knowledge(self, user_id, query, top_k):
        return [{"metadata": {"file_name": "Test.md", "H1": "报销"}, "content": "这部分是关于报销的规定。"}]

class MockLLM:
    def get_response(self, *args, **kwargs):
        return "好的"

def test_router():
    h = MessageHandler(None, None)
    h.memory = MockMemoryManager()
    h.rag = MockRAG()
    h.llm_service = MockLLM()
    
    print("Test 1: Normal chitchat")
    need_search, rewrite, cat = h._check_kb_intent("ShengZhiShuo", "你好啊", previous_context="")
    print(f"Need search: {need_search}, Rewrite: {rewrite}, Category: {cat}")
    
    print("\nTest 2: Implicit KB inquiry")
    need_search, rewrite, cat = h._check_kb_intent("ShengZhiShuo", "那个流程是什么？", previous_context="用户之前在问报销")
    print(f"Need search: {need_search}, Rewrite: {rewrite}, Category: {cat}")

if __name__ == "__main__":
    test_router()
