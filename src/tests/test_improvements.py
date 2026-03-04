"""
改进功能测试脚本
验证代码结构和基本逻辑
"""

import sys
import os
import ast

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


def check_file_syntax(filepath):
    """检查 Python 文件语法"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, str(e)


def test_file_syntax():
    """测试所有修改的文件语法"""
    print("\n=== 测试文件语法 ===")
    
    files_to_check = [
        'src/services/ai/embedding_service.py',
        'src/services/database.py',
        'src/services/memory_manager.py',
        'src/services/rag_engine.py',
        'src/wecom/handlers.py',
    ]
    
    all_passed = True
    for filepath in files_to_check:
        full_path = os.path.join(os.path.dirname(__file__), '../..', filepath)
        passed, error = check_file_syntax(full_path)
        if passed:
            print(f"[OK] {filepath}")
        else:
            print(f"[FAIL] {filepath}: {error}")
            all_passed = False
    
    return all_passed


def test_embedding_service_structure():
    """测试 EmbeddingService 结构"""
    print("\n=== 测试 EmbeddingService 结构 ===")
    
    filepath = os.path.join(os.path.dirname(__file__), '../../src/services/ai/embedding_service.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())
    
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    assert 'SiliconFlowEmbedding' in classes, "SiliconFlowEmbedding 类不存在"
    assert 'SiliconFlowReranker' in classes, "SiliconFlowReranker 类不存在"
    assert 'EmbeddingService' in classes, "EmbeddingService 类不存在"
    
    print("[OK] SiliconFlowEmbedding 类存在")
    print("[OK] SiliconFlowReranker 类存在")
    print("[OK] EmbeddingService 类存在")
    
    return True


def test_database_structure():
    """测试数据库模型结构"""
    print("\n=== 测试数据库模型结构 ===")
    
    filepath = os.path.join(os.path.dirname(__file__), '../../src/services/database.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())
    
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    assert 'ConversationCounter' in classes, "ConversationCounter 类不存在"
    assert 'KBSearchSession' in classes, "KBSearchSession 类不存在"
    
    print("[OK] ConversationCounter 类存在")
    print("[OK] KBSearchSession 类存在")
    
    return True


def test_memory_manager_structure():
    """测试 MemoryManager 结构"""
    print("\n=== 测试 MemoryManager 结构 ===")
    
    filepath = os.path.join(os.path.dirname(__file__), '../../src/services/memory_manager.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())
    
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    assert 'MemoryManager' in classes, "MemoryManager 类不存在"
    
    # 检查新方法
    assert '_get_or_create_counter' in functions, "_get_or_create_counter 方法不存在"
    assert '_increment_core_count' in functions, "_increment_core_count 方法不存在"
    assert '_increment_vector_count' in functions, "_increment_vector_count 方法不存在"
    assert '_do_update_vector_memory' in functions, "_do_update_vector_memory 方法不存在"
    assert '_do_update_core_memory' in functions, "_do_update_core_memory 方法不存在"
    
    print("[OK] MemoryManager 类存在")
    print("[OK] 双独立计数器方法存在")
    print("[OK] 向量/核心记忆更新子方法存在")
    
    return True


def test_rag_engine_structure():
    """测试 RAGEngine 结构"""
    print("\n=== 测试 RAGEngine 结构 ===")
    
    filepath = os.path.join(os.path.dirname(__file__), '../../src/services/rag_engine.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())
    
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    assert 'RAGEngine' in classes, "RAGEngine 类不存在"
    
    # 检查新方法
    assert 'get_document_outline' in functions, "get_document_outline 方法不存在"
    assert 'format_search_results' in functions, "format_search_results 方法不存在"
    
    print("[OK] RAGEngine 类存在")
    print("[OK] get_document_outline 方法存在")
    print("[OK] format_search_results 方法存在")
    
    return True


def test_handlers_structure():
    """测试 MessageHandler 结构"""
    print("\n=== 测试 MessageHandler 结构 ===")
    
    filepath = os.path.join(os.path.dirname(__file__), '../../src/wecom/handlers.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())
    
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    assert 'MessageHandler' in classes, "MessageHandler 类不存在"
    
    # 检查新方法
    assert '_check_kb_intent_v2' in functions, "_check_kb_intent_v2 方法不存在"
    assert '_handle_kb_outline_command' in functions, "_handle_kb_outline_command 方法不存在"

    print("[OK] MessageHandler 类存在")
    print("[OK] _check_kb_intent_v2 方法存在")
    print("[OK] 文档大纲命令处理方法存在")
    
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("AtriNexus Improvement Test")
    print("=" * 60)
    
    tests = [
        test_file_syntax,
        test_embedding_service_structure,
        test_database_structure,
        test_memory_manager_structure,
        test_rag_engine_structure,
        test_handlers_structure,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n[FAIL] {test.__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
