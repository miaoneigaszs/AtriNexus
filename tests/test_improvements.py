"""Structure checks for core AtriNexus modules."""

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_module(relative_path: str) -> ast.AST:
    return ast.parse((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


def class_names(tree: ast.AST) -> set[str]:
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def function_names(tree: ast.AST) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_core_files_have_valid_syntax():
    files_to_check = [
        "src/ai/embedding_service.py",
        "src/platform_core/database.py",
        "src/memory/memory_manager.py",
        "src/knowledge/rag_service.py",
        "src/conversation/message_handler.py",
        "src/conversation/pending_confirmation_handler.py",
    ]

    for relative_path in files_to_check:
        parse_module(relative_path)


def test_embedding_service_structure():
    classes = class_names(parse_module("src/ai/embedding_service.py"))

    assert "SiliconFlowEmbedding" in classes
    assert "SiliconFlowReranker" in classes
    assert "EmbeddingService" in classes


def test_database_structure():
    classes = class_names(parse_module("src/platform_core/database.py"))

    assert "ConversationCounter" in classes


def test_memory_manager_structure():
    tree = parse_module("src/memory/memory_manager.py")
    classes = class_names(tree)
    functions = function_names(tree)

    assert "MemoryManager" in classes
    assert "_increment_core_count" in functions
    assert "_increment_vector_count" in functions
    assert "_do_update_vector_memory" in functions
    assert "_do_update_core_memory" in functions
    assert "build_full_context" in functions


def test_rag_service_structure():
    classes = class_names(parse_module("src/knowledge/rag_service.py"))

    assert "SdkRAGService" in classes


def test_handlers_structure():
    tree = parse_module("src/conversation/message_handler.py")
    classes = class_names(tree)
    functions = function_names(tree)

    assert "MessageHandler" in classes
    assert "process_message" in functions
    assert "_handle_pending_action_confirmation" in functions


def test_pending_confirmation_handler_structure():
    tree = parse_module("src/conversation/pending_confirmation_handler.py")
    classes = class_names(tree)
    functions = function_names(tree)

    assert "PendingConfirmationHandler" in classes
    assert "handle" in functions
    assert "_extract_confirmation_id" in functions