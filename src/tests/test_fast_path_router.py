import ast
import importlib.util
from pathlib import Path
import sys
import unittest


ROUTER_PATH = Path(__file__).resolve().parents[1] / "conversation" / "fast_path_router.py"
INTENTS_PATH = Path(__file__).resolve().parents[1] / "conversation" / "fast_path_intents.py"
INTENTS_SPEC = importlib.util.spec_from_file_location("fast_path_intents", INTENTS_PATH)
INTENTS_MODULE = importlib.util.module_from_spec(INTENTS_SPEC)
assert INTENTS_SPEC and INTENTS_SPEC.loader
sys.modules["fast_path_intents"] = INTENTS_MODULE
INTENTS_SPEC.loader.exec_module(INTENTS_MODULE)
extract_directory_path = INTENTS_MODULE.extract_directory_path
extract_workspace_browse_request = INTENTS_MODULE.extract_workspace_browse_request


class FastPathRouterStructureTest(unittest.TestCase):
    def test_dispatch_helpers_do_not_exist_as_router_methods(self):
        tree = ast.parse(ROUTER_PATH.read_text(encoding="utf-8"))
        functions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        self.assertNotIn("_dispatch_before_remembered_action", functions)
        self.assertNotIn("_dispatch_after_remembered_action", functions)

    def test_try_handle_order_is_preserved_in_source(self):
        source = ROUTER_PATH.read_text(encoding="utf-8")
        markers = [
            "block_rewrite_request = self._extract_block_rewrite_request(normalized_message)",
            "replace_request = self._extract_replace_request(normalized_message)",
            "rewrite_request = self._extract_rewrite_request(normalized_message)",
            "append_request = self._extract_append_request(normalized_message)",
            "rename_paths = self._extract_rename_paths(normalized_message)",
            "browse_request = self._extract_workspace_browse_request(user_id, normalized_message)",
            "browse_reply = self._handle_workspace_browse_request(user_id, browse_request)",
        ]

        positions = [source.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))

    def test_try_handle_keeps_router_local_dispatch_and_pending_short_circuit(self):
        source = ROUTER_PATH.read_text(encoding="utf-8")
        self.assertIn("def dispatch_before_remembered_action(", source)
        self.assertIn("def dispatch_after_remembered_action(", source)
        self.assertEqual(source.count("pending_reply = self.path_resolver.take_pending_reply()"), 3)

    def test_router_local_dispatch_preserves_boundary_critical_semantics(self):
        source = ROUTER_PATH.read_text(encoding="utf-8")

        self.assertIn('self._promote_tool_profile(user_id, inferred_profile)', source)
        self.assertIn(
            'self.session_service.set_last_workspace_target(user_id, remembered_path(request), target_type)',
            source,
        )
        self.assertIn('if not reply.startswith(blocked_prefixes):', source)
        self.assertNotIn('from src.conversation.fast_path_dispatch import', source)

    def test_try_handle_uses_unified_workspace_browse_request(self):
        source = ROUTER_PATH.read_text(encoding="utf-8")
        self.assertIn('browse_request = self._extract_workspace_browse_request(user_id, normalized_message)', source)
        self.assertNotIn('dir_path = self._extract_directory_path(normalized_message)', source)
        self.assertNotIn('file_path = self._extract_read_file_path(normalized_message)', source)

    def test_directory_intent_accepts_followup_contents_phrase(self):
        self.assertEqual(extract_directory_path("docs里有什么"), "docs")
        self.assertEqual(extract_directory_path("docs里有哪些"), "docs")

    def test_extract_workspace_browse_request_uses_focus_for_followups(self):
        browser_state = {"focus": {"path": "docs", "type": "dir"}}
        request = extract_workspace_browse_request("看看它", browser_state)
        self.assertIsNotNone(request)
        self.assertEqual(request.intent, "browse_list")
        self.assertEqual(request.reference_mode, "focus_dir")

        browser_state = {"focus": {"path": "README.md", "type": "file"}}
        request = extract_workspace_browse_request("这个文件最后一行", browser_state)
        self.assertIsNotNone(request)
        self.assertEqual(request.intent, "browse_read_line")
        self.assertEqual(request.reference_mode, "focus_file")
        self.assertEqual(request.line_position, "last")

    def test_extract_workspace_browse_request_supports_search_with_focus_dir(self):
        browser_state = {"focus": {"path": "src", "type": "dir"}}
        request = extract_workspace_browse_request("在这个目录搜索 router", browser_state)
        self.assertIsNotNone(request)
        self.assertEqual(request.intent, "browse_search")
        self.assertEqual(request.reference_mode, "focus_dir")
        self.assertEqual(request.query, "router")


if __name__ == "__main__":
    unittest.main()
