import ast
from pathlib import Path
import unittest


ROUTER_PATH = Path(__file__).resolve().parents[1] / "conversation" / "fast_path_router.py"


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
            "read_line_request = self._extract_read_file_line_request(normalized_message)",
            "search_request = self._extract_search_request(normalized_message)",
            "file_path = self._extract_read_file_path(normalized_message)",
            "dir_path = self._extract_directory_path(normalized_message)",
            "followup_reply = self._handle_followup_reference(user_id, normalized_message)",
        ]

        positions = [source.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))

    def test_try_handle_keeps_router_local_dispatch_and_pending_short_circuit(self):
        source = ROUTER_PATH.read_text(encoding="utf-8")
        self.assertIn("def dispatch_before_remembered_action(", source)
        self.assertIn("def dispatch_after_remembered_action(", source)
        self.assertEqual(source.count("pending_reply = self.path_resolver.take_pending_reply()"), 2)

    def test_router_local_dispatch_preserves_boundary_critical_semantics(self):
        source = ROUTER_PATH.read_text(encoding="utf-8")

        self.assertIn('self._promote_tool_profile(user_id, inferred_profile)', source)
        self.assertIn(
            'self.session_service.set_last_workspace_target(user_id, remembered_path(request), target_type)',
            source,
        )
        self.assertIn('if not reply.startswith(blocked_prefixes):', source)
        self.assertNotIn('from src.conversation.fast_path_dispatch import', source)


if __name__ == "__main__":
    unittest.main()
