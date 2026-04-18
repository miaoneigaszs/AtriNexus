import ast
from pathlib import Path
import unittest


ROUTER_PATH = Path(__file__).resolve().parents[1] / "conversation" / "fast_path_router.py"


class FastPathRouterStructureTest(unittest.TestCase):
    def test_legacy_dispatch_helpers_do_not_exist(self):
        source = ROUTER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("def dispatch_before_remembered_action(", source)
        self.assertNotIn("def dispatch_after_remembered_action(", source)

    def test_try_handle_only_contains_env_and_overview_branches(self):
        source = ROUTER_PATH.read_text(encoding="utf-8")
        self.assertIn("read_fast_path_mode() == FAST_PATH_MODE_DISABLED", source)
        self.assertIn("is_tool_overview(normalized_message)", source)
        self.assertIn("is_profile_overview(normalized_message)", source)

    def test_browse_intent_routing_is_removed(self):
        source = ROUTER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("_extract_workspace_browse_request", source)
        self.assertNotIn("_handle_workspace_browse_request", source)
        self.assertNotIn("extract_workspace_browse_request", source)
        self.assertNotIn("WorkspaceBrowseRequest", source)

    def test_edit_intent_routing_is_removed(self):
        source = ROUTER_PATH.read_text(encoding="utf-8")
        for symbol in (
            "_extract_replace_request",
            "_extract_rewrite_request",
            "_extract_block_rewrite_request",
            "_extract_append_request",
            "_extract_rename_paths",
            "_extract_followup_rename_target",
            "extract_replace_request",
            "extract_rewrite_request",
            "extract_block_rewrite_request",
            "extract_append_request",
            "extract_rename_paths",
            "extract_followup_rename_target",
        ):
            self.assertNotIn(symbol, source, f"{symbol} 应已从 router 删除")

    def test_edit_intent_constants_are_removed_from_imports(self):
        source = ROUTER_PATH.read_text(encoding="utf-8")
        for symbol in (
            "INTENT_BLOCK_REWRITE",
            "INTENT_REPLACE",
            "INTENT_REWRITE",
            "INTENT_APPEND",
            "INTENT_RENAME",
        ):
            self.assertNotIn(symbol, source, f"{symbol} 应已从 router 删除")

    def test_state_machine_methods_preserved(self):
        tree = ast.parse(ROUTER_PATH.read_text(encoding="utf-8"))
        method_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for kept in (
            "try_handle",
            "try_handle_pending_resolution",
            "_execute_resolved_action",
            "_remember_browse_result",
            "_promote_tool_profile",
            "_handle_tool_overview",
            "_handle_profile_overview",
        ):
            self.assertIn(kept, method_names, f"{kept} 必须保留")

    def test_rewrite_helper_still_instantiated(self):
        source = ROUTER_PATH.read_text(encoding="utf-8")
        self.assertIn("self.rewrite_helper = FastPathRewriteHelper(", source)


if __name__ == "__main__":
    unittest.main()
