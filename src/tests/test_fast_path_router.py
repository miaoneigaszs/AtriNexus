import ast
from pathlib import Path
import unittest


ROUTER_PATH = Path(__file__).resolve().parents[1] / "conversation" / "fast_path_router.py"
TOOL_CATALOG_PATH = Path(__file__).resolve().parents[1] / "agent_runtime" / "tool_catalog.py"


class FastPathRouterStructureTest(unittest.TestCase):
    def test_legacy_dispatch_helpers_do_not_exist(self):
        source = ROUTER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("def dispatch_before_remembered_action(", source)
        self.assertNotIn("def dispatch_after_remembered_action(", source)

    def test_try_handle_only_contains_env_short_circuit(self):
        source = ROUTER_PATH.read_text(encoding="utf-8")
        self.assertIn("read_fast_path_mode() == FAST_PATH_MODE_DISABLED", source)
        self.assertNotIn("is_tool_overview(", source)
        self.assertNotIn("is_profile_overview(", source)

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
        ):
            self.assertNotIn(symbol, source, f"{symbol} 应已从 router 删除")

    def test_overview_intent_routing_is_removed(self):
        source = ROUTER_PATH.read_text(encoding="utf-8")
        for symbol in (
            "_handle_tool_overview",
            "_handle_profile_overview",
            "INTENT_TOOL_OVERVIEW",
            "INTENT_PROFILE_OVERVIEW",
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
        ):
            self.assertIn(kept, method_names, f"{kept} 必须保留")

    def test_rewrite_helper_still_instantiated(self):
        source = ROUTER_PATH.read_text(encoding="utf-8")
        self.assertIn("self.rewrite_helper = FastPathRewriteHelper(", source)


class ToolCatalogOverviewRemovedTest(unittest.TestCase):
    def test_overview_helpers_removed(self):
        source = TOOL_CATALOG_PATH.read_text(encoding="utf-8")
        self.assertNotIn("TOOL_OVERVIEW_HINTS", source)
        self.assertNotIn("looks_like_tool_overview", source)
        self.assertNotIn("format_tool_overview", source)


if __name__ == "__main__":
    unittest.main()
