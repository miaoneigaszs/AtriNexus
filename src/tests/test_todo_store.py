"""PR18 — todo_store 行为测试。"""

from __future__ import annotations

import unittest

from src.agent_runtime.todo_store import (
    TodoItem,
    TodoStore,
    VALID_STATUSES,
)


class TodoItemTest(unittest.TestCase):
    def test_from_raw_happy_path(self):
        item = TodoItem.from_raw({"id": "1", "content": "写 README", "status": "pending"})
        self.assertEqual(item, TodoItem(id="1", content="写 README", status="pending"))

    def test_from_raw_rejects_missing_fields(self):
        self.assertIsNone(TodoItem.from_raw({}))
        self.assertIsNone(TodoItem.from_raw({"id": "", "content": "x", "status": "pending"}))
        self.assertIsNone(TodoItem.from_raw({"id": "1", "content": "", "status": "pending"}))
        self.assertIsNone(TodoItem.from_raw("not a dict"))

    def test_from_raw_defaults_bad_status_to_pending(self):
        item = TodoItem.from_raw({"id": "1", "content": "x", "status": "nonsense"})
        self.assertEqual(item.status, "pending")

    def test_from_raw_accepts_all_valid_statuses(self):
        for status in VALID_STATUSES:
            item = TodoItem.from_raw({"id": "1", "content": "x", "status": status})
            self.assertEqual(item.status, status)


class TodoStoreSetGetTest(unittest.TestCase):
    def setUp(self):
        self.store = TodoStore()

    def test_get_empty_returns_empty_list(self):
        self.assertEqual(self.store.get("u1"), [])

    def test_set_replaces_all_items(self):
        self.store.set("u1", [TodoItem("1", "a", "pending")])
        self.store.set("u1", [TodoItem("2", "b", "in_progress")])
        self.assertEqual(self.store.get("u1"), [TodoItem("2", "b", "in_progress")])

    def test_set_dedupes_by_id_keeping_last(self):
        result = self.store.set(
            "u1",
            [
                TodoItem("1", "first", "pending"),
                TodoItem("1", "second", "completed"),
            ],
        )
        self.assertEqual(result, [TodoItem("1", "second", "completed")])

    def test_users_are_isolated(self):
        self.store.set("u1", [TodoItem("1", "a", "pending")])
        self.store.set("u2", [TodoItem("1", "b", "completed")])
        self.assertEqual(self.store.get("u1")[0].content, "a")
        self.assertEqual(self.store.get("u2")[0].content, "b")

    def test_clear_removes_user_entries(self):
        self.store.set("u1", [TodoItem("1", "a", "pending")])
        self.store.clear("u1")
        self.assertEqual(self.store.get("u1"), [])


class TodoStoreMergeTest(unittest.TestCase):
    def setUp(self):
        self.store = TodoStore()
        self.store.set(
            "u1",
            [
                TodoItem("1", "first", "pending"),
                TodoItem("2", "second", "pending"),
            ],
        )

    def test_merge_updates_existing_by_id(self):
        self.store.merge("u1", [TodoItem("1", "first-updated", "completed")])
        self.assertEqual(
            self.store.get("u1"),
            [
                TodoItem("1", "first-updated", "completed"),
                TodoItem("2", "second", "pending"),
            ],
        )

    def test_merge_adds_new_items_at_end(self):
        self.store.merge("u1", [TodoItem("3", "third", "in_progress")])
        self.assertEqual(
            [item.id for item in self.store.get("u1")],
            ["1", "2", "3"],
        )

    def test_merge_preserves_untouched_items(self):
        self.store.merge("u1", [TodoItem("2", "second-done", "completed")])
        self.assertEqual(self.store.get("u1")[0].content, "first")
        self.assertEqual(self.store.get("u1")[1].status, "completed")


class TodoStoreRenderTest(unittest.TestCase):
    def setUp(self):
        self.store = TodoStore()

    def test_render_empty_returns_empty_string(self):
        self.assertEqual(self.store.render("u1"), "")

    def test_render_uses_chinese_labels(self):
        self.store.set(
            "u1",
            [
                TodoItem("1", "写 README", "pending"),
                TodoItem("2", "跑测试", "in_progress"),
                TodoItem("3", "推 PR", "completed"),
            ],
        )
        rendered = self.store.render("u1")
        self.assertIn("未开始", rendered)
        self.assertIn("进行中", rendered)
        self.assertIn("已完成", rendered)
        self.assertIn("(1)", rendered)
        self.assertIn("写 README", rendered)


if __name__ == "__main__":
    unittest.main()
