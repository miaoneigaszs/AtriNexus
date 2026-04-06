from __future__ import annotations

import difflib
import os
import re
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional


class WorkspacePathResolver:
    """处理 workspace 路径归一化、候选解析和用户确认流。"""

    def __init__(self, runtime: Any, session_service: Any) -> None:
        self.runtime = runtime
        self.session_service = session_service
        self._current_user_id = ""
        self._pending_resolution_reply: Optional[str] = None

    def begin(self, user_id: str) -> None:
        self._current_user_id = user_id
        self._pending_resolution_reply = None

    @property
    def current_user_id(self) -> str:
        return self._current_user_id

    def take_pending_reply(self) -> Optional[str]:
        reply = self._pending_resolution_reply
        self._pending_resolution_reply = None
        return reply

    def normalize_path_fragment(self, fragment: str) -> str:
        normalized = (fragment or "").strip()
        if not normalized:
            return ""

        normalized = normalized.strip("`'\"“”‘’")
        normalized = normalized.replace("下的", "/")
        normalized = normalized.replace("下面的", "/")
        normalized = normalized.replace("目录里的", "/")
        normalized = normalized.replace("目录下的", "/")
        normalized = normalized.replace("目录", "")
        normalized = normalized.replace("文件", "")
        normalized = normalized.replace("\\", "/")
        normalized = re.sub(r"\s+", "", normalized)
        normalized = re.sub(r"/{2,}", "/", normalized)
        return normalized.strip("/")

    def normalize_request_text(self, message: str) -> str:
        normalized = (message or "").strip()
        normalized = re.sub(r"^(那你|那就|那|你先|你就|你)\s*", "", normalized)
        normalized = re.sub(r"^(帮我|麻烦你|麻烦|请你|请)\s*", "", normalized)
        normalized = re.sub(r"^(看一下|看下|看看|查看|读一下|读取|打开)\s*", "", normalized)
        return normalized.strip()

    def resolve_existing_path_hint(
        self,
        path: str,
        *,
        expect_file: bool = False,
        expect_dir: bool = False,
        action: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not path:
            return path

        candidate, error = self.runtime._resolve_path_or_error(path)
        if not error and candidate and candidate.exists():
            if expect_file and candidate.is_file():
                return path
            if expect_dir and candidate.is_dir():
                return path
            if not expect_file and not expect_dir:
                return path

        lowered = path.lower()
        normalized_key = self._normalize_lookup_key(path)
        if lowered == "readme":
            return "README.md"

        last_target = self.session_service.get_last_workspace_target(self._current_user_id)
        last_path = str(last_target.get("path", "")).strip()
        last_type = str(last_target.get("type", "")).strip()
        if last_path and "/" not in path:
            base_dir = PurePosixPath(last_path) if last_type == "dir" else PurePosixPath(last_path).parent
            candidate_path = str(base_dir / path) if str(base_dir) != "." else path
            candidate, candidate_error = self.runtime._resolve_path_or_error(candidate_path)
            if not candidate_error and candidate and candidate.exists():
                if expect_file and candidate.is_file():
                    return candidate_path
                if expect_dir and candidate.is_dir():
                    return candidate_path
                if not expect_file and not expect_dir:
                    return candidate_path

        if "/" in path:
            self._maybe_store_path_resolution(
                path,
                expect_file=expect_file,
                expect_dir=expect_dir,
                action=action,
                payload=payload,
            )
            return None if self._pending_resolution_reply else path

        exact_file_matches: List[str] = []
        for file_path in self.runtime._iter_files(self.runtime.workspace_root):
            file_name = file_path.name.lower()
            if file_name != lowered and self._normalize_lookup_key(file_name) != normalized_key:
                continue
            exact_file_matches.append(self.runtime._to_relative(file_path))
            if len(exact_file_matches) > 1:
                break
        if len(exact_file_matches) == 1:
            return exact_file_matches[0]

        if expect_dir:
            exact_dir_matches: List[str] = []
            for current_root, dirnames, _ in os.walk(self.runtime.workspace_root):
                dirnames[:] = [name for name in dirnames if name not in self.runtime.SKIP_DIRS]
                for dirname in dirnames:
                    if dirname.lower() != lowered:
                        continue
                    dir_path = PurePosixPath(os.path.relpath(os.path.join(current_root, dirname), self.runtime.workspace_root))
                    exact_dir_matches.append(str(dir_path))
                    if len(exact_dir_matches) > 1:
                        break
                if len(exact_dir_matches) > 1:
                    break
            if len(exact_dir_matches) == 1:
                return exact_dir_matches[0]

        self._maybe_store_path_resolution(
            path,
            expect_file=expect_file,
            expect_dir=expect_dir,
            action=action,
            payload=payload,
        )
        return None if self._pending_resolution_reply else path

    def looks_like_existing_file(self, path: str) -> bool:
        candidate, error = self.runtime._resolve_path_or_error(path)
        return not error and bool(candidate and candidate.exists() and candidate.is_file())

    def looks_like_existing_dir(self, path: str) -> bool:
        candidate, error = self.runtime._resolve_path_or_error(path)
        return not error and bool(candidate and candidate.exists() and candidate.is_dir())

    def parse_resolution_choice(self, message: str) -> Optional[int]:
        normalized = (message or "").strip().lower()
        if normalized in {"是", "对", "嗯", "嗯嗯", "好", "好的", "确认", "同意", "就这个", "就是这个"}:
            return 1
        if normalized in {"不是", "不对", "取消", "都不是", "算了"}:
            return 0

        match = re.fullmatch(r"(?:第)?([1-9])(?:个)?", normalized)
        if match:
            return int(match.group(1))
        return None

    def _maybe_store_path_resolution(
        self,
        original_input: str,
        *,
        expect_file: bool,
        expect_dir: bool,
        action: str,
        payload: Optional[Dict[str, Any]],
    ) -> None:
        candidates = self._find_path_candidates(
            original_input,
            expect_file=expect_file,
            expect_dir=expect_dir,
        )
        if not candidates:
            return

        self.session_service.set_pending_workspace_resolution(
            self._current_user_id,
            action=action,
            original_input=original_input,
            candidates=candidates,
            payload=payload or {},
        )
        self._pending_resolution_reply = self._format_resolution_prompt(
            original_input=original_input,
            candidates=candidates,
            expect_dir=expect_dir,
        )

    def _find_path_candidates(
        self,
        original_input: str,
        *,
        expect_file: bool,
        expect_dir: bool,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        query = self.normalize_path_fragment(original_input)
        query_name = PurePosixPath(query).name
        query_key = self._normalize_lookup_key(query_name)
        if not query_key:
            return []

        candidates: List[Dict[str, Any]] = []

        if not expect_dir:
            for file_path in self.runtime._iter_files(self.runtime.workspace_root):
                relative_path = self.runtime._to_relative(file_path)
                score = self._score_path_candidate(query_key, file_path.name, relative_path)
                if score < 0.62:
                    continue
                candidates.append({"path": relative_path, "type": "file", "score": score})

        if expect_dir or not expect_file:
            for current_root, dirnames, _ in os.walk(self.runtime.workspace_root):
                dirnames[:] = [name for name in dirnames if name not in self.runtime.SKIP_DIRS]
                for dirname in dirnames:
                    relative_path = str(
                        PurePosixPath(
                            os.path.relpath(
                                os.path.join(current_root, dirname),
                                self.runtime.workspace_root,
                            )
                        )
                    )
                    score = self._score_path_candidate(query_key, dirname, relative_path)
                    if score < 0.62:
                        continue
                    candidates.append({"path": relative_path, "type": "dir", "score": score})

        unique_by_path: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            existing = unique_by_path.get(candidate["path"])
            if existing is None or candidate["score"] > existing["score"]:
                unique_by_path[candidate["path"]] = candidate

        ordered = sorted(
            unique_by_path.values(),
            key=lambda item: (-float(item["score"]), len(str(item["path"]))),
        )
        return ordered[:limit]

    def _score_path_candidate(self, query_key: str, name: str, relative_path: str) -> float:
        name_key = self._normalize_lookup_key(name)
        path_key = self._normalize_lookup_key(relative_path)
        score = max(
            difflib.SequenceMatcher(None, query_key, name_key).ratio(),
            difflib.SequenceMatcher(None, query_key, path_key).ratio(),
        )
        if query_key == name_key:
            return 1.0
        if name_key.startswith(query_key):
            score = max(score, 0.9)
        if query_key in name_key:
            score = max(score, 0.85)
        return score

    def _format_resolution_prompt(
        self,
        *,
        original_input: str,
        candidates: List[Dict[str, Any]],
        expect_dir: bool,
    ) -> str:
        target_label = "目录" if expect_dir else "文件"
        lines = [f"没找到精确匹配的{target_label}：{original_input}", "", "你是不是指："]
        for index, candidate in enumerate(candidates, start=1):
            suffix = " [目录]" if candidate.get("type") == "dir" else ""
            lines.append(f"{index}. {candidate.get('path', '')}{suffix}")
        lines.append("")
        lines.append("回复“是”默认采用第 1 个，也可以回复序号选择，回复“不是”取消。")
        return "\n".join(lines)

    def _normalize_lookup_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", value.lower())
