from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PathResolutionResult:
    status: str
    path: str = ""
    target_type: str = ""
    reply: str = ""


class WorkspacePathResolver:
    """处理 workspace 路径归一化、候选解析和用户确认流。"""

    AUTO_REPAIR_SCORE = 0.86
    AUTO_REPAIR_TIE_GAP = 0.03
    DISAMBIGUATION_SCORE = 0.62

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
        normalized = "".join(ch for ch in normalized if not ch.isspace())
        while "//" in normalized:
            normalized = normalized.replace("//", "/")
        return normalized.strip("/")

    def normalize_request_text(self, message: str) -> str:
        normalized = (message or "").strip()
        for prefix in (
            "那你",
            "那就",
            "那",
            "你先",
            "你就",
            "你",
            "帮我",
            "麻烦你",
            "麻烦",
            "请你",
            "请",
            "看一下",
            "看下",
            "看看",
            "查看",
            "读一下",
            "读取",
            "打开",
        ):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].lstrip()
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

        resolved = self.resolve_path_hint(
            path,
            expect_file=expect_file,
            expect_dir=expect_dir,
            action=action,
            payload=payload,
        )
        if resolved.status == "resolved":
            return resolved.path
        if resolved.status == "ambiguous":
            return None
        return path if not resolved.reply else None

    def resolve_path_hint(
        self,
        path: str,
        *,
        expect_file: bool = False,
        expect_dir: bool = False,
        action: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> PathResolutionResult:
        normalized = self.normalize_path_fragment(path)
        if not normalized:
            return PathResolutionResult(status="missing")

        direct = self._resolve_existing_candidate(
            normalized,
            expect_file=expect_file,
            expect_dir=expect_dir,
        )
        if direct:
            return PathResolutionResult(
                status="resolved",
                path=direct,
                target_type="dir" if self.looks_like_existing_dir(direct) else "file",
            )

        repaired = self.repair_path_if_confident(
            normalized,
            expect_dir=expect_dir,
            allow_dir=expect_dir or not expect_file,
        )
        if repaired != normalized:
            return PathResolutionResult(
                status="resolved",
                path=repaired,
                target_type="dir" if self.looks_like_existing_dir(repaired) else "file",
            )

        candidates = self.find_path_candidates(
            normalized,
            expect_file=expect_file,
            expect_dir=expect_dir,
        )
        if candidates:
            self.session_service.set_pending_workspace_resolution(
                self._current_user_id,
                action=action,
                original_input=normalized,
                candidates=candidates,
                payload=payload or {},
            )
            self._pending_resolution_reply = self._format_resolution_prompt(
                original_input=normalized,
                candidates=candidates,
                expect_dir=expect_dir,
            )
            return PathResolutionResult(status="ambiguous", reply=self._pending_resolution_reply)

        return PathResolutionResult(status="missing")

    def repair_path_if_confident(
        self,
        value: str,
        *,
        expect_dir: bool,
        allow_dir: bool,
    ) -> str:
        normalized = self.normalize_path_fragment(value)
        if not normalized:
            return value
        if normalized.lower() == "readme":
            return "README.md"

        direct = self._resolve_existing_candidate(
            normalized,
            expect_file=not expect_dir,
            expect_dir=expect_dir,
            allow_dir=allow_dir,
        )
        if direct:
            return direct

        candidates = self.find_path_candidates(
            normalized,
            expect_file=not expect_dir,
            expect_dir=expect_dir,
            min_score=self.AUTO_REPAIR_SCORE,
        )
        if not candidates:
            return normalized

        best = candidates[0]
        if len(candidates) > 1 and abs(float(best["score"]) - float(candidates[1]["score"])) < self.AUTO_REPAIR_TIE_GAP:
            return normalized
        return str(best.get("path", normalized))

    def find_path_candidates(
        self,
        original_input: str,
        *,
        expect_file: bool,
        expect_dir: bool,
        min_score: float = DISAMBIGUATION_SCORE,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        query = self.normalize_path_fragment(original_input)
        query_name = PurePosixPath(query).name
        query_key = self.normalize_lookup_key(query_name)
        if not query_key:
            return []

        candidates: List[Dict[str, Any]] = []

        if not expect_dir:
            for file_path in self.runtime.iter_files(self.runtime.workspace_root):
                relative_path = self.runtime.to_relative(file_path)
                score = self.score_path_candidate(query_key, file_path.name, relative_path)
                if score < min_score:
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
                    score = self.score_path_candidate(query_key, dirname, relative_path)
                    if score < min_score:
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

    def score_path_candidate(self, query_key: str, name: str, relative_path: str) -> float:
        name_key = self.normalize_lookup_key(name)
        path_key = self.normalize_lookup_key(relative_path)
        score = max(
            difflib.SequenceMatcher(None, query_key, name_key).ratio(),
            difflib.SequenceMatcher(None, query_key, path_key).ratio(),
        )
        if query_key == name_key:
            return 1.0
        if name_key.startswith(query_key):
            return max(score, 0.9)
        if query_key in name_key:
            return max(score, 0.85)
        return score

    def normalize_lookup_key(self, value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")

    def looks_like_existing_file(self, path: str) -> bool:
        candidate, error = self.runtime.resolve_path_or_error(path)
        return not error and bool(candidate and candidate.exists() and candidate.is_file())

    def looks_like_existing_dir(self, path: str) -> bool:
        candidate, error = self.runtime.resolve_path_or_error(path)
        return not error and bool(candidate and candidate.exists() and candidate.is_dir())

    def parse_resolution_choice(self, message: str) -> Optional[int]:
        normalized = (message or "").strip().lower()
        if normalized in {"是", "对", "嗯", "嗯嗯", "好", "好的", "确认", "同意", "就这个", "就是这个"}:
            return 1
        if normalized in {"不是", "不对", "取消", "都不是", "算了"}:
            return 0

        for candidate in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
            if normalized in {candidate, f"第{candidate}", f"{candidate}个", f"第{candidate}个"}:
                return int(candidate)
        return None

    def _resolve_existing_candidate(
        self,
        path: str,
        *,
        expect_file: bool,
        expect_dir: bool,
        allow_dir: bool = False,
    ) -> Optional[str]:
        candidate, error = self.runtime.resolve_path_or_error(path)
        if not error and candidate and candidate.exists():
            if expect_file and candidate.is_file():
                return path
            if expect_dir and candidate.is_dir():
                return path
            if not expect_file and not expect_dir:
                return path
            if allow_dir and candidate.is_dir():
                return path

        lowered = path.lower()
        normalized_key = self.normalize_lookup_key(path)
        if lowered == "readme":
            return "README.md"

        last_target = self.session_service.get_last_workspace_target(self._current_user_id)
        last_path = str(last_target.get("path", "")).strip()
        last_type = str(last_target.get("type", "")).strip()
        if last_path and "/" not in path:
            base_dir = PurePosixPath(last_path) if last_type == "dir" else PurePosixPath(last_path).parent
            candidate_path = str(base_dir / path) if str(base_dir) != "." else path
            candidate, candidate_error = self.runtime.resolve_path_or_error(candidate_path)
            if not candidate_error and candidate and candidate.exists():
                if expect_file and candidate.is_file():
                    return candidate_path
                if expect_dir and candidate.is_dir():
                    return candidate_path
                if not expect_file and not expect_dir:
                    return candidate_path
                if allow_dir and candidate.is_dir():
                    return candidate_path

        exact_file_matches: List[str] = []
        if not expect_dir:
            for file_path in self.runtime.iter_files(self.runtime.workspace_root):
                file_name = file_path.name.lower()
                if file_name != lowered and self.normalize_lookup_key(file_name) != normalized_key:
                    continue
                exact_file_matches.append(self.runtime.to_relative(file_path))
                if len(exact_file_matches) > 1:
                    break
            if len(exact_file_matches) == 1:
                return exact_file_matches[0]

        if expect_dir or allow_dir:
            exact_dir_matches: List[str] = []
            for current_root, dirnames, _ in os.walk(self.runtime.workspace_root):
                dirnames[:] = [name for name in dirnames if name not in self.runtime.SKIP_DIRS]
                for dirname in dirnames:
                    if dirname.lower() != lowered and self.normalize_lookup_key(dirname) != normalized_key:
                        continue
                    dir_path = PurePosixPath(
                        os.path.relpath(
                            os.path.join(current_root, dirname),
                            self.runtime.workspace_root,
                        )
                    )
                    exact_dir_matches.append(str(dir_path))
                    if len(exact_dir_matches) > 1:
                        break
                if len(exact_dir_matches) > 1:
                    break
            if len(exact_dir_matches) == 1:
                return exact_dir_matches[0]

        return None

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
