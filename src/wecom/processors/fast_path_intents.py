from __future__ import annotations

from typing import List, Optional, Tuple


TOOL_OVERVIEW_HINTS = (
    "有哪些工具",
    "有什么工具",
    "能用什么工具",
    "可以用什么工具",
    "能做什么",
    "会什么",
    "能力有哪些",
)

PROFILE_OVERVIEW_HINTS = (
    "当前能力档位",
    "当前工具档位",
    "当前模式",
    "当前会话模式",
    "我现在是什么模式",
    "我现在是什么档位",
    "现在是什么模式",
    "现在是什么档位",
)

READ_FILE_VERBS = ("看看", "看下", "看一下", "查看", "读一下", "读取", "打开", "读")
READ_CONTENT_HINTS = ("写的什么", "写了什么", "写了啥", "写啥", "内容是什么", "里有什么", "里有哪些")
DIRECTORY_HINTS = ("目录", "目录里", "目录下", "下面有什么", "下面有哪些")
SEARCH_VERBS = ("搜索", "查找")
RENAME_VERBS = ("重命名", "改名", "改文件名", "移动到", "挪到")
FOLLOWUP_RENAME_PREFIXES = ("改为", "改成", "重命名为", "命名为")
BLOCK_TARGETS = ("第一段", "开头", "首段", "标题")
BLOCK_INSTRUCTIONS = (
    "更清楚",
    "更清晰",
    "更简洁",
    "更简短",
    "更短",
    "更正式",
    "更专业",
    "更口语化",
    "更自然",
    "更有条理",
    "改短一点",
    "改正式一点",
    "改清楚一点",
)
LINE_POSITIONS = ("最后一行", "末行", "最后1行", "第一行", "首行", "第1行")
START_END_HINTS = {"开头": "start", "前面": "start", "末尾": "end", "结尾": "end", "后面": "end"}
PATH_STOP_WORDS = (
    "写的什么",
    "写了什么",
    "写了啥",
    "写啥",
    "内容是什么",
    "里有什么",
    "里有哪些",
    "目录",
    "目录里",
    "目录下",
    "开头",
    "末尾",
    "结尾",
    "前面",
    "后面",
)

PUNCTUATION = " \t\r\n,，。！？!?:：；;（）()[]{}<>《》"
QUOTE_PAIRS = {"“": "”", '"': '"', "'": "'", "‘": "’"}


def is_tool_overview(message: str) -> bool:
    return any(hint in (message or "") for hint in TOOL_OVERVIEW_HINTS)


def is_profile_overview(message: str) -> bool:
    return any(hint in (message or "") for hint in PROFILE_OVERVIEW_HINTS)


def extract_read_file_path(message: str) -> Optional[str]:
    path = _extract_path_after_prefix(message, READ_FILE_VERBS)
    if path:
        return path
    if any(hint in message for hint in READ_CONTENT_HINTS):
        for hint in READ_CONTENT_HINTS:
            if hint not in message:
                continue
            left = message.split(hint, 1)[0]
            path = _extract_last_path_like_token(left)
            if path:
                return path
        return _extract_first_path_like_token(message)
    return None


def extract_directory_path(message: str) -> Optional[str]:
    for hint in DIRECTORY_HINTS:
        if hint in message:
            left = message.split(hint, 1)[0]
            path = _extract_last_path_like_token(left)
            if path:
                return path
    if any(prefix in message for prefix in ("看看", "查看", "列出")) and "目录" in message:
        left = message.split("目录", 1)[0]
        return _extract_last_path_like_token(left)
    return None


def extract_search_request(message: str) -> Optional[Tuple[str, str]]:
    for verb in SEARCH_VERBS:
        if verb not in message:
            continue

        if "在" in message and message.index("在") < message.index(verb):
            before, after = message.split(verb, 1)
            path = _extract_last_path_like_token(before)
            query = _clean_fragment(after)
            if query:
                return query, path or "."

        after = message.split(verb, 1)[1]
        query = _clean_fragment(after)
        if query:
            return query, "."
    return None


def extract_read_file_line_request(message: str) -> Optional[Tuple[str, str]]:
    for raw_position in LINE_POSITIONS:
        if raw_position not in message:
            continue
        left = message.split(raw_position, 1)[0]
        path = _extract_last_path_like_token(left)
        if not path:
            path = _extract_first_path_like_token(message)
        if not path:
            return None
        position = "first" if raw_position in {"第一行", "首行", "第1行"} else "last"
        return path, position
    return None


def extract_replace_request(message: str) -> Optional[Tuple[str, str, str]]:
    quoted = extract_quoted_strings(message)
    if len(quoted) < 2:
        return None
    marker = _first_present(message, ("改成", "替换成", "替换为"))
    if not marker or "把" not in message:
        return None
    path = _extract_path_between(message, "把", ("里的", "里面的", "中的", "内容里的"))
    if not path:
        return None
    return path, quoted[0], quoted[1]


def extract_rewrite_request(message: str) -> Optional[Tuple[str, str]]:
    quoted = extract_quoted_strings(message)
    if not quoted or "把" not in message:
        return None
    marker = _first_present(message, ("改成", "改为", "写成", "重写成", "覆盖成"))
    if not marker:
        return None
    path = _extract_path_between(message, "把", ("内容改成", "内容改为", "内容写成", "内容重写成", "内容覆盖成", "改成", "改为", "写成", "重写成", "覆盖成"))
    if not path:
        return None
    return path, quoted[0]


def extract_block_rewrite_request(message: str) -> Optional[Tuple[str, str, str]]:
    if "把" not in message:
        return None
    target = _first_present(message, BLOCK_TARGETS)
    instruction = _first_present(message, BLOCK_INSTRUCTIONS)
    if not target or not instruction:
        return None
    path = _extract_path_between(message, "把", BLOCK_TARGETS)
    if not path:
        return None
    return path, target, instruction


def extract_append_request(message: str) -> Optional[Tuple[str, str, str]]:
    quoted = extract_quoted_strings(message)
    if not quoted:
        return None
    position_hint = _first_present(message, tuple(START_END_HINTS))
    if not position_hint:
        return None
    action_hint = _first_present(message, ("追加", "加上", "补上", "添加"))
    if not action_hint:
        return None
    path = _extract_path_between(message, ("在", "给"), tuple(START_END_HINTS))
    if not path:
        return None
    return path, quoted[0], START_END_HINTS[position_hint]


def extract_rename_paths(message: str) -> Optional[Tuple[str, str]]:
    if "把" not in message:
        return None
    verb = _first_present(message, RENAME_VERBS)
    if not verb:
        return None
    path = _extract_path_between(message, "把", RENAME_VERBS)
    if not path:
        return None
    right = message.split(verb, 1)[1]
    target = _clean_fragment(right.removeprefix("为").removeprefix("成"))
    target = _extract_first_path_like_token(target) or target
    if not target:
        return None
    return path, target


def extract_followup_rename_target(message: str) -> Optional[str]:
    normalized = (message or "").strip()
    for prefix in FOLLOWUP_RENAME_PREFIXES:
        if normalized.startswith(prefix):
            target = _clean_fragment(normalized[len(prefix) :])
            return _extract_first_path_like_token(target) or target or None
    return None


def extract_quoted_strings(message: str) -> List[str]:
    items: List[str] = []
    text = message or ""
    index = 0
    while index < len(text):
        start = text[index]
        end = QUOTE_PAIRS.get(start)
        if not end:
            index += 1
            continue
        close = text.find(end, index + 1)
        if close <= index + 1:
            index += 1
            continue
        content = text[index + 1 : close].strip()
        if content:
            items.append(content)
        index = close + 1
    return items


def _extract_path_after_prefix(message: str, prefixes: Tuple[str, ...]) -> Optional[str]:
    for prefix in prefixes:
        if prefix not in message:
            continue
        after = message.split(prefix, 1)[1]
        path = _extract_first_path_like_token(after)
        if path:
            return path
    return None


def _extract_path_between(message: str, left_markers, right_markers) -> Optional[str]:
    left_candidates = left_markers if isinstance(left_markers, tuple) else (left_markers,)
    right_candidates = right_markers if isinstance(right_markers, tuple) else (right_markers,)
    start_index = -1
    for marker in left_candidates:
        idx = message.find(marker)
        if idx >= 0:
            start_index = idx + len(marker)
            break
    if start_index < 0:
        return None
    segment = message[start_index:]
    end_index = len(segment)
    for marker in right_candidates:
        idx = segment.find(marker)
        if idx >= 0:
            end_index = min(end_index, idx)
    return _extract_last_path_like_token(segment[:end_index])


def _extract_first_path_like_token(text: str) -> Optional[str]:
    for token in _iter_tokens(text):
        if _looks_like_path_token(token):
            return token
    return None


def _extract_last_path_like_token(text: str) -> Optional[str]:
    tokens = [token for token in _iter_tokens(text) if _looks_like_path_token(token)]
    return tokens[-1] if tokens else None


def _iter_tokens(text: str):
    current = []
    for ch in text or "":
        if ch in PUNCTUATION:
            if current:
                yield "".join(current)
                current = []
            continue
        current.append(ch)
    if current:
        yield "".join(current)


def _looks_like_path_token(token: str) -> bool:
    normalized = _clean_fragment(token)
    if not normalized:
        return False
    if normalized in {"README", "README.md", "readme", "readme.md"}:
        return True
    if any(stop in normalized for stop in PATH_STOP_WORDS):
        return False
    if "/" in normalized or "\\" in normalized or "." in normalized:
        return True
    letters = sum(1 for ch in normalized if ch.isalpha())
    digits = sum(1 for ch in normalized if ch.isdigit())
    return (letters + digits) >= 3


def _clean_fragment(text: str) -> str:
    value = (text or "").strip().strip("`'\"“”‘’")
    while value and value[0] in PUNCTUATION:
        value = value[1:]
    while value and value[-1] in PUNCTUATION:
        value = value[:-1]
    return value.strip()


def _first_present(message: str, candidates: Tuple[str, ...]) -> Optional[str]:
    for item in candidates:
        if item in message:
            return item
    return None
