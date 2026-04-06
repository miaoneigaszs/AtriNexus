from __future__ import annotations

import re


# 保持模式集中定义，避免主路由文件被正则淹没。
TOOL_OVERVIEW_PATTERN = re.compile(
    r"(有哪些工具|有什么工具|能用什么工具|可以用什么工具|能做什么|会什么|能力有哪些)"
)

PROFILE_OVERVIEW_PATTERN = re.compile(
    r"(当前能力档位|当前工具档位|当前模式|当前会话模式|我现在是什么模式|我现在是什么档位|现在是什么模式|现在是什么档位)"
)

READ_FILE_PATTERNS = (
    re.compile(
        r"(?:那你)?(?:帮我|麻烦|请)?(?:看看|看下|看一下|查看|读一下|读取|打开)\s*(?P<path>[A-Za-z0-9_./\\-]+)\s*(?:写的什么|写了什么|写了啥|写啥|内容是什么|里有什么|里有哪些)?(?:吧)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:读一下|读取|查看|看看|打开)\s*(?P<path>[^\s，。！？]+(?:\.[A-Za-z0-9_-]+|README(?:\.md)?))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<path>[^\s，。！？]+(?:\.[A-Za-z0-9_-]+|README(?:\.md)?))\s*(?:里写的什么|写了什么|内容是什么)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<path>[^\s，。！？]+(?:\.[A-Za-z0-9_-]+|README(?:\.md)?))\s*里(?:有什么|有哪些)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<path>[^\s，。！？]+)\s*(?:写的什么|写了什么|写了啥|写啥|内容是什么|里有什么|里有哪些)(?:吧)?",
        re.IGNORECASE,
    ),
)

READ_FILE_LINE_PATTERNS = (
    re.compile(
        r"(?P<path>[^\s，。！？]+(?:\.[A-Za-z0-9_-]+|README(?:\.md)?))\s*(?P<position>最后一行|末行|最后1行|第一行|首行|第1行)",
        re.IGNORECASE,
    ),
)

LIST_DIR_PATTERNS = (
    re.compile(r"(?:看看|查看|列出)\s*(?P<path>[^\s，。！？]+)\s*目录"),
    re.compile(r"(?P<path>[^\s，。！？]+)\s*目录(?:里|下)?(?:有什么|有哪些)?"),
    re.compile(r"(?P<path>[^\s，。！？]+)\s*里(?:写的什么|有什么|有哪些)"),
)

SEARCH_FILE_PATTERNS = (
    re.compile(
        r"在\s*(?P<path>[^\s，。！？]+)\s*(?:里|中|目录里|目录下|下面)?(?:搜索|查找)\s*(?P<query>[^\n，。！？]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:搜索|查找)\s*(?P<query>[^\s，。！？]+)\s*(?:内容|文本|关键词)?(?:在\s*(?P<path>[^\s，。！？]+))?",
        re.IGNORECASE,
    ),
)

REPLACE_PATTERNS = (
    re.compile(
        r"把\s*(?P<path>.+?)\s*(?:里的|里面的|中的|内容里的)\s*[\"“'‘](?P<find>[\s\S]+?)[\"”'’]\s*(?:改成|替换成|替换为)\s*[\"“'‘](?P<replace>[\s\S]+?)[\"”'’]",
        re.IGNORECASE,
    ),
)

REWRITE_PATTERNS = (
    re.compile(
        r"把\s*(?P<path>.+?)\s*(?:内容)?(?:改成|改为|写成|重写成|覆盖成)\s*[\"“'‘](?P<content>[\s\S]+?)[\"”'’]",
        re.IGNORECASE,
    ),
)

APPEND_PATTERNS = (
    re.compile(
        r"(?:在|给)\s*(?P<path>.+?)\s*(?P<position>末尾|结尾|开头|前面|后面)\s*(?:追加|加上|补上|添加)\s*[\"“'‘](?P<content>[\s\S]+?)[\"”'’]",
        re.IGNORECASE,
    ),
)

REWRITE_BLOCK_PATTERNS = (
    re.compile(
        r"把\s*(?P<path>.+?)\s*(?P<target>第一段|开头|首段|标题)\s*(?:改得|改成|改为)?\s*(?P<instruction>更清楚|更清晰|更简洁|更简短|更短|更正式|更专业|更口语化|更自然|更有条理|改短一点|改正式一点|改清楚一点)",
        re.IGNORECASE,
    ),
)

RENAME_PATTERNS = (
    re.compile(
        r"把\s*(?P<source>.+?)\s*(?:重命名|改名|改文件名)\s*(?:为|成)\s*(?P<target>[^\s，。！？]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"把\s*(?P<source>.+?)\s*(?:移动到|挪到)\s*(?P<target>[^\s，。！？]+)",
        re.IGNORECASE,
    ),
)

FOLLOWUP_RENAME_PATTERN = re.compile(
    r"^(?:改为|改成|重命名为|命名为)\s*(?P<target>[^\s，。！？]+)$",
    re.IGNORECASE,
)
