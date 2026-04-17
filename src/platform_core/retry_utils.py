"""抖动指数回退工具。

用于给未来自建的 LLM / tool 调用层做重试时间计算；当前 AtriNexus 的 LangChain
内部重试不受影响。

设计参考 hermes-agent/agent/retry_utils.py：把抖动放进每次 backoff，避免多会话
并发时 thundering herd。
"""

from __future__ import annotations

import random
import threading
import time


_jitter_counter = 0
_jitter_lock = threading.Lock()


def jittered_backoff(
    attempt: int,
    *,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    jitter_ratio: float = 0.5,
) -> float:
    """计算一次抖动指数回退时间。

    Args:
        attempt: 1 起的重试次数。
        base_delay: 第一次重试的基础秒数。
        max_delay: 单次最长等待秒数上限。
        jitter_ratio: 抖动占比；0.5 意味着在 [0, 0.5 * delay] 区间均匀采样。

    Returns:
        本次应等待的秒数：min(base * 2^(attempt-1), max_delay) + jitter。

    抖动让并发重试不会集中在同一瞬间击打 provider；即使多个会话在几乎同一时刻
    拿到 429，它们也会散开重试。
    """
    global _jitter_counter
    with _jitter_lock:
        _jitter_counter += 1
        tick = _jitter_counter

    exponent = max(0, attempt - 1)
    if exponent >= 63 or base_delay <= 0:
        delay = max_delay
    else:
        delay = min(base_delay * (2 ** exponent), max_delay)

    seed = (time.time_ns() ^ (tick * 0x9E3779B9)) & 0xFFFFFFFF
    rng = random.Random(seed)
    jitter = rng.uniform(0, jitter_ratio * delay)

    return delay + jitter
