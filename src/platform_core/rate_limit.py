"""Provider 速率限制头解析与展示。

从 API 响应头（`x-ratelimit-*`）解析出请求/token 的分钟/小时配额，生成
ASCII 进度条用于 `/usage` 类命令。参考 hermes-agent/agent/rate_limit_tracker.py。

**头名约定**（OpenRouter、Nous Portal、多数 OpenAI 兼容代理共用）：

    x-ratelimit-limit-requests          RPM 上限
    x-ratelimit-limit-requests-1h       RPH 上限
    x-ratelimit-limit-tokens            TPM 上限
    x-ratelimit-limit-tokens-1h         TPH 上限
    x-ratelimit-remaining-{requests,tokens}[-1h]
    x-ratelimit-reset-{requests,tokens}[-1h]

OpenAI 原生与 DeepSeek 的头格式略有差异；当前实现按 OpenRouter 风格解析，
不匹配的头会被忽略（`has_data` 保持 False）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass
class RateLimitBucket:
    """单个配额窗口（如 requests-per-minute）。"""

    limit: int = 0
    remaining: int = 0
    reset_seconds: float = 0.0
    captured_at: float = 0.0

    @property
    def used(self) -> int:
        return max(0, self.limit - self.remaining)

    @property
    def usage_pct(self) -> float:
        if self.limit <= 0:
            return 0.0
        return (self.used / self.limit) * 100.0

    @property
    def remaining_seconds_now(self) -> float:
        """估算距离重置还剩多少秒，补偿了抓取到现在已过的时间。"""
        elapsed = time.time() - self.captured_at
        return max(0.0, self.reset_seconds - elapsed)


@dataclass
class RateLimitState:
    """一次响应头解析出的全量速率限制快照。"""

    requests_min: RateLimitBucket = field(default_factory=RateLimitBucket)
    requests_hour: RateLimitBucket = field(default_factory=RateLimitBucket)
    tokens_min: RateLimitBucket = field(default_factory=RateLimitBucket)
    tokens_hour: RateLimitBucket = field(default_factory=RateLimitBucket)
    captured_at: float = 0.0
    provider: str = ""

    @property
    def has_data(self) -> bool:
        return self.captured_at > 0

    @property
    def age_seconds(self) -> float:
        if not self.has_data:
            return float("inf")
        return time.time() - self.captured_at


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_rate_limit_headers(
    headers: Mapping[str, str],
    provider: str = "",
) -> Optional[RateLimitState]:
    """把响应头解析成 RateLimitState；没有任何 x-ratelimit-* 头时返回 None。"""
    lowered = {k.lower(): v for k, v in headers.items()}
    if not any(k.startswith("x-ratelimit-") for k in lowered):
        return None

    now = time.time()

    def _bucket(resource: str, suffix: str = "") -> RateLimitBucket:
        tag = f"{resource}{suffix}"
        return RateLimitBucket(
            limit=_safe_int(lowered.get(f"x-ratelimit-limit-{tag}")),
            remaining=_safe_int(lowered.get(f"x-ratelimit-remaining-{tag}")),
            reset_seconds=_safe_float(lowered.get(f"x-ratelimit-reset-{tag}")),
            captured_at=now,
        )

    return RateLimitState(
        requests_min=_bucket("requests"),
        requests_hour=_bucket("requests", "-1h"),
        tokens_min=_bucket("tokens"),
        tokens_hour=_bucket("tokens", "-1h"),
        captured_at=now,
        provider=provider,
    )


# ── 展示 ────────────────────────────────────────────────────────────────


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_seconds(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{m}m {sec}s" if sec else f"{m}m"
    h, remainder = divmod(s, 3600)
    m = remainder // 60
    return f"{h}h {m}m" if m else f"{h}h"


def _bar(pct: float, width: int = 20) -> str:
    filled = max(0, min(width, int(pct / 100.0 * width)))
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def _bucket_line(label: str, bucket: RateLimitBucket, label_width: int = 14) -> str:
    if bucket.limit <= 0:
        return f"  {label:<{label_width}}  (无数据)"

    pct = bucket.usage_pct
    used = _fmt_count(bucket.used)
    limit = _fmt_count(bucket.limit)
    remaining = _fmt_count(bucket.remaining)
    reset = _fmt_seconds(bucket.remaining_seconds_now)
    return (
        f"  {label:<{label_width}} {_bar(pct)} {pct:5.1f}%  "
        f"{used}/{limit} 已用  （剩 {remaining}，{reset} 后重置）"
    )


def format_rate_limit_display(state: RateLimitState) -> str:
    """给终端/聊天窗口看的多行展示。"""
    if not state.has_data:
        return "当前无速率限制数据——先发一次 API 请求再试。"

    age = state.age_seconds
    if age < 5:
        freshness = "刚刚"
    elif age < 60:
        freshness = f"{int(age)}s 前"
    else:
        freshness = f"{_fmt_seconds(age)} 前"

    provider_label = state.provider.title() if state.provider else "Provider"
    lines = [
        f"{provider_label} 速率限制（{freshness}捕获）：",
        "",
        _bucket_line("Requests/min", state.requests_min),
        _bucket_line("Requests/hr", state.requests_hour),
        "",
        _bucket_line("Tokens/min", state.tokens_min),
        _bucket_line("Tokens/hr", state.tokens_hour),
    ]

    warnings = []
    for label, bucket in [
        ("requests/min", state.requests_min),
        ("requests/hr", state.requests_hour),
        ("tokens/min", state.tokens_min),
        ("tokens/hr", state.tokens_hour),
    ]:
        if bucket.limit > 0 and bucket.usage_pct >= 80:
            reset = _fmt_seconds(bucket.remaining_seconds_now)
            warnings.append(f"  ⚠ {label} 已达 {bucket.usage_pct:.0f}%，{reset} 后重置")

    if warnings:
        lines.append("")
        lines.extend(warnings)

    return "\n".join(lines)


def format_rate_limit_compact(state: RateLimitState) -> str:
    """给状态栏用的一行紧凑版。"""
    if not state.has_data:
        return "无速率限制数据"

    parts = []
    if state.requests_min.limit > 0:
        parts.append(f"RPM: {state.requests_min.remaining}/{state.requests_min.limit}")
    if state.tokens_min.limit > 0:
        parts.append(
            f"TPM: {_fmt_count(state.tokens_min.remaining)}/{_fmt_count(state.tokens_min.limit)}"
        )
    if state.requests_hour.limit > 0:
        parts.append(
            f"RPH: {_fmt_count(state.requests_hour.remaining)}/{_fmt_count(state.requests_hour.limit)}"
        )
    if state.tokens_hour.limit > 0:
        parts.append(
            f"TPH: {_fmt_count(state.tokens_hour.remaining)}/{_fmt_count(state.tokens_hour.limit)}"
        )
    return " | ".join(parts) if parts else "无速率限制数据"


# ── 进程级最新状态 ───────────────────────────────────────────────────────

_latest_state: Optional[RateLimitState] = None


def record_latest_state(state: Optional[RateLimitState]) -> None:
    """middleware 在每次响应后调一次；provider 没提供头则传 None。"""
    global _latest_state
    if state is not None:
        _latest_state = state


def get_latest_state() -> Optional[RateLimitState]:
    """供 /usage 之类命令读取当前最新状态。"""
    return _latest_state
