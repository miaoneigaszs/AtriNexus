"""Provider 与模型能力 registry。

不依赖外部 catalog（如 models.dev）；保持轻量、本地可控。新增 model 时直接
在这里登记一行；未登记的 model 走默认条目（OpenAI 兼容 + 32k 上下文）。

字段含义见 `ModelCapabilities`。`get_capabilities("gpt-4o")` 这种点查是主接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# ── 能力描述 ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelCapabilities:
    """单个 model 的运行时元信息。"""

    model_id: str
    """完整 model 名（如 `deepseek-ai/DeepSeek-V3`）。"""

    provider: str
    """provider 标识：openai / anthropic / deepseek / openai_compat / ..."""

    context_length: int
    """模型声明的上下文窗口（token 数）。"""

    supports_tools: bool = True
    """是否支持 OpenAI 风格的 tool calling。reasoning-only 模型常为 False。"""

    supports_cache_control: bool = False
    """是否支持 Anthropic 风格的 prompt caching breakpoint。"""

    is_reasoning: bool = False
    """是否为 reasoning 模型（DeepSeek-R1、o1 等），通常不能并发工具。"""


# ── 默认条目（未登记 model 的兜底）────────────────────────────────────────


DEFAULT_CAPABILITIES = ModelCapabilities(
    model_id="<unknown>",
    provider="openai_compat",
    context_length=32_000,
    supports_tools=True,
    supports_cache_control=False,
    is_reasoning=False,
)


# ── 注册表 ───────────────────────────────────────────────────────────────


_REGISTRY: Dict[str, ModelCapabilities] = {}


def register(cap: ModelCapabilities) -> None:
    _REGISTRY[cap.model_id.lower()] = cap


def get_capabilities(model: str) -> ModelCapabilities:
    """点查；未命中走前缀模糊匹配；再不命中返回默认条目。"""
    if not model:
        return DEFAULT_CAPABILITIES
    key = model.lower()
    cap = _REGISTRY.get(key)
    if cap is not None:
        return cap
    # 前缀匹配（如 `deepseek-ai/DeepSeek-V3.2-something` 命中 `deepseek-ai/deepseek-v3.2`）
    for registered_key, registered_cap in _REGISTRY.items():
        if key.startswith(registered_key):
            return registered_cap
    return DEFAULT_CAPABILITIES


def all_models() -> Dict[str, ModelCapabilities]:
    """供 /models 命令或诊断时枚举全表。"""
    return dict(_REGISTRY)


# ── 内置登记 ─────────────────────────────────────────────────────────────


# DeepSeek（OpenAI 兼容）
register(ModelCapabilities(
    model_id="deepseek-ai/deepseek-v3",
    provider="openai_compat",
    context_length=64_000,
    supports_tools=True,
))
register(ModelCapabilities(
    model_id="deepseek-ai/deepseek-v3.2",
    provider="openai_compat",
    context_length=128_000,
    supports_tools=True,
))
register(ModelCapabilities(
    model_id="pro/deepseek-ai/deepseek-v3.2",
    provider="openai_compat",
    context_length=128_000,
    supports_tools=True,
))
register(ModelCapabilities(
    model_id="deepseek-ai/deepseek-r1",
    provider="openai_compat",
    context_length=64_000,
    supports_tools=False,  # reasoning 模型不支持 tool calling
    is_reasoning=True,
))
register(ModelCapabilities(
    model_id="deepseek-reasoner",
    provider="openai_compat",
    context_length=64_000,
    supports_tools=False,
    is_reasoning=True,
))

# OpenAI
register(ModelCapabilities(
    model_id="gpt-4o",
    provider="openai_compat",
    context_length=128_000,
))
register(ModelCapabilities(
    model_id="gpt-4o-mini",
    provider="openai_compat",
    context_length=128_000,
))
register(ModelCapabilities(
    model_id="gpt-4-turbo",
    provider="openai_compat",
    context_length=128_000,
))

# Anthropic（PR11 不实现 native provider，但能力先登记好；PR13 接 native 时复用）
register(ModelCapabilities(
    model_id="claude-3-5-sonnet",
    provider="anthropic",
    context_length=200_000,
    supports_cache_control=True,
))
register(ModelCapabilities(
    model_id="claude-3-5-haiku",
    provider="anthropic",
    context_length=200_000,
    supports_cache_control=True,
))
register(ModelCapabilities(
    model_id="claude-3-opus",
    provider="anthropic",
    context_length=200_000,
    supports_cache_control=True,
))


__all__ = [
    "ModelCapabilities",
    "DEFAULT_CAPABILITIES",
    "register",
    "get_capabilities",
    "all_models",
]
