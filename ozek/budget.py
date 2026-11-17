"""Context / KV soft budget — refuse oversized prompts with a clear error.

This is a *machine* limit (RAM / KV growth), not the model architecture window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class ContextBudgetError(ValueError):
    """Raised when prompt (+ planned generation) exceeds ``max_context_tokens``."""

    def __init__(
        self,
        message: str,
        *,
        prompt_tokens: int,
        max_tokens: int,
        max_context_tokens: int,
    ) -> None:
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.max_context_tokens = max_context_tokens
        self.code = "context_budget"

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": str(self),
            "code": self.code,
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
            "max_context_tokens": self.max_context_tokens,
            "needed_tokens": self.prompt_tokens + self.max_tokens,
        }


@dataclass(frozen=True)
class ContextBudget:
    """Soft cap for prompt_tokens + max_tokens on this machine."""

    max_context_tokens: int

    def check(self, prompt_tokens: int, max_tokens: int) -> None:
        needed = int(prompt_tokens) + int(max_tokens)
        if needed <= self.max_context_tokens:
            return
        raise ContextBudgetError(
            (
                f"context budget exceeded: prompt_tokens={prompt_tokens} + "
                f"max_tokens={max_tokens} = {needed} > "
                f"max_context_tokens={self.max_context_tokens}. "
                "Shorten the prompt, lower max_tokens, or raise max_context_tokens "
                "(only if this Mac has enough unified memory — otherwise you get swap)."
            ),
            prompt_tokens=int(prompt_tokens),
            max_tokens=int(max_tokens),
            max_context_tokens=self.max_context_tokens,
        )


def count_tokens(tokenizer: Any, text: str) -> int:
    """Best-effort token count for a chat/prompt string."""
    if not text:
        return 0
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            ids = encode(text, add_special_tokens=False)
            return len(ids)
        except TypeError:
            ids = encode(text)
            return len(ids)
    # Fallback: rough char heuristic (Japanese-heavy ≈ 2 chars/token)
    return max(1, len(text) // 2)


def default_max_context_for_ram(ram_hint_gb: Optional[int] = None) -> int:
    """Conservative defaults for 8B-class 4-bit on Apple Silicon."""
    ram = int(ram_hint_gb or 16)
    if ram <= 16:
        return 4096
    if ram <= 36:
        return 16384
    if ram <= 64:
        return 32768
    return 65536
