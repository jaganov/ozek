"""Parse OpenAI Harmony-style channels (LLM-jp thinking models).

Typical completion::

    <|channel|>analysis<|message|>…reasoning…<|end|>
    <|start|>assistant<|channel|>final<|message|>…answer…

``analysis`` → ``thinking``; ``final`` → user-facing answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Channel name, optional attrs (e.g. commentary to=...), then message body until
# the next Harmony control token or end of string.
_CHANNEL_RE = re.compile(
    r"<\|channel\|>(?P<name>\w+)(?P<attrs>[^<]*?)<\|message\|>(?P<body>.*?)"
    r"(?=<\|end\|>|<\|return\|>|<\|call\|>|<\|channel\|>|<\|start\|>|$)",
    re.DOTALL,
)


@dataclass(frozen=True)
class HarmonyParts:
    thinking: Optional[str]
    final: Optional[str]
    commentary: tuple[str, ...]
    has_channels: bool

    @property
    def display(self) -> Optional[str]:
        """Preferred user-facing text when channels are present."""
        if self.final is not None:
            return self.final
        return None


def parse_harmony(text: str) -> HarmonyParts:
    """Split Harmony markup into thinking / final / commentary.

    If no ``<|channel|>`` markers are found, ``has_channels`` is False and
    ``thinking`` / ``final`` stay None (caller should use the raw text).
    """
    if not text or "<|channel|>" not in text:
        return HarmonyParts(
            thinking=None,
            final=None,
            commentary=(),
            has_channels=False,
        )

    thinking: Optional[str] = None
    final: Optional[str] = None
    commentary: list[str] = []

    for m in _CHANNEL_RE.finditer(text):
        name = m.group("name").lower()
        body = (m.group("body") or "").strip()
        if name == "analysis":
            # Keep the last analysis block if several appear.
            thinking = body
        elif name == "final":
            final = body
        elif name == "commentary":
            if body:
                commentary.append(body)

    return HarmonyParts(
        thinking=thinking,
        final=final,
        commentary=tuple(commentary),
        has_channels=True,
    )


_FINAL_OPEN = "<|channel|>final<|message|>"
_FINAL_CLOSE = ("<|return|>", "<|end|>", "<|call|>")


def final_channel_closed(text: str) -> tuple[bool, str]:
    """If a Harmony ``final`` channel has closed, return (True, truncated text).

    Truncation keeps the closing marker so parsers still see a complete message.
    Used to stop generation early instead of burning remaining ``max_tokens``.
    """
    if not text or _FINAL_OPEN not in text:
        return False, text
    start = text.rfind(_FINAL_OPEN)
    body_at = start + len(_FINAL_OPEN)
    body = text[body_at:]
    best: Optional[int] = None
    marker = ""
    for m in _FINAL_CLOSE:
        i = body.find(m)
        if i >= 0 and (best is None or i < best):
            best = i
            marker = m
    if best is None:
        return False, text
    cut = body_at + best + len(marker)
    return True, text[:cut]


def display_text(text: str, *, final_only: bool = False) -> str:
    """Return final channel when requested / available, else raw ``text``."""
    parts = parse_harmony(text)
    if final_only and parts.final is not None:
        return parts.final
    if final_only and parts.has_channels and parts.final is None and parts.thinking:
        # Truncated generation: only analysis so far — do not dump markers alone.
        return parts.thinking
    return text


def normalize_reasoning_effort(value: Optional[str]) -> Optional[str]:
    """Map CLI/API aliases → ``low``|``medium``|``high``|``off``."""
    if value is None:
        return None
    key = str(value).strip().lower()
    if key in ("", "none", "default"):
        return None
    if key in ("off", "direct", "final", "no-think", "nothink"):
        return "off"
    if key in ("low", "medium", "med", "high"):
        return "medium" if key == "med" else key
    raise ValueError(
        f"reasoning_effort must be low|medium|high|off, got {value!r}"
    )
