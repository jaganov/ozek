"""Light structured-output helpers (prompt nudge + JSON extract).

Not constrained decoding — the model can still emit invalid JSON.
Callers should check ``GenerateResult.parsed`` / ``parse_error``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Union

SchemaInput = Union[dict[str, Any], list[Any], str, None]

_JSON_NUDGE = (
    "Respond with valid JSON only. "
    "No markdown fences, no commentary before or after the JSON."
)

_FENCE_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)\s*```",
    re.IGNORECASE,
)


def schema_to_text(schema: SchemaInput) -> Optional[str]:
    if schema is None:
        return None
    if isinstance(schema, str):
        return schema.strip() or None
    return json.dumps(schema, ensure_ascii=False, indent=2)


def json_system_nudge(schema: SchemaInput = None) -> str:
    parts = [_JSON_NUDGE]
    text = schema_to_text(schema)
    if text:
        parts.append("Match this JSON shape / schema:")
        parts.append(text)
    return "\n".join(parts)


def merge_system(system: Optional[str], extra: str) -> str:
    system = (system or "").strip()
    extra = extra.strip()
    if not system:
        return extra
    if not extra:
        return system
    return f"{system}\n\n{extra}"


def apply_json_mode(
    *,
    system: Optional[str],
    messages: Optional[list[dict[str, str]]],
    schema: SchemaInput = None,
) -> tuple[Optional[str], Optional[list[dict[str, str]]]]:
    """Inject JSON-only instruction into system / messages (copy, no mutate)."""
    nudge = json_system_nudge(schema)
    if messages is None:
        return merge_system(system, nudge), None

    out = [dict(m) for m in messages]
    for i, m in enumerate(out):
        if m.get("role") == "system":
            out[i] = {
                **m,
                "content": merge_system(str(m.get("content") or ""), nudge),
            }
            return system, out
    out.insert(0, {"role": "system", "content": nudge})
    return system, out


def extract_json(text: str) -> Any:
    """Parse the first JSON value from model text. Raises ValueError on failure."""
    if text is None:
        raise ValueError("empty text")
    s = text.strip()
    if not s:
        raise ValueError("empty text")

    fence = _FENCE_RE.search(s)
    if fence:
        s = fence.group(1).strip()

    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch not in "{[":
            continue
        try:
            obj, _end = decoder.raw_decode(s[i:])
            return obj
        except json.JSONDecodeError:
            continue

    # whole string as JSON (e.g. "hello" / 42 / true)
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"no JSON object found: {e}") from e


def try_extract_json(text: str) -> tuple[Any, Optional[str]]:
    """Return (parsed, None) or (None, error message)."""
    try:
        return extract_json(text), None
    except ValueError as e:
        return None, str(e)
