"""Pinned Ozek HTTP/Python contract (v1).

Agents should depend on these field names and finish_reason values.
Bump CONTRACT_VERSION only on breaking renames / semantics changes.
"""

from __future__ import annotations

from typing import Any, Optional

CONTRACT_VERSION = "1"

# Ozek-native finish reasons (stats.finish_reason / ozek_stats).
FINISH_LENGTH = "length"
FINISH_STOP = "stop"
FINISH_CANCEL = "cancel"
FINISH_EOS = "eos"
FINISH_TIMEOUT = "timeout"
FINISH_FINAL = "final"

FINISH_REASONS = frozenset(
    {
        FINISH_LENGTH,
        FINISH_STOP,
        FINISH_CANCEL,
        FINISH_EOS,
        FINISH_TIMEOUT,
        FINISH_FINAL,
    }
)

# OpenAI chat.completion finish_reason subset we emit.
OPENAI_STOP = "stop"
OPENAI_LENGTH = "length"

# Documented request keys for POST /v1/generate (and shared chat knobs).
GENERATE_KEYS = frozenset(
    {
        "model",
        "prompt",
        "messages",
        "system",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "stop",
        "think",
        "final_only",
        "reasoning_effort",
        "stop_after_final",
        "max_context_tokens",
        "stream",
        "profile",
        "response_format",
        "schema",
        "preset",
        "timeout_s",
        "seed",
        "json_retries",
    }
)

# Documented stats keys (GenerateStats.as_dict).
STATS_KEYS = frozenset(
    {
        "model",
        "prompt_tokens",
        "generation_tokens",
        "ttft_s",
        "wall_s",
        "prompt_tps",
        "generation_tps",
        "decode_tps_wall",
        "peak_memory_gb",
        "enable_thinking",
        "finish_reason",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "stop",
        "seed",
        "timeout_s",
        "json_retries",
        "attempts",
        "reasoning_effort",
        "stop_after_final",
    }
)


def openai_finish_reason(ozek_reason: str) -> str:
    """Map Ozek finish_reason → OpenAI chat.completion finish_reason."""
    if ozek_reason in (FINISH_STOP, FINISH_EOS, FINISH_CANCEL, FINISH_FINAL):
        return OPENAI_STOP
    # length | timeout (and unknown) → length
    return OPENAI_LENGTH


def normalize_generate_kwargs(req: dict[str, Any]) -> dict[str, Any]:
    """Extract runtime.generate / stream_tokens kwargs from a JSON body.

    Unknown keys are ignored (forward-compatible). Validation errors raise ValueError.
    """
    kwargs: dict[str, Any] = {
        "max_tokens": _as_int(req.get("max_tokens"), 128, "max_tokens"),
        "temperature": _as_float(req.get("temperature"), 0.0, "temperature"),
        "top_p": _as_float(req.get("top_p"), 0.0, "top_p"),
        "top_k": _as_int(req.get("top_k"), 0, "top_k"),
        "min_p": _as_float(req.get("min_p"), 0.0, "min_p"),
        "stop": req.get("stop"),
        "enable_thinking": bool(req.get("think") or False),
        # Default UX: show final answer; pass final_only=false / raw for full Harmony.
        "final_only": True if req.get("final_only") is None else bool(req.get("final_only")),
        "stop_after_final": (
            True
            if req.get("stop_after_final") is None
            else bool(req.get("stop_after_final"))
        ),
    }
    if req.get("reasoning_effort") is not None:
        kwargs["reasoning_effort"] = str(req["reasoning_effort"])
    if req.get("max_context_tokens") is not None:
        kwargs["max_context_tokens"] = _as_int(
            req["max_context_tokens"], 0, "max_context_tokens"
        )
    if req.get("repetition_penalty") is not None:
        kwargs["repetition_penalty"] = _as_float(
            req["repetition_penalty"], None, "repetition_penalty"
        )
    if req.get("system") is not None:
        if not isinstance(req["system"], str):
            raise ValueError("system must be a string")
        kwargs["system"] = req["system"]

    rf = req.get("response_format")
    if rf is not None:
        if isinstance(rf, dict):
            # OpenAI-ish: {"type": "json_object"} | {"type": "json_schema", ...}
            rtype = str(rf.get("type") or "").lower()
            if rtype in ("json_object", "json", "json_schema"):
                kwargs["response_format"] = "json"
                if rf.get("json_schema") is not None:
                    js = rf["json_schema"]
                    if isinstance(js, dict) and "schema" in js:
                        kwargs["schema"] = js["schema"]
                    else:
                        kwargs["schema"] = js
                elif rf.get("schema") is not None:
                    kwargs["schema"] = rf["schema"]
            elif rtype in ("text", ""):
                kwargs["response_format"] = "text"
            else:
                raise ValueError("response_format.type must be text|json_object|json_schema")
        elif isinstance(rf, str):
            kwargs["response_format"] = rf.lower().strip()
        else:
            raise ValueError("response_format must be a string or object")

    if req.get("schema") is not None and "schema" not in kwargs:
        kwargs["schema"] = req["schema"]
    if req.get("timeout_s") is not None:
        kwargs["timeout_s"] = _as_float(req["timeout_s"], None, "timeout_s")
    if req.get("seed") is not None:
        kwargs["seed"] = _as_int(req["seed"], 0, "seed")
    if req.get("json_retries") is not None:
        kwargs["json_retries"] = _as_int(req["json_retries"], 0, "json_retries")
    return kwargs


def contract_meta() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "finish_reasons": sorted(FINISH_REASONS),
        "openai_finish_reasons": [OPENAI_STOP, OPENAI_LENGTH],
        "generate_keys": sorted(GENERATE_KEYS),
        "stats_keys": sorted(STATS_KEYS),
        "response_formats": ["text", "json"],
        "structured": {
            "mode": "prompt_nudge_plus_extract",
            "notes": "Not constrained decoding; check parsed / parse_error.",
        },
        "harmony": {
            "fields": ["thinking", "final", "raw"],
            "notes": (
                "LLM-jp / Harmony channels; final_only=true (default) shows answer; "
                "stop_after_final ends decode when final channel closes; "
                "reasoning_effort=low|medium|high|off."
            ),
        },
    }


def _as_int(value: Any, default: int, name: str) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be an int") from e


def _as_float(value: Any, default: Optional[float], name: str) -> float:
    if value is None or value == "":
        if default is None:
            raise ValueError(f"{name} required")
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be a number") from e
