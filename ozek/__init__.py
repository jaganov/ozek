"""Ozek — thin Mac-first MLX runtime for local agents."""

from .budget import ContextBudgetError
from .config import load_config
from .contract import CONTRACT_VERSION, openai_finish_reason
from .harmony import (
    HarmonyParts,
    final_channel_closed,
    normalize_reasoning_effort,
    parse_harmony,
)
from .profiles import DEFAULT_MODEL, DEFAULT_PROFILE_ID, get_profile, list_profiles
from .runtime import BusyError, GenerateResult, GenerateStats, OzekMLXRuntime, runtime
from .structured import (
    extract_json,
    json_system_nudge,
    try_extract_json,
    validate_against_schema,
)

__version__ = "0.1.1"
__all__ = [
    "BusyError",
    "CONTRACT_VERSION",
    "ContextBudgetError",
    "DEFAULT_MODEL",
    "DEFAULT_PROFILE_ID",
    "GenerateResult",
    "GenerateStats",
    "HarmonyParts",
    "OzekMLXRuntime",
    "extract_json",
    "final_channel_closed",
    "get_profile",
    "json_system_nudge",
    "list_profiles",
    "load_config",
    "normalize_reasoning_effort",
    "openai_finish_reason",
    "parse_harmony",
    "runtime",
    "try_extract_json",
    "validate_against_schema",
    "__version__",
]
