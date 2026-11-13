"""Ozek — thin Mac-first MLX runtime for local agents."""

from .config import load_config
from .contract import CONTRACT_VERSION, openai_finish_reason
from .profiles import DEFAULT_MODEL, DEFAULT_PROFILE_ID, get_profile, list_profiles
from .runtime import BusyError, GenerateResult, GenerateStats, OzekMLXRuntime, runtime
from .structured import extract_json, json_system_nudge, try_extract_json

__version__ = "0.1.1"
__all__ = [
    "BusyError",
    "CONTRACT_VERSION",
    "DEFAULT_MODEL",
    "DEFAULT_PROFILE_ID",
    "GenerateResult",
    "GenerateStats",
    "OzekMLXRuntime",
    "extract_json",
    "get_profile",
    "json_system_nudge",
    "list_profiles",
    "load_config",
    "openai_finish_reason",
    "runtime",
    "try_extract_json",
    "__version__",
]
