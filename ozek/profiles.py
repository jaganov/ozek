"""Named model profiles for typical Apple Silicon RAM budgets.

Use a profile id via CLI ``--profile`` / JSON ``"profile"``, or pass any HF / MLX id.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ModelProfile:
    id: str
    model: str
    label: str
    quant: str
    ram_hint_gb: int
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# Default for 16 GB Macs: fast, fits with headroom for OS + agent.
DEFAULT_PROFILE_ID = "qwen25-3b-4bit"

PROFILES: dict[str, ModelProfile] = {
    "qwen25-3b-4bit": ModelProfile(
        id="qwen25-3b-4bit",
        model="mlx-community/Qwen2.5-3B-Instruct-4bit",
        label="Qwen2.5-3B Instruct 4-bit",
        quant="4bit",
        ram_hint_gb=16,
        notes="Default. Best tok/s / memory tradeoff for agents on 16 GB.",
    ),
    "qwen25-7b-4bit": ModelProfile(
        id="qwen25-7b-4bit",
        model="mlx-community/Qwen2.5-7B-Instruct-4bit",
        label="Qwen2.5-7B Instruct 4-bit",
        quant="4bit",
        ram_hint_gb=16,
        notes="Stronger quality; expect higher RSS / lower tok/s on 16 GB.",
    ),
    "qwen35-9b-4bit": ModelProfile(
        id="qwen35-9b-4bit",
        model="mlx-community/Qwen3.5-9B-4bit",
        label="Qwen3.5-9B 4-bit",
        quant="4bit",
        ram_hint_gb=16,
        notes="Heavier; watch memory pressure. Prefer think=false unless needed.",
    ),
}

DEFAULT_MODEL = PROFILES[DEFAULT_PROFILE_ID].model


def get_profile(name: Optional[str] = None) -> ModelProfile:
    key = (name or DEFAULT_PROFILE_ID).strip().lower()
    if key not in PROFILES:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(f"unknown profile {name!r}; known: {known}")
    return PROFILES[key]


def resolve_model(
    *,
    model: Optional[str] = None,
    profile: Optional[str] = None,
) -> str:
    """Resolve HF/MLX model id from explicit model or profile name."""
    if model:
        return model
    if profile:
        return get_profile(profile).model
    return DEFAULT_MODEL


def list_profiles() -> list[dict[str, Any]]:
    return [p.as_dict() for p in PROFILES.values()]
