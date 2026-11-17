"""YAML config + named presets for Ozek.

Search order for config file (first hit wins):
  1. explicit ``--config`` / ``path=``
  2. ``$OZEK_CONFIG``
  3. ``./ozek.local.yml`` then ``./ozek.local.yaml``  (personal, gitignored)
  4. ``./ozek.yml`` then ``./ozek.yaml``              (personal, gitignored)
  5. ``./configs/local.yml`` then ``./configs/local.yaml``
  6. ``~/.config/ozek/ozek.yml`` then ``~/.config/ozek/ozek.yaml``
  7. bundled ``configs/ozek.example.yml`` (repo get-started fallback)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

CONFIG_FILENAMES = ("ozek.yml", "ozek.yaml")
LOCAL_FILENAMES = ("ozek.local.yml", "ozek.local.yaml")
CONFIGS_LOCAL_FILENAMES = ("local.yml", "local.yaml")

# Keys allowed inside a preset (forward-compatible: unknown keys ignored with warn?).
PRESET_KEYS = frozenset(
    {
        "model",
        "profile",
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
        "response_format",
        "schema",
        "timeout_s",
        "seed",
        "json_retries",
    }
)


def package_root() -> Path:
    """Repo / install root that contains ``configs/`` (parent of the ``ozek`` package)."""
    return Path(__file__).resolve().parent.parent


def bundled_example_path() -> Path:
    return package_root() / "configs" / "ozek.example.yml"


@dataclass
class OzekConfig:
    path: Optional[Path]
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def default_profile(self) -> Optional[str]:
        v = self.raw.get("default_profile")
        return str(v) if v else None

    @property
    def default_model(self) -> Optional[str]:
        v = self.raw.get("default_model")
        return str(v) if v else None

    @property
    def max_context_tokens(self) -> Optional[int]:
        v = self.raw.get("max_context_tokens")
        if v is None or v == "":
            return None
        return int(v)

    @property
    def server(self) -> dict[str, Any]:
        s = self.raw.get("server") or {}
        return s if isinstance(s, dict) else {}

    @property
    def presets(self) -> dict[str, dict[str, Any]]:
        p = self.raw.get("presets") or {}
        if not isinstance(p, dict):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for name, body in p.items():
            if isinstance(body, dict):
                out[str(name)] = body
        return out

    def list_presets(self) -> list[str]:
        return sorted(self.presets)

    def get_preset(self, name: str) -> dict[str, Any]:
        if name not in self.presets:
            known = ", ".join(self.list_presets()) or "(none)"
            raise KeyError(f"unknown preset {name!r}; known: {known}")
        body = self.presets[name]
        return {k: v for k, v in body.items() if k in PRESET_KEYS}

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path) if self.path else None,
            "default_profile": self.default_profile,
            "default_model": self.default_model,
            "max_context_tokens": self.max_context_tokens,
            "server": self.server,
            "presets": sorted(self.presets),
        }


def config_search_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get("OZEK_CONFIG")
    if env:
        paths.append(Path(env).expanduser())
    cwd = Path.cwd()
    for name in LOCAL_FILENAMES:
        paths.append(cwd / name)
    for name in CONFIG_FILENAMES:
        paths.append(cwd / name)
    configs_dir = cwd / "configs"
    for name in CONFIGS_LOCAL_FILENAMES:
        paths.append(configs_dir / name)
    home_cfg = Path.home() / ".config" / "ozek"
    for name in CONFIG_FILENAMES:
        paths.append(home_cfg / name)
    bundled = bundled_example_path()
    paths.append(bundled)
    return paths


def find_config_path(explicit: Optional[str | Path] = None) -> Optional[Path]:
    if explicit is not None:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"config not found: {p}")
        return p
    for p in config_search_paths():
        if p.is_file():
            return p
    return None


def load_config(path: Optional[str | Path] = None, *, required: bool = False) -> OzekConfig:
    """Load YAML config. If nothing found and not required, return empty config."""
    found = find_config_path(path)
    if found is None:
        if required:
            raise FileNotFoundError(
                "no ozek.yml found (cwd, $OZEK_CONFIG, ~/.config/ozek/, "
                "or configs/ozek.example.yml)"
            )
        return OzekConfig(path=None, raw={})

    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            "PyYAML required for config files; pip install pyyaml"
        ) from e

    text = found.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {found}")
    return OzekConfig(path=found, raw=data)


def merge_preset_with_overrides(
    preset: dict[str, Any],
    *,
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Merge preset + explicit overrides. ``None`` / missing override keeps preset."""
    out = dict(preset)
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        if k == "stop" and v == []:
            continue
        out[k] = v
    return out


def preset_to_generate_kwargs(merged: dict[str, Any]) -> dict[str, Any]:
    """Map preset/merged dict → runtime.generate kwargs (+ model/profile side keys)."""
    kwargs: dict[str, Any] = {}
    if "system" in merged:
        kwargs["system"] = merged["system"]
    if "max_tokens" in merged:
        kwargs["max_tokens"] = int(merged["max_tokens"])
    if "temperature" in merged:
        kwargs["temperature"] = float(merged["temperature"])
    if "top_p" in merged:
        kwargs["top_p"] = float(merged["top_p"])
    if "top_k" in merged:
        kwargs["top_k"] = int(merged["top_k"])
    if "min_p" in merged:
        kwargs["min_p"] = float(merged["min_p"])
    if merged.get("repetition_penalty") is not None:
        kwargs["repetition_penalty"] = float(merged["repetition_penalty"])
    if "stop" in merged:
        stop = merged["stop"]
        if isinstance(stop, str):
            kwargs["stop"] = [stop]
        elif stop is not None:
            kwargs["stop"] = list(stop)
    if "think" in merged:
        kwargs["enable_thinking"] = bool(merged["think"])
    if "final_only" in merged:
        kwargs["final_only"] = bool(merged["final_only"])
    else:
        kwargs["final_only"] = True
    if "reasoning_effort" in merged and merged["reasoning_effort"] is not None:
        kwargs["reasoning_effort"] = str(merged["reasoning_effort"])
    if "stop_after_final" in merged:
        kwargs["stop_after_final"] = bool(merged["stop_after_final"])
    if merged.get("max_context_tokens") is not None:
        kwargs["max_context_tokens"] = int(merged["max_context_tokens"])
    rf = merged.get("response_format")
    if rf is not None:
        if isinstance(rf, dict):
            rtype = str(rf.get("type") or "").lower()
            if rtype in ("json_object", "json", "json_schema"):
                kwargs["response_format"] = "json"
                if isinstance(rf.get("json_schema"), dict) and "schema" in rf["json_schema"]:
                    kwargs["schema"] = rf["json_schema"]["schema"]
                elif rf.get("schema") is not None:
                    kwargs["schema"] = rf["schema"]
            else:
                kwargs["response_format"] = "text"
        else:
            kwargs["response_format"] = str(rf).lower().strip()
    if "schema" in merged and "schema" not in kwargs:
        kwargs["schema"] = merged["schema"]
    if merged.get("timeout_s") is not None:
        kwargs["timeout_s"] = float(merged["timeout_s"])
    if merged.get("seed") is not None:
        kwargs["seed"] = int(merged["seed"])
    if merged.get("json_retries") is not None:
        kwargs["json_retries"] = int(merged["json_retries"])
    return kwargs


def resolve_model_from_config(
    cfg: OzekConfig,
    *,
    model: Optional[str] = None,
    profile: Optional[str] = None,
    preset: Optional[dict[str, Any]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve (model, profile) with precedence: explicit > preset > config defaults."""
    preset = preset or {}
    m = model or preset.get("model") or cfg.default_model
    p = profile or preset.get("profile") or cfg.default_profile
    return (
        str(m) if m else None,
        str(p) if p else None,
    )
