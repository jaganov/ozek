#!/usr/bin/env python3
"""Ozek CLI — generate, run presets, profiles, and runtime info."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional

from .config import (
    load_config,
    merge_preset_with_overrides,
    preset_to_generate_kwargs,
    resolve_model_from_config,
)
from .profiles import DEFAULT_PROFILE_ID, list_profiles, resolve_model
from .runtime import BusyError, GenerateResult, runtime


def _load_schema(raw: str | None):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _print_result(result: GenerateResult, *, as_json: bool) -> int:
    if as_json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    elif result.response_format == "json" and result.parsed is not None:
        print(json.dumps(result.parsed, ensure_ascii=False, indent=2))
        if result.parse_error:
            print(f"-- parse_error: {result.parse_error}", file=sys.stderr)
    else:
        print(result.text)
        if result.parse_error:
            print(f"-- parse_error: {result.parse_error}", file=sys.stderr)
        s = result.stats
        print(
            f"\n-- stats: gen_tps={s.generation_tps} "
            f"ttft_s={s.ttft_s} peak_gb={s.peak_memory_gb} "
            f"tokens={s.generation_tokens} finish={s.finish_reason}",
            file=sys.stderr,
        )
    return 0 if not result.parse_error else 3


def _do_generate(
    prompt: str,
    *,
    model: Optional[str],
    profile: Optional[str],
    gen_kwargs: dict[str, Any],
    as_json: bool,
) -> int:
    try:
        model_id = resolve_model(model=model, profile=profile)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    runtime.load(model_id)
    try:
        result = runtime.generate(prompt, **gen_kwargs)
    except BusyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return _print_result(result, as_json=as_json)


def _cli_overrides(args: argparse.Namespace, *, for_run: bool) -> dict[str, Any]:
    """Build override dict from CLI. For ``run``, only include flags the user set."""
    out: dict[str, Any] = {}

    def take(name: str, value: Any, *, always: bool = False) -> None:
        if always or value is not None:
            out[name] = value

    if for_run:
        # argparse defaults would clobber preset — only apply if dest was on argv
        # We use None defaults on run parser for optional knobs.
        take("system", args.system)
        take("max_tokens", args.max_tokens)
        take("temperature", args.temperature)
        take("top_p", args.top_p)
        take("top_k", args.top_k)
        take("min_p", args.min_p)
        take("repetition_penalty", args.repetition_penalty)
        take("stop", args.stop)
        if getattr(args, "think", False):
            out["think"] = True
        if getattr(args, "response_json", False):
            out["response_format"] = "json"
        if args.schema is not None:
            out["schema"] = _load_schema(args.schema)
        take("timeout_s", args.timeout)
        take("seed", args.seed)
        take("json_retries", args.json_retries)
        take("model", args.model)
        take("profile", args.profile)
    else:
        out["system"] = args.system
        out["max_tokens"] = args.max_tokens
        out["temperature"] = args.temperature
        out["top_p"] = args.top_p
        out["top_k"] = args.top_k
        out["min_p"] = args.min_p
        if args.repetition_penalty is not None:
            out["repetition_penalty"] = args.repetition_penalty
        if args.stop is not None:
            out["stop"] = args.stop
        if args.think:
            out["think"] = True
        out["response_format"] = "json" if args.response_json else "text"
        if args.schema is not None:
            out["schema"] = _load_schema(args.schema)
        if args.timeout is not None:
            out["timeout_s"] = args.timeout
        if args.seed is not None:
            out["seed"] = args.seed
        if args.json_retries:
            out["json_retries"] = args.json_retries
        out["model"] = args.model
        out["profile"] = args.profile
    return out


def cmd_generate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    overrides = _cli_overrides(args, for_run=False)
    model, profile = resolve_model_from_config(
        cfg,
        model=overrides.pop("model", None),
        profile=overrides.pop("profile", None),
    )
    gen_kwargs = preset_to_generate_kwargs(overrides)
    return _do_generate(
        args.prompt,
        model=model,
        profile=profile or DEFAULT_PROFILE_ID,
        gen_kwargs=gen_kwargs,
        as_json=args.json,
    )


def cmd_run(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config, required=True)
        preset = cfg.get_preset(args.preset)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    overrides = _cli_overrides(args, for_run=True)
    model_o = overrides.pop("model", None)
    profile_o = overrides.pop("profile", None)
    merged = merge_preset_with_overrides(preset, overrides=overrides)
    model, profile = resolve_model_from_config(
        cfg, model=model_o, profile=profile_o, preset=merged
    )
    gen_kwargs = preset_to_generate_kwargs(merged)
    # Defaults if preset omitted sampler fields
    gen_kwargs.setdefault("max_tokens", 128)
    gen_kwargs.setdefault("temperature", 0.0)
    return _do_generate(
        args.prompt,
        model=model,
        profile=profile or DEFAULT_PROFILE_ID,
        gen_kwargs=gen_kwargs,
        as_json=args.json,
    )


def cmd_info(args: argparse.Namespace) -> int:
    info = runtime.info()
    cfg = load_config(args.config)
    info["config"] = cfg.as_dict()
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


def cmd_profiles(_: argparse.Namespace) -> int:
    print(json.dumps({"profiles": list_profiles()}, ensure_ascii=False, indent=2))
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config, required=not args.optional)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = cfg.as_dict()
    if args.verbose:
        out["presets_detail"] = cfg.presets
        out["raw"] = cfg.raw
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_cancel(_: argparse.Namespace) -> int:
    runtime.cancel()
    print(json.dumps({"ok": True, "busy": runtime.busy}, ensure_ascii=False))
    return 0


def _add_common_gen_flags(p: argparse.ArgumentParser, *, run_mode: bool) -> None:
    p.add_argument("--config", default=None, help="path to ozek.yml")
    p.add_argument("--model", default=None, help="MLX / HF model id")
    if run_mode:
        p.add_argument("--profile", default=None, help="named model profile")
        p.add_argument("--system", default=None)
        p.add_argument("--max-tokens", type=int, default=None)
        p.add_argument("--temperature", type=float, default=None)
        p.add_argument("--top-p", type=float, default=None)
        p.add_argument("--top-k", type=int, default=None)
        p.add_argument("--min-p", type=float, default=None)
    else:
        p.add_argument(
            "--profile",
            default=None,
            help=f"named profile (default: config or {DEFAULT_PROFILE_ID})",
        )
        p.add_argument("--system", default=None)
        p.add_argument("--max-tokens", type=int, default=128)
        p.add_argument("--temperature", type=float, default=0.0)
        p.add_argument("--top-p", type=float, default=0.0)
        p.add_argument("--top-k", type=int, default=0)
        p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--stop", action="append", default=None)
    p.add_argument("--think", action="store_true")
    p.add_argument(
        "--response-json",
        action="store_true",
        help="JSON mode (overrides preset response_format when set)",
    )
    p.add_argument("--schema", default=None, help="JSON schema string or JSON object text")
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="wall-clock generate timeout in seconds",
    )
    p.add_argument("--seed", type=int, default=None, help="MLX RNG seed")
    p.add_argument(
        "--json-retries",
        type=int,
        default=None if run_mode else 0,
        help="extra generate attempts when JSON parse fails (json mode)",
    )
    p.add_argument("--json", action="store_true", help="print full result JSON with stats")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ozek",
        description="Ozek — thin Mac-first MLX runtime for local agents",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="load (if needed) and generate")
    g.add_argument("--prompt", required=True)
    _add_common_gen_flags(g, run_mode=False)
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="run a named preset from ozek.yml")
    r.add_argument("preset", help="preset name under presets: in ozek.yml")
    r.add_argument("--prompt", required=True)
    _add_common_gen_flags(r, run_mode=True)
    r.set_defaults(func=cmd_run)

    i = sub.add_parser("info", help="runtime info")
    i.add_argument("--config", default=None)
    i.set_defaults(func=cmd_info)

    pr = sub.add_parser("profiles", help="list built-in model profiles")
    pr.set_defaults(func=cmd_profiles)

    cf = sub.add_parser("config", help="show resolved ozek.yml")
    cf.add_argument("--config", default=None)
    cf.add_argument("-v", "--verbose", action="store_true", help="include preset bodies")
    cf.add_argument(
        "--optional",
        action="store_true",
        help="do not error if no config file is found",
    )
    cf.set_defaults(func=cmd_config)

    c = sub.add_parser(
        "cancel", help="request cancel of in-process generate (same process)"
    )
    c.set_defaults(func=cmd_cancel)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
