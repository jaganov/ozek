#!/usr/bin/env python3
"""Ozek CLI — generate and runtime info."""

from __future__ import annotations

import argparse
import json
import sys

from .runtime import DEFAULT_MODEL, runtime


def cmd_generate(args: argparse.Namespace) -> int:
    runtime.load(args.model)
    result = runtime.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system=args.system,
        enable_thinking=args.think,
    )
    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.text)
        s = result.stats
        print(
            f"\n-- stats: gen_tps={s.generation_tps} "
            f"ttft_s={s.ttft_s} peak_gb={s.peak_memory_gb} "
            f"tokens={s.generation_tokens}",
            file=sys.stderr,
        )
    return 0


def cmd_info(_: argparse.Namespace) -> int:
    print(json.dumps(runtime.info(), ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ozek",
        description="Ozek — thin Mac-first MLX runtime for local agents",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="load (if needed) and generate")
    g.add_argument("--model", default=DEFAULT_MODEL, help="MLX / Hugging Face model id")
    g.add_argument("--prompt", required=True)
    g.add_argument("--system", default=None)
    g.add_argument("--max-tokens", type=int, default=128)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument(
        "--think",
        action="store_true",
        help="enable Qwen3/3.5 chain-of-thought when supported",
    )
    g.add_argument("--json", action="store_true", help="print JSON with stats")
    g.set_defaults(func=cmd_generate)

    i = sub.add_parser("info", help="runtime info")
    i.set_defaults(func=cmd_info)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
