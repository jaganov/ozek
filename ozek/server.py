#!/usr/bin/env python3
"""Ozek local HTTP API for agents (stdlib only).

Endpoints:
  GET  /health
  GET  /v1/info
  GET  /v1/profiles
  GET  /v1/presets
  GET  /v1/contract
  POST /v1/generate
  POST /v1/chat/completions   (OpenAI-compatible subset; optional stream)
  POST /v1/cancel
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import urlparse

from .config import (
    OzekConfig,
    load_config,
    merge_preset_with_overrides,
    preset_to_generate_kwargs,
    resolve_model_from_config,
)
from .contract import contract_meta, normalize_generate_kwargs, openai_finish_reason
from .profiles import DEFAULT_PROFILE_ID, list_profiles, resolve_model
from .runtime import BusyError, runtime

# Set in main() from --config / discovery.
_CONFIG: OzekConfig = OzekConfig(path=None, raw={})


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length else b"{}"
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _send_json(handler: BaseHTTPRequestHandler, code: int, obj: Any) -> None:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _merge_request(req: dict[str, Any]) -> dict[str, Any]:
    """Apply optional YAML preset, then JSON body overrides → generate kwargs + model."""
    preset_name = req.get("preset")
    base: dict[str, Any] = {}
    if preset_name:
        base = _CONFIG.get_preset(str(preset_name))

    # Body fields that override preset (exclude meta keys).
    skip = {"preset", "messages", "stream", "prompt"}
    overrides = {k: v for k, v in req.items() if k not in skip and v is not None}
    merged = merge_preset_with_overrides(base, overrides=overrides)

    model, profile = resolve_model_from_config(
        _CONFIG,
        model=merged.get("model") or req.get("model") or runtime.model_id,
        profile=merged.get("profile") or req.get("profile"),
        preset=merged,
    )
    # normalize_generate_kwargs expects OpenAI-ish body; also accept preset-shaped.
    body_for_norm = dict(req)
    body_for_norm.update(merged)
    # Prefer preset-shaped conversion for sampler, then fill gaps via contract.
    kwargs = preset_to_generate_kwargs(merged)
    normalized = normalize_generate_kwargs(body_for_norm)
    for k, v in normalized.items():
        kwargs.setdefault(k, v)
    # Explicit merged values win
    kwargs.update(preset_to_generate_kwargs(merged))
    kwargs.setdefault("max_tokens", 128)
    kwargs.setdefault("temperature", 0.0)
    return {"model": model, "profile": profile, "kwargs": kwargs}


class Handler(BaseHTTPRequestHandler):
    server_version = "Ozek/0.1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/health", "/v1/health"):
            _send_json(
                self,
                200,
                {"ok": True, "ready": runtime.ready, "busy": runtime.busy},
            )
            return
        if path in ("/v1/info", "/info"):
            info = runtime.info()
            info["config"] = _CONFIG.as_dict()
            _send_json(self, 200, info)
            return
        if path == "/v1/profiles":
            _send_json(self, 200, {"profiles": list_profiles()})
            return
        if path == "/v1/presets":
            _send_json(
                self,
                200,
                {
                    "config": str(_CONFIG.path) if _CONFIG.path else None,
                    "presets": _CONFIG.presets,
                },
            )
            return
        if path == "/v1/contract":
            _send_json(self, 200, contract_meta())
            return
        _send_json(self, 404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            req = _read_json(self)
        except json.JSONDecodeError:
            _send_json(self, 400, {"error": "invalid json"})
            return

        if path == "/v1/cancel":
            runtime.cancel()
            _send_json(self, 200, {"ok": True, "busy": runtime.busy})
            return
        if path == "/v1/generate":
            self._generate_simple(req)
            return
        if path == "/v1/chat/completions":
            self._chat_completions(req)
            return
        _send_json(self, 404, {"error": "not found"})

    def _ensure_model(self, model: Optional[str], profile: Optional[str]) -> None:
        try:
            model_id = resolve_model(
                model=model,
                profile=profile or (_CONFIG.default_profile or DEFAULT_PROFILE_ID),
            )
        except KeyError as e:
            raise RuntimeError(str(e)) from e
        runtime.load(model_id)

    def _generate_simple(self, req: dict[str, Any]) -> None:
        try:
            prompt = req.get("prompt") or ""
            if not prompt:
                _send_json(self, 400, {"error": "prompt required"})
                return
            merged = _merge_request(req)
            self._ensure_model(merged["model"], merged["profile"])
            result = runtime.generate(prompt, **merged["kwargs"])
            _send_json(self, 200, {"ok": True, **result.as_dict()})
        except BusyError as e:
            _send_json(self, 503, {"ok": False, "error": str(e), "code": "busy"})
        except KeyError as e:
            _send_json(self, 400, {"ok": False, "error": str(e)})
        except ValueError as e:
            _send_json(self, 400, {"ok": False, "error": str(e)})
        except Exception as e:  # noqa: BLE001
            _send_json(self, 500, {"ok": False, "error": str(e)})

    def _chat_completions(self, req: dict[str, Any]) -> None:
        try:
            messages = req.get("messages") or []
            if not messages:
                _send_json(self, 400, {"error": "messages required"})
                return
            merged = _merge_request(req)
            self._ensure_model(merged["model"], merged["profile"])
            kwargs = merged["kwargs"]
            stream = bool(req.get("stream") or False)

            prompt = ""
            for m in messages:
                if m.get("role") == "user":
                    prompt = str(m.get("content") or "")

            if stream:
                self._chat_stream(messages, prompt, kwargs)
                return

            result = runtime.generate(prompt, messages=messages, **kwargs)
            _send_json(
                self,
                200,
                {
                    "id": f"ozek-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "model": runtime.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": result.text,
                            },
                            "finish_reason": openai_finish_reason(
                                result.stats.finish_reason
                            ),
                        }
                    ],
                    "usage": {
                        "prompt_tokens": result.stats.prompt_tokens,
                        "completion_tokens": result.stats.generation_tokens,
                        "total_tokens": result.stats.prompt_tokens
                        + result.stats.generation_tokens,
                    },
                    "ozek_stats": result.stats.as_dict(),
                    "parsed": result.parsed if result.response_format == "json" else None,
                    "parse_error": result.parse_error,
                },
            )
        except BusyError as e:
            _send_json(self, 503, {"error": str(e), "code": "busy"})
        except KeyError as e:
            _send_json(self, 400, {"error": str(e)})
        except ValueError as e:
            _send_json(self, 400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            _send_json(self, 500, {"error": str(e)})

    def _chat_stream(
        self,
        messages: list[dict[str, str]],
        prompt: str,
        kwargs: dict[str, Any],
    ) -> None:
        if runtime.busy:
            _send_json(self, 503, {"error": "runtime busy", "code": "busy"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        cid = f"ozek-{uuid.uuid4().hex[:12]}"
        try:
            for chunk in runtime.stream_tokens(
                prompt,
                messages=messages,
                **kwargs,
            ):
                payload = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "model": runtime.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }
                    ],
                }
                data = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                self.wfile.write(data.encode("utf-8"))
                self.wfile.flush()
            ozek_fr = runtime.last_stream_finish or "eos"
            done = {
                "id": cid,
                "object": "chat.completion.chunk",
                "model": runtime.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": openai_finish_reason(ozek_fr),
                    }
                ],
                "ozek_finish_reason": ozek_fr,
            }
            self.wfile.write(f"data: {json.dumps(done)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except BusyError as e:
            err = {"error": str(e), "code": "busy"}
            self.wfile.write(f"data: {json.dumps(err)}\n\n".encode("utf-8"))
        except Exception as e:  # noqa: BLE001
            err = {"error": str(e)}
            self.wfile.write(f"data: {json.dumps(err)}\n\n".encode("utf-8"))


def main(argv: list[str] | None = None) -> None:
    global _CONFIG

    p = argparse.ArgumentParser(description="Ozek local agent server (MLX)")
    p.add_argument("--config", default=None, help="path to ozek.yml")
    p.add_argument("--model", default=None, help="preload MLX / HF model id")
    p.add_argument(
        "--profile",
        default=None,
        help="preload named profile (see ozek profiles)",
    )
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument(
        "--no-preload",
        action="store_true",
        help="do not load model until first request",
    )
    args = p.parse_args(argv)

    _CONFIG = load_config(args.config)
    server_cfg = _CONFIG.server
    host = args.host or str(server_cfg.get("host") or "127.0.0.1")
    port = int(args.port if args.port is not None else server_cfg.get("port") or 8787)

    if _CONFIG.path:
        print(f"config: {_CONFIG.path}", flush=True)

    if not args.no_preload:
        model, profile = resolve_model_from_config(
            _CONFIG, model=args.model, profile=args.profile
        )
        model_id = resolve_model(
            model=model,
            profile=profile or DEFAULT_PROFILE_ID,
        )
        print(f"loading {model_id} ...", flush=True)
        secs = runtime.load(model_id)
        print(f"loaded in {secs:.2f}s", flush=True)

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(
        f"Ozek listening on http://{host}:{port} (model={runtime.model_id})",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)


if __name__ == "__main__":
    main()
