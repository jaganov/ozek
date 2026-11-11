#!/usr/bin/env python3
"""Ozek local HTTP API for agents (stdlib only).

Endpoints:
  GET  /health
  GET  /v1/info
  POST /v1/generate
  POST /v1/chat/completions   (OpenAI-compatible subset; optional stream)
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .runtime import DEFAULT_MODEL, runtime


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


class Handler(BaseHTTPRequestHandler):
    server_version = "Ozek/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/health", "/v1/health"):
            _send_json(self, 200, {"ok": True, "ready": runtime.ready})
            return
        if path in ("/v1/info", "/info"):
            _send_json(self, 200, runtime.info())
            return
        _send_json(self, 404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            req = _read_json(self)
        except json.JSONDecodeError:
            _send_json(self, 400, {"error": "invalid json"})
            return

        if path == "/v1/generate":
            self._generate_simple(req)
            return
        if path == "/v1/chat/completions":
            self._chat_completions(req)
            return
        _send_json(self, 404, {"error": "not found"})

    def _ensure_model(self, req: dict[str, Any]) -> None:
        model = req.get("model") or runtime.model_id
        if not model:
            raise RuntimeError("no model specified and none loaded")
        runtime.load(model)

    def _generate_simple(self, req: dict[str, Any]) -> None:
        try:
            self._ensure_model(req)
            prompt = req.get("prompt") or ""
            if not prompt:
                _send_json(self, 400, {"error": "prompt required"})
                return
            result = runtime.generate(
                prompt,
                max_tokens=int(req.get("max_tokens") or 128),
                temperature=float(req.get("temperature") or 0.0),
                system=req.get("system"),
                enable_thinking=bool(req.get("think") or False),
            )
            _send_json(self, 200, {"ok": True, **result.as_dict()})
        except Exception as e:  # noqa: BLE001
            _send_json(self, 500, {"ok": False, "error": str(e)})

    def _chat_completions(self, req: dict[str, Any]) -> None:
        try:
            self._ensure_model(req)
            messages = req.get("messages") or []
            if not messages:
                _send_json(self, 400, {"error": "messages required"})
                return
            max_tokens = int(req.get("max_tokens") or 128)
            temperature = float(req.get("temperature") or 0.0)
            think = bool(req.get("think") or False)
            stream = bool(req.get("stream") or False)

            prompt = ""
            for m in messages:
                if m.get("role") == "user":
                    prompt = str(m.get("content") or "")

            if stream:
                self._chat_stream(messages, prompt, max_tokens, temperature, think)
                return

            result = runtime.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                enable_thinking=think,
                messages=messages,
            )
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
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": result.stats.prompt_tokens,
                        "completion_tokens": result.stats.generation_tokens,
                        "total_tokens": result.stats.prompt_tokens
                        + result.stats.generation_tokens,
                    },
                    "ozek_stats": result.stats.as_dict(),
                },
            )
        except Exception as e:  # noqa: BLE001
            _send_json(self, 500, {"error": str(e)})

    def _chat_stream(
        self,
        messages: list[dict[str, str]],
        prompt: str,
        max_tokens: int,
        temperature: float,
        think: bool,
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        cid = f"ozek-{uuid.uuid4().hex[:12]}"
        try:
            for chunk in runtime.stream_tokens(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                enable_thinking=think,
                messages=messages,
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
            done = {
                "id": cid,
                "object": "chat.completion.chunk",
                "model": runtime.model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self.wfile.write(f"data: {json.dumps(done)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception as e:  # noqa: BLE001
            err = {"error": str(e)}
            self.wfile.write(f"data: {json.dumps(err)}\n\n".encode("utf-8"))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Ozek local agent server (MLX)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="preload MLX model at startup")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument(
        "--no-preload",
        action="store_true",
        help="do not load model until first request",
    )
    args = p.parse_args(argv)

    if not args.no_preload:
        print(f"loading {args.model} ...", flush=True)
        secs = runtime.load(args.model)
        print(f"loaded in {secs:.2f}s", flush=True)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"Ozek listening on http://{args.host}:{args.port} (model={runtime.model_id})",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)


if __name__ == "__main__":
    main()
