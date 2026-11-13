"""Ozek MLX runtime — one model slot, load / generate / stream / metrics.

Heavy compute runs in MLX Metal kernels. This module owns orchestration only:
sampler params, stop sequences, busy slot, cooperative cancel.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Optional, Sequence, Union

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .contract import CONTRACT_VERSION, contract_meta
from .profiles import DEFAULT_MODEL, DEFAULT_PROFILE_ID
from .structured import SchemaInput, apply_json_mode, try_extract_json

StopInput = Union[str, Sequence[str], None]
ResponseFormat = Optional[str]  # None | "text" | "json"


def _rss_bytes() -> int:
    try:
        import resource
        import sys

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # macOS: bytes; Linux: kilobytes
        if sys.platform != "darwin":
            rss *= 1024
        return rss
    except Exception:  # noqa: BLE001
        return 0


class BusyError(RuntimeError):
    """Raised when a generate/stream is already in progress."""


@dataclass
class GenerateStats:
    model: str
    prompt_tokens: int
    generation_tokens: int
    ttft_s: float
    wall_s: float
    prompt_tps: float
    generation_tps: float
    decode_tps_wall: float
    peak_memory_gb: float
    enable_thinking: bool
    finish_reason: str = "length"  # length | stop | cancel | eos | timeout
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: Optional[float] = None
    stop: list[str] = field(default_factory=list)
    seed: Optional[int] = None
    timeout_s: Optional[float] = None
    json_retries: int = 0
    attempts: int = 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerateResult:
    text: str
    stats: GenerateStats
    parsed: Any = None
    parse_error: Optional[str] = None
    response_format: str = "text"

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "text": self.text,
            "stats": self.stats.as_dict(),
            "response_format": self.response_format,
        }
        if self.response_format == "json":
            out["parsed"] = self.parsed
            if self.parse_error:
                out["parse_error"] = self.parse_error
        return out


def _normalize_stop(stop: StopInput) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    return [s for s in stop if s]


def _apply_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    """Truncate at the earliest stop match. Returns (text, hit)."""
    if not stops or not text:
        return text, False
    best: Optional[int] = None
    for s in stops:
        i = text.find(s)
        if i >= 0 and (best is None or i < best):
            best = i
    if best is None:
        return text, False
    return text[:best], True


class OzekMLXRuntime:
    """One loaded model at a time: load / generate / stream / metrics."""

    def __init__(self) -> None:
        # Generate/stream exclusive slot (busy → BusyError).
        self._slot = threading.Lock()
        # Protects model pointers; load fast-path does not take _slot.
        self._model_mu = threading.Lock()
        self._cancel = threading.Event()
        self.model_id: Optional[str] = None
        self._model = None
        self._tokenizer = None
        self.load_s: float = 0.0
        self.requests: int = 0
        self._last_stats: Optional[GenerateStats] = None
        self._last_stream_finish: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def busy(self) -> bool:
        return self._slot.locked()

    @property
    def last_stream_finish(self) -> Optional[str]:
        """finish_reason from the most recent ``stream_tokens`` call."""
        return self._last_stream_finish

    def cancel(self) -> None:
        """Request cooperative cancel of the in-flight generate/stream."""
        self._cancel.set()

    def load(self, model_id: str, force: bool = False) -> float:
        with self._model_mu:
            if self.ready and self.model_id == model_id and not force:
                return self.load_s

        # Swap only when no generate holds the slot.
        self._slot.acquire(blocking=True)
        try:
            with self._model_mu:
                if self.ready and self.model_id == model_id and not force:
                    return self.load_s
                from mlx_lm import load

                t0 = time.perf_counter()
                self._model, self._tokenizer = load(model_id)
                self.load_s = time.perf_counter() - t0
                self.model_id = model_id
                return self.load_s
        finally:
            self._slot.release()

    def unload(self) -> None:
        if not self._slot.acquire(blocking=False):
            raise BusyError("runtime busy")
        try:
            with self._model_mu:
                self._model = None
                self._tokenizer = None
                self.model_id = None
            try:
                import mlx.core as mx

                mx.metal.clear_cache()
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._slot.release()

    def info(self) -> dict[str, Any]:
        peak = 0.0
        try:
            import mlx.core as mx

            peak = float(mx.get_peak_memory()) / (1024**3)
        except Exception:  # noqa: BLE001
            pass
        rss = _rss_bytes()
        with self._model_mu:
            model = self.model_id
            ready = self.ready
            load_s = self.load_s
            last = self._last_stats
        last_summary = None
        if last is not None:
            last_summary = {
                "finish_reason": last.finish_reason,
                "generation_tokens": last.generation_tokens,
                "generation_tps": last.generation_tps,
                "ttft_s": last.ttft_s,
                "wall_s": last.wall_s,
                "peak_memory_gb": last.peak_memory_gb,
            }
        return {
            "name": "ozek",
            "version": "0.1.1",
            "contract_version": CONTRACT_VERSION,
            "backend": "mlx",
            "ready": ready,
            "busy": self.busy,
            "model": model,
            "default_profile": DEFAULT_PROFILE_ID,
            "default_model": DEFAULT_MODEL,
            "load_s": round(load_s, 4),
            "requests": self.requests,
            "peak_memory_gb": round(peak, 3),
            "rss_gb": round(rss / (1024**3), 3) if rss else 0.0,
            "last": last_summary,
            "contract": contract_meta(),
        }

    def _snapshot(self) -> tuple[Any, Any, str]:
        with self._model_mu:
            if self._model is None or self._tokenizer is None or not self.model_id:
                raise RuntimeError("model not loaded; call load() first")
            return self._model, self._tokenizer, self.model_id

    def _build_prompt(
        self,
        tokenizer: Any,
        prompt: str,
        *,
        system: Optional[str],
        enable_thinking: bool,
        messages: Optional[list[dict[str, str]]],
    ) -> str:
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

        if not hasattr(tokenizer, "apply_chat_template"):
            return prompt
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _acquire_slot(self) -> None:
        if not self._slot.acquire(blocking=False):
            raise BusyError("runtime busy")
        self._cancel.clear()

    def _release_slot(self) -> None:
        self._slot.release()

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: Optional[float] = None,
        stop: StopInput = None,
        system: Optional[str] = None,
        enable_thinking: bool = False,
        messages: Optional[list[dict[str, str]]] = None,
        response_format: ResponseFormat = None,
        schema: SchemaInput = None,
        timeout_s: Optional[float] = None,
        seed: Optional[int] = None,
        json_retries: int = 0,
    ) -> GenerateResult:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        fmt = (response_format or "text").lower().strip()
        if fmt not in ("text", "json"):
            raise ValueError("response_format must be 'text' or 'json'")
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if json_retries < 0:
            raise ValueError("json_retries must be >= 0")

        base_system = system
        base_messages = messages
        if fmt == "json":
            base_system, base_messages = apply_json_mode(
                system=system, messages=messages, schema=schema
            )

        stops = _normalize_stop(stop)
        retries = int(json_retries) if fmt == "json" else 0
        max_attempts = 1 + retries

        self._acquire_slot()
        try:
            model, tokenizer, model_id = self._snapshot()
            sampler = make_sampler(
                temp=temperature, top_p=top_p, top_k=top_k, min_p=min_p
            )
            logits_processors = None
            if repetition_penalty is not None:
                logits_processors = make_logits_processors(
                    repetition_penalty=repetition_penalty
                )

            result: Optional[GenerateResult] = None
            for attempt in range(1, max_attempts + 1):
                sys_prompt = base_system
                msgs = base_messages
                if fmt == "json" and attempt > 1:
                    repair = (
                        "Your previous reply was not valid JSON. "
                        "Reply again with valid JSON only — no markdown, no commentary."
                    )
                    if msgs is not None:
                        msgs = [dict(m) for m in msgs]
                        msgs.append({"role": "user", "content": repair})
                    else:
                        sys_prompt = (
                            f"{sys_prompt}\n\n{repair}" if sys_prompt else repair
                        )

                if seed is not None:
                    import mlx.core as mx

                    mx.random.seed(int(seed) + attempt - 1)

                text_prompt = self._build_prompt(
                    tokenizer,
                    prompt,
                    system=sys_prompt,
                    enable_thinking=enable_thinking,
                    messages=msgs,
                )

                pieces: list[str] = []
                first_token_s: Optional[float] = None
                last: Any = None
                finish_reason = "length"
                t0 = time.perf_counter()
                deadline = (t0 + timeout_s) if timeout_s is not None else None

                for resp in stream_generate(
                    model,
                    tokenizer,
                    prompt=text_prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                ):
                    if self._cancel.is_set():
                        finish_reason = "cancel"
                        break
                    if deadline is not None and time.perf_counter() >= deadline:
                        finish_reason = "timeout"
                        break
                    if first_token_s is None:
                        first_token_s = time.perf_counter() - t0
                    if resp.text:
                        pieces.append(resp.text)
                    last = resp

                    text_so_far = "".join(pieces)
                    text_so_far, hit = _apply_stop(text_so_far, stops)
                    if hit:
                        pieces = [text_so_far]
                        finish_reason = "stop"
                        break

                    fr = getattr(resp, "finish_reason", None)
                    if fr is not None:
                        finish_reason = "eos" if fr == "stop" else str(fr)
                        break

                wall_s = time.perf_counter() - t0
                self.requests += 1
                text = "".join(pieces)
                text, hit = _apply_stop(text, stops)
                if hit and finish_reason == "length":
                    finish_reason = "stop"

                gen_tokens = int(getattr(last, "generation_tokens", 0) or 0)
                ttft = float(first_token_s or 0.0)
                decode_tps = 0.0
                if (
                    first_token_s is not None
                    and gen_tokens > 1
                    and wall_s > first_token_s
                ):
                    decode_tps = (gen_tokens - 1) / (wall_s - first_token_s)

                stats = GenerateStats(
                    model=model_id,
                    prompt_tokens=int(getattr(last, "prompt_tokens", 0) or 0),
                    generation_tokens=gen_tokens,
                    ttft_s=round(ttft, 4),
                    wall_s=round(wall_s, 4),
                    prompt_tps=round(
                        float(getattr(last, "prompt_tps", 0.0) or 0.0), 2
                    ),
                    generation_tps=round(
                        float(getattr(last, "generation_tps", 0.0) or 0.0), 2
                    ),
                    decode_tps_wall=round(decode_tps, 2),
                    peak_memory_gb=round(
                        float(getattr(last, "peak_memory", 0.0) or 0.0), 3
                    ),
                    enable_thinking=enable_thinking,
                    finish_reason=finish_reason,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    stop=stops,
                    seed=seed,
                    timeout_s=timeout_s,
                    json_retries=retries,
                    attempts=attempt,
                )
                parsed = None
                parse_error = None
                if fmt == "json":
                    parsed, parse_error = try_extract_json(text)

                result = GenerateResult(
                    text=text,
                    stats=stats,
                    parsed=parsed,
                    parse_error=parse_error,
                    response_format=fmt,
                )
                self._last_stats = stats

                if finish_reason in ("cancel", "timeout"):
                    break
                if fmt != "json" or parse_error is None:
                    break

            assert result is not None
            return result
        finally:
            self._release_slot()

    def stream_tokens(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: Optional[float] = None,
        stop: StopInput = None,
        system: Optional[str] = None,
        enable_thinking: bool = False,
        messages: Optional[list[dict[str, str]]] = None,
        response_format: ResponseFormat = None,
        schema: SchemaInput = None,
        timeout_s: Optional[float] = None,
        seed: Optional[int] = None,
        json_retries: int = 0,  # ignored for stream; use generate for retries
    ) -> Iterator[str]:
        """Yield text chunks. Raises BusyError if another job holds the slot.

        Sets ``last_stream_finish`` when done (``length``|``stop``|``cancel``|``eos``|``timeout``).
        ``response_format=json`` only injects the prompt nudge; parse via ``generate``.
        """
        del json_retries  # stream does not retry; accepted for shared kwargs
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        fmt = (response_format or "text").lower().strip()
        if fmt not in ("text", "json"):
            raise ValueError("response_format must be 'text' or 'json'")
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if fmt == "json":
            system, messages = apply_json_mode(
                system=system, messages=messages, schema=schema
            )

        stops = _normalize_stop(stop)
        self._acquire_slot()
        finish_reason = "length"
        try:
            model, tokenizer, _model_id = self._snapshot()
            if seed is not None:
                import mlx.core as mx

                mx.random.seed(int(seed))
            text_prompt = self._build_prompt(
                tokenizer,
                prompt,
                system=system,
                enable_thinking=enable_thinking,
                messages=messages,
            )
            sampler = make_sampler(
                temp=temperature, top_p=top_p, top_k=top_k, min_p=min_p
            )
            logits_processors = None
            if repetition_penalty is not None:
                logits_processors = make_logits_processors(
                    repetition_penalty=repetition_penalty
                )

            emitted = ""
            t0 = time.perf_counter()
            deadline = (t0 + timeout_s) if timeout_s is not None else None
            for resp in stream_generate(
                model,
                tokenizer,
                prompt=text_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
            ):
                if self._cancel.is_set():
                    finish_reason = "cancel"
                    break
                if deadline is not None and time.perf_counter() >= deadline:
                    finish_reason = "timeout"
                    break
                if not resp.text:
                    if resp.finish_reason is not None:
                        finish_reason = (
                            "eos" if resp.finish_reason == "stop" else str(resp.finish_reason)
                        )
                        break
                    continue
                emitted += resp.text
                truncated, hit = _apply_stop(emitted, stops)
                if hit:
                    prev_len = len(emitted) - len(resp.text)
                    allowed = truncated[prev_len:] if prev_len < len(truncated) else ""
                    if allowed:
                        yield allowed
                    finish_reason = "stop"
                    break
                yield resp.text
                if resp.finish_reason is not None:
                    finish_reason = (
                        "eos" if resp.finish_reason == "stop" else str(resp.finish_reason)
                    )
                    break
            self.requests += 1
        finally:
            self._last_stream_finish = finish_reason
            self._release_slot()


runtime = OzekMLXRuntime()
