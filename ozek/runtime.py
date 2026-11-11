"""Ozek MLX runtime — one model slot, load / generate / stream / metrics.

Heavy compute runs in MLX Metal kernels. This module owns orchestration only.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Optional

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"


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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerateResult:
    text: str
    stats: GenerateStats

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "stats": self.stats.as_dict()}


class OzekMLXRuntime:
    """One loaded model at a time — thin agent runtime, not a model zoo."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model_id: Optional[str] = None
        self._model = None
        self._tokenizer = None
        self.load_s: float = 0.0
        self.requests: int = 0

    @property
    def ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def load(self, model_id: str, force: bool = False) -> float:
        with self._lock:
            if self.ready and self.model_id == model_id and not force:
                return self.load_s
            from mlx_lm import load

            t0 = time.perf_counter()
            self._model, self._tokenizer = load(model_id)
            self.load_s = time.perf_counter() - t0
            self.model_id = model_id
            return self.load_s

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._tokenizer = None
            self.model_id = None
            try:
                import mlx.core as mx

                mx.metal.clear_cache()
            except Exception:  # noqa: BLE001
                pass

    def info(self) -> dict[str, Any]:
        peak = 0.0
        try:
            import mlx.core as mx

            peak = float(mx.get_peak_memory()) / (1024**3)
        except Exception:  # noqa: BLE001
            pass
        return {
            "name": "ozek",
            "backend": "mlx",
            "ready": self.ready,
            "model": self.model_id,
            "load_s": round(self.load_s, 4),
            "requests": self.requests,
            "peak_memory_gb": round(peak, 3),
        }

    def _build_prompt(
        self,
        prompt: str,
        *,
        system: Optional[str],
        enable_thinking: bool,
        messages: Optional[list[dict[str, str]]],
    ) -> str:
        assert self._tokenizer is not None
        tok = self._tokenizer
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

        if not hasattr(tok, "apply_chat_template"):
            return prompt
        try:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        system: Optional[str] = None,
        enable_thinking: bool = False,
        messages: Optional[list[dict[str, str]]] = None,
    ) -> GenerateResult:
        if not self.ready:
            raise RuntimeError("model not loaded; call load() first")

        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        with self._lock:
            text_prompt = self._build_prompt(
                prompt,
                system=system,
                enable_thinking=enable_thinking,
                messages=messages,
            )
            sampler = make_sampler(temp=temperature)
            pieces: list[str] = []
            first_token_s: Optional[float] = None
            last: Any = None
            t0 = time.perf_counter()
            for resp in stream_generate(
                self._model,
                self._tokenizer,
                prompt=text_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                if first_token_s is None:
                    first_token_s = time.perf_counter() - t0
                if resp.text:
                    pieces.append(resp.text)
                last = resp
                if resp.finish_reason is not None:
                    break
            wall_s = time.perf_counter() - t0
            self.requests += 1

        gen_tokens = int(getattr(last, "generation_tokens", 0) or 0)
        ttft = float(first_token_s or 0.0)
        decode_tps = 0.0
        if first_token_s is not None and gen_tokens > 1 and wall_s > first_token_s:
            decode_tps = (gen_tokens - 1) / (wall_s - first_token_s)

        stats = GenerateStats(
            model=self.model_id or "",
            prompt_tokens=int(getattr(last, "prompt_tokens", 0) or 0),
            generation_tokens=gen_tokens,
            ttft_s=round(ttft, 4),
            wall_s=round(wall_s, 4),
            prompt_tps=round(float(getattr(last, "prompt_tps", 0.0) or 0.0), 2),
            generation_tps=round(
                float(getattr(last, "generation_tps", 0.0) or 0.0), 2
            ),
            decode_tps_wall=round(decode_tps, 2),
            peak_memory_gb=round(
                float(getattr(last, "peak_memory", 0.0) or 0.0), 3
            ),
            enable_thinking=enable_thinking,
        )
        return GenerateResult(text="".join(pieces), stats=stats)

    def stream_tokens(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        system: Optional[str] = None,
        enable_thinking: bool = False,
        messages: Optional[list[dict[str, str]]] = None,
    ) -> Iterator[str]:
        """Yield text chunks (SSE). Holds the runtime lock for the whole stream."""
        if not self.ready:
            raise RuntimeError("model not loaded; call load() first")

        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        with self._lock:
            text_prompt = self._build_prompt(
                prompt,
                system=system,
                enable_thinking=enable_thinking,
                messages=messages,
            )
            sampler = make_sampler(temp=temperature)
            for resp in stream_generate(
                self._model,
                self._tokenizer,
                prompt=text_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                if resp.text:
                    yield resp.text
                if resp.finish_reason is not None:
                    break
            self.requests += 1


runtime = OzekMLXRuntime()
