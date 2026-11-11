<p align="center">
  <img src="assets/ozek-cover.png" alt="Ozek — a thin MLX runtime for local agents" width="920" />
</p>

# Ozek (өзек)

Kazakh **өзек** — core / stem / essence.

**A lightweight inference runtime for developers building local AI agents on Apple Silicon.**

One model slot. CLI + local HTTP. Not a chat app, not a model zoo.

On Apple Silicon, the heavy work already lives in MLX Metal kernels. Ozek is the thin Python layer agents actually need: load → generate → stream → metrics.

| | Typical local stacks | **Ozek** |
|---|---|---|
| Focus | chat UI, many models | Mac agents, tok/s |
| Stack | llama.cpp + product surface | Python + **MLX** |
| Shape | full app | library + small server |

## Status

**Alpha** · `0.1.0` — small, usable, still evolving. Keep it a sharp tool, not a platform.

## Requirements

- Apple Silicon Mac (M1 / M2 / M3 / M4)
- Python 3.11+
- [`mlx-lm`](https://github.com/ml-explore/mlx-lm)

## Install

```bash
git clone https://github.com/jaganov/ozek.git
cd ozek
python3 -m pip install -e .
```

## CLI

```bash
ozek generate \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --prompt "Explain KV cache in one sentence." \
  --max-tokens 64

ozek generate --prompt "Hi" --json          # include stats
ozek generate --prompt "..." --think        # Qwen3/3.5 CoT when supported
ozek info
```

## Python API

```python
from ozek import runtime

runtime.load("mlx-community/Qwen2.5-3B-Instruct-4bit")
result = runtime.generate("Say hi in one word.", max_tokens=16)
print(result.text)
print(result.stats.generation_tps, "tok/s")
```

## Server

Local agent HTTP (stdlib only). Default bind: `127.0.0.1:8787`.

```bash
ozek-server --model mlx-community/Qwen2.5-3B-Instruct-4bit --port 8787
```

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | liveness + ready |
| `GET` | `/v1/info` | loaded model / peak mem |
| `POST` | `/v1/generate` | `{ "prompt", "max_tokens", "temperature", "system?", "think?" }` |
| `POST` | `/v1/chat/completions` | OpenAI-compatible subset; `stream: true` → SSE |

```bash
curl -s http://127.0.0.1:8787/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Say hi in one word.","max_tokens":16}'
```

## Performance (Apple M2 Pro, 16 GB)

Single-stream, warm-ish runs on **one machine**. Quant / chat template / sampler defaults are **not** fully identical across stacks — treat these as orientation, not a lab paper.

Workload: short prompt, `max_tokens=64`, thinking **off**.  
Model family: **Qwen2.5-3B Instruct**, 4-bit / Q4-class.

| Runtime | Backend | generation tok/s | decode tok/s (wall) | TTFT | Peak mem |
|---|---|---:|---:|---:|---:|
| **Ozek** | MLX (`mlx-lm`) | **78.5** | **73.3** | 0.58 s | 1.93 GB |
| Ollama | llama.cpp Metal (`qwen2.5:3b`) | 70.4 | 67.2 | **0.30 s** | — |

Ozek leads on generation throughput here; **Ollama wins TTFT** on this 3B run (0.30 s vs 0.58 s).

Larger check (**Qwen3.5-9B**, 4-bit, thinking off):

| Runtime | generation tok/s | decode tok/s (wall) |
|---|---:|---:|
| **Ozek (MLX)** | **~35** | **~35** |
| Ollama | ~25 | ~26 |

For Mac agents that care about sustained tok/s, MLX is usually the fast path — and Ozek is that path with a small CLI/API.

## License

MIT — free to use, fork, and ship.
