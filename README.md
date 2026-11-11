# Ozek (өзек)

Kazakh **өзек** — core / stem / essence.

**Thin Mac-first MLX runtime for local agents.** One model slot. CLI + local HTTP. Not another chat app.

On Apple Silicon, the heavy work already lives in MLX Metal kernels. Ozek is the thin Python layer agents actually need: load → generate → stream → metrics.

| | Ollama / LM Studio | **Ozek** |
|---|---|---|
| Focus | everyone, every model | Mac agents, tok/s |
| Stack | llama.cpp + UI / zoo | Python + **MLX** |
| Shape | product surface | library + small server |

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

Workload: same short prompt, `max_tokens=64`, generation with thinking **off**.

Model family: **Qwen2.5-3B Instruct**, 4-bit / Q4-class.

| Runtime | Backend | generation tok/s | decode tok/s (wall) | TTFT | Peak mem |
|---|---|---:|---:|---:|---:|
| **Ozek** | MLX (`mlx-lm`) | **78.5** | **73.3** | 0.58 s | 1.93 GB |
| Ollama | llama.cpp Metal (`qwen2.5:3b`) | 70.4 | 67.2 | 0.30 s | — |

Larger check (**Qwen3.5-9B**, 4-bit, thinking off):

| Runtime | generation tok/s | decode tok/s (wall) |
|---|---:|---:|
| **Ozek (MLX)** | **~35** | **~35** |
| Ollama | ~25 | ~26 |

Numbers are single-stream, warm-ish runs on one machine — orientation, not a lab paper. Quant / chat template / sampler defaults are not byte-identical across stacks. The point stands: **for Mac agents, MLX is usually the fast path, and Ozek is that path with a small CLI/API.**

## License

MIT — free to use, fork, and ship.
