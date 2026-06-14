# Fish Speech S2 Pro — Performance Baseline

Initial baseline for the Modal + vLLM-Omni TTS endpoint, captured right after the
first verified-working deploy. Use this as the reference point when changing the
GPU, the vLLM/vllm-omni versions, batching, or warm-pool settings.

- **Date:** 2026-06-14
- **Endpoint:** `https://huygbui--fish-s2-pro-tts-serve.modal.run` (`POST /v1/audio/speech`)
- **Benchmark script:** [`bench_tts.py`](./bench_tts.py) — re-run to compare against this baseline.

## Test environment (deployed config at time of test)

| Item | Value |
|---|---|
| Model | `fishaudio/s2-pro` (Dual-AR TTS + DAC codec, 44.1 kHz mono) |
| vLLM | `0.19.0` |
| vllm-omni | git commit `c93359bb354a6aa5c14d062430cb85b2c4db251e` (recipe-pinned) |
| torch | `2.10.0` (CUDA) |
| GPU | `H100:1` (80 GB) |
| Per-container concurrency | `@modal.concurrent(max_inputs=8)` |
| Warm pool | `min_containers=0` (scales to zero after `scaledown_window=15 min`) |
| Output format | `wav` (44.1 kHz mono, 16-bit PCM) |
| Request seed | `58842` (deterministic voice) |

Resource usage observed at load: model weights ~**10.9 GiB**; KV cache **47.5 GiB
/ 345,936 tokens** (~**21× concurrency headroom** for 16,384-token requests).

## Methodology

`bench_tts.py` sends non-streaming `POST /v1/audio/speech` requests (`voice=default`,
`response_format=wav`, `seed=58842`) and measures wall-clock latency. Audio duration
is read from the returned WAV header; **RTF = latency / audio_seconds** (lower is
better; <1.0 = faster than real time). Warm rows are the **median of 3 runs**. Text
sizes: short ≈ 7 words / 37 chars, medium ≈ 28 words / 158 chars, long ≈ 84 words /
477 chars.

## Results

| Scenario | Latency | Audio | RTF | Notes |
|---|---:|---:|---:|---|
| **Cold start** (1st request after scale-to-zero) | **209.84 s** | 2.97 s | — | container spin-up + model load + dual-engine init (weights & torch.compile cached) |
| Warm — short (~7 words) | 5.21 s | 2.79 s | 1.87 | fixed per-request overhead dominates |
| Warm — medium (~28 words) | 5.65 s | 10.63 s | **0.53** | ~2× faster than real time |
| Warm — long (~84 words) | 10.23 s | 19.13 s | **0.53** | scales linearly; no truncation observed up to ~19 s audio |
| Streaming (`stream=true`) | total 4.34 s | — | — | **time-to-first-audio ≈ 1.08 s** |
| 8 concurrent short (1 warm container) | wall **5.17 s** | — | — | per-req: min 3.14 s / median 4.25 s / max 5.16 s |

**Throughput (8 concurrent, batched):** ~**1.55 req/s** vs ~**0.24 req/s** sequential
→ continuous batching gives ~**6.5×** on short clips. 8 concurrent requests complete
in roughly the wall-time of a single request.

## Key takeaways

- **Warm RTF ≈ 0.53** for normal-length text (generates ~2× faster than real time).
- **~4–5 s fixed per-request overhead** (orchestrator + 2-stage pipeline) → short
  clips look slow by RTF; prefer **streaming** for interactive UX (first audio ≈ 1 s).
- **Batching is the strength**: up to 8 simultaneous requests are absorbed with
  negligible added latency. Rule of thumb: **1 warm H100 ≈ 8 concurrent requests ≈
  ~1.5 short-req/s** (much higher measured in audio-seconds/s for longer clips).
- **Cold start ≈ 3.5 min** is the dominant latency risk while `min_containers=0`.

## Production limits (as configured)

1. **Cold start ~3.5 min** after 15 min idle (`min_containers=0`). Fix for
   latency-sensitive use: `min_containers≥1` (cost: an always-on H100).
2. **Concurrency ceiling = 8 per container.** The 9th+ concurrent request triggers
   Modal autoscale → a **new container cold-starts (~3.5 min)** → burst tail latency.
   Provision ~⌈peak_concurrency / 8⌉ warm containers.
3. **No authentication** — the URL is public; anyone with it can use the GPU. Add
   token/proxy auth before production exposure.
4. **Cost** — H100 80 GB ≈ $3–4/hr; always-warm ≈ $2–3k/month. Model needs only
   ~11 GiB + KV cache, so a smaller 40–48 GB GPU (A100-40G / L40S) may fit and cut
   cost — untested.
5. **Long-input truncation** (vllm-omni [#2248](https://github.com/vllm-project/vllm-omni/issues/2248))
   — not hit up to ~19 s audio here, but chunk very long inputs client-side.
6. **No rate limiting / retries / multi-region / metrics** — single region, single
   GPU type.

## Caveats / known measurement artifacts

- The streaming row's reported audio duration was a parse artifact (the streamed
  body isn't a finalized WAV container, so the WAV-header duration was bogus). The
  meaningful streaming metric is **time-to-first-audio ≈ 1.08 s**; total latency 4.34 s.
- Single-run cold start (one sample); cold-start time varies with image-pull / load.
- Numbers are for `response_format=wav`, `seed=58842`, default voice, no voice cloning.

## How to reproduce

```bash
export TTS_URL="https://huygbui--fish-s2-pro-tts-serve.modal.run"
uv run python bench_tts.py
```
