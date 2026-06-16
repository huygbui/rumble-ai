# OmniVoice benchmark

_Measured 2026-06-14 against the live Modal endpoint
`https://huygbui--omnivoice-tts-serve.modal.run`, via `bench.py`
(`TTS_MODEL=omnivoice BENCH_QUALITY=1`). OmniVoice voice via DESIGN —
`instructions="child, australian accent, high pitch"`, no reference clip._

| Item | Value |
|---|---|
| GPU | A10G (24 GB) |
| Stack | vLLM 0.23.0 + vllm-omni 0.23.0rc1 |
| Output | 24 kHz mono |
| Voice path | voice-design (`instructions`) |

Cold-start and streaming rows are single samples; warm rows are the median of 3.

## 1. Latency / throughput

> **Historical — superseded by [§1b (re-measured 2026-06-16)](#1b-re-measured-2026-06-16--gpu--num_step-ab).**
> The current stack is ~2× faster than the table below.

| Scenario | OmniVoice | Read |
|---|---:|---|
| **Cold start** (scaled-to-zero -> 1st req) | **63.7 s** | weights load from cache on A10G |
| Warm — short (~7 w, 2.2 s audio) | **2.66 s** (RTF 1.21) | fixed overhead dominates |
| Warm — medium (~28 w, 9.6 s audio) | 4.77 s (**RTF 0.50**) | ~2x faster than real-time |
| Warm — long (~84 w, 28.8 s audio) | 11.0 s (**RTF 0.38**) | no truncation at ~29 s |
| **"Streaming" TTFA** | **3.22 s** | buffered full synth, not true streaming |
| 8 concurrent (1 container) | wall 9.48 s · **0.84 req/s** | batches cleanly |

**Correction (2026-06-14, post multi-agent review):** OmniVoice is a single-stage,
non-autoregressive masked-diffusion model. It unmasks the whole acoustic-token grid over
~32 diffusion steps, then runs one DAC codec decode. Consequences:

1. bf16 gave no win because the loop is graph-bound, not FLOP-bound.
2. True server-side streaming is architecturally unavailable; app-side clause chunking is
   the TTFA lever.
3. The real server-side latency knob is the diffusion `num_step` value (32 -> 24 -> 16),
   with a quality tradeoff. _(Update: §1b found this is **not** a measurable latency lever
   on the current stack — see below.)_

## 1b. Re-measured 2026-06-16 — GPU × num_step A/B

§1's numbers are stale. Re-running warm (median-of-3, on a clean pass *after* the cold-start
settles) against the current stack, plus a four-corner sweep of **{A10G, L40S} × {32, 24}
diffusion steps**, settles the GPU-tier and `num_step` questions.

| Warm, median-of-3 | A10G/32 (live) | A10G/24 | L40S/32 | L40S/24 |
|---|---:|---:|---:|---:|
| short (2.2 s audio) — **TTFA-relevant** | 1.14 s | 1.04 s | 1.39 s | 1.11 s |
| medium (9.6 s audio) | 2.28 s | 2.24 s | 1.48 s | 1.22 s |
| long (28.8 s audio) | 6.78 s | 6.74 s | 2.88 s | 2.86 s |
| cold start | ~12 s | ~119 s\* | ~106 s\* | ~110 s\* |
| Modal $/hr (approx) | ~1.10 | ~1.10 | ~1.95 | ~1.95 |

\* first-deploy cold start; the live A10G/32 was already warm.

**Findings:**

1. **Current A10G/32 is ~2× faster than §1** (warm short 1.14 s vs 2.66 s; cold ~12 s vs
   63.7 s). The stack improved since 2026-06-14 — treat §1 as historical.
2. **num_step 32→24 is not a latency lever.** On the most diffusion-bound metric (long) it is
   flat on both GPUs (A10G 6.78→6.74 s; L40S 2.88→2.86 s). Step count is not the wall-clock
   bottleneck; fixed pipeline cost dominates (consistent with "graph-bound, not FLOP-bound").
   Because of that, latency can't even confirm the override alters output — it may be quality-
   only or a no-op; verify by ear if ever used. `num_step` is **deploy-time** (HF config via
   `--hf-overrides`), **not** a request field: a body `num_step` is silently ignored.
3. **L40S helps only the long tail, never TTFA.** vs A10G/32: long 2.4× faster, medium 1.5×,
   but short is slightly *slower* (1.14→1.39 s) — short clauses are fixed-overhead bound, so a
   faster GPU can't help them. Cold start is ~10× worse (~106 s). Worth the ~$0.85/hr premium
   only for long single-shot utterances or high concurrency, not for chat TTFA.

**Verdict: stay on A10G/32, 32 steps.** Neither knob improves perceived (TTFA) latency, and
app-side synth-ahead (`tts_concurrency`) already hides the long-clause region where L40S would
help. Variance is high (single warm pass per corner: treat ±0.5 s short/medium, ±2 s long as
noise). Harness + full runbook to re-run this lives on branch `perf/tts-numstep-l40s-bench`
(`tts/omnivoice.py` env-parameterized by `OMNI_GPU` / `OMNI_NUM_STEP`;
`docs/tts-numstep-l40s-bench.md`).

## 2. Quality samples

Same 5 texts were generated into `../out/` as `omnivoice_NN_*.wav`. Objective sanity stats
catch gross failures; naturalness still needs ears.

| Clip | OmniVoice (24 kHz) |
|---|---:|
| 01 short greeting | 4.36 s · peak 0.91 · rms 0.12 · clip 0.0% |
| 02 medium narrative | 10.44 s · peak 0.85 · rms 0.11 · clip 0.0% |
| 03 long form | 26.68 s · peak 1.00 · rms 0.11 · clip 0.0% |
| 04 emotion tags | 10.00 s · peak 0.87 · rms 0.11 · clip 0.0% |
| 05 numbers/prosody | 11.72 s · peak 0.95 · rms 0.22 · clip 0.0% |

- Audio was clean: 0% clipping and no artifact flags.
- Subjective calls still need listening: naturalness, and whether the named
  `australian accent` + `child` design sounds good and Australian. Per
  [`tts-options.md`](./tts-options.md), AU fidelity is the highest-priority unknown and
  should be judged by Australian listeners.

## 3. Caching / performance notes

- **Weights cache works.** ~64 s cold start is consistent with weights loading from the
  `huggingface-cache` volume. A fresh ~3.25 GB download would inflate it noticeably.
- **No torch.compile cache needed.** OmniVoice's model-type deploy config forces `float32`
  + `enforce_eager` for the unused AR engine, while the diffusion loop is already
  CUDA-graphed by default.
- **Scale-to-zero works** with the 5-minute window used during this run.

Levers:

1. `float32` -> `bf16`: tested, no win; kept fp32.
2. Disable `enforce_eager`: dead end for this model; the relevant diffusion loop is already
   graphed.
3. Server-side streaming: not available for the NAR diffusion architecture; keep
   client-side chunking.
4. `num_step` (32 -> 24 -> 16): worth testing if TTFA needs to come down, with quality
   checks.
5. A10G is a sensible cost tier; L40S would be faster, L4 cheaper.

## Verdict

OmniVoice is a strong default for this repo: low GPU footprint, clean audio, Apache-2.0
licensing, and a native AU-accent + child design control. Open items before committing:

1. Listen to the generated samples, ideally with an Australian listener.
2. Decide whether client-chunked TTFA around ~2.2 s is acceptable for the chat UX.
3. Test zero-shot cloning of a consented AU clip if the design attribute is not consistent
   enough.

## Tuning experiments (2026-06-14)

### bf16 instead of fp32 — no win, kept fp32

A/B of a `--dtype bfloat16` variant (a separate `omnivoice-tts-bfloat16` app) vs the
validated fp32 default:

| | fp32 (default) | bf16 (experiment) |
|---|---:|---:|
| Cold start | 63.7 s | 67.4 s |
| Warm short | **2.66 s** | 2.84 s |
| Warm medium | **4.77 s** | 5.39 s |
| Warm long | **11.0 s** | 11.9 s |
| Streaming TTFA | 3.22 s | 3.55 s |
| 8-concurrent | 0.84 req/s | 0.83 req/s |
| Audio | clean | clean |

bf16 was marginally slower with no concurrency gain. The DAC codec is hard-coded fp32
anyway, so this stays fp32.

### App-side sentence chunking — adopted

Server-side streaming barely helps here. The app splits replies into clauses and
synthesizes them pipelined: first audio plays while the rest generate.

| approach | TTFA (first audio) |
|---|---:|
| single-shot (whole utterance) | 3.30 s and grows with length |
| server streaming | 3.22 s |
| **chunked app pipeline** | **2.21 s**, roughly flat regardless of reply length |

Run:

```bash
export TTS_URL="https://huygbui--omnivoice-tts-serve.modal.run"
TTS_MODEL=omnivoice BENCH_QUALITY=1 uv run python bench.py
```
