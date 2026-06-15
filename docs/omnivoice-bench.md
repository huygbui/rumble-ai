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
2. True server-side streaming is architecturally unavailable; client-side clause chunking
   (`app.cli.say`) is the TTFA lever.
3. The real server-side latency knob is the diffusion `num_step` value (32 -> 24 -> 16),
   with a quality tradeoff.

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

### Client-side sentence chunking — adopted (`app.cli.say`)

Server-side streaming barely helps here. `app.cli.say` instead splits the reply into
clauses and synthesizes them pipelined: first audio plays while the rest generate.

| approach | TTFA (first audio) |
|---|---:|
| single-shot (whole utterance) | 3.30 s and grows with length |
| server streaming | 3.22 s |
| **chunked (`app.cli.say`)** | **2.21 s**, roughly flat regardless of reply length |

Run:

```bash
export TTS_URL="https://huygbui--omnivoice-tts-serve.modal.run"
TTS_MODEL=omnivoice BENCH_QUALITY=1 uv run python bench.py

echo "..." | TTS_MODEL=omnivoice python -m app.cli.say
```
