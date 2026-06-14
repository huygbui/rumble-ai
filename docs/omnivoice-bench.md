# OmniVoice benchmark & head-to-head vs Fish S2 Pro

_Measured 2026-06-14 against the live Modal endpoint
`https://huygbui--omnivoice-tts-serve.modal.run`, via `bench.py`
(`TTS_MODEL=omnivoice BENCH_QUALITY=1`). OmniVoice voice via DESIGN —
`instructions="child, australian accent, high pitch"`, no reference clip._

| | OmniVoice (candidate) | Fish S2 Pro (baseline) |
|---|---|---|
| GPU | **A10G (24 GB)** | H100 (80 GB) |
| Stack | vLLM 0.23.0 + vllm-omni 0.23.0rc1 | vLLM 0.22.1 + vllm-omni 0.22.0 |
| Output | 24 kHz mono | 44.1 kHz mono |
| Voice path | voice-design (`instructions`) | registered/cloned reference |

Fish numbers are the documented current-stack figures from
[`fish-cold-start.md`](./fish-cold-start.md) (concurrency row is the older 0.19.0
[`perf-baseline.md`](./perf-baseline.md)). **Fish was not re-run** — this spends only the
cheap A10G, not the H100. Cold-start and streaming rows are single samples; warm rows are
the median of 3.

## 1. Latency / throughput

| Scenario | OmniVoice | Fish | Read |
|---|---|---|---|
| **Cold start** (scaled-to-zero → 1st req) | **63.7 s** | ~150–220 s | OmniVoice ~3× faster cold start, on a ~10× cheaper GPU |
| Warm — short (~7 w, 2.2 s audio) | **2.66 s** (RTF 1.21) | 3.93 s | OmniVoice faster despite the weaker card (tiny model) |
| Warm — medium (~28 w, 9.6 s audio) | 4.77 s (**RTF 0.50**) | RTF 0.40–0.55 | comparable; ~2× faster than real-time |
| Warm — long (~84 w, 28.8 s audio) | 11.0 s (**RTF 0.38**) | RTF ~0.53 | comparable; **no truncation** at ~29 s |
| **Streaming TTFA** | **3.22 s** | **1.16 s** | ⚠ OmniVoice's weak spot (see below) |
| 8 concurrent (1 container) | wall 9.48 s · **0.84 req/s** | wall 5.17 s · 1.55 req/s¹ | both batch ~5×; Fish higher abs. throughput on pricier HW |

¹ Fish concurrency is the 0.19.0-era baseline (not re-measured on the current stack).

**Takeaways.** OmniVoice matches or beats Fish on cold start and warm single-stream
latency at a fraction of the GPU cost. The one real regression is **streaming
time-to-first-audio (~3.2 s vs Fish's ~1.2 s)** — and streaming barely helps (3.22 s TTFA
vs 4.77 s full), so the two-stage eager codec seems to emit audio late. For a
back-and-forth kid chat, **TTFA is the metric to watch** — it's what makes a reply feel
instant.

## 2. Quality (A/B)

Same 5 texts on both models, side by side in `../out/`: `NN_*.wav` (Fish) vs
`omnivoice_NN_*.wav` (OmniVoice). Objective sanity stats (catch gross failures; naturalness
still needs ears):

| Clip | OmniVoice (24 kHz) | Fish (44.1 kHz) |
|---|---|---|
| 01 short greeting | 4.36 s · peak 0.91 · rms 0.12 · clip 0.0% | 5.20 s · 0.89 · 0.17 · 0.0% |
| 02 medium narrative | 10.44 s · 0.85 · 0.11 · 0.0% | 11.89 s · 0.84 · 0.15 · 0.0% |
| 03 long form | 26.68 s · 1.00 · 0.11 · 0.0% | 31.11 s · 0.80 · 0.13 · 0.0% |
| 04 emotion tags | 10.00 s · 0.87 · 0.11 · 0.0% | 10.96 s · 0.89 · 0.16 · 0.0% |
| 05 numbers/prosody | 11.72 s · 0.95 · 0.22 · 0.0% | 16.16 s · 0.66 · 0.13 · 0.0% |

- **Both clean** — 0% clipping, healthy RMS, no artifact flags. This confirms the 0.23
  stack fixed the broadband-artifact failure the 0.22.0 OmniVoice path hit.
- **Fish outputs 44.1 kHz vs OmniVoice's 24 kHz** — Fish has ~2× the audio bandwidth
  (crisper highs). 24 kHz is fine for speech intelligibility but less "airy".
- **Subjective calls are yours** (I can't listen): naturalness, and whether the named
  `australian accent` + `child` design actually sounds good and Australian. Per
  [`tts-options.md`](./tts-options.md), AU fidelity is the #1 unknown — **ideally judged by
  an Australian listener**. 04 also tests how OmniVoice handles Fish-style tags
  (`[laughing]`/`[whispering]`/`[angry]`) it doesn't natively use (its tag is `[laughter]`).

## 3. Caching / "as performant as it can be"

- **Weights cache works.** ~64 s cold start is consistent with weights loading from the
  `huggingface-cache` volume — a fresh ~3.25 GB download (2.45 GB model + 0.8 GB codec)
  would inflate it noticeably. No re-download per cold start.
- **No compile cache — by design.** OmniVoice's model-type deploy config forces
  `float32` + `enforce_eager`, so there's no torch.compile / CUDA-graph step (unlike Fish,
  which reloads compiled graphs in ~1.8 s). Upside: zero compile-cache fragility. Downside:
  it forgoes CUDA-graph speedups.
- **Scale-to-zero works** (5-min window); the follow-up run hit a still-warm container.

**Levers for more performance (untested, quality-risk):**
1. **`float32` → `bf16`** is the biggest one — could roughly halve warm latency and ~2× the
   concurrency, *if* OmniVoice/codec are validated in bf16. The deploy config pins fp32 and
   says don't override, so this needs a careful quality check, not a blind flip.
2. **Disable `enforce_eager`** for CUDA graphs — constrained by the custom modeling code;
   may not be safe.
3. **Streaming TTFA** is the highest-value fix for conversational use — check whether
   vllm-omni's OmniVoice path can emit earlier chunks; otherwise keep utterances short.
4. A10G is a sensible cost tier; L40S would be faster, L4 cheaper.

## Verdict

OmniVoice is a **strong, much cheaper** candidate: competitive-to-better latency than Fish
at ~10× lower GPU cost, clean audio, and (uniquely) a native AU-accent + child design
control. Open items before committing:
1. **Listen to the A/B** (`out/*` vs `out/omnivoice_*`), ideally with an AU listener — does
   the designed voice sound good and Australian?
2. **Decide if ~3.2 s streaming TTFA is acceptable** for the chat UX, or needs work.
3. Consider testing **zero-shot cloning** of a consented AU clip — likely a more consistent
   AU voice than the design attribute, and an easy add (`REF_AUDIO`+`REF_TEXT`).

## Tuning experiments (2026-06-14)

Two levers from §3 were tried live on the A10G endpoint.

### bf16 instead of fp32 — NO WIN, kept fp32

A/B of a `--dtype bfloat16` variant (a separate `omnivoice-tts-bfloat16` app) vs the
validated fp32 default:

| | fp32 (default) | bf16 (experiment) |
|---|---|---|
| Cold start | 63.7 s | 67.4 s |
| Warm short | **2.66 s** | 2.84 s |
| Warm medium | **4.77 s** | 5.39 s |
| Warm long | **11.0 s** | 11.9 s |
| Streaming TTFA | 3.22 s | 3.55 s |
| 8-concurrent | 0.84 req/s | 0.83 req/s |
| Audio (clip% / artifacts) | clean | clean (clip 04 slightly lower energy) |

bf16 was **marginally slower** with **no concurrency gain**. Why: OmniVoice runs
`enforce_eager` (no CUDA graphs) on a tiny 613M model, so latency is **overhead-bound — the
~2.5 s per-request 2-stage/codec orchestration — not compute-bound**. Lowering matmul
precision can't speed up an overhead-dominated workload, and it adds quality risk. **Kept
fp32.** The only remaining raw-perf avenue is CUDA graphs (disabling `enforce_eager`), which
the model's deploy config forces off — not pursued. To retry: re-add `--dtype` to
`tts/omnivoice.py`'s serve cmd and `modal deploy` (the experiment lever was reverted to keep
the serving file clean).

### Client-side sentence chunking — ADOPTED (`say.py`)

Server-side streaming barely helps here (TTFA ~3.2 s). `say.py` instead splits the reply
into clauses and synthesizes them **pipelined** — first audio plays while the rest generate:

| approach | TTFA (first audio) |
|---|---|
| single-shot (whole utterance) | 3.30 s — and grows with length (5.4 s medium, 11.9 s long) |
| server streaming | 3.22 s |
| **chunked (`say.py`)** | **2.21 s** — ≈ one short-clause synth; ~flat regardless of reply length |

The win grows with reply length: chunked TTFA stays ~2.2 s while single-shot scales with
text. The ~2.2 s floor is the per-request fixed overhead (can't go lower without true
server streaming, which this model lacks). This is the responsiveness pattern for the
eventual `chat.py` loop. Run: `echo "..." | TTS_URL=… TTS_MODEL=omnivoice python say.py`
(`PLAY=1` to hear it, `COMPARE=1` to also time single-shot).

## Reproduce

```bash
export TTS_URL="https://huygbui--omnivoice-tts-serve.modal.run"
TTS_MODEL=omnivoice BENCH_QUALITY=1 uv run python bench.py
# samples only (skip perf):  BENCH_PERF=0 BENCH_QUALITY=1 uv run python bench.py
```
