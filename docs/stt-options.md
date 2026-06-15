# STT options — Nemotron-3.5-ASR-Streaming-0.6B vs Parakeet-TDT-0.6B-v3

> **Update (2026-06-15): a third model, Qwen3-ASR (Alibaba Qwen), was evaluated after this
> doc and is now the STT pick.** It serves *natively* on vLLM (unlike both models below), is
> Apache-2.0, and posts the strongest documented English WER — and it's VERIFIED end-to-end in
> [`../stt/qwen3_asr.py`](../stt/qwen3_asr.py). The Parakeet-vs-Nemotron comparison below stays
> valid as the NVIDIA-model analysis; a full three-way rewrite with Qwen3-ASR is still pending.

Evaluation of the two candidate ASR models for the **STT stage** of the rumble-ai voice
loop (kid-facing, Australian-English, `STT → LLM → TTS` on Modal, scale-to-zero). Same
spirit as [`tts-options.md`](tts-options.md): ranked comparison with sources and caveats.

- **Date:** 2026-06-15
- **Candidates:** `nvidia/nemotron-3.5-asr-streaming-0.6b`, `nvidia/parakeet-tdt-0.6b-v3`
- **Method:** multi-agent research + adversarial verification of the contested
  WER / latency / license / serving claims (verdicts inline below).

## TL;DR

**Adopt `nvidia/parakeet-tdt-0.6b-v3` for the STT stage. Confidence ~70%.**
It wins on the two things we can verify with high confidence — serving fit and maturity —
while Nemotron's one real edge (native streaming latency) is the claim that is **not
validated on our 24GB GPU tier**, and the loop already tolerates ~1s TTS time-to-first-audio.

> **Pivotal infra fact:** Today (2026-06-15) **neither model is servable on `vllm serve`
> as a transcription endpoint.** vLLM's native STT model list is Whisper, Voxtral, Gemma3n,
> and Qwen3-Omni — all encoder-decoder or decoder-only. Both candidates are NeMo
> RNN-T/TDT/FastConformer; the TDT transducer decode (LSTM prediction net + token-duration
> inner loop) doesn't fit vLLM's autoregressive "one token per decode step" contract. So
> unlike the LLM (Qwen3.5-4B) and TTS (OmniVoice) stages, **STT needs a non-vLLM serving
> path today.**
>
> **But Parakeet is closer to the vLLM ecosystem than Nemotron** — relevant because the rest
> of the stack is vLLM-standardized:
> - Merged: Parakeet's FastConformer *encoder* is already in vLLM as the audio front-end for
>   Nemotron-Nano-VL ([PR #35100](https://github.com/vllm-project/vllm/pull/35100),
>   [#40007](https://github.com/vllm-project/vllm/pull/40007),
>   [#36671](https://github.com/vllm-project/vllm/pull/36671)) — *encoder only, not standalone ASR.*
> - In progress: [PR #41708](https://github.com/vllm-project/vllm/pull/41708) (WIP, open) adds
>   real `/v1/audio/transcriptions` support for `parakeet-tdt-0.6b-v3`, closing
>   [issue #23943](https://github.com/vllm-project/vllm/issues/23943). **Caveat:** its approach
>   precomputes the full greedy TDT transcript and replays forced logits through vLLM's loop —
>   i.e. *offline* transcription via vLLM's API, **no streaming partials** even once merged.
> - Nemotron-3.5-ASR-streaming has no equivalent issue/PR/encoder support in vLLM.
>
> Net: if PR #41708 lands, Parakeet could eventually unify onto `vllm serve` (offline path);
> Nemotron has no such track. This is a tiebreaker *toward Parakeet*, not a reason to wait.

## Pure head-to-head

| Axis | `nemotron-3.5-asr-streaming-0.6b` | `parakeet-tdt-0.6b-v3` | Winner |
|---|---|---|---|
| **Latency / streaming** | Native cache-aware FastConformer-**RNNT**; runtime-selectable 80/160/320/560/1120 ms chunks via `att_context_size`, switchable without retraining; each frame processed once. H100: 240 streams @80ms, 2,400 @1.12s. | **This checkpoint** ships a **buffered** chunked pipeline (`chunk_secs=2`, ~2s right-context, ~46% rel. WER hit: 9.22% vs 6.32%). Headline RTFx **3,332.74 is offline batch-64 on A100-80GB**, not streaming latency. | **Nemotron** |
| **Accuracy / WER** | Streaming-only FLEURS-en **9.43% @80ms → 7.91% @1.12s**; **no** offline / Open-ASR-Leaderboard number. 40 locales (incl. en-US, en-GB). | Offline Open-ASR avg **~6.34%** (paper 6.32%), LibriSpeech **1.93% / 3.59%**, FLEURS-en **4.85%**. 25 European langs + auto-LID. | **Parakeet** |
| **Infra / serving** | **No** `transformers` AutoModel → default is heavy `nemo_toolkit[asr]` + **DIY WebSocket** server (community ONNX-int4 + sherpa-onnx is brand-new). | **Native** `transformers` v5.12 (`ParakeetForTDT` / ASR pipeline) → serve **without NeMo**; ONNX exports + a community **OpenAI `/v1/audio/transcriptions`** FastAPI server exist. | **Parakeet** |
| **License** | **OpenMDW-1.1** (LF permissive; commercial OK, notice-retention only, **no attribution**). | **CC-BY-4.0** (commercial OK, **attribution required**: credit + license link + indicate changes). | **Nemotron** (lighter obligation) |
| **Maturity** | Released **~2026-06-04** (~10 days old at eval); serving recipes unproven. | Released **2025-08-14**; leaderboard-proven, broad ecosystem. | **Parakeet** |
| **Size / VRAM** | ~600M, 2.37GB `.nemo`; fits 24GB with headroom. | ~600M, 2.51GB; loads ~2GB; validated L4/A10G/T4. | Tie |

**General-purpose verdict:** Parakeet-TDT-0.6B-v3 is the stronger model overall (accuracy +
serving ease), ties on size; Nemotron is the better pick *only* when sub-second,
natively-streaming partials and 40-locale breadth are the dominant requirement.

### Verification notes (what the adversarial pass changed)

- **Nemotron "sub-100ms time-to-final" → partly-true.** That is a *chunk-mode label*, not a
  measured wall-clock. The nearest independent figure (~70 ms, Artificial Analysis) is for
  the **English-only Nemotron 3 predecessor** on H100/L40S; **no source validates it on a
  24GB L4/A10G**.
- **"Parakeet can only do buffered streaming" → refuted (at the family level).** Cache-aware
  streaming is a NeMo/FastConformer family feature and sibling Parakeet streaming checkpoints
  exist. But **this exact v3 checkpoint** ships only the buffered pipeline, so *out of the
  box* it is weaker for a snappy turn.
- **Both NeMo; neither is `vllm serve`-able for transcription today → confirmed, with nuance.**
  vLLM native STT = Whisper / Voxtral / Gemma3n / Qwen3-Omni (not "only Whisper/Voxtral" as an
  earlier draft said). Parakeet has merged *encoder* support (in Nemotron-Nano-VL) + an active
  WIP transcription PR ([#41708](https://github.com/vllm-project/vllm/pull/41708), offline-only
  approach); Nemotron-streaming has none. See the "Pivotal infra fact" callout above.
- **Neither has any published Australian-accent or child-speech WER → confirmed.** Nemotron
  exposes en-GB as a locale; Parakeet does not break out accents. Accuracy rankings above are
  adult/read/European benchmarks and are **not trustworthy for the target population.**

## Project fit (Modal + scale-to-zero + Qwen3.5-4B + OmniVoice)

| Criterion (project weight) | Nemotron-3.5-ASR | Parakeet-TDT-0.6b-v3 |
|---|---|---|
| **License — kids commercial** | **Edge.** OpenMDW-1.1: royalty-free commercial, no attribution, outputs unrestricted, no minors carve-out. Newer LF license; not Apache-2.0. | OK. CC-BY-4.0: commercial allowed but **attribution is a permanent compliance task** on a shipped product. Not Apache-2.0. |
| **AU-English / child robustness** | **Unverified** (en-GB proxy, no en-AU, no child eval). | **Unverified** (generic English, no en-AU, no child eval). |
| **Streaming / low-latency turn** | **Strong native spec** (80ms floor). *Caveat: sub-100ms is a mode label, no 24GB validation.* | Capable but **this v3 ships buffered** (~2s lookahead). Cache-aware Parakeet siblings exist. |
| **Modal / NeMo serving effort + cold start** | **Heaviest.** Full `nemo_toolkit`, big image, slower cold start; ready-made servers are **WebSocket-only** (no OpenAI-REST drop-in); per-stream session state fights scale-to-zero. | **Easiest.** Native `transformers`, smaller image, faster cold start; community **OpenAI `/v1/audio/transcriptions`** server; **stateless** request/response maps cleanly onto scale-to-zero. |
| **GPU / cost fit (24GB, scale-to-zero)** | Good (~600M, fits L4/A10G); vendor concurrency is H100-only, 24GB latency uninferred. | Good (~600M, ~2GB load, validated L4/A10G/T4); lighter runtime aids cold start. |

### Deciding factors

1. **Serving path (decisive → Parakeet).** Neither is `vllm serve`-able for transcription
   today, but Parakeet's native `transformers` integration + existing OpenAI-compatible server
   means a thin image (no `nemo_toolkit`), faster cold start, an HTTP shape that mirrors the
   existing stages, and **stateless** behaviour that fits scale-to-zero. Nemotron forces the
   full NeMo stack plus a self-built WebSocket layer with per-stream session state. Parakeet
   also has a live (offline-only) vLLM transcription PR in flight ([#41708](https://github.com/vllm-project/vllm/pull/41708));
   Nemotron has none — so Parakeet has the only credible future path back onto the project's
   vLLM standard.
2. **Latency (spec → Nemotron, but caveated).** Nemotron's native streaming is the cleaner
   spec, but the headline latency is unproven on cheap 24GB GPUs, and the loop already
   tolerates ~1s TTS TTFA — so a buffered few-hundred-ms STT is likely acceptable.
3. **License (soft edge → Nemotron).** Both commercial-usable; neither is Apache-2.0.
   OpenMDW has no attribution requirement; CC-BY does. Not a decider on its own.

### Recommendation

**Adopt `nvidia/parakeet-tdt-0.6b-v3`.** The high-confidence, verifiable factors (serving
fit + maturity) favor it; Nemotron's lone advantage is unvalidated on our tier.

**Conditions that flip the decision to Nemotron:**
1. A measured-latency requirement that buffered Parakeet can't meet on L4/A10G **and**
   native Nemotron clearly hits — *but first try a cache-aware Parakeet streaming sibling*
   (`parakeet_realtime_eou_120m-v1` / `multitalker-parakeet-streaming-0.6b-v1`) which keeps
   the Parakeet lineage while adding native streaming + EOU endpointing.
2. Legal rejects CC-BY-4.0 attribution but accepts OpenMDW-1.1.

**Hard gate before committing either way:** run an **in-domain eval on Australian children's
speech** — both models' AU-accent and child-speech robustness are unverified, and that is the
project's #2 hard preference.

## How to serve Parakeet v3 on Modal (sketch)

Fits the existing one-app-per-model + scale-to-zero + cheap-24GB pattern:

- **One Modal app, `stt/parakeet.py`.** Base image `nvidia/cuda:12.8.1-devel-ubuntu22.04`,
  `uv_pip_install` of `transformers>=5.12.0` (avoid yanked 5.10.0) + `torch` + audio libs
  (or `onnxruntime-gpu` for the lighter ONNX path). Add a `from_pretrained` **smoke test at
  image build** to catch checkpoint/load issues early. Cache the ~2.5GB weights on a named
  **Volume** with `hf_transfer` so cold starts skip the download.
- **GPU tier: L4 or A10G (24GB)** — same cheap tier as the LLM/OmniVoice apps; no H100.
- **Serving path: thin FastAPI exposing `/v1/audio/transcriptions`** (OpenAI-compatible).
  Reuse/fork the community `parakeet-tdt-0.6b-v3` FastAPI/ONNX server rather than hand-rolling
  — but **vet any third-party server before shipping in a kids' product**. No `nemo_toolkit`
  in the image; this mirrors the project's other OpenAI-HTTP endpoints.
- **Streaming approach:** start with **stateless per-utterance transcription** (client-side or
  lightweight server-side VAD/endpointing → send each finalized utterance). Maps cleanly onto
  Modal request/response + scale-to-zero with no WebSocket session state — lowest-risk ship.
  If turn latency proves too high, upgrade **in place** to a cache-aware Parakeet streaming
  sibling over the NeMo streaming path before considering Nemotron.
- **Scale-to-zero:** keep the 2–5 min scaledown window; stateless HTTP makes idle-to-zero and
  cold resume clean. Budget a longer first-request cold start than the vLLM apps (the
  `transformers`/NeMo image is heavier than pure vLLM); the Volume-cached weights keep it
  tolerable.

## Sources & caveats

- HF model cards: `nvidia/parakeet-tdt-0.6b-v3`, `nvidia/nemotron-3.5-asr-streaming-0.6b`;
  HF Open ASR Leaderboard; `transformers` v5.12 ASR docs.
- Papers: arXiv:2509.14128 (Parakeet/Canary multilingual), arXiv:2510.06961 (RTFx batch
  throughput), arXiv:2604.14493 (streaming WER deltas).
- **Unverified for our use case:** AU-accent and child-speech accuracy for *both* models; the
  community OpenAI-compatible Parakeet server was not deployment-tested here; exact 24GB-tier
  streaming latency for Nemotron has no independent measurement. Re-run an in-domain benchmark
  before final commitment.
