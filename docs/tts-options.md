# Audio conversation: TTS / S2S model options research

_Research date: 2026-06-14. Goal: pick the next speech model(s) to try for a natural,
safe, **kid-facing** conversation experience, focused on **English (preferably
Australian English)**, where ease of **self-hosting**, a **trusted provider**, and a
viable **commercial-license** path (open or by contact) are all pluses. Speech-to-speech
(S2S) and text-to-speech (TTS) are both in scope._

This was produced by a multi-agent research workflow: one researcher per model gathered
primary-source facts, an independent fact-checker re-verified the three highest-stakes
claims for each (license / commercial terms, weights availability, self-host
feasibility), and a synthesis stage ranked everything against the criteria above. Where
a researcher and verifier disagreed, the verifier's reading is used here.

Current repo default: **OmniVoice** (`k2-fsa/OmniVoice`) served on Modal via vLLM-Omni.
Qwen3-TTS is kept as the heavier comparison path. See `tts/omnivoice.py`,
`tts/qwen3.py`, and the top-level `README.md`.

---

## Two truths that hold regardless of which model you pick

1. **Australian-English fidelity is unverified for every model**, including OmniVoice's
   named "australian accent" control. The single highest-priority unknown is a **blind
   listening test with Australian listeners** (the named attribute vs a consented-AU
   voice clone). Do not trust any "natural AU" claim before that.
2. **Every TTS option is safety-neutral** — none add model-level guardrails. All
   kid-safety, boundary-keeping, and graceful out-of-distribution (OOD) handling must
   live in the **upstream STT → dialogue-LLM → moderation stack**. That architecture, not
   the voice model, decides whether the product is safe. (Full-duplex S2S like PersonaPlex
   is actually *harder* to bound, because there's no clean text checkpoint to moderate.)

---

## Ranking against the criteria

| # | Model | Type | AU English | License (commercial) | Self-host | Fit /10 |
|---|-------|------|-----------|----------------------|-----------|:--:|
| 1 | **OmniVoice** (k2-fsa / Next-gen Kaldi) | TTS | ✅ **named "australian accent"** + `child`/`teenager` age attrs | ✅ Apache-2.0 (open) | ⭐ ~0.6B / 2.45 GB, **official vLLM-Omni** | **8** |
| 2 | **Qwen3-TTS-1.7B** (Alibaba Qwen) | TTS | ⚠️ via cloning only (drifts to US) | ✅ Apache-2.0 (open) | ~1.7B / 3.8 GB, vLLM-Omni online | **7** |
| 3 | **PersonaPlex-7B** (NVIDIA) | **S2S full-duplex** | ❌ US-only; custom voice blocked | ✅ NVIDIA Open Model License (no contact) | 7B, Moshi/Mimi stack (not vLLM-Omni) | **6** |
| 4 | **Higgs Audio v3 TTS** (Boson AI) | TTS | ❌ via cloning only | ⚠️ non-commercial; commercial by contact | 4B / 9.3 GB, SGLang-Omni | **6** |
| 5 | **MOSS-TTS-v1.5** (OpenMOSS / Fudan) | TTS | ❌ weakest AU story | ✅ Apache-2.0 (open) | 8B / ~17 GB, Transformers/SGLang | **6** |
| 6 | **Nemotron 3 VoiceChat 12B** (NVIDIA) | S2S | ❌ US-only | ⚠️ permissive *in principle* — **gated early-access, no public weights** | not self-hostable now | **5** |
| 7 | **Magpie TTS Multilingual** (NVIDIA) | TTS | ❌ US-only; **cloning removed** in open ckpt | ❌ **"commercial" vs HF-gate "non-commercial ONLY" contradiction** | 357M, NeMo / Riva-NIM | **4** |

---

## Recommendation

**Spin up OmniVoice first, run Qwen3-TTS-1.7B alongside it as a de-risked hedge, and
treat PersonaPlex-7B as a separate, time-boxed S2S spike** (to learn the conversational
ceiling, not as a baseline replacement). Scaffolds for the first two are in
`tts/omnivoice.py` and `tts/qwen3.py`.

Concretely:
1. Stand both TTS models up on the existing vLLM-Omni / OpenAI-compatible path; benchmark
   cold-start, RTF, and concurrency (`bench.py` is a starting point).
2. Run a **blind AU listening test** — OmniVoice's `instructions="…australian accent…"`
   attribute *and* a consented-AU voice clone, for both models, judged by AU listeners.
3. Probe **OOD-text robustness** (gibberish, code-switching, very long, adversarial
   child-style inputs) for hallucination / token-skipping / timbre drift.
4. Defer Magpie and the gated Nemotron 12B.

---

## Per-model notes (verified)

### 1. OmniVoice — `k2-fsa/OmniVoice` (fit 8) — TOP PICK
- **Provider:** k2-fsa / "Next-gen Kaldi" (Daniel Povey et al.) — highly trusted speech lab.
- **Modality:** TTS (diffusion-LM, discrete acoustic tokens). Not S2S.
- **AU English:** the standout. `docs/voice-design.md` exposes a **named, selectable
  `australian accent`** attribute (alongside american/british/canadian/indian/…), plus
  **`child` / `teenager`** age attributes, 5 pitch levels, `whisper`, and inline
  non-verbals like `[laughter]`. Caveats: it's a prompt attribute, not a curated AU
  corpus — fidelity unverified; zero-shot cloning of a consented AU clip is likely the
  more consistent path.
- **License:** **Apache-2.0** (verified against the actual `LICENSE`, ©2026 Xiaomi Corp.;
  the arXiv "CC-BY-4.0" is the *paper* license, not the model). Open commercial use, no
  contact, no paywall. Non-binding ethical-use disclaimer only.
- **Self-host:** the field's easiest. ~613M params (Qwen3-0.6B base) fp32, `model.safetensors`
  2.45 GB + a 24 kHz codec (`audio_tokenizer/`, ~806 MB) that downloads with the repo.
  Ungated. Realistically runs on a 24 GB GPU (L4/A10G); RTF ~0.025. **Official
  vLLM-Omni support** (`OmniVoiceModel`, OpenAI-compatible `/v1/audio/speech`) —
  reuses your existing serving recipe.
- **Watch out for:** research-grade pace / informal support; AU quality is attribute-
  conditioned and unverified; the clean online voice-design/cloning request adapter
  (`instructions` / `ref_audio` mapping) landed 2026-06-13 in vLLM-Omni **v0.23.0rc1** /
  `main` — basic online TTS works on the proven v0.22.x stack, but for the full
  voice-design path you bump the version (see `tts/omnivoice.py` header).
- **Sources:** `huggingface.co/k2-fsa/OmniVoice`, `github.com/k2-fsa/OmniVoice`
  (`docs/voice-design.md`), `github.com/vllm-project/vllm-omni`
  (`docs/serving/speech_api.md`, `docs/models/supported_models.md`,
  `examples/online_serving/text_to_speech/omnivoice/`), arXiv 2604.00688.

### 2. Qwen3-TTS-12Hz-1.7B — `Qwen/Qwen3-TTS-12Hz-1.7B-{Base,CustomVoice,VoiceDesign}` (fit 7) — HEDGE
- **Provider:** Alibaba Cloud Qwen — top-tier, well-resourced; ~1.3M downloads on Base; paper + `qwen-tts` pip pkg.
- **Modality:** TTS (zero-shot 3 s cloning + natural-language voice design). Not S2S.
- **AU English:** no native AU. English is first-class and reviewers praise warm,
  natural, storytelling pacing, but it defaults to American (sometimes Chinese-accented);
  text-prompt accents drift to American. **AU only reliably via zero-shot cloning** of a
  consented AU reference (the `-Base` variant). Advertised "dialects" are Chinese, not
  English variants.
- **License:** **Apache-2.0** (HF metadata-declared; there is no standalone `LICENSE`
  file — the metadata is binding). Open commercial use. The `Qwen3-TTS-Flash` cloud/API
  variant is separate (Alibaba Cloud terms) and not needed for self-host.
- **Self-host:** easy. 1.7B, `model.safetensors` ~3.8 GB BF16 + separate `speech_tokenizer/`
  codec (~0.68 GB). Ungated. **vLLM-Omni now ships an online `/v1/audio/speech` path**
  (the earlier "offline-only" note is stale — confirmed by example scripts + CI e2e
  tests). Serve via the two-stage `qwen3_tts.yaml` deploy-config. The dual-engine design
  inflates VRAM (research saw the 0.6B hit ~22 GB on a 48 GB L40S), so budget ~0.9 of the
  card; comfortable on a 48 GB L40S/A6000, fine on 80 GB.
- **Variant → task:** `-CustomVoice` = preset speakers + `language` + style `instructions`;
  `-Base` = zero-shot cloning (`ref_audio` + `ref_text`) — **the AU path**; `-VoiceDesign`
  = natural-language voice design.
- **Watch out for:** single-request RTF was ~2.8–3× (not real-time without
  batching/optimization) — validate production streaming; no built-in safety/watermarking.
- **Sources:** `huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-*`, `github.com/QwenLM/Qwen3-TTS`,
  `pypi.org/project/qwen-tts`, vLLM-Omni `examples/online_serving/text_to_speech/qwen3_tts/`
  + `tests/e2e/online_serving/test_qwen3_tts_*.py`, arXiv 2601.15621.

### 3. NVIDIA PersonaPlex-7B-v1 — `nvidia/personaplex-7b-v1` (fit 6) — OPTIONAL S2S SPIKE
- **Modality:** the only released, self-hostable **full-duplex S2S** — barge-in, overlap,
  ~0.17–0.26 s turn-taking. Feels far more like "talking with" someone than read-aloud TTS.
  Moshi/Mimi + Helium backbone; ready Moshi server + web UI (port 8998).
- **AU English:** ❌. US-English only (Fisher telephone corpus); docs say it mispronounces
  non-English accents. **Custom voice cloning is NOT available to end users** — NVIDIA
  stated they "can't promise custom voice" due to legal approval — so you can't supply an
  AU reference. Fails the AU goal; English-only.
- **License:** NVIDIA Open Model License + CC-BY-4.0; "ready for commercial use," royalty-
  free, no business contact. **Conditions that matter for kids:** rights
  **auto-terminate** if you weaken/circumvent any safety guardrail; bound to NVIDIA
  Trustworthy-AI / AI-ethics terms; revocable, AS-IS. Get legal sign-off; a blog-vs-card
  commercial-status inconsistency is worth confirming in writing.
- **Self-host:** gated (accept license) Safetensors, official NVIDIA GitHub, single
  ~24 GB GPU real-time (community-reported; official test HW is A100 80 GB). Stack is Moshi
  PyTorch — **does not reuse vLLM-Omni** (support is an in-progress, turn-based-MVP issue).
  At-scale full-duplex serving is ~1 GPU/user.
- **Watch out for:** docs say it improvises "plausibly" on OOD prompts — the exact failure
  mode to fear with kids; end-to-end S2S has no text choke-point for moderation (wrap with
  NVIDIA Nemotron Content Safety / NeMo Guardrails on both transcript and audio).
- **Sources:** `huggingface.co/nvidia/personaplex-7b-v1`, `github.com/NVIDIA/personaplex`,
  `research.nvidia.com/labs/adlr/personaplex/`, NVIDIA Open Model License PDF.

### 4. Higgs Audio v3 TTS — `bosonai/higgs-audio-v3-tts-4b` (fit 6)
- **Modality:** real-time, chat-native streaming **TTS** with the **richest expressiveness
  controls** (21 emotions, styles, prosody, SFX, mid-utterance). Strong first-class English
  (Seed-TTS 1.11% WER). Zero-shot cloning. ~1 s E2E, RTF 0.262 on 1×H100.
- **AU English:** ❌ native — no AU preset, no accent-control token (controls cover
  emotion/style/prosody/SFX, not region). AU only via cloning (unverified; needs consent).
- **License:** v3 TTS = Boson Research & Non-Commercial License; commercial requires a
  separate license via `contact@boson.ai`. Mandates conspicuous **AI-generated disclosure**
  (a kids-app UX/compliance constraint); has explicit child-protection/consent clauses
  (policy, not enforcement).
  Fallback: the older **v2** model is commercially free under 100k annual active users.
- **Self-host:** **ungated/public** (corrected from "gated"); `model.safetensors` 9.31 GB,
  single 24–48 GB GPU. Official path is **SGLang-Omni** (not vLLM-Omni) —
  a new serving stack. Transformers serving is not officially documented.
- **Sources:** `huggingface.co/bosonai/higgs-audio-v3-tts-4b` (+ `LICENSE`),
  `github.com/boson-ai/higgs-audio`, `lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/`.

### 5. MOSS-TTS-v1.5 — `OpenMOSS-Team/MOSS-TTS-v1.5` (fit 6)
- **Modality:** 8B **TTS** with zero-shot cloning; strong benchmarked English (~1.84% WER),
  rich prosody/pause (`[pause X.Ys]`)/IPA control. Siblings: MOSS-TTSD (dialogue),
  MOSS-TTS-Realtime (1.7B streaming), Nano (0.1B). None are true S2S.
- **AU English:** ❌ weakest story — no AU mention, no accent control, China-heavy training;
  AU only via cloning (unverified). Biggest fit risk despite strong generic English.
- **License:** **Apache-2.0** (HF metadata + GitHub "MOSS-TTS Family … Apache License 2.0").
  Open commercial, no contact, no AUP (flip side: misuse responsibility entirely on you).
- **Self-host:** standard HF Transformers; ~17 GB BF16 (4 shards) on a 24 GB card; SGLang
  ~3×; torch-free llama.cpp/ONNX/TensorRT. Needs bleeding-edge Transformers 5.0; recipes
  still maturing. No documented vLLM-Omni path. (The "fits 8 GB quantized" claim is
  overstated — the real Q8 GGUF is ~12.6 GB.)
- **Provider caveat:** China-based academic lab (no commercial SLA) — a procurement/
  data-governance consideration for a children's product.
- **Sources:** `huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5`, `github.com/OpenMOSS/MOSS-TTS`,
  arXiv 2603.18090.

### 6. NVIDIA Nemotron 3 VoiceChat 12B (fit 5) — NOT ACTIONABLE NOW
- The flagship 12B full-duplex S2S **tops NVIDIA's conversational benchmarks** (Full Duplex
  Bench ~91%) but is **gated early-access with NO public weights** — can't prototype or
  self-host it. Its only downloadable family member is **PersonaPlex-7B** (rank 3). en-US
  only, so it fails the AU goal regardless. Revisit if/when weights are released.
- Same vendor ships complementary open-weights safety tooling worth knowing about for the
  upstream stack: **Nemotron 3 Content Safety (4B)** classifier and **NeMo Guardrails**.
- **Sources:** `developer.nvidia.com/nemotron-voicechat-early-access`,
  `huggingface.co/collections/nvidia/nemotron-speech`.

### 7. NVIDIA Magpie TTS Multilingual — `nvidia/magpie_tts_multilingual_357m` (fit 4) — AVOID
- **One genuine strength:** explicitly-engineered **OOD-text robustness / anti-hallucination**
  (monotonic alignment + CTC + attention priors) — valuable for unpredictable kid input.
- **But disqualifying for this use case:** en-US only, no accent control, AND the open 357M
  checkpoint had **zero-shot cloning removed** (only 5 fixed US voices) — so there is *no
  path to an AU voice at all*. Plus a **verified, unresolved license contradiction**: the
  model card says "ready for commercial use" (NVIDIA Open Model License) while the same HF
  repo's gating checkbox requires agreeing to "non-commercial use ONLY" — a real legal
  blocker to clear in writing before any commercial deployment. Best production path pushes
  you to Riva NIM (NVIDIA AI Enterprise; cost/lock-in).
- **Sources:** `huggingface.co/nvidia/magpie_tts_multilingual_357m`,
  `build.nvidia.com/nvidia/magpie-tts-multilingual`, NVIDIA NIM TTS support matrix.

---

## Why OmniVoice and Qwen stay in repo

The current repo keeps the two candidates that best match the product constraints:

- **Licensing:** OmniVoice and Qwen3-TTS are Apache-2.0, open for commercial use, and do
  not require a business-contact license path.
- **Serving reuse:** both run through Modal + vLLM-Omni + `/v1/audio/speech`. Higgs
  (SGLang-Omni), MOSS (Transformers/SGLang), and PersonaPlex (Moshi) introduce new
  serving stacks.
- **AU English:** OmniVoice is the only researched model with a named AU control plus
  child/teenager age attributes. Qwen3 stays as the cloning-capable hedge.
- **Cost / footprint:** OmniVoice is small enough for 24 GB GPUs; Qwen3 is heavier but
  still open and well-supported by the vLLM-Omni path.
- **Safety/OOD:** none of the TTS models add model-level guardrails. The real safety work
  remains upstream in STT -> dialogue-LLM -> moderation.

**Net:** OmniVoice is the default path; Qwen3-TTS is the comparison/hedge; PersonaPlex is
a separate S2S spike if the product needs full-duplex conversation later.

---

## Open questions to resolve before committing

1. **AU accent fidelity** — blind-test OmniVoice's named attribute and AU clones (all models)
   with Australian listeners. Highest-priority unknown for the primary criterion.
2. **OmniVoice on your stack** — does the official vLLM-Omni path run cleanly on your Modal
   image/pins, and what are real cold-start / RTF / concurrency numbers?
3. **Qwen3-TTS streaming** — is online vLLM-Omni serving production-ready under your concurrency
   targets, and does the 2-engine VRAM blow-up bite at 1.7B?
4. **OOD-text robustness** — empirically characterize hallucination/skipping/drift on
   adversarial child-style input for whichever you pick (unpublished for all).
5. **Kid-safety architecture** — all TTS options are safety-neutral; design the upstream
   STT → dialogue-LLM → moderation + boundary stack (consider NVIDIA Nemotron Content Safety /
   NeMo Guardrails).
6. **Compliance for a children's product** — confirm Apache-2.0 NOTICE/attribution for the exact
   repo snapshot you ship; for Higgs, the license mandates conspicuous AI-generated disclosure;
   for any NVIDIA model, get legal sign-off on the Trustworthy-AI / guardrail-non-circumvention
   clauses for under-13 use.
