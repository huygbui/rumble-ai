# Fish Speech S2 Pro on Modal (vLLM-Omni TTS)

Minimal demo serving [`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro)
as an OpenAI-compatible Text-to-Speech endpoint on Modal, via vLLM-Omni.

- Endpoint: `POST /v1/audio/speech` (OpenAI-compatible)
- Output: 44.1 kHz mono WAV (DAC codec)
- GPU: one 80GB card (~49 GiB peak)

> **Evaluating alternatives.** Fish S2 Pro is the baseline. This repo now also hosts two
> candidate models being evaluated for a **kid-facing, Australian-English** conversation
> experience. The full ranked comparison of 8 TTS/S2S options (with sources) lives in
> [`docs/tts-options-research.md`](docs/tts-options-research.md). All three models share
> the OpenAI-compatible `/v1/audio/speech` shape, so one `client.py` drives them via
> `TTS_MODEL` — see [Alternatives under evaluation](#alternatives-under-evaluation).

## Files

- `fish_s2_pro_modal.py` — Modal app: builds the image, caches weights on a Volume,
  runs `vllm serve fishaudio/s2-pro --omni` as a web server.
- `omnivoice_modal.py` — Modal app for **OmniVoice** (`k2-fsa/OmniVoice`), the top
  alternative: Apache-2.0, native `australian accent` voice-design. *(under evaluation)*
- `qwen3_tts_modal.py` — Modal app for **Qwen3-TTS-1.7B** (Alibaba), the de-risked
  hedge: Apache-2.0, natural streaming English. *(under evaluation)*
- `client.py` — generalized client for all three endpoints; picks the per-model request
  shape via `TTS_MODEL` (writes `tts.wav` / `cloned.wav` / `omnivoice_design.wav` / …).
- `bench_tts.py` — latency / streaming / concurrency benchmark.
- `docs/tts-options-research.md` — ranked evaluation of TTS/S2S options for this use case.

## 1. Prereqs

```bash
uv sync                    # installs modal + requests (or: pip install modal requests)
uv run modal token new     # authenticate the Modal CLI (one-time)
```

`fishaudio/s2-pro` is a **public** HF repo (`gated=false`), so the weights download
with **no HF token** — no license-acceptance step is required. The Fish Audio
Research License only governs *use*: research / non-commercial is free, commercial
use needs a separate license from `business@fish.audio`.

## 2. Serve / deploy

```bash
# ephemeral dev server (hot reload, prints the URL):
modal serve fish_s2_pro_modal.py

# OR persistent deployment:
modal deploy fish_s2_pro_modal.py
```

First cold start downloads the weights into the `huggingface-cache` Volume and can
take several minutes (covered by the 20-minute startup timeout). The CLI prints a
URL like `https://<workspace>--fish-s2-pro-tts-serve.modal.run`.

Health check:

```bash
modal run fish_s2_pro_modal.py
```

## 3. Run the client

S2 Pro is a **zero-shot** model with **no built-in voice**, so every request must
supply a voice via either a registered voice name or an inline reference clip. (The
old `voice:"default"` shortcut was removed upstream and now returns 400.)

```bash
export TTS_URL="https://<workspace>--fish-s2-pro-tts-serve.modal.run"

# A reference clip (10-30s) + its EXACT transcript are required.
# For inline cloning, REF_AUDIO must be reachable by the SERVER:
#   - a public https:// URL, OR
#   - a data:audio/wav;base64,... data URI, OR
#   - a LOCAL file path (client.py auto-encodes it into a data: URI before POST).
# Voice REGISTRATION uploads the bytes directly (multipart), so it needs a LOCAL file.
export REF_AUDIO="./reference.wav"   # local path, https:// URL, or data: URI
export REF_TEXT="The exact transcript of the reference audio."

python client.py
# -> tts.wav    (registers a reusable voice, then synthesizes by name; needs a LOCAL ref)
# -> cloned.wav (inline ref_audio + ref_text; skipped if REF_AUDIO/REF_TEXT unset)
```

Two ways to give S2 Pro a voice:

- **Registered voice** — `POST /v1/audio/voices` (multipart: `name`, `consent`,
  `audio_sample`, `ref_text`) once, then synthesize with `voice="<name>"` and no
  per-request reference. `GET /v1/audio/voices` lists them; `DELETE
  /v1/audio/voices/{name}` removes one.
- **Inline cloning (Base mode)** — pass `ref_audio` + `ref_text` on each request.

`bench_tts.py` is a quick latency/throughput benchmark; it registers one voice up
front and then synthesizes by name (set `REF_AUDIO`/`REF_TEXT` as above).

## Alternatives under evaluation

Two candidate models for a kid-facing, Australian-English conversation experience, served
the same way (OpenAI-compatible `/v1/audio/speech`). Both verified end-to-end on Modal.
See [`docs/tts-options-research.md`](docs/tts-options-research.md) for why these two.

`client.py` targets any endpoint via `TTS_MODEL`:

```bash
# OmniVoice — Australian-accent kid voice via DESIGN (no reference clip needed):
modal deploy omnivoice_modal.py
export TTS_URL="https://<workspace>--omnivoice-tts-serve.modal.run"
TTS_MODEL=omnivoice python client.py            # -> omnivoice_design.wav

# Qwen3-TTS — preset speaker (CustomVoice); AU via cloning needs the -Base variant:
modal deploy qwen3_tts_modal.py
export TTS_URL="https://<workspace>--qwen3-tts-serve.modal.run"
TTS_MODEL=qwen-customvoice TTS_VOICE=vivian python client.py   # -> qwen_customvoice.wav
```

`TTS_MODEL` values: `fish | omnivoice | qwen-customvoice | qwen-base | qwen-voicedesign`.
For voice **cloning** (Fish, `qwen-base`, or optional on OmniVoice), set `REF_AUDIO` +
`REF_TEXT` to a consented reference clip as in §3.

Two model-specific gotchas (documented in-file):

- **OmniVoice needs the vLLM-Omni 0.23 line.** Its online voice-design adapter
  ([PR #4330](https://github.com/vllm-project/vllm-omni/pull/4330)) is **not** in the
  0.22.0 stable wheel — on 0.22.0 the `instructions` path emits artifacts (noise), not a
  conditioned voice. The app installs `vllm-omni==0.23.0rc1` from its git tag (it isn't on
  PyPI) paired with `vllm==0.23.0`. Runs on a 24GB GPU (A10G), not 80GB.
- **Qwen3-TTS needs an 80GB H100.** Its shipped two-stage `qwen3_tts.yaml` deploy-config
  is "Verified on 1×H100" (talker + code2wav co-located at `0.3` memory each); a 48GB L40S
  OOMs the code2wav stage at startup. Do **not** pass `--gpu-memory-utilization` on the CLI
  — it overrides the per-stage split. Stays on the 0.22.0 stable stack (works there).

## Notes

- `--omni` is **mandatory** — plain `vllm serve` will not bring up the TTS stack.
- Pass a fixed `seed` for a deterministic voice.
- The image pins **vLLM 0.22.1 + vllm-omni 0.22.0** (both PyPI releases). vllm-omni
  is version-coupled to vLLM (its releases version-track the vLLM minor they rebase
  onto), so vllm-omni 0.22.0 pairs with the vLLM 0.22 line — keep the pair
  consistent if you bump either.
- **Long-form output works** on this stack: the older 0.19.0 build truncated s2-pro
  audio to ~5–9s ([vllm-omni #2248](https://github.com/vllm-project/vllm-omni/issues/2248),
  now fixed); verified producing 30s+ of continuous audio here.
- The two pip resolver `ERROR` lines during the image build (einx / pydantic /
  protobuf) are **expected and benign** — see the comments in `fish_s2_pro_modal.py`.
- If startup fails with Triton/attention errors, uncomment
  `VLLM_OMNI_FISH_KVCACHE_ATTN: "0"` in `fish_s2_pro_modal.py` to disable the Fish
  KV-cache attention fast path (installs cleanly on 0.22.x, so normally unneeded).
- The app scales to zero after 15 min idle; uncomment `min_containers=1` in
  `fish_s2_pro_modal.py` to keep one container always warm.
