# rumble-ai — self-hosted voice pipeline on Modal

A self-hosted **voice conversation** stack served on [Modal](https://modal.com), built up
one component at a time. Today it hosts the **TTS** stage as a 3-way model bake-off;
**STT** and **LLM** components — then a back-and-forth `app.cli.chat` loop wiring
STT → LLM → TTS — are next.

Every model is served behind the same OpenAI-compatible `POST /v1/audio/speech` shape, so
one `client.py` drives any of them via `TTS_MODEL`.

> **Evaluating models for a kid-facing, Australian-English experience.** Fish S2 Pro is the
> current TTS baseline; OmniVoice and Qwen3-TTS are candidates under evaluation. The full
> ranked comparison of 8 TTS/S2S options (with sources) lives in
> [`docs/tts-options.md`](docs/tts-options.md) — see [Alternatives under evaluation](#alternatives-under-evaluation).

## Repo layout

```
app/                  # FastAPI app package
  main.py             # FastAPI app + lifespan; run with `fastapi dev app/main.py`
  api/                # HTTP boundary: routers and Pydantic request/response schemas
  core/               # runtime helpers: config, llm, tts, text chunking, STT, warm-up
  cli/                # CLI entrypoints: `python -m app.cli.say`, `python -m app.cli.chat`
tts/                  # TTS component — one Modal app per candidate model
  fish_s2_pro.py      # fishaudio/s2-pro — baseline (research license, ~49 GiB, 80GB GPU)
  omnivoice.py        # k2-fsa/OmniVoice — top pick (Apache-2.0, native AU accent, 24GB GPU)
  qwen3.py            # Qwen3-TTS-1.7B (Alibaba) — hedge (Apache-2.0, natural English)
stt/                  # (next) speech-to-text candidates — one Modal app per model
llm/                  # (next) dialogue-LLM candidates — one Modal app per model
client.py             # drive any /v1/audio/speech endpoint; per-model shape via TTS_MODEL
bench.py              # latency / streaming / concurrency benchmark
docs/
  tts-options.md      # ranked evaluation of TTS/S2S options for this use case
  omnivoice-bench.md  # OmniVoice benchmark + head-to-head vs Fish
  perf-baseline.md    # Fish S2 Pro performance baseline
  fish-cold-start.md  # Fish S2 Pro cold-start investigation
out/                  # generated audio: client.py runs + demo clips (gitignored; set
                      #   TTS_OUT_DIR). out/MANIFEST.txt (tracked) lists the demo clips

```

Each pipeline stage is its own top-level folder holding one Modal app per candidate
model; `stt/` and `llm/` follow the same pattern as `tts/` when those components land.

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
modal serve tts/fish_s2_pro.py

# OR persistent deployment:
modal deploy tts/fish_s2_pro.py
```

First cold start downloads the weights into the `huggingface-cache` Volume and can
take several minutes (covered by the 20-minute startup timeout). The CLI prints a
URL like `https://<workspace>--fish-s2-pro-tts-serve.modal.run`.

Health check:

```bash
modal run tts/fish_s2_pro.py
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
# -> out/tts.wav    (registers a reusable voice, then synthesizes by name; needs a LOCAL ref)
# -> out/cloned.wav (inline ref_audio + ref_text; skipped if REF_AUDIO/REF_TEXT unset)
```

Two ways to give S2 Pro a voice:

- **Registered voice** — `POST /v1/audio/voices` (multipart: `name`, `consent`,
  `audio_sample`, `ref_text`) once, then synthesize with `voice="<name>"` and no
  per-request reference. `GET /v1/audio/voices` lists them; `DELETE
  /v1/audio/voices/{name}` removes one.
- **Inline cloning (Base mode)** — pass `ref_audio` + `ref_text` on each request.

`bench.py` is a quick latency/throughput benchmark; it registers one voice up
front and then synthesizes by name (set `REF_AUDIO`/`REF_TEXT` as above).

## Alternatives under evaluation

Two candidate models for a kid-facing, Australian-English conversation experience, served
the same way (OpenAI-compatible `/v1/audio/speech`). Both verified end-to-end on Modal.
See [`docs/tts-options.md`](docs/tts-options.md) for why these two.

`client.py` targets any endpoint via `TTS_MODEL`:

```bash
# OmniVoice — Australian-accent kid voice via DESIGN (no reference clip needed):
modal deploy tts/omnivoice.py
export TTS_URL="https://<workspace>--omnivoice-tts-serve.modal.run"
TTS_MODEL=omnivoice python client.py            # -> out/omnivoice_design.wav

# Qwen3-TTS — preset speaker (CustomVoice); AU via cloning needs the -Base variant:
modal deploy tts/qwen3.py
export TTS_URL="https://<workspace>--qwen3-tts-serve.modal.run"
TTS_MODEL=qwen-customvoice TTS_VOICE=vivian python client.py   # -> out/qwen_customvoice.wav
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
  protobuf) are **expected and benign** — see the comments in `tts/fish_s2_pro.py`.
- If startup fails with Triton/attention errors, uncomment
  `VLLM_OMNI_FISH_KVCACHE_ATTN: "0"` in `tts/fish_s2_pro.py` to disable the Fish
  KV-cache attention fast path (installs cleanly on 0.22.x, so normally unneeded).
- The app scales to zero after 15 min idle; uncomment `min_containers=1` in
  `tts/fish_s2_pro.py` to keep one container always warm.
