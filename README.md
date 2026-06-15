# rumble-ai — voice chat app on Modal

Rumble is a small production web app for low-latency voice conversation. The app runs
locally as FastAPI and talks to Modal-hosted STT, LLM, and TTS services:

```
browser -> FastAPI -> STT / LLM / TTS on Modal
```

OmniVoice is the default TTS runtime; Qwen3-TTS remains available as a heavier comparison
path. The old model scripts are still useful lab tools, but the product path is the web
app in `app/` and `web/`.

> **Evaluating models for a kid-facing, Australian-English experience.** OmniVoice is the
> current default because it has a native `australian accent` voice-design control, runs on
> a smaller GPU, and has an Apache-2.0 license. The broader model research lives in
> [`docs/tts-options.md`](docs/tts-options.md).

## Repo layout

```
app/                  # FastAPI app package
  main.py             # FastAPI app + lifespan; run with `fastapi dev app/main.py`
  api/                # HTTP boundary: routers and Pydantic request/response schemas
  core/               # turn pipeline, config, LLM/TTS/STT helpers, clause splitting
web/                  # product UI served by FastAPI
tts/                  # TTS component — one Modal app per candidate model
  omnivoice.py        # k2-fsa/OmniVoice — default (Apache-2.0, native AU accent, 24GB GPU)
  qwen3.py            # Qwen3-TTS-1.7B — hedge (Apache-2.0, natural English)
stt/                  # speech-to-text candidates — one Modal app per model
llm/                  # dialogue-LLM candidates — one Modal app per model
client.py, bench.py   # lab tools for model checks and one-off benchmarks
docs/
  tts-options.md      # ranked evaluation of TTS/S2S options for this use case
  omnivoice-bench.md  # OmniVoice benchmark, tuning notes, and chunking results
  modal-serving-notes.md
out/                  # generated audio (gitignored; set TTS_OUT_DIR)
```

Each pipeline stage is its own top-level folder holding one Modal app per candidate
model.

## 1. Prereqs

```bash
uv sync
uv run modal token new
```

## 2. Deploy Modal services

```bash
modal deploy stt/qwen3_asr.py
modal deploy llm/qwen3_5_4b.py
```

OmniVoice is the default TTS service:

```bash
modal deploy tts/omnivoice.py
export TTS_URL="https://<workspace>--omnivoice-tts-serve.modal.run"
```

Qwen3-TTS is available as a heavier comparison endpoint:

```bash
modal deploy tts/qwen3.py
export TTS_URL="https://<workspace>--qwen3-tts-serve.modal.run"
```

## 3. Run the app

```bash
export STT_URL="https://<workspace>--qwen3-asr-stt-serve.modal.run"
export LLM_URL="https://<workspace>--qwen3-5-4b-llm-serve.modal.run"
export TTS_URL="https://<workspace>--omnivoice-tts-serve.modal.run"

uv run fastapi dev app/main.py
```

Open the printed local URL, warm the services, then chat by typing or using the mic.

## Lab Tools

`client.py`, `bench.py`, and `app/cli/say.py` are kept for model checks and one-off
benchmarks. They are not the production app path.

### TTS client

```bash
TTS_MODEL=omnivoice python client.py
# -> out/omnivoice_design.wav

TTS_MODEL=qwen-customvoice TTS_VOICE=vivian python client.py
# -> out/qwen_customvoice.wav
```

`TTS_MODEL` values: `omnivoice | qwen-customvoice | qwen-base | qwen-voicedesign`.
For voice **cloning** (`qwen-base`, or optional on OmniVoice), set `REF_AUDIO` +
`REF_TEXT` to a consented reference clip. `REF_AUDIO` may be a public URL, a data URI, or
a local path that `client.py` auto-encodes into a data URI before posting.

### Chunked speech CLI

`app.cli.say` is the app-facing TTS path. It splits text into short clauses, synthesizes
them in a producer thread, and stitches the clips so the first audio is ready sooner than a
single long request.

```bash
echo "G'day! Want to hear a quick story? It begins on a windy hill by the sea." \
  | TTS_MODEL=omnivoice python -m app.cli.say
```

Useful flags:

```bash
PLAY=1      # play each generated clause with afplay
COMPARE=1   # also run a single-shot request for timing comparison
STITCH_ONLY=1 python -m app.cli.say
```

### Benchmark

```bash
TTS_MODEL=omnivoice BENCH_QUALITY=1 uv run python bench.py

# samples only, skip perf:
BENCH_PERF=0 BENCH_QUALITY=1 uv run python bench.py
```

The latest OmniVoice notes are in [`docs/omnivoice-bench.md`](docs/omnivoice-bench.md).

## Notes

- `--omni` is mandatory for these vLLM-Omni TTS services.
- OmniVoice uses the vLLM-Omni 0.23 line because the online voice-design adapter is not in
  the 0.22.0 stable wheel.
- Qwen3-TTS uses the vendor two-stage deploy config and still needs an 80GB H100 in this
  repo's current serving setup.
- Safety lives upstream in the STT -> LLM -> moderation stack. The TTS models are
  safety-neutral renderers.
