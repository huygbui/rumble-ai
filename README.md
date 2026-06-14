# Fish Speech S2 Pro on Modal (vLLM-Omni TTS)

Minimal demo serving [`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro)
as an OpenAI-compatible Text-to-Speech endpoint on Modal, via vLLM-Omni.

- Endpoint: `POST /v1/audio/speech` (OpenAI-compatible)
- Output: 44.1 kHz mono WAV (DAC codec)
- GPU: one 80GB card (~49 GiB peak)

## Files

- `fish_s2_pro_modal.py` — Modal app: builds the image, caches weights on a Volume,
  runs `vllm serve fishaudio/s2-pro --omni` as a web server.
- `client.py` — registers a reusable voice then synthesizes by name (`tts.wav`), and
  does inline voice cloning (`cloned.wav`).
- `bench_tts.py` — latency / streaming / concurrency benchmark.

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
