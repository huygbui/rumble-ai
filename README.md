# Fish Speech S2 Pro on Modal (vLLM-Omni TTS)

Minimal demo serving [`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro)
as an OpenAI-compatible Text-to-Speech endpoint on Modal, via vLLM-Omni.

- Endpoint: `POST /v1/audio/speech` (OpenAI-compatible)
- Output: 44.1 kHz mono WAV (DAC codec)
- GPU: one 80GB card (~49 GiB peak)

## Files

- `fish_s2_pro_modal.py` — Modal app: builds the image, caches weights on a Volume,
  runs `vllm serve fishaudio/s2-pro --omni` as a web server.
- `client.py` — calls the endpoint twice: plain TTS and voice cloning.

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

```bash
export TTS_URL="https://<workspace>--fish-s2-pro-tts-serve.modal.run"

# Voice cloning needs a reference clip (10-30s) + its EXACT transcript.
# Against a REMOTE Modal endpoint REF_AUDIO must be reachable by the server:
#   - a public https:// URL, OR
#   - a data:audio/wav;base64,... data URI, OR
#   - a LOCAL file path (client.py auto-encodes it into a data: URI before POST).
# A bare local path is NOT looked up on your machine by the server; vllm-omni's
# "local path auto-base64" resolves inside the container, so the client encodes it.
export REF_AUDIO="./reference.wav"   # local path, https:// URL, or data: URI
export REF_TEXT="The exact transcript of the reference audio."

python client.py
# -> tts.wav    (plain text-to-speech)
# -> cloned.wav (voice cloning; skipped if REF_AUDIO/REF_TEXT unset)
```

## Notes

- `--omni` is **mandatory** — plain `vllm serve` will not bring up the TTS stack.
- Voice cloning (Base mode) requires **both** `ref_audio` and `ref_text`.
- Pass a fixed `seed` for a deterministic voice (vLLM-Omni PR #2624).
- The image pins the recipe's tested combo: **vLLM 0.19.0 + vllm-omni commit
  `c93359bb…`**. `vllm-omni` is a real PyPI package, but its releases version-match
  a newer vLLM (PyPI latest pairs with vLLM 0.23.0), so do not mix a hard
  `vllm==0.19.0` pin with an unpinned `vllm-omni`.
- **Known bug ([vllm-omni #2248](https://github.com/vllm-project/vllm-omni/issues/2248)):**
  on the s2-pro path audio can be truncated to ~5–9s and inline emotion `[tag]`
  controls may have no effect. If you need long-form output or working emotion tags,
  try a newer vLLM / vllm-omni build (and update the pins in `fish_s2_pro_modal.py`).
- If startup fails with Triton/attention errors, uncomment
  `VLLM_OMNI_FISH_KVCACHE_ATTN: "0"` in `fish_s2_pro_modal.py` to disable the Fish
  KV-cache attention fast path (per the s2-pro recipe).
- The app scales to zero after 15 min idle; uncomment `min_containers=1` in
  `fish_s2_pro_modal.py` to keep one container always warm.
