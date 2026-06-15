# stt/qwen3_asr.py
# Serve Qwen3-ASR (Alibaba Qwen) as the STT/ASR stage on Modal via vLLM.
# Exposes the OpenAI-compatible POST /v1/audio/transcriptions endpoint on one small GPU.
#
#   modal deploy stt/qwen3_asr.py   # persistent endpoint
#   modal serve  stt/qwen3_asr.py   # ephemeral hot-reload dev server
#   modal run    stt/qwen3_asr.py   # health-check (+ optional real transcription) via local_entrypoint
#
# A/B the two sizes WITHOUT editing this file. The model id is read from an env var at
# deploy time, baked into the image env (so the container serves the SAME model), and the
# Modal app name is derived from it -- so 0.6B and 1.7B deploy as SEPARATE apps/URLs and can
# run side by side:
#   modal deploy stt/qwen3_asr.py                                    # -> Qwen3-ASR-0.6B (default)
#   QWEN_ASR_MODEL=Qwen/Qwen3-ASR-1.7B modal deploy stt/qwen3_asr.py # -> Qwen3-ASR-1.7B
#
# Why Qwen3-ASR (the STT pick for this pipeline -- see docs/stt-options.md):
#   - Apache-2.0 (open commercial use) -- cleaner than Parakeet's CC-BY-4.0 (attribution) or
#     Nemotron's OpenMDW-1.1, and the right license for a kid-facing commercial product.
#   - The ONLY shortlisted ASR that serves NATIVELY on vLLM (OpenAI /v1/audio/transcriptions,
#     plus streaming via the vLLM backend) -- so STT drops into the SAME serving pattern as
#     llm/qwen3_5_4b.py and the tts/ apps instead of needing the NeMo toolkit + a hand-rolled
#     WebSocket server (the blocker for both NVIDIA candidates).
#   - Same vendor/stack as the Qwen3.5-4B dialogue LLM -> one mental model for the pipeline.
#   - 0.6B default: lowest latency + fastest cold start for the turn loop; 1.7B is the
#     accuracy-headroom option. Pick by MEASUREMENT on AU/kid speech, not the spec sheet.
# This is the transcription stage only -- kid-safety lives in the upstream guardrail layer.
#
# STATUS: VERIFIED end-to-end on Modal (2026-06-15) on an L4. Bring-up findings:
#   - vLLM 0.23.0 loads Qwen3-ASR NATIVELY (engine config trust_remote_code=False) -- no version
#     bump, no --trust-remote-code. (Qwen3-ASR is Qwen3-Omni-thinker-based; the engine logs a
#     thinker_config + a deprecated-WhisperFeatureExtractor notice -- both benign.)
#   - The image MUST install vllm[audio] (NOT plain vllm): the /v1/audio/transcriptions decoder
#     needs PyAV (`av`) + soundfile, else every request 400s "Invalid or unsupported audio file"
#     ("No module named 'av'"). This bit the first attempt; fixed in the image below.
#   - fastapi<0.137 pin held (no Prometheus 500s), same as the sibling vLLM apps.
#   - Round-trip sanity: transcribing the Fish TTS clip out/01_short_greeting.wav returned
#     'Hello, welcome to the Fish Speech S Two Pro demo. How are you doing today?' -- near-perfect
#     ("S2"->"S Two" is just number verbalization). First-call execution ~2.4s on a ~5s clip
#     (RTF ~0.47) -- a LOOSE cold/first-call figure, NOT a tuned steady-state baseline (see PERF).
#   - Footprint: 1.53 GiB VRAM on the 24GB L4 -> huge headroom; the ~67s torch.compile graph is
#     cached to the vllm-cache Volume, so subsequent cold starts skip it.
# PERF TODO: running UNTUNED -- max_seq_len=65536 (uncapped) and @modal.concurrent=8 despite only
#   1.5GB used. Levers (measure single + concurrent first): cap --max-model-len, raise concurrency,
#   optional min_containers=1 warm floor + a post-boot warmup. Streaming transcription is the
#   bigger turn-loop win but needs chat.py work. Kid-safety stays in the upstream guardrail layer.

import os
import re
import subprocess

import modal

# --- Model + serving constants -------------------------------------------------
# Env-driven so you can A/B 0.6B vs 1.7B without touching code. Read on the CLIENT at deploy
# time AND re-read in the container -- where it resolves from the image env baked below, so both
# sides agree (a plain default would otherwise make the container fall back to 0.6B).
MODEL_NAME = os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")  # PUBLIC, ungated, Apache-2.0
VLLM_PORT = 8000  # plain vLLM's default (same as llm/qwen3_5_4b.py); ASR endpoint is /v1/audio/transcriptions
N_GPU = 1  # 0.6B (~1.2GB fp16) / 1.7B (~4GB bf16) -- one cheap 24GB card is plenty either way
MINUTES = 60  # seconds

# Distinct Modal app per model id -> 0.6B and 1.7B get separate URLs and can run side by side
# for A/B. "Qwen/Qwen3-ASR-0.6B" -> "qwen3-asr-0-6b-stt".
_SLUG = re.sub(r"[^a-z0-9]+", "-", MODEL_NAME.split("/")[-1].lower()).strip("-")

# --- vLLM version --------------------------------------------------------------
# Match the rest of the repo (shares the vllm-cache volume). See STATUS note #1 re: bumping.
VLLM_VERSION = "0.23.0"

# --- Container image -----------------------------------------------------------
# Same CUDA 12.8 base + hf_transfer discipline as llm/qwen3_5_4b.py, plus audio I/O libs
# (ffmpeg/libsndfile1 + librosa/soundfile) that vLLM needs to decode uploaded audio for the
# transcription endpoint -- the one real delta from the text-LLM image.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git", "ffmpeg", "libsndfile1")
    # No `--torch-backend=auto`: Modal's image builder has no GPU, so `auto` would install a
    # CPU-only torch that fails on the GPU at runtime (same note as the sibling apps).
    .uv_pip_install(
        # vllm[audio] (NOT plain vllm): the /v1/audio/transcriptions decoder loads uploaded audio
        # via soundfile and falls back to PyAV (`av`); plain vllm ships neither, so every request
        # 400s with "No module named 'av'" / "Please install vllm[audio] for audio support".
        # The [audio] extra pulls av + soundfile + librosa. (Verified: plain vllm fails this exact
        # way on Modal, 2026-06-15.)
        f"vllm[audio]=={VLLM_VERSION}",
        # Pin FastAPI < 0.137: vLLM 0.23.0's Prometheus middleware 500s every request on 0.137's
        # new lazy router. Drop the pin once the instrumentator ships a fix. See llm/qwen3_5_4b.py.
        "fastapi<0.137",
        "huggingface_hub[hf_transfer]",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Bake the resolved model id into the container so serve() loads the SAME model that
            # was deployed (the module-level os.environ read above resolves to THIS in-container).
            "QWEN_ASR_MODEL": MODEL_NAME,
        }
    )
)

# --- Caches (persisted across cold starts) ------------------------------------
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App(f"{_SLUG}-stt")


@app.function(
    image=image,
    gpu=f"L4:{N_GPU}",  # 24GB, ~$0.80/hr -- cheapest fit; same tier as the Qwen3.5-4B LLM. A10G is
    #                     a drop-in alternative. The 0.6B leaves huge headroom (raise concurrency).
    # No secret needed: Qwen/Qwen3-ASR-* are PUBLIC HF repos, download without a token.
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    # Idle-cost control: stay warm 2 min after the last request, then scale to ZERO. STT is hit at
    # the START of every user turn, so during an active conversation (turns seconds apart) the 2-min
    # window holds it warm; between conversations it idles to zero. Set min_containers=1 if first-turn
    # cold-start latency matters more than idle cost.
    scaledown_window=2 * MINUTES,
    timeout=20 * MINUTES,
    # min_containers=1,  # uncomment for always-warm (no cold-start latency, costs $)
)
@modal.concurrent(max_inputs=8)  # 0.6B is light + vLLM continuous-batches well; tune up/down
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve():
    # Non-blocking launch: Modal's web_server waits for the port, the function returns.
    # Base command follows the documented vLLM recipe: `vllm serve Qwen/Qwen3-ASR-0.6B`.
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        # Optional flags to try if first boot misbehaves (see STATUS in the header):
        #   "--trust-remote-code",                 # if vLLM rejects the custom Qwen3-ASR audio arch
        #   "--gpu-memory-utilization", "0.8",     # the repo's qwen-asr-serve wrapper uses 0.8
        #   "--max-model-len", "8192",             # only if you must cap KV; leave default for audio
    ]
    if N_GPU > 1:
        cmd += ["--tensor-parallel-size", str(N_GPU)]
    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
def main():
    # Smoke test: confirm the server is up. If ASR_SAMPLE is set (local WAV path or http URL),
    # also run a REAL transcription and assert a non-empty transcript -- the end-to-end check.
    import urllib.request

    url = serve.get_web_url()
    print(f"Server URL: {url}")
    print(f"Model: {MODEL_NAME}  (app: {_SLUG}-stt)")
    print(f"Transcription endpoint: {url}/v1/audio/transcriptions\n")

    # /health blocks (up to startup_timeout) until vLLM has loaded the weights and is ready.
    with urllib.request.urlopen(f"{url}/health", timeout=20 * MINUTES) as r:
        assert r.status == 200, f"health check failed: {r.status}"
    print("Health check OK -- model loaded.\n")

    sample = os.environ.get("ASR_SAMPLE")
    if not sample:
        print("Set ASR_SAMPLE to a WAV (local path or https URL) to run a real transcription, e.g.:")
        print("  ASR_SAMPLE=out/chat.wav modal run stt/qwen3_asr.py   # TTS->STT round-trip sanity check")
        print("\nOr call it directly (OpenAI-compatible multipart form):")
        print(f"  curl -s {url}/v1/audio/transcriptions -F model={MODEL_NAME} -F file=@your_audio.wav")
        return

    # requests is a local (client-side) dep; local_entrypoint runs on YOUR machine, not the GPU.
    import requests

    if sample.startswith(("http://", "https://")):
        audio = requests.get(sample, timeout=2 * MINUTES).content
        name = sample.rsplit("/", 1)[-1] or "audio.wav"
    else:
        with open(sample, "rb") as f:
            audio = f.read()
        name = os.path.basename(sample)

    print(f"Transcribing {sample} ({len(audio)} bytes)...")
    resp = requests.post(
        f"{url}/v1/audio/transcriptions",
        files={"file": (name, audio, "audio/wav")},
        data={"model": MODEL_NAME},  # language is auto-detected; add "language": "en" to force English
        timeout=5 * MINUTES,
    )
    assert resp.status_code == 200, f"transcription failed: {resp.status_code} {resp.text[:300]}"
    text = (resp.json().get("text") or "").strip()
    print(f"Transcript: {text!r}\n")
    assert text, "empty transcript -- server returned no text"
    print("END-TO-END OK: server up, audio transcribed.")
