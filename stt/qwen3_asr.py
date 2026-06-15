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
# ---------------------------------------------------------------------------------------
# COLD START: GPU MEMORY SNAPSHOTS (the big lever).
# The HF weights cache makes DOWNLOAD cheap, but it does nothing for the phases that
# dominate a scale-to-zero cold start: loading weights into VRAM, memory profiling, and the
# first-request Triton-JIT / audio-decoder kernel compiles. GPU memory snapshots capture the
# POST-INITIALIZATION GPU+CPU state and restore it directly, so a warm-from-snapshot cold
# start skips all of that. This mirrors llm/qwen3_5_4b.py, which is plain vLLM on the SAME
# 0.23 stack -- Qwen3-ASR is also plain vLLM (vllm[audio]) serving over the OpenAI API, so the
# pattern (sleep mode + /sleep + /wake_up) transfers DIRECTLY. There (the LLM) this took the
# scale-from-zero cold start from ~330s to ~25s while STILL scaling to zero.
#
# To get GPU snapshots we need the documented vLLM pattern, which is why this file no longer
# uses the simpler @modal.web_server + Popen shape it started with:
#   - A CLASS (@app.cls) with enable_memory_snapshot=True and
#     experimental_options={"enable_gpu_snapshot": True}  (the latter is ALPHA).
#   - @modal.experimental.http_server (Flash) instead of @modal.web_server -- this is the
#     server shape the GPU-snapshot examples use; it is region-pinned (see REGION).
#   - @modal.enter(snap=True): start vLLM (--enable-sleep-mode), wait ready, WARM IT UP with a
#     real transcription (bakes the audio-decoder + Triton-JIT kernels into the snapshot), then
#     PUT IT TO SLEEP (/sleep) so the snapshot is taken with weights offloaded GPU->CPU.
#   - @modal.enter(snap=False): WAKE IT UP (/wake_up) after the snapshot is restored.
# vLLM's /sleep + /wake_up are dev endpoints -- they exist only with VLLM_SERVER_DEV_MODE=1
# (set in the image env below). The snapshot becomes available after a few cold starts.
# Still scale-to-zero (MIN_CONTAINERS=0) -- snapshots give a fast cold start WHILE idle is
# free, which is exactly the cost posture we want (no always-warm GPU bill).
#
# STATUS: VERIFIED end-to-end on Modal (2026-06-15) on an L4, on the earlier plain web_server
# shape. Bring-up findings (still hold):
#   - vLLM 0.23.0 loads Qwen3-ASR NATIVELY (engine config trust_remote_code=False) -- no version
#     bump, no --trust-remote-code. (Qwen3-ASR is Qwen3-Omni-thinker-based; the engine logs a
#     thinker_config + a deprecated-WhisperFeatureExtractor notice -- both benign.)
#   - The image MUST install vllm[audio] (NOT plain vllm): the /v1/audio/transcriptions decoder
#     needs PyAV (`av`) + soundfile, else every request 400s "Invalid or unsupported audio file"
#     ("No module named 'av'"). This bit the first attempt; fixed in the image below. KEEP [audio].
#   - fastapi<0.137 pin held (no Prometheus 500s), same as the sibling vLLM apps.
#   - Round-trip sanity: transcribing the Fish TTS clip out/01_short_greeting.wav returned
#     'Hello, welcome to the Fish Speech S Two Pro demo. How are you doing today?' -- near-perfect.
#   - Footprint: 1.53 GiB VRAM on the 24GB L4 -> huge headroom.
# The GPU-snapshot rewrite here MIRRORS the VERIFIED llm/qwen3_5_4b.py pattern and is now ALSO
# VERIFIED live on Modal for THIS app (2026-06-15): one-time snapshot BUILD ~236s, then a
# scale-from-zero RESTORE in ~1-2s (the 0.6B is tiny, so it restores even faster than the 4B
# LLM's ~25s), 0 restore failures, sleep/wake working. TWO gotchas are REQUIRED to make the
# snapshot actually engage (both encoded below); without either, every cold start silently
# full-rebuilds with NO snapshot benefit:
#   (1) Do NOT mount the vLLM compile-cache Volume -- the snapshot can't reconcile the 9p
#       mount on restore ("CompleteRestore ... failed to walk torch_compile_cache"). The
#       vllm-cache Volume that the old web_server shape mounted is DROPPED here on purpose.
#   (2) --enforce-eager -- torch.compile breaks GPU memory snapshot CREATION, so the snapshot
#       never becomes usable. (See the Caches note and the serve-flags comment.)

import os
import re
import subprocess
import time
import urllib.error
import urllib.request

import modal
import modal.experimental  # http_server lives here; `import modal` alone does NOT pull it in

# --- Model + serving constants -------------------------------------------------
# Env-driven so you can A/B 0.6B vs 1.7B without touching code. Read on the CLIENT at deploy
# time AND re-read in the container -- where it resolves from the image env baked below, so both
# sides agree (a plain default would otherwise make the container fall back to 0.6B).
MODEL_NAME = os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")  # PUBLIC, ungated, Apache-2.0
VLLM_PORT = 8000  # plain vLLM's default (same as llm/qwen3_5_4b.py); ASR endpoint is /v1/audio/transcriptions
N_GPU = 1  # 0.6B (~1.2GB fp16) / 1.7B (~4GB bf16) -- one cheap 24GB card is plenty either way
GPU = "L4"  # 24GB, ~$0.80/hr -- cheapest fit; same tier as the Qwen3.5-4B LLM. A10G is a drop-in
#             alternative. The 0.6B leaves huge headroom (raise concurrency).
MINUTES = 60  # seconds

# Distinct Modal app per model id -> 0.6B and 1.7B get separate URLs and can run side by side
# for A/B. "Qwen/Qwen3-ASR-0.6B" -> "qwen3-asr-0-6b-stt".
_SLUG = re.sub(r"[^a-z0-9]+", "-", MODEL_NAME.split("/")[-1].lower()).strip("-")

# Context length. The old shape ran UNCAPPED (vLLM's default 65536), which reserves a large KV
# cache. We leave it uncapped here on purpose: Qwen3-ASR's audio context is consumed as encoded
# audio tokens and capping --max-model-len too low would TRUNCATE longer utterances. The voice
# loop sends short turns, so the unused KV headroom is the only cost; if you want to shrink the
# snapshot / raise concurrency, measure the longest real utterance first, then cap with margin.
# (Left as None == do not pass --max-model-len; see serve flags below.)
MAX_MODEL_LEN = None

# --- Snapshot / Flash knobs ----------------------------------------------------
# GPU snapshots ride on Modal's Flash http_server, which is REGION-PINNED: the GPU container
# and its proxy live in one region. Pick the region nearest your users; the named HF Volume
# below is global so it works from any region.
REGION = "us-east"
MIN_CONTAINERS = 0  # scale to ZERO when idle (no GPU bill); snapshots make the cold start fast
TARGET_INPUTS = 8  # Flash autoscale target: scale out when a container exceeds ~8 in-flight
#                    requests. 0.6B is light + vLLM continuous-batches well; tune up/down.

# --- vLLM version --------------------------------------------------------------
# Match the rest of the repo. See STATUS note #1 re: bumping.
VLLM_VERSION = "0.23.0"

# --- Container image -----------------------------------------------------------
# Same CUDA 12.8 base + hf_transfer discipline as llm/qwen3_5_4b.py, plus audio I/O libs
# (ffmpeg/libsndfile1 + vllm[audio]'s av/soundfile/librosa) that vLLM needs to decode uploaded
# audio for the transcription endpoint -- the one real delta from the text-LLM image.
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
        # way on Modal, 2026-06-15.) KEEP the [audio] extra -- the snapshot does not change this.
        f"vllm[audio]=={VLLM_VERSION}",
        # Pin FastAPI < 0.137: vLLM 0.23.0's Prometheus middleware 500s every request on 0.137's
        # new lazy router. Drop the pin once the instrumentator ships a fix. See llm/qwen3_5_4b.py.
        "fastapi<0.137",
        "huggingface_hub[hf_transfer]",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Bake the resolved model id into the container so the class loads the SAME model that
            # was deployed (the module-level os.environ read above resolves to THIS in-container).
            "QWEN_ASR_MODEL": MODEL_NAME,
            # REQUIRED for the snapshot dance: /sleep + /wake_up are vLLM "dev" endpoints that are
            # only mounted when this is set. Without it the @modal.enter sleep/wake calls below 404
            # and no GPU snapshot is taken.
            "VLLM_SERVER_DEV_MODE": "1",
        }
    )
)

# --- Caches -------------------------------------------------------------------
# ONLY the HF weights volume is mounted. We deliberately DROPPED the vLLM torch.compile cache
# Volume (/root/.cache/vllm) that the old web_server shape mounted -- it is the bug that breaks
# GPU snapshots:
#   --enable-sleep-mode forces vLLM's cumem allocator, which CHANGES the torch.compile cache
#   key. So the snapshot-build boot gets a cache MISS, writes fresh files into the compile
#   cache, and if that cache is a Modal Volume (9p) the snapshot freezes a process referencing
#   files that are NOT in the Volume's committed state, and RESTORE aborts with:
#     vfs.CompleteRestore() ... 9p: failed to walk ".../torch_compile_cache/...": no such file
#   -> it then falls back to a full cold boot, i.e. no snapshot win at all.
# Keeping the compile cache on the container's OWN filesystem lets the snapshot capture the
# compiled graphs in-process (no 9p mount to reconcile), so restores succeed. (We also pass
# --enforce-eager below, which disables torch.compile entirely -- belt and braces.) The HF
# weights are downloaded/committed once and are stable, so that Volume restores fine and stays.
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App(f"{_SLUG}-stt")


# --- In-container helpers (talk to the local vLLM over loopback) ----------------
# Pure stdlib (urllib) so they have zero dependency surface inside the container.
def _vllm_url(path: str) -> str:
    return f"http://127.0.0.1:{VLLM_PORT}{path}"


def _post(path: str, body: bytes = b"", timeout: int = 60) -> None:
    req = urllib.request.Request(
        _vllm_url(path),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()


def _check_running(p: subprocess.Popen) -> None:
    # Fail fast if vLLM died during startup instead of silently waiting out the timeout.
    rc = p.poll()
    if rc is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def _wait_ready(p: subprocess.Popen, timeout: int = 15 * MINUTES) -> None:
    # Poll /health until vLLM has loaded the weights and is serving (or the proc dies).
    deadline = time.time() + timeout
    while time.time() < deadline:
        _check_running(p)
        try:
            with urllib.request.urlopen(_vllm_url("/health"), timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM not ready within {timeout}s")


def _tiny_wav_bytes(seconds: float = 0.3, sample_rate: int = 16000) -> bytes:
    # Build a tiny 16kHz mono PCM16 WAV entirely in-process with the stdlib (no audio libs, no
    # disk). ASR warmup needs REAL audio to exercise the decoder + encoder kernels, so we
    # synthesize a short low-amplitude 440Hz sine (a touch of signal beats pure silence at
    # waking the feature extractor). Returned as raw WAV bytes for the multipart upload below.
    import io
    import math
    import struct
    import wave

    n = int(seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit PCM
        w.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n):
            sample = int(0.05 * 32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
            frames += struct.pack("<h", sample)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def _post_multipart(path: str, fields: dict, file_field: str,
                    file_name: str, file_bytes: bytes,
                    content_type: str = "audio/wav", timeout: int = 120) -> None:
    # Minimal multipart/form-data POST (stdlib only) for the OpenAI-compatible
    # /v1/audio/transcriptions endpoint, which takes an uploaded file + form fields (the JSON
    # _post helper above can't do file uploads). Used by warmup; the local_entrypoint uses the
    # `requests` client lib instead (it runs on your laptop, not the GPU).
    boundary = "----rumbleaiBoundary7MA4YWxkTrZu0gW"
    parts = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\n")
        parts.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n')
        parts.append(f"{v}\r\n")
    head = "".join(parts).encode()
    file_head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{file_field}"; filename="{file_name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    body = head + file_head + file_bytes + tail
    req = urllib.request.Request(
        _vllm_url(path),
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()


def _warmup() -> None:
    # Send a couple of REAL transcriptions on a tiny synthesized clip so the one-time audio
    # decoder + feature-extractor + Triton-JIT kernels compile and the CUDA paths are exercised
    # BEFORE the snapshot is taken -- this bakes the "first request eats JIT spikes" cost into
    # the snapshot. (A short low-volume sine; the transcript itself is irrelevant.)
    wav = _tiny_wav_bytes()
    for _ in range(2):
        _post_multipart(
            "/v1/audio/transcriptions",
            fields={"model": MODEL_NAME},
            file_field="file",
            file_name="warmup.wav",
            file_bytes=wav,
            timeout=120,
        )


def _sleep(level: int = 1) -> None:
    # level=1: offload weights GPU->CPU (kept in RAM), drop KV cache. The CPU memory snapshot
    # captures the RAM; the GPU snapshot then has far less to checkpoint.
    _post(f"/sleep?level={level}")


def _wake_up() -> None:
    _post("/wake_up")


@app.cls(
    image=image,
    gpu=f"{GPU}:{N_GPU}",
    # No secret needed: Qwen/Qwen3-ASR-* are PUBLIC HF repos, download without a token.
    volumes={
        # HF weights only -- the vLLM compile cache stays container-local (see Caches note).
        "/root/.cache/huggingface": hf_cache_vol,
    },
    # Idle-cost control: stay warm 2 min after the last request, then scale to ZERO. STT is hit
    # at the START of every user turn, so during an active conversation (turns seconds apart) the
    # 2-min window holds it warm; between conversations it idles to zero. Snapshots make the
    # cold start fast, so scale-to-zero stays cheap WITHOUT a big first-turn latency hit.
    scaledown_window=2 * MINUTES,
    timeout=20 * MINUTES,
    min_containers=MIN_CONTAINERS,
    region=REGION,
    # The two snapshot switches: CPU memory checkpoint + the ALPHA GPU memory checkpoint.
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.experimental.http_server(
    port=VLLM_PORT,
    proxy_regions=[REGION],
    # First (pre-snapshot) boot loads + warms + sleeps vLLM, which takes minutes; the 30s
    # default would time out. Restores from snapshot come up in seconds, well under this.
    startup_timeout=20 * MINUTES,
    exit_grace_period=5,
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class QwenAsrVllm:
    @modal.enter(snap=True)
    def startup(self):
        # Runs BEFORE the snapshot is captured. Launch vLLM, warm it with a real transcription,
        # then sleep so the snapshot is taken in the offloaded state. Base command follows the
        # documented vLLM recipe: `vllm serve Qwen/Qwen3-ASR-0.6B` (no --trust-remote-code --
        # vLLM 0.23.0 loads Qwen3-ASR natively), plus the two snapshot-required flags:
        #   --enable-sleep-mode : REQUIRED for the /sleep + /wake_up snapshot dance.
        #   --enforce-eager     : REQUIRED for GPU snapshots to actually engage. Modal's docs warn
        #     "torch.compile can cause Memory Snapshot creation to fail" -- vLLM's inductor compile
        #     spawns multi-process workers that CRIU can't checkpoint, so the snapshot silently
        #     fails to become usable and EVERY cold start full-rebuilds. --enforce-eager skips
        #     torch.compile AND CUDA-graph capture, so snapshot creation succeeds and restores are
        #     fast. Cost is modestly lower throughput -- negligible for a single-user voice loop
        #     (ASR is already a small model with a low RTF). It also drops the ~67s compile from
        #     every build.
        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--enable-sleep-mode",
            "--enforce-eager",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
        ]
        # Only cap context if MAX_MODEL_LEN is set -- left None by default so we don't truncate
        # ASR audio context (see the MAX_MODEL_LEN note above).
        if MAX_MODEL_LEN is not None:
            cmd += ["--max-model-len", str(MAX_MODEL_LEN)]
        if N_GPU > 1:
            cmd += ["--tensor-parallel-size", str(N_GPU)]
        # Optional flags to try if first boot misbehaves (see STATUS in the header):
        #   "--trust-remote-code",              # if vLLM ever rejects the Qwen3-ASR audio arch
        #   "--gpu-memory-utilization", "0.8",  # the repo's qwen-asr-serve wrapper uses 0.8
        self.process = subprocess.Popen(cmd)
        _wait_ready(self.process)
        _warmup()
        _sleep(level=1)

    @modal.enter(snap=False)
    def restore(self):
        # Runs AFTER the snapshot is restored (GPU state already back). Move weights CPU->GPU so
        # the server is immediately ready to transcribe.
        _wake_up()

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.local_entrypoint()
def main():
    # Smoke test: confirm the server is up. If ASR_SAMPLE is set (local WAV path or http URL),
    # also run a REAL transcription and assert a non-empty transcript -- the end-to-end check.

    # http_server (Flash) URLs are NOT the classic `<workspace>--app-fn.modal.run` form;
    # fetch them from the class. (Also printed in the `modal deploy` output.)
    url = QwenAsrVllm._experimental_get_flash_urls()[0].rstrip("/")
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

    # httpx is a local (client-side) dep; local_entrypoint runs on YOUR machine, not the GPU.
    import httpx

    if sample.startswith(("http://", "https://")):
        audio = httpx.get(sample, timeout=2 * MINUTES).content
        name = sample.rsplit("/", 1)[-1] or "audio.wav"
    else:
        with open(sample, "rb") as f:
            audio = f.read()
        name = os.path.basename(sample)

    print(f"Transcribing {sample} ({len(audio)} bytes)...")
    resp = httpx.post(
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
