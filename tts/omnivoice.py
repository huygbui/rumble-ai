# tts/omnivoice.py
# Serve OmniVoice (k2-fsa/OmniVoice) TTS on Modal via vLLM-Omni.
# Exposes the OpenAI-compatible POST /v1/audio/speech endpoint on a single small GPU.
#
#   modal deploy tts/omnivoice.py   # persistent endpoint
#   modal serve  tts/omnivoice.py   # ephemeral hot-reload dev server
#   modal run    tts/omnivoice.py   # health-check via local_entrypoint
#
# Why OmniVoice (vs the tts/fish_s2_pro.py baseline):
#   - Apache-2.0 (open commercial use, no contact) vs Fish's research-only license.
#   - ~613M params / 2.45GB weights + ~806MB 24kHz codec -> runs on a 24GB card, not 80GB.
#   - The ONLY candidate with a NATIVE "australian accent" voice-design attribute, plus
#     "child"/"teenager" age attributes -- directly serves the kid + AU-English goal.
# It is pure TTS (no S2S, no built-in safety): all kid-safety/boundary/OOD handling must
# live in your upstream dialogue + moderation layer. See docs/tts-options.md.
#
# STATUS: VERIFIED end-to-end on Modal (2026-06-14), on the 0.23 stack (vLLM 0.23.0 +
# vllm-omni 0.23.0rc1) + A10G. On the older 0.22.0 stack the voice-design `instructions`
# path produced broadband ARTIFACTS (clipping noise, not voice); moving to the 0.23 line
# (PR #4330 adapter) restored a clean voiced-speech output profile. See the version block.
#
# ---------------------------------------------------------------------------------------
# COLD START: GPU MEMORY SNAPSHOTS, WITHOUT vLLM SLEEP MODE (the big lever for this app).
# The disk caches make DOWNLOAD cheap, but they do nothing for the phases that dominate a
# scale-to-zero cold start: loading weights into VRAM, the diffusion pipeline's profiling
# pass, and first-request Triton-JIT (the OmniVoice generator/decoder kernels). GPU memory
# snapshots capture the POST-INITIALIZATION GPU+CPU state and restore it directly, so a
# warm-from-snapshot cold start skips all of that (Modal reports vLLM cold starts dropping
# from ~45s to ~5s). See llm/qwen3_5_4b.py for the plain-vLLM reference of this pattern.
# VERIFIED live on Modal (2026-06-15): one-time snapshot BUILD ~187s, then a scale-from-zero
# RESTORE ~15s (measured cold: 502 -> serving; ~3.3GB stays VRAM-resident so it restores slower
# than the tiny STT app but still ~12x faster than the build), 0 restore failures. The
# OMNIVOICE_CUDA_GRAPH=0 guard (below) held on the deployed 0.23 image -- no cudagraph capture.
#
# WHY "WITHOUT SLEEP" -- this is the key difference from the qwen LLM app:
#   - OmniVoice runs as a single PURE-DIFFUSION stage (vllm_omni/deploy/omnivoice.yaml:
#     stage_type: diffusion, model_class_name OmniVoicePipeline), and that deploy YAML
#     already pins `enforce_eager: true` + `dtype: float32` (auto-resolved -- we pass no
#     --deploy-config). enforce_eager means OmniVoice NEVER runs torch.compile, so THAT
#     blocker (the one that forced --enforce-eager in qwen3_5_4b.py) is already satisfied.
#     CAVEAT: enforce_eager disables only vLLM's OWN cudagraph capture. OmniVoice ships a
#     SEPARATE self-managed cudagraph path (omnivoice_generator.py _OmniVoiceCUDAGraphForward)
#     that is ON by default and is NOT gated by enforce_eager -- captured CUDA graphs are the
#     SAME CRIU/cuda-checkpoint hazard torch.compile was for Qwen. We disable it via env
#     OMNIVOICE_CUDA_GRAPH=0 (see the image .env() guard below) so the snapshot CREATION
#     succeeds; with that guard, a GPU snapshot taken AFTER load+warm captures the warm GPU
#     state cleanly.
#   - The omni serve path does NOT mount vLLM's stock /sleep, /wake_up, /is_sleeping dev
#     endpoints in a usable form. --enable-sleep-mode IS accepted and IS wired to vLLM's
#     CuMemAllocator for the diffusion stage (vllm_omni/worker/diffusion_worker.py
#     _maybe_get_memory_pool_context + sleep/wake_up), but the HTTP surface is the
#     STAGE-AWARE /v1/omni/sleep + /v1/omni/wakeup (body: {"stage_ids":[0]}), NOT the
#     parameterless /sleep the qwen snapshot dance POSTs to. vLLM's stock dev /sleep, even
#     though build_app still mounts it under VLLM_SERVER_DEV_MODE, calls
#     engine_client.sleep(int(level)) positionally -- which binds to AsyncOmni.sleep()'s
#     FIRST param `stage_ids`, a signature mismatch. So the qwen sleep/wake snapshot dance
#     can't be lifted verbatim, and we don't need it: snapshot the WARM (loaded) state.
#   Tradeoff: the snapshot is taken with weights resident in VRAM (no GPU->CPU offload), so
#   it's larger to capture/restore than a slept snapshot. For OmniVoice (~3.3GB total) that
#   is a small, acceptable cost; the win is a far simpler, less fragile cold start.
#
# To get GPU snapshots we use the documented vLLM-on-Modal pattern (same shape as the qwen
# app), minus the sleep/wake steps:
#   - A CLASS (@app.cls) with enable_memory_snapshot=True and
#     experimental_options={"enable_gpu_snapshot": True}  (the latter is ALPHA).
#   - @modal.experimental.http_server (Flash) instead of @modal.web_server -- this is the
#     server shape the GPU-snapshot examples use; it is region-pinned (see REGION).
#   - @modal.enter(snap=True): start vLLM, wait ready, WARM IT UP (bakes the Triton-JIT
#     kernels + first diffusion forward into the snapshot). NO /sleep call.
#   - @modal.enter(snap=False): nothing to wake -- weights were never offloaded -- so the
#     server is ready as soon as the GPU snapshot is restored.
# Still scale-to-zero (MIN_CONTAINERS=0) -- snapshots give a fast cold start WHILE idle is
# free, which is exactly the cost posture we want (no always-warm GPU bill).
#
# Cache note: as in qwen3_5_4b.py, do NOT mount a vLLM compile-cache Volume when
# snapshotting (the snapshot can't reconcile the 9p mount on restore). OmniVoice is
# enforce_eager so there's nothing to compile anyway -- we simply drop that Volume here.

import subprocess
import time
import urllib.error
import urllib.request

import modal
import modal.experimental  # http_server lives here; `import modal` alone does NOT pull it in

# --- Model + serving constants -------------------------------------------------
MODEL_NAME = "k2-fsa/OmniVoice"  # PUBLIC, ungated HF repo (Apache-2.0); the 24kHz codec
#                                  in audio_tokenizer/ downloads automatically with it.
VLLM_PORT = 8091  # vllm-omni's OmniVoice example (run_server.sh) defaults to 8091
N_GPU = 1  # ~613M params fp32 (~2.45GB) + ~806MB codec -> one small card is plenty
GPU = "A10G"  # 24GB is ample for ~613M fp32 + codec; L4 works too. (Fish needed an 80GB
#               card; OmniVoice does not -- big cost win.)
MINUTES = 60  # seconds

# --- Snapshot / Flash knobs ----------------------------------------------------
# GPU snapshots ride on Modal's Flash http_server, which is REGION-PINNED: the GPU
# container and its proxy live in one region. Pick the region nearest your users; the named
# Volume below is global so it works from any region. (Same posture as llm/qwen3_5_4b.py.)
REGION = "us-east"
MIN_CONTAINERS = 0  # scale to ZERO when idle (no GPU bill); snapshots make the cold start fast
TARGET_INPUTS = 8  # Flash autoscale target: scale out when a container exceeds ~8 in-flight
#                    requests. Small/fast model; tune up/down.

# --- vLLM / vllm-omni version pair ---------------------------------------------
# OmniVoice's clean online voice-design/cloning adapter (PR #4330: maps `instructions`
# = accent/age/pitch and `ref_audio`/`ref_text` onto /v1/audio/speech) is ONLY in the
# 0.23 line, NOT in 0.22.0. On 0.22.0 the voice-design path produced high-energy
# ARTIFACTS (noise, not a conditioned voice) -- verified by listening 2026-06-14. So we
# run OmniVoice on the 0.23.0rc1 tag, installed from GIT (it is NOT on PyPI yet), paired
# with vLLM 0.23.0 (which IS on PyPI). vllm-omni declares no vllm dep, so we re-pin vllm
# explicitly -- same discipline as the Fish recipe.
# (Qwen3-TTS deliberately stays on the 0.22.0 STABLE stack -- it works well there; no
# reason to chase an RC for a working setup. Only the broken model moves.)
VLLM_VERSION = "0.23.0"  # on PyPI; pairs with the vllm-omni 0.23 line
VLLM_OMNI_REF = "v0.23.0rc1"  # git tag carrying PR #4330 (online voice-design adapter)

# --- Container image -----------------------------------------------------------
# Much simpler than the Fish image: OmniVoice has none of fish-speech's einx/pydantic/
# protobuf conflicts, and vLLM-Omni loads the model itself (no standalone `omnivoice`
# pip package needed for the serve path). We add ffmpeg (OmniVoice audio I/O) alongside
# the libsndfile1 the Fish image already used.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git", "ffmpeg", "libsndfile1")
    # We deliberately do NOT pass `--torch-backend=auto`: Modal's image builder has no
    # GPU, so `auto` would install a CPU-only torch that fails on the GPU at runtime.
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",
    )
    .uv_pip_install(
        # Install vllm-omni from the git tag (the 0.23.0rc1 wheel is not on PyPI yet).
        # It is pure-Python (py3-none-any), so this clones + installs without compiling.
        f"vllm-omni @ git+https://github.com/vllm-project/vllm-omni@{VLLM_OMNI_REF}",
        f"vllm=={VLLM_VERSION}",  # re-pin so vllm-omni's resolve can't drift vllm
        "fastapi<0.137",  # FastAPI 0.137 breaks vLLM's Prometheus middleware -> every request 500s; see llm/qwen3_5_4b.py
        "huggingface_hub[hf_transfer]",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # SNAPSHOT-CREATION GUARD (the OmniVoice analog of qwen3_5_4b.py's --enforce-eager).
            # enforce_eager (auto-set in omnivoice.yaml) only disables vLLM's OWN cudagraph
            # capture -- it does NOT touch OmniVoice's SEPARATE, self-managed cudagraph path.
            # OmniVoice's generator pre-captures CUDA graphs for fixed seq-length buckets
            # (vllm_omni .../models/omnivoice/omnivoice_generator.py _OmniVoiceCUDAGraphForward),
            # gated NOT by enforce_eager but by config.enable_cuda_graph, which DEFAULTS ON
            # (configs/omnivoice.py reads env OMNIVOICE_CUDA_GRAPH, "1" unless overridden).
            # Captured CUDA graphs hold raw pool handles + replay state that CRIU/cuda-checkpoint
            # cannot round-trip cleanly -- the SAME failure class that made torch.compile break
            # GPU-snapshot CREATION for Qwen (every cold start then silently full-rebuilds with
            # NO snapshot benefit). Force the eager-replay path before we snapshot by disabling
            # those graphs. Cost: a small per-step latency bump, negligible for this voice loop
            # and well worth a snapshot that actually engages. (Drop this only if a live deploy
            # proves OmniVoice's captured graphs survive cuda-checkpoint.)
            "OMNIVOICE_CUDA_GRAPH": "0",
        }
    )
)

# --- Caches -------------------------------------------------------------------
# ONLY the HF weights volume is mounted. As in llm/qwen3_5_4b.py we deliberately do NOT
# mount a Volume for vLLM's torch.compile cache when snapshotting: the snapshot can't
# reconcile a 9p mount on restore ("CompleteRestore ... failed to walk torch_compile_cache")
# and falls back to a full cold boot. For OmniVoice this costs nothing -- the diffusion
# stage runs enforce_eager (no torch.compile), so there's no compile cache to keep anyway.
# The HF weights are downloaded/committed once and stable, so that Volume restores fine.
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App("omnivoice-tts")


# --- In-container helpers (talk to the local vLLM-Omni over loopback) ------------
# Pure stdlib (urllib) so they have zero dependency surface inside the container.
def _vllm_url(path: str) -> str:
    return f"http://127.0.0.1:{VLLM_PORT}{path}"


def _check_running(p: subprocess.Popen) -> None:
    # Fail fast if vLLM died during startup instead of silently waiting out the timeout.
    rc = p.poll()
    if rc is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def _wait_ready(p: subprocess.Popen, timeout: int = 15 * MINUTES) -> None:
    # Poll /health until vLLM-Omni has loaded the model and is serving (or the proc dies).
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
    raise TimeoutError(f"vLLM-Omni not ready within {timeout}s")


def _warmup() -> None:
    # Synthesize a couple of REAL short utterances so the one-time Triton-JIT kernels (the
    # OmniVoice generator's iterative-unmask step + the RVQ/DAC decoder) compile and the
    # first diffusion forward is exercised BEFORE the snapshot is taken -- this is what
    # bakes the "first request eats JIT spikes" cost into the snapshot. We read the WAV
    # body so the full synth path (not just request acceptance) is warmed.
    import json

    payload = json.dumps(
        {
            "input": "G'day! This is a warmup line.",
            "instructions": "child, australian accent, high pitch",
            "language": "English",
            "response_format": "wav",
        }
    ).encode()
    req = urllib.request.Request(
        _vllm_url("/v1/audio/speech"),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for _ in range(2):
        with urllib.request.urlopen(req, timeout=120) as r:
            r.read()


@app.cls(
    image=image,
    gpu=f"{GPU}:{N_GPU}",
    # No secret needed: k2-fsa/OmniVoice is a PUBLIC HF repo, downloads without a token.
    volumes={
        # HF weights only -- the vLLM compile cache stays container-local (see Caches note).
        "/root/.cache/huggingface": hf_cache_vol,
    },
    scaledown_window=2 * MINUTES,  # stay warm 2 min after last request (cost-saving; chat keepalive holds it open)
    timeout=20 * MINUTES,
    min_containers=MIN_CONTAINERS,
    region=REGION,
    # The two snapshot switches: CPU memory checkpoint + the ALPHA GPU memory checkpoint.
    # No --enable-sleep-mode needed: we snapshot the WARM (weights-resident) state directly.
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.experimental.http_server(
    port=VLLM_PORT,
    proxy_regions=[REGION],
    # First (pre-snapshot) boot loads + warms vLLM-Omni, which takes minutes; the 30s
    # default would time out. Restores from snapshot come up in seconds, well under this.
    startup_timeout=20 * MINUTES,
    exit_grace_period=5,
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class OmniVoiceVllm:
    @modal.enter(snap=True)
    def startup(self):
        # Runs BEFORE the snapshot is captured. Launch vLLM-Omni, wait ready, then warm it
        # so the snapshot is taken in the fully-warm (loaded + JIT-baked) state.
        # OmniVoice-specific deltas vs the qwen LLM app:
        #   --omni              : MANDATORY (enables the TTS/diffusion stack), same as Fish.
        #   --trust-remote-code : REQUIRED here -- OmniVoice ships custom modeling code
        #                         (model_type "omnivoice"); Fish s2-pro did not need it.
        # OmniVoice's deploy config (dtype float32, enforce_eager) is auto-resolved from the
        # model type, so -- unlike qwen3_tts -- no explicit --deploy-config is passed, and we
        # pass NO --enforce-eager (it's already on in the YAML). Do NOT override dtype. Add
        # --gpu-memory-utilization here only if you hit OOM/over-reserve. We deliberately do
        # NOT pass --enable-sleep-mode: the omni server has no parameterless /sleep to drive
        # the qwen-style sleep/wake dance, and snapshotting the warm GPU state is simpler.
        # The snapshot-creation guard is NOT a serve flag here -- it's the image env
        # OMNIVOICE_CUDA_GRAPH=0 (see .env() above), which disables OmniVoice's own
        # self-managed CUDA graphs that enforce_eager does NOT cover. Without that guard the
        # snapshot can silently fail to engage (see the COLD START block).
        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--omni",
            "--trust-remote-code",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
        ]
        if N_GPU > 1:
            cmd += ["--tensor-parallel-size", str(N_GPU)]
        self.process = subprocess.Popen(cmd)
        _wait_ready(self.process)
        _warmup()

    @modal.enter(snap=False)
    def restore(self):
        # Runs AFTER the snapshot is restored. Nothing to wake: weights were never offloaded
        # (no sleep), so the GPU snapshot comes back with the model resident and the server
        # is immediately ready to synthesize.
        pass

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.local_entrypoint()
def main():
    # Smoke test: confirm the server is up, then print example request payloads.
    # (Audio synthesis itself is best driven from a small client like client.py.)
    # http_server (Flash) URLs are NOT the classic `<workspace>--app-fn.modal.run` form;
    # fetch them from the class. (Also printed in the `modal deploy` output.)
    url = OmniVoiceVllm._experimental_get_flash_urls()[0].rstrip("/")
    print(f"Server URL: {url}")
    print(f"Speech endpoint: {url}/v1/audio/speech")
    # /health blocks (up to startup_timeout) until vLLM-Omni has loaded the model.
    with urllib.request.urlopen(f"{url}/health", timeout=20 * MINUTES) as r:
        assert r.status == 200, f"health check failed: {r.status}"
    print("Health check OK.\n")

    print("Example requests (POST JSON to /v1/audio/speech, response is WAV):\n")
    print("# 1) Voice DESIGN -- the kid + Australian-English path. Attributes go in")
    print("#    `instructions` as a comma-separated string; non-verbals like [laughter]")
    print("#    go inline in `input`. Accent vocab incl. 'australian accent'; age incl.")
    print("#    'child'/'teenager'; pitch 'very low'..'very high'. (Needs the #4330")
    print("#    online adapter -- see the version block if it has no effect.)")
    print(
        '  {"input": "G\'day! Wanna hear a story? [laughter]",\n'
        '   "instructions": "child, australian accent, high pitch",\n'
        '   "language": "English", "response_format": "wav"}\n'
    )
    print("# 2) Zero-shot CLONING -- the more reliable consistent-AU-voice path. Supply a")
    print("#    consented AU reference clip; ref_text optional (Whisper auto-transcribes).")
    print(
        '  {"input": "This line is spoken in the cloned voice.",\n'
        '   "ref_audio": "https://…/au_reference.wav", "ref_text": "transcript of the clip",\n'
        '   "language": "English", "response_format": "wav"}'
    )
