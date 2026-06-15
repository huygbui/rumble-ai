# tts/fish_s2_pro.py
# Serve Fish Speech S2 Pro (fishaudio/s2-pro) TTS on Modal via vLLM-Omni.
# Exposes the OpenAI-compatible POST /v1/audio/speech endpoint on a single 80GB GPU.
#
#   modal deploy tts/fish_s2_pro.py   # persistent endpoint
#   modal serve  tts/fish_s2_pro.py   # ephemeral hot-reload dev server
#   modal run    tts/fish_s2_pro.py   # health-check via local_entrypoint

import subprocess

import modal

# --- Model + serving constants -------------------------------------------------
MODEL_NAME = "fishaudio/s2-pro"  # public HF repo; Fish Audio Research License governs *use* (research/non-commercial free)
VLLM_PORT = 8091  # recipe uses 8091 for s2-pro
N_GPU = 1  # ~48.9 GiB peak -> one 80GB card is enough
MINUTES = 60  # seconds

# Pin the vLLM / vllm-omni pair. vllm-omni is version-coupled to vllm: its releases
# version-track the vLLM minor they rebase onto (e.g. vllm-omni 0.16.0 rebased onto
# vLLM 0.16), so vllm-omni 0.22.0 pairs with the vLLM 0.22 line -- we use vllm 0.22.1
# (latest 0.22.x stable). vllm-omni 0.22.0's PyPI metadata declares NO vllm
# dependency, so installing vllm first is not silently overridden -- but we still
# re-pin vllm explicitly in BOTH uv steps below so the resolver can never drift it.
# This 0.22.x stack fixed the s2-pro long-form truncation bug (vllm-omni #2248) that
# capped output at ~5-9s on the older 0.19.0 build; long-form output now works.
VLLM_VERSION = "0.22.1"  # latest 0.22.x stable; pairs with vllm-omni 0.22.0
VLLM_OMNI_VERSION = "0.22.0"  # latest stable PyPI release; "improved s2-pro path"

# --- Container image -----------------------------------------------------------
# CUDA 12.8 base. The PyPI torch that vllm pulls bundles its own CUDA runtime libs,
# so the base image mainly supplies the toolkit/driver surface; 12.8 is compatible
# with the vllm 0.22.x default torch. If startup ever fails with a CUDA/torch ABI
# error, bump this base tag to match torch's CUDA build.
#
# Install order matters -- the dependency tensions below are inherent to pairing
# vllm-omni with fish-speech, independent of version:
#   1. vllm pinned to 0.22.1. We deliberately do NOT pass `--torch-backend=auto`:
#      that flag detects CUDA from the *build* host, but Modal's image builder has no
#      GPU, so `auto` installs a CPU-only torch that fails to import on the GPU at
#      runtime. Omitting it pulls the default PyPI (CUDA) torch.
#   2. vllm-omni 0.22.0 from PyPI (provides the --omni TTS stack). We re-pin
#      vllm==0.22.1 in this step too so uv cannot drift it while resolving
#      vllm-omni's deps (x-transformers>=2.12.2 -> einx>=0.3.0, etc.).
#   3. fish-speech (DAC codec) installed SEPARATELY with permissive pip. It is NOT in
#      the uv steps: fish-speech hard-pins einx==0.2.2, mutually exclusive with
#      vllm-omni's einx>=0.3.0, so uv's strict resolver would abort. pip just
#      downgrades einx to 0.2.2 (what the s2-pro DAC path needs) and warns.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("libportaudio2", "portaudio19-dev", "git", "libsndfile1")
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",
    )
    .uv_pip_install(
        f"vllm-omni=={VLLM_OMNI_VERSION}",
        f"vllm=={VLLM_VERSION}",  # re-pin so vllm-omni's resolve can't drift vllm
        "fastapi<0.137",  # FastAPI 0.137 breaks vLLM's Prometheus middleware -> every request 500s; see llm/qwen3_5_4b.py
        "huggingface_hub[hf_transfer]",
    )
    # fish-speech pulls pyaudio, which builds a C extension -> needs a compiler.
    # Add it AFTER the uv steps so the cached vllm/vllm-omni layers aren't
    # invalidated; portaudio19-dev (its headers) is already in the apt step above.
    .apt_install("build-essential", "clang")
    # Permissive pip step (NOT uv): tolerates the einx==0.2.2 vs >=0.3.0 conflict.
    .pip_install("fish-speech")
    # fish-speech's heavy dep tree (gradio/tensorboard/wandb/...) downgrades two libs
    # that vllm needs newer, which would crash vllm at import:
    #   - pydantic: fish-speech hard-pins ==2.9.2; vllm needs >=2.12.0.
    #   - protobuf: pulled down to 3.19.6; vllm needs >=5.29.6 (protobuf 3 vs 5
    #     generated code is ABI-incompatible).
    # Restore both LAST. Safe: the serving path is vllm + the DAC codec
    # (descript-audio-codec), not fish-speech's pydantic schemas / protobuf users
    # (tensorboard/wandb/gradio webui), none of which run during TTS serving.
    # (These two pip resolver "ERROR" lines at build time are expected and benign.)
    # Backstop fastapi<0.137 here too: fish-speech's gradio dep can pull fastapi forward,
    # and 0.137 breaks vLLM's Prometheus middleware (every request 500s) -- same restore-
    # LAST discipline as pydantic/protobuf above.
    .pip_install("pydantic>=2.12.0", "protobuf>=5.29.6,<6", "fastapi<0.137")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Escape hatch: if startup fails with Triton/attention errors, disable the
            # Fish KV-cache attention fast path by uncommenting the next line.
            # "=required" hard-fails if the path is unavailable; "=0" disables it.
            # (On the 0.22.x stack the fast path installs cleanly, so this stays off.)
            # "VLLM_OMNI_FISH_KVCACHE_ATTN": "0",
        }
    )
)

# --- Caches (persisted across cold starts) ------------------------------------
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("fish-s2-pro-tts")


@app.function(
    image=image,
    gpu=f"H100:{N_GPU}",  # any 80GB card (H100/A100-80GB/A800); s2-pro needs ~49GB peak
    # No secret needed: fishaudio/s2-pro is a PUBLIC HF repo (gated=false), so the
    # weights download without an HF token. If you later swap in a gated model, add:
    #   secrets=[modal.Secret.from_name("huggingface-secret")],  # injects HF_TOKEN
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    # Idle-cost control: after the last request the container stays warm for this
    # long, then scales to ZERO -> no GPU cost while idle. 5 min trims the idle tail
    # while still absorbing brief gaps between requests in a session. Cold start is
    # ~150s (weights load from cache + model init; the first boot after a vllm bump
    # also recompiles CUDA graphs once). Raise this if cold starts bite too often;
    # lower it (1-2 min) to save more when usage is sparse.
    scaledown_window=5 * MINUTES,
    timeout=20 * MINUTES,
    # min_containers=1,  # KEEP COMMENTED: always-warm = 24/7 GPU cost. Uncomment only
    # if you need zero cold-start latency and accept paying continuously.
)
@modal.concurrent(max_inputs=8)  # vLLM batches; s2-pro is heavy so keep modest
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve():
    # Non-blocking launch: Modal's web_server waits for the port, the function returns.
    # Recipe-exact single-GPU invocation: `vllm serve fishaudio/s2-pro --omni
    # --host 0.0.0.0 --port 8091`. --omni is MANDATORY (enables the TTS stack).
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--omni",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]
    if N_GPU > 1:
        cmd += ["--tensor-parallel-size", str(N_GPU)]
    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
def main():
    # Smoke test: confirm the server is up. Audio synthesis is done by client.py.
    import urllib.request

    url = serve.get_web_url()
    print(f"Server URL: {url}")
    print(f"OpenAI base_url: {url}/v1")
    print(f"Speech endpoint: {url}/v1/audio/speech")
    with urllib.request.urlopen(f"{url}/health", timeout=20 * MINUTES) as r:
        assert r.status == 200, f"health check failed: {r.status}"
    print("Health check OK.")
