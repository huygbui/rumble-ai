# fish_s2_pro_modal.py
# Serve Fish Speech S2 Pro (fishaudio/s2-pro) TTS on Modal via vLLM-Omni.
# Exposes the OpenAI-compatible POST /v1/audio/speech endpoint on a single 80GB GPU.
#
#   modal deploy fish_s2_pro_modal.py   # persistent endpoint
#   modal serve  fish_s2_pro_modal.py   # ephemeral hot-reload dev server
#   modal run    fish_s2_pro_modal.py   # health-check via local_entrypoint

import subprocess

import modal

# --- Model + serving constants -------------------------------------------------
MODEL_NAME = "fishaudio/s2-pro"  # public HF repo; Fish Audio Research License governs *use* (research/non-commercial free)
VLLM_PORT = 8091  # recipe uses 8091 for s2-pro
N_GPU = 1  # ~48.9 GiB peak -> one 80GB card is enough
MINUTES = 60  # seconds

# Pin to the s2-pro recipe's exact tested combo: vLLM 0.19.0 + a SPECIFIC vllm-omni
# commit. vllm-omni is NOT independent of vllm -- its setup.py manages the vllm
# dependency dynamically, and PyPI's latest vllm-omni (0.22.0) pairs with vllm 0.23.0.
# Mixing a hard vllm==0.19.0 pin with an unpinned `vllm-omni` from PyPI produces a
# resolver conflict / mismatched install, so we pin both to the recipe's combo.
VLLM_VERSION = "0.19.0"  # recipe-stated minimum + tested version for s2-pro
VLLM_OMNI_COMMIT = "c93359bb354a6aa5c14d062430cb85b2c4db251e"  # recipe-pinned commit

# --- Container image -----------------------------------------------------------
# CUDA 12.8 per recipe. Install order matters:
#   1. vllm pinned to 0.19.0 with --torch-backend=auto so uv resolves the torch
#      wheel matching the CUDA 12.8 base image (vllm compiles CUDA kernels and is
#      binary-incompatible across CUDA/torch builds; the recipe + install doc both
#      use this flag).
#   2. vllm-omni from the recipe-pinned git commit (provides the --omni TTS stack
#      and pulls its own correct transformers; do NOT hand-pin transformers here --
#      vllm-omni excludes the 5.3.* line, which an earlier draft floor required).
#   3. fish-speech for the DAC codec the model loads at startup.
# NOTE: `vllm-omni` IS a real PyPI package, but the PyPI release version-matches a
# newer vllm (latest -> vLLM 0.23.0). The risk is version pairing, not the package
# name. If you want PyPI latest instead, install vllm==0.23.0 (with
# --torch-backend=auto) and then plain `vllm-omni` -- keep the pair consistent.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("libportaudio2", "portaudio19-dev", "git", "libsndfile1")
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",
        extra_options="--torch-backend=auto",
    )
    .uv_pip_install(
        f"vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@{VLLM_OMNI_COMMIT}",
        "fish-speech",
        "huggingface_hub[hf_transfer]",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Escape hatch: if startup fails with Triton/attention errors, set the
            # Fish KV-cache attention fast path off by uncommenting the next line
            # (documented in the s2-pro recipe). "=required" hard-fails if the path
            # is unavailable; "=0" disables it.
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
    scaledown_window=15 * MINUTES,  # stay warm 15 min after last request
    timeout=20 * MINUTES,
    # min_containers=1,  # uncomment for always-warm (no cold-start latency, costs $)
)
@modal.concurrent(max_inputs=8)  # vLLM batches; s2-pro is heavy so keep modest
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve():
    # Non-blocking launch: Modal's web_server waits for the port, the function returns.
    # Recipe-exact single-GPU invocation: `vllm serve fishaudio/s2-pro --omni
    # --host 0.0.0.0 --port 8091`. No --served-model-name (vLLM serves under the
    # model id by default) and no --tensor-parallel-size (no-op for 1 GPU); add the
    # latter only when N_GPU > 1.
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--omni",  # MANDATORY: enables the vLLM-Omni TTS stack
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
