# tts/qwen3.py
# Serve Qwen3-TTS-12Hz-1.7B (Alibaba Qwen) TTS on Modal via vLLM-Omni.
# Exposes the OpenAI-compatible POST /v1/audio/speech endpoint on a single GPU.
#
#   modal deploy tts/qwen3.py   # persistent endpoint
#   modal serve  tts/qwen3.py   # ephemeral hot-reload dev server
#   modal run    tts/qwen3.py   # health-check via local_entrypoint
#
# Why Qwen3-TTS (the de-risked hedge alongside tts/omnivoice.py):
#   - Apache-2.0 (open commercial use, no contact) vs Fish's research-only license.
#   - Top-tier, well-maintained provider (Alibaba Qwen); natural, warm streaming English.
#   - ~1.7B / ~3.8GB weights + ~0.68GB codec -> a 48GB card, not the 80GB Fish needs.
# No NATIVE Australian English: AU is reached via zero-shot cloning of a consented AU
# clip (the -Base variant). Pure TTS, no built-in safety -- kid-safety/boundary/OOD must
# live upstream. See docs/tts-options.md.
#
# STATUS: VERIFIED end-to-end on Modal (2026-06-14). The online /v1/audio/speech path for
# Qwen3-TTS is shipped in vllm-omni 0.22.0 (the old "offline-only" note is stale). The
# deploy-config resolves from the installed package, and a CustomVoice cold-start request
# returned a valid 24kHz WAV. REQUIRES an 80GB H100: the shipped qwen3_tts.yaml is "Verified
# on 1x H100" (two stages co-located at 0.3 mem each); a 48GB L40S OOMs the code2wav stage
# during READY. Do not pass --gpu-memory-utilization on the CLI (it breaks the per-stage split).

import subprocess

import modal

# --- Model + serving constants -------------------------------------------------
# Three public, ungated, Apache-2.0 variants share one architecture; pick by TASK:
#   -CustomVoice : preset speakers + `language` + style `instructions`   (default below)
#   -Base        : zero-shot voice cloning (ref_audio + ref_text)  <- the AU-via-clone path
#   -VoiceDesign : natural-language voice design via `instructions`
# Each online server serves ONE variant's task -- deploy a second app for -Base if you
# want cloning at the same time. For the kid + AU goal, an AU clone via -Base is the
# recommended route (no native AU accent exists in this model).
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
VLLM_PORT = 8091  # vllm-omni's qwen3_tts example (run_server.sh) defaults to 8091
N_GPU = 1
MINUTES = 60  # seconds

# --- vLLM / vllm-omni version pair ---------------------------------------------
# Same proven pair as the Fish recipe -- vllm-omni 0.22.0 (stable) now ships the online
# Qwen3-TTS serving path. vllm-omni declares no vllm dep, so we re-pin vllm explicitly.
VLLM_VERSION = "0.22.1"  # latest 0.22.x stable
VLLM_OMNI_VERSION = "0.22.0"  # latest stable PyPI release; pairs with the vLLM 0.22 line

# --- Container image -----------------------------------------------------------
# Simpler than the Fish image: Qwen3-TTS has none of fish-speech's einx/pydantic/protobuf
# conflicts, and vLLM-Omni loads the model itself (no `qwen-tts` pip pkg needed to serve).
# Add ffmpeg (Qwen3-TTS audio I/O) alongside the libsndfile1 the Fish image already used.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git", "ffmpeg", "libsndfile1")
    # No `--torch-backend=auto` (Modal's builder has no GPU -> would install CPU-only torch).
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",
    )
    .uv_pip_install(
        f"vllm-omni=={VLLM_OMNI_VERSION}",
        f"vllm=={VLLM_VERSION}",  # re-pin so vllm-omni's resolve can't drift vllm
        "fastapi<0.137",  # FastAPI 0.137 breaks vLLM's Prometheus middleware -> every request 500s; see llm/qwen3_5_4b.py
        "huggingface_hub[hf_transfer]",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

# --- Caches (persisted across cold starts) ------------------------------------
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("qwen3-tts")


@app.function(
    image=image,
    gpu=f"H100:{N_GPU}",  # 80GB. The shipped qwen3_tts.yaml is "Verified on 1x H100" and
    #                       co-locates BOTH stages (talker + code2wav) on GPU 0 at
    #                       gpu_memory_utilization 0.3 each (~24GB/stage). A 48GB L40S gives
    #                       each stage only ~14.4GB, too tight for stage 1's cudagraph
    #                       capture + 65536 KV cache -> the child stage dies during READY.
    # No secret needed: the Qwen3-TTS repos are PUBLIC, download without a token.
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    scaledown_window=5 * MINUTES,  # stay warm 5 min after last request (cost-saving)
    timeout=20 * MINUTES,
    # min_containers=1,  # uncomment for always-warm (no cold-start latency, costs $)
)
@modal.concurrent(max_inputs=8)  # vLLM batches; tune against the dual-engine VRAM budget
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve():
    import os

    # Qwen3-TTS REQUIRES a two-stage deploy-config (Talker LM -> Code2Wav codec); without
    # it the model won't serve correctly. It ships INSIDE the installed vllm-omni package,
    # so resolve it dynamically rather than hard-coding a repo-relative path.
    import vllm_omni

    deploy_cfg = os.path.join(
        os.path.dirname(vllm_omni.__file__), "deploy", "qwen3_tts.yaml"
    )

    # Use the `vllm-omni serve` entrypoint (the tested run_server.sh path); `vllm serve
    # … --omni` is the docs alternative. --trust-remote-code is MANDATORY for Qwen3-TTS.
    cmd = ["vllm-omni", "serve", MODEL_NAME]
    if os.path.isfile(deploy_cfg):
        cmd += ["--deploy-config", deploy_cfg]
    else:
        # Wheel didn't bundle the yaml: vendor it from the vllm-omni repo
        # (vllm_omni/deploy/qwen3_tts.yaml) next to this file and point --deploy-config at
        # it, or try the docs short-form (no deploy-config) and validate output quality.
        print(f"WARNING: deploy-config not found at {deploy_cfg} -- serving without it; "
              "vendor vllm_omni/deploy/qwen3_tts.yaml if audio is wrong.")
    # NOTE: do NOT pass --gpu-memory-utilization here. The deploy-config sets a per-stage
    # 0.3 split (talker + code2wav co-located on GPU 0); a CLI override applies globally to
    # both stages and breaks that split (each tries to grab the whole card -> stage 1 OOMs
    # during READY). Let the yaml govern. Tune memory in the yaml, not on the CLI.
    cmd += [
        "--omni",
        "--trust-remote-code",
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
    # Smoke test: confirm the server is up, then print example request payloads.
    import urllib.request

    url = serve.get_web_url()
    print(f"Server URL: {url}")
    print(f"Speech endpoint: {url}/v1/audio/speech")
    print(f"Serving variant: {MODEL_NAME}")
    with urllib.request.urlopen(f"{url}/health", timeout=20 * MINUTES) as r:
        assert r.status == 200, f"health check failed: {r.status}"
    print("Health check OK.\n")

    print("Example requests (POST JSON to /v1/audio/speech, 24kHz WAV out).")
    print("The payload depends on which VARIANT (MODEL_NAME) this app is serving:\n")
    print("# -CustomVoice (this default): preset speaker + language + optional style.")
    print(
        '  {"input": "Hello! Want to hear a story?", "voice": "vivian",\n'
        '   "language": "English", "instructions": "speak in a warm, friendly tone",\n'
        '   "response_format": "wav"}\n'
    )
    print("# -Base (deploy with MODEL_NAME=…-Base): zero-shot cloning -> the AU path.")
    print("#   ref_audio = consented AU clip (path / URL / data: URI); ref_text = transcript.")
    print(
        '  {"task_type": "Base", "input": "This line is in the cloned AU voice.",\n'
        '   "ref_audio": "https://…/au_reference.wav", "ref_text": "transcript of the clip",\n'
        '   "language": "English", "response_format": "wav"}\n'
    )
    print("# -VoiceDesign (deploy with MODEL_NAME=…-VoiceDesign): design a voice from text.")
    print(
        '  {"task_type": "VoiceDesign",\n'
        '   "instructions": "A cheerful young Australian woman, gentle and clear",\n'
        '   "input": "Nice to meet you!", "language": "English", "response_format": "wav"}'
    )
