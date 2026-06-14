# omnivoice_modal.py
# Serve OmniVoice (k2-fsa/OmniVoice) TTS on Modal via vLLM-Omni.
# Exposes the OpenAI-compatible POST /v1/audio/speech endpoint on a single small GPU.
#
#   modal deploy omnivoice_modal.py   # persistent endpoint
#   modal serve  omnivoice_modal.py   # ephemeral hot-reload dev server
#   modal run    omnivoice_modal.py   # health-check via local_entrypoint
#
# Why OmniVoice (vs the fish_s2_pro_modal.py baseline):
#   - Apache-2.0 (open commercial use, no contact) vs Fish's research-only license.
#   - ~613M params / 2.45GB weights + ~806MB 24kHz codec -> runs on a 24GB card, not 80GB.
#   - The ONLY candidate with a NATIVE "australian accent" voice-design attribute, plus
#     "child"/"teenager" age attributes -- directly serves the kid + AU-English goal.
# It is pure TTS (no S2S, no built-in safety): all kid-safety/boundary/OOD handling must
# live in your upstream dialogue + moderation layer. See docs/tts-options-research.md.
#
# STATUS: VERIFIED end-to-end on Modal (2026-06-14), on the 0.23 stack (vLLM 0.23.0 +
# vllm-omni 0.23.0rc1) + A10G. On the older 0.22.0 stack the voice-design `instructions`
# path produced broadband ARTIFACTS (clipping noise, not voice); moving to the 0.23 line
# (PR #4330 adapter) restored a clean voiced-speech output profile. See the version block.

import subprocess

import modal

# --- Model + serving constants -------------------------------------------------
MODEL_NAME = "k2-fsa/OmniVoice"  # PUBLIC, ungated HF repo (Apache-2.0); the 24kHz codec
#                                  in audio_tokenizer/ downloads automatically with it.
VLLM_PORT = 8091  # vllm-omni's OmniVoice example (run_server.sh) defaults to 8091
N_GPU = 1  # ~613M params fp32 (~2.45GB) + ~806MB codec -> one small card is plenty
MINUTES = 60  # seconds

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

app = modal.App("omnivoice-tts")


@app.function(
    image=image,
    gpu=f"A10G:{N_GPU}",  # 24GB is ample for ~613M fp32 + codec; L4 works too. (Fish
    #                       needed an 80GB card; OmniVoice does not -- big cost win.)
    # No secret needed: k2-fsa/OmniVoice is a PUBLIC HF repo, downloads without a token.
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    scaledown_window=5 * MINUTES,  # stay warm 5 min after last request (cost-saving)
    timeout=20 * MINUTES,
    # min_containers=1,  # uncomment for always-warm (no cold-start latency, costs $)
)
@modal.concurrent(max_inputs=8)  # small/fast model; can likely raise this -- tune it
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve():
    # Mirrors the Fish recipe's non-blocking launch, with two OmniVoice-specific deltas:
    #   --omni              : MANDATORY (enables the TTS stack), same as Fish.
    #   --trust-remote-code : REQUIRED here -- OmniVoice ships custom modeling code
    #                         (model_type "omnivoice"); Fish s2-pro did not need it.
    # OmniVoice's deploy config (dtype float32, enforce_eager) is auto-resolved from the
    # model type, so -- unlike qwen3_tts -- no explicit --deploy-config is passed. Do NOT
    # override dtype. Add --gpu-memory-utilization here only if you hit OOM/over-reserve.
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
    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
def main():
    # Smoke test: confirm the server is up, then print example request payloads.
    # (Audio synthesis itself is best driven from a small client like client.py.)
    import urllib.request

    url = serve.get_web_url()
    print(f"Server URL: {url}")
    print(f"Speech endpoint: {url}/v1/audio/speech")
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
