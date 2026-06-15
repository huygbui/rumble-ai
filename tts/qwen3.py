# Modal vLLM-Omni endpoint for Qwen3-TTS.
#   modal deploy tts/qwen3.py

import subprocess

import modal

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
VLLM_PORT = 8091
N_GPU = 1
MINUTES = 60

VLLM_VERSION = "0.22.1"  # latest 0.22.x stable
VLLM_OMNI_VERSION = "0.22.0"  # latest stable PyPI release; pairs with the vLLM 0.22 line

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git", "ffmpeg", "libsndfile1")
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

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("qwen3-tts")


@app.function(
    image=image,
    gpu=f"H100:{N_GPU}",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    scaledown_window=5 * MINUTES,  # stay warm 5 min after last request (cost-saving)
    timeout=20 * MINUTES,
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve():
    import os

    import vllm_omni

    deploy_cfg = os.path.join(
        os.path.dirname(vllm_omni.__file__), "deploy", "qwen3_tts.yaml"
    )

    cmd = ["vllm-omni", "serve", MODEL_NAME]
    if os.path.isfile(deploy_cfg):
        cmd += ["--deploy-config", deploy_cfg]
    else:
        print(f"WARNING: deploy-config not found at {deploy_cfg} -- serving without it; "
              "vendor vllm_omni/deploy/qwen3_tts.yaml if audio is wrong.")
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
    subprocess.Popen(cmd)
