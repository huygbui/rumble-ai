# Modal vLLM-Omni endpoint for OmniVoice TTS.
#   modal deploy tts/omnivoice.py
# Deployment notes: docs/modal-serving-notes.md.

import subprocess
import time
import urllib.error
import urllib.request

import modal
import modal.experimental  # http_server lives here; `import modal` alone does NOT pull it in

MODEL_NAME = "k2-fsa/OmniVoice"
VLLM_PORT = 8091
N_GPU = 1
GPU = "A10G"
MINUTES = 60

REGION = "us-east"
MIN_CONTAINERS = 0
TARGET_INPUTS = 8

VLLM_VERSION = "0.23.0"  # on PyPI; pairs with the vllm-omni 0.23 line
VLLM_OMNI_REF = "v0.23.0rc1"  # git tag carrying PR #4330 (online voice-design adapter)

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
        f"vllm-omni @ git+https://github.com/vllm-project/vllm-omni@{VLLM_OMNI_REF}",
        f"vllm=={VLLM_VERSION}",  # re-pin so vllm-omni's resolve can't drift vllm
        "fastapi<0.137",  # FastAPI 0.137 breaks vLLM's Prometheus middleware -> every request 500s; see llm/qwen3_5_4b.py
        "huggingface_hub[hf_transfer]",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "OMNIVOICE_CUDA_GRAPH": "0",
        }
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App("omnivoice-tts")


def _vllm_url(path: str) -> str:
    return f"http://127.0.0.1:{VLLM_PORT}{path}"


def _check_running(p: subprocess.Popen) -> None:
    rc = p.poll()
    if rc is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def _wait_ready(p: subprocess.Popen, timeout: int = 15 * MINUTES) -> None:
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
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
    scaledown_window=2 * MINUTES,  # stay warm 2 min after last request (cost-saving; chat keepalive holds it open)
    timeout=20 * MINUTES,
    min_containers=MIN_CONTAINERS,
    region=REGION,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.experimental.http_server(
    port=VLLM_PORT,
    proxy_regions=[REGION],
    startup_timeout=20 * MINUTES,
    exit_grace_period=5,
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class OmniVoiceVllm:
    @modal.enter(snap=True)
    def startup(self):
        # OmniVoice snapshots the warm model; CUDA graphs stay disabled by image env.
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
        pass

    @modal.exit()
    def stop(self):
        self.process.terminate()
