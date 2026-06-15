# Modal vLLM endpoint for Qwen3.5-4B.
#   modal deploy llm/qwen3_5_4b.py
# Deployment notes: docs/modal-serving-notes.md.

import subprocess
import time
import urllib.error
import urllib.request

import modal
import modal.experimental  # http_server lives here; `import modal` alone does NOT pull it in

MODEL_NAME = "Qwen/Qwen3.5-4B"
VLLM_PORT = 8000
N_GPU = 1
GPU = "L4"
MINUTES = 60
MAX_MODEL_LEN = 16384

REGION = "us-east"
MIN_CONTAINERS = 0
TARGET_INPUTS = 16

VLLM_VERSION = "0.23.0"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",
        "fastapi<0.137",
        "huggingface_hub[hf_transfer]",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_SERVER_DEV_MODE": "1",
        }
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App("qwen3-5-4b-llm")


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
    raise TimeoutError(f"vLLM not ready within {timeout}s")


def _warmup() -> None:
    import json

    payload = json.dumps(
        {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Say g'day in one short sentence."}
            ],
            "max_tokens": 16,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    for _ in range(3):
        _post("/v1/chat/completions", payload, timeout=120)


def _sleep(level: int = 1) -> None:
    _post(f"/sleep?level={level}")


def _wake_up() -> None:
    _post("/wake_up")


@app.cls(
    image=image,
    gpu=f"{GPU}:{N_GPU}",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
    scaledown_window=5 * MINUTES,
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
class QwenVllm:
    @modal.enter(snap=True)
    def startup(self):
        # Sleep mode + eager mode are required for reliable GPU snapshots.
        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--language-model-only",
            "--reasoning-parser",
            "qwen3",
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--enable-sleep-mode",
            "--enforce-eager",
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
        _sleep(level=1)

    @modal.enter(snap=False)
    def restore(self):
        _wake_up()

    @modal.exit()
    def stop(self):
        self.process.terminate()
