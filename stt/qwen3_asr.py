# Modal vLLM endpoint for Qwen3-ASR.
#   modal deploy stt/qwen3_asr.py
# Set QWEN_ASR_MODEL to deploy the 1.7B variant separately.
# Deployment notes: docs/modal-serving-notes.md.

import os
import re
import subprocess
import time
import urllib.error
import urllib.request

import modal
import modal.experimental  # http_server lives here; `import modal` alone does NOT pull it in

MODEL_NAME = os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
VLLM_PORT = 8000
N_GPU = 1
GPU = "L4"
MINUTES = 60
_SLUG = re.sub(r"[^a-z0-9]+", "-", MODEL_NAME.split("/")[-1].lower()).strip("-")
MAX_MODEL_LEN = None

REGION = "us-east"
MIN_CONTAINERS = 0
TARGET_INPUTS = 8

VLLM_VERSION = "0.23.0"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git", "ffmpeg", "libsndfile1")
    .uv_pip_install(
        f"vllm[audio]=={VLLM_VERSION}",
        "fastapi<0.137",
        "huggingface_hub[hf_transfer]",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "QWEN_ASR_MODEL": MODEL_NAME,
            "VLLM_SERVER_DEV_MODE": "1",
        }
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App(f"{_SLUG}-stt")


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


def _tiny_wav_bytes(seconds: float = 0.3, sample_rate: int = 16000) -> bytes:
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


def _post_multipart(
    path: str,
    fields: dict,
    file_field: str,
    file_name: str,
    file_bytes: bytes,
    content_type: str = "audio/wav",
    timeout: int = 120,
) -> None:
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
    _post(f"/sleep?level={level}")


def _wake_up() -> None:
    _post("/wake_up")


@app.cls(
    image=image,
    gpu=f"{GPU}:{N_GPU}",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
    scaledown_window=2 * MINUTES,
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
class QwenAsrVllm:
    @modal.enter(snap=True)
    def startup(self):
        # Sleep mode + eager mode are required for reliable GPU snapshots.
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
        if MAX_MODEL_LEN is not None:
            cmd += ["--max-model-len", str(MAX_MODEL_LEN)]
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
