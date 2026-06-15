# llm/qwen3_5_4b.py
# Serve Qwen3.5-4B (Alibaba Qwen) as the dialogue LLM on Modal via vLLM.
# Exposes the OpenAI-compatible POST /v1/chat/completions endpoint on a single small GPU.
#
#   modal deploy llm/qwen3_5_4b.py   # persistent endpoint
#   modal serve  llm/qwen3_5_4b.py   # ephemeral hot-reload dev server
#   modal run    llm/qwen3_5_4b.py   # health-check via local_entrypoint
#
# Why Qwen3.5-4B (the Stage-1 dialogue baseline -- see docs/llm-finetune-tinker.md):
#   - Apache-2.0 (open commercial use, no contact), top-tier provider (Alibaba Qwen).
#   - 4B beats 9B for THIS use case: a turn-by-turn voice loop wants low TTFT + fast
#     cold-start, and the 4B/9B gap on instruction-following (what persona adherence
#     rides on) is only ~1.7 IFEval points; 9B's lead is in hard reasoning a kids' chat
#     barely exercises. 4B fits a cheap 24GB card; 9B needs ~2x the GPU and ~2x latency.
#   - Tinker-supported for LoRA ($0.67/M train) when you later graduate to fine-tuning a
#     persona/AU-English-register adapter (Stage 3). The adapter merges onto THIS base.
# This model has NO built-in safety. Per docs/llm-finetune-tinker.md the kid-safety
# boundary lives in a SEPARATE guardrail classifier (input + output), NOT in this model
# or any future fine-tune. This app is the persona/dialogue engine only.
#
# ---------------------------------------------------------------------------------------
# COLD START: GPU MEMORY SNAPSHOTS (the big lever).
# The disk caches below (HF weights + vLLM compiled graphs) make DOWNLOAD and COMPILE
# cheap, but they do nothing for the phases that dominate a scale-to-zero cold start:
# loading weights into VRAM, memory profiling, CUDA-graph capture, and first-request
# Triton-JIT (the Gated-DeltaNet conv/KV kernels). GPU memory snapshots capture the
# POST-INITIALIZATION GPU+CPU state and restore it directly, so a warm-from-snapshot cold
# start skips all of that (Modal reports vLLM cold starts dropping from ~45s to ~5s).
#
# To get GPU snapshots we need the documented vLLM pattern, which is why this file looks
# different from the tts/ apps (still on the simpler @modal.web_server + Popen shape):
#   - A CLASS (@app.cls) with enable_memory_snapshot=True and
#     experimental_options={"enable_gpu_snapshot": True}  (the latter is ALPHA).
#   - @modal.experimental.http_server (Flash) instead of @modal.web_server -- this is the
#     server shape the GPU-snapshot examples use; it is region-pinned (see REGION).
#   - @modal.enter(snap=True): start vLLM (--enable-sleep-mode), wait ready, WARM IT UP
#     (bakes the Triton-JIT kernels into the snapshot), then PUT IT TO SLEEP (/sleep) so
#     the snapshot is taken with weights offloaded GPU->CPU (smaller/faster to capture).
#   - @modal.enter(snap=False): WAKE IT UP (/wake_up) after the snapshot is restored.
# vLLM's /sleep + /wake_up are dev endpoints -- they exist only with VLLM_SERVER_DEV_MODE=1
# (set in the image env below). The snapshot becomes available after a few cold starts.
# Still scale-to-zero (MIN_CONTAINERS=0) -- snapshots give a fast cold start WHILE idle is
# free, which is exactly the cost posture we want (no always-warm GPU bill).
#
# STATUS: dialogue path VERIFIED end-to-end on Modal (2026-06-15) on an L4 with the plain
# web_server shape (vLLM 0.23.0 resolves Qwen3_5ForConditionalGeneration, loads text-only
# via --language-model-only, returns a clean kid-appropriate reply with thinking disabled).
# Serve flags follow the official Qwen3.5-4B HF model-card command
# (huggingface.co/Qwen/Qwen3.5-4B). The fastapi<0.137 pin is REQUIRED (see that comment).
# The GPU-snapshot rewrite here is VERIFIED on Modal (2026-06-15): scale-from-zero cold start
# RESTORES the snapshot in ~25s (warm chat TTFT ~1.3s), vs ~330s for a full build -- a ~13x
# cold-start win while STILL scaling to zero. TWO gotchas were REQUIRED to make the snapshot
# actually engage (both encoded below); without either, every cold start silently full-
# rebuilds with NO snapshot benefit:
#   (1) Do NOT mount the vLLM compile-cache Volume -- the snapshot can't reconcile the 9p
#       mount on restore ("CompleteRestore ... failed to walk torch_compile_cache").
#   (2) --enforce-eager -- torch.compile breaks GPU memory snapshot CREATION, so the snapshot
#       never becomes usable. (See the Caches note and the serve-flags comment.)

import subprocess
import time
import urllib.error
import urllib.request

import modal
import modal.experimental  # http_server lives here; `import modal` alone does NOT pull it in

# --- Model + serving constants -------------------------------------------------
MODEL_NAME = "Qwen/Qwen3.5-4B"  # PUBLIC, ungated HF repo (Apache-2.0); downloads w/o token
VLLM_PORT = 8000  # vLLM's own default (the HF card example uses 8000); the tts/ apps use
#                   8091 because that's vllm-omni's default -- this is a plain vLLM server.
N_GPU = 1  # 4B text-only ~14GB at bf16 -> one 24GB card is plenty
GPU = "L4"  # 24GB, ~$0.80/hr -- cheapest fit for 4B text-only at bf16 (~14GB weights +
#             small KV at the capped context). A10G (24GB, what tts/omnivoice.py uses) is a
#             drop-in alternative; L40S/A100 give headroom if you raise MAX_MODEL_LEN.
MINUTES = 60  # seconds

# Cap the context LOW for a voice loop: short turns + a little history. 262K native would
# reserve a huge KV cache and waste VRAM/latency headroom. 16K is comfortable; drop to
# 8192 to raise concurrency / save VRAM (and shrink the snapshot), or raise it if you carry
# long dialogue histories.
MAX_MODEL_LEN = 16384

# --- Snapshot / Flash knobs ----------------------------------------------------
# GPU snapshots ride on Modal's Flash http_server, which is REGION-PINNED: the GPU
# container and its proxy live in one region. Pick the region nearest your users; the named
# Volumes below are global so they work from any region.
REGION = "us-east"
MIN_CONTAINERS = 0  # scale to ZERO when idle (no GPU bill); snapshots make the cold start fast
TARGET_INPUTS = 16  # Flash autoscale target: scale out when a container exceeds ~16 in-flight
#                     requests. 4B is light + vLLM continuous-batches well; tune up/down.

# --- vLLM version --------------------------------------------------------------
# Plain vLLM (NO vllm-omni / --omni -- that stack is only for the TTS audio models).
# Qwen3.5's hybrid Gated-DeltaNet + full-attention architecture needs a recent vLLM; the
# 0.23 line (same vLLM the tts/omnivoice.py app runs) postdates Qwen3.5's ~Mar-2026
# release and should recognize it. Bump if the architecture isn't recognized at startup.
VLLM_VERSION = "0.23.0"

# --- Container image -----------------------------------------------------------
# Simpler than the tts/ images: a text LLM needs only vLLM itself (no vllm-omni, no codec,
# no audio libs). Same CUDA 12.8 base + hf_transfer discipline as the rest of the repo.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git")
    # No `--torch-backend=auto`: Modal's image builder has no GPU, so `auto` would install
    # a CPU-only torch that fails on the GPU at runtime (same note as the tts/ apps).
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",
        # Pin FastAPI < 0.137. vLLM 0.23.0 mounts a Prometheus metrics middleware
        # (prometheus-fastapi-instrumentator) that runs on EVERY request; FastAPI
        # 0.137.0's new lazy router inclusion (`_IncludedRouter`) breaks its route-name
        # resolution, so every request -- including /health and /sleep -- 500s with
        # "'_IncludedRouter' object has no attribute 'path'". 0.137.0 landed 2026-06-15;
        # builds a day earlier (e.g. tts/omnivoice.py, 2026-06-14) resolved 0.136.x and
        # work fine on the same vLLM. Drop this pin once the instrumentator ships a
        # 0.137-compatible release.
        "fastapi<0.137",
        "huggingface_hub[hf_transfer]",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # REQUIRED for the snapshot dance: /sleep + /wake_up are vLLM "dev" endpoints
            # that are only mounted when this is set. Without it the @modal.enter sleep/wake
            # calls below 404 and no GPU snapshot is taken.
            "VLLM_SERVER_DEV_MODE": "1",
        }
    )
)

# --- Caches -------------------------------------------------------------------
# ONLY the HF weights volume is mounted. We deliberately do NOT mount a Volume for vLLM's
# torch.compile cache when snapshotting -- this was the bug that broke the first snapshot:
#   --enable-sleep-mode forces vLLM's cumem allocator, which CHANGES the torch.compile cache
#   key. So the snapshot-build boot gets a cache MISS, RECOMPILES (~90s), and writes fresh
#   files into the compile cache. If that cache is a Modal Volume (9p), the snapshot freezes
#   a process referencing files that are NOT in the Volume's committed state, and RESTORE
#   aborts with:
#     vfs.CompleteRestore() ... 9p: failed to walk ".../torch_compile_cache/...": no such file
#   -> it then falls back to a full cold boot, i.e. no snapshot win at all.
# Keeping the compile cache on the container's OWN filesystem lets the snapshot capture the
# compiled graphs in-process (no 9p mount to reconcile), so restores succeed. The HF weights
# are downloaded/committed once and are stable, so that Volume restores fine and stays.
# Cost: each fresh snapshot BUILD recompiles (~90s), folded into snapshot creation; every
# RESTORE skips it. (To re-add a persistent compile-cache Volume later you'd have to commit
# it after warmup AND ensure the snap boot hits the cache -- not worth the fragility here.)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App("qwen3-5-4b-llm")


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


def _warmup() -> None:
    # Send a few REAL non-thinking completions so the one-time Triton-JIT kernel compiles
    # (Gated-DeltaNet conv/KV) and CUDA paths are exercised BEFORE the snapshot is taken --
    # this is what bakes the "first request eats JIT spikes" cost into the snapshot.
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
    # level=1: offload weights GPU->CPU (kept in RAM), drop KV cache. The CPU memory
    # snapshot captures the RAM; the GPU snapshot then has far less to checkpoint.
    _post(f"/sleep?level={level}")


def _wake_up() -> None:
    _post("/wake_up")


@app.cls(
    image=image,
    gpu=f"{GPU}:{N_GPU}",
    # No secret needed: Qwen/Qwen3.5-4B is a PUBLIC HF repo, downloads without a token.
    # If you later serve a gated/private fine-tune, add:
    #   secrets=[modal.Secret.from_name("huggingface-secret")],  # injects HF_TOKEN
    volumes={
        # HF weights only -- the vLLM compile cache stays container-local (see Caches note).
        "/root/.cache/huggingface": hf_cache_vol,
    },
    # Idle-cost control: stay warm 5 min after the last request, then scale to ZERO.
    scaledown_window=5 * MINUTES,
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
class QwenVllm:
    @modal.enter(snap=True)
    def startup(self):
        # Runs BEFORE the snapshot is captured. Launch vLLM, warm it, then sleep so the
        # snapshot is taken in the offloaded state. Flags mirror the official Qwen3.5-4B HF
        # card command, tuned for a text voice loop:
        #   --language-model-only : drop the vision tower (this is a multimodal checkpoint;
        #                           we only do text -> saves VRAM + load time).
        #   --reasoning-parser qwen3 : if a request leaves thinking ON, split the chain-of-
        #                           thought into `reasoning_content` instead of polluting the
        #                           spoken reply. Harmless when thinking is disabled.
        #   --max-model-len       : capped low (see MAX_MODEL_LEN) for the voice loop.
        #   --enable-sleep-mode   : REQUIRED for the /sleep + /wake_up snapshot dance.
        #   --enforce-eager       : REQUIRED for GPU snapshots to actually engage. Modal's
        #     docs warn "torch.compile can cause Memory Snapshot creation to fail" -- vLLM's
        #     inductor compile spawns multi-process workers that CRIU can't checkpoint, so the
        #     snapshot silently fails to become usable and EVERY cold start full-rebuilds
        #     (measured: 333s/140s/422s, never converging). --enforce-eager skips torch.compile
        #     AND CUDA-graph capture, so snapshot creation succeeds and restores are fast. Cost
        #     is modestly lower throughput / slightly higher inter-token latency -- negligible
        #     for a single-user voice loop (warm TTFT stays ~1-2s). It also drops the ~80-90s
        #     compile from every build. (To get CUDA graphs back, drop this and instead set
        #     TORCHINDUCTOR_COMPILE_THREADS=1 in the image env -- single-threaded compile is
        #     CRIU-checkpointable -- but that path is slower to build and less certain.)
        # NOTE on THINKING: Qwen3.5 has thinking ON by default and NO `/think` soft switch.
        # Disable it PER REQUEST with `chat_template_kwargs: {"enable_thinking": false}`
        # (see the local_entrypoint + app.core.dialogue). There is no serve-time flag to force it off.
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
        # Escape hatch: prefix caching is flagged experimental on the Gated-DeltaNet/hybrid
        # KV path -- if you hit correctness/stability issues, add "--no-enable-prefix-caching".
        self.process = subprocess.Popen(cmd)
        _wait_ready(self.process)
        _warmup()
        _sleep(level=1)

    @modal.enter(snap=False)
    def restore(self):
        # Runs AFTER the snapshot is restored (GPU state already back). Move weights
        # CPU->GPU so the server is immediately ready to generate.
        _wake_up()

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.local_entrypoint()
def main():
    # End-to-end smoke test: resolve the Flash URL, wait for the server, then make a REAL
    # chat completion and assert the model generates a non-empty reply with thinking off.
    import json

    # http_server (Flash) URLs are NOT the classic `<workspace>--app-fn.modal.run` form;
    # fetch them from the class. (Also printed in the `modal deploy` output.)
    url = QwenVllm._experimental_get_flash_urls()[0].rstrip("/")
    print(f"Server URL: {url}")
    print(f"OpenAI base_url: {url}/v1")
    print(f"Chat endpoint: {url}/v1/chat/completions\n")

    # /health blocks (up to startup_timeout) until vLLM has loaded the weights and is ready.
    with urllib.request.urlopen(f"{url}/health", timeout=20 * MINUTES) as r:
        assert r.status == 200, f"health check failed: {r.status}"
    print("Health check OK -- model loaded.\n")

    # The two voice-loop essentials: enable_thinking=false (no slow chain-of-thought) and
    # the Qwen3.5 recommended non-thinking sampling params. Persona/AU register go in
    # `system`; kid-safety stays in your SEPARATE guardrail layer, not here.
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a friendly Australian helper for kids. "
                "Speak simply, warmly, in one or two short sentences.",
            },
            {"role": "user", "content": "Can you tell me a tiny story about a wombat?"},
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5 * MINUTES) as r:
        assert r.status == 200, f"chat completion failed: {r.status}"
        body = json.loads(r.read())

    msg = body["choices"][0]["message"]
    reply = (msg.get("content") or "").strip()
    print("Prompt:  Can you tell me a tiny story about a wombat?")
    print(f"Reply:   {reply}\n")

    # Verify generation actually happened and thinking was OFF (no leaked reasoning).
    assert reply, "empty completion -- model returned no content"
    assert "<think>" not in reply, "thinking leaked into content (enable_thinking not honored)"
    assert not msg.get("reasoning_content"), (
        "reasoning_content present -- thinking was NOT disabled; check chat_template_kwargs"
    )
    usage = body.get("usage", {})
    print(f"Tokens:  {usage.get('completion_tokens', '?')} completion / "
          f"{usage.get('prompt_tokens', '?')} prompt")
    print("END-TO-END OK: server up, model generated a reply, thinking disabled.")
