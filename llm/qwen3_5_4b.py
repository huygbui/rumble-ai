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
# STATUS: VERIFIED end-to-end on Modal (2026-06-15) on an L4. vLLM 0.23.0 resolves the
# architecture (Qwen3_5ForConditionalGeneration), loads in text-only mode
# (--language-model-only worked), and a live POST /v1/chat/completions returned a clean,
# kid-appropriate reply with thinking disabled (no <think>, no reasoning_content). Serve
# flags follow the official Qwen3.5-4B HF model-card command
# (huggingface.co/Qwen/Qwen3.5-4B). Two notes from the bring-up:
#   - The fastapi<0.137 pin in the image is REQUIRED (see that comment) -- without it every
#     request 500s on a Prometheus-middleware bug.
#   - The first request after a cold start eats a few Triton-JIT latency spikes (the
#     hybrid Gated-DeltaNet conv/KV kernels compile on first use); measure STEADY-STATE
#     tok/s + TTFT separately, and consider a warmup request after cold start.

import subprocess

import modal

# --- Model + serving constants -------------------------------------------------
MODEL_NAME = "Qwen/Qwen3.5-4B"  # PUBLIC, ungated HF repo (Apache-2.0); downloads w/o token
VLLM_PORT = 8000  # vLLM's own default (the HF card example uses 8000); the tts/ apps use
#                   8091 because that's vllm-omni's default -- this is a plain vLLM server.
N_GPU = 1  # 4B text-only ~14GB at bf16 -> one 24GB card is plenty
MINUTES = 60  # seconds

# Cap the context LOW for a voice loop: short turns + a little history. 262K native would
# reserve a huge KV cache and waste VRAM/latency headroom. 16K is comfortable; drop to
# 8192 to raise concurrency / save VRAM, or raise it if you carry long dialogue histories.
MAX_MODEL_LEN = 16384

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
        # resolution, so every request -- including /health -- 500s with
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
        }
    )
)

# --- Caches (persisted across cold starts) ------------------------------------
# Reuse the repo's shared named volumes so weights/compiled graphs survive scale-to-zero.
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("qwen3-5-4b-llm")


@app.function(
    image=image,
    gpu=f"L4:{N_GPU}",  # 24GB, ~$0.80/hr -- cheapest fit for 4B text-only at bf16 (~14GB
    #                     weights + small KV at the capped context). A10G (24GB, what
    #                     tts/omnivoice.py uses) is a drop-in alternative; L40S/A100 give
    #                     headroom if you raise MAX_MODEL_LEN or concurrency.
    # No secret needed: Qwen/Qwen3.5-4B is a PUBLIC HF repo, downloads without a token.
    # If you later serve a gated/private fine-tune, add:
    #   secrets=[modal.Secret.from_name("huggingface-secret")],  # injects HF_TOKEN
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    # Idle-cost control: stay warm 5 min after the last request, then scale to ZERO (no GPU
    # cost while idle). A 4B is small so cold start is modest, but restoring weights into
    # VRAM is still bandwidth-bound -- measure it. For a latency-sensitive voice loop you
    # may want a small warm floor (min_containers=1) instead of pure scale-to-zero.
    scaledown_window=5 * MINUTES,
    timeout=20 * MINUTES,
    # min_containers=1,  # uncomment for always-warm (no cold-start latency, costs $)
)
@modal.concurrent(max_inputs=16)  # 4B is light + vLLM continuous-batches well; tune up/down
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve():
    # Non-blocking launch: Modal's web_server waits for the port, the function returns.
    # Flags mirror the official Qwen3.5-4B HF card command, tuned for a text voice loop:
    #   --language-model-only : drop the vision tower (this is a multimodal checkpoint;
    #                           we only do text dialogue -> saves VRAM + load time).
    #   --reasoning-parser qwen3 : if a request leaves thinking ON, split the chain-of-
    #                           thought into `reasoning_content` instead of polluting the
    #                           spoken reply. Harmless when thinking is disabled.
    #   --max-model-len       : capped low (see MAX_MODEL_LEN) for the voice loop.
    # NOTE on THINKING: Qwen3.5 has thinking ON by default and NO `/think` soft switch.
    # Disable it PER REQUEST with `chat_template_kwargs: {"enable_thinking": false}` (see
    # the local_entrypoint example) -- essential for low-latency speech. There is no
    # serve-time flag to force it off; the client must send it.
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--language-model-only",
        "--reasoning-parser",
        "qwen3",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]
    if N_GPU > 1:
        cmd += ["--tensor-parallel-size", str(N_GPU)]
    # Escape hatch: prefix caching is flagged experimental on the Gated-DeltaNet/hybrid KV
    # path -- if you hit correctness/stability issues, add "--no-enable-prefix-caching".
    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
def main():
    # End-to-end smoke test: wait for the server, then make a REAL chat completion and
    # assert the model actually generates a non-empty reply with thinking disabled.
    import json
    import urllib.request

    url = serve.get_web_url()
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
