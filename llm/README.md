Dialogue-LLM candidate serving apps go here — one Modal app per model, mirroring `tts/`.

- `qwen3_5_4b.py` — **Qwen3.5-4B** (Alibaba Qwen, Apache-2.0) served on Modal via vLLM as the
  OpenAI-compatible `POST /v1/chat/completions` dialogue baseline. Text-only
  (`--language-model-only`), thinking disabled per request, context capped for a voice loop,
  scale-to-zero on a 24GB L4. **Dialogue path verified end-to-end on Modal 2026-06-15.**
  Cold start is attacked with **GPU memory snapshots** (`enable_memory_snapshot` +
  `experimental_options={"enable_gpu_snapshot": True}`): served as an `@app.cls` behind
  `@modal.experimental.http_server` (Flash, region-pinned) with a `@modal.enter(snap=True)`
  start→warmup→`/sleep` and a `@modal.enter(snap=False)` `/wake_up`, so a scale-from-zero
  cold start restores the warmed GPU state instead of reloading/recompiling (Modal reports
  ~45s→~5s for vLLM). Get the endpoint URL from the `modal deploy` output. **Verified
  2026-06-15: scale-from-zero cold start restores in ~25s** (warm chat TTFT ~1.3s) vs ~330s
  for a full build — ~13× faster, still scaling to zero. Two non-obvious requirements made
  the snapshot actually engage: (1) **do not** mount the vLLM compile-cache Volume (9p
  restore-reconcile failure) and (2) **`--enforce-eager`** (torch.compile breaks GPU-snapshot
  creation); without either, every cold start silently full-rebuilds. This is Stage 1 of
  [`docs/llm-finetune-tinker.md`](../docs/llm-finetune-tinker.md): a strong base + system
  prompt is the day-one architecture; a Tinker LoRA is a later, eval-gated upgrade for
  persona/AU-English consistency. Safety is **not** in this model — it belongs in a separate
  guardrail/moderation layer.
