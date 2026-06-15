Dialogue-LLM candidate serving apps go here — one Modal app per model, mirroring `tts/`.

- `qwen3_5_4b.py` — **Qwen3.5-4B** (Alibaba Qwen, Apache-2.0) served on Modal via vLLM as the
  OpenAI-compatible `POST /v1/chat/completions` dialogue baseline. Text-only
  (`--language-model-only`), thinking disabled per request, context capped for a voice loop,
  scale-to-zero on a 24GB L4. **Verified end-to-end on Modal 2026-06-15.** This is Stage 1 of
  [`docs/llm-finetune-tinker.md`](../docs/llm-finetune-tinker.md): a strong base + system
  prompt is the day-one architecture; a Tinker LoRA is a later, eval-gated upgrade for
  persona/AU-English consistency. Safety is **not** in this model — it belongs in a separate
  guardrail/moderation layer.
