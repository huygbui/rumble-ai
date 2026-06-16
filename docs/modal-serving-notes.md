# Modal serving notes

Keep only deploy-critical constraints in the serving scripts. Historical findings live here.

## Shared vLLM constraints

- Do not use `--torch-backend=auto` during Modal image builds. The builder has no GPU and can resolve CPU-only torch.
- Pin `fastapi<0.137` with vLLM 0.23 until the Prometheus middleware route-name issue is fixed.
- For GPU snapshots, avoid mounting `/root/.cache/vllm`; the 9p volume can break restore. Keep only the Hugging Face weights volume mounted.
- `VLLM_SERVER_DEV_MODE=1` is required when a snapshot flow calls `/sleep` or `/wake_up`.

## LLM

`llm/qwen3_5_4b.py` was verified on Modal on 2026-06-15. The snapshot path uses `--enable-sleep-mode` and `--enforce-eager`; without eager mode, snapshot creation fell back to full cold boots. Qwen3.5 thinking is disabled per request with `chat_template_kwargs`.

## STT

`stt/qwen3_asr.py` was verified on Modal on 2026-06-15. Qwen3-ASR needs `vllm[audio]` so uploaded audio can be decoded by the transcription endpoint. The 0.6B model restored from snapshot much faster than a full cold boot.

## OmniVoice

`tts/omnivoice.py` was verified on Modal on 2026-06-15. It snapshots the warm resident model instead of using the vLLM sleep endpoints. `OMNIVOICE_CUDA_GRAPH=0` is kept because OmniVoice has a separate CUDA graph path not covered by vLLM `enforce_eager`.

GPU-tier / `num_step` were A/B'd on 2026-06-16 (see [`omnivoice-bench.md` §1b](./omnivoice-bench.md#1b-re-measured-2026-06-16--gpu--num_step-ab)): **stay on A10G, 32 steps.** `num_step` is deploy-time only — OmniVoice reads `config.num_step` (override at load via `--hf-overrides '{"num_step": N}'`); a request-body `num_step` is ignored. It showed no measurable latency win 32→24. L40S is ~2.4× faster on long synth but slightly slower on short clauses (TTFA), so it's a throughput/long-form lever, not a TTFA one.
