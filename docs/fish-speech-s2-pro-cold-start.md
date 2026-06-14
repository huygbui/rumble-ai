# Fish Speech S2 Pro — cold-start investigation & current-stack notes

_Investigated 2026-06-14, on the **upgraded** stack: vLLM **0.22.1** + vllm-omni
**0.22.0**, `H100:1`, Modal app `fish-s2-pro-tts`. This supersedes the 0.19.0-era
numbers in [`../PERF_BASELINE.md`](../PERF_BASELINE.md) (which predates the upgrade
and the `voice:"default"` removal). One of several models in this repo — see
`fish_s2_pro_modal.py`; siblings being tested: `qwen3_tts_modal.py`,
`omnivoice_modal.py`._

## TL;DR

- **Cold start (~150–220 s) is NOT a bug.** The torch.compile/inductor cache is
  healthy and reused every boot — it loads compiled graphs in **~1.8 s**, it does not
  recompile. The earlier worry about cache "churn" was disproven.
- The time is **inherent 2-stage (AR + DAC) model boot**: container spin-up + weight
  load + a ~66 s memory-profiling/warmup/stage-init phase + graph capture.
- **The one real lever** is `--max-model-len` (currently 16384) — lowering it should
  shrink the profiling phase and the KV-cache reservation. **Untested.**
- **A100 caveat:** the compile cache is keyed by GPU type, so the *first* A100 boot
  recompiles once (one slow boot), then every later A100 cold start is fast.

## Evidence: the compile cache is reused, not regenerated

From the boot log of a clean cold start (153 s total):

```
07:24:09 [backends.py:1089] Using cache directory: /root/.cache/vllm/torch_compile_cache/53ee2bbcce/...
07:24:11 [backends.py:292]  Directly load the compiled graph(s) for compile range (1, 16384) from the cache, took 1.768 s
07:24:11 [monitor.py:53]    torch.compile took 4.89 s in total
```

- A real recompile would add **60–120 s**; we see a **1.8 s cache load**.
- The `vllm-cache` volume holds a **stable set of ~10 compile-hash dirs** under
  `torch_compile_cache/` (the 2-stage omni model's legitimate compile units). A fresh
  cold start added **zero** new keys — no churn.
- Cache lives on the `vllm-cache` Modal Volume, mounted at `/root/.cache/vllm`.

## Where the ~150 s actually goes (from the 153 s boot)

| Phase | ~Time | Notes |
|---|---:|---|
| Modal container spin-up | ~25 s | scheduling + image/GPU acquisition (before vLLM logs) |
| vLLM start → weights loaded | ~33 s | engine init + ~8.5 GiB checkpoint from cache |
| **Compile-cache load (HIT)** | **~2 s** | `Directly load the compiled graph(s) … took 1.768 s` |
| Memory profiling + warmup + 2-stage (AR+DAC) init | **~66 s** | largest single chunk; tied to `--max-model-len` |
| CUDA graph capture | ~4 s | `Graph capturing finished in 4 secs` (sizes [1,2,4,8]) |
| KV-cache sizing + flashinfer autotune + finalize | ~15 s | KV cache 47.51 GiB / 345,936 tokens; 21.1× concurrency @ 16k |

Cold-start observations across boots: **265 s** (first-ever boot, populated the
compile cache), **218 s** (perf-test boot — inflated by first-inference warmup), and
**153 s** (clean boot). Treat steady-state as **~150–220 s**; the spread is Modal
scheduling/image-pull variance, not recompilation.

## Open lever: `--max-model-len`

vLLM profiles and sizes the KV cache for **16,384 tokens** (reserving 47.5 GiB /
345k tokens — far more than TTS needs: a long paragraph is ~1–2k tokens). Lowering
`--max-model-len` to e.g. **4096** in `fish_s2_pro_modal.py`'s `vllm serve` args
should trim the ~66 s profiling phase and free GPU memory. Cost is one cold start to
measure; helps on **either** GPU. **Not yet tried.**

## Current-stack H100 perf snapshot (2026-06-14)

Quick perf/quality run (registered-voice path; audio saved to `../samples/`,
gitignored, with `MANIFEST.txt`):

| Metric | Value |
|---|---|
| Cold start | ~150–220 s |
| Warm latency, short clip | **3.93 s** median (3 runs, stable) |
| RTF, sentence-length (32–71 words) | **0.40–0.55** (faster than real-time) |
| Streaming time-to-first-audio | **1.16 s** |
| Long-form (71 words) | 31.1 s audio — **not truncated** (vllm-omni #2248 fixed on this stack) |

Note: the first request after a cold start carries one-time inference warmup (showed
RTF ≈ 2.0); the **warm 3.9 s** is the true short-clip number.

## A100-80GB decision (in progress)

- Pricing: H100 $0.001097/s vs A100-80GB $0.000694/s (~37% cheaper/s). A100-40GB
  won't fit (~49 GiB peak).
- Break-even: A100 is cheaper per request while it's **< 1.58×** slower than H100;
  expected slowdown is ~1.4–1.7× (no FA3/FP8) → active compute ≈ break-even.
- A100 clearly wins on the **idle tail + cold start** (~37% cheaper at the same
  seconds), which dominate a sparse/dev endpoint → net **A100 saves money** for this
  usage, at some latency cost.
- **First A100 boot recompiles once** (GPU-keyed cache), then fast.
- Validated on Ampere: the s2-pro recipe's reference config is **A800 80GB**.

## Pointers

- App / serving: [`../fish_s2_pro_modal.py`](../fish_s2_pro_modal.py)
- Client + benchmark: [`../client.py`](../client.py), [`../bench_tts.py`](../bench_tts.py)
- Older (0.19.0) baseline: [`../PERF_BASELINE.md`](../PERF_BASELINE.md)
- Model-selection research: [`./tts-options-research.md`](./tts-options-research.md)
