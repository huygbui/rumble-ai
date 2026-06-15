# Dialogue LLM: should we fine-tune Qwen3.6-35B-A3B via Tinker?

_Research date: 2026-06-14. Goal: decide whether to **fine-tune** the dialogue LLM
(candidate: the Qwen3 MoE the team calls "Qwen3.6 35B A3B") via **Tinker** (Thinking
Machines Lab) for a self-hosted, **kid-facing, Australian-English** voice pipeline
(STT → dialogue-LLM → TTS) on Modal — answering two questions: (1) is the fine-tune
**doable on a limited budget** (cost / effort / risk, and what Tinker abstracts vs what
we still own)? (2) what **benefit** would fine-tuning bring for steering kid
conversation, and is it the right lever **vs a prompt + guardrail/moderation stack**?_

This was produced by a multi-agent research workflow: five dimension researchers gathered
primary-source facts (Tinker-as-a-service, the Qwen model itself, the FT-vs-prompting
steering question, Modal serving, and the total cost/effort/risk budget), an independent
fact-checker adversarially re-verified each researcher's highest-stakes claims, and this
synthesis ranks everything against the two questions. **Where a verifier REFUTED or marked
a claim UNCERTAIN, the verifier's corrected reading is used here** — those spots are
called out inline. Most load-bearing facts here postdate the Jan-2026 model cutoff and
were verified against live primary sources (Tinker docs/homepage, the Qwen HF card, vLLM
recipes/PRs, Modal pricing, arXiv); re-check the time-sensitive ones (pricing, model
support, credits) on the live pages before committing spend.

---

## Bottom line

**Fine-tuning is technically DOABLE and cheap on compute, but it is the wrong FIRST
lever and the wrong place for safety. Do a prompt + external-guardrail baseline first,
build a kid-safety/AU eval harness, and only graduate to a Tinker LoRA when that eval
shows a persistent persona/AU-register gap prompting cannot close.** Tinker genuinely
de-risks the hard part (it owns the training GPUs, distributed training, failure
recovery, checkpointing — you never touch an H100 to *train*), the team's exact model is
live on Tinker for LoRA at **$1.07 / M training tokens**, and the trained adapter exports
as portable PEFT/HF weights you self-host on Modal — so there is no contractual lock-in
and compute for a first useful LoRA is **~$2–$50 per run / ~$100–$300 to iterate**. The
real cost is **data curation + ML-eng time (~3–6 person-weeks)**, not GPUs. But the
literature is decisive on the two questions: SFT reliably buys **style/persona/register**
(LIMA, ~1k examples) — which maps perfectly onto a consistent AU-English kid persona — and
it just as reliably **erodes safety even on benign data** (Qi et al., ICLR 2024), so a
kid-product's safety boundary must live in a **separate input/output guardrail
classifier**, never in the fine-tuned weights.

- **Q1 — Doable with limited resources? → YES, with caveats.** Compute and lock-in are
  non-issues; budget for **data + eval engineering + a safety-regression gate**, plus two
  infra risks specific to this model: a recently-added/fragile MoE-LoRA serving path on
  vLLM, and Tinker base-model deprecation churn (pin your data + config, plan to re-tune).
- **Q2 — Benefit, and is FT the right lever? → FT is the right lever for PERSONA/STYLE,
  the WRONG lever for SAFETY, and PREMATURE as a first step.** Prompt + a guardrail model
  is the correct day-one architecture; FT is a later, eval-gated upgrade for persona/AU
  consistency only.

---

## Truths that hold regardless of which path you pick

1. **Fine-tuning is not a safety mechanism — it can quietly make safety WORSE.** Qi et al.
   2023 (arXiv 2310.03693, peer-reviewed at ICLR 2024) show ~10 adversarial examples strip
   guardrails for <$0.20, and **even benign, ordinary SFT data degrades safety alignment**
   ("to a lesser extent," and the effect is hyperparameter-sensitive — high LR / small
   batch worsen it). LoRA is **not** a free pass: follow-on peer-reviewed work (Safe LoRA,
   NeurIPS 2024; arXiv 2511.00382) finds LoRA can degrade safety as much as or more than
   full FT because it readily overwrites the narrow low-rank "safety direction." **So the
   safety boundary must be an independent layer, and every fine-tune must pass a
   safety-regression eval vs the base model.** (Verifier note: "naive SFT degrades safety"
   = CONFIRMED; the only qualifier is that degradation is real-but-not-strictly-inevitable
   with careful hyperparameters — which strengthens, not weakens, the case for the eval
   gate.)
2. **What light SFT reliably buys is STYLE / PERSONA / REGISTER, not knowledge or safety.**
   The LIMA "superficial alignment hypothesis" (arXiv 2305.11206, 65B model, exactly 1,000
   examples, no RL) — and even its leading *refutation* (arXiv 2410.03717) — agree that
   style/format **saturates fast (~100–1,000 examples)** while reasoning/knowledge keeps
   scaling with far more data. So persona, AU-English idiom, sentence length, and
   "stay-in-character" are exactly what a small, high-quality LoRA delivers. Honest hedge:
   persona SFT has documented side effects — warmth tuning can increase sycophancy /
   false-belief affirmation (arXiv 2507.21919) and personas drift at larger token budgets
   — both directly relevant to a kid assistant, so eval for them.
3. **You essentially cannot train on real children's conversations.** The FTC's 2025 COPPA
   amendments (effective 2025-06-23; full compliance deadline **2026-04-22**) now classify
   **voiceprints** as covered personal information — directly relevant to a voice pipeline.
   **Verifier correction (UNCERTAIN → narrowed):** the rule does NOT broadly require
   parental consent for *any* AI-training use of a child's data; its specific new
   requirement is **separate verifiable parental consent to DISCLOSE a child's data to a
   THIRD PARTY to train/develop AI**. That is exactly the Tinker pathway — sending real
   child transcripts/audio to a hosted third-party fine-tuning service squarely triggers
   it. Net: **build the SFT set from synthetic, adult-authored / teacher-reviewed
   AU-English dialogue; never send real child data to Tinker.** (Legal-counsel question,
   not just technical.)
4. **The model's "3B active" does NOT shrink the serving footprint.** All 35B expert
   weights are resident: ~**70 GB at bf16**, ~**37.5 GB at FP8** (Qwen ships an official
   FP8 build). It is fast (compute is ~3B-class) but it is not a 3B-sized deployment.
5. **The named model resolves to a real model — but it's a NEW hybrid architecture, not
   the older plain MoE.** See naming section below. This changes serving maturity and the
   LoRA-on-MoE story.

---

## What "Qwen3.6 35B A3B" actually is (naming resolved)

**Confirmed (high confidence):** the user's "Qwen3.6 35B A3B" is real and essentially
correctly named. Canonical id: **`Qwen/Qwen3.6-35B-A3B`** (HF model card; Apache-2.0;
released ~2026-04-16; ~35B total / ~3B active). It is **NOT** the older plain-transformer
`Qwen3-30B-A3B` (April 2025). Differences that matter:

- **Hybrid linear-attention + sparse MoE:** 40 layers as 10× [3× Gated DeltaNet → MoE) + 1×
  (Gated Attention/GQA → MoE)], **256 experts (8 routed + 1 shared)**, only 10/40 layers use
  full attention. (source: https://huggingface.co/Qwen/Qwen3.6-35B-A3B,
  https://github.com/QwenLM/Qwen3.6)
- **Multimodal** (vision encoder); for this text-only pipeline serve with vLLM's
  `--language-model-only` to drop the vision tower. (source:
  https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B)
- **Hybrid thinking model:** thinking ON by default; disable for low-latency voice via
  `chat_template_kwargs:{"enable_thinking": false}`. Community reports note some runtimes
  still emit `<think>` despite the flag — verify your serving stack honors it. (source:
  https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- 262K native context (extensible toward ~1M). For a short-turn voice app, cap context low
  (8–32K) to preserve VRAM/latency headroom.

> **One researcher initially mis-resolved this** to "Qwen3-30B-A3B" / "no verifiable
> Qwen3.6 35B exists." The other four researchers AND every fact-check verdict confirm
> `Qwen/Qwen3.6-35B-A3B` is real and Tinker-supported. The model name is correct; only the
> exact Tinker base-model *string* (bare `Qwen3.6-35B-A3B` on the pricing table vs the
> HF-style `Qwen/Qwen3.6-35B-A3B` used elsewhere) is **medium confidence** — confirm the
> precise `base_model=` string before the first run.

**Fit caveat:** this is a coding/agentic flagship (SWE-bench Verified 73.4) — somewhat
over-spec for kid dialogue. A smaller Tinker model (e.g. Qwen3.5-9B dense or Qwen3.5-4B)
may serve kid dialogue as well, cost far less to serve, and avoid the hybrid-MoE serving
edges. Validate model fit before committing to the 35B MoE.

---

## Decision table

| Option | What it steers | Cost / effort | Risk | When to choose |
|--------|----------------|---------------|------|----------------|
| **1. Prompt-only baseline** | Persona, AU register, tone, refusal phrasing — but **brittle**: "swings" under prompt changes; few-shot bloats context/latency (arXiv 2502.11789) | ~Free; hours–days. No training. Served on the Apache-2.0 base via vLLM on Modal | Style instability; no safety enforcement; prompt-injection / jailbreak exposure | **Day one, always.** The mandatory starting point and the thing a FT must beat |
| **2. Prompt + guardrail model** | Same persona steering as (1), **plus a real safety boundary** via an independent input/output classifier (Llama Guard-style / ShieldGemma / NeMo Guardrails) with a custom kid taxonomy | Low; days. One extra small classifier endpoint on Modal | Guard latency + an AU/kid taxonomy may need its own tuning; over/under-blocking to calibrate | **Day one for any kid-facing launch.** This is the correct safety architecture regardless of whether you ever fine-tune |
| **3. LoRA SFT via Tinker** | **Durable** persona / AU-English register / concise spoken style / stay-in-character (LIMA: ~1k good examples) | Compute **~$2–$50/run, ~$100–$300 to iterate**; **~3–6 person-weeks** dominated by synthetic-data curation + eval | **Safety regression** (must gate); over-refusal harming kid UX; MoE-LoRA serving fragility; Tinker base deprecation | **Only after** the eval shows a persistent persona/AU gap prompting can't close, AND a safety-regression gate is in place. Keep guardrail (2) on top |
| **4. Preference / RL (DPO/GRPO) via Tinker** | Finer refusal *quality* — graceful, age-appropriate, explained refusals; selective refusal prompting can't reliably enforce | Higher than SFT (sampling + reward/judge loop adds prefill $0.36/M + sample $0.89/M usage); more eng + a reward signal you must build | All of (3) plus over-refusal (XSTest/OR-Bench) and reward-hacking; harder to debug | **Last.** Only if SFT + guardrail still leave a measured refusal-quality gap and you have a trustworthy preference/eval signal |

Safety is a **layer (2)**, not a row you "win" by fine-tuning. Defense-in-depth — tuned
persona model (3/4) *behind* an independent guard (2) — beats either alone (survey +
Llama Guard, arXiv 2312.06674; MinorBench arXiv 2503.10242; Safe-Child-LLM arXiv
2506.13510, which finds even frontier models have child-facing vulnerabilities and that
adult-centric refusal metrics are inadequate for minors).

---

## What Tinker abstracts vs what you still own

**Tinker owns (you never touch training hardware):** the GPU clusters, distributed
training, scheduling/resource allocation, transparent recovery from GPU crashes,
tokenization, loss compute, and remote weight persistence/checkpointing. Your scripts run
**CPU-only** and call four low-level primitives — `forward_backward`, `optim_step`,
`sample`, and weight persistence (`save_weights_and_get_sampling_client` /
`weights.download`) — with the open-source "Tinker Cookbook" wrapping common recipes.
(source: https://tinker-docs.thinkingmachines.ai/tinker/,
https://github.com/thinking-machines-lab/tinker-cookbook)

**You still own:** the dataset (synthetic AU-English kid dialogue + safety/refusal data),
the training loop and loss/eval code, model selection, the eval + safety-regression
harness, and **serving** on your own Modal infra.

**Confirmed hard constraints (all three high-stakes Tinker claims verified):**
- **Tinker is LoRA-ONLY — no full fine-tuning.** Verbatim from docs: "Tinker implements
  low-rank adaptation (LoRA) fine-tuning, not full fine-tuning." Any plan assuming full FT
  is wrong. Nuance: TML's own research says LoRA *matches* full FT for RL and
  small-to-medium SFT, but can underperform once a supervised set exceeds LoRA capacity —
  relevant if your kid corpus grows large (tune rank/LR, follow "LoRA Without Regret").
  (source: https://tinker-docs.thinkingmachines.ai/tinker/,
  https://thinkingmachines.ai/blog/lora/)
- **`Qwen/Qwen3.6-35B-A3B` is live on Tinker** at **$1.07/M train, $0.89/M sample, $0.36/M
  prefill, 64K context** (CONFIRMED on the Tinker homepage across repeated fetches; MoE is
  "priced by active params," which is why a 35B-total MoE trains as cheaply as a ~3B
  model). Caveats: figures read from the **homepage**, not independently re-confirmed on a
  separate docs pricing page (which 404'd) — **re-check live before budgeting**; do not
  confuse these FT rates with the cheaper per-token *inference* prices other providers
  quote for serving this model. (source: https://thinkingmachines.ai/tinker/)
- **Weights are portable — no inference lock-in.** `weights.build_hf_model()` writes a
  standard merged HF dir (config + tokenizer + safetensors, optional
  `serving_format="vllm"`); `weights.build_lora_adapter()` writes a standard PEFT adapter
  (`adapter_config.json` + `adapter_model.safetensors`). Both download to your disk via
  `weights.download` and run on vLLM/SGLang/transformers — i.e. on Modal. **ToS caveat: on
  contract termination Tinker commits only to a 30-day export window — keep local copies.**
  (source: https://tinker-docs.thinkingmachines.ai/tutorials/deployment/export-hf/,
  https://tinker-docs.thinkingmachines.ai/tutorials/deployment/lora-adapter/)

**Concrete scale anchor:** Tinker's own "first SFT" example used ~7,244 train + 500 val
examples, rank-32 LoRA, 904 iters × 4 epochs, **~3 hours** wall-clock; a comparable
walkthrough was ~300 lines of Python. New users have reported **~$150 starter credits**
(reported during late-2025 waitlist clearance; **not re-confirmed at GA — open question**).
(source: https://tinker-docs.thinkingmachines.ai/tutorials/basics/first-sft/,
https://www.datacamp.com/tutorial/tinker-tutorial)

---

## Serving the result on Modal (verified)

- **Single 80 GB GPU at FP8 is the recommended config.** Qwen ships an official
  `Qwen/Qwen3.6-35B-A3B-FP8` build (~**37.5 GB**); the vLLM recipe explicitly lists
  "Hardware (FP8): single **H100/H200**." Because only 10/40 layers carry growing KV cache,
  KV is tiny (~5 GB even at 262K ctx), leaving ample headroom after the ~37.5 GB of
  weights. **Verifier correction:** the FP8 path is an **H100/H200** plan, **not** an
  A100-80GB plan — Ampere (A100) has no native FP8 compute and only runs FP8 weight-only
  via Marlin (slower). bf16 (~70 GB) fits one 80 GB card but with tight KV; the FP8 card's
  TP=8 example is a throughput default, not a fit requirement. (source:
  https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8,
  https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B)
- **Modal GPU cost (per-second, $0 while scaled to zero):** H100 ~**$3.95/hr**, H200
  ~$4.54/hr, A100-80GB ~$2.50/hr (Marlin-FP8 / bf16-TP2 only), L40S ~$1.95/hr. Back of
  envelope on H100: ~$118/mo at 1 active hr/day, ~$948/mo at 8 hr/day, ~$2,844/mo if 24×7.
  With the team's scale-to-zero preference, a bursty kid app likely costs **tens of
  $/month plus cold-start overhead**. (source: https://modal.com/pricing)
- **Serving the LoRA: hot-load is now possible but is the riskiest link — keep merge as
  the safe fallback.** CONFIRMED in code/release notes: a single current vLLM (**v0.22.0+**,
  current v0.23.0) supports BOTH the hybrid Gated-DeltaNet architecture (since v0.17.0) AND
  expert-layer MoE LoRA hot-load via `--enable-mixed-moe-lora-format` /
  `is_3d_lora_weight` (PR #42242, shipped v0.22.0; a test even references
  `Qwen/Qwen3.6-35B-A3B`). **But:** that full-model LoRA test is `@pytest.mark.skip` ("too
  big") so it's not CI-validated at 35B; a sibling-arch report (vLLM #38520, Qwen3.5-35B-A3B
  LoRA silently no-op) and PEFT-merge failures (tinker-cookbook #75) exist; and vLLM
  **trusts `is_3d_lora_weight` blindly — a wrong flag silently produces garbage.** So
  **smoke-test the Tinker→vLLM hot-load round-trip on this exact model first; if it
  misbehaves, merge the LoRA into the base (`build_hf_model`) and serve a full model** —
  the verifier's recommended safe route for this architecture. (source:
  https://github.com/vllm-project/vllm/pull/42242,
  https://github.com/vllm-project/vllm/releases/tag/v0.22.0,
  https://github.com/vllm-project/vllm/issues/38520,
  https://github.com/thinking-machines-lab/tinker-cookbook/issues/75)
- **Cold-start under scale-to-zero: don't over-promise.** **Verifier downgraded this to
  UNCERTAIN.** Modal's headline "~12s" cold-start figure is for a **3B** model, not a
  37–70 GB MoE; GPU memory snapshots are an **alpha** feature and Modal's own docs warn
  they "generally will not improve — and may even worsen" cold starts when **weight loading
  dominates** (which it does at this size, since restoring tens of GB into VRAM is
  bandwidth-bound). Realistic expectation: **low-tens-of-seconds at best, measured on the
  actual model.** Modal recommends a small **warm floor (`min_containers`)** for
  latency-sensitive interactive loops rather than relying on pure scale-to-zero — a real
  tension with the idle-cost preference for a voice product. (source:
  https://modal.com/docs/guide/memory-snapshots, https://modal.com/blog/gpu-mem-snapshots)
- **Latency for the voice loop is plausible but UNVERIFIED** for this exact model: ~60–120
  tok/s single-stream decode and sub-second TTFT on one H100 at FP8 is an *estimate* (no
  primary benchmark exists). Run non-thinking mode, keep generations short, and **measure
  in the full STT→LLM→TTS loop** before trusting any number. (source:
  https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B)

---

## Recommended staged plan (cheapest-first, tailored to this repo)

The `llm/` stage is currently empty (`llm/README.md`: "one Modal app per model, mirroring
`tts/`"). Mirror the `tts/` pattern: one Modal app per candidate, with an
OpenAI-compatible endpoint. Spend stays at ~$0 training until a fine-tune is *earned*.

**Stage 0 — Build the eval set FIRST (no training).** A few hundred AU-English kid prompts
+ adversarial/safety probes with a rubric: age-appropriateness, AU idiom/register, concise
spoken-style output, and *correct* refusals. Include an **over-refusal** set
(XSTest/OR-Bench-style) so a future FT can't silently make the bot refuse innocent
questions. This is the asset that decides everything downstream; it is reusable across all
stages. (Closest external benchmark: Safe-Child-LLM, US-centric — you'll build the AU
piece yourselves.)

**Stage 1 — Prompt + guardrail baseline on Modal (the launchable architecture).** Serve
the Apache-2.0 `Qwen/Qwen3.6-35B-A3B-FP8` base via vLLM on a single H100, non-thinking
mode, `--language-model-only`, capped context — as `llm/qwen3_dialogue.py` mirroring the
`tts/` apps. Put an **independent guard classifier** (Llama Guard-style / ShieldGemma /
NeMo Guardrails) on input AND output with a custom kid/AU taxonomy as a second small Modal
app. Tune the system prompt + few-shot persona. **Measure against the Stage-0 eval.** This
is the day-one product; safety lives here, not in any weights.

**Stage 2 — De-risk before you ever fine-tune.** While Stage 1 runs: (a) confirm the exact
Tinker `base_model=` string and re-check live pricing + starter-credit status; (b) build
the **synthetic, adult-authored / teacher-reviewed** AU-English kid-dialogue SFT set
(quality ≫ quantity; ~few-hundred-to-few-thousand multi-turn examples; mix in 5–15% general
instruction data to blunt catastrophic forgetting) — **never real child data** (COPPA
third-party-disclosure trigger); (c) smoke-test the Tinker→vLLM serving round-trip
(adapter hot-load vs merged-model) on this exact model with a throwaway tiny LoRA.

**Stage 3 — Graduate to a Tinker LoRA SFT ONLY on a measured trigger.** **Trigger =
repeated, reproducible Stage-1 baseline failures on the eval set** — consistent persona /
AU-register drift or concise-spoken-style failures that prompting + few-shot cannot close
— **AND** a safety-regression gate is wired up. Then: a rank-32 LoRA SFT run on Tinker
(~$2–$50; iterate ~$100–$300), export the adapter/merged weights, serve on Modal *behind
the same Stage-1 guardrail*. **Gate every fine-tune on a safety-regression eval vs the
base** (it must not get worse) and on the over-refusal set. Keep the dataset + training
config reproducible to survive Tinker base-model deprecation (its predecessor
Qwen3.5-35B-A3B already retired ~2026-06-12).

**Stage 4 — Preference/RL (DPO/GRPO) via Tinker, only if needed.** Pursue only if SFT +
guardrail still leave a measured refusal-quality gap (graceful, explained, selective
refusals) and you have a trustworthy preference/eval signal. Highest effort/risk; do last.

**Decision rule in one line:** *prompt + guard ships the product; an eval-gated LoRA is an
optimization for persona/AU consistency, never the safety boundary; full FT is not on the
table (Tinker is LoRA-only).*

---

## Open questions to resolve before committing

1. **Exact Tinker `base_model=` string** for Qwen3.6-35B-A3B (bare name on pricing table vs
   `Qwen/...` HF form), and **re-confirm live pricing** ($1.07/M train) + whether the
   **$150 starter credit** still applies at GA — read from homepage, not an independent
   docs pricing page. (medium confidence)
2. **Tinker data-residency / privacy terms** for uploading (even synthetic) kid-related
   training data, and COPPA / Australian Privacy Act posture for the **third-party
   AI-training-disclosure** trigger — legal-counsel question, especially given voiceprints
   are now covered. (open)
3. **MoE-LoRA serving round-trip on THIS model:** does Tinker's exported adapter actually
   target expert layers (needing the 3D `is_3d_lora_weight` path), and does the
   Tinker→vLLM hot-load work end-to-end — or must you merge? Smoke-test before relying on
   hot-load. (medium confidence; merge is the safe fallback)
4. **Measured latency** of non-thinking Qwen3.6-35B-A3B in the full STT→LLM→TTS loop on
   Modal's H100, and whether `enable_thinking=false` is honored by your serving stack.
   (low confidence — estimate only)
5. **Cold-start at 37–70 GB on Modal under scale-to-zero** — measure it; expect
   low-tens-of-seconds and consider a small warm floor for the voice loop rather than pure
   scale-to-zero. (uncertain per verifier)
6. **Is the 35B MoE even the right model?** A smaller Tinker model (Qwen3.5-9B / 4B) may
   match kid-dialogue quality, serve cheaper, and dodge the hybrid-MoE serving edges and
   the vision encoder you don't need. Validate fit in Stage 1. (open)
7. **No off-the-shelf AU-English kid-dialogue safety/quality benchmark exists** — the
   Stage-0 custom eval is unavoidable; scope its build cost. (open)
