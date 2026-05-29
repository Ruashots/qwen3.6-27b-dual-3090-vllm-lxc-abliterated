# Gotchas

What we learned the hard way building this recipe. Companion to the [base-recipe GOTCHAS.md](https://github.com/Ruashots/qwen3.6-27b-dual-3090-vllm-lxc/blob/main/GOTCHAS.md) — those apply identically here (e.g. CUDA toolkit via .run not apt, the `/dev/nvidia-modeset` lazy-device gotcha, NCCL_P2P_DISABLE on PCIe-only dual-3090). The entries below are specific to this variant.

---

## 1. 350W power is a thermal trap, not the peak

The intuitive belief: "uncapped = fastest." On this hardware with FTW3 + Turbo blower it's wrong. At sustained c=4 generation under 350W cap, the FTW3 hits thermal throttle, and **c=4 aggregate collapses from 463 t/s (at 340W) to 342 t/s (at 350W)** — a 26% throughput loss for 3% more nameplate power.

Visible in the power-cap matrix:

| 340W | 350W |
|---|---|
| c=4 agg = **462.8 t/s** | c=4 agg = **341.7 t/s** ⚠️ |
| c=4 per-stream = 116.0 | c=4 per-stream = 86.4 ⚠️ |

**Fix:** never run above 340W on this card combination at sustained c=4. The `nvidia-pl250.service` in this repo applies a 250W cap (silent operation, ~5% off peak); 320W is also fine if you want the absolute peak throughput.

This is a case-airflow + cooler-design issue, not a chip issue. A different cooler choice (two blowers, or two open-air with stronger case airflow) would shift the throttle threshold up. With the FTW3's open-air design dumping into the case sandwich, 350W is just over the edge.

---

## 2. Both quants are mixed-precision (corrected after HF discussion)

**Original version of this entry was wrong.** We initially described shawnw3i as "pure INT4 weights" vs cyankiwi's "hybrid BF16-INT4." After publishing, shawnw3i (the model author) clarified on HuggingFace that their quant is **also mixed-precision**: `config.json` lists 101 entries in `modules_to_not_convert`, keeping most linear_attn layers (48/63), `lm_head`, the vision encoder, and the MTP heads in float16.

So the actual structure is two similar-in-spirit mixed-precision AWQ schemes with different preserved-layer choices:

| | shawnw3i | cyankiwi |
|---|---|---|
| Preservation scope | Most linear_attn + lm_head + vision encoder + MTP heads in fp16 | BF16 on linear-attn layers |
| Net size on disk | ~19 GB | ~28 GB |
| Difference is mostly | which non-attention modules get downcast | the BF16 dtype choice for attention paths |

This makes the quality parity we observed (deterministic math/code/reasoning A/B + 5000-word long-form generation + vision) unsurprising rather than counterintuitive. We expected a pure-INT4 quant to lag a hybrid; instead, both are hybrids with different recipes, and they perform comparably on day-to-day workloads.

The original observations still hold:
- Math reasoning (both gave identical wrong answers at identical step — same model, same mistake)
- Code generation
- Factual recall
- Long-context coherence (4/4 needles at 41K tokens)
- Vision accuracy

If you have a production workload sensitive to the gap, run the same A/B before committing — but don't expect the "pure INT4" alarm bells the first version of this gotcha implied.

---

## 3. MTP n=3 actually works (not just labeled)

shawnw3i ships MTP heads as a **separate file** (`model_extra_tensors.safetensors`, ~850 MB) that vLLM loads alongside the main weights when `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` is set. We bumped from cyankiwi's n=2 to shawnw3i's recommended n=3 and measured:
- Acceptance rate: 60-80% across workloads (structured JSON higher, prose lower)
- No degradation in output quality from the larger speculative window
- Visible contribution to throughput vs `num_speculative_tokens` lower or off

If you see weird "drafter not loading" warnings in vLLM startup logs, verify the model directory has `model_extra_tensors.safetensors` present and `--trust-remote-code` is set (shawnw3i requires it for custom MTP config).

---

## 4. The shawnw3i model card under-reports throughput

The model card cites **110+ tok/s on a single A800 80GB** as the headline number. We measured **144 t/s c=1 / 463 t/s c=4 aggregate on dual RTX 3090** — significantly faster despite being on consumer Ampere instead of datacenter Ampere.

Possible reasons:
- Their reference was at fp8 KV but maybe different vLLM version
- A800 has higher memory bandwidth than RTX 3090, but our dual-GPU TP doubles compute
- Their "110 t/s" might have been measured at a sub-optimal config

Doesn't matter much — just don't anchor your expectations to the model card. The recipe in this repo measures higher.

---

## 5. Abliteration method differs from huihui's standard

shawnw3i used [Sumandora's `remove-refusals-with-transformers`](https://github.com/Sumandora/remove-refusals-with-transformers) (single-pass orthogonal projection) as the abliteration method, not huihui-ai's two-pass technique. Despite the name `Huihui-Qwen3.6-27B-abliterated-AWQ-MTP`, this is a different abliteration of the underlying Qwen3.6-27B model than huihui's official `Huihui-Qwen3.6-27B-abliterated-MTP-GGUF`.

In practice: behavior on refusal-edge prompts may differ slightly between shawnw3i's variant and a llama.cpp-served huihui variant. Both are abliterated; both will respond to prompts that base Qwen3.6-27B refuses. If you have specific edge-case prompts that matter, A/B them.

---

## 6. `gpu_memory_utilization=0.98` is fine but tight

Inherited from the base recipe verbatim. 0.98 reserves 23.6 GB of 24 GB per card upfront for KV cache + activations + cudagraph workspace. We never OOM'd in our testing including 5K-word generations and 41K-token contexts.

**The shawnw3i quant has more KV headroom than cyankiwi** (591K tokens vs 375K tokens at fp8 KV) because the smaller weights leave more room. So even at 0.98 GMU you have substantially more breathing room than the base recipe.

If you want safer headroom at the cost of 5-7% throughput, drop to 0.93 — that's what the [base recipe's GOTCHAS.md](https://github.com/Ruashots/qwen3.6-27b-dual-3090-vllm-lxc/blob/main/GOTCHAS.md) calls out as "safer for 24/7." We didn't see the need on this quant.

---

## 7. Vision works alongside MTP — verified

Cited as "image-text-to-text" in the shawnw3i model card but the README doesn't explicitly say MTP + vision coexist. We tested:
- Send `image_url` payload with base64-encoded PNG + text instruction
- Model correctly describes the image
- MTP still engages during the text response generation (visible in vLLM logs)
- Both GPUs balanced at ~14 GB VRAM each

No special flags needed beyond what's already in `vllm-serve.sh`. The `--mm-encoder-tp-mode data` flag is what makes vision work cleanly on TP=2.

---

## 8. Power cap persistence requires nvidia-persistenced

`nvidia-smi -pl <W>` only works if persistence mode is on. On a Proxmox host with the .run-installer NVIDIA driver, this needs the `nvidia-persistenced.service` unit (the .run installer doesn't ship one — see the base recipe). The `nvidia-pl250.service` in this repo has `After=nvidia-persistenced.service` + `Requires=nvidia-persistenced.service` so the ordering is correct, but if you skipped the persistenced unit, the pl250 service will silently fail.

Verify with `systemctl status nvidia-persistenced.service` and `systemctl status nvidia-pl250.service` after reboot.
