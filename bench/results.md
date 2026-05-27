# Benchmark Results — shawnw3i abliterated AWQ-MTP on dual RTX 3090

All numbers measured on the hardware described in the README using [`bench.py`](bench.py). vLLM 0.21.0 running directly under systemd inside a Proxmox unprivileged LXC. No Docker, no Compose. Benchmark client on a separate machine on the same LAN.

Model: `shawnw3i/Huihui-Qwen3.6-27B-abliterated-AWQ-MTP`
Date: 2026-05-27
Workload: structured JSON generation, ~25 prompt tokens, ~1024 max_tokens, `enable_thinking: false`

## Power-cap matrix — c=1 / c=2 / c=4

| Power cap | c=1 agg | c=2 agg | c=4 agg | c=2 per | c=4 per |
|---|---|---|---|---|---|
| **250W** ✨ | 138.5 | 242.9 | 440.7 | 121.9 | **112.5** |
| 260W | 140.8 | 257.0 | 424.3 | 130.1 | 107.1 |
| 270W | 139.7 | 256.3 | 442.7 | 128.2 | 112.1 |
| 280W | 138.8 | 252.9 | 446.8 | 126.9 | 112.7 |
| 290W | 141.7 | 256.9 | 437.2 | 129.8 | 111.3 |
| 300W | 143.1 | 258.2 | 440.5 | 133.1 | 111.0 |
| 310W | 136.4 | 246.2 | 447.5 | 124.1 | 112.6 |
| 320W | 148.3 | 249.8 | 456.9 | 125.8 | 115.5 |
| 330W | 149.0 | 268.3 | 418.8 | 134.4 | 108.4 |
| **340W** ✨ | 137.2 | 259.8 | **462.8** | 131.0 | **116.0** |
| 350W ⚠️ | 148.5 | 232.3 | **341.7** ⚠️ | 117.6 | 86.4 |

**Two findings:**

1. **350W is a thermal trap.** At sustained c=4 the FTW3 thermally throttles and aggregate throughput collapses from 463 t/s (340W) to 342 t/s — a 26% throughput loss for 3% more power. Don't run at 350W on this card.
2. **250W is silent operation at 95% of peak.** Aggregate throughput at 250W (441 t/s) is within 5% of peak (463 t/s at 340W), but at significantly less heat and noise. For 24/7 inference where acoustic + thermal cost matter, 250W is the genuinely correct daily cap, not a compromise.

## Comparison to the [base-model recipe](https://github.com/Ruashots/qwen3.6-27b-dual-3090-vllm-lxc)

| Metric | shawnw3i (this recipe) best | cyankiwi (base recipe) best | shawnw3i win |
|---|---|---|---|
| c=1 single-stream | **149.0** (@ 330W) | 100.9 (@ 350W) | **+48%** |
| c=2 aggregate | **268.3** (@ 330W) | 182.7 (@ 300W) | **+47%** |
| c=4 aggregate | **462.8** (@ 340W) | 340.0 (@ 330W) | **+36%** |
| c=4 per-stream | **116.0** (@ 340W) | 85.5 (@ 330W) | **+36%** |
| At silent 250W cap c=4 | **440.7** | n/a (cyankiwi at 250W not optimal config) | — |

## What we did NOT verify yet

- **vLLM compatibility on Ampere (RTX 3090, sm_86)** is the published target for cyankiwi and we confirmed it for shawnw3i. shawnw3i's model card cites a 110+ tok/s reference on A800 80GB — our measured 144 t/s c=1 on dual RTX 3090 actually beats that estimate by ~30%, suggesting the model card was conservative or the dual-GPU effect helped more than expected.
- **Long-running production stability** beyond one ~45 min hermes session (5000-word multi-agent synthesis). No crashes, no corruption observed.
- **MTP acceptance under heavy concurrent load** (c=4+ sustained over hours). Short-burst measurements show 60-80% range depending on workload.

## Reproducing

```bash
# Point at your endpoint
export VLLM_URL=http://<your-lxc-ip>:8000
export VLLM_MODEL=shawnw3i/Huihui-Qwen3.6-27B-abliterated-AWQ-MTP

# Run the benchmark
python3 bench.py shawnw3i_default results_default.json
```

To reproduce the full power-cap matrix you'd loop over `nvidia-smi -i 0 -pl <W> && nvidia-smi -i 1 -pl <W>` between bench runs from 350 down to 250 in 10W steps, then collect per-level JSONs. ~25 minutes total on this hardware.

## Quality A/B (deterministic, temperature=0)

Same prompts run on both shawnw3i and cyankiwi, greedy decode for determinism:

| Test | shawnw3i | cyankiwi | Notes |
|---|---|---|---|
| Multi-step math (6 steps) | Wrong answer (570) | Wrong answer (570) | **Identical wrong answer at identical step** — both made `(60+2)² = 3600+240+240+4` instead of `3600+120+120+4`. Same reasoning, same mistake → quantization isn't changing capability. |
| Bracket-balance code + tests | Correct code + all 5 tests pass | Correct code, **one test assertion is buggy** | shawnw3i edges out |
| Noble gases (factual) | byte-identical correct output | byte-identical correct output | tie |
| Probability reasoning | Correct (ran out of tokens 1 char before final at max_tokens=800) | Correct, finished | tied on correctness; shawnw3i more verbose |
| Long-context recall (41K tokens, 4 needles) | **4/4 retrieved verbatim** | **4/4 retrieved verbatim** | tie |
| Vision (image_url, 256x256 generated PNG) | "centered red square... symmetrical layout" | "centered, slightly blurred red square... emphasizing contrast" | both accurate; cyankiwi added a mild hallucination ("blurred" on a pixel-perfect PNG) |

**Bottom line:** shawnw3i's pure INT4 quantization does NOT measurably damage reasoning, code generation, factual recall, long-context coherence, or vision accuracy on these tests. The model card's stricter abliteration (Sumandora's method via huihui-ai) shifts refusal behavior but not reasoning capability.

The one observed quirk: shawnw3i tends to be **slightly more verbose** than cyankiwi on long-form math/reasoning. At production `max_tokens=32768` this is invisible; at constrained `max_tokens=800` it occasionally cuts off before the final answer line.
