# Qwen3.6-27B Abliterated on Dual RTX 3090 via vLLM in a Proxmox LXC

Reproducible recipe for serving **`shawnw3i/Huihui-Qwen3.6-27B-abliterated-AWQ-MTP`** from a Proxmox LXC on **2× RTX 3090** with **MTP n=3 speculative decoding**, full vision + tool-calling + reasoning, **256K context**, and a **250W per-GPU power cap for silent 24/7 operation**.

This is the abliterated companion to the [base-model recipe](https://github.com/Ruashots/qwen3.6-27b-dual-3090-vllm-lxc). Same hardware, same vLLM stack, same LXC pattern — different model and a tuned power profile that turns out to be *faster* than the base.

## Headline numbers

Measured on this exact config, dual RTX 3090, 250W cap per card, vLLM 0.21.0 — see [`bench/results.md`](bench/results.md) for the full data and reproduction steps.

| Concurrency | Aggregate t/s | Per-stream t/s |
|---|---|---|
| 1 | **138.5** | 138.5 |
| 2 | 242.9 | 121.9 |
| 4 | **440.7** | 112.5 |

Peak at the optimal 340W cap is **463 t/s aggregate at c=4** with 116 t/s per stream. At the silent 250W operation cap it's **440 t/s aggregate** — still **+30% faster than the [base-model recipe](https://github.com/Ruashots/qwen3.6-27b-dual-3090-vllm-lxc) at its own optimal cap.**

Other measured:
- **MTP acceptance:** 60-80% across workloads (n=3 speculative tokens, embedded in the checkpoint via `model_extra_tensors.safetensors`)
- **KV cache pool:** 591K tokens at fp8 KV — 2.26× concurrency at full 262K context
- **Vision:** validated working alongside tool-calling + MTP
- **Sustained thermal:** G0 67°C, G1 63°C, CPU 75°C under c=4 hermes workload

## Why this is faster than the base AWQ-BF16-INT4 quant

| | shawnw3i (this recipe) | [cyankiwi base-model recipe](https://github.com/Ruashots/qwen3.6-27b-dual-3090-vllm-lxc) |
|---|---|---|
| Format | AWQ mixed-precision (most layers INT4; linear-attn 48/63 + lm_head + vision encoder + MTP heads kept in float16) | AWQ-BF16-INT4 hybrid (BF16 on linear-attn layers) |
| Size on disk | ~19 GB | ~28 GB |
| MTP n | 3 | 2 |
| c=1 single-stream | **149 t/s** (peak) | 101 t/s (peak) |
| c=4 aggregate | **463 t/s** (peak @ 340W) | 340 t/s (peak @ 330W) |
| KV pool | 591K tokens | 375K tokens |
| Abliterated | ✅ (huihui via Sumandora method) | ❌ base |

The smaller checkpoint means less memory bandwidth per token and more VRAM left for KV cache. **Both quants are mixed-precision** — shawnw3i corrected us on this after a friendly HF discussion: their `config.json` lists 101 entries in `modules_to_not_convert`, keeping most linear_attn layers (48/63), `lm_head`, the vision encoder, and MTP heads in float16, while cyankiwi's recipe preserves BF16 specifically on linear-attn. So the quant-quality picture is "two similar-in-spirit mixed-precision schemes with different preserved-layer choices," not "pure INT4 vs hybrid." This now makes the observed quality parity unsurprising rather than counterintuitive. See [GOTCHAS.md](GOTCHAS.md) for details.

## Hardware and software tested

Same hardware as the base recipe — see the [base recipe's hardware table](https://github.com/Ruashots/qwen3.6-27b-dual-3090-vllm-lxc#hardware-and-software-tested). Specifically: ASUS ProArt X870E-Creator, AMD Ryzen 7 9800X3D, 64 GB DDR5-5200, EVGA RTX 3090 FTW3 + reference Turbo blower RTX 3090, no NVLink, Corsair RM1200e PSU, Antec P1 Silent case, Proxmox VE 9.2.

## Quickstart

Assuming you already have the [base recipe's host-side setup](https://github.com/Ruashots/qwen3.6-27b-dual-3090-vllm-lxc#quickstart-proxmox-host--lxc--vllm) (NVIDIA driver on host, nvidia-uvm-init systemd unit, etc.), the abliterated variant only needs:

### Clone an existing vLLM LXC to a new VMID (or build fresh)

Easiest path — clone your base-recipe LXC:

```bash
pct clone <base-recipe-vmid> 5064 --hostname vllm-abliterated --full
pct set 5064 \
  --net0 name=eth0,bridge=vmbr0,gw=<your-gw>,ip=<your-new-ip>/24,hwaddr=<new-mac>,type=veth \
  --onboot 0
pct start 5064
```

### Swap the launch script

Inside the LXC, replace `/usr/local/bin/vllm-serve` with [this version](vllm-serve.sh). The only delta from the base recipe is:

- Model name: `shawnw3i/Huihui-Qwen3.6-27B-abliterated-AWQ-MTP`
- `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` (was n=2)

Then restart `vllm.service`. First start downloads ~20 GB from HuggingFace; subsequent starts are fast.

### Apply the 250W power cap (host side)

Drop in [`systemd/nvidia-pl250.service`](systemd/nvidia-pl250.service) on the **Proxmox host** (not inside the LXC):

```bash
sudo cp systemd/nvidia-pl250.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-pl250.service
sudo nvidia-smi --query-gpu=index,power.limit --format=csv
```

Power cap survives reboots via `WantedBy=multi-user.target` + `After=nvidia-persistenced.service`. If you ever want to uncap, just `systemctl disable --now nvidia-pl250.service` and reboot (or manually `nvidia-smi -pl <vbios-default>`).

## Why 250W and not "uncapped"

The full power-cap matrix we benched (350W → 250W in 10W steps) shows:

| Power cap | c=4 aggregate | Notes |
|---|---|---|
| 350W (≈uncapped) | **342 t/s** | FTW3 thermally throttles at sustained c=4 load |
| **340W** | **463 t/s** ✨ | Peak |
| 320W | 457 t/s | Within 1% of peak |
| 300W | 441 t/s | -5% from peak |
| **250W** ✨ | **441 t/s** | **Same as 300W — clean break of the throttle curve** |

Two findings:
1. **350W is a thermal trap.** The FTW3 dumps so much heat into the case at uncapped power that it throttles under sustained c=4 — you actually get *less* throughput than at any lower cap.
2. **250W is the silent-operation sweet spot.** GPU fans drop noticeably, case fans relax, and you're still within ~5% of the absolute peak. For 24/7 inference where total power, heat, and acoustic footprint matter, 250W is the genuine best choice — not a compromise.

This is specific to the FTW3 + Turbo blower combination on this case. A different cooler choice (e.g., two blower cards or two open-air with better case airflow) would shift the throttle threshold up.

## Repository layout

```
.
├── README.md                              # you are here
├── LICENSE                                # MIT
├── vllm-serve.sh                          # launch script (shawnw3i model + MTP n=3)
├── GOTCHAS.md                             # quality A/B, thermal throttle math, what we tried
├── systemd/
│   ├── vllm.service                       # same as base recipe
│   └── nvidia-pl250.service               # 250W power cap, host-side, persistent
├── host-setup/                            # see base recipe — identical content
└── bench/
    ├── bench.py                           # concurrency benchmark
    └── results.md                         # full power-cap matrix + comparison to base recipe
```

The host-setup files (LXC config example, nvidia-uvm-init.service, nvidia-persistenced.service) are identical to the [base recipe](https://github.com/Ruashots/qwen3.6-27b-dual-3090-vllm-lxc). They're not duplicated here — if you're starting from scratch, follow the base recipe's host setup first, then come back here for the model/serve/cap pieces.

## What this is and isn't

**Is**:
- A working, verified, reproducible recipe for the abliterated variant of Qwen3.6-27B on dual RTX 3090 in a Proxmox LXC, with a 250W silent-operation power profile.
- Faster than the base-model AWQ recipe on this exact hardware.

**Isn't**:
- A Docker / Compose recipe — see Dzombak's blog for that.
- For non-abliterated workloads — if you don't need refusal-free output, the [base recipe](https://github.com/Ruashots/qwen3.6-27b-dual-3090-vllm-lxc) is the right starting point.
- A safety guarantee — abliteration removes the model's refusal training. Use accordingly.

## Credit

- **Quant**: [shawnw3i](https://huggingface.co/shawnw3i) — `Huihui-Qwen3.6-27B-abliterated-AWQ-MTP` is the missing piece that makes abliterated serving on vLLM possible without a custom quantization job. The reference 110+ tok/s on A800 in their model card is what made this recipe worth trying.
- **Abliteration source**: [huihui-ai](https://huggingface.co/huihui-ai), method via [Sumandora's `remove-refusals-with-transformers`](https://github.com/Sumandora/remove-refusals-with-transformers).
- **Base recipe**: [Chris Dzombak's dual-3090 vLLM config](https://www.dzombak.com/blog/2026/04/a-vllm-docker-compose-recipe-for-running-qwen-3-6-27b-on-dual-rtx-3090s-opencode-configuration/) — the shape this builds on.
- **Cross-stack benchmark context**: [noonghunna/qwen36-dual-3090](https://github.com/noonghunna/qwen36-dual-3090).

## License

MIT — see [LICENSE](LICENSE).
