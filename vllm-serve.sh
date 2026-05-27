#!/bin/bash
# vLLM launch script — shawnw3i/Huihui-Qwen3.6-27B-abliterated-AWQ-MTP on 2× RTX 3090
#
# Deltas vs the base-model recipe (cyankiwi/Qwen3.6-27B-AWQ-BF16-INT4):
#   - Model id swapped
#   - --speculative-config num_speculative_tokens bumped from 2 to 3 (per shawnw3i model card)
# Everything else identical — TP=2, fp8 KV, 262K context, max-num-seqs=4, performance-mode interactivity.
#
# GPUs running at 250W cap via /etc/systemd/system/nvidia-pl250.service on the host.
# Persistent across reboots. To uncap, disable that service.
#
# Measured: 144 t/s c=1, 463 t/s c=4 peak (@ 340W cap). At 250W: 138 t/s c=1, 441 t/s c=4 — silent.

set -e

# CUDA toolkit at /opt/cuda (installed via standalone .run, not Debian apt — see base recipe)
export CUDA_HOME=/opt/cuda
export PATH=/opt/cuda/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/opt/cuda/lib64

# Required for max-model-len 262144 on a model card that declares less
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# 2× 3090 with no NVLink — disable NCCL peer-to-peer
export NCCL_P2P_DISABLE=1

exec /opt/vllm/bin/vllm serve shawnw3i/Huihui-Qwen3.6-27B-abliterated-AWQ-MTP \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.98 \
  --mm-encoder-tp-mode data \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --max-num-seqs 4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --performance-mode interactivity \
  --disable-custom-all-reduce \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
