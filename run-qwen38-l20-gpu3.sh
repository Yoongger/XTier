#!/usr/bin/env bash
# Qwen3.8-Flash-Next (FP8, unsloth) on one L20 48G (GPU3) with the XTier packs on the official SGLang main.
#
# Derived from the author's configs/qwen38-rtx4090-3090/run-qwen.sh (Qwen3.8 on a 24 GB card, ~120 GB of host
# memory) and adapted to this box:
#
#   * GPU3 only (TP=1). GPU3 is on NUMA node 1; the CPU engine's workers must sit on the same node, otherwise every
#     expert swap crosses UPI. The engine reads XTIER_NODES (csrc/engine/engine.cpp: detect_topology), and with TP=1
#     the placement does not set it, so without this the workers would be spread over both nodes.
#   * XTIER_THREADS: node 1 has 32 physical cores; 24 of them (the physical cores, one worker each) — the DeepSeek
#     rank on GPU2 uses the same node's cores, so the two models share them while both are running.
#   * XTIER_SPIN_COUNT=32768 (the library default) instead of the production 1048576: 64 spinning workers kept
#     this host at 80% user CPU; the throughput was the same.
#   * The expert slots (48/layer, the author's TP=1 recipe for a 24 GB card) plus the elastic cache; the SSM state
#     cache is capped at 16 entries (the author's production long-context recipe) so the memory goes to the experts.
#   * --mem-fraction-static 0.85 (the card is twice as large as the author's, but the elastic cache needs room).
#
# Settings from the environment: PORT, CONTEXT, KV_POOL, SLOTS, THREADS, MEM, MAX_REQ, MTP=1, XTIER_* (as usual).
set -eo pipefail

M=${MODEL:-/opt/models/unsloth/Qwen3.8-Flash-Next-FP8}
ENV=${XTIER_ENV:-/opt/miniforge3/envs/SGLang}
PORT=${PORT:-8002}
CONTEXT=${CONTEXT:-262144}
KV_POOL=${KV_POOL:-524288}

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13}
export PATH=$CUDA_HOME/bin:$ENV/bin:$PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export no_proxy=127.0.0.1,localhost,::1 NO_PROXY=127.0.0.1,localhost,::1

# GPU3 is on NUMA node 1: the CPU engine (its experts, its shards) belongs to that node too.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export XTIER_NODES=${XTIER_NODES:-1}
export XTIER=${XTIER:-1}
export XTIER_THREADS=${XTIER_THREADS:-24}
export XTIER_MAIN_CPUS=${XTIER_MAIN_CPUS:-auto}
export XTIER_SPIN_COUNT=${XTIER_SPIN_COUNT:-32768}

# prefill as in the author's config (the RTX PRO 5000 recipe)
export XTIER_GPU_PREFILL_CHUNK=${XTIER_GPU_PREFILL_CHUNK:-8192}
export XTIER_PREFILL_EXPERT_CHUNK=${XTIER_PREFILL_EXPERT_CHUNK:-32}
export XTIER_PREFILL_PREFETCH=${XTIER_PREFILL_PREFETCH:-1}
export XTIER_PREFILL_RELEASE_WORKSPACE=${XTIER_PREFILL_RELEASE_WORKSPACE:-1}
export XTIER_PREFILL_ROUTE_ACCUM=${XTIER_PREFILL_ROUTE_ACCUM:-1}
export XTIER_PREFILL_UNPACK=${XTIER_PREFILL_UNPACK:-1}
export XTIER_CPU_PREFILL_BATCH=${XTIER_CPU_PREFILL_BATCH:-1024}
export XTIER_GPU_PREFILL_MIN=${XTIER_GPU_PREFILL_MIN:-1024}

# PLE n-gram rows: read from the checkpoint on the data disk, a row cache in pinned host memory
export XTIER_PLE_ROWS=${XTIER_PLE_ROWS:-nvme}
export XTIER_PLE_CACHE_GIB=${XTIER_PLE_CACHE_GIB:-4}

# hot experts: 48 slots per layer (the author's TP=1 recipe) plus the elastic cache
export XTIER_CACHE_SLOTS=${XTIER_CACHE_SLOTS:-48}
export XTIER_CACHE_SWAPS=${XTIER_CACHE_SWAPS:-32}
export XTIER_CACHE_LOG=${XTIER_CACHE_LOG:-500}
export XTIER_CACHE_ELASTIC=${XTIER_CACHE_ELASTIC:-1}

SPEC=()
if [ "${MTP:-0}" = 1 ]; then   # the checkpoint carries an MTP layer; NEXTN 3 steps as in production
  export XTIER_DRAFT=${XTIER_DRAFT:-1}
  SPEC=(--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1
        --speculative-num-draft-tokens 4 --speculative-draft-model-quantization modelopt_mixed
        --speculative-draft-kv-cache-dtype fp8_e4m3)
fi

ARGS=(
  --model-path $M
  --served-model-name Qwen3.8-Flash-Next
  --host 127.0.0.1 --port $PORT
  --tp-size 1
  --trust-remote-code
  --skip-server-warmup
  --cuda-graph-backend-prefill disabled
  --max-running-requests ${MAX_REQ:-4}
  --cuda-graph-max-bs-decode ${GRAPH_BS:-4}
  --page-size 64
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
  --enable-metrics --enable-cache-report
  --weight-loader-drop-cache-after-load
  --context-length $CONTEXT
  --max-total-tokens $KV_POOL
  --kv-cache-dtype fp8_e4m3
  --chunked-prefill-size ${CHUNK:-8192}
  --mamba-ssm-dtype bfloat16
  --max-mamba-cache-size 16
  --mem-fraction-static ${MEM:-0.85}
)
[ "${MTP:-0}" = 1 ] && ARGS+=("${SPEC[@]}")

if [ "${DRYRUN:-0}" = 1 ]; then
  printf '%s\n' "${ARGS[@]}"
  exit 0
fi

exec $ENV/bin/python -m sglang.launch_server "${ARGS[@]}" "$@"
