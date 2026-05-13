#!/bin/bash
# Reproduces the PyTorch CUDACachingAllocator NVML assertion when llama.cpp
# already holds a CUDA context on the same device.
#
# Sequence:
#   1. Clean slate (kill stray procs)
#   2. Start llama-server (working dev-config: -ngl 28 ctx 1024 batch 128)
#   3. Wait for /health
#   4. From a Python process, try to load a NeMo PyTorch ASR on CUDA
#   5. Observe: PyTorch fails with NVML_SUCCESS == r assertion
#
# Compare with isolated-asr-load.py, which loads the same ASR in isolation
# and succeeds.
#
# Usage: GGUF=... LLAMA_BIN=... ./concurrent-loadtest.sh [model-id]
set -u

GGUF="${GGUF:?set GGUF to your GGUF path}"
LLAMA_BIN="${LLAMA_BIN:?set LLAMA_BIN to your llama-server path}"
MODEL_ID="${1:-nvidia/stt_fa_fastconformer_hybrid_large}"

LLAMA_LOG=/tmp/concurrent_llama.log

echo "=== [1/5] clean slate ==="
pkill -f llama-server 2>/dev/null
pkill -f isolated-asr-load 2>/dev/null
sleep 2

echo "=== [2/5] start llama-server (dev sweet-spot: -ngl 28 ctx 1024 batch 128) ==="
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
export LLAMA_ARG_FIT=off

# Evict GGUF from page cache
python3 -c "import os; fd=os.open('$GGUF', os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)"

nohup "$LLAMA_BIN" -m "$GGUF" \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 1024 --parallel 1 \
  --batch-size 128 --ubatch-size 128 \
  --fit off -ngl 28 \
  --threads 4 --no-warmup \
  --reasoning off --reasoning-budget 0 > "$LLAMA_LOG" 2>&1 &
LLAMA_PID=$!
echo "llama-server pid=$LLAMA_PID"

echo "=== [3/5] wait for /health (up to 180s) ==="
ok=0
for i in $(seq 1 90); do
  if curl -s -m 3 http://127.0.0.1:8080/health 2>/dev/null | grep -q status; then
    ok=1; break
  fi
  if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
    echo "llama-server died. log tail:"; tail -20 "$LLAMA_LOG"
    exit 2
  fi
  sleep 2
done
[ "$ok" = 1 ] || { echo "llama-server not healthy"; kill "$LLAMA_PID" 2>/dev/null; exit 3; }
echo "llama-server healthy after ~$((i*2))s"

echo
echo "=== [4/5] attempt to load NeMo ASR on CUDA WHILE llama-server runs ==="
export MODEL_ID
python3 - <<'PYEOF' || PY_EXIT=$?
import os
import sys
import traceback
import time

try:
    import nemo.collections.asr as nemo_asr
except Exception:
    print("NeMo not installed in this Python; this test needs nemo_toolkit.")
    sys.exit(4)

model_id = os.environ["MODEL_ID"]
print(f"loading {model_id} on CUDA (llama-server is up)...")
t0 = time.time()
try:
    m = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
        model_id, map_location="cuda",
    )
    print(f"SUCCESS in {time.time()-t0:.1f}s: model is on CUDA")
    print("This is unexpected based on my findings. Please file an issue / PR.")
    sys.exit(0)
except Exception:
    print(f"FAILED in {time.time()-t0:.1f}s:")
    traceback.print_exc()
    print()
    print("If the error contains 'NVML_SUCCESS == r INTERNAL ASSERT FAILED at")
    print("CUDACachingAllocator.cpp:838', you've reproduced the bug.")
    sys.exit(5)
PYEOF

PY_EXIT="${PY_EXIT:-0}"
echo
echo "=== [5/5] teardown ==="
kill "$LLAMA_PID" 2>/dev/null
sleep 1
pkill -f llama-server 2>/dev/null
echo "Python ASR-load exit code: $PY_EXIT"
echo
if [ "$PY_EXIT" = "5" ]; then
  echo "Bug reproduced (PyTorch NVML assertion). See docs/pytorch-nvml-conflict.md"
elif [ "$PY_EXIT" = "0" ]; then
  echo "Bug NOT reproduced. Model loaded fine. Please share your version info!"
else
  echo "Test inconclusive (exit code $PY_EXIT). See output above."
fi
echo "DONE"
