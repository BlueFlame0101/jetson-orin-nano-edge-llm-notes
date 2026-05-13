#!/bin/bash
# Single-shot llama-server config probe on Jetson.
# Tests: starts server with a given (-ngl, ctx-size, batch-size), waits for
# /health, fires one translation call, prints timing + tegrastats, kills.
#
# Usage: GGUF=/path/to/model.gguf LLAMA_BIN=/path/to/llama-server \
#        ./try-llama-server.sh [ngl] [ctx-size] [batch-size]
#
# Defaults: ngl=28, ctx=1024, batch=128 — the working sweet spot for Gemma 4 E4B
# Q4_K_M on Orin Nano 8 GB with cma=512M.
set -u

NGL="${1:-28}"
CTX="${2:-1024}"
BATCH="${3:-128}"
GGUF="${GGUF:?set GGUF to your GGUF path}"
BIN="${LLAMA_BIN:?set LLAMA_BIN to your llama-server path}"

pkill -f llama-server 2>/dev/null; sleep 1

# Evict the GGUF from page cache so cudaMalloc doesn't compete for memory
python3 -c "import os; fd=os.open('$GGUF', os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd); print('GGUF evicted from page cache')"

echo "=== baseline tegrastats ==="
timeout 1 tegrastats --interval 500 2>/dev/null | head -1
echo

export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
export LLAMA_ARG_FIT=off

echo "=== starting llama-server: -ngl $NGL --ctx-size $CTX --batch-size $BATCH ==="
nohup "$BIN" -m "$GGUF" \
  --host 127.0.0.1 --port 8080 \
  --ctx-size "$CTX" --parallel 1 \
  --batch-size "$BATCH" --ubatch-size "$BATCH" \
  --fit off -ngl "$NGL" \
  --threads 4 --no-warmup \
  --reasoning off --reasoning-budget 0 > /tmp/llama_try.log 2>&1 &
LP=$!
echo "llama-server pid=$LP"

# Wait for /health (up to 180s)
HEALTHY=0
for i in $(seq 1 90); do
  if curl -s -m 3 http://127.0.0.1:8080/health 2>/dev/null | grep -q status; then
    HEALTHY=1
    echo "HEALTHY after ~$((i*2))s"
    break
  fi
  if ! kill -0 "$LP" 2>/dev/null; then
    echo "llama-server DIED — log tail:"
    tail -20 /tmp/llama_try.log
    exit 2
  fi
  sleep 2
done

if [ "$HEALTHY" = 1 ]; then
  echo
  echo "=== tegrastats after load ==="
  timeout 1 tegrastats --interval 500 2>/dev/null | head -1
  echo

  echo "=== single translation call (Danish -> Arabic medical phrase) ==="
  t0=$(date +%s%3N)
  curl -s -m 180 http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"g","messages":[
      {"role":"system","content":"You translate Danish medical instructions to Arabic. Reply with only the translation."},
      {"role":"user","content":"Du fik to tabletter paracetamol klokken otte i morges."}
    ],"temperature":0,"max_tokens":128}'
  t1=$(date +%s%3N)
  echo
  echo "call took $((t1-t0)) ms"
  echo
  echo "=== tegrastats during/after call ==="
  timeout 1 tegrastats --interval 500 2>/dev/null | head -1
fi

echo
echo "=== llama-server log tail ==="
tail -20 /tmp/llama_try.log

kill "$LP" 2>/dev/null
pkill -f llama-server 2>/dev/null
echo
echo "DONE"
