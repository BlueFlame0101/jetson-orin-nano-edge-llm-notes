#!/bin/bash
# NVML-bug diagnostic capture — wraps concurrent-loadtest.sh with parallel
# tegrastats logging + pre/post memory snapshots + stack info.
#
# Goal: produce hard evidence about whether the PyTorch
# CUDACachingAllocator NVML assertion is an OOM or something more subtle.
#
# Usage:
#   GGUF=/path/to/gemma.gguf \
#   LLAMA_BIN=/path/to/llama-server \
#   VENV_PYTHON=/path/to/venv/bin/python \
#   ./nvml-diag-capture.sh [nemo-model-id]
#
# Output: /tmp/nvml-diag-<timestamp>/
#
# See docs/captures/2026-05-14-nvml-repro-with-diagnostics.md for an
# example annotated run.

set -u

GGUF="${GGUF:?set GGUF to your GGUF path}"
LLAMA_BIN="${LLAMA_BIN:?set LLAMA_BIN to your llama-server path}"
VENV_PYTHON="${VENV_PYTHON:?set VENV_PYTHON to your venv python (NeMo installed)}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL_ID="${1:-nvidia/stt_fa_fastconformer_hybrid_large}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="/tmp/nvml-diag-${TS}"
mkdir -p "$OUT"
echo "Capture dir: $OUT"

snapshot() {
    local tag="$1"
    {
        echo "=== ${tag} @ $(date -u +%FT%TZ) ==="
        echo "--- free -m ---"
        free -m
        echo
        echo "--- /proc/meminfo (key fields) ---"
        grep -E '^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal|SwapFree|Dirty|Mlocked|HardwareCorrupted|CmaTotal|CmaFree|NFS_Unstable)' /proc/meminfo
        echo
        echo "--- /proc/buddyinfo ---"
        cat /proc/buddyinfo
        echo
        echo "--- tegrastats single sample ---"
        timeout 2 tegrastats 2>&1 | head -1
        echo
        echo "--- pgrep llama-server / python ---"
        pgrep -af llama-server || echo "(none)"
        pgrep -af 'python.*nemo|isolated-asr-load|concurrent' || echo "(none)"
        echo
    } > "${OUT}/${tag}.txt"
    echo "  snapshot: ${tag}"
}

echo "=== [1/8] clean slate ==="
pkill -f llama-server >/dev/null 2>&1 || true
pkill -f tegrastats >/dev/null 2>&1 || true
pkill -f concurrent >/dev/null 2>&1 || true
pkill -f isolated-asr >/dev/null 2>&1 || true
sleep 2

# Free page cache of the GGUF (avoids races between page cache and CUDA
# allocations on the same physical pages; see docs/llama-cpp-orin-nano-8gb.md)
python3 -c "
import os
fd = os.open('${GGUF}', os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
print('GGUF page-cache evicted')
" || true

echo "=== [2/8] snapshot pre-test ==="
snapshot "00-pre-clean"

echo "=== [3/8] start tegrastats background logger (500ms interval) ==="
nohup tegrastats --interval 500 > "${OUT}/tegrastats.log" 2>&1 < /dev/null &
TEGRA_PID=$!
disown
echo "  tegrastats pid=${TEGRA_PID}"
sleep 1

echo "=== [4/8] note dmesg cursor (for post-test extraction) ==="
DMESG_START_TS="$(date +%s)"
echo "  dmesg cursor at unix ts ${DMESG_START_TS}"

echo "=== [5/8] run concurrent-loadtest.sh ==="
# Prepend venv/bin to PATH so concurrent-loadtest.sh's `python3` resolves
# to the NeMo-equipped venv Python.
VENV_BIN="$(dirname "$VENV_PYTHON")"
export PATH="${VENV_BIN}:${PATH}"
echo "  python3 resolves to: $(which python3)"

cd "$REPO_DIR"
export GGUF
export LLAMA_BIN
bash scripts/concurrent-loadtest.sh "$MODEL_ID" 2>&1 | tee "${OUT}/concurrent-loadtest.out"
sleep 1

echo "=== [6/8] snapshot post-failure ==="
snapshot "01-post-fail"

echo "=== [7/8] stop tegrastats logger ==="
kill "${TEGRA_PID}" 2>/dev/null || true
sleep 1
pkill -f tegrastats >/dev/null 2>&1 || true
echo "  tegrastats lines logged: $(wc -l < "${OUT}/tegrastats.log")"
tail -20 "${OUT}/tegrastats.log" > "${OUT}/tegrastats-tail.log"

# dmesg from this test window (kernel buffer may have rolled, so also
# capture the unfiltered tail for safety)
sudo dmesg 2>/dev/null | tail -100 > "${OUT}/dmesg-tail.log" 2>/dev/null \
    || dmesg 2>&1 | tail -100 > "${OUT}/dmesg-tail.log"

echo "=== [8/8] stack-info ==="
{
    echo "--- /etc/nv_tegra_release ---"
    cat /etc/nv_tegra_release 2>/dev/null
    echo
    echo "--- uname -a ---"
    uname -a
    echo
    echo "--- /proc/cmdline ---"
    cat /proc/cmdline
    echo
    echo "--- venv python version ---"
    "$VENV_PYTHON" --version
    echo
    echo "--- venv pip torch/nemo/numpy ---"
    "$VENV_PYTHON" -m pip list 2>/dev/null | grep -iE '^(torch|nemo|numpy)\s' || true
    echo
    echo "--- llama-server binary version ---"
    "$LLAMA_BIN" --version 2>&1 | head -3
    echo
    echo "--- nvpmodel state ---"
    sudo /usr/sbin/nvpmodel -q 2>/dev/null | head -5
} > "${OUT}/stack-info.txt"

echo
echo "=== CAPTURE COMPLETE ==="
echo "Output directory: ${OUT}"
ls -la "${OUT}/"
