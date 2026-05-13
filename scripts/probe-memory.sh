#!/bin/bash
# Diagnose memory state on a Jetson before running CUDA workloads.
# Read tegrastats lfb (largest free block) carefully: anything below
# ~8 MB suggests fragmentation that will bite CUDA allocators.
set -u

echo "=== /proc/cmdline (look for cma= param) ==="
cat /proc/cmdline
echo
echo "=== /proc/meminfo (CMA + general free) ==="
grep -E "MemTotal|MemFree|MemAvailable|Cached|CmaTotal|CmaFree" /proc/meminfo
echo
echo "=== systemd default target ==="
systemctl get-default
echo
echo "=== top RAM consumers (top 10) ==="
ps -eo pmem,rss,comm --sort=-rss | head -11
echo
echo "=== tegrastats baseline (3 samples, watch lfb) ==="
timeout 2 tegrastats --interval 500 2>/dev/null | head -3
echo
echo "=== uptime ==="
uptime
echo
echo "=== any llama-server / ASR processes running? ==="
pgrep -fa 'llama-server|python3' | grep -vE "$0|grep " || echo "(none)"
