# 2026-05-14 — NVML repro with full diagnostics

Follow-up capture for [`pytorch-nvml-conflict.md`](../pytorch-nvml-conflict.md),
prompted by NVIDIA staff request (forum thread) to verify the failure is
not a simple OOM. This run captures `tegrastats` at 500 ms cadence
alongside snapshots of `/proc/meminfo`, `/proc/buddyinfo`, and `free -m`
immediately before and after the reproducer fires.

**TL;DR:** It is not OOM in the classical sense (no kills, no swap pressure,
`MemAvailable` ≈ 6 GB at the snapshot). It **is** contiguous-memory
pressure: during PyTorch's CUDA context init the largest-free-block
collapses to a single 4 MB chunk, NvMap fails to allocate a ~1 GB DMA
buffer (`error 12`), and PyTorch's `CUDACachingAllocator` then asserts
on its NVML query because the device is in a bad state.

## Stack

```
Hardware:   Jetson Orin Nano Super 8 GB, MAXN_SUPER
OS:         JetPack 6.2.2 (L4T R36.5.0)
Kernel:     5.15.148-tegra, cmdline: cma=512M, multi-user.target
Python:     3.10.12
PyTorch:    2.5.0a0+872d972e41.nv24.8 (official NVIDIA Jetson wheel)
NumPy:      1.26.4 (locked <2 per wheel requirement)
NeMo:       2.0.0
llama.cpp:  commit f3c3e0e, built with GGML_CUDA=ON CUDA_ARCH=87
Model:      Gemma 4 E4B Q4_K_M (5.0 GB GGUF), -ngl 28 ctx 1024 batch 128
ASR model:  nvidia/stt_fa_fastconformer_hybrid_large
```

## Reproducer

Used the existing [`scripts/concurrent-loadtest.sh`](../../scripts/concurrent-loadtest.sh)
from this repo, wrapped with parallel tegrastats logging + pre/post
snapshots.

## Pre-test memory state (system idle, llama.cpp NOT running)

```
$ free -m
               total  used  free  shared  buff/cache  available
Mem:            7607  1293  5299      18        1014       6082

$ grep -E '(MemTotal|MemFree|MemAvailable|CmaTotal|CmaFree|SwapTotal)' /proc/meminfo
MemTotal:        7790364 kB
MemFree:         5426612 kB
MemAvailable:    6228472 kB
CmaTotal:         524288 kB
CmaFree:          472596 kB
SwapTotal:             0 kB

$ cat /proc/buddyinfo
Node 0, zone      DMA   3947   2901   1951   1179    972    626    463    289    129    144     77     20     19
Node 0, zone   Normal  24348  17561  15245  21586   8356   2270    871    586    224    206     55     20      8

$ tegrastats (single sample)
RAM 1366/7608MB (lfb 132x4MB) GR3D_FREQ 0%
```

Plenty of headroom, plenty of high-order buddy blocks, lfb is 132 free 4 MB
chunks.

## Test sequence (annotated tegrastats)

llama-server boots at the start; PyTorch attempts `model.to("cuda")` ~10 s
later. Selected samples around the failure window:

```
# llama-server steady-state idle (Gemma fully resident on GPU via UMA)
10:33:48  RAM 2880/7608MB (lfb 1x4MB) GR3D_FREQ 0%
                                ^^^^^
                                lfb already collapsed because llama-server's
                                CUDA context + Gemma weights take ~1.5 GB
                                of NvMap/CMA pool, leaving very little
                                contiguous physical memory.

# PyTorch loads NeMo model on CUDA: pressure builds rapidly
10:34:16  RAM 4032/7608MB (lfb 1x4MB) CPU [96%@1344, ...]
10:34:17  RAM 4094/7608MB (lfb 1x4MB)
10:34:17  RAM 4093/7608MB (lfb 5x4MB)
10:34:18  RAM 4092/7608MB (lfb 5x4MB)
10:34:19  RAM 4091/7608MB (lfb 5x4MB)
10:34:19  RAM 4074/7608MB (lfb 4x4MB)
                              ^^^^^
                              Peak RAM 4094 MB (~54% utilization), but
                              lfb is 1-5 free 4 MB chunks. Contiguous
                              memory is the bottleneck, not raw RAM.

# Failure fires (NvMap rejects allocation, PyTorch NVML asserts, NeMo
# Python frees its CPU-side tensors)
10:34:20  RAM 1738/7608MB (lfb 42x4MB)
10:34:20  RAM 1454/7608MB (lfb 44x4MB)
10:34:21  RAM 1377/7608MB (lfb 48x4MB) (steady — only llama-server remains)
```

## NvMap errors in stderr immediately before the PyTorch traceback

```
NvMapMemAllocInternalTagged: 1075072515 error 12
NvMapMemHandleAlloc: error 0
NvMapMemAllocInternalTagged: 1075072515 error 12
NvMapMemHandleAlloc: error 0
NvMapMemAllocInternalTagged: 1075072515 error 12
NvMapMemHandleAlloc: error 0
NvMapMemAllocInternalTagged: 1075072515 error 12
NvMapMemHandleAlloc: error 0
```

`1075072515 bytes ≈ 1.0 GB`. `error 12` is `ENOMEM`. NvMap is trying to
allocate ~1 GB contiguous DMA buffer for PyTorch's CUDA context and
failing — not because system RAM is exhausted (`MemAvailable` is 6 GB at
this moment), but because the CMA pool that NvMap draws from is largely
already pinned by llama.cpp's ggml-CUDA context.

## PyTorch traceback (verbatim)

```
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at
"/opt/pytorch/pytorch/c10/cuda/CUDACachingAllocator.cpp":838,
please report a bug to PyTorch.

File "torch/nn/modules/module.py", line 1318, in to
    return self._apply(convert)
File "torch/nn/modules/module.py", line 897, in _apply
    module._apply(fn)
File "torch/nn/modules/module.py", line 924, in _apply
    param_applied = fn(param)
File "torch/nn/modules/module.py", line 1304, in convert
    return t.to(...)
```

The assertion fires inside `CUDACachingAllocator::cuda_property` (or
similar — line 838 of CUDACachingAllocator.cpp on the wheel build) which
queries NVML for device properties during caching-allocator init. The
NVML call returns non-`NVML_SUCCESS`, which my read is a consequence of
the NvMap allocation failure leaving the CUDA device in a bad state for
NVML state-queries.

## Post-failure memory state

```
$ free -m
               total  used  free  shared  buff/cache  available
Mem:            7607  1287  3094      18        3226       6012

$ grep -E '(MemFree|MemAvailable|CmaFree)' /proc/meminfo
MemFree:         3168552 kB
MemAvailable:    6156320 kB
CmaFree:          168452 kB
                                  ^^^^^^
                                  CMA pool drained from 472 MB → 168 MB
                                  (~300 MB still held by llama-server's
                                  CUDA context, even after PyTorch backed
                                  out).

$ cat /proc/buddyinfo
Node 0, zone      DMA   3952   2239   1702   1244    608     68     70     23     10     31     28      6      1
Node 0, zone   Normal  59440  50533  38883  22684   2068    402     96     32     87    147     20      6      0
                                                            ^^^
                                                            Higher-order
                                                            blocks recover
                                                            once PyTorch
                                                            releases its
                                                            CPU-side state.
```

## Interpretation

1. The system never runs out of usable RAM. `MemAvailable` stays ≥ 6 GB
   throughout. No OOM-killer activity in `dmesg`. No swap (Jetson has
   none). Without llama.cpp running, the same NeMo ASR loads on CUDA
   alone in 9.8 s with no NvMap errors (see
   [`scripts/isolated-asr-load.py`](../../scripts/isolated-asr-load.py)).
2. The failure is **contiguous-memory exhaustion in the CMA / NvMap pool**.
   `cma=512M` minus llama.cpp's ~300 MB context leaves NvMap unable to
   satisfy a 1 GB DMA-buffer request for PyTorch's CUDA context init.
3. PyTorch's `CUDACachingAllocator::cuda_property` NVML query asserts as
   a downstream symptom of the NvMap failure. It is not robust to
   NvMap-failure device state.

So while the *root* cause is contiguous-memory contention between
ggml-CUDA and PyTorch-CUDA on a Tegra UMA platform with a constrained CMA
pool, the *user-visible* failure is a PyTorch internal assertion that
gives no hint about NvMap.

## Mitigations we have tried

- `PYTORCH_NO_CUDA_NVML=1` env var → no effect on the official Jetson
  wheel.
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` on the llama.cpp side → helps
  llama-server's own weight-buffer allocation, no effect on PyTorch.
- `cma=1G` on kernel cmdline → kernel cannot reserve 1 GB contiguous
  physical memory at boot on Tegra234 (carveouts). `cma=512M` is the
  practical maximum. See [`docs/cma-tuning-tegra234.md`](../cma-tuning-tegra234.md).
- Reverse the load order (PyTorch first, then llama.cpp) → llama.cpp
  fails with `cudaMalloc failed: out of memory` on its 929 MB weight
  buffer. Same underlying contention, different symptom.

## Open questions

1. Is there a PyTorch build-time flag or runtime knob to **disable NVML
   tracking in `CUDACachingAllocator`** for embedded targets where NVML
   semantics differ from discrete GPUs?
2. Are there NvMap tunables (sysfs, kernel module params, `/proc/sys`)
   that would let PyTorch and ggml-CUDA cooperate on a shared CMA pool?
3. Does the JetPack 7.2 NVIDIA llama.cpp container
   (`ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-orin`) bypass this by
   using a different CUDA memory paradigm, or will the same contention
   apply once two CUDA contexts coexist?

## Files

Raw output from the capture run (timestamps in filenames are UTC):

- `00-pre-clean.txt` — `free`, `meminfo`, `buddyinfo`, `tegrastats`
  before anything starts
- `01-post-fail.txt` — same set, immediately after the assertion
- `concurrent-loadtest.out` — full stdout/stderr from the reproducer
- `tegrastats.log` — 500 ms-cadence samples across the entire run
- `stack-info.txt` — `nv_tegra_release`, `uname`, `cmdline`, package
  versions

These are excluded from the repo via `.gitignore`'s `*.log` rule but the
key excerpts are embedded inline above. The capture wrapper used to
generate them is at
[`scripts/nvml-diag-capture.sh`](../../scripts/nvml-diag-capture.sh).
