# `llama.cpp` dev-build on Orin Nano 8 GB — what works

Configuration matrix for `llama.cpp` server built from source at commit
`f3c3e0e` (master as of 2026-05-09) running Gemma 4 E4B Q4_K_M on Jetson Orin
Nano Super 8 GB, JetPack 6.2.2.

Production runtime is the NVIDIA container
`ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-orin`, which is a different
beast — these notes are about the dev-build path before JetPack 7.2 lands and
makes the container deployable.

## Prerequisites

- `cma=512M` on the kernel cmdline (see
  [cma-tuning-tegra234.md](cma-tuning-tegra234.md)). Without it most configs
  below fail non-deterministically.
- `multi-user.target` boot. Saves ~1 GB of baseline RAM vs `graphical.target`.
- Build: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 && cmake --build build -j4`.
- Page-cache evict the GGUF immediately before `llama-server` starts (see
  scripts/try-llama-server.sh) — without this, `cudaMalloc` for the weight
  buffer competes with the kernel's page cache copy of the same file.

## Working config (sweet spot)

```bash
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
export LLAMA_ARG_FIT=off

llama-server \
  -m /path/to/gemma-4-E4B-it-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 1024 --parallel 1 \
  --batch-size 128 --ubatch-size 128 \
  --fit off -ngl 28 \
  --threads 4 --no-warmup \
  --reasoning off --reasoning-budget 0
```

**Performance** (single-shot translation call, ~92 prompt tokens → ~48 output
tokens, Danish→Arabic medical phrase):

- Server startup → `/health` ready: ~6–8 s
- Prompt eval: **48 tok/s**
- Generation: **5.9 tok/s** (CPU-bound — adding GPU layers beyond ~12 has no
  speedup; see ngl-sweep below)
- Total call: ~10 s
- Deterministic 3× consecutive (per `§4` of our recovery plan)
- Memory: ~3 GB resident during inference

## Why each flag

| Flag | Why |
|---|---|
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` (env) | Routes the big weight buffer through `cudaMallocManaged` instead of plain `cudaMalloc`. Helps but is **not sufficient** alone — compute buffers still use plain `cudaMalloc` and need CMA headroom. |
| `LLAMA_ARG_FIT=off` (env) + `--fit off` | Disables the auto-fit pre-flight that builds a probe compute-graph. The probe OOMs on Jetson before the real model even loads. |
| `--ctx-size 1024` | 2048 + full offload triggers a too-large compute-buffer reservation. 1024 fits. Yes, that limits the conversation, but for short translation calls 1024 is fine. |
| `--parallel 1` | The default is `-1` (auto) which allocates 4 parallel slots → 4× the KV-cache + 4× the compute-buffer pressure. We make sequential calls only. |
| `--batch-size 128 --ubatch-size 128` | Smaller batches reduce the compute-buffer footprint. Tried `64`; perf identical. Tried `--batch-size 256`; OOMs. |
| `-ngl 28` | The sweet spot — see sweep below. |
| `--threads 4` | Orin Nano has 6 CPU cores; 4 for inference, 2 for OS/scheduling. |
| `--no-warmup` | The warmup runs a dummy generation that allocates intermediate buffers, and on Jetson it sometimes fails. Skipping is safe. |
| `--reasoning off --reasoning-budget 0` | **Critical for Gemma.** Gemma 4's chat template has thinking-mode enabled by default. For complex translation prompts the `<think>` block can fill the entire token budget, leaving `message.content` empty (success=True, text=""). Disabling thinking at template level is necessary unless you raise `max_tokens` very high. |

## `-ngl` sweep

| `-ngl` | Status | Notes |
|---|---|---|
| `999` (full offload, all 26 layers) | ❌ DIES | ~3 GB contiguous CUDA weight buffer — even with `cma=512M`, doesn't fit. |
| `28` | ✅ works | Sweet spot. Some non-Transformer layers go to GPU too, hence "28" of 26. |
| `16` | ❌ DIES | Weights load OK, then `ggml_abort` in `ggml_backend_sched_split_graph` — compute-graph allocation fails. Even with `--batch-size 64`. |
| `12` | ✅ works | Same perf as `-ngl 28` — system is CPU-bound at this point, not GPU. |
| `8` | ✅ works | Works even without `cma=512M`. Slowest perf — 4.6 tok/s gen vs 5.9 at `-ngl 28`. |
| `0` (CPU-only) | ✅ works alone, ❌ DIES with concurrent ASR | Even CPU-only requires ~593 MB CUDA compute-buffer for I/O scratch. |

The CPU-bound plateau between `-ngl 8` and `-ngl 28` is interesting — adding
GPU layers gives no speedup until you have a clear majority on GPU. We
attribute this to memory-bandwidth limitations on Tegra's unified memory:
adding GPU compute capacity doesn't help when the bottleneck is moving
activations back and forth.

## What does NOT work

- `-ngl 999` (full offload) — even with all the tuning above. The 3 GB
  contiguous weight buffer is too large for the post-carveout Tegra memory
  layout to provide. `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` should route this
  through `cudaMallocManaged` (which is fragmentation-tolerant), but on
  Tegra the managed allocator still bottoms out in NvMap/CMA at some point
  and fails. We didn't dig deeper — production runtime (the NVIDIA container)
  is the path for full-offload Gemma on this hardware.
- Concurrent `llama.cpp` + any PyTorch-CUDA workload — see
  [pytorch-nvml-conflict.md](pytorch-nvml-conflict.md). The dev-build
  cannot coexist with PyTorch-based ASR on the same device, regardless of
  `-ngl` setting. CPU-based ASR (CTranslate2 family) is fine.

## Why `f3c3e0e` specifically

We were on this commit when we ran these tests; it's also `origin/master`
HEAD as of 2026-05-12 (no newer commits available at time of testing). If
you're on a newer commit, some of this may have improved — particularly the
unified-memory handling for compute buffers, which is the lingering hot spot.
A PR with updated numbers from a newer commit would be welcome.
