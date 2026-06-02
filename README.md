# Jetson Orin Nano (8 GB) — edge-LLM + ASR engineering notes

Field notes from running Gemma 4 E4B inference via
[`llama.cpp`](https://github.com/ggml-org/llama.cpp) alongside NeMo ASR
models on the NVIDIA Jetson Orin Nano Super 8 GB on JetPack 6.2.2.

This is stuff I hit while trying to ship a real product on this hardware
and couldn't find collected anywhere else. Hopefully saves the next person
a few days.

If you're here from a Google search for `cudaMalloc failed on Tegra` or
`NVML_SUCCESS == r INTERNAL ASSERT FAILED`: yes, this is the rabbit hole.
Read on.

---

## TL;DR: the four findings

1. The kernel cmdline needs `cma=512M` before `llama.cpp` dev-build will
   reliably load anything Gemma-sized (~3 GB CUDA weight buffer) on the 8 GB
   Orin Nano. The default 256 MB region leaves NvMap unable to satisfy the
   contiguous allocation. I tried `cma=1G` and it failed at boot (Tegra234
   carveouts mean the kernel can't reserve that much), so 512M is what I
   landed on. See [docs/cma-tuning-tegra234.md](docs/cma-tuning-tegra234.md).
   **(Mechanism correction, 2026-06-02:** NVIDIA staff state in [forum thread
   370049](https://forums.developer.nvidia.com/t/pytorch-cudacachingallocator-nvml-assertion-when-sharing-cuda-context-with-llama-cpp-on-orin-nano-8-gb-jetpack-6-2-2/370049/14)
   that NvMap does not allocate from CMA. The `cma=512M` bump still reliably
   helps in practice, but *why* it helps is now an open question — likely
   general contiguity, not the CMA reserve. See the doc for details.)**

2. On an NVMe-booted Orin Nano there are two `extlinux.conf` files: one on
   the NVMe rootfs (`/dev/nvme0n1p1`), one on eMMC (`/dev/mmcblk0p1`).
   L4TLauncher reads the eMMC copy. I edited the NVMe one, rebooted, and
   nothing changed, which cost me an hour before I figured out why. See
   [docs/cma-tuning-tegra234.md](docs/cma-tuning-tegra234.md#the-extlinuxconf-trap).

3. `llama.cpp` dev-build and PyTorch-based ASR cannot share a CUDA context
   on Jetson (verified against llama.cpp `f3c3e0e` and PyTorch from
   NVIDIA's official Jetson wheel). PyTorch's `CUDACachingAllocator` makes
   an NVML query at init that asserts (`NVML_SUCCESS == r INTERNAL ASSERT
   FAILED at CUDACachingAllocator.cpp:838`) when `llama.cpp` already holds
   a CUDA context on the same device. Every PyTorch-CUDA ASR provider I
   tried hits this: fa-fastconformer, ar-fastconformer, Parakeet TDT v3.
   CTranslate2-based providers (faster-whisper, Røst-CT2) don't hit it. I
   don't have a fix; reproducer and hypotheses in
   [docs/pytorch-nvml-conflict.md](docs/pytorch-nvml-conflict.md).

4. The dev-build config that worked for Gemma 4 E4B Q4_K_M on this
   hardware: `-ngl 28 --ctx-size 1024 --batch-size 128 --fit off`, with env
   vars `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 LLAMA_ARG_FIT=off`. I get
   ~10 s/call, ~48 tok/s on prompt-eval, ~5.9 tok/s on generation. Full
   offload (`-ngl 999`) doesn't fit; the 3 GB contiguous weight buffer
   fails even with `cma=512M`. See
   [docs/llama-cpp-orin-nano-8gb.md](docs/llama-cpp-orin-nano-8gb.md).

---

## Memory budget (8 GB unified): what actually fits

| Component | Size on Orin Nano 8 GB | Notes |
|---|---|---|
| Total unified RAM | 7.6 GB usable | Reported as 7608 MB |
| Kernel + system services (multi-user.target) | ~320 MB | Rises to ~1.4 GB on `graphical.target` |
| CMA region (recommended) | 512 MB | Bumped from 256 MB default |
| Page cache (after `vm.drop_caches=3`) | ~200 MB | |
| Gemma 4 E4B Q4_K_M weights resident (`-ngl 28`) | ~929 MB CUDA buf | ~2.1 GB total RSS during inference |
| Gemma 4 E4B Q4_K_M weights resident (`-ngl 999`) | ~3 GB CUDA buf | Does not fit (contiguity, not size) |
| NeMo fa-fastconformer-hybrid-large on CUDA | 458 MB allocated, 480 MB reserved | PyTorch caching allocator footprint |
| Røst-v3-whisper-1.5b CT2 int8 on CPU | ~1.5 GB (estimated) | Non-CUDA, coexists fine |

What does not fit on dev-build:
- Gemma at `-ngl 999` (full GPU offload).
- Gemma anywhere + PyTorch-based ASR on CUDA simultaneously. That's the
  allocator-init issue from finding #3, not a memory-size issue.

What might fit, untested by me:
- The production NVIDIA container
  `ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-orin`. It needs a driver
  version that ships with JetPack 7.2, which doesn't exist for Orin Nano
  yet (see below). It uses a different CUDA allocator strategy and may
  not hit the PyTorch/llama.cpp NVML conflict. If you've run it
  alongside PyTorch ASR on the same device, PRs welcome.

---

## Why dev-build and not containers

The standard "just use a container" advice doesn't work cleanly on Orin
Nano right now, which is why these notes exist. Specifically:

- **JetPack 7.x is Thor-only.** JetPack 7.0 and 7.1 list support for
  Jetson AGX Thor, T5000 and T4000 only ([Jetson Linux releases](https://developer.nvidia.com/embedded/jetson-linux)).
  Orin Nano is on the 6.2.x branch (latest: 6.2.2, L4T R36.5.0). The
  next JetPack expected to bring the newer driver (R595+) to Orin Nano
  is 7.2, which was originally targeted at Q1 2026, slipped to Q2 2026,
  and was still unreleased as of mid-May 2026 when these notes were
  written.
- **`nvcr.io/nvidia/nemo:26.02` (ARM64) is built for ARM SBSA servers**
  (GH200 / GB200), not Jetson. It requires driver 580.95+ which isn't
  available on Orin Nano until JetPack 7.2.
- **`dustynv/nemo` community container's latest tag is `r36.2.0` from
  December 2023** with a PyTorch 2.2 base. It loads but isn't compatible
  with the ASR models I need.
- **`jetson-ai-lab.io` PyTorch wheels** are generic ARM64 builds. They
  install but fail with `CUBLAS_STATUS_ALLOC_FAILED` on the first matmul
  on Jetson, even on a trivial 100×100 matrix. NVIDIA's official Jetson
  PyTorch wheel is the only one that works.

So the dev-build path documented here isn't a stylistic choice. It's the
only path that currently works for the model + ASR + JetPack 6.2.2
combination on Orin Nano. When 7.2 lands, most of this becomes
re-evaluable.

---

## Test environment

- **Board:** NVIDIA Jetson Orin Nano Super Developer Kit, 8 GB
- **Storage:** Kingston KC3000 NVMe M.2 SSD (1 TB) for rootfs; eMMC for boot
- **OS:** Ubuntu 22.04, L4T R36.5.0 (JetPack 6.2.2)
- **Kernel:** 5.15.148-tegra
- **Power mode:** MAXN_SUPER
- **`llama.cpp`:** built from source at commit `f3c3e0e` with
  `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87`
- **PyTorch:** NVIDIA official Jetson wheel
  ([download index](https://developer.download.nvidia.com/compute/redist/jp/v62/)),
  pinned to `numpy<2`
- **NeMo:** `nemo_toolkit==2.0.0`
- **Model:** Gemma 4 E4B (`gemma-4-E4B-it-Q4_K_M.gguf`, ~5.3 GB on disk)
- **ASR (for conflict reproduction):**
  [`nvidia/stt_fa_fastconformer_hybrid_large`](https://huggingface.co/nvidia/stt_fa_fastconformer_hybrid_large).
  The bug isn't specific to fa; any NeMo PyTorch-CUDA model triggers it.

---

## Reproducing

Scripts in [`scripts/`](scripts/) are runnable on a Jetson with the stack
above. Each takes its config from env vars at the top. Set `GGUF`,
`LLAMA_BIN` etc. to your local paths and they should work end-to-end.

```bash
# Quick memory characterisation
./scripts/probe-memory.sh

# Single-shot llama-server config probe
GGUF=/path/to/gemma.gguf LLAMA_BIN=/path/to/llama-server \
  ./scripts/try-llama-server.sh 28 1024

# Isolated ASR CUDA load: does it work on its own?
./scripts/isolated-asr-load.py

# Concurrent load (llama-server + PyTorch ASR): reproduces the NVML conflict
./scripts/concurrent-loadtest.sh
```

Detailed methodology and raw output examples in each `docs/` page.

---

## About

These notes were collected while building an airgapped edge-LLM + ASR
product on the Orin Nano 8 GB. I needed to know exactly how far I could
push the hardware before committing to a tier, and the published material
on the topic was thin, so I wrote my own.

Shipping a product, not selling notes. Happy to take corrections,
contributions, or "this is solved upstream now" PRs.

Maintainer: Malthe Skriver Pedersen ([@BlueFlame0101](https://github.com/BlueFlame0101))

---

## License

[Apache License 2.0](LICENSE). Use, fork, redistribute. Explicit patent grant.
