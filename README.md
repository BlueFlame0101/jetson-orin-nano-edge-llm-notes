# Jetson Orin Nano (8 GB) — edge-LLM + ASR engineering notes

Field notes from running Gemma 4 E4B inference via
[`llama.cpp`](https://github.com/ggml-org/llama.cpp) alongside NeMo
ASR models on the NVIDIA Jetson Orin Nano Super 8 GB on JetPack 6.2.2.

Stuff documented here we hit while trying to ship a real product on this hardware
and couldn't find collected anywhere else. Hopefully saves the next team a few days.

If you're here from a Google search for `cudaMalloc failed on Tegra` or
`NVML_SUCCESS == r INTERNAL ASSERT FAILED` — yes, this is the rabbit hole. Read
on.

---

## TL;DR — the four findings

1. **`cma=512M` on the kernel cmdline is the missing piece** to reliably load
   Gemma-class (~3 GB CUDA weight buffer) models with `llama.cpp` dev-build on
   8 GB Orin Nano. The default 256 MB CMA region leaves NvMap unable to
   satisfy the contiguous allocation. Bumping to `cma=1G` fails (kernel
   reserve OOM at boot due to Tegra234 carveouts); **512M is the practical
   ceiling**. See [docs/cma-tuning-tegra234.md](docs/cma-tuning-tegra234.md).

2. **There are TWO `extlinux.conf` on an NVMe-booted Orin Nano** — one on the
   rootfs (NVMe `/dev/nvme0n1p1`), one on eMMC `/dev/mmcblk0p1`. L4TLauncher
   reads the **eMMC** one. Edit the wrong file and your reboot has no effect.
   Cost us an hour the first time. See
   [docs/cma-tuning-tegra234.md](docs/cma-tuning-tegra234.md#the-extlinuxconf-trap).

3. **`llama.cpp` dev-build + PyTorch-based ASR cannot share a CUDA context
   on Jetson** as of llama.cpp `f3c3e0e` + PyTorch from NVIDIA's official
   Jetson wheel. PyTorch's `CUDACachingAllocator` makes an NVML query at init
   that asserts (`NVML_SUCCESS == r INTERNAL ASSERT FAILED at
   CUDACachingAllocator.cpp:838`) when `llama.cpp` has already initialised a
   CUDA context on the same device. Affects **all PyTorch-CUDA-based ASR
   providers** (fa-fastconformer, ar-fastconformer, Parakeet TDT v3, …).
   Unaffected: CTranslate2-based providers (faster-whisper, Røst-CT2).
   Reproducer + open-for-help in
   [docs/pytorch-nvml-conflict.md](docs/pytorch-nvml-conflict.md).

4. **Working dev-build config** for Gemma 4 E4B Q4_K_M on this hardware:
   `-ngl 28 --ctx-size 1024 --batch-size 128 --fit off` with env
   `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 LLAMA_ARG_FIT=off`. Yields ~10 s/call,
   ~48 tok/s prompt-eval, ~5.9 tok/s generation. `-ngl 999` (full offload) does
   not fit — the 3 GB contiguous weight buffer fails even with `cma=512M`.
   See [docs/llama-cpp-orin-nano-8gb.md](docs/llama-cpp-orin-nano-8gb.md).

---

## Memory budget (8 GB unified) — what actually fits

| Component | Size on Orin Nano 8GB | Notes |
|---|---|---|
| Total unified RAM | 7.6 GB usable | Reported as 7608 MB |
| Kernel + system services (multi-user.target) | ~320 MB | Drops to ~1.4 GB if `graphical.target` |
| CMA region (recommended) | 512 MB | Bumped from 256 MB default |
| Page cache (post-eviction) | ~200 MB | Drops on `vm.drop_caches=3` |
| Gemma 4 E4B Q4_K_M weights resident (`-ngl 28`) | ~929 MB CUDA buf | ~2.1 GB total RSS during inference |
| Gemma 4 E4B Q4_K_M weights resident (`-ngl 999`) | ~3 GB CUDA buf | Does not fit (contiguity, not size) |
| NeMo fa-fastconformer-hybrid-large on CUDA | 458 MB allocated, 480 MB reserved | PyTorch caching allocator footprint |
| Røst-v3-whisper-1.5b CT2 int8 on CPU | ~1.5 GB | Non-CUDA — coexists fine |

What does NOT fit on dev-build:
- Gemma at `-ngl 999` (full GPU offload).
- Gemma anywhere + PyTorch-based ASR on CUDA simultaneously (see finding #3 —
  not a memory issue, an allocator-init issue).

What MIGHT fit, untested by us:
- The production NVIDIA container
  `ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-orin` (delayed until JetPack
  7.2 per [Jetson SDK roadmap](https://developer.nvidia.com/embedded/jetson-linux-r3672)).
  Uses a different CUDA allocator strategy and may not hit the
  PyTorch/llama.cpp NVML conflict. If you've tested this combo on 7.2, PRs
  welcome.

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
  [`nvidia/stt_fa_fastconformer_hybrid_large`](https://huggingface.co/nvidia/stt_fa_fastconformer_hybrid_large)
  (the bug is not specific to fa — any NeMo PyTorch-CUDA model triggers it)

---

## Reproducing

Scripts in [`scripts/`](scripts/) are runnable on a Jetson with the stack
above. Each takes its config from env vars at the top — set `GGUF`,
`LLAMA_BIN`, etc. to your local paths and they should work end-to-end.

```bash
# Quick memory characterisation
./scripts/probe-memory.sh

# Single-shot llama-server config probe
GGUF=/path/to/gemma.gguf LLAMA_BIN=/path/to/llama-server \
  ./scripts/try-llama-server.sh 28 1024

# Isolated ASR CUDA load — does it work in isolation?
./scripts/isolated-asr-load.py

# Concurrent load: llama-server + PyTorch ASR — triggers the NVML conflict
./scripts/concurrent-loadtest.sh
```

Detailed methodology + raw output examples in each `docs/` page.

---

## About

These notes were collected by the team building [LOQUA](https://loqua.dk) — an
airgapped speech-to-speech translation device for Danish institutions
(hospitals, police, municipalities). We needed to know exactly how far we could
push edge-LLM + ASR on the Orin Nano 8 GB before committing to a hardware tier,
and the published material on the topic was thin. So we wrote our own.

We're shipping a product, not selling notes — happy to take corrections,
contributions, or "this is solved upstream now" PRs.

Maintainer: Malthe

---

## License

[Apache License 2.0](LICENSE). Use, fork, redistribute — explicit patent grant.
