# Memory budget — Gemma 4 E4B + ASR on Orin Nano 8 GB

Numbers from running on a Jetson Orin Nano Super 8 GB, JetPack 6.2.2,
`multi-user.target` boot (no desktop), `cma=512M`. Your numbers will be in the
same ballpark; exact figures vary with kernel version + system services + which
ASR backend.

## System overhead

| State | RAM used | Notes |
|---|---|---|
| Fresh boot, `multi-user.target`, no workload | ~320 MB | Best baseline |
| Fresh boot, `graphical.target`, no workload | ~1.4 GB | GNOME shell + Xorg + dockerd eat ~1 GB |
| After running `llama.cpp` + ASR test once | ~900 MB | Various caches stick around |

## Page cache pressure

The Gemma GGUF is ~5.3 GB on disk. Once you've read it once (via
`llama-server`'s `mmap`), the kernel page cache happily holds it in RAM,
competing with future CUDA allocations. Always evict before launching
`llama-server`:

```python
import os
fd = os.open("/path/to/gemma.gguf", os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
```

Without this we observed `cudaMalloc` failures right after a previous
successful run, even though we'd killed `llama-server`. The page cache held
the GGUF and NvMap couldn't get a clean contiguous block.

## `llama.cpp` Gemma 4 E4B Q4_K_M, `-ngl 28` config (recommended)

| Component | Memory |
|---|---|
| GGUF on disk | 5.3 GB |
| Resident memory during inference (total RSS) | ~3 GB |
| CUDA0 buffer (28 layers) | ~929 MB |
| CPU side (remaining ~7 layers + KV cache + compute scratch) | ~2 GB |
| Compute buffer on CUDA | ~130 MB |

Per-call (single sequential call, batch 128, ctx 1024):

- Prompt eval: 48 tok/s
- Generation: 5.9 tok/s
- Latency for ~50 input + ~50 output tokens: ~10 s

## ASR (post-Phase-0 baseline)

CUDA-resident:

| Model | CUDA allocated | CUDA reserved | Load time | WER (Common Voice 17) |
|---|---|---|---|---|
| `nvidia/stt_fa_fastconformer_hybrid_large` | 458 MB | 480 MB | 9.8 s | 15.5% |
| `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` | ~440 MB | ~470 MB | ~9 s | 13.6% |
| `nvidia/parakeet-tdt-0.6b-v3` | ~750 MB | ~800 MB | ~12 s | 11.4% (uk) / 12.9% (en) / 17.8% (ru) |

CPU-resident (CTranslate2):

| Model | RSS | Notes |
|---|---|---|
| `CoRal-project/roest-v3-whisper-1.5b` (CT2 int8) | ~1.5 GB | Danish; 9.47% WER on CoRal-v3 |
| `selimc/whisper-large-v3-tr-turbo` (CT2 int8) | ~1.6 GB | Turkish; 28.5% WER on CV17 |

## What can coexist?

Without the PyTorch-NVML conflict (see
[pytorch-nvml-conflict.md](pytorch-nvml-conflict.md)) the budget would be:

- Gemma `-ngl 28` (~929 MB CUDA)
- + PyTorch ASR on CUDA (~500 MB)
- + ~2 GB CPU-side overhead
- = ~3.5 GB total

That fits comfortably in 7 GB available unified RAM. **Memory is not the
bottleneck on this hardware** — the bottleneck is the allocator-interaction
bug. We could've shipped on much smaller hardware if not for that.

CPU-side ASR (Røst-CT2 for staff-side, in our case) coexists fine with
`llama.cpp` `-ngl 28`:

- Røst-CT2 on CPU: ~1.5 GB RSS
- Gemma `-ngl 28`: ~3 GB total
- System: ~320 MB
- = ~5 GB total, ~2.5 GB headroom for KV cache growth, GGUF page cache,
  user processes

So if you can architect your patient-side ASR as CTranslate2-based (e.g.
faster-whisper or a CT2-converted custom model), you have a fully working
end-to-end pipeline on the dev-build with current tools. The NVML conflict
only bites when both sides need PyTorch CUDA.

## Fragmentation reality check

Even after a clean boot with `cma=512M`:

```
$ tegrastats --interval 500 | head -1
RAM 320/7608MB (lfb 6x4MB) ...
```

`lfb 6x4MB` = 6 free blocks of 4 MB each in the general kernel allocator,
i.e. **24 MB of contiguous space in 4 MB chunks**. That's tiny. Anything
contiguity-sensitive past 4 MB has to go through CMA (now 512 MB) or
through `cudaMallocManaged` (with its own constraints on Tegra).

The 4 MB block size is structural — it's the page-block-order setting in the
kernel. Compaction (`vm.compact_memory=1`) doesn't produce larger free
blocks on Tegra in our experience. This is why bumping CMA is the right
fix rather than trying to defragment.
