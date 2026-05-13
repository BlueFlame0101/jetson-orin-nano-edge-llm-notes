# PyTorch `CUDACachingAllocator` NVML assertion when sharing CUDA with `llama.cpp`

**Status:** unresolved. Reproducer below. I list workaround ideas at the
bottom, none of them fully tested. PRs and pointers welcome.

## Symptom

When running `llama.cpp` server (CUDA-enabled, dev-build) and then trying
to load a PyTorch-based ASR model on the same GPU device, PyTorch fails to
move the model to CUDA with:

```
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at
"/opt/pytorch/pytorch/c10/cuda/CUDACachingAllocator.cpp":838,
please report a bug to PyTorch.
```

Full traceback (truncated):

```
File "torch/nn/modules/module.py", line 1318, in to
    return self._apply(convert)
File "torch/nn/modules/module.py", line 897, in _apply
    module._apply(fn)
File "torch/nn/modules/module.py", line 924, in _apply
    param_applied = fn(param)
File "torch/nn/modules/module.py", line 1304, in convert
    return t.to(...)
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at
"/opt/pytorch/pytorch/c10/cuda/CUDACachingAllocator.cpp":838
```

That line is where PyTorch's `CUDACachingAllocator` queries NVML for
device properties (UUID, memory-tracking info) during its first
allocation. The query returns a non-`NVML_SUCCESS` code, the assertion
fires, and PyTorch can't initialise its CUDA caching allocator on that
process.

## Reproducer

The trigger is `llama.cpp` having already initialised a CUDA context on the
same device. I verified the pieces in isolation:

1. `fa-fastconformer` (or any other NeMo PyTorch-CUDA model) loads on
   CUDA alone in 9.8 s, 458 MB allocated, 480 MB reserved. No assertion.
2. `llama.cpp` server `-ngl 28` boots alone, runs translation calls
   deterministically. No issue.
3. Start `llama.cpp` first, then try `model.to("cuda")` from PyTorch: NVML
   assertion fires.
4. Start PyTorch first (CUDA initialised, model on CUDA), then start
   `llama.cpp`: `cudaMalloc failed: out of memory` on the 929 MB weight
   buffer. That's a different failure mode (NvMap fragmentation, since
   PyTorch's allocator has subdivided the pool), but still a failure.

Either order is broken. There's no working order with these two stacks on
this device. The exact reproducer:

```bash
# Terminal 1: start llama.cpp server
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 LLAMA_ARG_FIT=off \
  llama-server -m /path/to/gemma.gguf --host 127.0.0.1 --port 8080 \
    --ctx-size 1024 --parallel 1 --batch-size 128 --fit off -ngl 28 \
    --threads 4 --no-warmup --reasoning off --reasoning-budget 0

# Terminal 2: try loading any NeMo PyTorch-CUDA model
python3 -c "
import nemo.collections.asr as nemo_asr
m = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
    'nvidia/stt_fa_fastconformer_hybrid_large'
)
m.to('cuda')  # <-- explodes here
"
```

Or use the bundled
[`scripts/concurrent-loadtest.sh`](../scripts/concurrent-loadtest.sh).

## What's not affected

CTranslate2-based ASR providers (`faster-whisper`, `Røst-CT2`, `selimc
tr-turbo-CT2`, any custom CT2-converted model). These use CTranslate2's
own CUDA binding, not PyTorch's caching allocator, and so don't hit the
NVML assertion. They coexist fine with `llama.cpp` provided you have
memory headroom (which you do, with `cma=512M`).

This is a useful fallback if you can run with CT2-based ASRs only. The
catch: the major NeMo families (FastConformer, Conformer-CTC, Parakeet-TDT)
are PyTorch-only.

## Hypotheses for the root cause

I don't have this fully nailed down. The best guesses I have:

1. NVML state corruption by `llama.cpp`'s CUDA usage. `llama.cpp` uses
   `cudaMalloc` / `cudaMallocManaged` directly. PyTorch's allocator at
   init queries NVML for device properties (UUID, memory accounting). On
   Tegra, NVML is provided through JetPack-specific libraries, and there's
   known interaction between the NVML query and active CUDA contexts that
   doesn't exist on x86 with discrete GPUs.

2. `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` specifically. When `llama.cpp`
   uses `cudaMallocManaged`, it may pin some unified-memory state that
   NVML's later query reads inconsistently. I didn't test this directly:
   turning off unified memory prevents `llama.cpp` from starting at all
   without first fixing the underlying CMA issue, so I couldn't isolate
   this on my hardware in a reasonable timeframe.

3. JetPack 6.2.2's NVML build. PyTorch was built against a specific NVML
   version when NVIDIA released the official Jetson wheel. JetPack 6.2.2
   might ship a slightly newer NVML that PyTorch wasn't expecting, leading
   to the unexpected return code. JetPack 7.2, when it lands for Orin
   Nano, may fix this.

## Workarounds (none fully tested)

In rough order from least to most invasive:

1. `PYTORCH_NO_CUDA_NVML=1` (or a similar env var). I tried this and it
   didn't change behaviour; PyTorch still asserts. There may be a build
   flag or different env name I missed. If you find one that works,
   please PR.

2. Use CTranslate2 alternatives. As noted above, the CT2 family doesn't
   hit this. If you can ship with `faster-whisper` for staff-side or
   Røst-CT2 for Danish-only deployments and skip the PyTorch ASR stack
   entirely, this dodges the problem.

3. Run patient-side ASR on CPU instead of CUDA. PyTorch's CPU allocator
   doesn't have this issue. Slower (roughly 3-5× depending on model), but
   functional. For the smaller FastConformer models this can be acceptable.

4. Rebuild PyTorch from source with the assertion patched. Modifying
   `CUDACachingAllocator.cpp:838` to handle the NVML failure gracefully
   (downgrade the assertion to a warning, fall back to a no-tracking mode)
   should let allocation proceed. Untested. Be careful with what you ship:
   you're masking a CUDA-state-detection failure that might have
   downstream memory-accounting implications.

5. Use the production NVIDIA container. `llama.cpp` inside
   `ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-orin` may use a
   different CUDA-init path that avoids the conflict. The container
   needs the driver from JetPack 7.2. JetPack 7.0 and 7.1 are
   Thor-only, and 7.2 still hadn't shipped for Orin Nano when I wrote
   this, so I can't test it. If you've run that container alongside
   PyTorch on the same device, I'd love to know the result.

## Why upstream PyTorch hasn't fixed this

It may not be considered a bug from PyTorch's perspective. On x86 with
discrete GPUs the NVML query is reliable and the assertion is appropriate
defensive code. On Tegra / Jetson, NVML behaves differently and the
assertion is over-strict. A proper fix probably needs a Tegra-aware code
path in PyTorch's allocator. Worth filing an issue on
[pytorch/pytorch](https://github.com/pytorch/pytorch/issues) with this
reproducer if there isn't one already.

## If you've solved this

PR welcome. I'd much rather ship on dev-build `llama.cpp` than wait on
JetPack 7.2 landing for Orin Nano.
