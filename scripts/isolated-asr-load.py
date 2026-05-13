#!/usr/bin/env python3
"""Isolated test: can a PyTorch-based NeMo ASR model load on CUDA?

Tests the model load on CPU first as a sanity check, then on CUDA. Run with
NO llama-server (or other CUDA workload) running. That's the whole point
of the isolation.

Compare with concurrent-loadtest.sh, which runs the same load alongside
llama-server and reproduces the NVML assertion.

Usage:
    ./isolated-asr-load.py [model-id]

Default model is nvidia/stt_fa_fastconformer_hybrid_large (Persian).
Substitute any PyTorch-CUDA NeMo model.
"""
import sys
import time
import traceback
import gc


def mem_mb() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "nvidia/stt_fa_fastconformer_hybrid_large"
    print(f"[{mem_mb()} MB] loading NeMo")
    import nemo.collections.asr as nemo_asr  # noqa: E402

    # CPU first
    print(f"\n[{mem_mb()} MB] --- attempt CPU load of {model_id} ---")
    t0 = time.time()
    try:
        m_cpu = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_id, map_location="cpu",
        )
        print(f"[{mem_mb()} MB] CPU load OK in {time.time() - t0:.1f}s")
        del m_cpu
        gc.collect()
        print(f"[{mem_mb()} MB] CPU model deleted")
    except Exception:
        print(f"[{mem_mb()} MB] CPU load FAILED:")
        traceback.print_exc()
        return 2

    # CUDA
    print(f"\n[{mem_mb()} MB] --- attempt CUDA load of {model_id} ---")
    t0 = time.time()
    try:
        m_cuda = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_id, map_location="cuda",
        )
        import torch
        if torch.cuda.is_available():
            alloc_mb = torch.cuda.memory_allocated() // (1024 * 1024)
            reserved_mb = torch.cuda.memory_reserved() // (1024 * 1024)
            print(
                f"[{mem_mb()} MB] CUDA load OK in {time.time() - t0:.1f}s "
                f"(torch.cuda: allocated={alloc_mb} MB, reserved={reserved_mb} MB)"
            )
    except Exception:
        print(f"[{mem_mb()} MB] CUDA LOAD FAILED:")
        traceback.print_exc()
        print(
            "\nIf the failure says 'NVML_SUCCESS == r INTERNAL ASSERT FAILED' you've\n"
            "hit the PyTorch/llama.cpp CUDA-context conflict. Make sure NO\n"
            "llama-server (or other CUDA process) is running. If still failing,\n"
            "see docs/pytorch-nvml-conflict.md for hypotheses.\n"
        )
        return 3

    print("\nDONE: model loaded successfully on CUDA.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
