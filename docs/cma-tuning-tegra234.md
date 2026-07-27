# CMA tuning on Tegra234 (Orin Nano 8 GB)

> **Correction (2026-06-02), see [forum thread 370049 post #14](https://forums.developer.nvidia.com/t/pytorch-cudacachingallocator-nvml-assertion-when-sharing-cuda-context-with-llama-cpp-on-orin-nano-8-gb-jetpack-6-2-2/370049/14):**
> NVIDIA staff state that **NvMap does not allocate from the CMA pool**. The
> contiguous-allocation failure is NvMap forwarding a contiguous-buffer request
> to the kernel, which then can't provide contiguous memory — *not* the `cma=`
> reserve being drained. That contradicts the "NvMap bottoms out in CMA"
> mechanism I describe just below. The **empirical results in this doc still
> hold** (bumping `cma=` to 512 MB and running `compact_memory` both reliably
> change load success, reproducibly), but the **causal explanation is now
> uncertain** — it may be general buddy-allocator contiguity rather than the CMA
> reserve specifically. Treat the mechanism here as a working hypothesis pending
> NVIDIA's debug-print findings; the tuning steps remain useful regardless.

> **Platform update (2026-07-27).** This page is JetPack 6.2.2 / L4T R36.5.0.
> The board has since gone to JetPack 7.2 (L4T R39.2.0, Ubuntu 24.04, kernel
> 6.8.12-1021-tegra, CUDA 13.2) via an in-place APT `dist-upgrade`, and
> **`cma=512M` did not survive it**: `/proc/cmdline` no longer carries a `cma=`
> parameter and the pool is back at the 256 MB default (`CmaTotal: 262144 kB`,
> measured). I did not inspect the eMMC `extlinux.conf` after the upgrade, so I
> can't say from measurement what dropped the parameter — only that it is gone.
> Two consequences: if you upgrade, expect to redo the edit below; and the
> `extlinux.conf` procedure here has not been re-verified on the 7.2 boot chain,
> so check the paths before trusting the `sed` line. One idle reading on 7.2
> showed `CmaFree: 222628 kB` — 222 MB of the 256 MB free, against the 13 MB
> recorded below on 6.2.2 — but I have not attempted a model load on 7.2, so
> whether the allocation failure still reproduces is untested.

The Contiguous Memory Allocator (CMA) is a kernel-level pool used by NvMap
(NVIDIA's Tegra memory allocator) to satisfy large contiguous physical
allocations. CUDA on Jetson bottoms out in NvMap, which bottoms out in CMA
when the request is contiguity-sensitive. This applies to both `cudaMalloc`
from llama.cpp and PyTorch's caching allocator.

The default CMA region on JetPack 6.2.2 / L4T R36.5.0 on Orin Nano 8 GB is
256 MB. That's too small once you start loading anything Gemma-sized.

## Symptom

Fresh boot, no other GPU work running, plenty of free RAM (~6 GB
`MemAvailable`), but:

```
NvMapMemAllocInternalTagged: 1075072515 error 12
NvMapMemHandleAlloc: error 0
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 929.82 MiB on device 0: cudaMalloc failed: out of memory
```

The kernel can't satisfy a contiguous CUDA allocation. Run `tegrastats` and
you'll see `lfb 11x4MB` or similar: the largest free block in the regular
allocator is only 4 MB, even on a fresh boot. This is how Tegra's memory
layout looks at boot, not something that builds up over time.

## Diagnosis

`/proc/meminfo` shows:

```
MemTotal:        7790360 kB
MemFree:         2185504 kB
MemAvailable:    6137448 kB
CmaTotal:         262144 kB     # 256 MB — too small for Gemma weight buffer
CmaFree:           13640 kB     # only 13 MB free
```

CMA has plenty allocated *internally* by some Tegra subsystem, leaving 13 MB
free out of 256 MB. NvMap can't get a contiguous block large enough.

## Fix: bump CMA via kernel cmdline

`cma=512M` is what I settled on:

- `cma=1G` fails at boot with `cma: Failed to reserve 1024 MiB`. Tegra234 has
  fixed hardware carveouts (NvMap pre-reserve, GPU firmware regions, secure
  carveouts) at specific physical addresses, which fragments the available
  contiguous space at boot time. The kernel can't find a single 1 GB block to
  reserve for CMA. (Verify in `dmesg | grep -i cma`.)
- `cma=512M` works. Gives `CmaTotal: 524288 kB, CmaFree: 482224 kB`, which is
  about 36× the default free contiguous space.
- `cma=768M` is untested by me. May work; please file a PR if you check.

## The extlinux.conf trap

On an NVMe-booted Orin Nano there are two `extlinux.conf` files, and the one
you want to edit is the one you'd probably never guess.

```
/                  → mounted from /dev/nvme0n1p1   (rootfs on NVMe)
/boot/efi          → mounted from /dev/mmcblk0p10  (UEFI System Partition on eMMC)
/dev/mmcblk0p1     → ext4, NOT mounted by default  (eMMC rootfs copy)
```

L4TLauncher (the bootloader handler) reads `extlinux.conf` from **eMMC
`/dev/mmcblk0p1`**, not from the NVMe rootfs. If you edit
`/boot/extlinux/extlinux.conf` directly (which is on the running NVMe
rootfs), your changes will appear to be saved but the next reboot ignores
them entirely.

I spent an hour debugging "I edited extlinux.conf, rebooted, /proc/cmdline
unchanged" before noticing this. The eMMC copy is the canonical one.

### Edit procedure

```bash
# 1. Mount the eMMC rootfs
sudo mkdir -p /mnt/emmc
sudo mount /dev/mmcblk0p1 /mnt/emmc

# 2. Back up first
sudo cp /mnt/emmc/boot/extlinux/extlinux.conf \
        /mnt/emmc/boot/extlinux/extlinux.conf.bak-$(date +%Y%m%d)

# 3. Append cma=512M to the APPEND line (the live primary kernel section)
sudo sed -i 's/console=tty0$/console=tty0 cma=512M/' \
        /mnt/emmc/boot/extlinux/extlinux.conf

# 4. Verify
grep -E 'APPEND' /mnt/emmc/boot/extlinux/extlinux.conf
# Expected: APPEND ${cbootargs} ... console=tty0 cma=512M

# 5. Unmount and reboot
sudo umount /mnt/emmc
sudo reboot
```

### Verify after reboot

```bash
cat /proc/cmdline
# Should contain: ... console=tty0 cma=512M bl_prof_*

grep -E "CmaTotal|CmaFree" /proc/meminfo
# CmaTotal: 524288 kB
# CmaFree:  482224 kB  (varies a bit)

sudo dmesg | grep -i cma
# Should show: "cma: Reserved 512 MiB at 0x..."
```

If `CmaTotal: 0` after reboot, the kernel rejected your size (Tegra carveout
conflict). Try a smaller value (e.g. `cma=384M`).

### Rollback

```bash
sudo mount /dev/mmcblk0p1 /mnt/emmc
sudo cp /mnt/emmc/boot/extlinux/extlinux.conf.bak-* \
        /mnt/emmc/boot/extlinux/extlinux.conf
sudo umount /mnt/emmc
sudo reboot
```

## What CMA tuning does and doesn't fix

What it fixes:
- `llama.cpp` dev-build at moderate `-ngl` (e.g. 28) loading Gemma 4 E4B
  Q4_K_M reliably. Without the bump it fails non-deterministically on the
  ~929 MB contiguous CUDA weight buffer.
- General contiguity headroom for any large CUDA workload.

What it doesn't fix:
- `-ngl 999` (full GPU offload) for Gemma 4 E4B Q4_K_M. The ~3 GB contiguous
  weight buffer still fails; even 512 MB CMA isn't enough.
- Concurrent `llama.cpp` + PyTorch-based ASR on CUDA. That's a different bug
  (see [pytorch-nvml-conflict.md](pytorch-nvml-conflict.md)), not a
  memory-size issue.

## Trade-offs

You're stealing 512 MB from general RAM for the CMA pool. On 8 GB Orin Nano:

| | Before | After |
|---|---|---|
| Usable general RAM | ~7.5 GB | ~7.0 GB |
| CMA pool | 256 MB | 512 MB |
| `lfb` (largest free contiguous in general) | ~4 MB on fresh boot | similar |
| Big contiguous CUDA allocs work | flaky | reliable up to ~929 MB |

If you're already memory-pressured for non-GPU work, 512 MB might be too much
to give up. Adjust to taste.

## Also useful: boot to text mode

Switching the default systemd target to `multi-user.target` (i.e. no GNOME
desktop) drops the system RAM baseline from ~1.4 GB to ~320 MB. Permanent
change, recommended for production-style deployments anyway:

```bash
sudo systemctl set-default multi-user.target
sudo reboot
```

Revert to desktop: `sudo systemctl set-default graphical.target`.

This isn't strictly necessary if you've bumped CMA, but it gives you ~1 GB of
extra general RAM which can absorb fragmentation pressure during model loads.
