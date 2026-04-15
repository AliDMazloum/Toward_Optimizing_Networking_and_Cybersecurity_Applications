# Toward Optimizing Networking and Cybersecurity Applications Using Domain-Specific Accelerators for Dynamic Programming

This repository contains the CUDA source code accompanying the manuscript:

> **Toward Optimizing Networking and Cybersecurity Applications Using Domain-Specific Accelerators for Dynamic Programming** (under review, IEEE TNSM).

The scripts implement GPU-accelerated versions of two dynamic-programming (DP) workloads that are core to modern networking and cybersecurity stacks:

1. **Smith–Waterman** – used for Deep Packet Inspection (DPI) signature matching.
2. **Floyd–Warshall** – used for all-pairs shortest-path routing.

Each DP algorithm is provided in multiple GPU-optimized variants to study the design trade-offs discussed in the paper (memory utilization vs. occupancy, regex support, and energy measurement).

---

## Repository contents

| File | Algorithm | Focus |
|------|-----------|-------|
| [dpi_memory_focused.cu](dpi_memory_focused.cu) | Smith–Waterman DPI | Memory-bandwidth optimized kernel (large payload / signature set, instrumented with NVML for power/energy logging). |
| [dpi_memory_focused_energy.cu](dpi_memory_focused_energy.cu) | Smith–Waterman DPI | Memory-focused kernel with a dedicated NVML power-polling thread used for the energy-consumption measurements reported in the paper. |
| [dpi_occupancy_focused.cu](dpi_occupancy_focused.cu) | Smith–Waterman DPI | Occupancy-optimized kernel (smaller block size, higher active warps per SM). |
| [dpi_occupancy_focused_variant.cu](dpi_occupancy_focused_variant.cu) | Smith–Waterman DPI | Alternate occupancy-focused configuration used for ablation runs. |
| [dpi_regex_matching.cu](dpi_regex_matching.cu) | Smith–Waterman DPI | DPI variant with regular-expression (character-class / metacharacter) support in the signature set. |
| [floyd_warshall_routing.cu](floyd_warshall_routing.cu) | Floyd–Warshall | GPU all-pairs shortest-path implementation used for the routing case study. |

---

## Requirements

- **NVIDIA GPU** with CUDA Compute Capability ≥ 7.0 (the kernels use `__vibmin_s32` and other intrinsics available on Volta and newer).
- **CUDA Toolkit** 11.0 or newer (`nvcc`).
- **NVML** (ships with the NVIDIA driver) – required by the memory-focused and energy-measurement variants as well as the Floyd–Warshall script.
- **POSIX threads** (`pthread`) – used by the NVML power-polling thread.
- A Linux environment is recommended. On Windows, use WSL2 with the CUDA-on-WSL driver, or MSVC + CUDA with the equivalent compiler flags.

---

## Building

Each `.cu` file is self-contained and can be compiled directly with `nvcc`.

Generic build (no NVML, no pthreads):

```bash
nvcc -O3 -arch=sm_90 dpi_occupancy_focused.cu -o dpi_occupancy_focused
nvcc -O3 -arch=sm_90 dpi_occupancy_focused_variant.cu -o dpi_occupancy_focused_variant
nvcc -O3 -arch=sm_90 dpi_regex_matching.cu -o dpi_regex_matching
```

Builds that link against NVML and pthreads:

```bash
nvcc -O3 -arch=sm_90 dpi_memory_focused.cu        -lnvidia-ml -lpthread -o dpi_memory_focused
nvcc -O3 -arch=sm_90 dpi_memory_focused_energy.cu -lnvidia-ml -lpthread -o dpi_memory_focused_energy
nvcc -O3 -arch=sm_90 floyd_warshall_routing.cu    -lnvidia-ml           -o floyd_warshall_routing
```

The results in the paper were produced on an **NVIDIA H100** (`-arch=sm_90`). Replace the flag with the architecture of your GPU if needed (e.g. `sm_70` for V100, `sm_80` for A100, `sm_86` for RTX 30xx, `sm_89` for RTX 40xx).

On systems where `libnvidia-ml.so` is not on the default library path, add `-L/usr/lib/x86_64-linux-gnu/` (Linux) or point `LIBRARY_PATH` at the directory shipped with your driver.

---

## Running

All binaries are self-contained: input payloads, signature databases, and graphs are generated randomly inside `main()` using the compile-time constants defined at the top of each file. To sweep a different problem size, edit the macros and recompile.

### DPI (Smith–Waterman) variants

Relevant compile-time parameters (top of each DPI file):

| Macro | Meaning |
|-------|---------|
| `PayloadSize` | Length of each packet payload (bytes). |
| `NumberOfSignatures` | Total number of signatures in the database. |
| `MaxSignatureLength` | Maximum signature length (bytes). |
| `MatchingIndex` | Index of the signature that is forced to match (used to verify correctness). |
| `midPoint` | Number of signatures processed per kernel launch. |
| `blockSize` / `gridSize` | CUDA launch configuration. |

Run, for example:

```bash
./dpi_memory_focused
./dpi_occupancy_focused
./dpi_regex_matching
```

Each program prints the measured kernel time and, for NVML-enabled variants, the sampled power / energy consumption during the kernel window.

### Floyd–Warshall

Edit `Ver` (graph vertex count) at the top of [floyd_warshall_routing.cu](floyd_warshall_routing.cu) and rebuild. Then:

```bash
./floyd_warshall_routing
```

The program prints the kernel execution time for computing the all-pairs shortest paths.

---

## Reproducing the paper results

The configurations used in the paper correspond to the constants already set in each file (e.g. `PayloadSize = 512`, `NumberOfSignatures = 20,000,000` for the large-scale DPI runs, `Ver = 12000` for Floyd–Warshall). To reproduce a specific figure or table, adjust these constants to the value reported in the paper, recompile, and run.

For power/energy figures, use `dpi_memory_focused_energy.cu`; it spawns an NVML polling thread that logs instantaneous GPU power during the kernel window and reports the integrated energy.

---

## Citation

A citation entry will be added here once the paper is accepted for publication.

---

## License

Released for academic and research use. Please contact the authors for other uses.
