# Toward Optimizing Networking and Cybersecurity Applications Using Domain-Specific Accelerators for Dynamic Programming

This repository contains the CUDA source code accompanying the manuscript:

> **Toward Optimizing Networking and Cybersecurity Applications Using Domain-Specific Accelerators for Dynamic Programming** (under review, IEEE Access).

The scripts implement GPU-accelerated versions of two dynamic-programming (DP) workloads that are core to modern networking and cybersecurity stacks:

1. **Smith–Waterman**, used for Deep Packet Inspection (DPI) signature matching.
2. **Floyd–Warshall**, used for all-pairs shortest-path routing.

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

- **NVIDIA GPU with compute capability 9.0** (Hopper, for example the H100) to execute the DPX
  instructions in hardware. The kernels call two DPX intrinsics: `__vimax3_s16x2_relu` in the DPI
  kernels and `__vibmin_s32` in the Floyd–Warshall kernel. DPX was introduced with the NVIDIA
  Hopper architecture, so a pre-Hopper GPU does not run these operations on DPX hardware.
- **CUDA Toolkit 12.0 or newer** (`nvcc`). The DPX math APIs used here are exposed by CUDA 12.
- **NVML** (ships with the NVIDIA driver), required by the two memory-focused DPI variants.
- **POSIX threads** (`pthread`), used by the NVML power-polling thread.
- A Linux environment. The energy-measurement path uses `clock_gettime(CLOCK_MONOTONIC)`,
  `nanosleep` and pthreads, so it does not build unmodified on Windows; use WSL2 with the
  CUDA-on-WSL driver if a Linux host is not available.

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
```

Floyd–Warshall needs neither library:

```bash
nvcc -O3 -arch=sm_90 floyd_warshall_routing.cu -o floyd_warshall_routing
```

The results in the paper were produced on an **NVIDIA H100** (`-arch=sm_90`). Replace the flag with the architecture of your GPU if needed (e.g. `sm_70` for V100, `sm_80` for A100, `sm_86` for RTX 30xx, `sm_89` for RTX 40xx).

### Toolchain used for the reported results

The evaluation ran on a cluster that provides its toolchain through environment modules. The
measurements in the paper were taken after loading:

```bash
module unload gcc
module load gcc/12.2.0
module load cuda12.4/
```

so `nvcc` came from **CUDA Toolkit 12.4** and the host compiler was **GCC 12.2.0**. No flags beyond
those in the build commands above were used: `-O3` for optimization, `-arch=sm_90` to target Hopper,
and `-lnvidia-ml -lpthread` where NVML is linked.

On systems where `libnvidia-ml.so` is not on the default library path, add `-L/usr/lib/x86_64-linux-gnu/` (Linux) or point `LIBRARY_PATH` at the directory shipped with your driver.

---

## Running

All binaries are self-contained: input payloads, signature databases, and graphs are generated
inside `main()` from the compile-time constants at the top of each file, so no external dataset is
needed. The DPI programs seed the generator with `srand(time(NULL))`, which means the signature
set differs between runs; the signature at `MatchingIndex` is planted so that a match always
exists. Floyd–Warshall builds a deterministic graph and takes no seed. To sweep a different
problem size, edit the constants and recompile, except in the regex variant, which also accepts
them as command-line options.

### DPI (Smith–Waterman) variants

Relevant compile-time parameters (top of each DPI file):

| Macro | Meaning |
|-------|---------|
| `PayloadSize` | Length of each packet payload (bytes). |
| `NumberOfSignatures` | Total number of signatures in the database. |
| `MaxSignatureLength` | Maximum signature length (bytes). |
| `MatchingIndex` | Index of the signature that is forced to match (used to verify correctness). |
| `midPoint` | Half of the signature set, and the number of threads launched. Each thread scores signature `t` in the low halfword and signature `t + midPoint` in the high halfword of one 32-bit word, which is what `__vimax3_s16x2_relu` operates on. |
| `blockSize` / `gridSize` | CUDA launch configuration. |

`dpi_memory_focused`, `dpi_memory_focused_energy`, `dpi_occupancy_focused` and
`dpi_occupancy_focused_variant` take no command-line options:

```bash
./dpi_memory_focused
./dpi_occupancy_focused
```

`dpi_regex_matching` overrides its constants from the command line, and prints the configuration
it used on the first line of output:

```bash
./dpi_regex_matching --p_size 512 --s_count 10000 --s_len 16 --m_idx 1356                      --block 32 1 1 --grid 313 1 1 --verbose
```

| Option | Meaning |
|--------|---------|
| `--p_size <int>` | Payload size in bytes. |
| `--s_count <int>` | Number of signatures. |
| `--s_len <int>` | Maximum signature length in bytes. |
| `--m_idx <int>` | Index of the planted matching signature, which must be below `--s_count`. |
| `--block <x> <y> <z>` | Block dimensions. |
| `--grid <x> <y> <z>` | Grid dimensions. If the grid is left at one block, it is set to `ceil((s_count / 2) / blockSize)`. |
| `--verbose` | Print processing time and other execution details. |
| `--help` | Print the option list. |

Each program prints the measured kernel time, and the index and score of the signature the kernel
reported as matching. The two memory-focused variants also print the sampled power and the energy
integrated over the kernel window.

### Floyd–Warshall

Edit `Ver` (graph vertex count) at the top of [floyd_warshall_routing.cu](floyd_warshall_routing.cu) and rebuild. Then:

```bash
./floyd_warshall_routing
```

The program prints the total execution time for computing the all-pairs shortest paths. It builds a
directed chain graph, in which vertex `i` has a single outgoing edge to vertex `i + 1` of weight 1,
so the correct distance matrix is known in closed form. After the GPU run it asserts that every
entry equals `j - i` for `j >= i` and `INF` otherwise, and aborts if any entry disagrees.

---

## Reproducing the paper results

Every file is committed with the configuration of one point in the paper's sweeps. The table below
lists what is set in this release, so that a reported number can be traced to a build.

| File | Problem size as committed | Launch configuration |
|------|---------------------------|----------------------|
| `dpi_memory_focused.cu` | `PayloadSize 512`, `MaxSignatureLength 16`, `NumberOfSignatures 20000000`, `midPoint 10000000`, `MatchingIndex 99501` | `blockSize 32`, `gridSize = ceil(midPoint / blockSize) = 312500`, launched with `gridSize + 1` blocks |
| `dpi_memory_focused_energy.cu` | same as above | same as above, plus an NVML polling thread |
| `dpi_occupancy_focused.cu` | `PayloadSize 512`, `MaxSignatureLength 16`, `NumberOfSignatures 10000000`, `midPoint 5000000`, `MatchingIndex 99152` | `blockSize 64`, `gridSize = 78125` |
| `dpi_occupancy_focused_variant.cu` | same as above | same as above |
| `dpi_regex_matching.cu` | `PayloadSize 50`, `MaxSignatureLength 20`, `NumberOfSignatures 10000`, `MatchingIndex 1356` | default block `(10, 1, 1)`, grid derived from the signature count; both overridable on the command line |
| `floyd_warshall_routing.cu` | `Ver 12000`, `INF 99999` | `blockSize 256`, `gridSize 2048`, one kernel launch per intermediate vertex |

Scoring parameters are `match = 2`, `mismatch = -1` and `indel = -1` in the four plain DPI variants,
and `match = 6`, `mismatch = -3` and `indel = -2` in the regex variant.

The paper's second DPI configuration, `(1024, 32)`, is obtained by setting `PayloadSize` to 1024 and
`MaxSignatureLength` to 32. Each signature-count point in the sweeps is obtained by setting
`NumberOfSignatures`, with `midPoint` at half that value, and recompiling. For Floyd–Warshall, each
topology size is obtained by setting `Ver`.

### How time and energy are measured

- **DPI processing time** comes from CUDA events (`cudaEventRecord` before and after the kernel
  launch, then `cudaEventElapsedTime`). Signature generation and the host-to-device copies sit
  outside the timed region, so the figure is kernel time alone.
- **Floyd–Warshall processing time** comes from `clock()` around the whole loop of per-vertex kernel
  launches, each followed by `cudaDeviceSynchronize()`. The host-to-device and device-to-host copies
  sit outside the timed region, so the figure covers the kernel launches and their synchronization.
- **Energy** is measured by a separate pthread that calls `nvmlDeviceGetPowerUsage` every 1 ms on
  device 0 and appends to `DPI_Power_Log.csv` with monotonic timestamps. The thread starts 100 ms
  before the kernel and stops 100 ms after it, and the program then integrates the samples that fall
  inside the kernel window. Use `dpi_memory_focused_energy.cu` for the energy figures.

---

## Citation

The revision submitted to IEEE Access cites this repository at tag **`v1.0`**. Use that tag, rather
than the tip of `main`, to obtain the sources exactly as they were evaluated in the paper:

```bash
git clone https://github.com/AliDMazloum/Toward_Optimizing_Networking_and_Cybersecurity_Applications.git
cd Toward_Optimizing_Networking_and_Cybersecurity_Applications
git checkout v1.0
```

A full citation entry will be added here once the paper is accepted for publication.

---

## License

Released for academic and research use. Please contact the authors for other uses.
