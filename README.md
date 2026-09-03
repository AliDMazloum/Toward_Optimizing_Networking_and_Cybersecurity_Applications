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
| [dpi_memory_focused.cu](App2/dpi_memory_focused.cu) | Smith–Waterman DPI | Memory-bandwidth optimized kernel (large payload / signature set, instrumented with NVML for power/energy logging). |
| [dpi_memory_focused_energy.cu](App2/dpi_memory_focused_energy.cu) | Smith–Waterman DPI | Memory-focused kernel with a dedicated NVML power-polling thread used for the energy-consumption measurements reported in the paper. |
| [dpi_occupancy_focused.cu](App2/dpi_occupancy_focused.cu) | Smith–Waterman DPI | Occupancy-optimized kernel (smaller block size, higher active warps per SM). |
| [dpi_occupancy_focused_variant.cu](App2/dpi_occupancy_focused_variant.cu) | Smith–Waterman DPI | Alternate occupancy-focused configuration used for ablation runs. |
| [dpi_regex_matching.cu](App2/dpi_regex_matching.cu) | Smith–Waterman DPI | DPI variant with regular-expression (character-class / metacharacter) support in the signature set. |
| [floyd_warshall_routing.cu](App1/floyd_warshall_routing.cu) | Floyd–Warshall | GPU all-pairs shortest path for the routing case study. Carries both thread-to-data mappings and both DPX states, selected by command-line flags, with repeated trials and optional NVML energy sampling. |

---

## Requirements

- **NVIDIA GPU with compute capability 9.0** (Hopper, for example the H100 or H200) to execute the
  DPX instructions in hardware. The kernels call two DPX intrinsics: `__vimax3_s16x2_relu` in the
  DPI kernels and `__viaddmin_s32` in the Floyd–Warshall kernel. DPX was introduced with the NVIDIA
  Hopper architecture, so a pre-Hopper GPU does not run these operations on DPX hardware. The
  routing program can also be built and run with `--dpx off`, which computes the same values with
  an ordinary add and minimum on the same GPU.
- **CUDA Toolkit 12.0 or newer** (`nvcc`). The DPX math APIs used here are exposed by CUDA 12.
- **NVML** (ships with the NVIDIA driver), required by the two memory-focused DPI
  variants and by the routing program.
- **POSIX threads** (`pthread`), used by the NVML power-polling thread.
- A Linux environment. The energy-measurement path uses `clock_gettime(CLOCK_MONOTONIC)`,
  `nanosleep` and pthreads, so it does not build unmodified on Windows; use WSL2 with the
  CUDA-on-WSL driver if a Linux host is not available.

---

## Building

The repository is split by application: `App1` holds the network resilience system and `App2` holds
the deep packet inspection system. A `Makefile` at the top level builds both, and places each binary
next to its source:

```bash
make            # build everything
make app1       # the network resilience system only
make app2       # the deep packet inspection system only
make check      # build App1 and run its four kernel variants
make clean
```

`ARCH` selects the target architecture and defaults to `sm_90`, so `make ARCH=sm_80` builds for an
A100. Each file is also self-contained and can be compiled directly:

```bash
nvcc -O3 -arch=sm_90 App2/dpi_occupancy_focused.cu         -o App2/dpi_occupancy_focused
nvcc -O3 -arch=sm_90 App2/dpi_occupancy_focused_variant.cu -o App2/dpi_occupancy_focused_variant
nvcc -O3 -arch=sm_90 App2/dpi_regex_matching.cu            -o App2/dpi_regex_matching
nvcc -O3 -arch=sm_90 App2/dpi_memory_focused.cu        -lnvidia-ml -lpthread -o App2/dpi_memory_focused
nvcc -O3 -arch=sm_90 App2/dpi_memory_focused_energy.cu -lnvidia-ml -lpthread -o App2/dpi_memory_focused_energy
nvcc -O3 -arch=sm_90 App1/floyd_warshall_routing.cu    -lnvidia-ml -lpthread -o App1/floyd_warshall_routing
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
./App2/dpi_memory_focused
./App2/dpi_occupancy_focused
```

`dpi_regex_matching` overrides its constants from the command line, and prints the configuration
it used on the first line of output:

```bash
./App2/dpi_regex_matching --p_size 512 --s_count 10000 --s_len 16 --m_idx 1356 --block 32 1 1 --grid 313 1 1 --verbose
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

The routing program takes every parameter on the command line, so a sweep needs no edits and no
rebuilds:

```bash
./App1/floyd_warshall_routing --nodes 24000 --layout coalesced --dpx on --trials 10 --warmup 1 --energy --csv results.csv
```

| Option | Meaning |
|--------|---------|
| `--nodes <int>` | Number of vertices. Default 12000. |
| `--layout <name>` | `coalesced`, in which a block strides over rows and a thread over columns, or `strided`, the non-coalesced mapping in which consecutive threads address entries `nodes` apart. Default `coalesced`. |
| `--dpx <state>` | `on` uses `__viaddmin_s32`; `off` computes the same value with an add and a minimum, which is the DPX-off arm on the same GPU. Default `on`. |
| `--trials <int>` | Measured repetitions. Default 1. |
| `--warmup <int>` | Unmeasured repetitions run first. Default 1. |
| `--cpu` | Run the serial host reference instead of the GPU. |
| `--energy` | Sample GPU power with NVML and report energy per trial. |
| `--poll-ms <int>` | NVML sampling interval in milliseconds. Default 1. |
| `--device <int>` | CUDA and NVML device index. Default 0. |
| `--csv <path>` | Append one row per trial, with every setting, to this file. |
| `--power-csv <path>` | Write every power sample to this file. |
| `--no-verify` | Skip the correctness check. |

The block size is fixed at 256 threads, which performed best in our measurements. The block count is
not fixed and not hand-picked: it is the smaller of the work available and the number of blocks the
device can hold resident for the selected kernel, obtained from the occupancy API and the SM count.
Every run prints the derived value along with all other settings, so the output documents the
configuration that produced it.

The topology is a directed chain in which vertex `i` has one outgoing edge to vertex `i + 1` of
weight 1, so the correct distance matrix is known in closed form. Unless `--no-verify` is given,
every trial is checked entry by entry against `j - i` for `j >= i` and `INF` otherwise, and the
number of mismatching entries is reported.

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
| `floyd_warshall_routing.cu` | Set by `--nodes`, default 12000. `INF 99999` | 256 threads per block, block count derived per run, one kernel launch per intermediate vertex |

Scoring parameters are `match = 2`, `mismatch = -1` and `indel = -1` in the four plain DPI variants,
and `match = 6`, `mismatch = -3` and `indel = -2` in the regex variant.

The paper's second DPI configuration, `(1024, 32)`, is obtained by setting `PayloadSize` to 1024 and
`MaxSignatureLength` to 32. Each signature-count point in the sweeps is obtained by setting
`NumberOfSignatures`, with `midPoint` at half that value, and recompiling. For Floyd–Warshall,
nothing needs recompiling: each topology size, mapping and DPX state is a command-line flag.

### How time and energy are measured

- **DPI processing time** comes from CUDA events (`cudaEventRecord` before and after the kernel
  launch, then `cudaEventElapsedTime`). Signature generation and the host-to-device copies sit
  outside the timed region, so the figure is kernel time alone.
- **Floyd–Warshall processing time** is reported twice per trial, so the measurement window is never
  ambiguous. The *kernel* time comes from CUDA events around the per-vertex launches alone. The
  *end to end* time comes from a monotonic host clock and additionally covers the host-to-device copy
  of the distance matrix and the copy of the result back. Rebuilding the input matrix between trials
  falls outside both windows.
- **Energy** is measured by a separate pthread that calls `nvmlDeviceGetPowerUsage` every 1 ms and
  keeps its samples in memory, so no file writing happens inside a measured window. Energy for a
  trial is the trapezoidal integral of the samples whose timestamps fall inside that trial's end to
  end window; a window holding fewer than two samples is reported as missing rather than estimated.
  Use `dpi_memory_focused_energy.cu` for the DPI energy figures and `--energy` for the routing ones.

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
