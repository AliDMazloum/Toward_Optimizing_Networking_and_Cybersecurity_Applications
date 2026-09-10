# Toward Optimizing Networking and Cybersecurity Applications Using Domain-Specific Accelerators for Dynamic Programming

This repository contains the CUDA source code accompanying the manuscript:

> **Toward Optimizing Networking and Cybersecurity Applications Using Domain-Specific Accelerators for Dynamic Programming** (under review, IEEE Access).

The scripts implement GPU-accelerated versions of two dynamic-programming (DP) workloads that are core to modern networking and cybersecurity stacks:

1. **Smith–Waterman**, used for Deep Packet Inspection (DPI) signature matching.
2. **Floyd–Warshall**, used for all-pairs shortest-path routing.

Each workload is one program that takes every design choice on the command line, so a point in the
paper's sweeps is a set of flags rather than a rebuild.

---

## Repository contents

| File | Algorithm | What it is |
|------|-----------|------------|
| [floyd_warshall_routing.cu](App1/floyd_warshall_routing.cu) | Floyd–Warshall | GPU all-pairs shortest path for the routing case study. Three memory layouts, both DPX states, two store policies, two topologies, repeated trials, an optional host reference and optional NVML energy sampling, all selected by flag. |
| [smith_waterman_dpi.cu](App2/smith_waterman_dpi.cu) | Smith–Waterman DPI | GPU signature matching for the deep packet inspection case study. Literal and regex scoring, both DPX states, DP rows in registers or in global memory, an optional host reference and optional NVML energy sampling, all selected by flag. |

Several separate DPI programs, each carrying its problem size as compile-time constants, stood here
until the revision. They are kept under `Old_files/App2/` for reference and are not built by the
Makefile. The single program above computes what they computed; where it deliberately differs, and
why, is documented in its source at the point of the difference.

---

## Requirements

- **NVIDIA GPU with compute capability 9.0** (Hopper, for example the H100 or H200) to execute the
  DPX instructions in hardware. The kernels call two DPX intrinsics: `__vimax3_s16x2_relu` in the
  DPI kernels and `__viaddmin_s32` in the Floyd–Warshall kernel. DPX was introduced with the NVIDIA
  Hopper architecture, so a pre-Hopper GPU does not run these operations on DPX hardware. The
  routing program can also be built and run with `--dpx off`, which computes the same values with
  an ordinary add and minimum on the same GPU.
- **CUDA Toolkit 12.0 or newer** (`nvcc`). The DPX math APIs used here are exposed by CUDA 12.
- **NVML** (ships with the NVIDIA driver), linked by both programs and used only when `--energy`
  is given.
- **POSIX threads** (`pthread`), used by the NVML power-polling thread.
- A Linux environment. The energy-measurement path uses `clock_gettime(CLOCK_MONOTONIC)`,
  `nanosleep` and pthreads, so it does not build unmodified on Windows; use WSL2 with the
  CUDA-on-WSL driver if a Linux host is not available.

---

## Building

The repository is split by application: `App1` holds the network resilience system and `App2` holds
the deep packet inspection system. A `Makefile` at the top level builds both, and places each binary
next to its source:

### One target per machine

The two GPUs used for the reported results sit on nodes that share a file system, so a build for one
would overwrite the binary the other is running. The Makefile therefore names every binary after the
machine it was built for, and offers one target per machine that sets the architecture, the name and
the output file together. Nothing has to be remembered at the prompt:

```bash
make h200         # build both applications for the H200
make h200-app1    # the network resilience system only
make h200-check   # run all twenty kernel variants and check each one
make h200-sweep   # run the App1 sweep reported in the paper
make h200-sweep2  # run the App2 sweep reported in the paper
make h200-clean   # remove this machine's binaries and nothing else
```

and the same six with `a100`. `make h200` produces `App1/floyd_warshall_routing-h200`, `make a100`
produces `App1/floyd_warshall_routing-a100`, and both can be built and measured at the same time.

Sweeps write into `Results/`, which is also where the measurements behind the paper live.
`h200-sweep` writes `Results/app1_final_h200.csv` and `a100-sweep` writes
`Results/app1_final_a100.csv`, so the two machines never append to one file. A program also refuses
to append to a file whose header is not the one it writes, which stops a binary built from an older
revision from adding rows that its header no longer describes.
Before running anything, a sweep asks the node which GPU it has
and refuses if the answer does not match the target's name, which catches a job that landed on the
wrong machine before it produces a mislabelled number. `SWEEP_NODES`, `SWEEP_TRIALS`, `SWEEP_WARMUP`,
`SWEEP_FLAGS` and `SWEEP_CSV` override what it runs and where it writes, and the `SWEEP2_`
equivalents do the same for App2. A sweep also refuses to run on a GPU another process is computing
on; see **Reproducing the paper results** for why and for how to name a free card.

### The general targets

The machine targets are shorthand for these, which take the architecture and the tag by hand:

```bash
make ARCHES=90 TAG=h200            # build everything
make ARCHES=90 TAG=h200 app1       # the network resilience system only
make ARCHES=90 TAG=h200 app2       # the deep packet inspection system only
make ARCHES=90 TAG=h200 check      # run all twenty kernel variants and check each one
make ARCHES=90 TAG=h200 sweep      # run the reported App1 sweep
make ARCHES=90 TAG=h200 sweep2     # run the reported App2 sweep
make ARCHES=90 TAG=h200 clean      # remove the binaries for this tag only
make clean-all                     # remove every tag, including other machines'
```

`clean` removes only the binaries for the tag it is given. `clean-all` removes every tag, which on a
shared file system deletes the binaries another machine may be running; use it once before the first
tagged build, and not while a measurement is in progress.

With neither `ARCHES` nor `TAG` the Makefile builds one binary carrying native code for both `sm_80` and `sm_90`,
named `floyd_warshall_routing-sm80-sm90`, which runs on an A100 and on an H100 or H200 without
recompiling. Building for a single architecture and running on the other still works, because the
driver compiles the embedded PTX, but it no longer measures the same machine code. Each file is also
self-contained and can be compiled directly:

```bash
TAG=h200   # or a100, so the two machines do not overwrite each other
nvcc -O3 -arch=sm_90 -Xcompiler -fopenmp,-march=native \
     App1/floyd_warshall_routing.cu -lnvidia-ml -lpthread -o App1/floyd_warshall_routing-$TAG
nvcc -O3 -arch=sm_90 -Xcompiler -fopenmp,-march=native \
     App2/smith_waterman_dpi.cu     -lnvidia-ml -lpthread -o App2/smith_waterman_dpi-$TAG
```

`-Xcompiler -fopenmp,-march=native` applies to the host reference that `--cpu` runs, not to the
kernels. It is what the reported CPU baselines were built with, so leave it in when comparing against
them and drop it for a serial, portable host build. Because `-march=native` targets the machine doing
the compiling, build on the node that runs, or the binary can trap on an instruction the run node
lacks.

The two GPUs behind the reported results are an **NVIDIA H200 NVL** and an **NVIDIA A100-SXM4-40GB**,
built with `-arch=sm_90` and `-arch=sm_80` respectively; the machine targets above set that for you.
For another GPU, replace the flag with its architecture, for example `sm_70` for a V100 or `sm_89` for
an RTX 40xx. Only Hopper and later execute the DPX intrinsics in hardware.

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

Both programs are self-contained: payloads, signature sets and graphs are generated inside `main()`,
so no external dataset is needed. Both take a `--seed`, and generation is a deterministic function of
it, so a run is reproducible from the settings it prints. Every run begins by printing those settings
as comment lines, which means the output documents the configuration that produced it.

### DPI (Smith–Waterman)

```bash
./App2/smith_waterman_dpi-h200 --signatures 10000000 --payload 512 --sig-len 16 \
    --mode literal --rows registers --dpx on --trials 5 --warmup 1 --csv Results/my_run.csv
```

| Option | Meaning |
|--------|---------|
| `--signatures <int>` | Number of signatures, even. Default 20000000. |
| `--payload <int>` | Payload length in bytes. Default 512. |
| `--sig-len <int>` | Signature length, 16 or 32. Both are compile-time bounds on the register-resident rows. |
| `--mode <name>` | `literal` or `regex`. Regex adds `*`, `.` and `~`, which score zero and so preserve the reading of the threshold as a fraction of literal agreement. |
| `--rows <where>` | `registers` or `global`: where the two DP rows live. |
| `--dpx <state>` | `on` uses `__vimax3_s16x2_relu`, packing two signatures into one 32-bit word; `off` computes the same values with ordinary integer operations on the same GPU. |
| `--alpha <float>` | Detection threshold as a fraction of the maximum score. Default 0.8. |
| `--block <int>` | Threads per block. Default 32. |
| `--exit <policy>` | `first` stops a thread at its first report, `never` scans everything. |
| `--plant <int>` | Force this signature to match the payload. Default none, so nothing matches, which is the worst case the throughput figures describe. |
| `--alphabet <name>` | `lower26`, `ascii95` or `bytes256`: the bytes the generated data is drawn from. It sets how often unrelated strings agree by chance, so any false positive rate depends on it. Default `lower26`. |
| `--trials`, `--warmup` | Measured and unmeasured repetitions. |
| `--repeat <int>`, `--min-window <sec>` | Repeat the scan inside one timed window, so a run too short for the energy instruments can be lengthened. Every reported figure is still per scan. |
| `--cpu` | Run the host reference instead of the GPU. |
| `--energy`, `--poll-ms`, `--power-csv` | NVML power sampling and its interval and dump file. |
| `--device <int>`, `--seed <int>`, `--csv <path>` | Device, generator seed, and the results file to append to. |
| `--verify <what>` | `report`, `all` or `off`: recheck the reported crossing, or every signature, against the host reference. |

The program prints one row per trial with the kernel and end to end time, and the index, score and
position of any signature it reported. With verification on it also prints a mismatch count against
the host reference, which must be zero.

### Floyd–Warshall

The routing program takes every parameter on the command line, so a sweep needs no edits and no
rebuilds:

```bash
./App1/floyd_warshall_routing-h200 --nodes 24000 --layout coalesced --dpx on --trials 10 --warmup 1 --energy --csv Results/my_run_h200.csv
```

The sweep reported in the paper is `make h200-sweep` or `make a100-sweep`, which fills these flags
in for you and writes one file per machine under `Results/`.

If you call the program directly, give each machine its own `--csv` file. Two runs appending to one
file on a shared file system interleave their rows and can tear a line. Every row records the GPU it
came from in the `gpu` column, so a merge afterwards is safe and an accidental mixture is still
readable.

| Option | Meaning |
|--------|---------|
| `--nodes <int>` | Number of vertices. Default 12000. |
| `--layout <name>` | `coalesced`, in which a block strides over rows and a thread over columns; `strided`, the non-coalesced mapping in which consecutive threads address entries `nodes` apart; or `tiled`, the blocked formulation. Default `coalesced`. |
| `--dpx <state>` | `on` uses a DPX instruction; `off` computes the same value with an ordinary add and minimum, which is the DPX-off arm on the same GPU. Default `on`. |
| `--store <policy>` | `always` writes every cell on every pass. `changed` writes only the cells whose value improves, which removes most of the write traffic and is what the originally published kernel did. With `--dpx on` the two policies use different instructions, `__viaddmin_s32` and `__vibmin_s32` respectively, because only the second returns the comparison alongside the minimum. Default `always`. |
| `--trials <int>` | Measured repetitions. Default 1. |
| `--warmup <int>` | Unmeasured repetitions run first. Default 1. |
| `--cpu` | Run the host reference instead of the GPU. It is the textbook triple loop, spread across OpenMP threads when the build enables OpenMP and serial when it does not. |
| `--energy` | Sample GPU power with NVML and report energy per trial. |
| `--poll-ms <int>` | NVML sampling interval in milliseconds. Default 1. |
| `--device <int>` | CUDA and NVML device index. Default 0. |
| `--csv <path>` | Append one row per trial, with every setting, to this file. A file whose header is not the one this build writes is refused, so rows never land under a header that does not describe them. |
| `--power-csv <path>` | Write every power sample to this file. |
| `--topology <name>` | `chain` or `scale-free`. Default `chain`. |
| `--attach <int>` | Scale-free only: edges each new vertex adds. |
| `--seed <int>` | Scale-free only: generator seed. Default 1. |
| `--block <int>` | Threads per block for the flat layouts. |
| `--sync <state>` | `per-launch` or `none`: whether the host synchronizes after each launch. |
| `--no-verify` | Skip the correctness check. |

The block size is fixed at 256 threads, which performed best in our measurements. The block count is
not fixed and not hand-picked: it is the smaller of the work available and the number of blocks the
device can hold resident for the selected kernel, obtained from the occupancy API and the SM count.
Every run prints the derived value along with all other settings, so the output documents the
configuration that produced it.

Two topologies are available and they are checked differently, which is the point of having both.
`chain` is a directed chain in which vertex `i` has one outgoing edge to vertex `i + 1` of weight 1,
so the correct distance matrix is known in closed form, and unless `--no-verify` is given every trial
is checked entry by entry against `j - i` for `j >= i` and `INF` otherwise. `scale-free` is a
Barabasi-Albert graph grown by preferential attachment, which has no closed-form distance matrix, so
verification there runs the host triple loop once per invocation and compares against that. The
generated graph is the paper's primary evaluation because it is the realistic one; the chain carries
the stronger correctness claim because its answer is known independently of any implementation.

---

## Reproducing the paper results

Nothing needs recompiling to move between points. Every configuration in the paper is a set of flags,
and the Makefile's sweep targets encode the reported protocol so a whole sweep is one command:

```bash
make h200-sweep     # the App1 sweep, every topology size, both DPX states
make h200-sweep2    # the App2 sweep, both configurations, every signature count
```

and the same two with `a100`. What those targets set:

| | App1 | App2 |
|---|---|---|
| sizes | 1000, 2000, 3000, 6000, 12000, 24000 vertices | 10K to 50M signatures |
| configurations | three layouts, two store policies, two topologies | (512, 16) and (1024, 32) |
| DPX | on and off at every point | on and off at every point |
| trials | 5 measured after 1 warm-up | 5 measured after 1 warm-up |

Override any of them with `SWEEP_NODES`, `SWEEP_TRIALS`, `SWEEP_WARMUP`, `SWEEP_FLAGS`, `SWEEP_CSV`
and the `SWEEP2_` equivalents.

**A sweep refuses to run on a GPU another process is computing on.** A timing measurement taken
beside another job reports the sharing rather than the program, and the results file it writes cannot
afterwards be told from a good one, so the check happens before anything is written. On a shared node,
find a free card and name it:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
make a100-sweep2 SWEEP_GPU=3
```

The card is then pinned by its UUID rather than by an index, because CUDA and NVML number devices
differently and an index that names the free card to one can name a busy card to the other.
`ALLOW_BUSY=1` overrides the refusal, for a deliberate measurement of a shared card.

Scoring parameters are `match = 1`, `mismatch = -2` in literal mode and `match = 6`, `mismatch = -3`,
`indel = -2` in regex mode, where the three metacharacters contribute zero.

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
  Both programs take `--energy`, and both resolve the NVML handle by the PCI bus id of the CUDA
  device they ran on rather than by device index, because CUDA and NVML order devices differently and
  an index can therefore name a card that ran nothing.
- **A run shorter than the driver's power refresh interval cannot be measured this way**, whatever the
  sampler does, because the value it reads was computed before the run began. That interval is a
  property of the card and driver rather than a constant, so measure it rather than assuming it. Where
  a scan is too short, `--repeat` or `--min-window` repeats it inside one timed window and every
  reported figure is divided back down to one scan.
- **CPU energy** comes from the powercap RAPL counters, read at the window edges rather than sampled,
  so it carries no such lower limit. The counter wraps, and the reader unwraps it per package.

---

## Citation

The revision submitted to IEEE Access cites this repository at tag **`v2.0`**. Use that tag, rather
than the tip of `main`, to obtain the sources exactly as they were evaluated in the revision:

```bash
git clone https://github.com/AliDMazloum/Toward_Optimizing_Networking_and_Cybersecurity_Applications.git
cd Toward_Optimizing_Networking_and_Cybersecurity_Applications
git checkout v2.0
```

`v1.0` is the earlier tag and it does **not** reproduce the revision. It predates the rewrite of the
DPI system into one program, the tiled layout and the generated topology in the routing system, the
resolution of the NVML handle by PCI bus id, the unwrapping of the RAPL counter, and the correctness
fixes made during the revision. It is kept because it is what the original submission cited.

A full citation entry will be added here once the paper is accepted for publication.

---

## License

Released for academic and research use. Please contact the authors for other uses.
