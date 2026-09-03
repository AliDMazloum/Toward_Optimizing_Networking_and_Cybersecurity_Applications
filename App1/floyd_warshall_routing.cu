// floyd_warshall_routing.cu
//
// All-pairs shortest path on the GPU, for the network resilience case study of
// "Toward Optimizing Networking and Cybersecurity Applications Using
// Domain-Specific Accelerators for Dynamic Programming".
//
// One program covers the kernel variants used in the paper, plus a blocked
// (tiled) formulation added during the revision. Three flags pick one at run
// time, so a sweep does not require editing and recompiling:
//
//   --layout coalesced | strided | tiled   thread to data mapping
//   --dpx    on | off              DPX instruction, or its plain equivalent
//   --store  always | changed      write every cell, or only the improvements
//
// Every run repeats the whole computation --trials times and reports two
// timings per trial:
//
//   kernel      the |V| kernel launches only
//   end to end  host to device copy, the launches, and the copy back
//
// Build:
//   nvcc -O3 -arch=sm_90 floyd_warshall_routing.cu -lnvidia-ml -lpthread \
//        -o floyd_warshall_routing
//
// Example:
//   ./floyd_warshall_routing --nodes 24000 --layout coalesced --dpx on \
//                            --trials 10 --warmup 1 --energy --csv results.csv

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>

#include <nvml.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Fixed parameters
// ---------------------------------------------------------------------------

// 256 threads per block was the best performing block size in our measurements.
// The block count is not fixed: it is derived per run from the problem size and
// from what the device can hold, in derive_grid() below.
#define BLOCK_THREADS 256

// Tile width of the tiled layout. 32 is the warp width, so one tile row is one
// warp and a 32 x 32 block is 1024 threads, the hardware maximum per block.
// Two int tiles in shared memory are 8 KB. The tiled layout pads the matrix up
// to a whole number of tiles; the padding is INF in both directions, so it is
// unreachable and cannot change a real distance.
#define TILE 32

// Distance used for "no path". Any pair sum stays below 2*INF, so the relaxation
// never overflows a 32-bit signed integer.
#define INF 99999

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Monotonic clock helper
// ---------------------------------------------------------------------------

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

// ---------------------------------------------------------------------------
// NVML power sampling
//
// The sampler runs for the whole program and keeps its samples in memory. The
// original version appended to a file and flushed on every sample, which put
// disk I/O inside the measured window; energy for one trial is obtained here by
// integrating the samples whose timestamps fall inside that trial's window.
// ---------------------------------------------------------------------------

typedef struct {
    double t;  // seconds, same clock as now_seconds()
    double w;  // watts
} PowerSample;

static volatile bool  g_poll_running = false;
static nvmlDevice_t   g_nvml_device;
static pthread_t      g_poll_thread;
static PowerSample   *g_samples      = NULL;
static size_t         g_sample_count = 0;
static size_t         g_sample_cap   = 0;
static long           g_poll_interval_ms = 1;

// The sampler appends while the main thread integrates a finished window, and
// an append can reallocate the buffer, so both sides take this lock.
static pthread_mutex_t g_sample_lock = PTHREAD_MUTEX_INITIALIZER;

static void nvml_check(nvmlReturn_t r, const char *what)
{
    if (r != NVML_SUCCESS) {
        fprintf(stderr, "NVML error in %s: %s\n", what, nvmlErrorString(r));
        exit(EXIT_FAILURE);
    }
}

static void *power_polling_func(void *unused)
{
    (void)unused;
    while (g_poll_running) {
        unsigned int milliwatts = 0;
        nvmlReturn_t r = nvmlDeviceGetPowerUsage(g_nvml_device, &milliwatts);
        const double stamp = now_seconds();
        if (r == NVML_SUCCESS) {
            pthread_mutex_lock(&g_sample_lock);
            if (g_sample_count == g_sample_cap) {
                size_t next = g_sample_cap ? g_sample_cap * 2 : 65536;
                PowerSample *grown =
                    (PowerSample *)realloc(g_samples, next * sizeof(PowerSample));
                if (grown == NULL) {
                    fprintf(stderr, "Power sample buffer allocation failed\n");
                    pthread_mutex_unlock(&g_sample_lock);
                    break;
                }
                g_samples = grown;
                g_sample_cap = next;
            }
            g_samples[g_sample_count].t = stamp;
            g_samples[g_sample_count].w = (double)milliwatts / 1000.0;
            g_sample_count++;
            pthread_mutex_unlock(&g_sample_lock);
        } else {
            fprintf(stderr, "NVML warning (nvmlDeviceGetPowerUsage): %s\n",
                    nvmlErrorString(r));
        }

        if (g_poll_interval_ms > 0) {
            struct timespec ts;
            ts.tv_sec  = g_poll_interval_ms / 1000;
            ts.tv_nsec = (g_poll_interval_ms % 1000) * 1000000L;
            nanosleep(&ts, NULL);
        } else {
            sched_yield();
        }
    }
    return NULL;
}

static void power_start(unsigned int device_index, long interval_ms)
{
    unsigned int device_count = 0;
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];

    nvml_check(nvmlInit(), "nvmlInit");
    nvml_check(nvmlDeviceGetCount(&device_count), "nvmlDeviceGetCount");
    if (device_index >= device_count) {
        fprintf(stderr, "NVML device index %u is out of range, %u present\n",
                device_index, device_count);
        exit(EXIT_FAILURE);
    }
    nvml_check(nvmlDeviceGetHandleByIndex(device_index, &g_nvml_device),
               "nvmlDeviceGetHandleByIndex");
    nvml_check(nvmlDeviceGetName(g_nvml_device, name, sizeof(name)),
               "nvmlDeviceGetName");
    printf("# nvml_device        : %s (index %u)\n", name, device_index);

    g_poll_interval_ms = interval_ms;
    g_poll_running = true;
    if (pthread_create(&g_poll_thread, NULL, power_polling_func, NULL) != 0) {
        fprintf(stderr, "Could not start the power polling thread\n");
        exit(EXIT_FAILURE);
    }
}

static void power_stop(void)
{
    if (!g_poll_running) return;
    g_poll_running = false;
    pthread_join(g_poll_thread, NULL);
    nvml_check(nvmlShutdown(), "nvmlShutdown");
}

// Trapezoidal integration of power over [t0, t1]. Returns false when the window
// holds fewer than two samples, which is the case the paper reports as an
// unreliable energy reading at small problem sizes.
static bool power_window(double t0, double t1, double *energy_j,
                         double *mean_w, int *n_samples)
{
    double energy = 0.0, sum = 0.0;
    int n = 0;
    bool have_prev = false;
    PowerSample prev = { 0.0, 0.0 };

    pthread_mutex_lock(&g_sample_lock);
    for (size_t i = 0; i < g_sample_count; i++) {
        const PowerSample s = g_samples[i];
        if (s.t < t0 || s.t > t1) continue;
        if (have_prev) energy += 0.5 * (s.w + prev.w) * (s.t - prev.t);
        sum += s.w;
        n++;
        prev = s;
        have_prev = true;
    }
    pthread_mutex_unlock(&g_sample_lock);

    *n_samples = n;
    if (n < 2) return false;
    *energy_j = energy;
    *mean_w   = sum / n;
    return true;
}

static void power_dump_csv(const char *path)
{
    FILE *f = fopen(path, "w");
    if (f == NULL) { perror("power log"); return; }
    fprintf(f, "timestamp_s,power_w\n");
    for (size_t i = 0; i < g_sample_count; i++)
        fprintf(f, "%.6f,%.3f\n", g_samples[i].t, g_samples[i].w);
    fclose(f);
}

// ---------------------------------------------------------------------------
// Kernels
//
// Every layout performs the same relaxation for a fixed intermediate vertex k:
//
//     D[i][j] = min(D[i][k] + D[k][j], D[i][j])
//
// Two template parameters select the four ways of expressing it.
//
// USE_DPX picks whether a DPX instruction is used. DPX needs compute capability
// 9.0 to run in hardware, so the off arm is the same computation on the same
// chip without it.
//
// STORE_CHANGED picks whether every cell is written or only the cells whose
// value improves. Skipping the rest removes most of the write traffic, and it
// is where the DPX predicate form earns its keep, because __vibmin_s32 returns
// the comparison alongside the minimum. NVIDIA defines it as min(a, b) with the
// predicate set to (a <= b), so the predicate is true exactly when the new
// candidate wins.
// ---------------------------------------------------------------------------

template <bool USE_DPX, bool STORE_CHANGED>
__device__ __forceinline__ void relax_cell(int a, int b, int *cell)
{
    const int c = *cell;
    if (STORE_CHANGED) {
        if (USE_DPX) {
            bool improved;
            const int r = __vibmin_s32(a + b, c, &improved);
            if (improved) *cell = r;
        } else {
            const int t = a + b;
            if (t < c) *cell = t;
        }
    } else {
        if (USE_DPX) {
            *cell = __viaddmin_s32(a, b, c);
        } else {
            const int t = a + b;
            *cell = t < c ? t : c;
        }
    }
}

// Coalesced mapping: a block strides over the rows, a thread strides over the
// columns. Threads of a warp therefore address adjacent entries of a row.
template <bool USE_DPX, bool STORE_CHANGED>
__global__ void fw_coalesced(int V, int k, int *dis)
{
    for (int i = blockIdx.x; i < V; i += gridDim.x) {
        const long long row = (long long)i * V;
        const int d_ik = dis[row + k];
        const long long krow = (long long)k * V;
        for (int j = threadIdx.x; j < V; j += blockDim.x) {
            relax_cell<USE_DPX, STORE_CHANGED>(d_ik, dis[krow + j], &dis[row + j]);
        }
    }
}

// Strided mapping: one flat loop over the |V|^2 entries in which the row index
// varies fastest, so consecutive threads address entries |V| apart. This is the
// non-coalesced arm of the memory access experiment.
template <bool USE_DPX, bool STORE_CHANGED>
__global__ void fw_strided(int V, int k, int *dis)
{
    const long long n = (long long)V * V;
    const long long stride = (long long)gridDim.x * blockDim.x;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < n; idx += stride) {
        const int row = (int)(idx % V);
        const int col = (int)(idx / V);
        const long long cell = (long long)row * V + col;
        relax_cell<USE_DPX, STORE_CHANGED>(dis[(long long)row * V + k],
                                           dis[(long long)k * V + col],
                                           &dis[cell]);
    }
}

// ---------------------------------------------------------------------------
// Tiled layout: blocked Floyd-Warshall, three kernels per round of TILE
// intermediate vertices (Katz and Kider's formulation).
//
// The flat layouts stream the whole matrix from global memory once per
// intermediate vertex, so they are bandwidth-bound and a fused add+min has
// nothing to save. The blocked formulation loads TILE x TILE tiles into shared
// memory and reuses each element TILE times, which raises the arithmetic
// intensity by the tile width. It is the standard high-performance GPU
// formulation, and it is the regime where the DPX instruction can matter.
//
// Round r finalises the TILE intermediate vertices of diagonal tile (r, r):
//   phase 1: the diagonal tile itself, one block;
//   phase 2: the other tiles of block-row r and block-column r, which need the
//            finished diagonal tile;
//   phase 3: every remaining tile, which needs its row tile (bi, r) and its
//            column tile (r, bj), both finished in phase 2.
//
// Phases 1 and 2 update their tile in shared memory, so each step computes
// into a register and stores after a barrier, with a second barrier after the
// store. Phase 3 has no in-tile dependency: the accumulator stays in a
// register for all TILE steps, and that loop is where the DPX arm differs, one
// fused instruction against an add, a compare and a select. Unlike the
// conditional store of the flat layouts, the minimum the instruction returns
// is exactly the value wanted here, so it does no redundant work.
// ---------------------------------------------------------------------------

template <bool USE_DPX>
__device__ __forceinline__ int add_min(int a, int b, int c)
{
    if (USE_DPX) return __viaddmin_s32(a, b, c);
    const int t = a + b;
    return t < c ? t : c;
}

// Phase 1: the diagonal tile.
template <bool USE_DPX, bool STORE_CHANGED>
__global__ void fw_tiled_diag(int Vp, int r, int *dis)
{
    __shared__ int tile[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int base = r * TILE;
    const long long idx = (long long)(base + ty) * Vp + base + tx;

    tile[ty][tx] = dis[idx];
    __syncthreads();
    const int before = tile[ty][tx];

    for (int kk = 0; kk < TILE; kk++) {
        const int t = add_min<USE_DPX>(tile[ty][kk], tile[kk][tx], tile[ty][tx]);
        __syncthreads();
        tile[ty][tx] = t;
        __syncthreads();
    }

    if (STORE_CHANGED) {
        if (tile[ty][tx] < before) dis[idx] = tile[ty][tx];
    } else {
        dis[idx] = tile[ty][tx];
    }
}

// Phase 2: block-row r (blockIdx.y == 0) and block-column r (blockIdx.y == 1).
// The grid x dimension has nb - 1 blocks and skips the diagonal tile.
template <bool USE_DPX, bool STORE_CHANGED>
__global__ void fw_tiled_line(int Vp, int r, int *dis)
{
    __shared__ int own[TILE][TILE];
    __shared__ int diag[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int base = r * TILE;

    int m = blockIdx.x;
    if (m >= r) m++;  // skip the diagonal tile

    diag[ty][tx] = dis[(long long)(base + ty) * Vp + base + tx];

    long long idx;
    if (blockIdx.y == 0)  // tile (r, m): the rows are the intermediate vertices
        idx = (long long)(base + ty) * Vp + m * TILE + tx;
    else                  // tile (m, r): the columns are
        idx = (long long)(m * TILE + ty) * Vp + base + tx;
    own[ty][tx] = dis[idx];
    __syncthreads();
    const int before = own[ty][tx];

    for (int kk = 0; kk < TILE; kk++) {
        const int a = (blockIdx.y == 0) ? diag[ty][kk] : own[ty][kk];
        const int b = (blockIdx.y == 0) ? own[kk][tx]  : diag[kk][tx];
        const int t = add_min<USE_DPX>(a, b, own[ty][tx]);
        __syncthreads();
        own[ty][tx] = t;
        __syncthreads();
    }

    if (STORE_CHANGED) {
        if (own[ty][tx] < before) dis[idx] = own[ty][tx];
    } else {
        dis[idx] = own[ty][tx];
    }
}

// Phase 3: the remaining (nb - 1)^2 tiles. Both grid dimensions skip index r.
template <bool USE_DPX, bool STORE_CHANGED>
__global__ void fw_tiled_rest(int Vp, int r, int *dis)
{
    __shared__ int rowt[TILE][TILE];  // tile (bi, r), holds d[i][kk]
    __shared__ int colt[TILE][TILE];  // tile (r, bj), holds d[kk][j]
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int base = r * TILE;

    int bi = blockIdx.y; if (bi >= r) bi++;
    int bj = blockIdx.x; if (bj >= r) bj++;

    rowt[ty][tx] = dis[(long long)(bi * TILE + ty) * Vp + base + tx];
    colt[ty][tx] = dis[(long long)(base + ty) * Vp + bj * TILE + tx];
    __syncthreads();

    const long long idx = (long long)(bi * TILE + ty) * Vp + bj * TILE + tx;
    int c = dis[idx];
    const int before = c;

    for (int kk = 0; kk < TILE; kk++)
        c = add_min<USE_DPX>(rowt[ty][kk], colt[kk][tx], c);

    if (STORE_CHANGED) {
        if (c < before) dis[idx] = c;
    } else {
        dis[idx] = c;
    }
}

// Serial reference on the host, used to check the GPU result and as the CPU
// baseline. It is the textbook triple loop.
static void fw_cpu(int V, int *dis)
{
    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            const long long row = (long long)i * V;
            const int d_ik = dis[row + k];
            const long long krow = (long long)k * V;
            for (int j = 0; j < V; j++) {
                const int t = d_ik + dis[krow + j];
                if (t < dis[row + j]) dis[row + j] = t;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Problem set-up and checking
//
// The topology is a directed chain: vertex i has one outgoing edge to vertex
// i+1 of weight 1. Its all-pairs distance matrix is known in closed form,
// D[i][j] = j - i for j >= i and INF otherwise, which gives an exact oracle for
// every entry rather than a spot check.
// ---------------------------------------------------------------------------

// The matrix is pitch x pitch. The tiled layout pads pitch above V; padded
// cells (row or column at or beyond V) are INF in both directions, so no real
// path can improve through them. The other layouts pass pitch == V.
static void init_matrix(int V, int pitch, int *dis)
{
    for (int i = 0; i < pitch; i++) {
        for (int j = 0; j < pitch; j++) {
            int v;
            if (i >= V || j >= V) v = INF;
            else if (j == i + 1)  v = 1;
            else if (i != j)      v = INF;
            else                  v = 0;
            dis[(long long)i * pitch + j] = v;
        }
    }
}

static long long count_mismatches(int V, int pitch, const int *dis)
{
    long long bad = 0;
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            const int expected = (j >= i) ? (j - i) : INF;
            if (dis[(long long)i * pitch + j] != expected) bad++;
        }
    }
    return bad;
}

// ---------------------------------------------------------------------------
// Launch configuration
//
// The block count is derived, never hand-picked. It is the smaller of the work
// available and the number of blocks the device can hold resident for this
// kernel, which comes from the occupancy API and the SM count.
// ---------------------------------------------------------------------------

typedef enum { LAYOUT_COALESCED, LAYOUT_STRIDED, LAYOUT_TILED } Layout;

static const char *layout_name(Layout l)
{
    if (l == LAYOUT_COALESCED) return "coalesced";
    if (l == LAYOUT_STRIDED)   return "strided";
    return "tiled";
}

// Both flat kernels have this signature, so one pointer type serves for the
// occupancy query and for dispatch. The tiled kernels are launched by
// launch_tiled_round below and need neither the pointer nor the occupancy
// query, because their grids are fixed by the tile geometry.
typedef void (*FwKernel)(int, int, int *);

// The occupancy query needs the instantiation that will actually run.
static FwKernel select_kernel(bool coalesced, bool use_dpx, bool store_changed)
{
    if (coalesced) {
        if (use_dpx)
            return store_changed ? fw_coalesced<true, true> : fw_coalesced<true, false>;
        return store_changed ? fw_coalesced<false, true> : fw_coalesced<false, false>;
    }
    if (use_dpx)
        return store_changed ? fw_strided<true, true> : fw_strided<true, false>;
    return store_changed ? fw_strided<false, true> : fw_strided<false, false>;
}

// A launch cannot go through a function pointer, so the same choice is spelled
// out once here rather than at every call site.
static void launch_relaxation(bool coalesced, bool use_dpx, bool store_changed,
                              int grid, int V, int k, int *dis_d)
{
    if (coalesced) {
        if (use_dpx) {
            if (store_changed) fw_coalesced<true, true><<<grid, BLOCK_THREADS>>>(V, k, dis_d);
            else               fw_coalesced<true, false><<<grid, BLOCK_THREADS>>>(V, k, dis_d);
        } else {
            if (store_changed) fw_coalesced<false, true><<<grid, BLOCK_THREADS>>>(V, k, dis_d);
            else               fw_coalesced<false, false><<<grid, BLOCK_THREADS>>>(V, k, dis_d);
        }
    } else {
        if (use_dpx) {
            if (store_changed) fw_strided<true, true><<<grid, BLOCK_THREADS>>>(V, k, dis_d);
            else               fw_strided<true, false><<<grid, BLOCK_THREADS>>>(V, k, dis_d);
        } else {
            if (store_changed) fw_strided<false, true><<<grid, BLOCK_THREADS>>>(V, k, dis_d);
            else               fw_strided<false, false><<<grid, BLOCK_THREADS>>>(V, k, dis_d);
        }
    }
}

// One round of the tiled layout: the three phases for diagonal index r. The
// synchronization option is applied inside, so --sync means the same thing in
// every layout: a host wait after every kernel launch.
template <bool USE_DPX, bool STORE_CHANGED>
static void tiled_round(int nb, int Vp, int r, int *dis_d, bool sync_each)
{
    const dim3 threads(TILE, TILE);
    fw_tiled_diag<USE_DPX, STORE_CHANGED><<<1, threads>>>(Vp, r, dis_d);
    if (sync_each) CUDA_CHECK(cudaDeviceSynchronize());
    if (nb > 1) {
        fw_tiled_line<USE_DPX, STORE_CHANGED>
            <<<dim3(nb - 1, 2), threads>>>(Vp, r, dis_d);
        if (sync_each) CUDA_CHECK(cudaDeviceSynchronize());
        fw_tiled_rest<USE_DPX, STORE_CHANGED>
            <<<dim3(nb - 1, nb - 1), threads>>>(Vp, r, dis_d);
        if (sync_each) CUDA_CHECK(cudaDeviceSynchronize());
    }
}

static void launch_tiled_round(bool use_dpx, bool store_changed,
                               int nb, int Vp, int r, int *dis_d, bool sync_each)
{
    if (use_dpx) {
        if (store_changed) tiled_round<true, true>(nb, Vp, r, dis_d, sync_each);
        else               tiled_round<true, false>(nb, Vp, r, dis_d, sync_each);
    } else {
        if (store_changed) tiled_round<false, true>(nb, Vp, r, dis_d, sync_each);
        else               tiled_round<false, false>(nb, Vp, r, dis_d, sync_each);
    }
}

static int derive_grid(FwKernel kernel, int V, bool strided_layout,
                       int *blocks_per_sm_out, int *sm_count_out)
{
    int blocks_per_sm = 0;
    cudaDeviceProp prop;
    int device = 0;

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kernel, BLOCK_THREADS, 0));

    const long long resident = (long long)prop.multiProcessorCount * blocks_per_sm;

    // Work available, in blocks, if every thread took one item.
    long long work;
    if (strided_layout) {
        work = (((long long)V * V) + BLOCK_THREADS - 1) / BLOCK_THREADS;
    } else {
        work = V;  // one block per row
    }

    long long grid = work < resident ? work : resident;
    if (grid < 1) grid = 1;

    *blocks_per_sm_out = blocks_per_sm;
    *sm_count_out = prop.multiProcessorCount;
    return (int)grid;
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

static double mean_of(const double *v, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += v[i];
    return s / n;
}

static double stddev_of(const double *v, int n, double mean)
{
    if (n < 2) return 0.0;
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (v[i] - mean) * (v[i] - mean);
    return sqrt(s / (n - 1));  // sample standard deviation
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

static void print_usage(const char *prog)
{
    printf("\nUsage: %s [options]\n\n", prog);
    printf("  --nodes <int>       number of vertices (default 12000)\n");
    printf("  --layout <name>     coalesced | strided | tiled (default coalesced)\n");
    printf("  --dpx <state>       on | off (default on)\n");
    printf("  --store <policy>    always | changed: write every cell, or only\n");
    printf("                      the cells whose value improves (default always)\n");
    printf("  --trials <int>      measured repetitions (default 1)\n");
    printf("  --warmup <int>      unmeasured repetitions first (default 1)\n");
    printf("  --sync <state>      per-launch | none: host synchronization after\n");
    printf("                      each launch (default per-launch)\n");
    printf("  --cpu               run the serial host reference instead of the GPU\n");
    printf("  --energy            sample GPU power with NVML and report energy\n");
    printf("  --poll-ms <int>     NVML sampling interval, ms (default 1)\n");
    printf("  --device <int>      CUDA and NVML device index (default 0)\n");
    printf("  --csv <path>        append one row per trial to this file\n");
    printf("  --power-csv <path>  write every power sample to this file\n");
    printf("  --no-verify         skip the closed-form correctness check\n");
    printf("  --help              show this message\n\n");
}

int main(int argc, char **argv)
{
    int   V           = 12000;
    Layout layout     = LAYOUT_COALESCED;
    bool  use_dpx     = true;
    int   trials      = 1;
    int   warmup      = 1;
    bool  run_cpu     = false;
    bool  measure_energy = false;
    long  poll_ms     = 1;
    int   device      = 0;
    bool  verify      = true;
    bool  sync_each   = true;
    bool  store_changed = false;
    const char *csv_path = NULL;
    const char *power_csv_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            print_usage(argv[0]);
            return 0;
        } else if (!strcmp(argv[i], "--nodes") && i + 1 < argc) {
            V = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--layout") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "coalesced"))    layout = LAYOUT_COALESCED;
            else if (!strcmp(v, "strided")) layout = LAYOUT_STRIDED;
            else if (!strcmp(v, "tiled"))   layout = LAYOUT_TILED;
            else { fprintf(stderr, "Unknown layout: %s\n", v); return 1; }
        } else if (!strcmp(argv[i], "--dpx") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "on"))       use_dpx = true;
            else if (!strcmp(v, "off")) use_dpx = false;
            else { fprintf(stderr, "Unknown dpx state: %s\n", v); return 1; }
        } else if (!strcmp(argv[i], "--trials") && i + 1 < argc) {
            trials = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--store") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "always"))       store_changed = false;
            else if (!strcmp(v, "changed")) store_changed = true;
            else { fprintf(stderr, "Unknown store policy: %s\n", v); return 1; }
        } else if (!strcmp(argv[i], "--sync") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "per-launch"))  sync_each = true;
            else if (!strcmp(v, "none"))   sync_each = false;
            else { fprintf(stderr, "Unknown sync state: %s\n", v); return 1; }
        } else if (!strcmp(argv[i], "--cpu")) {
            run_cpu = true;
        } else if (!strcmp(argv[i], "--energy")) {
            measure_energy = true;
        } else if (!strcmp(argv[i], "--poll-ms") && i + 1 < argc) {
            poll_ms = atol(argv[++i]);
        } else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            device = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--csv") && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (!strcmp(argv[i], "--power-csv") && i + 1 < argc) {
            power_csv_path = argv[++i];
        } else if (!strcmp(argv[i], "--no-verify")) {
            verify = false;
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (V < 2)       { fprintf(stderr, "--nodes must be at least 2\n"); return 1; }
    if (trials < 1)  { fprintf(stderr, "--trials must be at least 1\n"); return 1; }
    if (warmup < 0)  { fprintf(stderr, "--warmup cannot be negative\n"); return 1; }

    // The tiled layout computes on a matrix padded up to a whole number of
    // tiles; the correctness check still reads only the real V x V entries.
    // Every other layout uses the exact size, so Vp == V there.
    const bool tiled = (layout == LAYOUT_TILED) && !run_cpu;
    const int  Vp    = tiled ? ((V + TILE - 1) / TILE) * TILE : V;
    const int  nb    = Vp / TILE;
    const size_t bytes = (size_t)Vp * (size_t)Vp * sizeof(int);
    int *dis = (int *)malloc(bytes);
    if (dis == NULL) {
        fprintf(stderr, "Host allocation of %.2f GB failed\n", bytes / 1e9);
        return 1;
    }

    int *dis_d = NULL;
    int grid = 0, blocks_per_sm = 0, sm_count = 0;
    cudaDeviceProp prop;

    if (!run_cpu) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        CUDA_CHECK(cudaMalloc((void **)&dis_d, bytes));

        if (tiled) {
            // The tiled grids are fixed by the geometry: one block, then
            // (nb - 1) x 2, then (nb - 1)^2 per round. The grid recorded in
            // the CSV is the phase 3 grid, which does almost all of the work.
            grid = nb > 1 ? (nb - 1) * (nb - 1) : 1;
        } else {
            FwKernel kernel = select_kernel(layout == LAYOUT_COALESCED,
                                            use_dpx, store_changed);
            grid = derive_grid(kernel, V, layout == LAYOUT_STRIDED,
                               &blocks_per_sm, &sm_count);
        }
    }

    // Every setting that affects a number is printed, so a run documents itself.
    printf("# program           : floyd_warshall_routing\n");
    printf("# nodes             : %d\n", V);
    printf("# matrix_bytes      : %zu\n", bytes);
    printf("# target            : %s\n", run_cpu ? "cpu (serial reference)" : "gpu");
    if (!run_cpu) {
        printf("# gpu               : %s\n", prop.name);
        printf("# compute_capability: %d.%d\n", prop.major, prop.minor);
        printf("# layout            : %s\n", layout_name(layout));
        printf("# dpx               : %s\n", use_dpx ? "on" : "off");
        printf("# store             : %s\n",
               store_changed ? "only cells that improve" : "every cell");
        if (tiled) {
            printf("# tile              : %d x %d, matrix padded to %d nodes, %d rounds\n",
                   TILE, TILE, Vp, nb);
            printf("# block_threads     : %d (%d x %d)\n", TILE * TILE, TILE, TILE);
            printf("# grid_blocks       : 1, then %d x 2, then %d x %d per round\n",
                   nb - 1, nb - 1, nb - 1);
            printf("# launches_per_trial: %d\n", nb > 1 ? 3 * nb : 1);
        } else {
            printf("# block_threads     : %d\n", BLOCK_THREADS);
            printf("# grid_blocks       : %d (min of work and %d SMs x %d resident blocks)\n",
                   grid, sm_count, blocks_per_sm);
            printf("# launches_per_trial: %d\n", V);
        }
        printf("# sync              : %s\n",
               sync_each ? "per launch" : "none, launches are queued");
    }
    printf("# trials            : %d\n", trials);
    printf("# warmup            : %d (not reported)\n", warmup);
    printf("# verify            : %s\n", verify ? "on" : "off");
    printf("# kernel_window     : the %s only\n",
           run_cpu ? "serial triple loop" : "kernel launches");
    if (run_cpu)
        printf("# endtoend_window   : the serial triple loop\n");
    else
        printf("# endtoend_window   : host to device copy, the kernel launches,"
               " device to host copy\n");

    if (measure_energy && !run_cpu) {
        printf("# poll_interval_ms  : %ld\n", poll_ms);
        power_start((unsigned int)device, poll_ms);
        // Let the sampler produce a first reading before any trial starts.
        struct timespec settle = { 0, 100 * 1000000L };
        nanosleep(&settle, NULL);
    }

    double *kernel_s   = (double *)calloc(trials, sizeof(double));
    double *endtoend_s = (double *)calloc(trials, sizeof(double));
    double *energy_j   = (double *)calloc(trials, sizeof(double));
    double *power_w    = (double *)calloc(trials, sizeof(double));
    bool   *energy_ok  = (bool *)calloc(trials, sizeof(bool));

    cudaEvent_t ev_start, ev_stop;
    if (!run_cpu) {
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
    }

    printf("trial,kernel_s,endtoend_s");
    if (measure_energy && !run_cpu) printf(",energy_j,mean_power_w,power_samples");
    if (verify) printf(",mismatches");
    printf("\n");

    for (int t = -warmup; t < trials; t++) {
        const bool measured = (t >= 0);

        // Rebuilding the matrix is outside both windows.
        init_matrix(V, Vp, dis);

        double t0 = 0.0, t1 = 0.0, kernel_seconds = 0.0;

        if (run_cpu) {
            t0 = now_seconds();
            fw_cpu(V, dis);
            t1 = now_seconds();
            kernel_seconds = t1 - t0;
        } else {
            t0 = now_seconds();
            CUDA_CHECK(cudaMemcpy(dis_d, dis, bytes, cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaEventRecord(ev_start, 0));
            if (tiled) {
                for (int r = 0; r < nb; r++)
                    launch_tiled_round(use_dpx, store_changed, nb, Vp, r,
                                       dis_d, sync_each);
            } else {
                for (int k = 0; k < V; k++) {
                    launch_relaxation(layout == LAYOUT_COALESCED, use_dpx,
                                      store_changed, grid, V, k, dis_d);
                    // Launches on one stream already run in order, so the
                    // synchronization is not needed for correctness. It is the
                    // default because it is what the originally reported
                    // measurements did, and --sync none drops it so that the
                    // host can queue the launches instead of waiting for each
                    // one.
                    if (sync_each) CUDA_CHECK(cudaDeviceSynchronize());
                }
            }
            CUDA_CHECK(cudaEventRecord(ev_stop, 0));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            kernel_seconds = ms / 1000.0;

            CUDA_CHECK(cudaMemcpy(dis, dis_d, bytes, cudaMemcpyDeviceToHost));
            t1 = now_seconds();
        }

        long long bad = verify ? count_mismatches(V, Vp, dis) : 0;

        if (!measured) continue;

        kernel_s[t]   = kernel_seconds;
        endtoend_s[t] = t1 - t0;

        int nsamp = 0;
        if (measure_energy && !run_cpu)
            energy_ok[t] = power_window(t0, t1, &energy_j[t], &power_w[t], &nsamp);

        printf("%d,%.6f,%.6f", t + 1, kernel_s[t], endtoend_s[t]);
        if (measure_energy && !run_cpu) {
            if (energy_ok[t]) printf(",%.3f,%.3f,%d", energy_j[t], power_w[t], nsamp);
            else              printf(",,,%d", nsamp);
        }
        if (verify) printf(",%lld", bad);
        printf("\n");
        fflush(stdout);

        if (csv_path != NULL) {
            FILE *f = fopen(csv_path, "a");
            if (f == NULL) {
                perror("csv");
            } else {
                fseek(f, 0, SEEK_END);
                if (ftell(f) == 0)
                    fprintf(f, "nodes,target,gpu,layout,dpx,store,sync,block_threads,"
                               "grid_blocks,trial,kernel_s,endtoend_s,energy_j,"
                               "mean_power_w,power_samples,mismatches\n");
                fprintf(f, "%d,%s,%s,%s,%s,%s,%s,%d,%d,%d,%.6f,%.6f,",
                        V, run_cpu ? "cpu" : "gpu",
                        run_cpu ? "n/a" : prop.name,
                        run_cpu ? "n/a" : layout_name(layout),
                        run_cpu ? "n/a" : (use_dpx ? "on" : "off"),
                        run_cpu ? "n/a" : (store_changed ? "changed" : "always"),
                        run_cpu ? "n/a" : (sync_each ? "per-launch" : "none"),
                        run_cpu ? 0 : (tiled ? TILE * TILE : BLOCK_THREADS),
                        run_cpu ? 0 : grid,
                        t + 1, kernel_s[t], endtoend_s[t]);
                if (measure_energy && !run_cpu && energy_ok[t])
                    fprintf(f, "%.3f,%.3f,%d,", energy_j[t], power_w[t], nsamp);
                else
                    fprintf(f, ",,%d,", nsamp);
                if (verify) fprintf(f, "%lld\n", bad); else fprintf(f, "\n");
                fclose(f);
            }
        }
    }

    if (measure_energy && !run_cpu) {
        power_stop();
        if (power_csv_path != NULL) power_dump_csv(power_csv_path);
    }

    const double k_mean = mean_of(kernel_s, trials);
    const double e_mean = mean_of(endtoend_s, trials);
    printf("# kernel_s   mean %.6f  std %.6f  over %d trials\n",
           k_mean, stddev_of(kernel_s, trials, k_mean), trials);
    printf("# endtoend_s mean %.6f  std %.6f  over %d trials\n",
           e_mean, stddev_of(endtoend_s, trials, e_mean), trials);

    if (measure_energy && !run_cpu) {
        int good = 0;
        double sum = 0.0;
        for (int t = 0; t < trials; t++) if (energy_ok[t]) { sum += energy_j[t]; good++; }
        if (good >= 1) {
            double emean = sum / good;
            double *tmp = (double *)calloc(good, sizeof(double));
            int m = 0;
            for (int t = 0; t < trials; t++) if (energy_ok[t]) tmp[m++] = energy_j[t];
            printf("# energy_j   mean %.3f  std %.3f  over %d of %d trials\n",
                   emean, stddev_of(tmp, good, emean), good, trials);
            free(tmp);
        } else {
            printf("# energy_j   not reported: fewer than two power samples per window\n");
        }
    }

    if (!run_cpu) {
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
        CUDA_CHECK(cudaFree(dis_d));
    }
    free(dis);
    free(kernel_s);
    free(endtoend_s);
    free(energy_j);
    free(power_w);
    free(energy_ok);
    free(g_samples);
    return 0;
}
