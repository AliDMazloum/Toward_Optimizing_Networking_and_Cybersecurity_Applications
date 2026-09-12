// smith_waterman_dpi.cu
//
// Packet-payload signature matching on the GPU.
//
// One program covers all the kernel variants. Flags pick one at run time, so
// a sweep does not require editing and recompiling:
//
//   --mode literal | regex       plain signatures, or the regex scoring
//                                formulation (*, ., ~ contribute specially)
//   --rows registers | global    where the two DP rows live: in registers
//                                (the memory-focused kernel) or in a coalesced
//                                global-memory buffer (the occupancy-focused
//                                kernel)
//   --dpx  on | off              the DPX halfword instructions
//                                __viaddmax_s16x2 and __viaddmax_s16x2_relu,
//                                or the same per-halfword computation
//                                without them
//
// Two signatures are packed per 32-bit word (one per 16-bit halfword), so one
// thread scores two signatures at once; the packing is identical in both
// --dpx arms.
//
// Every run repeats the whole scan --trials times and reports two timings per
// trial:
//
//   kernel      the kernel launch only
//   end to end  payload upload, the kernel, and the report copy back
//
// The signature database upload is one-time setup, reported once, because a
// deployment loads its signatures once and then streams payloads.
//
// A match is not part of a timing run unless one is planted: by default no
// signature matches the random payload, every thread scans everything, and the
// measured time is the worst case. --plant <index> makes that signature match
// (its text is copied into the payload), which is how detection is checked.
// The first thread whose score crosses its threshold claims a report word by
// atomic compare-and-swap, so the record is coherent; --exit picks whether
// that thread then stops or keeps scanning. --verify recomputes the reported
// signature's score with a host implementation of the same recurrence and
// counts a mismatch if the two disagree.
//
// Build:
//   nvcc -O3 -arch=sm_90 smith_waterman_dpi.cu -lnvidia-ml -lpthread \
//        -o smith_waterman_dpi
//
// Example:
//   ./smith_waterman_dpi --signatures 20000000 --payload 512 --sig-len 16 \
//                        --mode literal --rows registers --dpx on \
//                        --trials 10 --warmup 1 --energy --csv results.csv

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>

#include <nvml.h>
#include <cuda_runtime.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Fixed parameters
//
// Literal mode adds 1 per matching character, on the diagonal only, so a
// signature's maximum score is its length and the threshold is a fraction of
// that length. Regex mode uses the larger match reward so that wildcard
// positions, which contribute 0, still leave literal matches room to
// dominate; its maximum is RE_MATCH times the literal count.
// ---------------------------------------------------------------------------

#define LIT_MATCH     1   // literal mode: diagonal score for a matching character
#define LIT_MISMATCH -2   // literal mode: diagonal score for a mismatch
#define LIT_GAP      -1   // literal mode: penalty for a vertical or horizontal move

#define RE_MATCH      6   // regex mode: diagonal score for a matching literal
#define RE_MISMATCH  -3   // regex mode: diagonal score for a mismatched literal
#define RE_INDEL     -2   // regex mode: gap penalty, 0 in a * column

// Signature lengths the register-row kernel is compiled for. The DP rows and
// the cached signature bytes can only stay in registers when the inner loop
// has compile-time bounds, so each supported length is its own instantiation.
#define SIG_LEN_A 16
#define SIG_LEN_B 32
#define MAX_SIG_LEN 32

// The payload lives in constant memory; every thread reads the same byte at
// the same time, which is what constant memory broadcasts.
#define MAX_PAYLOAD 4096

__constant__ char c_payload[MAX_PAYLOAD];

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
// Same sampler as the routing program: it runs for the whole program and keeps
// its samples in memory, and energy for one trial is obtained by integrating
// the samples whose timestamps fall inside that trial's window.
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

// A CUDA device index and an NVML device index are not the same name for the
// same card. CUDA_VISIBLE_DEVICES renumbers what CUDA can see and NVML ignores
// that mask, so handing one index to both libraries can leave the sampler
// watching a card that is running nothing. The PCI bus id is the one identifier
// both agree on, so the handle is resolved through it and printed, which makes
// a mismatch visible in the run header instead of silent in the numbers.
static void power_start(int cuda_device, long interval_ms)
{
    char pci_id[32];
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];

    CUDA_CHECK(cudaDeviceGetPCIBusId(pci_id, (int)sizeof pci_id, cuda_device));

    nvml_check(nvmlInit(), "nvmlInit");
    nvml_check(nvmlDeviceGetHandleByPciBusId(pci_id, &g_nvml_device),
               "nvmlDeviceGetHandleByPciBusId");
    nvml_check(nvmlDeviceGetName(g_nvml_device, name, sizeof(name)),
               "nvmlDeviceGetName");
    printf("# nvml_device        : %s at %s\n", name, pci_id);

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

// Cumulative device energy since the driver was last loaded, in millijoules.
// Volta and newer expose this counter; where it is missing the read fails and
// the column is left empty rather than the run failing. It integrates inside
// the device, so unlike the sampled figure it does not depend on how often the
// driver refreshes its power reading: bracketing a window with two reads gives
// that window's energy even when the window is shorter than one refresh.
static bool energy_counter_mj(unsigned long long *mj)
{
    return nvmlDeviceGetTotalEnergyConsumption(g_nvml_device, mj) == NVML_SUCCESS;
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
// The recurrence
//
// For payload p (length P) and signature s (length L), with H[0][*] and
// H[*][0] fixed at 0, the Smith-Waterman cell:
//
//   H[i][j] = max(H[i-1][j-1] + score(p[i-1], s[j-1]),
//                 H[i-1][j]   + gap(s[j-1]),
//                 H[i][j-1]   + gap(s[j-1]),
//                 0)
//
// The character score enters on the diagonal only, so a signature position is
// credited at most once per alignment and a score never exceeds the match
// reward times the literal count. Literal mode scores LIT_MATCH or
// LIT_MISMATCH and charges LIT_GAP. Regex mode, by signature character:
//
//   literal   RE_MATCH or RE_MISMATCH, gap RE_INDEL
//   *         score 0 and gap 0, so it absorbs any run of payload bytes,
//             including none
//   .         score 0 against any byte, gap RE_INDEL
//   ~         score 0 against a digit and RE_MISMATCH otherwise, gap RE_INDEL
//
// Every payload byte enters the recurrence, and both modes test the newly
// computed cell: literal with strict greater-than against floor(alpha * L),
// regex with greater-or-equal against a per-signature threshold.
//
// The score functions are shared, host and device, so the host reference and
// the kernels cannot drift apart, and the only thing --dpx changes is whether
// the cell is computed by the DPX instructions or by the plain equivalent.
// ---------------------------------------------------------------------------

__host__ __device__ __forceinline__ bool is_ascii_digit(char c)
{
    return c >= '0' && c <= '9';
}

__host__ __device__ __forceinline__ int score_literal(char p, char s)
{
    return p == s ? LIT_MATCH : LIT_MISMATCH;
}

__host__ __device__ __forceinline__ int score_regex(char p, char s, bool p_is_digit)
{
    if (s == '*' || s == '.') return 0;
    if (s == '~') return p_is_digit ? 0 : RE_MISMATCH;
    return p == s ? RE_MATCH : RE_MISMATCH;
}

__host__ __device__ __forceinline__ int gap_regex(char s)
{
    return s == '*' ? 0 : RE_INDEL;
}

// One cell from its three neighbours, the diagonal score and the gap.
__host__ __device__ __forceinline__ int cell_plain(int n, int nw, int w,
                                                   int score, int gap)
{
    int m = nw + score;
    if (n + gap > m) m = n + gap;
    if (w + gap > m) m = w + gap;
    return m > 0 ? m : 0;
}

// Two signed 16-bit values in one word: signature gid in the low halfword,
// signature gid + midpoint in the high one.
__host__ __device__ __forceinline__ uint32_t pack2(int lo, int hi)
{
    return (uint32_t)(uint16_t)(int16_t)lo | ((uint32_t)(uint16_t)(int16_t)hi << 16);
}

// The packed cell, both halfwords at once. The DPX arm is two instructions on
// compute capability 9.0: __viaddmax_s16x2 gives max(n + gap, w + gap) per
// halfword and __viaddmax_s16x2_relu gives max(nw + score, that, 0). The
// plain arm is what --dpx off measures on the same chip.
template <bool USE_DPX>
__device__ __forceinline__ uint32_t cell_packed(uint32_t n, uint32_t nw, uint32_t w,
                                                uint32_t score2, uint32_t gap2)
{
    if (USE_DPX) {
        const uint32_t from_gap = __viaddmax_s16x2(n, gap2, __vadd2(w, gap2));
        return __viaddmax_s16x2_relu(nw, score2, from_gap);
    }
    const int lo = cell_plain((int)(int16_t)(n  & 0xFFFFu),
                              (int)(int16_t)(nw & 0xFFFFu),
                              (int)(int16_t)(w  & 0xFFFFu),
                              (int)(int16_t)(score2 & 0xFFFFu),
                              (int)(int16_t)(gap2 & 0xFFFFu));
    const int hi = cell_plain((int)(int16_t)(n  >> 16),
                              (int)(int16_t)(nw >> 16),
                              (int)(int16_t)(w  >> 16),
                              (int)(int16_t)(score2 >> 16),
                              (int)(int16_t)(gap2 >> 16));
    return pack2(lo, hi);
}

// ---------------------------------------------------------------------------
// The report
//
// The first crossing claims the report by atomic compare-and-swap, so exactly
// one thread fills it and the three fields belong to one detection.
// ---------------------------------------------------------------------------

typedef struct {
    int claimed;  // 0 until a thread wins the claim
    int score;    // the crossing score
    int sig;      // which signature crossed
    int pos;      // payload index (0-based) of the crossing character
} Report;

__device__ __forceinline__ void claim_report(Report *r, int score, int sig, int pos)
{
    if (atomicCAS(&r->claimed, 0, 1) == 0) {
        r->score = score;
        r->sig   = sig;
        r->pos   = pos;
    }
}

// ---------------------------------------------------------------------------
// The kernel
//
// One thread scores two signatures: signature gid in the low halfword and
// signature gid + midpoint in the high halfword, where midpoint is half the
// signature count. ROWS_REG keeps the two DP rows and the two signatures in
// registers (the memory-focused kernel); otherwise the rows live in a global
// buffer laid out so that consecutive threads touch consecutive words (the
// occupancy-focused kernel), two row blocks used alternately by payload-index
// parity. REGEX picks the score step and a per-signature threshold; literal
// mode shares one threshold because every signature has the same length.
// ---------------------------------------------------------------------------

template <int SIG_LEN, bool USE_DPX, bool ROWS_REG, bool REGEX, bool EXIT_FIRST>
__global__ void sw_scan(int midpoint, int payload_len,
                        const char *__restrict__ signatures,
                        uint32_t *rows, long long row_size,
                        const int *__restrict__ thresholds, int threshold_lit,
                        bool exit_first, Report *report)
{
    // Optional compile-time pinning: -DPIN_SIGNATURES=10000000
    // -DPIN_PAYLOAD=512 -DPIN_THRESHOLD=12 -DPIN_EXIT_FIRST=1 turns these
    // parameters into constants the compiler can optimize against. The host
    // refuses flags that contradict a pin, so a pinned binary cannot measure
    // the wrong configuration. An unpinned build is unaffected.
#ifdef PIN_SIGNATURES
    midpoint = (int)((long long)PIN_SIGNATURES / 2);
#endif
#ifdef PIN_PAYLOAD
    payload_len = PIN_PAYLOAD;
#endif
#ifdef PIN_THRESHOLD
    threshold_lit = PIN_THRESHOLD;
#endif
#ifdef PIN_EXIT_FIRST
    exit_first = (PIN_EXIT_FIRST != 0);
#endif

    // The early exit is a return from inside the unrolled inner loop, and its
    // predicate is available two ways: as the run-time argument and as the
    // template parameter it was instantiated from. They always carry the same
    // value, so this chooses only how the branch is compiled.
    //
    // It matters because a run-time predicate leaves the compiler unable to
    // prove the inner loop runs to completion, so it keeps the row registers
    // live across every possible exit and the unroll's register allocation
    // suffers. Which form is faster depends on the architecture and on the
    // mode, so each variant takes the one that is faster for it: the literal
    // path below sm_90 uses the template parameter, everything else uses the
    // argument.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    const bool exit_now = REGEX ? exit_first : EXIT_FIRST;
#else
    const bool exit_now = exit_first;
#endif

    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= midpoint) return;

    const long long off0 = (long long)gid * (SIG_LEN + 1);
    const long long off1 = (long long)(gid + midpoint) * (SIG_LEN + 1);

    char sig0[SIG_LEN];
    char sig1[SIG_LEN];
    if (ROWS_REG) {
#pragma unroll
        for (int j = 0; j < SIG_LEN; j++) {
            sig0[j] = signatures[off0 + j];
            sig1[j] = signatures[off1 + j];
        }
    }

    int thr0, thr1;
    if (REGEX) {
        thr0 = thresholds[gid];
        thr1 = thresholds[gid + midpoint];
    } else {
        thr0 = thr1 = threshold_lit;
    }

    uint32_t rowA[SIG_LEN + 1];
    uint32_t rowB[SIG_LEN + 1];
    if (ROWS_REG) {
#pragma unroll
        for (int j = 0; j <= SIG_LEN; j++) { rowA[j] = 0u; rowB[j] = 0u; }
    }

    bool even = false;

    // i starts at 1, so payload byte 0 is scored like every other byte. The
    // loop bound is a compile-time-friendly constant start rather than a
    // parameter, deliberately: a run-time loop start would leave the compiler
    // unable to bound the trip count, which is the same thing that cost this
    // kernel its register allocation once already.
    for (int i = 1; i <= payload_len; i++) {
        const char p = c_payload[i - 1];
        const bool pdig = REGEX ? is_ascii_digit(p) : false;
        even = !even;

        // Row blocks for this payload character: cur is being written, prev
        // was written for the previous character. Column 0 of both blocks
        // stays 0, the DP boundary, because no j in this loop writes it.
        long long cur = 0, prev = 0;
        if (!ROWS_REG) {
            cur  = even ? row_size : 0;
            prev = even ? 0 : row_size;
        }

#pragma unroll
        for (int j = 1; j <= SIG_LEN; j++) {
            uint32_t n, nw, w;
            if (ROWS_REG) {
                if (even) { nw = rowA[j - 1]; n = rowA[j]; w = rowB[j - 1]; }
                else      { nw = rowB[j - 1]; n = rowB[j]; w = rowA[j - 1]; }
            } else {
                nw = rows[prev + (long long)(j - 1) * midpoint + gid];
                n  = rows[prev + (long long)j * midpoint + gid];
                w  = rows[cur  + (long long)(j - 1) * midpoint + gid];
            }

            const char s0 = ROWS_REG ? sig0[j - 1] : signatures[off0 + j - 1];
            const char s1 = ROWS_REG ? sig1[j - 1] : signatures[off1 + j - 1];

            const uint32_t score2 = REGEX
                ? pack2(score_regex(p, s0, pdig), score_regex(p, s1, pdig))
                : pack2(score_literal(p, s0), score_literal(p, s1));
            const uint32_t gap2 = REGEX ? pack2(gap_regex(s0), gap_regex(s1))
                                        : pack2(LIT_GAP, LIT_GAP);

            const uint32_t packed = cell_packed<USE_DPX>(n, nw, w, score2, gap2);
            const int t0 = (int)(int16_t)(packed & 0xFFFFu);
            const int t1 = (int)(int16_t)(packed >> 16);
            if (ROWS_REG) {
                if (even) rowB[j] = packed; else rowA[j] = packed;
            } else {
                rows[cur + (long long)j * midpoint + gid] = packed;
            }

            // Detection, per mode, on the new cell: regex with >=, literal
            // with strict >.
            if (REGEX) {
                if (t0 >= thr0) {
                    claim_report(report, t0, gid, i - 1);
                    if (exit_now) return;
                } else if (t1 >= thr1) {
                    claim_report(report, t1, gid + midpoint, i - 1);
                    if (exit_now) return;
                }
            } else {
                if (t0 > thr0) {
                    claim_report(report, t0, gid, i - 1);
                    if (exit_now) return;
                } else if (t1 > thr1) {
                    claim_report(report, t1, gid + midpoint, i - 1);
                    if (exit_now) return;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host reference
//
// The same recurrence on the host, one signature at a time, through the same
// score functions the kernel uses, with the same detection semantics: every
// payload byte scored and the new cell tested, literal with strict > and
// regex with >=. Returns true at the first threshold
// crossing with its score and position, which for a given signature is
// deterministic, so a reported detection must reproduce exactly. When nothing
// crosses, best_out holds the maximum cell value seen.
// ---------------------------------------------------------------------------

static bool host_scan_signature(bool regex_mode, const char *payload, int P,
                                const char *sig, int L, int thr,
                                int *score_out, int *pos_out, int *best_out)
{
    int prev[MAX_SIG_LEN + 1] = { 0 };
    int cur[MAX_SIG_LEN + 1]  = { 0 };
    int best = 0;

    // Starts at 1 for the same reason as the kernel, and must match it: this is
    // the reference every reported crossing is checked against, so a difference
    // of one column here would report itself as a mismatch on every trial.
    for (int i = 1; i <= P; i++) {
        const char p = payload[i - 1];
        const bool pdig = is_ascii_digit(p);
        cur[0] = 0;
        for (int j = 1; j <= L; j++) {
            const char s = sig[j - 1];
            const int score = regex_mode ? score_regex(p, s, pdig)
                                         : score_literal(p, s);
            const int gap   = regex_mode ? gap_regex(s) : LIT_GAP;
            const int t = cell_plain(prev[j], prev[j - 1], cur[j - 1], score, gap);
            cur[j] = t;
            if (t > best) best = t;
            if (regex_mode ? (t >= thr) : (t > thr)) {
                *score_out = t;
                *pos_out   = i - 1;
                *best_out  = best;
                return true;
            }
        }
        memcpy(prev, cur, sizeof(int) * (L + 1));
    }
    *best_out = best;
    return false;
}

// ---------------------------------------------------------------------------
// RAPL energy for the host path
//
// The kernel's powercap interface counts package energy in microjoules at
// /sys/class/powercap/intel-rapl:<n>/energy_uj (the name is historical; the
// same driver serves AMD packages). Each counter wraps at its
// max_energy_range_uj, and a wrap shows up only as a decrease, so bracketing a
// long run with one read at each end loses a whole range per wrap without any
// sign that it happened. A poller therefore reads the counters ten times a
// second and accumulates the unwrapped differences, which keeps every interval
// several orders of magnitude below one range. Package energy covers
// everything on the socket, not just this process. rapl_zones() returning 0
// means the counters are absent or not readable here.
// ---------------------------------------------------------------------------

#define RAPL_MAX_PKGS 8
#define RAPL_POLL_MS  100
static int           g_rapl_pkgs = 0;
static long long     g_rapl_range[RAPL_MAX_PKGS];
static long long     g_rapl_last[RAPL_MAX_PKGS];
static long long     g_rapl_total_uj = 0;
static volatile bool g_rapl_running = false;
static pthread_t     g_rapl_thread;

static bool rapl_read_pkg(int i, const char *file, long long *out)
{
    char path[96];
    snprintf(path, sizeof path, "/sys/class/powercap/intel-rapl:%d/%s", i, file);
    FILE *f = fopen(path, "r");
    if (f == NULL) return false;
    const bool ok = (fscanf(f, "%lld", out) == 1);
    fclose(f);
    return ok;
}

// Counts the readable packages once, caching each one's counter range.
static int rapl_zones(void)
{
    long long v;
    g_rapl_pkgs = 0;
    while (g_rapl_pkgs < RAPL_MAX_PKGS &&
           rapl_read_pkg(g_rapl_pkgs, "energy_uj", &v) &&
           rapl_read_pkg(g_rapl_pkgs, "max_energy_range_uj",
                         &g_rapl_range[g_rapl_pkgs]))
        g_rapl_pkgs++;
    return g_rapl_pkgs;
}

// Adds the microjoules each package consumed since it was last read.
static void rapl_accumulate(void)
{
    for (int i = 0; i < g_rapl_pkgs; i++) {
        long long now = g_rapl_last[i];
        if (!rapl_read_pkg(i, "energy_uj", &now)) continue;
        long long d = now - g_rapl_last[i];
        if (d < 0) d += g_rapl_range[i];
        g_rapl_total_uj += d;
        g_rapl_last[i] = now;
    }
}

static void *rapl_polling_func(void *unused)
{
    (void)unused;
    while (g_rapl_running) {
        struct timespec ts;
        ts.tv_sec  = RAPL_POLL_MS / 1000;
        ts.tv_nsec = (RAPL_POLL_MS % 1000) * 1000000L;
        nanosleep(&ts, NULL);
        rapl_accumulate();
    }
    return NULL;
}

static void rapl_begin(void)
{
    for (int i = 0; i < g_rapl_pkgs; i++)
        rapl_read_pkg(i, "energy_uj", &g_rapl_last[i]);
    g_rapl_total_uj = 0;
    g_rapl_running = true;
    if (pthread_create(&g_rapl_thread, NULL, rapl_polling_func, NULL) != 0) {
        fprintf(stderr, "Could not start the RAPL polling thread\n");
        exit(EXIT_FAILURE);
    }
}

// Joules over all packages since rapl_begin, unwrapped per package. The poller
// is stopped and joined first, so the final accumulate races with nothing.
static double rapl_end_joules(void)
{
    if (g_rapl_running) {
        g_rapl_running = false;
        pthread_join(g_rapl_thread, NULL);
    }
    rapl_accumulate();
    return (double)g_rapl_total_uj / 1e6;
}

// Names how the host path runs, for the header and the csv: the OpenMP
// thread count when the build enables it, serial otherwise.
static const char *cpu_desc(void)
{
#ifdef _OPENMP
    static char buf[32];
    snprintf(buf, sizeof buf, "openmp-%d", omp_get_max_threads());
    return buf;
#else
    return "serial";
#endif
}

// ---------------------------------------------------------------------------
// Problem set-up
//
// Signatures and payload are random bytes from a seeded generator, so a run is
// reproducible from its printed settings. Planting a match copies the planted
// signature's text into the payload (literal mode), or copies a regex and its
// matching payload text over the planted slots (regex mode), one pair per
// supported signature length.
//
// The alphabet those bytes come from is a run-time choice rather than a
// constant, because how often two unrelated strings agree by chance is set by
// it: a chance agreement at one position has probability one over the alphabet
// size. Any false positive figure is therefore a statement about the alphabet
// as much as about the detector, so the alphabet is named wherever one is
// reported. The default stays at the 26 letters every timing sweep has used, so
// those remain reproducible, and the detection work varies it deliberately.
//
// Two interactions to keep in mind when comparing across alphabets. The larger
// alphabets contain the three regex metacharacters, so in regex mode a random
// signature can acquire a wildcard and fall under the full-literal-score rule
// instead of the alpha rule. And a signature drawn from the whole byte range can
// contain a zero byte, which nothing here treats as a terminator, because every
// length is carried explicitly.
// ---------------------------------------------------------------------------

typedef enum {
    ALPHABET_LOWER26,    // 'a' to 'z'
    ALPHABET_ASCII95,    // printable ASCII, 0x20 to 0x7e
    ALPHABET_BYTES256    // the whole byte range
} Alphabet;

static Alphabet g_alphabet = ALPHABET_LOWER26;

static const char *alphabet_name(Alphabet a)
{
    switch (a) {
        case ALPHABET_ASCII95:  return "ascii95";
        case ALPHABET_BYTES256: return "bytes256";
        default:                return "lower26";
    }
}

static int alphabet_size(Alphabet a)
{
    switch (a) {
        case ALPHABET_ASCII95:  return 95;
        case ALPHABET_BYTES256: return 256;
        default:                return 26;
    }
}

// Returns false for an unknown name, so the caller reports it rather than
// silently measuring a different alphabet from the one that was asked for.
static bool alphabet_from_name(const char *s, Alphabet *out)
{
    if (!strcmp(s, "lower26"))  { *out = ALPHABET_LOWER26;  return true; }
    if (!strcmp(s, "ascii95"))  { *out = ALPHABET_ASCII95;  return true; }
    if (!strcmp(s, "bytes256")) { *out = ALPHABET_BYTES256; return true; }
    return false;
}

static void fill_random_bytes(char *dst, int n)
{
    switch (g_alphabet) {
        case ALPHABET_ASCII95:
            for (int i = 0; i < n; i++) dst[i] = (char)(rand() % 95 + 0x20);
            break;
        case ALPHABET_BYTES256:
            for (int i = 0; i < n; i++) dst[i] = (char)(rand() % 256);
            break;
        default:
            for (int i = 0; i < n; i++) dst[i] = (char)(rand() % 26 + 'a');
            break;
    }
}

// Each signature reaches its full literal score against its payload text, so
// a planted pair is detectable under the full-literal-score threshold.
static const char REGEX_SIG_16[] = "goo.leM*l.ci*c~m";
static const char REGEX_PAT_16[] = "goosleMaliciou.c1m";
static const char REGEX_SIG_32[] = "This*malware*fro*goo.leM*l.c*c~m";
static const char REGEX_PAT_32[] = "ThisisamalwareobtainedfromgoosleMaliciou.c1m";

static int literal_count(const char *sig, int L)
{
    int n = 0;
    for (int j = 0; j < L; j++)
        if (sig[j] != '*' && sig[j] != '.' && sig[j] != '~') n++;
    return n;
}

// Literal mode: the kernel tests the cell against alpha * L; for integer
// scores that is strict > against floor(alpha * L), which is what is passed
// to the kernel.
static int literal_threshold(int L, double alpha)
{
    return (int)floor(alpha * (double)L);
}

// Regex mode: a signature made only of literals gets alpha of its maximum
// score with integer truncation (count * RE_MATCH * pct / 100, pct = alpha
// as a percentage); a signature containing any wildcard must reach its full
// literal score.
static int regex_threshold(const char *sig, int L, double alpha)
{
    const int count = literal_count(sig, L);
    const int pct   = (int)(alpha * 100.0 + 0.5);
    if (count == L) return count * RE_MATCH * pct / 100;
    return count * RE_MATCH;
}

// ---------------------------------------------------------------------------
// Launch dispatch
// ---------------------------------------------------------------------------

typedef struct {
    int midpoint;
    int payload_len;
    const char *signatures_d;
    uint32_t *rows_d;
    long long row_size;
    const int *thresholds_d;
    int threshold_lit;
    bool exit_first;
    Report *report_d;
} LaunchArgs;

template <int SIG_LEN>
static void launch_sig_len(bool use_dpx, bool rows_reg, bool regex_mode,
                           int grid, int block, const LaunchArgs *a)
{
// The early-exit flag is passed twice, as a template parameter and as a
// run-time argument, from the same value. The kernel chooses which of the two
// its branch reads, for the reason recorded there. This doubles the
// instantiations to sixteen per signature length, which costs compile time.
#define SW_LAUNCH(D, R, X, E)                                                  \
    sw_scan<SIG_LEN, D, R, X, E><<<grid, block>>>(                             \
        a->midpoint, a->payload_len, a->signatures_d, a->rows_d, a->row_size,  \
        a->thresholds_d, a->threshold_lit, a->exit_first, a->report_d)

#define SW_LAUNCH_X(D, R, X)                                                   \
    do {                                                                       \
        if (a->exit_first) SW_LAUNCH(D, R, X, true);                           \
        else               SW_LAUNCH(D, R, X, false);                          \
    } while (0)

    if (use_dpx) {
        if (rows_reg) { if (regex_mode) SW_LAUNCH_X(true, true, true);
                        else            SW_LAUNCH_X(true, true, false); }
        else          { if (regex_mode) SW_LAUNCH_X(true, false, true);
                        else            SW_LAUNCH_X(true, false, false); }
    } else {
        if (rows_reg) { if (regex_mode) SW_LAUNCH_X(false, true, true);
                        else            SW_LAUNCH_X(false, true, false); }
        else          { if (regex_mode) SW_LAUNCH_X(false, false, true);
                        else            SW_LAUNCH_X(false, false, false); }
    }
#undef SW_LAUNCH_X
#undef SW_LAUNCH
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
// Results file
// ---------------------------------------------------------------------------
// A results csv is appended to across many runs, and a binary built from a
// different commit writes a different set of columns. Appending under a header
// that describes the older set leaves every later reader one field out of step,
// and nothing reports it, because a csv carries no statement of how many
// columns a row should have: the file still parses, and the numbers land under
// the wrong names. The header that would be written is therefore compared with
// the one already in the file, and a run stops before it starts rather than
// after the work is done.

static const char CSV_HEADER[] =
    "signatures,payload,sig_len,mode,rows,dpx,alpha,"
    "block_threads,grid_blocks,exit,plant,seed,"
    "target,gpu,repeat,trial,kernel_s,endtoend_s,energy_j,"
    "mean_power_w,power_samples,energy_counter_j,"
    "found,report_sig,"
    "report_score,report_pos,mismatches\n";

// True when the file is absent or empty, because the run then writes the header
// itself, and when the header it already holds is the one this program writes.
static bool csv_header_matches(const char *path)
{
    FILE *f = fopen(path, "r");
    if (f == NULL) return true;
    char line[512];
    const char *got = fgets(line, sizeof line, f);
    fclose(f);
    if (got == NULL) return true;

    char want[512];
    snprintf(want, sizeof want, "%s", CSV_HEADER);
    line[strcspn(line, "\r\n")] = '\0';
    want[strcspn(want, "\r\n")] = '\0';
    if (strcmp(line, want) == 0) return true;

    fprintf(stderr, "\nThe csv already exists and its header is not the one this"
                    " program writes,\nso appending to it would put every new row"
                    " out of step with the header\nthat is supposed to describe"
                    " it.\n");
    fprintf(stderr, "  file     : %s\n", path);
    fprintf(stderr, "  it has   : %s\n", line);
    fprintf(stderr, "  we write : %s\n", want);
    fprintf(stderr, "Point --csv at a new file, or move this one aside and let"
                    " the run recreate it.\n");
    return false;
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

typedef enum { VERIFY_OFF, VERIFY_REPORT, VERIFY_ALL } VerifyMode;

static void print_usage(const char *prog)
{
    printf("\nUsage: %s [options]\n\n", prog);
    printf("  --signatures <int>  number of signatures, even (default 20000000)\n");
    printf("  --payload <int>     payload length in bytes (default 512, max %d)\n",
           MAX_PAYLOAD);
    printf("  --sig-len <int>     signature length, %d or %d (default %d)\n",
           SIG_LEN_A, SIG_LEN_B, SIG_LEN_A);
    printf("  --mode <name>       literal | regex (default literal)\n");
    printf("  --rows <where>      registers | global: where the DP rows live\n");
    printf("                      (default registers)\n");
    printf("  --dpx <state>       on | off (default on)\n");
    printf("  --alpha <float>     detection threshold fraction, in (0, 1]\n");
    printf("                      (default 0.8): literal detects when a\n");
    printf("                      cell score exceeds alpha * length; a regex\n");
    printf("                      signature with wildcards needs its full\n");
    printf("                      literal score\n");
    printf("  --block <int>       threads per block (default 32)\n");
    printf("  --exit <policy>     first | never: a matching thread stops at its\n");
    printf("                      first report, or scans everything (default first)\n");
    printf("  --plant <int>       make this signature match the payload\n");
    printf("                      (default: none, nothing matches)\n");
    printf("  --trials <int>      measured repetitions (default 1)\n");
    printf("  --warmup <int>      unmeasured repetitions first (default 1)\n");
    printf("  --repeat <int>      scans inside one timed window (default 1).\n");
    printf("                      Raise it when a scan finishes faster than\n");
    printf("                      the energy instruments resolve; every\n");
    printf("                      reported figure is still per scan.\n");
    printf("  --min-window <sec>  choose --repeat from the warm-up scan, so\n");
    printf("                      the window reaches this length (default 0,\n");
    printf("                      off). Needs at least one warm-up.\n");
    printf("  --cpu               run the host reference instead of the GPU\n");
    printf("                      (OpenMP when the build enables it, else serial)\n");
    printf("  --energy            report per-trial energy: NVML power sampling\n");
    printf("                      on the GPU, RAPL package counters with --cpu\n");
    printf("  --poll-ms <int>     NVML sampling interval, ms (default 1)\n");
    printf("  --device <int>      CUDA and NVML device index (default 0)\n");
    printf("  --csv <path>        append one row per trial to this file\n");
    printf("  --power-csv <path>  write every power sample to this file\n");
    printf("  --seed <int>        seed for the generated data (default 1)\n");
    printf("  --alphabet <name>   bytes the generated signatures and payload\n");
    printf("                      are drawn from: lower26 | ascii95 | bytes256\n");
    printf("                      (default lower26, the 26 lowercase letters,\n");
    printf("                      which every timing sweep has used). It sets\n");
    printf("                      how often unrelated strings agree by chance,\n");
    printf("                      so any false positive rate depends on it.\n");
    printf("  --verify <what>     report | all | off: recheck the reported\n");
    printf("                      signature on the host, additionally scan every\n");
    printf("                      signature on the host, or skip (default report)\n");
    printf("  --help              show this message\n\n");
}

int main(int argc, char **argv)
{
    long long N      = 20000000;
    int   P          = 512;
    int   L          = SIG_LEN_A;
    bool  regex_mode = false;
    bool  rows_reg   = true;
    bool  use_dpx    = true;
    double alpha     = 0.8;
    int   block      = 32;
    bool  exit_first = true;
    long long plant  = -1;
    int   trials     = 1;
    int    warmup       = 1;
    int    repeat       = 1;
    double min_window_s = 0.0;
    bool  run_cpu    = false;
    bool  measure_energy = false;
    long  poll_ms    = 1;
    int   device     = 0;
    unsigned int seed = 1;
    VerifyMode verify = VERIFY_REPORT;
    const char *csv_path = NULL;
    const char *power_csv_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            print_usage(argv[0]);
            return 0;
        } else if (!strcmp(argv[i], "--signatures") && i + 1 < argc) {
            N = atoll(argv[++i]);
        } else if (!strcmp(argv[i], "--payload") && i + 1 < argc) {
            P = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--sig-len") && i + 1 < argc) {
            L = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "literal"))    regex_mode = false;
            else if (!strcmp(v, "regex")) regex_mode = true;
            else { fprintf(stderr, "Unknown mode: %s\n", v); return 1; }
        } else if (!strcmp(argv[i], "--rows") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "registers"))   rows_reg = true;
            else if (!strcmp(v, "global")) rows_reg = false;
            else { fprintf(stderr, "Unknown rows placement: %s\n", v); return 1; }
        } else if (!strcmp(argv[i], "--dpx") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "on"))       use_dpx = true;
            else if (!strcmp(v, "off")) use_dpx = false;
            else { fprintf(stderr, "Unknown dpx state: %s\n", v); return 1; }
        } else if (!strcmp(argv[i], "--alpha") && i + 1 < argc) {
            alpha = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--block") && i + 1 < argc) {
            block = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--exit") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "first"))      exit_first = true;
            else if (!strcmp(v, "never")) exit_first = false;
            else { fprintf(stderr, "Unknown exit policy: %s\n", v); return 1; }
        } else if (!strcmp(argv[i], "--plant") && i + 1 < argc) {
            plant = atoll(argv[++i]);
        } else if (!strcmp(argv[i], "--trials") && i + 1 < argc) {
            trials = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) {
            repeat = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--min-window") && i + 1 < argc) {
            min_window_s = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) {
            warmup = atoi(argv[++i]);
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
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--alphabet") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!alphabet_from_name(v, &g_alphabet)) {
                fprintf(stderr, "Unknown alphabet: %s. One of lower26,"
                                " ascii95, bytes256.\n", v);
                return 1;
            }
        } else if (!strcmp(argv[i], "--verify") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "off"))         verify = VERIFY_OFF;
            else if (!strcmp(v, "report")) verify = VERIFY_REPORT;
            else if (!strcmp(v, "all"))    verify = VERIFY_ALL;
            else { fprintf(stderr, "Unknown verify mode: %s\n", v); return 1; }
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (N < 2 || N % 2 != 0) {
        fprintf(stderr, "--signatures must be even and at least 2 (one thread"
                        " scores two signatures)\n");
        return 1;
    }
    if (P < 1 || P > MAX_PAYLOAD) {
        fprintf(stderr, "--payload must be between 1 and %d\n", MAX_PAYLOAD);
        return 1;
    }
    if (L != SIG_LEN_A && L != SIG_LEN_B) {
        fprintf(stderr, "--sig-len must be %d or %d (compile-time bounds keep"
                        " the DP rows in registers)\n", SIG_LEN_A, SIG_LEN_B);
        return 1;
    }
    if (P < L) { fprintf(stderr, "--payload must be at least --sig-len\n"); return 1; }
    if (alpha <= 0.0 || alpha > 1.0) {
        fprintf(stderr, "--alpha must be in (0, 1]\n");
        return 1;
    }
    if (block < 1 || block > 1024) {
        fprintf(stderr, "--block must be between 1 and 1024\n");
        return 1;
    }
    if (plant >= N) {
        fprintf(stderr, "--plant must be below --signatures\n");
        return 1;
    }
    if (trials < 1) { fprintf(stderr, "--trials must be at least 1\n"); return 1; }
    if (warmup < 0) { fprintf(stderr, "--warmup cannot be negative\n"); return 1; }
    if (repeat < 1) { fprintf(stderr, "--repeat must be at least 1\n"); return 1; }
    if (min_window_s < 0.0) {
        fprintf(stderr, "--min-window cannot be negative\n");
        return 1;
    }
    if (min_window_s > 0.0 && warmup < 1) {
        fprintf(stderr, "--min-window needs at least one warm-up: the repeat"
                        " count is read off\nthe warm-up scan rather than"
                        " guessed.\n");
        return 1;
    }
    // Repeating inside the window is only the same work repeated when each
    // launch starts from the state the one before it started from. With the DP
    // rows in registers every launch zeroes its own, so it does. With the rows
    // in global memory a launch reads what the previous one left behind, and
    // the memset that clears them sits outside the timed window where it
    // belongs, so the second scan onward would not be the scan being measured.
    if ((repeat > 1 || min_window_s > 0.0) && !rows_reg) {
        fprintf(stderr, "--repeat above 1 needs --rows registers: with the rows"
                        " in global memory\neach scan would read what the one"
                        " before it left behind.\n");
        return 1;
    }
    // The CPU path reads its counters at the edges of the window, so it has no
    // shortest usable run and nothing to gain here.
    if ((repeat > 1 || min_window_s > 0.0) && run_cpu) {
        fprintf(stderr, "--repeat above 1 does not apply to --cpu: RAPL is read"
                        " at the window\nedges and resolves a run of any"
                        " length.\n");
        return 1;
    }
    // Checked here, with the rest of the settings, so that a mismatch costs
    // nothing: the run has not started and no measurement is lost.
    if (csv_path != NULL && !csv_header_matches(csv_path)) return 1;

    const int midpoint = (int)(N / 2);
    const size_t slot = (size_t)L + 1;               // L characters plus a NUL
    const size_t sig_bytes = (size_t)N * slot;

    // ------------------------------------------------------------------
    // Generate the data
    // ------------------------------------------------------------------

    srand(seed);

    char *signatures = (char *)malloc(sig_bytes);
    char *payload    = (char *)malloc((size_t)P);
    if (signatures == NULL || payload == NULL) {
        fprintf(stderr, "Host allocation of %.2f GB failed\n", sig_bytes / 1e9);
        return 1;
    }
    for (long long k = 0; k < N; k++) {
        fill_random_bytes(signatures + (size_t)k * slot, L);
        signatures[(size_t)k * slot + L] = '\0';
    }
    fill_random_bytes(payload, P);

    if (plant >= 0) {
        char *slot_p = signatures + (size_t)plant * slot;
        if (regex_mode) {
            const char *sig_text = (L == SIG_LEN_B) ? REGEX_SIG_32 : REGEX_SIG_16;
            const char *pat_text = (L == SIG_LEN_B) ? REGEX_PAT_32 : REGEX_PAT_16;
            // The payload text is planted at offset 5. Offset 0 would work now
            // that every byte is scored, but the offset is kept where the
            // measured program put it so a planted run stays comparable with
            // every earlier one.
            if ((int)(5 + strlen(pat_text)) > P) {
                fprintf(stderr, "--payload too short for the planted regex"
                                " payload text (%zu bytes at offset 5)\n",
                        strlen(pat_text));
                return 1;
            }
            memcpy(slot_p, sig_text, (size_t)L);     // both texts are exactly L
            memcpy(payload + 5, pat_text, strlen(pat_text));
        } else {
            // The planted signature's text becomes the start of the payload,
            // and every one of its L characters scores, so a perfect plant
            // reaches L. The test is strict against floor(alpha * L), so it is
            // detected at every alpha below 1.0.
            memcpy(payload, slot_p, (size_t)L);
        }
    }

    // ------------------------------------------------------------------
    // Thresholds
    // ------------------------------------------------------------------

    const int threshold_lit = literal_threshold(L, alpha);

    int *thresholds = NULL;
    if (regex_mode) {
        thresholds = (int *)malloc((size_t)N * sizeof(int));
        if (thresholds == NULL) {
            fprintf(stderr, "Threshold allocation failed\n");
            return 1;
        }
        for (long long k = 0; k < N; k++)
            thresholds[k] = regex_threshold(signatures + (size_t)k * slot,
                                            L, alpha);
    }

    // A pinned binary refuses a run whose settings differ from its pins, so
    // it cannot silently measure the wrong configuration.
#ifdef PIN_SIGNATURES
    if (N != (long long)PIN_SIGNATURES) {
        fprintf(stderr, "This binary is pinned to --signatures %lld\n",
                (long long)PIN_SIGNATURES);
        return 1;
    }
#endif
#ifdef PIN_PAYLOAD
    if (P != PIN_PAYLOAD) {
        fprintf(stderr, "This binary is pinned to --payload %d\n", PIN_PAYLOAD);
        return 1;
    }
#endif
#ifdef PIN_THRESHOLD
    if (regex_mode || threshold_lit != PIN_THRESHOLD) {
        fprintf(stderr, "This binary is pinned to literal mode with threshold"
                        " %d\n", PIN_THRESHOLD);
        return 1;
    }
#endif
#ifdef PIN_EXIT_FIRST
    if (exit_first != (PIN_EXIT_FIRST != 0)) {
        fprintf(stderr, "This binary is pinned to --exit %s\n",
                PIN_EXIT_FIRST ? "first" : "never");
        return 1;
    }
#endif

    // The threshold the verifier applies to signature k.
    #define THR_OF(k) (regex_mode ? thresholds[k] : threshold_lit)

    // ------------------------------------------------------------------
    // Host pre-checks, once: the inputs are identical in every trial
    // ------------------------------------------------------------------

    bool plant_crosses = false;
    int plant_score = 0, plant_pos = 0, plant_best = 0;
    if (verify != VERIFY_OFF && plant >= 0) {
        plant_crosses = host_scan_signature(regex_mode, payload, P,
                                            signatures + (size_t)plant * slot, L,
                                            THR_OF(plant), &plant_score,
                                            &plant_pos, &plant_best);
    }

    long long host_crossers = -1;
    if (verify == VERIFY_ALL) {
        host_crossers = 0;
        for (long long k = 0; k < N; k++) {
            int sc, po, be;
            if (host_scan_signature(regex_mode, payload, P,
                                    signatures + (size_t)k * slot, L,
                                    THR_OF(k), &sc, &po, &be))
                host_crossers++;
        }
    }

    // ------------------------------------------------------------------
    // Device set-up
    // ------------------------------------------------------------------

    const int grid = (midpoint + block - 1) / block;
    const long long row_size = (long long)(L + 1) * midpoint;  // words per row block
    const size_t rows_bytes = rows_reg ? 0
                            : (size_t)2 * (size_t)row_size * sizeof(uint32_t);

    char     *signatures_d = NULL;
    int      *thresholds_d = NULL;
    uint32_t *rows_d       = NULL;
    Report   *report_d     = NULL;
    cudaDeviceProp prop;
    double sig_upload_s = 0.0;

    if (!run_cpu) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        CUDA_CHECK(cudaMalloc((void **)&signatures_d, sig_bytes));
        CUDA_CHECK(cudaMalloc((void **)&report_d, sizeof(Report)));
        if (regex_mode)
            CUDA_CHECK(cudaMalloc((void **)&thresholds_d, (size_t)N * sizeof(int)));
        if (!rows_reg)
            CUDA_CHECK(cudaMalloc((void **)&rows_d, rows_bytes));

        const double u0 = now_seconds();
        CUDA_CHECK(cudaMemcpy(signatures_d, signatures, sig_bytes,
                              cudaMemcpyHostToDevice));
        if (regex_mode)
            CUDA_CHECK(cudaMemcpy(thresholds_d, thresholds,
                                  (size_t)N * sizeof(int),
                                  cudaMemcpyHostToDevice));
        sig_upload_s = now_seconds() - u0;
    }

    // ------------------------------------------------------------------
    // Header: every setting that affects a number is printed
    // ------------------------------------------------------------------

    printf("# program           : smith_waterman_dpi\n");
#if defined(PIN_SIGNATURES) || defined(PIN_PAYLOAD) || \
    defined(PIN_THRESHOLD) || defined(PIN_EXIT_FIRST)
    printf("# pinned            : compile-time constants baked into the"
           " kernel\n");
#endif
    printf("# signatures        : %lld\n", N);
    printf("# payload_bytes     : %d\n", P);
    printf("# sig_len           : %d\n", L);
    printf("# mode              : %s\n", regex_mode ? "regex" : "literal");
    printf("# alphabet          : %s (%d symbols; a chance agreement at one"
           " position has probability 1/%d)\n",
           alphabet_name(g_alphabet), alphabet_size(g_alphabet),
           alphabet_size(g_alphabet));
    printf("# alpha             : %g\n", alpha);
    if (regex_mode) {
        // Every random signature is all-literal, so any non-planted slot
        // shows the threshold they all share.
        const long long rnd = (plant == 0) ? 1 : 0;
        if (plant >= 0)
            printf("# threshold         : %d (planted regex), %d (random signatures)\n",
                   thresholds[plant], thresholds[rnd]);
        else
            printf("# threshold         : %d (random signatures)\n",
                   thresholds[rnd]);
    } else {
        printf("# threshold         : score > %d, of a maximum %d\n",
               threshold_lit, L);
    }
    printf("# cell_updates      : %.0f per trial (every payload byte, 0..%d)\n",
           (double)N * (double)P * (double)L, P - 1);
    printf("# target            : %s\n", run_cpu ? "cpu" : "gpu");
    if (run_cpu)
        printf("# cpu_run           : %s\n", cpu_desc());
    if (!run_cpu) {
        printf("# gpu               : %s\n", prop.name);
        printf("# compute_capability: %d.%d\n", prop.major, prop.minor);
        printf("# rows              : %s\n",
               rows_reg ? "registers" : "global memory");
        if (!rows_reg)
            printf("# rows_bytes        : %zu\n", rows_bytes);
        printf("# dpx               : %s\n", use_dpx ? "on" : "off");
        printf("# block_threads     : %d\n", block);
        printf("# grid_blocks       : %d (one thread per two signatures)\n", grid);
        printf("# signature_bytes   : %zu\n", sig_bytes);
        printf("# signature_upload_s: %.6f (one-time set-up, outside both windows)\n",
               sig_upload_s);
    }
    printf("# exit              : %s\n",
           exit_first ? "a matching thread stops at its first report"
                      : "every thread scans everything");
    if (plant >= 0) printf("# plant             : signature %lld\n", plant);
    else            printf("# plant             : none\n");
    printf("# seed              : %u\n", seed);
    printf("# trials            : %d\n", trials);
    printf("# warmup            : %d (not reported)\n", warmup);
    if (min_window_s > 0.0)
        printf("# repeat            : chosen after the warm-up, to fill"
               " %.3f s\n", min_window_s);
    else
        printf("# repeat            : %d scan%s per timed window%s\n",
               repeat, repeat == 1 ? "" : "s",
               repeat == 1 ? "" : "; every figure below is per scan");
    printf("# verify            : %s\n",
           verify == VERIFY_OFF ? "off" :
           verify == VERIFY_REPORT ? "the reported signature, on the host"
                                   : "every signature, on the host");
    if (verify != VERIFY_OFF && plant >= 0) {
        if (plant_crosses)
            printf("# plant_host        : crosses at score %d, payload index %d\n",
                   plant_score, plant_pos);
        else
            printf("# plant_host        : does not cross (best score %d); the"
                   " planted text does not reach its threshold\n", plant_best);
    }
    if (host_crossers >= 0)
        printf("# host_crossers     : %lld of %lld signatures cross\n",
               host_crossers, N);
    printf("# kernel_window     : the %s only\n",
           run_cpu ? "host scan" : "kernel launch");
    if (run_cpu)
        printf("# endtoend_window   : the host scan\n");
    else
        printf("# endtoend_window   : payload upload, the kernel launch,"
               " report copy back\n");

    if (measure_energy && !run_cpu) {
        printf("# poll_interval_ms  : %ld\n", poll_ms);
        power_start(device, poll_ms);
        struct timespec settle = { 0, 100 * 1000000L };
        nanosleep(&settle, NULL);
    }
    if (measure_energy && run_cpu) {
        if (rapl_zones() == 0) {
            fprintf(stderr, "--cpu --energy needs readable RAPL counters at"
                            " /sys/class/powercap/intel-rapl:*/energy_uj\n");
            return 1;
        }
        printf("# energy_source     : rapl, %d package(s), read at the window"
               " edges\n", g_rapl_pkgs);
    }

    // ------------------------------------------------------------------
    // Trials
    // ------------------------------------------------------------------

    double *kernel_s   = (double *)calloc(trials, sizeof(double));
    double *endtoend_s = (double *)calloc(trials, sizeof(double));
    double *energy_j   = (double *)calloc(trials, sizeof(double));
    double *power_w    = (double *)calloc(trials, sizeof(double));
    bool   *energy_ok  = (bool *)calloc(trials, sizeof(bool));
    double *energy_ctr_j  = (double *)calloc(trials, sizeof(double));
    bool   *energy_ctr_ok = (bool *)calloc(trials, sizeof(bool));

    cudaEvent_t ev_start, ev_stop;
    if (!run_cpu) {
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
    }

    LaunchArgs args;
    args.midpoint      = midpoint;
    args.payload_len   = P;
    args.signatures_d  = signatures_d;
    args.rows_d        = rows_d;
    args.row_size      = row_size;
    args.thresholds_d  = thresholds_d;
    args.threshold_lit = threshold_lit;
    args.exit_first    = exit_first;
    args.report_d      = report_d;

    printf("trial,kernel_s,endtoend_s");
    if (measure_energy) printf(",energy_j,mean_power_w,power_samples,energy_counter_j");
    printf(",found,report_sig,report_score,report_pos");
    if (verify != VERIFY_OFF) printf(",mismatches");
    printf("\n");

    for (int t = -warmup; t < trials; t++) {
        const bool measured = (t >= 0);
        Report rep = { 0, 0, 0, 0 };
        unsigned long long ctr0 = 0, ctr1 = 0;
        bool ctr_ok = false;
        double t0 = 0.0, t1 = 0.0, kernel_seconds = 0.0;
        double trial_joules = 0.0;

        if (run_cpu) {
            if (measure_energy) rapl_begin();
            // The signatures are independent, so the loop is spread across
            // OpenMP threads when the build enables it; without OpenMP the
            // pragma is ignored and the loop runs serially. The first
            // crossing to reach the critical section claims the report, as
            // the kernel's atomic compare-and-swap does, so which signature
            // claims may differ between runs, and the claimed signature
            // must still reproduce exactly under the verifier below.
            volatile bool stop = false;
            t0 = now_seconds();
            #pragma omp parallel for schedule(static)
            for (long long k = 0; k < N; k++) {
                if (stop) continue;
                int sc, po, be;
                if (host_scan_signature(regex_mode, payload, P,
                                        signatures + (size_t)k * slot, L,
                                        THR_OF(k), &sc, &po, &be)) {
                    #pragma omp critical
                    if (rep.claimed == 0) {
                        rep.claimed = 1;
                        rep.score = sc;
                        rep.sig = (int)k;
                        rep.pos = po;
                    }
                    if (exit_first) stop = true;
                }
            }
            t1 = now_seconds();
            kernel_seconds = t1 - t0;
            if (measure_energy) trial_joules = rapl_end_joules();
        } else {
            // Resetting the report and the row buffer is state clean-up
            // between trials, outside both windows.
            CUDA_CHECK(cudaMemset(report_d, 0, sizeof(Report)));
            if (!rows_reg) CUDA_CHECK(cudaMemset(rows_d, 0, rows_bytes));
            CUDA_CHECK(cudaDeviceSynchronize());

            const bool ctr_started = measure_energy && energy_counter_mj(&ctr0);
            t0 = now_seconds();
            CUDA_CHECK(cudaMemcpyToSymbol(c_payload, payload, (size_t)P));

            CUDA_CHECK(cudaEventRecord(ev_start, 0));
            // The scan does not modify its inputs and, with the rows held in
            // registers, each launch zeroes its own state, so running it
            // repeatedly inside one window is the same work done repeatedly.
            // That is the point: a scan too short for the energy instruments
            // to resolve becomes a window they can, and the per-scan figures
            // below divide back out. The repeat count is 1 unless asked for.
            for (int rep_i = 0; rep_i < repeat; rep_i++) {
                if (L == SIG_LEN_A)
                    launch_sig_len<SIG_LEN_A>(use_dpx, rows_reg, regex_mode,
                                              grid, block, &args);
                else
                    launch_sig_len<SIG_LEN_B>(use_dpx, rows_reg, regex_mode,
                                              grid, block, &args);
            }
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaEventRecord(ev_stop, 0));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            kernel_seconds = ms / 1000.0;

            CUDA_CHECK(cudaMemcpy(&rep, report_d, sizeof(Report),
                                  cudaMemcpyDeviceToHost));
            t1 = now_seconds();
            if (ctr_started) ctr_ok = energy_counter_mj(&ctr1);
        }

        // Verification. The reported crossing is deterministic per signature,
        // so the host reference must reproduce its score and position; which
        // signature wins the claim may differ between trials.
        long long bad = 0;
        if (verify != VERIFY_OFF) {
            if (rep.claimed) {
                int sc, po, be;
                const bool crossed =
                    host_scan_signature(regex_mode, payload, P,
                                        signatures + (size_t)rep.sig * slot, L,
                                        THR_OF(rep.sig), &sc, &po, &be);
                if (!crossed || sc != rep.score || po != rep.pos) bad++;
            }
            if (plant >= 0 && plant_crosses && !rep.claimed) bad++;  // a miss
            if (host_crossers >= 0 &&
                (rep.claimed != 0) != (host_crossers > 0)) bad++;
        }

        // The repeat count is read off the last warm-up rather than chosen: it
        // is whatever this card, at this size, needs to fill the requested
        // window. A scan already long enough keeps a count of one. Doing it
        // here means every measured trial uses the same count, and the count
        // itself is never a number somebody typed.
        if (!measured && t == -1 && min_window_s > 0.0) {
            const double one = kernel_seconds / (double)repeat;
            if (one > 0.0 && one * repeat < min_window_s) {
                repeat = (int)ceil(min_window_s / one);
                printf("# repeat            : %d, to fill %.3f s from a warm-up"
                       " scan of %.6f s\n", repeat, min_window_s, one);
                fflush(stdout);
            }
        }

        if (!measured) continue;

        // Everything below is per scan, so a repeated window reports the same
        // quantities a single scan would and stays comparable with every row
        // ever written. Power is not divided: it is a rate, and the rate over
        // the window is what the device drew. The repeat column records how
        // many scans the window held, so its length is recoverable.
        const double per = (double)repeat;
        kernel_s[t]   = kernel_seconds / per;
        endtoend_s[t] = (t1 - t0) / per;

        int nsamp = 0;
        if (measure_energy && !run_cpu) {
            energy_ok[t] = power_window(t0, t1, &energy_j[t], &power_w[t], &nsamp);
            if (energy_ok[t]) energy_j[t] /= per;
        }
        if (measure_energy && run_cpu) {
            energy_j[t]  = trial_joules / per;
            power_w[t]   = trial_joules / (t1 - t0);
            energy_ok[t] = true;
        }
        if (ctr_ok) {
            energy_ctr_j[t]  = (double)(ctr1 - ctr0) / 1000.0 / per;
            energy_ctr_ok[t] = true;
        }

        printf("%d,%.6f,%.6f", t + 1, kernel_s[t], endtoend_s[t]);
        if (measure_energy) {
            if (energy_ok[t]) printf(",%.3f,%.3f,%d", energy_j[t], power_w[t], nsamp);
            else              printf(",,,%d", nsamp);
            if (energy_ctr_ok[t]) printf(",%.3f", energy_ctr_j[t]);
            else                  printf(",");
        }
        if (rep.claimed) printf(",1,%d,%d,%d", rep.sig, rep.score, rep.pos);
        else             printf(",0,,,");
        if (verify != VERIFY_OFF) printf(",%lld", bad);
        printf("\n");
        fflush(stdout);

        if (csv_path != NULL) {
            FILE *f = fopen(csv_path, "a");
            if (f == NULL) {
                perror("csv");
            } else {
                fseek(f, 0, SEEK_END);
                if (ftell(f) == 0) fputs(CSV_HEADER, f);
                fprintf(f, "%lld,%d,%d,%s,%s,%s,%g,%d,%d,%s,%lld,%u,%s,%s,%d,%d,"
                           "%.6f,%.6f,",
                        N, P, L, regex_mode ? "regex" : "literal",
                        run_cpu ? "n/a" : (rows_reg ? "registers" : "global"),
                        run_cpu ? "n/a" : (use_dpx ? "on" : "off"),
                        alpha, run_cpu ? 0 : block, run_cpu ? 0 : grid,
                        exit_first ? "first" : "never", plant, seed,
                        run_cpu ? "cpu" : "gpu",
                        run_cpu ? cpu_desc() : prop.name,
                        repeat, t + 1, kernel_s[t], endtoend_s[t]);
                if (measure_energy && energy_ok[t])
                    fprintf(f, "%.3f,%.3f,%d,", energy_j[t], power_w[t], nsamp);
                else
                    fprintf(f, ",,%d,", nsamp);
                if (energy_ctr_ok[t]) fprintf(f, "%.3f,", energy_ctr_j[t]);
                else                  fprintf(f, ",");
                if (rep.claimed)
                    fprintf(f, "1,%d,%d,%d,", rep.sig, rep.score, rep.pos);
                else
                    fprintf(f, "0,,,,");
                if (verify != VERIFY_OFF) fprintf(f, "%lld\n", bad);
                else                      fprintf(f, "\n");
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

    if (measure_energy) {
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

        int cgood = 0;
        double csum = 0.0;
        for (int t = 0; t < trials; t++)
            if (energy_ctr_ok[t]) { csum += energy_ctr_j[t]; cgood++; }
        if (cgood >= 1) {
            double cmean = csum / cgood;
            double *tmp = (double *)calloc(cgood, sizeof(double));
            int m = 0;
            for (int t = 0; t < trials; t++)
                if (energy_ctr_ok[t]) tmp[m++] = energy_ctr_j[t];
            printf("# energy_ctr mean %.3f  std %.3f  over %d of %d trials\n",
                   cmean, stddev_of(tmp, cgood, cmean), cgood, trials);
            free(tmp);
        } else if (!run_cpu) {
            printf("# energy_ctr not available: this driver does not expose "
                   "nvmlDeviceGetTotalEnergyConsumption\n");
        }
    }

    if (!run_cpu) {
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
        CUDA_CHECK(cudaFree(signatures_d));
        CUDA_CHECK(cudaFree(report_d));
        if (thresholds_d != NULL) CUDA_CHECK(cudaFree(thresholds_d));
        if (rows_d != NULL)       CUDA_CHECK(cudaFree(rows_d));
    }
    free(signatures);
    free(payload);
    free(thresholds);
    free(kernel_s);
    free(endtoend_s);
    free(energy_ctr_j);
    free(energy_ctr_ok);
    free(energy_j);
    free(power_w);
    free(energy_ok);
    free(g_samples);
    return 0;
}
