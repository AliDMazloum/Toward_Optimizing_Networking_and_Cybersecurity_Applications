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
//   --dpx  on | off              the __vimax3_s16x2_relu instruction, or the
//                                same per-halfword computation without it
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
// Literal mode adds 1 per matching character, so a signature's maximum score
// is its length and the threshold is a fraction of that length. Regex mode
// uses the larger match reward so that wildcard positions, which contribute
// 0, still leave literal matches room to dominate. The >1, >3 and >4 guards
// keep penalties from acting before an alignment has started.
// ---------------------------------------------------------------------------

#define LIT_MATCH     1   // literal mode: score added per matching character
#define LIT_MISMATCH -2   // literal mode: penalty per mismatch, once started

#define RE_MATCH      6   // regex mode: score per matching literal character
#define RE_MISMATCH  -3   // regex mode: mismatch penalty, once started
#define RE_INDEL     -2   // regex mode: gap penalty, suppressed by *

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
// The recurrence
//
// For payload p (length P) and signature s (length L), with H[0][*] and
// H[*][0] fixed at 0:
//
//   base    = max(H[i-1][j], H[i-1][j-1], H[i][j-1], 0)
//   H[i][j] = base + score(base, p[i-1], s[j-1])
//
// Three details of the recurrence:
//
//   - scanning starts at the second payload character, so payload byte 0
//     never enters the recurrence,
//   - literal mode tests the BASE value (the max of the three neighbours)
//     against its threshold with strict greater-than,
//   - regex mode tests the newly computed cell with greater-or-equal against
//     a per-signature threshold.
//
// The score step is shared, host and device, so the host reference and the
// kernels cannot drift apart, and the only thing --dpx changes is whether
// the base is computed by the DPX instruction or by the plain equivalent.
// ---------------------------------------------------------------------------

__host__ __device__ __forceinline__ bool is_ascii_digit(char c)
{
    return c >= '0' && c <= '9';
}

__host__ __device__ __forceinline__ int step_literal(int base, char p, char s)
{
    if (p == s) return base + LIT_MATCH;
    return base > 1 ? base + LIT_MISMATCH : base;
}

// Regex scoring: '*' contributes 0 and suppresses the gap penalty, '.' matches
// any one character contributing 0 (the gap penalty still applies), and '~'
// matches any one digit contributing 0.
__host__ __device__ __forceinline__ int step_regex(int base, char p, char s,
                                                   bool p_is_digit)
{
    const bool star  = (s == '*');
    const bool any1  = (s == '.');
    const bool digit = (s == '~');

    int t = base;
    if (base > 3 && p != s && !star) t += RE_INDEL;
    if (!star && !any1 && !digit) {
        if (p == s)        t += RE_MATCH;
        else if (base > 4) t += RE_MISMATCH;
    }
    if (digit && base > 4 && !p_is_digit) t += RE_MISMATCH;
    return t;
}

__host__ __device__ __forceinline__ int max3_relu(int a, int b, int c)
{
    int m = a > b ? a : b;
    if (c > m) m = c;
    return m > 0 ? m : 0;
}

// The packed base: max of three values and 0, per signed 16-bit halfword. The
// DPX arm is one instruction on compute capability 9.0; the plain arm is what
// --dpx off measures on the same chip.
template <bool USE_DPX>
__device__ __forceinline__ uint32_t base_packed(uint32_t n, uint32_t nw, uint32_t w)
{
    if (USE_DPX) return __vimax3_s16x2_relu(n, nw, w);
    const int lo = max3_relu((int)(int16_t)(n  & 0xFFFFu),
                             (int)(int16_t)(nw & 0xFFFFu),
                             (int)(int16_t)(w  & 0xFFFFu));
    const int hi = max3_relu((int)(int16_t)(n  >> 16),
                             (int)(int16_t)(nw >> 16),
                             (int)(int16_t)(w  >> 16));
    return (uint32_t)(uint16_t)lo | ((uint32_t)(uint16_t)hi << 16);
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

template <int SIG_LEN, bool USE_DPX, bool ROWS_REG, bool REGEX>
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

    // i starts at 2: payload byte 0 is never scored.
    for (int i = 2; i <= payload_len; i++) {
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

            const uint32_t base2 = base_packed<USE_DPX>(n, nw, w);
            const int b0 = (int)(int16_t)(base2 & 0xFFFFu);
            const int b1 = (int)(int16_t)(base2 >> 16);

            const char s0 = ROWS_REG ? sig0[j - 1] : signatures[off0 + j - 1];
            const char s1 = ROWS_REG ? sig1[j - 1] : signatures[off1 + j - 1];

            const int t0 = REGEX ? step_regex(b0, p, s0, pdig)
                                 : step_literal(b0, p, s0);
            const int t1 = REGEX ? step_regex(b1, p, s1, pdig)
                                 : step_literal(b1, p, s1);

            const uint32_t packed =
                (uint32_t)(uint16_t)t0 | ((uint32_t)(uint16_t)t1 << 16);
            if (ROWS_REG) {
                if (even) rowB[j] = packed; else rowA[j] = packed;
            } else {
                rows[cur + (long long)j * midpoint + gid] = packed;
            }

            // Detection, per mode: regex tests the new cell with >=, literal
            // tests the base with strict > and reports the base.
            if (REGEX) {
                if (t0 >= thr0) {
                    claim_report(report, t0, gid, i - 1);
                    if (exit_first) return;
                } else if (t1 >= thr1) {
                    claim_report(report, t1, gid + midpoint, i - 1);
                    if (exit_first) return;
                }
            } else {
                if (b0 > thr0) {
                    claim_report(report, b0, gid, i - 1);
                    if (exit_first) return;
                } else if (b1 > thr1) {
                    claim_report(report, b1, gid + midpoint, i - 1);
                    if (exit_first) return;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host reference
//
// The same recurrence on the host, one signature at a time, through the same
// score-step functions the kernel uses, with the same detection semantics:
// payload byte 0 unscored, literal tested on the base with strict >, regex
// tested on the new cell with >=. Returns true at the first threshold
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

    for (int i = 2; i <= P; i++) {
        const char p = payload[i - 1];
        const bool pdig = is_ascii_digit(p);
        cur[0] = 0;
        for (int j = 1; j <= L; j++) {
            const int base = max3_relu(prev[j], prev[j - 1], cur[j - 1]);
            const int t = regex_mode ? step_regex(base, p, sig[j - 1], pdig)
                                     : step_literal(base, p, sig[j - 1]);
            cur[j] = t;
            if (t > best) best = t;
            if (regex_mode ? (t >= thr) : (base > thr)) {
                *score_out = regex_mode ? t : base;
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
// Signatures and payload are random lowercase letters from a seeded generator,
// so a run is reproducible from its printed settings. Planting a match copies
// the planted signature's text into the payload (literal mode), or copies a
// regex and its matching payload text over the planted slots (regex mode),
// one pair per supported signature length.
// ---------------------------------------------------------------------------

static void fill_random_lowercase(char *dst, int n)
{
    for (int i = 0; i < n; i++) dst[i] = (char)(rand() % 26 + 'a');
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

// Literal mode: the kernel tests the base against alpha * L; for integer
// bases that is strict > against floor(alpha * L), which is what is passed
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
#define SW_LAUNCH(D, R, X)                                                     \
    sw_scan<SIG_LEN, D, R, X><<<grid, block>>>(                                \
        a->midpoint, a->payload_len, a->signatures_d, a->rows_d, a->row_size,  \
        a->thresholds_d, a->threshold_lit, a->exit_first, a->report_d)

    if (use_dpx) {
        if (rows_reg) { if (regex_mode) SW_LAUNCH(true, true, true);
                        else            SW_LAUNCH(true, true, false); }
        else          { if (regex_mode) SW_LAUNCH(true, false, true);
                        else            SW_LAUNCH(true, false, false); }
    } else {
        if (rows_reg) { if (regex_mode) SW_LAUNCH(false, true, true);
                        else            SW_LAUNCH(false, true, false); }
        else          { if (regex_mode) SW_LAUNCH(false, false, true);
                        else            SW_LAUNCH(false, false, false); }
    }
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
    printf("                      base score exceeds alpha * length; a regex\n");
    printf("                      signature with wildcards needs its full\n");
    printf("                      literal score\n");
    printf("  --block <int>       threads per block (default 32)\n");
    printf("  --exit <policy>     first | never: a matching thread stops at its\n");
    printf("                      first report, or scans everything (default first)\n");
    printf("  --plant <int>       make this signature match the payload\n");
    printf("                      (default: none, nothing matches)\n");
    printf("  --trials <int>      measured repetitions (default 1)\n");
    printf("  --warmup <int>      unmeasured repetitions first (default 1)\n");
    printf("  --cpu               run the host reference instead of the GPU\n");
    printf("                      (OpenMP when the build enables it, else serial)\n");
    printf("  --energy            sample GPU power with NVML and report energy\n");
    printf("  --poll-ms <int>     NVML sampling interval, ms (default 1)\n");
    printf("  --device <int>      CUDA and NVML device index (default 0)\n");
    printf("  --csv <path>        append one row per trial to this file\n");
    printf("  --power-csv <path>  write every power sample to this file\n");
    printf("  --seed <int>        seed for the generated data (default 1)\n");
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
    int   warmup     = 1;
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
        fill_random_lowercase(signatures + (size_t)k * slot, L);
        signatures[(size_t)k * slot + L] = '\0';
    }
    fill_random_lowercase(payload, P);

    if (plant >= 0) {
        char *slot_p = signatures + (size_t)plant * slot;
        if (regex_mode) {
            const char *sig_text = (L == SIG_LEN_B) ? REGEX_SIG_32 : REGEX_SIG_16;
            const char *pat_text = (L == SIG_LEN_B) ? REGEX_PAT_32 : REGEX_PAT_16;
            // The payload text is planted at offset 5: payload byte 0 is
            // never scored, so offset 0 would lose the pattern's first
            // character.
            if ((int)(5 + strlen(pat_text)) > P) {
                fprintf(stderr, "--payload too short for the planted regex"
                                " payload text (%zu bytes at offset 5)\n",
                        strlen(pat_text));
                return 1;
            }
            memcpy(slot_p, sig_text, (size_t)L);     // both texts are exactly L
            memcpy(payload + 5, pat_text, strlen(pat_text));
        } else {
            // The planted signature's text becomes the start of the payload.
            // Payload byte 0 is never scored, so the usable match is L - 1
            // characters and the plant is detected for alpha up to about
            // (L - 2) / L.
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
        printf("# threshold         : base score > %d, of a maximum %d\n",
               threshold_lit, L);
    }
    printf("# cell_updates      : %.0f per trial (payload bytes 1..%d;"
           " byte 0 is never scored)\n",
           (double)N * (double)(P - 1) * (double)L, P - 1);
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
        power_start((unsigned int)device, poll_ms);
        struct timespec settle = { 0, 100 * 1000000L };
        nanosleep(&settle, NULL);
    }

    // ------------------------------------------------------------------
    // Trials
    // ------------------------------------------------------------------

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
    if (measure_energy && !run_cpu) printf(",energy_j,mean_power_w,power_samples");
    printf(",found,report_sig,report_score,report_pos");
    if (verify != VERIFY_OFF) printf(",mismatches");
    printf("\n");

    for (int t = -warmup; t < trials; t++) {
        const bool measured = (t >= 0);
        Report rep = { 0, 0, 0, 0 };
        double t0 = 0.0, t1 = 0.0, kernel_seconds = 0.0;

        if (run_cpu) {
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
        } else {
            // Resetting the report and the row buffer is state clean-up
            // between trials, outside both windows.
            CUDA_CHECK(cudaMemset(report_d, 0, sizeof(Report)));
            if (!rows_reg) CUDA_CHECK(cudaMemset(rows_d, 0, rows_bytes));
            CUDA_CHECK(cudaDeviceSynchronize());

            t0 = now_seconds();
            CUDA_CHECK(cudaMemcpyToSymbol(c_payload, payload, (size_t)P));

            CUDA_CHECK(cudaEventRecord(ev_start, 0));
            if (L == SIG_LEN_A)
                launch_sig_len<SIG_LEN_A>(use_dpx, rows_reg, regex_mode,
                                          grid, block, &args);
            else
                launch_sig_len<SIG_LEN_B>(use_dpx, rows_reg, regex_mode,
                                          grid, block, &args);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaEventRecord(ev_stop, 0));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            kernel_seconds = ms / 1000.0;

            CUDA_CHECK(cudaMemcpy(&rep, report_d, sizeof(Report),
                                  cudaMemcpyDeviceToHost));
            t1 = now_seconds();
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
                if (ftell(f) == 0)
                    fprintf(f, "signatures,payload,sig_len,mode,rows,dpx,alpha,"
                               "block_threads,grid_blocks,exit,plant,seed,"
                               "target,gpu,trial,kernel_s,endtoend_s,energy_j,"
                               "mean_power_w,power_samples,found,report_sig,"
                               "report_score,report_pos,mismatches\n");
                fprintf(f, "%lld,%d,%d,%s,%s,%s,%g,%d,%d,%s,%lld,%u,%s,%s,%d,"
                           "%.6f,%.6f,",
                        N, P, L, regex_mode ? "regex" : "literal",
                        run_cpu ? "n/a" : (rows_reg ? "registers" : "global"),
                        run_cpu ? "n/a" : (use_dpx ? "on" : "off"),
                        alpha, run_cpu ? 0 : block, run_cpu ? 0 : grid,
                        exit_first ? "first" : "never", plant, seed,
                        run_cpu ? "cpu" : "gpu",
                        run_cpu ? cpu_desc() : prop.name,
                        t + 1, kernel_s[t], endtoend_s[t]);
                if (measure_energy && !run_cpu && energy_ok[t])
                    fprintf(f, "%.3f,%.3f,%d,", energy_j[t], power_w[t], nsamp);
                else
                    fprintf(f, ",,%d,", nsamp);
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
    free(energy_j);
    free(power_w);
    free(energy_ok);
    free(g_samples);
    return 0;
}
