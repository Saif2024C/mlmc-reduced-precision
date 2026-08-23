////////////////////////////////////////////////////////////////////////////
// Level-0 kernel benchmark: SCALAR LOOKBACK, fp32 vs fp16+Kahan.
//
// Performance_analysis.txt item 1d -- the real l=0 production kernel in
// isolation, NOT the MLMC driver.  Companion to kernel_bench_basket_l0.cpp
// (item 1a, basket European), chosen as the second kernel because it is the
// most DIFFERENT from that baseline: it adds an exact Brownian-bridge minimum
// (an extra log-uniform draw and a sqrt per step) on top of the same
// RNG -> Milstein spine.  If the fp16 advantage degrades anywhere, here is
// where it should show.
//
//   P = e^{-rT} ( X_T - M ),    M = running minimum over the path
//
// ONE DELIBERATE DEVIATION FROM THE PRODUCTION KERNEL.
// nested_scalar_milstein_fp16_avx512.cpp's l==0 branch serves BOTH scalar
// payoffs from one code path: it computes the Asian time-integral Af AND the
// Lookback minimum Mf unconditionally, then selects between them in the final
// pay32()/pay16() call on the runtime global `option`.  Unlike the basket
// file -- where the Asian-only work sits behind `if (opt == 2)` and European
// genuinely skips it -- there is no guard here, so the shipped Lookback path
// really does compute Af and throw it away.
//
// This benchmark STRIPS Af, to profile a pure Lookback kernel rather than
// Lookback-plus-dead-Asian-work.  Consequence: it is faster than the shipped
// l==0 branch by the cost of one trapezoid, one bridge multiply-add and one
// N(0, h^3/12) draw.  Everything else -- the Milstein step, the bridge
// minimum, the RNG, the lane layout -- is verbatim.  Quote this as the cost of
// the Lookback KERNEL, not as the cost of the level-0 estimator call.
//
// NOTE ON THE fp16 PATH: dW is drawn as native fp16 (normal16_fp16, Giles'
// norminv_fp16 spline).  The log-uniform driving the bridge minimum has NO
// fp16 spline, so it is drawn in fp32 and narrowed -- verbatim from the
// production file.  That is a genuine fp32 dependency inside the fp16
// Lookback kernel, and unlike the basket European kernel (which has no
// log-uniform at all) it is expected to leave residual fp32 work in a profile.
//
// N_0 per eps comes from the driver's own complexity table (adaptive mode,
// outputs/avx512/scalar/nested_scalar_fp16_avx512_adaptive_2.txt), so every
// row is a real workload.  `sink` stops the loop being optimised away; `chk`
// (mean discounted payoff) catches a fast-but-broken kernel -- both precisions
// must price the same option.
//
// BUILD / RUN (mimic, after `source /opt/intel/oneapi/setvars.sh`):
//   make kbench_scalar           # serial sweep over eps
//   make kbench_scalar_scaling   # 1 vs 32 threads (OMP)
//   ./kb_scalar_l0 --kernel fp32 # one kernel, one N: perf/VTune target
////////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <vector>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../my_mm512.h"
#include "../avx512_rng.h"
#include "../options_mm512.h"

//==========================================================================
// 1. MATHEMATICAL OBJECTIVE
//==========================================================================
//
// One asset under risk-neutral GBM,
//
//       dS = r S dt + sigma S dW,      S(0) = K,
//
// discretised with ONE Milstein step over [0,T] (level 0, n_f = 1):
//
//       X_T = X_0 + r X_0 h + sigma X_0 dW + (1/2) sigma^2 X_0 (dW^2 - h)
//
// The Lookback payoff needs the minimum of the path, not just its endpoint.
// Conditional on the two endpoints, the minimum of the Brownian bridge on the
// step has an exact sampling formula driven by L = log U < 0:
//
//       M = (1/2)[ X_0 + X_T - sqrt( (X_T - X_0)^2 - 2 h (sigma X_0)^2 L ) ]
//       P = e^{-rT} ( X_T - M )
//
// This is minbr32/minbr16 in options_mm512.h.  The sqrt and the extra
// log-uniform draw are exactly what make this kernel more expensive per path
// than the basket European baseline, which needs neither.
//
// This file estimates chk = (1/N) sum_p P^{(p)} over N independent paths and
// times how long that takes in fp32 versus fp16+Kahan.
//
//==========================================================================
// 2. WHAT THE TWO KERNELS DIFFER IN
//==========================================================================
//
// kernel_fp32      : __m512  state, 16 lanes.
// kernel_fp16_kahan: __m256h state, 16 lanes.
//
// The fp16 state is deliberately 16-wide (__m256h) rather than the 32 lanes a
// __m512h would allow, matching the production estimator: one fp16 lane per
// fp32 lane keeps per-path work identical so the timing is like-for-like.
//
// At level 0 there is a single timestep, so there is no running accumulation
// for Kahan compensation to act on -- the "+Kahan" in the fp16 kernel's name
// is inherited from the production design, and at l=0 it costs and does
// nothing.  That is a property of level 0, not an omission here.
//==========================================================================

// ---- model constants: identical to nested_scalar_milstein_fp16_avx512.cpp
static const float K = 100.0f, T = 1.0f, r = 0.05f, sig = 0.2f;

namespace S = opt::scalar;

//==========================================================================
// 3. THE TWO KERNELS
//==========================================================================

// ---- fp32 -----------------------------------------------------------------
static double kernel_fp32(long long N)
{
    const int   nf   = 1;                     // level 0: one timestep
    const float hf_f = T / (float)nf;
    const float disc = std::exp(-r * T);

    const S::C32 cf32   = S::make32(r, sig, hf_f);
    const __m512 vK32   = _mm512_set1_ps(K);
    const __m512 vSig32 = _mm512_set1_ps(sig);

    double sink = 0.0;
    const long long nblk = (N + 15) / 16;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:sink) schedule(static)
#endif
    for (long long b = 0; b < nblk; ++b) {
        avx_rng g;
#ifdef _OPENMP
        rng_seed(g, (unsigned)(b * 2654435761u + omp_get_thread_num()));
#else
        rng_seed(g, (unsigned)(b * 2654435761u));
#endif
        // dW ~ N(0,h), and L = log U < 0 for the exact bridge minimum
        __m512 dW  = _mm512_mul_ps(cf32.sqhf, normal16_fp32(g));
        __m512 Lrv = loguniform16_fp32(g);

        // X(0) = K;  v = sigma X, the local diffusion coefficient
        __m512 X0  = vK32, v = _mm512_mul_ps(vSig32, X0);
        // Milstein step
        __m512 Xf  = _mm512_add_ps(X0, S::mil_incr32(X0, dW, cf32));
        // Bridge minimum over the step (the sqrt lives in here)
        __m512 Mf  = S::minbr32(vK32, X0, Xf, v, Lrv, cf32);

        alignas(64) float Xa[16], Ma[16];
        _mm512_store_ps(Xa, Xf);
        _mm512_store_ps(Ma, Mf);

        double li = 0.0;
        for (int j = 0; j < 16; ++j)
            li += (double)disc * (double)(Xa[j] - Ma[j]);   // Lookback payoff
        sink += li;
    }
    return sink / (double)(nblk * 16);
}

// ---- fp16 + Kahan ---------------------------------------------------------
static double kernel_fp16_kahan(long long N)
{
    const int   nf   = 1;
    const float hf_f = T / (float)nf;
    const float disc = std::exp(-r * T);

    const S::C16  cf16    = S::make16(r, sig, (_Float16)hf_f);
    const __m256h vK16    = _mm256_set1_ph((_Float16)K);
    const __m256h vSig16  = _mm256_set1_ph((_Float16)sig);
    const __m256h vSqhf16 = _mm256_set1_ph((_Float16)std::sqrt(hf_f));

    double sink = 0.0;
    const long long nblk = (N + 15) / 16;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:sink) schedule(static)
#endif
    for (long long b = 0; b < nblk; ++b) {
        avx_rng g;
#ifdef _OPENMP
        rng_seed(g, (unsigned)(b * 2654435761u + omp_get_thread_num()));
#else
        rng_seed(g, (unsigned)(b * 2654435761u));
#endif
        // dW: native fp16 draw (norminv_fp16 spline, no fp32 detour)
        __m256h dW  = _mm256_mul_ph(vSqhf16, normal16_fp16(g));
        // Lrv: no fp16 spline for the log-uniform, so fp32 then narrow --
        // verbatim from production, a real fp32 dependency in this kernel
        __m256h Lrv = opt::narrow16(loguniform16_fp32(g));

        __m256h X0 = vK16, v = _mm256_mul_ph(vSig16, X0);
        __m256h Xf = _mm256_add_ph(X0, S::mil_incr16(X0, dW, cf16));
        __m256h Mf = S::minbr16(vK16, X0, Xf, v, Lrv, cf16);

        alignas(64) _Float16 Xa[16], Ma[16];
        _mm256_store_ph(Xa, Xf);
        _mm256_store_ph(Ma, Mf);

        double li = 0.0;
        for (int j = 0; j < 16; ++j)
            li += (double)disc * (double)(_Float16)(Xa[j] - Ma[j]);
        sink += li;
    }
    return sink / (double)(nblk * 16);
}

//==========================================================================
// 4. TIMING HARNESS
//==========================================================================

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

struct result { double best, med, chk; };

static result time_kernel(double (*fn)(long long), long long N, int reps)
{
    std::vector<double> t;
    t.reserve(reps);
    double chk = 0.0;
    for (int i = 0; i < reps; ++i) {
        double w = now_sec();
        chk = fn(N);
        t.push_back(now_sec() - w);
    }
    std::sort(t.begin(), t.end());
    result R;
    R.best = t.front();
    R.med  = t[t.size() / 2];
    R.chk  = chk;
    return R;
}

// Driver's own level-0 sample counts for scalar Lookback, from the adaptive
// complexity table's N_l column.
struct work { double eps; long long n0; };

static const work W[] = {
    {0.100,     37506},
    {0.050,    148880},
    {0.020,   1016136},
    {0.010,   4175386},
    {0.005,  17031064},
};
static const int NW = 5;

int main(int argc, char **argv)
{
    long long N    = ((long long)1) << 22;   // 4.19M paths, matching the
    int       reps = 5;                      // basket benchmark's default
    int       do32 = 1, do16 = 1, sweep = 0, scaling = 0;

    for (int a = 1; a < argc; ++a) {
        if      (!strcmp(argv[a], "--n")    && a+1 < argc) N    = atoll(argv[++a]);
        else if (!strcmp(argv[a], "--reps") && a+1 < argc) reps = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--sweep"))   sweep   = 1;
        else if (!strcmp(argv[a], "--scaling")) scaling = 1;
        else if (!strcmp(argv[a], "--kernel") && a+1 < argc) {
            ++a;
            do32 = !strcmp(argv[a], "fp32");
            do16 = !strcmp(argv[a], "fp16");
        }
    }

    printf("Level-0 kernel benchmark -- SCALAR LOOKBACK (option 2)\n");
    printf("  RNG -> Milstein (1 step) -> bridge minimum (sqrt) -> payoff\n");
    printf("  (pure Lookback: the shipped l=0 branch also computes the Asian\n");
    printf("   integral and discards it; that dead work is stripped here)\n");
#ifdef _OPENMP
    printf("  threads = %d (OpenMP build)\n", omp_get_max_threads());
#else
    printf("  threads = 1 (serial build)\n");
#endif
    printf("  reps    = %d per timing\n\n", reps);

    // ---- eps sweep -------------------------------------------------------
    if (sweep) {
        printf("Level-0 sample counts N_0 as requested by the MLMC driver\n");
        printf("(scalar Lookback, from the adaptive complexity table's N_l column)\n\n");
        printf("    eps       N_0 |  fp32(ms)  ns/path     Mp/s |"
               "  fp16(ms)  ns/path     Mp/s |  ratio\n");

        std::vector<result> ra(NW), rb(NW);
        for (int i = 0; i < NW; ++i) {
            long long ns = W[i].n0;
            if (do32) ra[i] = time_kernel(kernel_fp32,       ns, reps);
            if (do16) rb[i] = time_kernel(kernel_fp16_kahan, ns, reps);
            double a_ns = ra[i].best * 1e9 / (double)ns;
            double b_ns = rb[i].best * 1e9 / (double)ns;
            printf("  %5.3f %9lld | %9.4f %8.2f %8.2f | %9.4f %8.2f %8.2f | %5.2fx\n",
                   W[i].eps, ns,
                   ra[i].best * 1e3, a_ns, 1e3 / a_ns,
                   rb[i].best * 1e3, b_ns, 1e3 / b_ns,
                   (b_ns > 0.0) ? a_ns / b_ns : 0.0);
        }

        printf("\n  Correctness check (chk = mean discounted payoff):\n");
        printf("  %5s %9s | %11s %11s %9s\n",
               "eps", "N_0", "fp32 chk", "fp16 chk", "gap");
        for (int i = 0; i < NW; ++i)
            printf("  %5.3f %9lld | %11.6f %11.6f %9.2e\n",
                   W[i].eps, W[i].n0, ra[i].chk, rb[i].chk,
                   std::fabs(ra[i].chk - rb[i].chk));

        printf("\n  ns/p = nanoseconds per path.  A flat ns/p column means the\n"
               "  kernel's efficiency is size-independent; a rising one means\n"
               "  something degrades as the trip count grows.  The two smallest\n"
               "  N_0 inflate ns/p in BOTH kernels -- per-call overhead not yet\n"
               "  amortised -- so quote eps <= 0.02 for the steady-state cost.\n");
        return 0;
    }

    // ---- OpenMP scaling: 1 vs 32 threads ---------------------------------
    if (scaling) {
#ifndef _OPENMP
        printf("--scaling needs the OpenMP build (-qopenmp).\n");
        return 1;
#else
        // 1 and 32 threads; 32 == one full socket on mimic, so no HT
        // contention and no NUMA crossing.  Both counts run in ONE process
        // (omp_set_num_threads, not separate OMP_NUM_THREADS runs) so they
        // share a warm-up and neither pays its own first-touch cost.
        //
        // MEDIAN over TRIALS passes, thread counts INTERLEAVED within each --
        // same protocol as the basket benchmark, where best-of-N produced a
        // physically impossible row: at sub-millisecond times one disturbed
        // pass dominates, and best-of is robust to slow outliers but not fast
        // ones.  Output format matches basket exactly so that
        // python/kernel_scaling_plot_python.py parses both unchanged.
        const int tc[2] = {1, 32};
        const int TRIALS = 5;

        printf("OpenMP scaling at the driver's own level-0 sample counts\n");
        printf("(scalar Lookback; 32 threads = one full socket of physical\n");
        printf(" cores on mimic -- no hyperthreading, no NUMA crossing)\n");
        printf("median of %d independent trials, %d reps each, interleaved\n\n",
               TRIALS, reps);

        double t[NW][2][2];                       // [eps][kernel][threadcount]
        for (int i = 0; i < NW; ++i) {
            long long Ns = (W[i].n0 + 15) & ~((long long)15);
            std::vector<double> v[2][2];
            for (int tr = 0; tr < TRIALS; ++tr)
                for (int ti = 0; ti < 2; ++ti) {
                    omp_set_num_threads(tc[ti]);
                    v[0][ti].push_back(time_kernel(kernel_fp32,       Ns, reps).best);
                    v[1][ti].push_back(time_kernel(kernel_fp16_kahan, Ns, reps).best);
                }
            for (int kk = 0; kk < 2; ++kk)
                for (int ti = 0; ti < 2; ++ti) {
                    std::sort(v[kk][ti].begin(), v[kk][ti].end());
                    t[i][kk][ti] = v[kk][ti][TRIALS/2];
                }
        }

        // Two stacked blocks: eleven columns on one line does not fit.
        printf("  Wall clock (ms):\n");
        printf("  %5s %9s | %9s %9s | %9s %9s | %6s\n",
               "eps", "N_0", "fp32 1thr", "fp32 32t", "fp16 1thr", "fp16 32t",
               "ratio32");
        for (int i = 0; i < NW; ++i)
            printf("  %5.3f %9lld | %9.4f %9.4f | %9.4f %9.4f | %5.2fx\n",
                   W[i].eps, W[i].n0,
                   1000.0*t[i][0][0], 1000.0*t[i][0][1],
                   1000.0*t[i][1][0], 1000.0*t[i][1][1],
                   t[i][0][1]/t[i][1][1]);

        printf("\n  Times in ms.  ratio32 = fp32/fp16 wall clock at 32\n"
               "  threads; compare it with the serial ratio (~1.21x) to see\n"
               "  whether threading erodes the fp16 advantage.\n"
               "  The smallest N_0 runs for well under a millisecond at 32\n"
               "  threads, where thread-startup overhead is a large fraction\n"
               "  of the measurement.  Quote eps <= 0.02.\n");
        return 0;
#endif
    }

    // ---- single N: the target for perf / VTune ---------------------------
    if (do32) {
        result R = time_kernel(kernel_fp32, N, reps);
        printf("fp32         N = %-11lld best %8.4f ms  %6.2f ns/path  chk %.6f\n",
               N, R.best * 1e3, R.best * 1e9 / (double)N, R.chk);
    }
    if (do16) {
        result R = time_kernel(kernel_fp16_kahan, N, reps);
        printf("fp16+Kahan   N = %-11lld best %8.4f ms  %6.2f ns/path  chk %.6f\n",
               N, R.best * 1e3, R.best * 1e9 / (double)N, R.chk);
    }
    return 0;
}
