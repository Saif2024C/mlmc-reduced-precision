////////////////////////////////////////////////////////////////////////////
// Level-0 kernel benchmark: BASKET EUROPEAN, fp32 vs fp16+Kahan.
//
// Performance_analysis.txt item 1a -- the real l=0 production kernel in
// isolation, NOT the MLMC driver:  RNG -> Milstein -> terminal payoff.
// 
// N_0 per eps comes from the driver, so every row is a real workload.  `sink`
// stops the loop being optimised away; `chk` (mean discounted payoff) catches
// a fast-but-broken kernel -- both precisions must price the same option.
//
// BUILD / RUN (mimic, after `source /opt/intel/oneapi/setvars.sh`):
//   make kbench                  # serial sweep over eps -- the serial result
//   make kbench_scaling          # 1 vs 32 threads, wall clock (OMP)
//   ./kb_basket_l0 --kernel fp32 # one kernel, one N: a target for perf/VTune
//                                # (profiling both at once mixes them in the
//                                #  same hotspot report)
////////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
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
//   P = e^{-rT} max( sum_{i=1..5} w_i S_i(T) - K , 0 )
//
//   and this file estimates its Monte Carlo mean
//
//       chk = (1/N) sum_{p=1..N} P^{(p)}
//
//   over N independent paths, timing how long that takes in fp32 vs fp16.

// Level 0 alone contributes E[P_0] to the MLMC telescoping sum, and it is
// where the driver puts the overwhelming majority of its samples (N_0 is
// 4.6M at eps=0.01), which is why isolating and profiling it is worthwhile.
//
//
//==========================================================================
// 2. CODE WALKTHROUGH (execution order)
//==========================================================================
//
//  Model constants        K, T, r, sigma_i, w_i, rho -- the option and the SDE.
//  build_cholesky()       Factors Sigma = L L^T once, so every path can turn
//                         5 independent Normals into 5 correlated increments.
//  kernel_*()             Per 16-lane batch:
//    Initialisation       S_i(0) = S0 = K for all 5 assets, all 16 lanes.
//    Correlated draws     dW_i = sqrt(h) (L Z)_i  -- the driving randomness.
//    Milstein update      S_i <- S_i + dS_i, one step, h = T.
//    (fp16 only) Kahan    Same update, with a carried compensation term
//                         absorbing what each fp16 add rounds away.
//    Basket value         B = sum_i w_i S_i(T).
//    Payoff               P = e^{-rT} max(B - K, 0), per lane.
//    Reduction            sum_p P^{(p)}, divided by N at the end.
//  time_kernel()          Warm-up, then `reps` timed repetitions; reports the
//                         best/median wall time and `chk` (the MC mean).
//  main()                 --sweep: cost vs the driver's real N_0 per eps.
//                         --scaling: 1 vs 32 thread wall clock.
//
//==========================================================================

// Model constants -- identical to nested_basket_milstein_fp16_avx512.cpp
// K = strike AND initial price S_i(0) (at-the-money); Tm = maturity T;
// R = risk-free rate r appearing in both the drift and the discount factor.
static const float K   = 100.0f, Tm = 1.0f, R = 0.05f;
static const float SIG[5] = {0.25f, 0.30f, 0.35f, 0.40f, 0.45f};  // sigma_i
static const float W[5]   = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};       // w_i, sum = 1
static const float RHO    = 0.25f;                                // rho_ij, i != j
// Lower-triangular Cholesky factor L of the correlation matrix: Sigma = L L^T.
static float Lc[5][5];

// The per-path maths lives in ../options_mm512.h, shared with the production
// estimator, so this benchmark cannot drift away from the code it measures.
namespace B = opt::basket;

// Solve Sigma = L L^T once. Thereafter dW = sqrt(h) L Z turns 5 independent
// Normals Z into 5 increments with the required correlation structure.
static void build_cholesky() { B::cholesky(RHO, Lc); }

// Which Normal transform each precision uses: fp32 goes through SVML's
// cdfnorminv, fp16 through Giles' superdyadic spline.  Passed to draw_corr*
// so the header itself stays RNG-agnostic.
static inline __m512  n32(avx_rng &g) { return normal16_fp32(g); }
static inline __m256h n16(avx_rng &g) { return normal16_fp16(g); }

// -------------------------------------------------------------------------
// KERNEL A: pure fp32.
//
//   Computes  (1/N) sum_p e^{-rT} max( sum_i w_i S_i^{(p)}(T) - K , 0 )
//   with S(T) obtained from S(0) by ONE Milstein step of size h = T,
//   the entire path state S_i held in fp32.
//
// Verbatim from the adaptive k>=l_star l==0 branch, opt==1 -- the opt==2
// integral-draw and wdot bridge terms are dead, elided.
// -------------------------------------------------------------------------
static double kernel_fp32(long long N)
{
    const int   nf     = 1;                    // 2^0 fine timesteps: level 0
    const float hf_f   = Tm / (float)nf;       // h = T / n_f  (= T here)
    const float disc_f = std::exp(-R * Tm);    // discount factor e^{-rT}
    const float sf_f   = std::sqrt(hf_f);      // sqrt(h): the dW scaling

    const __m512 vHf32 = _mm512_set1_ps(hf_f);
    const __m512 vK32  = _mm512_set1_ps(K);    // S_i(0) = K, at-the-money

    double sink = 0.0;                         // running sum_p P^{(p)}

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:sink)
#endif
    // One iteration = 16 independent Monte Carlo paths, one per SIMD lane.
    for (long long i0 = 0; i0 < N; i0 += 16) {
        int nb = (int)std::min((long long)16, N - i0);
        // Independent Philox substream per batch: paths must be i.i.d.
        avx_rng g; rng_seed(g, (unsigned)((i0 >> 4) * 97) + 1u);

        // Initialise all 5 asset prices S_i(0) = S0 = K for these 16 paths
        __m512 xf[5]; for (int i = 0; i < 5; ++i) xf[i] = vK32;
        __m512 dw[5], sigx[5];
        // Correlated Brownian increments  dW_i = sqrt(h) * (L Z)_i
        B::draw_corr32(g, n32, sf_f, Lc, dw);
        // Milstein update, per asset, in place:
        //   S_i <- S_i + r S_i h + sigma_i S_i dW_i
        //              + (1/2) sigma_i^2 S_i (dW_i^2 - h)
        B::step32(xf, dw, vHf32, R, SIG, sigx);
        // Weighted basket value at maturity  B = sum_i w_i S_i(T)
        __m512 bf = B::value32(xf, W);

        alignas(64) float Ba[16];
        _mm512_store_ps(Ba, bf);
        double li = 0.0;
        for (int j = 0; j < nb; ++j)
            // Discounted European call payoff  e^{-rT} max(B - K, 0)
            li += (double)(disc_f * std::fmax(0.0f, Ba[j] - K));
        // Reduction: accumulate sum_p P^{(p)} (fp64, so the MC sum is exact
        // to well beyond the path precision under test)
        sink += li;
    }
    return sink / (double)N;                   // the MC estimate of E[P]
}

// -------------------------------------------------------------------------
// KERNEL B: fp16 + Kahan.
//
// Same mathematics as Kernel A, term for term -- same SDE, same Milstein
// scheme, same basket, same payoff.  The ONLY difference is that the path
// state S_i, the increment, the basket value and the payoff are all fp16,
// and the state update
//
//       S_i <- S_i + dS_i
//
// is done by compensated (Kahan) summation: a carried term c_i tracks what
// each fp16 add rounded away and is subtracted back into the next increment,
//
//       y = dS_i - c_i ;  t = S_i + y ;  c_i = (t - S_i) - y ;  S_i = t
//
// which recovers most of the accuracy an uncompensated fp16 accumulation
// would lose.  (At level 0 there is a single step, so Kahan has almost
// nothing to correct here -- its cost, not its benefit, is what this
// benchmark measures.)
//
// MODIFIED (2026-08-25): xf/xf_c are __m512h -- 32 lanes, NOT the 16-lane
// __m256h of the original benchmark.  The motivation is a measurement, not a
// guess: norminv_fp16 (my_mm512.h) reads its __m512i argument as 32 x 16-bit
// integers and returns a full __m512h, but normal16_fp16_truncated() then
// throws the upper 16 lanes away (`_mm512_castph512_ph256`).  perf counted
// 32.87 M 512b-packed-half instructions in the original fp16 kernel -- ~25
// per norminv_fp16 call -- i.e. the superdyadic spline was already running
// 32-wide with half its output discarded.
//
// One Philox draw is 512 bits = 32 independent 16-bit uniforms, so 32 lanes
// need NO extra RNG call: the second half of the transform is work already
// being paid for.  avx512_rng.h states this directly ("this consumes a full
// 512-bit Philox draw to return 16 fp16 lanes").
//
// COMPARABILITY CAVEAT: this kernel now retires 32 paths per iteration
// against fp32's 16, so ns/path no longer isolates arithmetic the way the
// 16-lane version did -- the RNG is amortised over twice as many paths here.
// That is the point of the experiment, but it means the speedup from THIS
// file and from the original are not the same measurement.
//
// The 32-lane helpers below are local to this benchmark; options_mm512.h is
// deliberately left untouched so the production estimators are unaffected.
// -------------------------------------------------------------------------

// 32-lane counterparts of opt::kahan_accum16 / basket::{draw_corr16, incr16,
// value16}, identical arithmetic with __m512h in place of __m256h.
static inline void kahan_accum32h(__m512h &sum, __m512h &comp, __m512h term) {
    __m512h y = _mm512_sub_ph(term, comp);
    __m512h t = _mm512_add_ph(sum, y);
    comp      = _mm512_sub_ph(_mm512_sub_ph(t, sum), y);
    sum       = t;
}

// Full 32-lane fp16 Normal draw: norminv_fp16 without the truncating cast.
static inline __m512h normal32_fp16_full(avx_rng &g) {
    return norminv_fp16(philox_next(g));
}

static inline void draw_corr32h(avx_rng &g, _Float16 scale,
                                const float L[5][5], __m512h dw[5]) {
    __m512h y[5]; for (int i = 0; i < 5; ++i) y[i] = normal32_fp16_full(g);
    __m512h sc = _mm512_set1_ph(scale);
    for (int i = 0; i < 5; ++i) {
        __m512h s = _mm512_setzero_ph();
        for (int j = 0; j <= i; ++j)
            s = _mm512_fmadd_ph(_mm512_set1_ph((_Float16)L[i][j]), y[j], s);
        dw[i] = _mm512_mul_ph(sc, s);
    }
}

static inline void incr32h(const __m512h x0[5], const __m512h dw[5], __m512h h,
                           float r, const float sig[5], __m512h out[5]) {
    __m512h vR = _mm512_set1_ph((_Float16)r);
    for (int i = 0; i < 5; ++i) {
        __m512h vS = _mm512_set1_ph((_Float16)sig[i]);
        __m512h vH = _mm512_set1_ph((_Float16)(0.5f*sig[i]*sig[i]));
        __m512h sigx  = _mm512_mul_ph(vS, x0[i]);
        __m512h drift = _mm512_mul_ph(_mm512_mul_ph(x0[i], vR), h);
        __m512h diff  = _mm512_mul_ph(sigx, dw[i]);
        __m512h d2h   = _mm512_sub_ph(_mm512_mul_ph(dw[i], dw[i]), h);
        __m512h mil   = _mm512_mul_ph(_mm512_mul_ph(x0[i], vH), d2h);
        out[i] = _mm512_add_ph(_mm512_add_ph(drift, diff), mil);
    }
}

static inline __m512h value32h(const __m512h x[5], const float w[5]) {
    __m512h s = _mm512_setzero_ph();
    for (int i = 0; i < 5; ++i)
        s = _mm512_fmadd_ph(_mm512_set1_ph((_Float16)w[i]), x[i], s);
    return s;
}

static double kernel_fp16_kahan(long long N)
{
    const int   nf     = 1;                    // 2^0 fine timesteps: level 0
    const float hf_f   = Tm / (float)nf;       // h = T / n_f
    const float disc_f = std::exp(-R * Tm);    // e^{-rT}
    const float sf_f   = std::sqrt(hf_f);      // sqrt(h)

    // Same constants, narrowed once: everything downstream stays fp16.
    const _Float16 disc_h = (_Float16)disc_f;
    const _Float16 K_h    = (_Float16)K;
    const __m512h  vHf16  = _mm512_set1_ph((_Float16)hf_f);
    const __m512h  vK16   = _mm512_set1_ph(K_h);

    double sink = 0.0;                         // running sum_p P^{(p)}

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:sink)
#endif
    // One iteration = 32 independent Monte Carlo paths, one per fp16 lane.
    for (long long i0 = 0; i0 < N; i0 += 32) {
        int nb = (int)std::min((long long)32, N - i0);
        avx_rng g; rng_seed(g, (unsigned)((i0 >> 5) * 131) + 1u);

        // Initialise all 5 asset prices S_i(0) = S0 = K for these 32 paths
        __m512h xf[5];   for (int i = 0; i < 5; ++i) xf[i]   = vK16;
        // Kahan compensation c_i, one per asset, zero at t = 0
        __m512h xf_c[5]; for (int i = 0; i < 5; ++i) xf_c[i] = _mm512_setzero_ph();

        __m512h dw[5];
        // Correlated Brownian increments dW_i = sqrt(h) (L Z)_i, all 32 lanes
        // of norminv_fp16 kept (the original discarded the upper 16)
        draw_corr32h(g, (_Float16)sf_f, Lc, dw);

        __m512h incr[5];
        // Milstein INCREMENT only (not applied):
        //   dS_i = r S_i h + sigma_i S_i dW_i + (1/2) sigma_i^2 S_i (dW_i^2 - h)
        incr32h(xf, dw, vHf16, R, SIG, incr);
        // Apply S_i <- S_i + dS_i by compensated summation, carrying c_i
        for (int i = 0; i < 5; ++i) kahan_accum32h(xf[i], xf_c[i], incr[i]);
        // Weighted basket value at maturity  B = sum_i w_i S_i(T)
        __m512h bf = value32h(xf, W);

        alignas(64) _Float16 Ba[32];
        _mm512_store_ph(Ba, bf);
        double li = 0.0;
        for (int j = 0; j < nb; ++j) {
            // Discounted European call payoff e^{-rT} max(B - K, 0), with the
            // max taken in fp16 so the payoff itself is genuinely fp16 too
            _Float16 v = Ba[j] - K_h;
            li += (double)disc_h * (double)(v > (_Float16)0.0f ? v : (_Float16)0.0f);
        }
        // Reduction in fp64: the MC sum is deliberately never the bottleneck
        sink += li;
    }
    return sink / (double)N;                   // the MC estimate of E[P]
}

// timing driver
static double now_sec() {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
#endif
}

// best/med = wall time in seconds; chk = the MC mean (1/N) sum_p P^{(p)},
// which both kernels must agree on -- it is the option price, not a timing.
struct result { double best, med, chk; };

static result time_kernel(double (*fn)(long long), long long N, int reps)
{
    fn(std::min(N, (long long)1 << 16));      // untimed warm-up
    std::vector<double> t;
    double chk = 0.0;
    for (int r = 0; r < reps; ++r) {
        double w = now_sec();
        chk = fn(N);
        t.push_back(now_sec() - w);
    }
    std::sort(t.begin(), t.end());
    result R;
    R.best = t.front();
    R.med  = t[t.size()/2];
    R.chk  = chk;
    return R;
}

int main(int argc, char **argv)
{
    long long N    = ((long long)1) << 22;   // 4.19M paths
    int       reps = 5;
    int       do32 = 1, do16 = 1, sweep = 0, scaling = 0;

    for (int a = 1; a < argc; ++a) {
        if (!strcmp(argv[a], "--n")      && a+1 < argc) N    = atoll(argv[++a]);
        else if (!strcmp(argv[a], "--reps") && a+1 < argc) reps = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--sweep"))   sweep   = 1;
        else if (!strcmp(argv[a], "--scaling")) scaling = 1;
        else if (!strcmp(argv[a], "--kernel") && a+1 < argc) {
            ++a;
            do32 = !strcmp(argv[a], "fp32");
            do16 = !strcmp(argv[a], "fp16");
        }
    }
    N = (N + 31) & ~((long long)31);          // round up to a whole 32-lane batch
                                              // (32, not 16: the fp16 kernel here is 32-lane
                                              //  and both kernels must see identical N)

    build_cholesky();

    printf("Level-0 kernel benchmark -- BASKET EUROPEAN (option 1)\n");
    printf("  RNG -> per-asset Milstein (1 step) -> terminal payoff\n");
#ifdef _OPENMP
    printf("  threads = %d (OpenMP)\n", omp_get_max_threads());
#else
    printf("  threads = 1 (serial build)\n");
#endif
    printf("  reps    = %d per timing\n\n", reps);


#ifdef _OPENMP
    if (scaling) {
        // 1 and 32 threads; 32 == one full socket on mimic, so no HT
        // contention and no NUMA crossing.  Both counts run in ONE process
        // (omp_set_num_threads, not separate OMP_NUM_THREADS runs) so they
        // share a warm-up and neither pays its own first-touch cost.
        //
        // MEDIAN over TRIALS passes, thread counts INTERLEAVED within each.
        // Best-of-N gave a physically impossible row here (fp32 37.6% beside
        // fp16 98.7% at the same eps): at 0.02-7 ms one disturbed pass
        // dominates, and best-of is robust to slow outliers but not fast ones.
        struct { double eps; long long n0; } W[] = {
            {0.20,     9704},
            {0.10,    44758},
            {0.05,   173030},
            {0.02,  1169929},
            {0.01,  4658857},
        };
        const int NW = (int)(sizeof(W)/sizeof(W[0]));
        const int tc[2] = {1, 32};
        const int TRIALS = 5;

        printf("OpenMP scaling at the driver's own level-0 sample counts\n");
        printf("(basket European; 32 threads = one full socket of physical\n");
        printf(" cores on mimic -- no hyperthreading, no NUMA crossing)\n");
        printf("median of %d independent trials, %d reps each, interleaved\n\n",
               TRIALS, reps);

        double t[NW][2][2];                       // [eps][kernel][threadcount]
        for (int i = 0; i < NW; ++i) {
            long long Ns = (W[i].n0 + 31) & ~((long long)31);
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
               "  threads; compare it with the serial ratio (~2.35x) to see\n"
               "  whether threading erodes the fp16 advantage.\n"
               "  The two smallest N_0 run for well under a millisecond at 32\n"
               "  threads, where thread-startup overhead is a large fraction\n"
               "  of the measurement.  Quote eps <= 0.02.\n");
        return 0;
    }
#endif

    if (sweep) {
        // N_0 per eps, from the N_l column of
        // outputs/avx512/basket/nested_basket_fp16_avx512_kahan_1.txt.
        // Re-derive if the driver's Eps[], model constants, or beta change.
        struct { double eps; long long n0; } W[] = {
            {0.20,     9704},
            {0.10,    44758},
            {0.05,   173030},
            {0.02,  1169929},
            {0.01,  4658857},
        };
        printf("Level-0 sample counts N_0 as requested by the MLMC driver\n");
        printf("(basket European, from the complexity table's N_l column)\n\n");

        const int NW = (int)(sizeof(W)/sizeof(W[0]));
        std::vector<result> ra(NW), rb(NW);
        std::vector<long long> ns(NW);
        for (int i = 0; i < NW; ++i) {
            ns[i] = (W[i].n0 + 31) & ~((long long)31);
            if (do32) ra[i] = time_kernel(kernel_fp32,       ns[i], reps);
            if (do16) rb[i] = time_kernel(kernel_fp16_kahan, ns[i], reps);
        }

        // --- timing / throughput (Mp/s = millions of paths/sec) ---
        printf("  %5s %9s | %9s %8s %8s | %9s %8s %8s | %6s\n",
               "eps", "N_0",
               "fp32(ms)", "ns/path", "Mp/s",
               "fp16(ms)", "ns/path", "Mp/s", "ratio");
        for (int i = 0; i < NW; ++i) {
            printf("  %5.3f %9lld | %9.4f %8.2f %8.2f | %9.4f %8.2f %8.2f | %5.2fx\n",
                   W[i].eps, W[i].n0,
                   1000.0*ra[i].best, 1e9*ra[i].best/(double)ns[i],
                   1e-6*(double)ns[i]/ra[i].best,
                   1000.0*rb[i].best, 1e9*rb[i].best/(double)ns[i],
                   1e-6*(double)ns[i]/rb[i].best,
                   (do32 && do16) ? ra[i].best/rb[i].best : 0.0);
        }

        // --- correctness: a large gap means fp16 is skipping work, not
        // that it is efficient.  Pure payoff discretisation, stays O(1e-2). ---
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

    // Default / --kernel: one kernel, one N -- a bare workload for perf or
    // VTune to attach to.  --sweep and --scaling give the reportable numbers.
    const bool only16 = do16 && !do32;
    result r = only16 ? time_kernel(kernel_fp16_kahan, N, reps)
                      : time_kernel(kernel_fp32,       N, reps);
    printf("%-11s  N = %-10lld  best %8.4f ms  %6.2f ns/path  chk %.6f\n",
           only16 ? "fp16+Kahan" : "fp32", N,
           1000.0*r.best, 1e9*r.best/(double)N, r.chk);
    return 0;
}
