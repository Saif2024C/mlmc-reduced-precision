/*
 * nested_basket_milstein_fp16_avx512.cpp -- unified AVX-512 basket fp16
 * template.  5-asset correlated GBM; option 1 = European, 2 = Asian.
 *
 * Path state (per-asset x[5], Asian running sum A, payoff) is ALL __m256h --
 * genuine 16-wide fp16 arithmetic, not fp32 with narrowed storage.
 *
 * Per-path maths lives in options_mm512.h, shared with the scalar template
 * and the benchmarks; the forwarders below bind this file's globals to it.
 *
 * Model: S0=K=100, T=1, r=0.05, sigma={0.25..0.45}, weights 0.2, rho=0.25.
 * OpenMP is unconditional (active iff _OPENMP): one Philox stream per 16-lane
 * block, per-block accumulators reduced into sums[].  No separate _omp file.
 *
 * Build/run:  make nbh_avx  (add -qopenmp -> nbh_omp);  see src/AVX-512/Makefile.
 * Writes nested_basket_fp16_avx512_{nokahan,kahan,adaptive}_{1,2}.txt to cwd.
 *
 *==========================================================================
 * 1. MATHEMATICAL OBJECTIVE
 *==========================================================================
 *
 * Five correlated GBM assets under the risk-neutral measure
 *
 *       dS_i = r S_i dt + sigma_i S_i dW_i ,   d<W_i,W_j> = rho_ij dt,
 *       rho_ij = rho (i != j), 1 (i = j),      S_i(0) = S0 = K
 *
 * with the basket value
 *
 *       B(t) = sum_{i=1..5} w_i S_i(t) ,       sum_i w_i = 1
 *
 * and the two payoffs priced:
 *
 *   option 1, BASKET EUROPEAN call:
 *       P = e^{-rT} max( B(T) - K , 0 )
 *
 *   option 2, BASKET ASIAN call:
 *       P = e^{-rT} max( Abar - K , 0 ),   Abar = (1/T) int_0^T B(t) dt
 *
 * European depends only on the terminal basket; Asian is path-dependent and
 * so carries a running time-integral A, exactly as the scalar Asian does.
 * There is no lookback here, hence no bridge minimum M.
 *
 * ---- Correlation ----
 * Correlated increments come from the Cholesky factor Sigma = L L^T:
 *
 *       dW = sqrt(h) * L Z ,      Z ~ N(0, I_5)
 *
 * so Cov(dW) = h L L^T = h Sigma as required.  L is built once by
 * build_cholesky() and reused by every path.
 *
 * ---- Discretisation ----
 * MILSTEIN, per asset, step h:
 *
 *   S_i(t+h) = S_i + r S_i h + sigma_i S_i dW_i
 *                  + (1/2) sigma_i^2 S_i ( dW_i^2 - h )
 *
 * The Asian time-integral uses the trapezoid plus the WEIGHTED bridge term
 *
 *       A += h B(t+h) + sum_i w_i (sigma_i S_i) dI_i ,  dI_i ~ N(0, h^3/12)
 *
 * The weights on the bridge term are load-bearing (Giles' alf'*(vf.*dIf));
 * an unweighted version is a documented past bug -- see CLAUDE.md.  A final
 * -(h/2)B(T) closes the trapezoid.
 *
 * ---- The NESTED (super-level) MLMC telescoping sum ----
 * Identical in structure to the scalar template.  k = l/2 is the grid level,
 * n_f = 2^k fine steps, n_c = n_f/2 coarse steps, and E[P] = sum_l E[Y_l]:
 *
 *   l = 0        Y_0 = P^{h}_0             (fp16 base level, 1 step)
 *   l even >= 2  Y_l = P^{h}_f - P^{h}_c   (Milstein MLMC correction, fp16)
 *   l odd        Y_l = P^{f32} - P^{f16}     (precision correction, same draws)
 *
 * even levels correct DISCRETISATION error, odd levels correct PRECISION
 * error.  Target rates: alpha ~ 1, beta ~ 2 (Milstein), gamma ~ 1.
 *
 */

#include "../core/nested_mlmc_test.cpp"
#include "avx512_rng.h"
#include "options_mm512.h"
#include "modes_mm512.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace B = opt::basket;   // per-path maths: ../options_mm512.h

int option;                                   // 1 = European, 2 = Asian
// K = strike AND S_i(0) (at-the-money); Tm = maturity T; R = risk-free rate r.
static const float K = 100.0f, Tm = 1.0f, R = 0.05f;
static const float SIG[5] = {0.25f, 0.30f, 0.35f, 0.40f, 0.45f};  // sigma_i
static const float W[5]   = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};       // w_i, sum = 1
static const float RHO    = 0.25f;                                // rho_ij, i != j
static float Lc[5][5];                        // Cholesky factor (fp32)

// Solve Sigma = L L^T once; thereafter dW = sqrt(h) L Z has Cov = h Sigma.
static void build_cholesky() { B::cholesky(RHO, Lc); }

void nested_basket_fp16_avx_l(int, int, double *);


int main(int argc, char **argv)
{
    // l_star=6 for both options
    static const float EPS[] = { 0.01f, 0.02f, 0.05f, 0.1f, 0.2f, 0.0f };

    SweepConfig cfg;
    cfg.prefix      = "nested_basket_fp16_avx512";
    cfg.family      = "basket";
    cfg.opt_name[0] = "European";
    cfg.opt_name[1] = "Asian";
    cfg.estimator   = nested_basket_fp16_avx_l;
    cfg.eps         = EPS;
    cfg.l_star[1]   = 6;
    cfg.l_star[2]   = 6;
    cfg.setup       = build_cholesky;
    return run_sweep(cfg, argc, argv);
}

// fp32 per-asset helpers (16-wide __m512), used only above the l_star cutoff.
static inline __m512  n32_(avx_rng &g) { return normal16_fp32(g); }
static inline __m256h n16_(avx_rng &g) { return normal16_fp16(g); }
static inline void draw_corr_f32(avx_rng &g, float scale, __m512 dw[5]) {
    B::draw_corr32(g, n32_, scale, Lc, dw);
}

// Native fp16 correlated draw: 5 Normals via norminv_fp16 (Giles' spline, not
// a downcast), Cholesky-combined in fp16, on its own Philox bits.
static inline void draw_corr_fp16(avx_rng &g, _Float16 scale, __m256h dw[5]) {
    B::draw_corr16(g, n16_, scale, Lc, dw);
}


// Coupled draw for the odd (precision-correction) levels: fp32 and fp16 each
// apply their own inverse-CDF to the SAME Philox draw per asset, then each
// Cholesky-combines its own set.  Neither is a downcast of the other.

static inline void draw_corr_coupled(avx_rng &g, float scale_f32, _Float16 scale_f16,
                                      __m512 dw_f32[5], __m256h dw_f16[5]) {
    __m512  y32[5];
    __m256h y16[5];
    // 5 independent standard Normals Z_i, each materialised in BOTH precisions
    // from one shared Philox draw (two inverse-CDFs, not a downcast)
    for (int i = 0; i < 5; ++i) {
        __m512i bits = philox_next(g);
        y32[i] = normal16_fp32_bits(bits);
        y16[i] = normal16_fp16_aligned(bits);
    }
    __m512  sc32 = _mm512_set1_ps(scale_f32);
    __m256h sc16 = _mm256_set1_ph(scale_f16);
    // Correlate and scale in each precision separately:
    //   dW_i = scale * sum_{j<=i} L_ij Z_j ,  scale = sqrt(h)
    for (int i = 0; i < 5; ++i) {
        __m512  s32 = _mm512_setzero_ps();
        __m256h s16 = _mm256_setzero_ph();
        for (int j = 0; j <= i; ++j) {
            s32 = _mm512_fmadd_ps(_mm512_set1_ps(Lc[i][j]), y32[j], s32);
            s16 = _mm256_fmadd_ph(_mm256_set1_ph((_Float16)Lc[i][j]), y16[j], s16);
        }
        dw_f32[i] = _mm512_mul_ps(sc32, s32);
        dw_f16[i] = _mm256_mul_ph(sc16, s16);
    }
}
// Thin forwarders onto options_mm512.h: bind this file's globals (R/SIG/W/K/Lc)
static inline __m512  basket32(const __m512  x[5]) { return B::value32(x, W); }
static inline __m256h basket16(const __m256h x[5]) { return B::value16(x, W); }
static inline __m512  wdot32(const __m512  a[5], const __m512  b[5]) { return B::wdot32(a, b, W); }
static inline __m256h wdot16(const __m256h a[5], const __m256h b[5]) { return B::wdot16(a, b, W); }

static inline void step32(__m512 x[5], const __m512 dw[5], __m512 h, __m512 sigx[5]) {
    B::step32(x, dw, h, R, SIG, sigx);
}
static inline void mil_incr16_asset(const __m256h x0[5], const __m256h dw[5], __m256h h,
                                     __m256h out[5], __m256h sigx[5]) {
    B::incr16(x0, dw, h, R, SIG, out, sigx);
}

static inline double pay32(int opt, float basket_val, float A) {
    return B::pay32(opt, K, basket_val, A);
}
static inline double pay16(int opt, _Float16 basket_val, _Float16 A) {
    return B::pay16(opt, (_Float16)K, basket_val, A);
}

// kahan_mode picks the compensated or plain accumulator; the header exposes
// both as separate functions rather than hiding the branch in one.
static inline void kahan_accum16(__m256h &sum, __m256h &comp, __m256h term) {
    if (kahan_mode) opt::kahan_accum16(sum, comp, term);
    else            opt::plain_accum16(sum, comp, term);
}
static inline void kahan_accum16_asset(__m256h sum[5], __m256h comp[5], const __m256h term[5]) {
    for (int i = 0; i < 5; ++i) kahan_accum16(sum[i], comp[i], term[i]);
}


void nested_basket_fp16_avx_l(int l, int N, double *sums)
{
    const int k  = l / 2;        // grid level: two super-levels per grid level
    const int nf = 1 << k;       // n_f = 2^k fine timesteps
    const int nc = nf / 2;       // n_c = n_f/2 coarse timesteps (0 at k=0)

    const float hf_f   = Tm / (float)nf;             // fine step   h_f = T/n_f
    const float hc_f   = (nc > 0) ? Tm / (float)nc : 0.0f; // coarse step h_c = 2h_f
    const float disc_f = std::exp(-R * Tm);          // discount factor e^{-rT}
    // sd of the Asian bridge draw dI ~ N(0, h^3/12):  si = sqrt(h/12) * h
    const float si_f   = sqrtf(hf_f / 12.0f) * hf_f;
    const float sf_f   = sqrtf(hf_f);                // sqrt(h): the dW scaling

    const _Float16 hf_h   = (_Float16)hf_f;
    const _Float16 hc_h   = (_Float16)hc_f;
    const _Float16 disc_h = (_Float16)disc_f;
    const _Float16 K_h    = (_Float16)K;

    for (int i = 0; i < 7; ++i) sums[i] = 0.0;

    const int opt = option;

    // Adaptive cutoff, k >= l_star: pure fp32. 
    
    if (adaptive_mode && k >= l_star) {
        // Above the cutoff the path is FP32 throughout, so the scheme is
        // standard non-nested MLMC: one correction P_k - P_{k-1} per grid
        // level, carried by the even super-level, and no precision correction
        // at all.  The odd super-level therefore has no work to do.  It is
        // returned empty rather than run: sums stay zero, so the driver's
        // zero-variance guard allocates it no samples and it contributes
        // nothing to the cost.  Running it (two identical FP32 chains
        // differenced to an exact zero) would charge 2*n_f per path for a
        // level that cannot affect the estimate.
        if (l % 2 == 1) return;

        const __m512 vHalf32 = _mm512_set1_ps(0.5f);
        const __m512 vHf32 = _mm512_set1_ps(hf_f), vHc32 = _mm512_set1_ps(hc_f);
        const __m512 v025hc32 = _mm512_set1_ps(0.25f*hc_f);
        const __m512 vK32 = _mm512_set1_ps(K);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:sums[0:7])
#endif
        for (int i0 = 0; i0 < N; i0 += 16) {
            int nb = (N - i0 < 16) ? (N - i0) : 16;
            avx_rng g; rng_seed(g, (unsigned)((i0 >> 4) * 97 + l) + 1u);
            opt::Moments mom;
            auto accum = [&](double dP, double Pfv, double cost) { mom.add(dP, Pfv, cost); };

            if (l == 0) {
                __m512 xf[5]; for (int i = 0; i < 5; ++i) xf[i] = vK32;
                __m512 dw[5], di[5], sigx[5];
                draw_corr_f32(g, sf_f, dw);
                if (opt == 2) draw_corr_f32(g, si_f, di);
                __m512 b0 = basket32(xf);
                step32(xf, dw, vHf32, sigx);
                __m512 bf = basket32(xf);
                __m512 Af = _mm512_mul_ps(vHalf32, _mm512_mul_ps(vHf32, _mm512_add_ps(b0, bf)));
                if (opt == 2) Af = _mm512_add_ps(Af, wdot32(sigx, di));
                alignas(64) float Aa[16], Ba[16];
                _mm512_store_ps(Aa, Af); _mm512_store_ps(Ba, bf);
                for (int j = 0; j < nb; ++j) { double P = disc_f*pay32(opt,Ba[j],Aa[j]); accum(P,P,(double)nf); }
                mom.flush(sums);
                continue;
            }

            if (l % 2 == 0) {
                __m512 xf[5], xc[5]; for (int i = 0; i < 5; ++i) { xf[i] = vK32; xc[i] = vK32; }
                __m512 Af = _mm512_mul_ps(vHalf32, _mm512_mul_ps(vHf32, basket32(xf)));
                __m512 Ac = _mm512_mul_ps(vHalf32, _mm512_mul_ps(vHc32, basket32(xc)));

                for (int n = 0; n < nc; ++n) {
                    __m512 dw0[5], dw1[5], di0[5], di1[5];
                    draw_corr_f32(g, sf_f, dw0); if (opt == 2) draw_corr_f32(g, si_f, di0);
                    draw_corr_f32(g, sf_f, dw1); if (opt == 2) draw_corr_f32(g, si_f, di1);
                    B::step_pair32(xf, xc, Af, Ac, dw0, dw1, di0, di1,
                                   vHf32, vHc32, v025hc32, R, SIG, W, opt == 2);
                }
                __m512 bf = basket32(xf), bc = basket32(xc);
                if (opt == 2) {
                    Af = _mm512_sub_ps(Af, _mm512_mul_ps(_mm512_mul_ps(vHalf32,vHf32), bf));
                    Ac = _mm512_sub_ps(Ac, _mm512_mul_ps(_mm512_mul_ps(vHalf32,vHc32), bc));
                }
                alignas(64) float Afa[16],Aca[16],Bfa[16],Bca[16];
                _mm512_store_ps(Afa,Af); _mm512_store_ps(Aca,Ac);
                _mm512_store_ps(Bfa,bf); _mm512_store_ps(Bca,bc);
                for (int j = 0; j < nb; ++j) {
                    double Pf = pay32(opt,Bfa[j],Afa[j]), Pc = pay32(opt,Bca[j],Aca[j]);
                    accum(disc_f*(Pf-Pc), disc_f*Pf, (double)nf);
                }
                mom.flush(sums);
            } else {
                // Odd, k>=l_star: pure-fp32 chain run TWICE on the same
                // draws -- fp32-self-consistency check, not a real fp16 gap.
                alignas(64) float Pf_run[2][16], Pc_run[2][16];
                for (int run = 0; run < 2; ++run) {
                    // Reseed identically per run: `g` is shared, so without
                    // this the second run continues the stream and the two
                    // chains see DIFFERENT draws, making dP a difference of
                    // two independent samples (variance ~2*Var(dP)) rather
                    // than the exact zero an fp32-vs-fp32 check must give.
                    rng_seed(g, (unsigned)((i0 >> 4) * 97 + l) + 1u);
                    __m512 xf[5], xc[5]; for (int i = 0; i < 5; ++i) { xf[i] = vK32; xc[i] = vK32; }
                    __m512 Af = _mm512_mul_ps(vHalf32, _mm512_mul_ps(vHf32, basket32(xf)));
                    __m512 Ac = _mm512_mul_ps(vHalf32, _mm512_mul_ps(vHc32, basket32(xc)));

                    for (int n = 0; n < nc; ++n) {
                        __m512 dw0[5], dw1[5], di0[5], di1[5];
                        draw_corr_f32(g, sf_f, dw0); if (opt == 2) draw_corr_f32(g, si_f, di0);
                        draw_corr_f32(g, sf_f, dw1); if (opt == 2) draw_corr_f32(g, si_f, di1);
                        B::step_pair32(xf, xc, Af, Ac, dw0, dw1, di0, di1,
                                       vHf32, vHc32, v025hc32, R, SIG, W, opt == 2);
                    }
                    __m512 bf = basket32(xf), bc = basket32(xc);
                    if (opt == 2) {
                        Af = _mm512_sub_ps(Af, _mm512_mul_ps(_mm512_mul_ps(vHalf32,vHf32), bf));
                        Ac = _mm512_sub_ps(Ac, _mm512_mul_ps(_mm512_mul_ps(vHalf32,vHc32), bc));
                    }
                    alignas(64) float Afa[16],Aca[16],Bfa[16],Bca[16];
                    _mm512_store_ps(Afa,Af); _mm512_store_ps(Aca,Ac);
                    _mm512_store_ps(Bfa,bf); _mm512_store_ps(Bca,bc);
                    for (int j = 0; j < nb; ++j) {
                        Pf_run[run][j] = (float)pay32(opt,Bfa[j],Afa[j]);
                        Pc_run[run][j] = (float)pay32(opt,Bca[j],Aca[j]);
                    }
                }
                for (int j = 0; j < nb; ++j) {
                    double dP0 = disc_f * (Pf_run[0][j] - Pc_run[0][j]);
                    double dP1 = disc_f * (Pf_run[1][j] - Pc_run[1][j]);
                    double dP  = dP0 - dP1;
                    double Pfv = disc_f * Pf_run[0][j];
                    accum(dP, Pfv, 2.0*(double)nf);
                }
                mom.flush(sums);
            }
        }
        return;
    }

    // Pure fp16 path: state, increment, payoff all __m256h (16 lanes, matching
    // the fp32 RNG width), per asset.  Kahan on/off via kahan_mode.
    const __m256h vHalf16 = _mm256_set1_ph((_Float16)0.5f);
    const __m256h vHf16   = _mm256_set1_ph(hf_h);
    const __m256h vHc16   = _mm256_set1_ph(hc_h);
    const __m256h v025hc16= _mm256_set1_ph((_Float16)(0.25f*(float)hc_h));
    const __m256h vK16    = _mm256_set1_ph(K_h);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:sums[0:7])
#endif
    for (int i0 = 0; i0 < N; i0 += 16) {
        int nb = (N - i0 < 16) ? (N - i0) : 16;
        avx_rng g; rng_seed(g, (unsigned)((i0 >> 4) * 131 + l) + 1u);
        opt::Moments mom;
        auto accum = [&](double dP, double Pfv, double cost) { mom.add(dP, Pfv, cost); };
    
        // Y_0 = P_0 = e^{-rT} * payoff( one Milstein step of size h = T ).
        if (l == 0) {
            // Initialise all 5 asset prices S_i(0) = S0 = K for these 16 paths
            __m256h xf[5]; for (int i = 0; i < 5; ++i) xf[i] = vK16;
            // Kahan compensation c_i, one per asset, zero at t = 0
            __m256h xf_c[5]; for (int i = 0; i < 5; ++i) xf_c[i] = _mm256_setzero_ph();

            __m256h dw[5], di[5];
            // Correlated increments dW_i = sqrt(h) (L Z)_i
            draw_corr_fp16(g, (_Float16)sf_f, dw);
            // Asian only: bridge draws dI_i ~ N(0, h^3/12)
            if (opt == 2) draw_corr_fp16(g, (_Float16)si_f, di);

            // Basket value before the step, B(0) = sum_i w_i S_i(0)
            __m256h b0 = basket16(xf);
            __m256h incr[5], sigx[5];
            // Per-asset Milstein increment (sigx returns the PRE-step
            // sigma_i S_i, which the Asian bridge term needs):
            //   dS_i = r S_i h + sigma_i S_i dW_i
            //          + (1/2) sigma_i^2 S_i (dW_i^2 - h)
            mil_incr16_asset(xf, dw, vHf16, incr, sigx);
            // Apply S_i <- S_i + dS_i by compensated summation
            for (int i = 0; i < 5; ++i) kahan_accum16(xf[i], xf_c[i], incr[i]);
            // Terminal basket value B(T) = sum_i w_i S_i(T)
            __m256h bf = basket16(xf);
            // Asian integral over the single step: trapezoid (h/2)(B(0)+B(T))
            __m256h Af = _mm256_mul_ph(vHalf16, _mm256_mul_ph(vHf16, _mm256_add_ph(b0, bf)));
            // plus the WEIGHTED bridge term sum_i w_i (sigma_i S_i) dI_i
            if (opt == 2) Af = _mm256_add_ph(Af, wdot16(sigx, di));

            alignas(64) _Float16 Aa[16], Ba[16];
            _mm256_store_ph(Aa, Af); _mm256_store_ph(Ba, bf);
            for (int j = 0; j < nb; ++j) {
                // P = e^{-rT} max(B(T)-K,0) (European) or e^{-rT} max(A-K,0) (Asian)
                double P = (double)disc_h * pay16(opt, Ba[j], Aa[j]);
                // Level 0: Y = P itself; cost = n_f timesteps
                accum(P, P, (double)nf);
            }
        // l = 1 (odd, k=0): fp32-minus-fp16 precision correction, 1 step.
        //Var(Y_1) is an O(fp16 eps) quantity.
        } else if (l == 1) {

            __m512 dw_f32[5], di_f32[5];
            __m256h dw_h[5], di_h[5];
            draw_corr_coupled(g, sf_f, (_Float16)sf_f, dw_f32, dw_h);
            if (opt == 2) draw_corr_coupled(g, si_f, (_Float16)si_f, di_f32, di_h);

            __m256h xf_h[5]; for (int i = 0; i < 5; ++i) xf_h[i] = vK16;
            __m256h xf_hc[5]; for (int i = 0; i < 5; ++i) xf_hc[i] = _mm256_setzero_ph();
            __m256h b0_h = basket16(xf_h);
            __m256h incr_h[5], sigx_h[5];
            mil_incr16_asset(xf_h, dw_h, vHf16, incr_h, sigx_h);
            for (int i = 0; i < 5; ++i) kahan_accum16(xf_h[i], xf_hc[i], incr_h[i]);
            __m256h bf_h = basket16(xf_h);
            __m256h Af_h = _mm256_mul_ph(vHalf16, _mm256_mul_ph(vHf16, _mm256_add_ph(b0_h, bf_h)));
            if (opt == 2) Af_h = _mm256_add_ph(Af_h, wdot16(sigx_h, di_h));

            // fp32 reference chain: unchanged, pure fp32.
            __m512 xf_f[5]; for (int i = 0; i < 5; ++i) xf_f[i] = _mm512_set1_ps(K);
            __m512 b0_f = basket32(xf_f);
            __m512 sigx_f[5];
            step32(xf_f, dw_f32, _mm512_set1_ps(hf_f), sigx_f);
            __m512 bf_f = basket32(xf_f);
            __m512 Af_f = _mm512_mul_ps(_mm512_set1_ps(0.5f), _mm512_mul_ps(_mm512_set1_ps(hf_f), _mm512_add_ps(b0_f, bf_f)));
            if (opt == 2) Af_f = _mm512_add_ps(Af_f, wdot32(sigx_f, di_f32));

            alignas(64) _Float16 Aah[16], Bah[16];
            alignas(64) float    Aaf[16], Baf[16];
            _mm256_store_ph(Aah, Af_h); _mm256_store_ph(Bah, bf_h);
            _mm512_store_ps(Aaf, Af_f); _mm512_store_ps(Baf, bf_f);
            for (int j = 0; j < nb; ++j) {
                // The same payoff, evaluated in each precision on the same draws
                double dP_h = (double)disc_h * pay16(opt, Bah[j], Aah[j]);
                double dP_f = (double)disc_f * pay32(opt, Baf[j], Aaf[j]);
                // Y_1 = P^{f32} - P^{fp16}: the precision correction
                double dP   = dP_f - dP_h;
                // Cost 2*n_f: both precision chains are simulated
                accum(dP, dP, 2.0*(double)nf);
            }
        // l even >= 2: fp16 Milstein MLMC correction (nf fine, nc coarse).
        // Y_l = P^h_f - P^h_c, the fine path (n_f steps of h_f) minus the
        // coarse path (n_c steps of h_c = 2h_f) driven by the SAME correlated
        // increments.  That coupling gives Var(Y_l) = O(h^2), i.e. beta ~ 2.
        } else if (l % 2 == 0) {
            // Both paths start at S_i(0) = K for all 5 assets
            __m256h xf[5], xc[5]; for (int i = 0; i < 5; ++i) { xf[i] = vK16; xc[i] = vK16; }
            // Per-asset Kahan compensation terms for both paths
            __m256h xf_c[5], xc_c[5];
            for (int i = 0; i < 5; ++i) { xf_c[i] = _mm256_setzero_ph(); xc_c[i] = _mm256_setzero_ph(); }
            // Asian sums pre-loaded with the trapezoid's first half-term
            // (h/2)B(0); the matching -(h/2)B(T) is subtracted after the loop
            __m256h Af = _mm256_mul_ph(vHalf16, _mm256_mul_ph(vHf16, basket16(xf)));
            __m256h Ac = _mm256_mul_ph(vHalf16, _mm256_mul_ph(vHc16, basket16(xc)));
            __m256h Af_c = _mm256_setzero_ph(), Ac_c = _mm256_setzero_ph();

            // One iteration = one COARSE step = two FINE steps
            for (int n = 0; n < nc; ++n) {
                __m256h dw0[5], dw1[5], di0[5], di1[5];
                // Two fine correlated increments ~ N(0, h_f Sigma); the coarse
                // step is driven by their sum ~ N(0, h_c Sigma) -- the shared
                // randomness that couples the fine and coarse paths.
                draw_corr_fp16(g, (_Float16)sf_f, dw0); if (opt == 2) draw_corr_fp16(g, (_Float16)si_f, di0);
                draw_corr_fp16(g, (_Float16)sf_f, dw1); if (opt == 2) draw_corr_fp16(g, (_Float16)si_f, di1);
                __m256h dwc[5], ddw[5];
                for (int i = 0; i < 5; ++i) { dwc[i] = _mm256_add_ph(dw0[i],dw1[i]); ddw[i] = _mm256_sub_ph(dw0[i],dw1[i]); }

                // Advance both paths one coarse step: two fine Milstein steps
                // per asset (dw0, then dw1) and one coarse Milstein step
                // (dw0+dw1), updating the Asian sums along the way.  All state
                // updates route through the Kahan accumulators.
                B::step_pair16(xf, xc, Af, Ac, xf_c, xc_c, Af_c, Ac_c,
                               dw0, dw1, di0, di1,
                               kahan_accum16, kahan_accum16_asset,
                               vHf16, vHc16, v025hc16, R, SIG, W, opt == 2);
            }

            // Terminal basket values B(T) for the fine and coarse paths
            __m256h bf = basket16(xf), bc = basket16(xc);
            __m256h Aft = Af, Act = Ac;
            // Asian only: close the trapezoid by subtracting (h/2)B(T)
            if (opt == 2) {
                Aft = _mm256_sub_ph(Af, _mm256_mul_ph(vHalf16, _mm256_mul_ph(vHf16, bf)));
                Act = _mm256_sub_ph(Ac, _mm256_mul_ph(vHalf16, _mm256_mul_ph(vHc16, bc)));
            }

            alignas(64) _Float16 Aa[16], Aca[16], Ba[16], Bca[16];
            _mm256_store_ph(Aa, Aft); _mm256_store_ph(Aca, Act);
            _mm256_store_ph(Ba, bf);  _mm256_store_ph(Bca, bc);
            for (int j = 0; j < nb; ++j) {
                // Fine and coarse payoffs from the same driving randomness
                double Pf  = pay16(opt, Ba[j], Aa[j]);
                double Pc  = pay16(opt, Bca[j], Aca[j]);
                // Y_l = e^{-rT}(P_f - P_c): the MLMC Milstein correction
                double dP  = (double)disc_h * (Pf - Pc);
                // P_f alone, for the telescoping-sum consistency check
                double Pfv = (double)disc_h * Pf;
                accum(dP, Pfv, (double)nf);   // cost = n_f fine timesteps
            }

        // l odd >= 3: fp32-minus-fp16 precision correction.
        //
        //   Y_l = (P^{f32}_f - P^{f32}_c) - (P^{fp16}_f - P^{fp16}_c)
        } else {
            __m256h xf_h[5], xc_h[5]; for (int i = 0; i < 5; ++i) { xf_h[i] = vK16; xc_h[i] = vK16; }
            __m256h xf_hc[5], xc_hc[5];
            for (int i = 0; i < 5; ++i) { xf_hc[i] = _mm256_setzero_ph(); xc_hc[i] = _mm256_setzero_ph(); }
            __m256h Af_h = _mm256_mul_ph(vHalf16, _mm256_mul_ph(vHf16, basket16(xf_h)));
            __m256h Ac_h = _mm256_mul_ph(vHalf16, _mm256_mul_ph(vHc16, basket16(xc_h)));
            __m256h Af_hc = _mm256_setzero_ph(), Ac_hc = _mm256_setzero_ph();

            __m512 xf_f[5], xc_f[5]; for (int i = 0; i < 5; ++i) { xf_f[i] = _mm512_set1_ps(K); xc_f[i] = _mm512_set1_ps(K); }
            __m512 Af_f = _mm512_mul_ps(_mm512_set1_ps(0.5f), _mm512_mul_ps(_mm512_set1_ps(hf_f), basket32(xf_f)));
            __m512 Ac_f = _mm512_mul_ps(_mm512_set1_ps(0.5f), _mm512_mul_ps(_mm512_set1_ps(hc_f), basket32(xc_f)));

            for (int n = 0; n < nc; ++n) {
                // fp32 and fp16 each generate their own correlated Normals,
                // independently, from the same underlying Philox bits per
                // asset (draw_corr_coupled) -- not narrowed one from the other.
                __m512 dw0_f32[5], dw1_f32[5], di0_f32[5], di1_f32[5];
                __m256h dw0_h[5], dw1_h[5], di0_h[5], di1_h[5];
                draw_corr_coupled(g, sf_f, (_Float16)sf_f, dw0_f32, dw0_h);
                if (opt == 2) draw_corr_coupled(g, si_f, (_Float16)si_f, di0_f32, di0_h);
                draw_corr_coupled(g, sf_f, (_Float16)sf_f, dw1_f32, dw1_h);
                if (opt == 2) draw_corr_coupled(g, si_f, (_Float16)si_f, di1_f32, di1_h);
                __m256h dwc_h[5], ddw_h[5];
                for (int i = 0; i < 5; ++i) { dwc_h[i] = _mm256_add_ph(dw0_h[i],dw1_h[i]); ddw_h[i] = _mm256_sub_ph(dw0_h[i],dw1_h[i]); }
                __m512 dwc_f[5], ddw_f[5];
                for (int i = 0; i < 5; ++i) { dwc_f[i] = _mm512_add_ps(dw0_f32[i],dw1_f32[i]); ddw_f[i] = _mm512_sub_ps(dw0_f32[i],dw1_f32[i]); }

                // fp16 chain: the whole coupled pair in one call. 
                B::step_pair16(xf_h, xc_h, Af_h, Ac_h, xf_hc, xc_hc, Af_hc, Ac_hc,
                               dw0_h, dw1_h, di0_h, di1_h,
                               kahan_accum16, kahan_accum16_asset,
                               vHf16, vHc16, v025hc16, R, SIG, W, opt == 2);

                // ---- fp32 fine step 0 (unchanged reference) ----
                __m512 s0_f[5]; step32(xf_f, dw0_f32, _mm512_set1_ps(hf_f), s0_f);
                if (opt == 2) Af_f = _mm512_add_ps(Af_f, _mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(hf_f),basket32(xf_f)), wdot32(s0_f,di0_f32)));

                // ---- fp32 fine step 1 (unchanged reference) ----
                __m512 s1_f[5]; step32(xf_f, dw1_f32, _mm512_set1_ps(hf_f), s1_f);
                if (opt == 2) Af_f = _mm512_add_ps(Af_f, _mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(hf_f),basket32(xf_f)), wdot32(s1_f,di1_f32)));

                // ---- fp32 coarse step (unchanged reference) ----
                __m512 sc_f[5]; step32(xc_f, dwc_f, _mm512_set1_ps(hc_f), sc_f);
                if (opt == 2) {
                    __m512 br_f[5];
                    for (int i = 0; i < 5; ++i) br_f[i] = _mm512_add_ps(_mm512_add_ps(di0_f32[i],di1_f32[i]), _mm512_mul_ps(_mm512_set1_ps(0.25f*hc_f),ddw_f[i]));
                    Ac_f = _mm512_add_ps(Ac_f, _mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(hc_f),basket32(xc_f)), wdot32(sc_f,br_f)));
                }
            }

            __m256h bf_h = basket16(xf_h), bc_h = basket16(xc_h);
            __m512  bf_f = basket32(xf_f), bc_f = basket32(xc_f);
            __m256h Af_ht = Af_h, Ac_ht = Ac_h;
            __m512  Af_ft = Af_f, Ac_ft = Ac_f;
            if (opt == 2) {
                Af_ht = _mm256_sub_ph(Af_h, _mm256_mul_ph(vHalf16, _mm256_mul_ph(vHf16, bf_h)));
                Ac_ht = _mm256_sub_ph(Ac_h, _mm256_mul_ph(vHalf16, _mm256_mul_ph(vHc16, bc_h)));
                Af_ft = _mm512_sub_ps(Af_f, _mm512_mul_ps(_mm512_set1_ps(0.5f), _mm512_mul_ps(_mm512_set1_ps(hf_f), bf_f)));
                Ac_ft = _mm512_sub_ps(Ac_f, _mm512_mul_ps(_mm512_set1_ps(0.5f), _mm512_mul_ps(_mm512_set1_ps(hc_f), bc_f)));
            }

            alignas(64) _Float16 Ahf[16], Ahc[16], Bhf[16], Bhc[16];
            alignas(64) float    Aff[16], Afc[16], Bff[16], Bfc[16];
            _mm256_store_ph(Ahf, Af_ht); _mm256_store_ph(Ahc, Ac_ht);
            _mm256_store_ph(Bhf, bf_h);  _mm256_store_ph(Bhc, bc_h);
            _mm512_store_ps(Aff, Af_ft); _mm512_store_ps(Afc, Ac_ft);
            _mm512_store_ps(Bff, bf_f);  _mm512_store_ps(Bfc, bc_f);

            for (int j = 0; j < nb; ++j) {
                // Four payoffs: {fine, coarse} x {fp16, fp32}
                double Ph_fine = pay16(opt, Bhf[j], Ahf[j]);
                double Ph_cors = pay16(opt, Bhc[j], Ahc[j]);
                double Pf_fine = pay32(opt, Bff[j], Aff[j]);
                double Pf_cors = pay32(opt, Bfc[j], Afc[j]);

                // The MLMC correction as each precision computes it
                double dP_h = (double)disc_h * (Ph_fine - Ph_cors);
                double dP_f = disc_f * (Pf_fine - Pf_cors);
                // Y_l = fp32 correction - fp16 correction: pure rounding gap
                double dP   = dP_f - dP_h;
                // Pfv must be a PAYOFF value (fp16 convention, matching the
                // even levels), not a precision gap -- see CLAUDE.md, the
                // 2026-07-31 sums[5] unit-mismatch fix.
                double Pfv  = (double)disc_h*Ph_fine;
                // Cost 2*n_f: both precision chains are simulated
                accum(dP, Pfv, 2.0*(double)nf);
            }
        }
        mom.flush(sums);
    }
}
