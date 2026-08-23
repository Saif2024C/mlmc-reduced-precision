/*
 * nested_scalar_milstein_fp16_avx512.cpp -- unified AVX-512 scalar fp16
 * template.  1-asset GBM; option 1 = Asian, 2 = Lookback.
 *
 * Path state (X, Asian running sum A, Lookback minimum M, payoff) is ALL
 * __m256h -- genuine 16-wide fp16 arithmetic, not fp32 with narrowed storage.
 *
 * Per-path maths lives in options_mm512.h, shared with the basket template
 * and the benchmarks; the forwarders below bind this file's globals to it.
 *
 * OpenMP is unconditional (active iff _OPENMP): one Philox stream per 16-lane
 * block, per-block accumulators reduced into sums[].  No separate _omp file.
 *
 * Build/run:  make nsh_avx  (add -qopenmp -> nsh_omp);  see src/AVX-512/Makefile.
 * Writes nested_scalar_fp16_avx512_{nokahan,kahan,adaptive}_{1,2}.txt to cwd.
 *
 *==========================================================================
 * 1. MATHEMATICAL OBJECTIVE
 *==========================================================================
 *
 * The underlying is a single GBM asset under the risk-neutral measure
 *
 *       dS = r S dt + sigma S dW ,        S(0) = S0 = K
 *
 * and the two payoffs priced are
 *
 *   option 1, ASIAN call:      P = e^{-rT} max( Abar - K , 0 ),
 *                              Abar = (1/T) int_0^T S(t) dt
 *
 *   option 2, LOOKBACK call:   P = e^{-rT} ( S(T) - min_{0<=t<=T} S(t) )
 *
 * Both are path-DEPENDENT: the payoff needs the whole trajectory, not just
 * S(T), which is why the state carries a running time-integral A and a
 * running minimum M alongside the path position X.
 *
 * ---- Discretisation ----
 * MILSTEIN, step h:
 *
 *   X_{n+1} = X_n + r X_n h + sigma X_n dW_n
 *                 + (1/2) sigma^2 X_n ( dW_n^2 - h )
 *
 * The (dW^2 - h) term is what raises the strong order from O(sqrt h) (Euler)
 * to O(h), and hence gives the MLMC variance decay beta ~ 2.  Note E[dW^2-h]
 * = 0: the term is a pure cancellation, which is exactly why quantising dW
 * itself to fp16 destroys it (see the header notes and CLAUDE.md).
 *
 * A (Asian) uses the trapezoidal rule plus a Brownian-bridge correction that
 * captures the sub-step variation of the integral:
 *
 *       A += h X_{n+1} + (sigma X_n) dI_n ,   dI_n ~ N(0, h^3/12)
 *
 * with a final -(h/2) X_T applied at the end so the sum telescopes to the
 * trapezoid  A = sum h (X_n + X_{n+1})/2  plus the bridge terms.
 *
 * M (Lookback) uses the exact Brownian-bridge minimum over each step: given
 * the endpoints X_n, X_{n+1} and a log-uniform L = log U < 0,
 *
 *   m_n = (1/2)[ X_n + X_{n+1} - sqrt( (X_{n+1}-X_n)^2 - 2 h (sigma X_n)^2 L ) ]
 *   M   = min(M, m_n)
 *
 * which is unbiased for the continuous minimum of the bridged path and is
 * what preserves the fine/coarse coupling for the lookback payoff.
 *
 * ---- The NESTED (super-level) MLMC telescoping sum ----
 * The estimator index l runs over SUPER-levels; k = l/2 is the grid level,
 * n_f = 2^k fine steps, n_c = n_f/2 coarse steps.  The driver receives
 * Y_l = the level-l correction, and E[P] = sum_l E[Y_l]:
 *
 *   l = 0        Y_0 = P^{h}_0             (fp16 base level, 1 step)
 *   l even >= 2  Y_l = P^{h}_f - P^{h}_c   (Milstein MLMC correction, fp16)
 *   l odd        Y_l = P^{f32} - P^{h}     (PRECISION correction: the same
 *                                           payoff evaluated in fp32 minus
 *                                           in fp16, on the same draws)
 *
 * So the even levels correct the DISCRETISATION error (h -> h/2) and the odd
 * levels correct the PRECISION error (fp16 -> fp32).  Both telescope, and the
 * sum is an unbiased estimate of the fp32-accurate price.  The odd levels are
 * cheap precisely because fp16 and fp32 on identical draws differ only by
 * rounding, so Var(Y_odd) is tiny -- that is the whole point of the design.
 *
 * MLMC's three rates, which every diagnostic in the output table targets:
 *       |E[P - P_l]| = O(2^{-alpha l}),  alpha ~ 1
 *       Var(Y_l)     = O(2^{-beta l}),   beta  ~ 2 for Milstein
 *       C_l          = O(2^{gamma l}),   gamma ~ 1 (cost = timestep count)
 *==========================================================================
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

namespace S = opt::scalar;   // per-path maths: options_mm512.h

int option;                                   // 1 = Asian, 2 = Lookback
// K = strike AND S(0) (at-the-money); T = maturity; r = risk-free rate (drift
// and discount); sig = sigma, the GBM volatility.
static const float K = 100.0f, T = 1.0f, r = 0.05f, sig = 0.2f;

void nested_scalar_fp16_avx_l(int, int, double *);

// Runtime Kahan on/off switch -- see kahan_accum16() below.

int main(int argc, char **argv)
{
    // l_star per option: re-derived from this file's own same-RNG
    // fp16-Kahan-vs-fp32 ratio crossover (>2x).
    static const float EPS[] = { 0.005f, 0.01f, 0.02f, 0.05f, 0.1f, 0.0f };

    SweepConfig cfg;
    cfg.prefix      = "nested_scalar_fp16_avx512";
    cfg.family      = "scalar";
    cfg.opt_name[0] = "Asian";
    cfg.opt_name[1] = "Lookback";
    cfg.estimator   = nested_scalar_fp16_avx_l;
    cfg.eps         = EPS;
    cfg.l_star[1]   = 5;   // Asian
    cfg.l_star[2]   = 7;   // Lookback
    return run_sweep(cfg, argc, argv);
}

// fp32 Milstein helpers (16-wide __m512), used only above the l_star cutoff.
// Thin forwarders onto options_mm512.h: bind this file's globals (r/sig/K) and
// the kahan_mode switch, which the header deliberately does not know about.
typedef S::C32 FC32;
typedef S::C16 FC16;

static inline FC32 make_fc32(float h)    { return S::make32(r, sig, h); }
static inline FC16 make_fc16(_Float16 h) { return S::make16(r, sig, h); }

// mil_incr32/16 and minbr32/16 are used unwrapped as S::mil_incr32 etc --
// a forwarder here would duplicate the header's signature exactly (FC32 is
// S::C32), making every call site ambiguous by ADL.

static inline double pay32(int opt, float A, float X, float M) {
    return S::pay32(opt, K, A, X, M);
}
static inline double pay16(int opt, _Float16 A, _Float16 X, _Float16 M) {
    return S::pay16(opt, (_Float16)K, A, X, M);
}

// kahan_mode picks the compensated or plain accumulator.
static inline void kahan_accum16(__m256h &sum, __m256h &comp, __m256h term) {
    if (kahan_mode) opt::kahan_accum16(sum, comp, term);
    else            opt::plain_accum16(sum, comp, term);
}

static inline __m256h narrow16(__m512 x_f32) { return opt::narrow16(x_f32); }

/* ------------------------------------------------------------------
 * sums[] layout: [0] cost, [1] Y, [2] Y^2, [3] Y^3, [4] Y^4,
 *                [5] Pf, [6] Pf^2  (all fp64)
 * ------------------------------------------------------------------ */
void nested_scalar_fp16_avx_l(int l, int N, double *sums)
{
    const int k  = l / 2;        // grid level: two super-levels per grid level
    const int nf = 1 << k;       // n_f = 2^k fine timesteps
    const int nc = nf / 2;       // n_c = n_f/2 coarse timesteps (0 at k=0)

    const float hf_f   = T / (float)nf;              // fine step   h_f = T/n_f
    const float hc_f   = (nc > 0) ? T / (float)nc : 0.0f;  // coarse step h_c = 2h_f
    const float disc_f = std::exp(-r * T);           // discount factor e^{-rT}
    // sd of the Asian bridge draw dI ~ N(0, h^3/12):  si = sqrt(h/12) * h
    const float si_f   = sqrtf(hf_f / 12.0f) * hf_f;

    const _Float16 hf_h   = (_Float16)hf_f;
    const _Float16 hc_h   = (_Float16)hc_f;
    const _Float16 disc_h = (_Float16)disc_f;
    const _Float16 K_h    = (_Float16)K;

    for (int i = 0; i < 7; ++i) sums[i] = 0.0;

    const int opt = option;

    // Adaptive cutoff, k >= l_star: pure fp32.  Odd levels run the fp32 chain
    // twice on the same draws -- a self-consistency check, not a real gap.
    
    if (adaptive_mode && k >= l_star) {
        const FC32 cf32 = make_fc32(hf_f), cc32 = make_fc32(hc_f);
        const __m512 vsi32 = _mm512_set1_ps(si_f), vHalf32 = _mm512_set1_ps(0.5f);
        const __m512 vHf32 = _mm512_set1_ps(hf_f), vHc32 = _mm512_set1_ps(hc_f);
        const __m512 v025hc32 = _mm512_set1_ps(0.25f*hc_f), vSig32 = _mm512_set1_ps(sig);
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
                __m512 dW  = _mm512_mul_ps(cf32.sqhf, normal16_fp32(g));
                __m512 dI  = _mm512_mul_ps(vsi32, normal16_fp32(g));
                __m512 Lrv = loguniform16_fp32(g);
                __m512 X0  = vK32, v = _mm512_mul_ps(vSig32, X0);
                __m512 Xf  = _mm512_add_ps(X0, S::mil_incr32(X0, dW, cf32));
                __m512 Af  = _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(vHalf32,vHf32), _mm512_add_ps(X0,Xf)),
                                            _mm512_mul_ps(v, dI));
                __m512 Mf  = S::minbr32(vK32, X0, Xf, v, Lrv, cf32);
                alignas(64) float Aa[16], Xa[16], Ma[16];
                _mm512_store_ps(Aa, Af); _mm512_store_ps(Xa, Xf); _mm512_store_ps(Ma, Mf);
                for (int j = 0; j < nb; ++j) { double P = disc_f*pay32(opt,Aa[j],Xa[j],Ma[j]); accum(P,P,(double)nf); }
                mom.flush(sums);
                continue;
            }

            if (l % 2 == 0) {
                __m512 Xf=vK32, Xc=vK32;
                __m512 Af=_mm512_mul_ps(vHalf32,_mm512_mul_ps(vHf32,vK32));
                __m512 Ac=_mm512_mul_ps(vHalf32,_mm512_mul_ps(vHc32,vK32));
                __m512 Mf=vK32, Mc=vK32;

                for (int n = 0; n < nc; ++n) {
                    S::Pair32 d;
                    d.dW0 = _mm512_mul_ps(cf32.sqhf, normal16_fp32(g));
                    d.dW1 = _mm512_mul_ps(cf32.sqhf, normal16_fp32(g));
                    d.dI0 = _mm512_mul_ps(vsi32, normal16_fp32(g));
                    d.dI1 = _mm512_mul_ps(vsi32, normal16_fp32(g));
                    d.Lrv0 = loguniform16_fp32(g);
                    d.Lrv1 = loguniform16_fp32(g);
                    S::step_pair32(Xf, Xc, Af, Ac, Mf, Mc, d,
                                   vSig32, vHf32, vHc32, v025hc32, cf32, cc32);
                }
                Af=_mm512_sub_ps(Af, _mm512_mul_ps(_mm512_mul_ps(vHalf32,vHf32),Xf));
                Ac=_mm512_sub_ps(Ac, _mm512_mul_ps(_mm512_mul_ps(vHalf32,vHc32),Xc));
                alignas(64) float Afa[16],Aca[16],Xfa[16],Xca[16],Mfa[16],Mca[16];
                _mm512_store_ps(Afa,Af);_mm512_store_ps(Aca,Ac);_mm512_store_ps(Xfa,Xf);
                _mm512_store_ps(Xca,Xc);_mm512_store_ps(Mfa,Mf);_mm512_store_ps(Mca,Mc);
                for (int j = 0; j < nb; ++j) {
                    double Pf=pay32(opt,Afa[j],Xfa[j],Mfa[j]), Pc=pay32(opt,Aca[j],Xca[j],Mca[j]);
                    accum(disc_f*(Pf-Pc), disc_f*Pf, (double)nf);
                }
                mom.flush(sums);
            } else {
                // Odd, k>=l_star: fp32 chain twice on the same draws.  Per-lane
                // payoffs are kept side by side so each lane contributes its own diff.
                alignas(64) float Pf_run[2][16], Pc_run[2][16];
                for (int run = 0; run < 2; ++run) {
                    __m512 Xf=vK32, Xc=vK32;
                    __m512 Af=_mm512_mul_ps(vHalf32,_mm512_mul_ps(vHf32,vK32));
                    __m512 Ac=_mm512_mul_ps(vHalf32,_mm512_mul_ps(vHc32,vK32));
                    __m512 Mf=vK32, Mc=vK32;
                    for (int n = 0; n < nc; ++n) {
                        S::Pair32 d;
                        d.dW0 = _mm512_mul_ps(cf32.sqhf, normal16_fp32(g));
                        d.dW1 = _mm512_mul_ps(cf32.sqhf, normal16_fp32(g));
                        d.dI0 = _mm512_mul_ps(vsi32, normal16_fp32(g));
                        d.dI1 = _mm512_mul_ps(vsi32, normal16_fp32(g));
                        d.Lrv0 = loguniform16_fp32(g);
                        d.Lrv1 = loguniform16_fp32(g);
                        S::step_pair32(Xf, Xc, Af, Ac, Mf, Mc, d,
                                       vSig32, vHf32, vHc32, v025hc32, cf32, cc32);
                    }
                    Af=_mm512_sub_ps(Af, _mm512_mul_ps(_mm512_mul_ps(vHalf32,vHf32),Xf));
                    Ac=_mm512_sub_ps(Ac, _mm512_mul_ps(_mm512_mul_ps(vHalf32,vHc32),Xc));
                    alignas(64) float Afa[16],Aca[16],Xfa[16],Xca[16],Mfa[16],Mca[16];
                    _mm512_store_ps(Afa,Af);_mm512_store_ps(Aca,Ac);_mm512_store_ps(Xfa,Xf);
                    _mm512_store_ps(Xca,Xc);_mm512_store_ps(Mfa,Mf);_mm512_store_ps(Mca,Mc);
                    for (int j = 0; j < nb; ++j) {
                        Pf_run[run][j] = (float)pay32(opt,Afa[j],Xfa[j],Mfa[j]);
                        Pc_run[run][j] = (float)pay32(opt,Aca[j],Xca[j],Mca[j]);
                    }
                }
                for (int j = 0; j < nb; ++j) {
                    double dP0  = disc_f * (Pf_run[0][j] - Pc_run[0][j]);
                    double dP1  = disc_f * (Pf_run[1][j] - Pc_run[1][j]);
                    double dP   = dP0 - dP1;
                    double Pfv  = disc_f * Pf_run[0][j];
                    accum(dP, Pfv, 2.0*(double)nf);
                }
                mom.flush(sums);
            }
        }
        return;
    }

    // Pure fp16 path: state, increment, payoff all __m256h (16 lanes, matching
    // the fp32 RNG width).  Kahan on/off via kahan_mode.
    const FC16 cf16 = make_fc16(hf_h), cc16 = make_fc16(hc_h);
    const __m256h vsi16   = _mm256_set1_ph((_Float16)si_f);
    const __m256h vHalf16 = _mm256_set1_ph((_Float16)0.5f);
    const __m256h vHf16   = _mm256_set1_ph(hf_h);
    const __m256h vHc16   = _mm256_set1_ph(hc_h);
    const __m256h v025hc16= _mm256_set1_ph((_Float16)(0.25f*(float)hc_h));
    const __m256h vSig16  = _mm256_set1_ph((_Float16)sig);
    const __m256h vK16    = _mm256_set1_ph(K_h);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:sums[0:7])
#endif
    for (int i0 = 0; i0 < N; i0 += 16) {
        int nb = (N - i0 < 16) ? (N - i0) : 16;
        avx_rng g; rng_seed(g, (unsigned)((i0 >> 4) * 131 + l) + 1u);
        opt::Moments mom;
        auto accum = [&](double dP, double Pfv, double cost) { mom.add(dP, Pfv, cost); };
        // l = 0 (even, k=0): fp16 Milstein base level, 1 fine step.
        // Y_0 = P_0 = e^{-rT} * payoff( one Milstein step of size h = T ).
        if (l == 0) {
            // dW ~ N(0,h): scale a standard Normal by sqrt(h)
            __m256h dW  = _mm256_mul_ph(_mm256_set1_ph((_Float16)sqrtf(hf_f)), normal16_fp16(g));
            // dI ~ N(0,h^3/12): the Asian time-integral bridge draw
            __m256h dI  = _mm256_mul_ph(_mm256_set1_ph((_Float16)si_f), normal16_fp16(g));
            // L = log U < 0: drives the exact Brownian-bridge minimum
            __m256h Lrv = narrow16(loguniform16_fp32(g));

            // X(0) = K;  v = sigma X, the local diffusion coefficient
            __m256h X0 = vK16, v = _mm256_mul_ph(vSig16, X0);
            // Milstein: X(T) = X0 + r X0 h + sigma X0 dW + (1/2)sigma^2 X0 (dW^2-h)
            __m256h Xf = _mm256_add_ph(X0, S::mil_incr16(X0, dW, cf16));
            // Asian integral over the single step: trapezoid (h/2)(X0+X_T)
            // plus the bridge correction (sigma X0) dI
            __m256h Af = _mm256_add_ph(_mm256_mul_ph(_mm256_mul_ph(vHalf16,vHf16), _mm256_add_ph(X0,Xf)),
                                        _mm256_mul_ph(v, dI));
            // Running minimum: min(X0, bridge minimum over [0,T])
            __m256h Mf = S::minbr16(vK16, X0, Xf, v, Lrv, cf16);

            alignas(64) _Float16 Aa[16], Xa[16], Ma[16];
            _mm256_store_ph(Aa, Af); _mm256_store_ph(Xa, Xf); _mm256_store_ph(Ma, Mf);
            for (int j = 0; j < nb; ++j) {
                // P = e^{-rT} max(A-K,0)  (Asian)  or  e^{-rT}(X_T - M)  (Lookback)
                double P = (double)disc_h * pay16(opt, Aa[j], Xa[j], Ma[j]);
                // Level 0: Y = P itself; cost = n_f timesteps
                accum(P, P, (double)nf);
            }
        // l = 1 (odd, k=0): fp32-minus-fp16 precision correction, 1 step.
        // Y_1 = P^{f32} - P^{fp16}, both from ONE Milstein step on the SAME
        // Philox bits.  Since the two chains differ only by rounding, Y_1 is
        // an O(fp16 eps) quantity with correspondingly tiny variance.
        } else if (l == 1) {
            // fp32/fp16 Normals independently transformed from the same Philox
            // bits (normal16_coupled), not narrowed one from the other.  The
            // log-uniform term has no fp16 spline, so it stays fp32-generated.
            __m512 zW_f32, zI_f32; __m256h zW_f16, zI_f16;
            normal16_coupled(g, zW_f32, zW_f16);
            normal16_coupled(g, zI_f32, zI_f16);
            __m512 dW_f32  = _mm512_mul_ps(_mm512_set1_ps(sqrtf(hf_f)), zW_f32);
            __m512 dI_f32  = _mm512_mul_ps(_mm512_set1_ps(si_f), zI_f32);
            __m512 Lrv_f32 = loguniform16_fp32(g);

            __m256h dW  = _mm256_mul_ph(_mm256_set1_ph((_Float16)sqrtf(hf_f)),
                                         zW_f16);
            __m256h dI  = _mm256_mul_ph(_mm256_set1_ph((_Float16)si_f),
                                         zI_f16);
            __m256h Lrv = narrow16(Lrv_f32);

            __m256h X0h = vK16, vh = _mm256_mul_ph(vSig16, X0h);
            __m256h Xfh = _mm256_add_ph(X0h, S::mil_incr16(X0h, dW, cf16));
            __m256h Afh = _mm256_add_ph(_mm256_mul_ph(_mm256_mul_ph(vHalf16,vHf16), _mm256_add_ph(X0h,Xfh)),
                                         _mm256_mul_ph(vh, dI));
            __m256h Mfh = S::minbr16(vK16, X0h, Xfh, vh, Lrv, cf16);

            // fp32 reference chain: unchanged, pure fp32.
            const FC32 cf32 = make_fc32(hf_f);
            __m512 X0f = _mm512_set1_ps(K), vf = _mm512_mul_ps(_mm512_set1_ps(sig), X0f);
            __m512 Xff = _mm512_add_ps(X0f, S::mil_incr32(X0f, dW_f32, cf32));
            __m512 Aff = _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.5f),_mm512_set1_ps(hf_f)),
                                                       _mm512_add_ps(X0f,Xff)),
                                        _mm512_mul_ps(vf, dI_f32));
            __m512 Mff = S::minbr32(_mm512_set1_ps(K), X0f, Xff, vf, Lrv_f32, cf32);

            alignas(64) _Float16 Aah[16], Xah[16], Mah[16];
            alignas(64) float    Aaf[16], Xaf[16], Maf[16];
            _mm256_store_ph(Aah, Afh); _mm256_store_ph(Xah, Xfh); _mm256_store_ph(Mah, Mfh);
            _mm512_store_ps(Aaf, Aff); _mm512_store_ps(Xaf, Xff); _mm512_store_ps(Maf, Mff);
            for (int j = 0; j < nb; ++j) {
                // The same payoff, evaluated in each precision on the same draws
                double dP_h = (double)disc_h * pay16(opt, Aah[j], Xah[j], Mah[j]);
                double dP_f = (double)disc_f * pay32(opt, Aaf[j], Xaf[j], Maf[j]);
                // Y_1 = P^{f32} - P^{fp16}: the precision correction
                double dP   = dP_f - dP_h;
                // Cost 2*n_f: both chains are simulated on this level
                accum(dP, dP, 2.0*(double)nf);
            }
        // l even >= 2: fp16 Milstein MLMC correction (nf fine, nc coarse).
        // Y_l = P^h_f - P^h_c, the fine path (n_f steps of h_f) minus the
        // coarse path (n_c steps of h_c = 2h_f) driven by the SAME Brownian
        // increments.  That coupling is what makes Var(Y_l) = O(h^2), i.e.
        // beta ~ 2; give the two paths independent draws and it collapses.
        } else if (l % 2 == 0) {
            // Both paths start at X(0) = K
            __m256h Xf = vK16, Xc = vK16;
            // Asian sums pre-loaded with the trapezoid's first half-term
            // (h/2)X(0); the matching -(h/2)X(T) is subtracted after the loop
            __m256h Af = _mm256_mul_ph(vHalf16, _mm256_mul_ph(vHf16, vK16));
            __m256h Ac = _mm256_mul_ph(vHalf16, _mm256_mul_ph(vHc16, vK16));
            // Running minima start at X(0)
            __m256h Mf = vK16, Mc = vK16;
            // Kahan compensation terms for the four running fp16 accumulators
            __m256h Xf_c = _mm256_setzero_ph(), Xc_c = _mm256_setzero_ph();
            __m256h Af_c = _mm256_setzero_ph(), Ac_c = _mm256_setzero_ph();

            // One iteration = one COARSE step = two FINE steps
            for (int n = 0; n < nc; ++n) {
                // The two fine Brownian increments dW0, dW1 ~ N(0,h_f).  The
                // coarse step is driven by their sum dW0+dW1 ~ N(0,h_c) --
                // the shared randomness that couples the two paths.
                __m256h dW0 = _mm256_mul_ph(_mm256_set1_ph((_Float16)sqrtf(hf_f)), normal16_fp16(g));
                __m256h dW1 = _mm256_mul_ph(_mm256_set1_ph((_Float16)sqrtf(hf_f)), normal16_fp16(g));
                // Asian bridge draws dI ~ N(0, h_f^3/12), one per fine step
                __m256h dI0 = _mm256_mul_ph(_mm256_set1_ph((_Float16)si_f), normal16_fp16(g));
                __m256h dI1 = _mm256_mul_ph(_mm256_set1_ph((_Float16)si_f), normal16_fp16(g));
                // log-uniforms for the two fine-step bridge minima
                __m256h Lrv0 = narrow16(loguniform16_fp32(g));
                __m256h Lrv1 = narrow16(loguniform16_fp32(g));

                // Advance both paths one coarse step: two fine Milstein steps
                // (dW0, then dW1) and one coarse Milstein step (dW0+dW1),
                // updating X, the Asian sum A and the running minimum M for
                // each.  All state updates route through kahan_accum16.
                S::Pair16 d; d.dW0=dW0; d.dW1=dW1; d.dI0=dI0; d.dI1=dI1;
                d.Lrv0=Lrv0; d.Lrv1=Lrv1;
                S::step_pair16(Xf, Xc, Af, Ac, Mf, Mc, Xf_c, Xc_c, Af_c, Ac_c,
                               d, kahan_accum16,
                               vSig16, vHf16, vHc16, v025hc16, cf16, cc16);
            }

            // Close the trapezoid: the loop added h*X_{n+1} each step, so
            // subtract (h/2)X(T) to leave sum h(X_n + X_{n+1})/2 + bridge.
            __m256h Aft = _mm256_sub_ph(Af, _mm256_mul_ph(vHalf16, _mm256_mul_ph(vHf16, Xf)));
            __m256h Act = _mm256_sub_ph(Ac, _mm256_mul_ph(vHalf16, _mm256_mul_ph(vHc16, Xc)));

            alignas(64) _Float16 Aa[16], Aca[16], Xa[16], Xca[16], Ma[16], Mca[16];
            _mm256_store_ph(Aa, Aft); _mm256_store_ph(Aca, Act);
            _mm256_store_ph(Xa, Xf);  _mm256_store_ph(Xca, Xc);
            _mm256_store_ph(Ma, Mf);  _mm256_store_ph(Mca, Mc);
            for (int j = 0; j < nb; ++j) {
                // Fine and coarse payoffs from the same driving randomness
                double Pf  = pay16(opt, Aa[j], Xa[j], Ma[j]);
                double Pc  = pay16(opt, Aca[j], Xca[j], Mca[j]);
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
            const FC32 cf32 = make_fc32(hf_f), cc32 = make_fc32(hc_f);
            const __m512 vsi32 = _mm512_set1_ps(si_f), vHalf32 = _mm512_set1_ps(0.5f);
            const __m512 vHf32 = _mm512_set1_ps(hf_f), vHc32 = _mm512_set1_ps(hc_f);
            const __m512 v025hc32 = _mm512_set1_ps(0.25f*hc_f), vSig32 = _mm512_set1_ps(sig);
            const __m512 vK32 = _mm512_set1_ps(K);

            __m256h Xf_h = vK16, Xc_h = vK16;
            __m256h Af_h = _mm256_mul_ph(vHalf16, _mm256_mul_ph(vHf16, vK16));
            __m256h Ac_h = _mm256_mul_ph(vHalf16, _mm256_mul_ph(vHc16, vK16));
            __m256h Mf_h = vK16, Mc_h = vK16;
            __m256h Xf_hc = _mm256_setzero_ph(), Xc_hc = _mm256_setzero_ph();
            __m256h Af_hc = _mm256_setzero_ph(), Ac_hc = _mm256_setzero_ph();

            __m512 Xf_f = vK32, Xc_f = vK32;
            __m512 Af_f = _mm512_mul_ps(vHalf32, _mm512_mul_ps(vHf32, vK32));
            __m512 Ac_f = _mm512_mul_ps(vHalf32, _mm512_mul_ps(vHc32, vK32));
            __m512 Mf_f = vK32, Mc_f = vK32;

            for (int n = 0; n < nc; ++n) {
                // fp32 and fp16 Normals: independently transformed from the
                // same underlying Philox bits (normal16_coupled) at each of
                // the four draw points, not narrowed one from the other.
                __m512 zW0_f32, zW1_f32, zI0_f32, zI1_f32;
                __m256h zW0_f16, zW1_f16, zI0_f16, zI1_f16;
                normal16_coupled(g, zW0_f32, zW0_f16);
                normal16_coupled(g, zW1_f32, zW1_f16);
                normal16_coupled(g, zI0_f32, zI0_f16);
                normal16_coupled(g, zI1_f32, zI1_f16);

                __m512 dW0_f32 = _mm512_mul_ps(_mm512_set1_ps(sqrtf(hf_f)), zW0_f32);
                __m512 dW1_f32 = _mm512_mul_ps(_mm512_set1_ps(sqrtf(hf_f)), zW1_f32);
                __m512 dI0_f32 = _mm512_mul_ps(_mm512_set1_ps(si_f), zI0_f32);
                __m512 dI1_f32 = _mm512_mul_ps(_mm512_set1_ps(si_f), zI1_f32);
                __m512 Lrv0_f32 = loguniform16_fp32(g);
                __m512 Lrv1_f32 = loguniform16_fp32(g);

                // fp16 draws: norminv_fp16 on the same bits as the fp32 draws
                // above, not a narrow of dW*_f32.  Log-uniform term stays fp32.
                __m256h vsqhf16 = _mm256_set1_ph((_Float16)sqrtf(hf_f));
                __m256h vsi16h  = _mm256_set1_ph((_Float16)si_f);
                __m256h dW0 = _mm256_mul_ph(vsqhf16, zW0_f16);
                __m256h dW1 = _mm256_mul_ph(vsqhf16, zW1_f16);
                __m256h dI0 = _mm256_mul_ph(vsi16h,  zI0_f16);
                __m256h dI1 = _mm256_mul_ph(vsi16h,  zI1_f16);
                __m256h Lrv0 = narrow16(Lrv0_f32), Lrv1 = narrow16(Lrv1_f32);
                __m256h dWc = _mm256_add_ph(dW0, dW1), ddW = _mm256_sub_ph(dW0, dW1);

                // fp16 chain: the whole coupled pair in one call (same body as
                // the even-level branch).  The fp32 reference chain below is
                // interleaved with it and driven by the same coupled draws.
                {
                    S::Pair16 dh; dh.dW0=dW0; dh.dW1=dW1; dh.dI0=dI0; dh.dI1=dI1;
                    dh.Lrv0=Lrv0; dh.Lrv1=Lrv1;
                    S::step_pair16(Xf_h, Xc_h, Af_h, Ac_h, Mf_h, Mc_h,
                                   Xf_hc, Xc_hc, Af_hc, Ac_hc, dh, kahan_accum16,
                                   vSig16, vHf16, vHc16, v025hc16, cf16, cc16);
                }

                // ---- fp32 fine step 0 (unchanged reference) ----
                __m512 X0a_f = Xf_f, v0_f = _mm512_mul_ps(vSig32, X0a_f);
                Xf_f = _mm512_add_ps(X0a_f, S::mil_incr32(X0a_f, dW0_f32, cf32));
                Af_f = _mm512_add_ps(Af_f, _mm512_add_ps(_mm512_mul_ps(vHf32,Xf_f), _mm512_mul_ps(v0_f,dI0_f32)));
                Mf_f = S::minbr32(Mf_f, X0a_f, Xf_f, v0_f, Lrv0_f32, cf32);

                // ---- fp32 fine step 1 (unchanged reference) ----
                __m512 X0b_f = Xf_f, v1_f = _mm512_mul_ps(vSig32, X0b_f);
                Xf_f = _mm512_add_ps(X0b_f, S::mil_incr32(X0b_f, dW1_f32, cf32));
                Af_f = _mm512_add_ps(Af_f, _mm512_add_ps(_mm512_mul_ps(vHf32,Xf_f), _mm512_mul_ps(v1_f,dI1_f32)));
                Mf_f = S::minbr32(Mf_f, X0b_f, Xf_f, v1_f, Lrv1_f32, cf32);

                // ---- fp32 coarse step (unchanged reference) ----
                __m512 Xc0_f = Xc_f, vc_f = _mm512_mul_ps(vSig32, Xc0_f);
                __m512 dWc_f32 = _mm512_add_ps(dW0_f32, dW1_f32), ddW_f32 = _mm512_sub_ps(dW0_f32, dW1_f32);
                Xc_f = _mm512_add_ps(Xc0_f, S::mil_incr32(Xc0_f, dWc_f32, cc32));
                __m512 cb_f = _mm512_add_ps(_mm512_add_ps(dI0_f32,dI1_f32), _mm512_mul_ps(v025hc32, ddW_f32));
                Ac_f = _mm512_add_ps(Ac_f, _mm512_add_ps(_mm512_mul_ps(vHc32,Xc_f), _mm512_mul_ps(vc_f,cb_f)));
                __m512 Xc1_f = _mm512_mul_ps(vHalf32, _mm512_add_ps(_mm512_add_ps(Xc0_f,Xc_f), _mm512_mul_ps(vc_f,ddW_f32)));
                Mc_f = S::minbr32(Mc_f, Xc0_f, Xc1_f, vc_f, Lrv0_f32, cf32);
                Mc_f = S::minbr32(Mc_f, Xc1_f, Xc_f,  vc_f, Lrv1_f32, cf32);
            }

            __m256h Af_ht = _mm256_sub_ph(Af_h, _mm256_mul_ph(vHalf16, _mm256_mul_ph(vHf16, Xf_h)));
            __m256h Ac_ht = _mm256_sub_ph(Ac_h, _mm256_mul_ph(vHalf16, _mm256_mul_ph(vHc16, Xc_h)));
            __m512  Af_ft = _mm512_sub_ps(Af_f, _mm512_mul_ps(vHalf32, _mm512_mul_ps(vHf32, Xf_f)));
            __m512  Ac_ft = _mm512_sub_ps(Ac_f, _mm512_mul_ps(vHalf32, _mm512_mul_ps(vHc32, Xc_f)));

            alignas(64) _Float16 Ahf[16], Ahc[16], Xhf[16], Xhc[16], Mhf[16], Mhc[16];
            alignas(64) float    Aff[16], Afc[16], Xff[16], Xfc[16], Mff[16], Mfc[16];
            _mm256_store_ph(Ahf, Af_ht); _mm256_store_ph(Ahc, Ac_ht);
            _mm256_store_ph(Xhf, Xf_h);  _mm256_store_ph(Xhc, Xc_h);
            _mm256_store_ph(Mhf, Mf_h);  _mm256_store_ph(Mhc, Mc_h);
            _mm512_store_ps(Aff, Af_ft); _mm512_store_ps(Afc, Ac_ft);
            _mm512_store_ps(Xff, Xf_f);  _mm512_store_ps(Xfc, Xc_f);
            _mm512_store_ps(Mff, Mf_f);  _mm512_store_ps(Mfc, Mc_f);

            for (int j = 0; j < nb; ++j) {
                // Four payoffs: {fine, coarse} x {fp16, fp32}
                double Ph_fine = pay16(opt, Ahf[j], Xhf[j], Mhf[j]);
                double Ph_cors = pay16(opt, Ahc[j], Xhc[j], Mhc[j]);
                double Pf_fine = pay32(opt, Aff[j], Xff[j], Mff[j]);
                double Pf_cors = pay32(opt, Afc[j], Xfc[j], Mfc[j]);

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
