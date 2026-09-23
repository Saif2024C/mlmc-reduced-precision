#ifndef OPTIONS_MM512_H
#define OPTIONS_MM512_H
////////////////////////////////////////////////////////////////////////////
// options_mm512.h -- the per-path Milstein/payoff helpers shared by every
// AVX-512 estimator and benchmark in this directory, in one place.
//
// TWO NAMESPACES, one per payoff family -- they share the Kahan accumulator
// and nothing else (scalar has no Cholesky and no per-asset arrays).  Each
// program uses the one it needs:
//
//   opt::scalar   1-asset GBM, Asian                (sig scalar)
//   opt::basket   5-asset correlated GBM, European / Asian (Cholesky, weights)
//
// Include AFTER immintrin.h / avx512_rng.h (needs __m512, __m256h, _Float16).
////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <immintrin.h>

namespace opt {

// ===========================================================================
// shared: compensated / plain accumulation on native fp16 state
// ===========================================================================

// Classical 3-line Kahan compensated summation.  `comp` carries what the
// previous add rounded away and must persist across steps alongside `sum`.
static inline void kahan_accum16(__m256h &sum, __m256h &comp, __m256h term) {
    __m256h y = _mm256_sub_ph(term, comp);
    __m256h t = _mm256_add_ph(sum, y);
    comp      = _mm256_sub_ph(_mm256_sub_ph(t, sum), y);
    sum       = t;
}

// Uncompensated counterpart, so a caller with a runtime kahan on/off switch
// picks between two named functions rather than branching inside one.
static inline void plain_accum16(__m256h &sum, __m256h &, __m256h term) {
    sum = _mm256_add_ph(sum, term);
}

// Per-block MLMC moment accumulator.
// sums[] layout: [0] cost, [1] Y, [2] Y^2, [3] Y^3, [4] Y^4, [5] Pf, [6] Pf^2,
// all fp64 regardless of the path precision -- MLMC's own summation is never
// the accuracy bottleneck (see the mcsum experiment) and cost must stay exact.
struct Moments {
    double m[7];
    Moments() { for (int i = 0; i < 7; ++i) m[i] = 0.0; }

    // dP = Y (the MLMC correction), Pfv = the fine-level payoff, cost in
    // timesteps.
    inline void add(double dP, double Pfv, double cost) {
        m[0] += cost;
        m[1] += dP;    m[2] += dP*dP;  m[3] += dP*dP*dP;  m[4] += dP*dP*dP*dP;
        m[5] += Pfv;   m[6] += Pfv*Pfv;
    }
    inline void flush(double *sums) const {
        for (int i = 0; i < 7; ++i) sums[i] += m[i];
    }
};

static inline __m256h narrow16(__m512 x) { return _mm512_cvtxps_ph(x); }
static inline __m512  widen32(__m256h x) {
    return _mm512_cvtph_ps(_mm256_castph_si256(x));
}

// ===========================================================================
// scalar: 1-asset GBM,  option 1 = Asian
// ===========================================================================
namespace scalar {

struct C32 { __m512  r, sig, halfsig2, hf, sqhf; };
struct C16 { __m256h r, sig, halfsig2, hf; };

static inline C32 make32(float r, float sig, float h) {
    C32 c;
    c.r        = _mm512_set1_ps(r);
    c.sig      = _mm512_set1_ps(sig);
    c.halfsig2 = _mm512_set1_ps(0.5f*sig*sig);
    c.hf       = _mm512_set1_ps(h);
    c.sqhf     = _mm512_set1_ps(std::sqrt(h));
    return c;
}
static inline C16 make16(float r, float sig, _Float16 h) {
    C16 c;
    c.r        = _mm256_set1_ph((_Float16)r);
    c.sig      = _mm256_set1_ph((_Float16)sig);
    c.halfsig2 = _mm256_set1_ph((_Float16)(0.5f*sig*sig));
    c.hf       = _mm256_set1_ph(h);
    return c;
}

// Milstein increment:  X*r*h + X*sig*dW + 0.5*sig^2*X*(dW^2 - h)
static inline __m512 mil_incr32(__m512 X, __m512 dW, const C32 &c) {
    __m512 drift = _mm512_mul_ps(_mm512_mul_ps(X, c.r), c.hf);
    __m512 diff  = _mm512_mul_ps(_mm512_mul_ps(X, c.sig), dW);
    __m512 d2h   = _mm512_sub_ps(_mm512_mul_ps(dW, dW), c.hf);
    __m512 mil   = _mm512_mul_ps(_mm512_mul_ps(X, c.halfsig2), d2h);
    return _mm512_add_ps(_mm512_add_ps(drift, diff), mil);
}
static inline __m256h mil_incr16(__m256h X, __m256h dW, const C16 &c) {
    __m256h drift = _mm256_mul_ph(_mm256_mul_ph(X, c.r), c.hf);
    __m256h diff  = _mm256_mul_ph(_mm256_mul_ph(X, c.sig), dW);
    __m256h d2h   = _mm256_sub_ph(_mm256_mul_ph(dW, dW), c.hf);
    __m256h mil   = _mm256_mul_ph(_mm256_mul_ph(X, c.halfsig2), d2h);
    return _mm256_add_ph(_mm256_add_ph(drift, diff), mil);
}

//
// State is updated in place.  Af/Ac accumulate the Asian time-integral
// (bridge term v*dI per fine step, and the coarse bridge dI0+dI1+0.25*hc*ddW).

struct Pair32 { __m512 dW0, dW1, dI0, dI1; };

static inline void step_pair32(__m512 &Xf, __m512 &Xc, __m512 &Af, __m512 &Ac,
                               const Pair32 &d,
                               __m512 sig, __m512 hf, __m512 hc, __m512 q25hc,
                               const C32 &cf, const C32 &cc)
{
    __m512 dWc = _mm512_add_ps(d.dW0, d.dW1);
    __m512 ddW = _mm512_sub_ps(d.dW0, d.dW1);

    // fine step 1
    __m512 Xa = Xf, va = _mm512_mul_ps(sig, Xa);
    Xf = _mm512_add_ps(Xa, mil_incr32(Xa, d.dW0, cf));
    Af = _mm512_add_ps(Af, _mm512_add_ps(_mm512_mul_ps(hf, Xf),
                                         _mm512_mul_ps(va, d.dI0)));

    // fine step 2
    __m512 Xb = Xf, vb = _mm512_mul_ps(sig, Xb);
    Xf = _mm512_add_ps(Xb, mil_incr32(Xb, d.dW1, cf));
    Af = _mm512_add_ps(Af, _mm512_add_ps(_mm512_mul_ps(hf, Xf),
                                         _mm512_mul_ps(vb, d.dI1)));

    // coarse step, driven by dW0+dW1
    __m512 Xc0 = Xc, vc = _mm512_mul_ps(sig, Xc0);
    __m512 nxc = _mm512_add_ps(Xc0, mil_incr32(Xc0, dWc, cc));
    __m512 cb  = _mm512_add_ps(_mm512_add_ps(d.dI0, d.dI1),
                               _mm512_mul_ps(q25hc, ddW));
    Ac = _mm512_add_ps(Ac, _mm512_add_ps(_mm512_mul_ps(hc, nxc),
                                         _mm512_mul_ps(vc, cb)));
    Xc = nxc;
}

// fp16 counterpart of step_pair32.  Two differences from the fp32 version, both
// deliberate and load-bearing:
//   * every state update goes through `acc`, so the caller passes
//     kahan_accum16 or plain_accum16 and the compensation terms travel with
//     the state
struct Pair16 { __m256h dW0, dW1, dI0, dI1; };

template <class AccFn>
static inline void step_pair16(__m256h &Xf, __m256h &Xc, __m256h &Af, __m256h &Ac,
                               __m256h &Xf_c, __m256h &Xc_c,
                               __m256h &Af_c, __m256h &Ac_c,
                               const Pair16 &d, AccFn acc,
                               __m256h sig, __m256h hf, __m256h hc, __m256h q25hc,
                               const C16 &cf, const C16 &cc)
{
    // fine step 0
    __m256h Xa = Xf, va = _mm256_mul_ph(sig, Xa);
    acc(Xf, Xf_c, mil_incr16(Xa, d.dW0, cf));
    acc(Af, Af_c, _mm256_add_ph(_mm256_mul_ph(hf, Xf), _mm256_mul_ph(va, d.dI0)));

    // fine step 1
    __m256h Xb = Xf, vb = _mm256_mul_ph(sig, Xb);
    acc(Xf, Xf_c, mil_incr16(Xb, d.dW1, cf));
    acc(Af, Af_c, _mm256_add_ph(_mm256_mul_ph(hf, Xf), _mm256_mul_ph(vb, d.dI1)));

    // coarse step, driven by dW0+dW1
    __m256h Xc0 = Xc, vc = _mm256_mul_ph(sig, Xc0);
    __m256h ddW = _mm256_sub_ph(d.dW0, d.dW1);
    __m256h dWc = _mm256_add_ph(d.dW0, d.dW1);
    acc(Xc, Xc_c, mil_incr16(Xc0, dWc, cc));
    __m256h cb = _mm256_add_ph(_mm256_add_ph(d.dI0, d.dI1),
                               _mm256_mul_ph(q25hc, ddW));
    acc(Ac, Ac_c, _mm256_add_ph(_mm256_mul_ph(hc, Xc), _mm256_mul_ph(vc, cb)));
}

// opt 1 = Asian (running average A).
static inline double pay32(int o, float K, float A) {
    (void)o;
    return (double)std::fmax(0.0f, A - K);
}
static inline double pay16(int o, _Float16 K, _Float16 A) {
    (void)o;
    _Float16 v = A - K;
    return (double)(v > (_Float16)0.0f ? v : (_Float16)0.0f);
}

} // namespace scalar

// ===========================================================================
// basket: 5-asset correlated GBM,  option 1 = European, 2 = Asian
// ===========================================================================
namespace basket {

static const int NA = 5;      // assets

// Cholesky factor of the equicorrelation matrix, Sigma = L L^T.
// Written into L[NA][NA], lower triangular, upper zeroed.
static inline void cholesky(float rho, float L[NA][NA]) {
    float S[NA][NA];
    for (int i = 0; i < NA; ++i)
        for (int j = 0; j < NA; ++j) S[i][j] = (i == j) ? 1.0f : rho;
    for (int i = 0; i < NA; ++i) {
        for (int j = 0; j <= i; ++j) {
            float s = S[i][j];
            for (int m = 0; m < j; ++m) s -= L[i][m] * L[j][m];
            L[i][j] = (i == j) ? std::sqrt(s) : s / L[j][j];
        }
        for (int j = i + 1; j < NA; ++j) L[i][j] = 0.0f;
    }
}

// NA correlated increments: NA independent Normals, Cholesky-combined, scaled.
// The Normal transform is passed in (`draw`), since the fp32 and fp16 paths
// use different ones -- this header stays RNG-agnostic.
template <class Rng, class DrawFn>
static inline void draw_corr32(Rng &g, DrawFn draw, float scale,
                               const float L[NA][NA], __m512 dw[NA]) {
    __m512 y[NA]; for (int i = 0; i < NA; ++i) y[i] = draw(g);
    __m512 sc = _mm512_set1_ps(scale);
    for (int i = 0; i < NA; ++i) {
        __m512 s = _mm512_setzero_ps();
        for (int j = 0; j <= i; ++j)
            s = _mm512_fmadd_ps(_mm512_set1_ps(L[i][j]), y[j], s);
        dw[i] = _mm512_mul_ps(sc, s);
    }
}
template <class Rng, class DrawFn>
static inline void draw_corr16(Rng &g, DrawFn draw, _Float16 scale,
                               const float L[NA][NA], __m256h dw[NA]) {
    __m256h y[NA]; for (int i = 0; i < NA; ++i) y[i] = draw(g);
    __m256h sc = _mm256_set1_ph(scale);
    for (int i = 0; i < NA; ++i) {
        __m256h s = _mm256_setzero_ph();
        for (int j = 0; j <= i; ++j)
            s = _mm256_fmadd_ph(_mm256_set1_ph((_Float16)L[i][j]), y[j], s);
        dw[i] = _mm256_mul_ph(sc, s);
    }
}

// weighted basket value  sum_i w_i x_i
static inline __m512 value32(const __m512 x[NA], const float w[NA]) {
    __m512 s = _mm512_setzero_ps();
    for (int i = 0; i < NA; ++i) s = _mm512_fmadd_ps(_mm512_set1_ps(w[i]), x[i], s);
    return s;
}
static inline __m256h value16(const __m256h x[NA], const float w[NA]) {
    __m256h s = _mm256_setzero_ph();
    for (int i = 0; i < NA; ++i)
        s = _mm256_fmadd_ph(_mm256_set1_ph((_Float16)w[i]), x[i], s);
    return s;
}

// weighted dot  sum_i w_i a_i b_i -- the Asian time-average bridge term.
// Must carry the weights (Giles' alf'*(vf.*dIf)); an unweighted version is a
// documented past bug that inflated the bridge and broke low-level coupling.
static inline __m512 wdot32(const __m512 a[NA], const __m512 b[NA], const float w[NA]) {
    __m512 s = _mm512_setzero_ps();
    for (int i = 0; i < NA; ++i)
        s = _mm512_fmadd_ps(_mm512_set1_ps(w[i]), _mm512_mul_ps(a[i], b[i]), s);
    return s;
}
static inline __m256h wdot16(const __m256h a[NA], const __m256h b[NA], const float w[NA]) {
    __m256h s = _mm256_setzero_ph();
    for (int i = 0; i < NA; ++i)
        s = _mm256_fmadd_ph(_mm256_set1_ph((_Float16)w[i]),
                            _mm256_mul_ph(a[i], b[i]), s);
    return s;
}

// Per-asset Milstein step, in place:
//   x_i += r*x_i*h + sig_i*x_i*dw_i + 0.5*sig_i^2*x_i*(dw_i^2 - h)
// sigx[i] returns sig_i*x_i taken PRE-step -- the Asian bridge needs it.
static inline void step32(__m512 x[NA], const __m512 dw[NA], __m512 h,
                          float r, const float sig[NA], __m512 sigx[NA]) {
    __m512 vR = _mm512_set1_ps(r);
    for (int i = 0; i < NA; ++i) {
        __m512 x0 = x[i];
        __m512 vS = _mm512_set1_ps(sig[i]);
        __m512 vH = _mm512_set1_ps(0.5f*sig[i]*sig[i]);
        sigx[i]      = _mm512_mul_ps(vS, x0);
        __m512 drift = _mm512_mul_ps(_mm512_mul_ps(x0, vR), h);
        __m512 diff  = _mm512_mul_ps(sigx[i], dw[i]);
        __m512 d2h   = _mm512_sub_ps(_mm512_mul_ps(dw[i], dw[i]), h);
        __m512 mil   = _mm512_mul_ps(_mm512_mul_ps(x0, vH), d2h);
        x[i] = _mm512_add_ps(x0, _mm512_add_ps(_mm512_add_ps(drift, diff), mil));
    }
}

// fp16 counterpart returning the INCREMENT rather than stepping in place, so
// the caller can route it through kahan_accum16 / plain_accum16.
static inline void incr16(const __m256h x0[NA], const __m256h dw[NA], __m256h h,
                          float r, const float sig[NA],
                          __m256h out[NA], __m256h sigx[NA]) {
    __m256h vR = _mm256_set1_ph((_Float16)r);
    for (int i = 0; i < NA; ++i) {
        __m256h vS = _mm256_set1_ph((_Float16)sig[i]);
        __m256h vH = _mm256_set1_ph((_Float16)(0.5f*sig[i]*sig[i]));
        sigx[i]       = _mm256_mul_ph(vS, x0[i]);
        __m256h drift = _mm256_mul_ph(_mm256_mul_ph(x0[i], vR), h);
        __m256h diff  = _mm256_mul_ph(sigx[i], dw[i]);
        __m256h d2h   = _mm256_sub_ph(_mm256_mul_ph(dw[i], dw[i]), h);
        __m256h mil   = _mm256_mul_ph(_mm256_mul_ph(x0[i], vH), d2h);
        out[i] = _mm256_add_ph(_mm256_add_ph(drift, diff), mil);
    }
}

// One coarse timestep of the fine/coarse COUPLED pair, per asset: two fine
// steps driven by dw0,dw1 and one coarse step driven by their sum, sharing the
// same driving randomness -- that sharing is what makes Var(Pf-Pc) decay.
//
// The Asian time-integral terms (Af/Ac and the di/bridge inputs) are only
// touched when asian is true; for European they are dead and the caller passes
// asian=false, matching the `if (opt == 2)` gating in the estimators.

static inline void step_pair32(__m512 xf[NA], __m512 xc[NA],
                               __m512 &Af, __m512 &Ac,
                               const __m512 dw0[NA], const __m512 dw1[NA],
                               const __m512 di0[NA], const __m512 di1[NA],
                               __m512 hf, __m512 hc, __m512 q25hc,
                               float r, const float sig[NA], const float w[NA],
                               bool asian)
{
    __m512 dwc[NA], ddw[NA];
    for (int i = 0; i < NA; ++i) {
        dwc[i] = _mm512_add_ps(dw0[i], dw1[i]);
        ddw[i] = _mm512_sub_ps(dw0[i], dw1[i]);
    }

    __m512 s0[NA]; step32(xf, dw0, hf, r, sig, s0);
    if (asian) Af = _mm512_add_ps(Af, _mm512_add_ps(_mm512_mul_ps(hf, value32(xf, w)),
                                                    wdot32(s0, di0, w)));
    __m512 s1[NA]; step32(xf, dw1, hf, r, sig, s1);
    if (asian) Af = _mm512_add_ps(Af, _mm512_add_ps(_mm512_mul_ps(hf, value32(xf, w)),
                                                    wdot32(s1, di1, w)));

    __m512 sc[NA]; step32(xc, dwc, hc, r, sig, sc);
    if (asian) {
        __m512 br[NA];
        for (int i = 0; i < NA; ++i)
            br[i] = _mm512_add_ps(_mm512_add_ps(di0[i], di1[i]),
                                  _mm512_mul_ps(q25hc, ddw[i]));
        Ac = _mm512_add_ps(Ac, _mm512_add_ps(_mm512_mul_ps(hc, value32(xc, w)),
                                             wdot32(sc, br, w)));
    }
}

// fp16 counterpart of the basket step_pair32.  
template <class AccFn, class AccVecFn>
static inline void step_pair16(__m256h xf[NA], __m256h xc[NA],
                               __m256h &Af, __m256h &Ac,
                               __m256h xf_c[NA], __m256h xc_c[NA],
                               __m256h &Af_c, __m256h &Ac_c,
                               const __m256h dw0[NA], const __m256h dw1[NA],
                               const __m256h di0[NA], const __m256h di1[NA],
                               AccFn acc, AccVecFn accv,
                               __m256h hf, __m256h hc, __m256h q25hc,
                               float r, const float sig[NA], const float w[NA],
                               bool asian)
{
    __m256h dwc[NA], ddw[NA];
    for (int i = 0; i < NA; ++i) {
        dwc[i] = _mm256_add_ph(dw0[i], dw1[i]);
        ddw[i] = _mm256_sub_ph(dw0[i], dw1[i]);
    }

    __m256h incr0[NA], s0[NA];
    incr16(xf, dw0, hf, r, sig, incr0, s0);
    accv(xf, xf_c, incr0);
    if (asian) acc(Af, Af_c, _mm256_add_ph(_mm256_mul_ph(hf, value16(xf, w)),
                                           wdot16(s0, di0, w)));

    __m256h incr1[NA], s1[NA];
    incr16(xf, dw1, hf, r, sig, incr1, s1);
    accv(xf, xf_c, incr1);
    if (asian) acc(Af, Af_c, _mm256_add_ph(_mm256_mul_ph(hf, value16(xf, w)),
                                           wdot16(s1, di1, w)));

    __m256h incrc[NA], sc[NA];
    incr16(xc, dwc, hc, r, sig, incrc, sc);
    accv(xc, xc_c, incrc);
    if (asian) {
        __m256h br[NA];
        for (int i = 0; i < NA; ++i)
            br[i] = _mm256_add_ph(_mm256_add_ph(di0[i], di1[i]),
                                  _mm256_mul_ph(q25hc, ddw[i]));
        acc(Ac, Ac_c, _mm256_add_ph(_mm256_mul_ph(hc, value16(xc, w)),
                                    wdot16(sc, br, w)));
    }
}

// opt 1 = European (terminal basket B), 2 = Asian (time average A).
static inline double pay32(int o, float K, float B, float A) {
    return (o == 1) ? (double)std::fmax(0.0f, B - K)
                    : (double)std::fmax(0.0f, A - K);
}
static inline double pay16(int o, _Float16 K, _Float16 B, _Float16 A) {
    _Float16 v = (o == 1) ? (_Float16)(B - K) : (_Float16)(A - K);
    return (double)(v > (_Float16)0.0f ? v : (_Float16)0.0f);
}

} // namespace basket
} // namespace opt

#endif // OPTIONS_MM512_H
