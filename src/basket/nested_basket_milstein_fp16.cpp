/*
 * nested_basket_milstein_fp16.cpp -- unified basket fp16 template.
 *
 * Basket analogue of src/scalar/nested_scalar_milstein_fp16.cpp: same three
 * runtime-switchable sweep modes (see main()), same bracketed all-half (and
 * all-fp32) Milstein arithmetic, applied per-asset to the 5-asset correlated
 * GBM basket (option 1 = European, option 2 = Asian). Model parameters and
 * correlation/Cholesky match basket_scalar.cpp / the old basket_code.cpp:
 * S0=K=100, T=1, r=0.05, sigma={0.25..0.45}, weights=0.2, pairwise rho=0.25.
 *
 *   nokahan  : plain += per-asset accumulation, convergence table only (a
 *              zero-terminated Eps array skips mlmc_test's complexity
 *              section -- beta is often non-positive here, same reason as
 *              the scalar file: mlmc()'s adaptive complexity-test sampler
 *              never returns against a non-positive beta).
 *   kahan    : Kahan-compensated accumulation, per asset, in BOTH the fp16
 *              chain and its fp32 twin (path state x[5] and the Asian
 *              running sum af/ac) -- beta is genuinely positive, so the
 *              real complexity test (mlmc_test) runs fine.
 *   adaptive : kahan ON below grid level l_star, PLUS a hard cutoff at/above
 *              l_star (l_star=6 for both options, matching
 *              nested_basket_milstein_adaptive.cpp's existing tuning): even
 *              levels run pure fp32 (no fp16 at all) instead of the fp16
 *              chain; odd levels run the fp32 chain twice on the same draws
 *              as a genuine fp32-self-consistency check. Real complexity
 *              test. Gives beta close to the theoretical 2 (dominated by the
 *              clean pure-fp32 levels above cutoff) -- see CLAUDE.md.
 *
 * Build (from repo root):
 *   g++ -O0 -std=c++11 -mf16c -fno-fast-math \
 *       src/basket/nested_basket_milstein_fp16.cpp \
 *       -o build/nested_basket_milstein_fp16
 *   -- or, for a much faster run (fuses the bracketed half temporaries, so
 *      per-bracket fp16 rounding fidelity is not guaranteed -- fine for a
 *      quick look, not for citing exact rounding behaviour):
 *   g++ -O3 -std=c++11 -mf16c \
 *       src/basket/nested_basket_milstein_fp16.cpp \
 *       -o build/nested_basket_milstein_fp16_O3
 *
 * Run (writes nested_basket_fp16_{nokahan,kahan,adaptive}_{1,2}.txt/
 * _diag.txt/_moments.txt into the current directory -- run from outputs/ so
 * they land there; each sweep runs both options):
 *   cd outputs && ../build/nested_basket_milstein_fp16_O3
 *
 * Plot (3x2 layout when the complexity-test section has rows -- kahan and
 * adaptive pass a populated Eps array so it does; nokahan passes a zero-
 * terminated Eps array, so the section header prints but stays empty -- 2x2
 * otherwise):
 *   for f in nested_basket_fp16_nokahan_1  nested_basket_fp16_nokahan_2 \
 *            nested_basket_fp16_kahan_1    nested_basket_fp16_kahan_2   \
 *            nested_basket_fp16_adaptive_1 nested_basket_fp16_adaptive_2; do
 *     MPLBACKEND=Agg python3 python/nested_mlmc_plot_python.py outputs/$f
 *   done
 *
 * All output/plots for the latest run of this file live in
 * outputs/nested/basket/fp16/.
 */

#include "../core/nested_mlmc_test.cpp"
#include "../core/mlmc_rng.cpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

using half = _Float16;

// 1 = Basket European, 2 = Basket Asian
int option;

// Per-level diagnostic file (point 3): mean(Y), var(Y), mean(Pf) per level.
static FILE *diagfp = nullptr;

// Real per-sample moments file (matches nested_scalar_fp16_diag_plot_python.py's
// "level N: cost=... del1=... del2=... var1=... var2=... s3=... s4=..." parser).
static FILE *momfp = nullptr;

void nested_basket_fp16_l(int, int, double *);

// Runtime Kahan on/off switch (set per main() sweep). See kahan_accum().
static bool kahan_mode = false;

// Runtime Adaptive on/off switch: at/above grid level l_star, even levels run
// the pure-fp32 chain instead of the fp16 chain, and odd levels run the fp32
// chain twice (same draws) as a genuine fp32-self-consistency check. Below
// l_star: unchanged (today's fp16 chain, still gated by kahan_mode). l_star=6
// for both options here -- matches nested_basket_milstein_adaptive.cpp's
// existing per-asset cutoff tuning (not re-derived from this file's own
// nokahan data, since that file already established it for this same model).
// Only ever run paired with kahan_mode=true in main(), same reasoning as the
// scalar template.
static bool adaptive_mode = false;
static int  l_star = 0;
static const int l_star_by_option[] = {0, 6, 6};  // index by option 1/2

// ============================================================
// Model constants (match basket_scalar.cpp)
// ============================================================
static const int    kDim  = 5;
static const float  kK_f  = 100.0f, kT_f = 1.0f, kR_f = 0.05f;
static const float  kSig_f[5] = {0.25f, 0.30f, 0.35f, 0.40f, 0.45f};
static const float  kW_f[5]   = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};
static const float  kRho_f    = 0.25f;

static float L_f[5][5];

static void build_cholesky()
{
    float a[5][5];
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            a[i][j] = (i == j) ? 1.0f : kRho_f;

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j <= i; ++j) {
            float s = a[i][j];
            for (int k = 0; k < j; ++k) s -= L_f[i][k] * L_f[j][k];
            L_f[i][j] = (i == j) ? sqrtf(s) : s / L_f[j][j];
        }
    }
}

static inline void draw_increment_f(float scale, float dw[5])
{
    float y[5];
    for (int i = 0; i < kDim; ++i) y[i] = next_normal();
    for (int i = 0; i < kDim; ++i) {
        float s = 0.0f;
        for (int j = 0; j <= i; ++j) s += L_f[i][j] * y[j];
        dw[i] = scale * s;
    }
}

static inline void draw_integral_f(float scale, float di[5])
{
    float y[5];
    for (int i = 0; i < kDim; ++i) y[i] = next_normal();
    for (int i = 0; i < kDim; ++i) {
        float s = 0.0f;
        for (int j = 0; j <= i; ++j) s += L_f[i][j] * y[j];
        di[i] = scale * s;
    }
}

static inline half basket_h(const half x[5])
{ half s=(half)0.0f16; for(int i=0;i<kDim;i++) s=s+(half)kW_f[i]*x[i]; return s; }

static inline float basket_f(const float x[5])
{ float s=0; for(int i=0;i<kDim;i++) s+=kW_f[i]*x[i]; return s; }

static inline void sigma_x_h(const half x[5], half v[5])
{ for(int i=0;i<kDim;i++) v[i]=(half)kSig_f[i]*x[i]; }

static inline void sigma_x_f(const float x[5], float v[5])
{ for(int i=0;i<kDim;i++) v[i]=kSig_f[i]*x[i]; }

// Weighted dot: sum_i w_i * a_i * b_i (Asian bridge correction)
static inline half wdot_h(const half a[5], const half b[5])
{ half s=(half)0.0f16; for(int i=0;i<kDim;i++) s=s+(half)kW_f[i]*a[i]*b[i]; return s; }

static inline float wdot_f(const float a[5], const float b[5])
{ float s=0; for(int i=0;i<kDim;i++) s+=kW_f[i]*a[i]*b[i]; return s; }

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    build_cholesky();

    int   N    = 20000;
    int   L    = 10;
    char  filename[64], diagname[64], momname[64];
    FILE *fp;

    const bool kahan_modes[3] = { false, true, true  };
    const bool adapt_modes[3] = { false, false, true };
    const char *mode_tag[3]   = { "nokahan", "kahan", "adaptive" };

    for (int m = 0; m < 3; ++m) {
        kahan_mode    = kahan_modes[m];
        adaptive_mode = adapt_modes[m];

        for (option = 1; option <= 2; ++option) {
            l_star = l_star_by_option[option];
            rng_initialisation();

            std::sprintf(filename, "nested_basket_fp16_%s_%d.txt", mode_tag[m], option);
            fp = std::fopen(filename, "w");
            if (!fp) { std::perror("fopen"); return EXIT_FAILURE; }

            std::sprintf(diagname, "nested_basket_fp16_%s_%d_diag.txt", mode_tag[m], option);
            diagfp = std::fopen(diagname, "w");
            if (!diagfp) { std::perror("fopen"); return EXIT_FAILURE; }
            std::fprintf(diagfp, "# l  k  parity   N   mean(Y)       var(Y)        mean(Pf)\n");

            std::sprintf(momname, "nested_basket_fp16_%s_%d_moments.txt", mode_tag[m], option);
            momfp = std::fopen(momname, "w");
            if (!momfp) { std::perror("fopen"); return EXIT_FAILURE; }

            std::printf("\n ---- Nested basket fp16/fp32 %s (Milstein, bracketed, %s%s) ----\n",
                        option == 1 ? "European" : "Asian",
                        kahan_mode ? "Kahan ON (fp16+fp32)" : "Kahan OFF",
                        adaptive_mode ? ", Adaptive l_star" : "");
            if (adaptive_mode) std::printf("      l_star = %d\n", l_star);

            if (kahan_mode) {
                int   N0   = 200;
                int   Lmin = 2;
                int   Lmax = 20;
                float Eps[] = { 0.01f, 0.02f, 0.05f, 0.1f, 0.2f, 0.0f };
                mlmc_test(nested_basket_fp16_l, N, L, N0, Eps, Lmin, Lmax, fp);
            } else {
                int   N0   = 200;
                int   Lmin = 2;
                int   Lmax = 20;
                float Eps[] = { 0.0f };
                mlmc_test(nested_basket_fp16_l, N, L, N0, Eps, Lmin, Lmax, fp);
            }
            std::fclose(fp);
            std::fclose(diagfp);
            std::fclose(momfp);
            rng_termination();
        }
    }
    return EXIT_SUCCESS;
}

// kahan_mode is declared earlier (near main()). The bracketed all-half (or
// all-fp32) arithmetic that PRODUCES each step's per-asset increment is
// unchanged either way; this only controls how that increment is folded into
// the running state (x[5] path position, af/ac Asian running sum), for both
// the fp16 chain and its fp32 twin. OFF = plain += (today's baseline). ON =
// classical 3-line compensated summation, applied in the state's own type.
static inline void kahan_accum(half &sum, half &comp, half term) {
    if (!kahan_mode) { sum = sum + term; return; }
    half y = term - comp;
    half t = sum + y;
    comp   = (t - sum) - y;
    sum    = t;
}

static inline void kahan_accum(float &sum, float &comp, float term) {
    if (!kahan_mode) { sum = sum + term; return; }
    float y = term - comp;
    float t = sum + y;
    comp    = (t - sum) - y;
    sum     = t;
}

// Bracketed per-asset Milstein step, fp16 state, Kahan-accumulated iff
// kahan_mode (point 5/7 from the scalar debug spec, extended to 5 lanes).
// comp[5] is the per-asset Kahan compensation (inert when kahan_mode==false).
static inline void milstein_h(half x[5], half comp[5], const half dw[5], half h)
{
    for (int i = 0; i < kDim; ++i) {
        half xi   = x[i];
        half sig  = (half)kSig_f[i];
        half r    = (half)kR_f;

        // Bracketed: Xnew = Xold + (sig*Xold*dW + (r*Xold*h + 0.5*sig*sig*Xold*(dW*dW - h)))
        half dWdW       = dw[i] * dw[i];
        half dWdWmh     = dWdW - h;
        half halfsigsig = (half)0.5f16 * sig * sig;
        half milcorrX   = halfsigsig * dWdWmh;
        half milcorr    = milcorrX * xi;
        half rh         = r * h;
        half rXh        = rh * xi;
        half inner      = rXh + milcorr;
        half sigX       = sig * xi;
        half diff       = sigX * dw[i];
        half outer      = diff + inner;
        kahan_accum(x[i], comp[i], outer);
    }
}

// Bracketed per-asset Milstein step, fp32 state, same Kahan on/off switch.
static inline void milstein_f(float x[5], float comp[5], const float dw[5], float h)
{
    for (int i = 0; i < kDim; ++i) {
        float xi = x[i], sig = kSig_f[i];
        float incr = kR_f*xi*h + sig*xi*dw[i] + 0.5f*sig*sig*xi*(dw[i]*dw[i] - h);
        kahan_accum(x[i], comp[i], incr);
    }
}

/* ------------------------------------------------------------------
 * nested_basket_fp16_l: level estimator for fp16/fp32 nested basket MLMC.
 *
 * sums[] layout:
 *   [0] cost,  [1] Y,  [2] Y^2,  [3] Y^3,  [4] Y^4,
 *   [5] Pf,   [6] Pf^2   (fp64, point 2 -- unchanged)
 * ------------------------------------------------------------------ */
void nested_basket_fp16_l(int l, int N, double *sums)
{
    const int nf = 1 << (l / 2);
    const int nc = nf / 2;
    const int k  = l / 2;

    const float  hf_f  = kT_f / (float)nf;
    const float  hc_f  = (nc > 0) ? kT_f / (float)nc : 0.0f;
    const half   hf_h  = (half)hf_f;
    const half   hc_h  = (half)hc_f;
    const float  disc  = std::exp(-kR_f * kT_f);
    const half   disc_h = (half)disc;

    const float  sf_f  = sqrtf(hf_f);
    const float  si_f  = sqrtf(hf_f / 12.0f) * hf_f;
    const half   halfhf = (half)0.5f16 * hf_h;
    const half   halfhc = (half)0.5f16 * hc_h;

    for (int i = 0; i < 7; ++i) sums[i] = 0.0;

    // Diagnostic accumulators (point 3): plain fp64.
    double dY = 0.0, dY2 = 0.0, dY3 = 0.0, dY4 = 0.0, dPfsum = 0.0, dPf2sum = 0.0;

    // ----------------------------------------------------------------------
    // Adaptive cutoff: at/above grid level l_star, skip the fp16 chain
    // entirely. Even levels: pure fp32 Milstein MLMC correction. Odd levels:
    // the SAME fp32 chain twice on the same draws, giving genuine fp32-self-
    // consistency noise rather than a hardcoded zero (see scalar template).
    // ----------------------------------------------------------------------
    if (adaptive_mode && k >= l_star) {
        for (int np = 0; np < N; ++np) {
            double dP, Pfv;

            if (l == 0) {
                float xf_f[5]; for(int i=0;i<kDim;i++) xf_f[i]=kK_f;
                float dw_f[5], di_f[5];
                draw_increment_f(sf_f, dw_f);
                if (option == 2) draw_integral_f(si_f, di_f);

                float af_f = 0.5f*hf_f*basket_f(xf_f);
                float vf_f[5]; sigma_x_f(xf_f, vf_f);
                float comp0[5] = {0,0,0,0,0};
                milstein_f(xf_f, comp0, dw_f, hf_f);

                float Pf;
                if (option == 2) {
                    af_f += 0.5f*hf_f*basket_f(xf_f) + wdot_f(vf_f, di_f);
                    Pf = std::fmaxf(0.0f, af_f - kK_f);
                } else {
                    Pf = std::fmaxf(0.0f, basket_f(xf_f) - kK_f);
                }
                dP = disc * Pf; Pfv = dP;

            } else if (l % 2 == 0) {
                float xf_f[5], xc_f[5];
                for(int i=0;i<kDim;i++){ xf_f[i]=kK_f; xc_f[i]=kK_f; }
                float af_f = 0.5f*hf_f*basket_f(xf_f);
                float ac_f = 0.5f*hc_f*basket_f(xc_f);
                float xfc[5]={0,0,0,0,0}, xcc[5]={0,0,0,0,0};

                for (int n = 0; n < nc; ++n) {
                    float dw1_f[5], dw2_f[5], di1_f[5], di2_f[5];
                    draw_increment_f(sf_f, dw1_f);
                    if (option == 2) draw_integral_f(si_f, di1_f);
                    draw_increment_f(sf_f, dw2_f);
                    if (option == 2) draw_integral_f(si_f, di2_f);

                    float dwc_f[5], ddw_f[5];
                    for(int i=0;i<kDim;i++){ dwc_f[i]=dw1_f[i]+dw2_f[i]; ddw_f[i]=dw1_f[i]-dw2_f[i]; }

                    float vf1_f[5]; sigma_x_f(xf_f, vf1_f);
                    milstein_f(xf_f, xfc, dw1_f, hf_f);
                    if (option==2) af_f += hf_f*basket_f(xf_f) + wdot_f(vf1_f, di1_f);

                    float vf2_f[5]; sigma_x_f(xf_f, vf2_f);
                    milstein_f(xf_f, xfc, dw2_f, hf_f);
                    if (option==2) af_f += hf_f*basket_f(xf_f) + wdot_f(vf2_f, di2_f);

                    float vc_f[5]; sigma_x_f(xc_f, vc_f);
                    milstein_f(xc_f, xcc, dwc_f, hc_f);
                    if (option == 2) {
                        float bridge_f[5];
                        for(int i=0;i<kDim;i++) bridge_f[i] = di1_f[i]+di2_f[i]+0.25f*hc_f*ddw_f[i];
                        ac_f += hc_f*basket_f(xc_f) + wdot_f(vc_f, bridge_f);
                    }
                }
                if (option == 2) {
                    af_f -= 0.5f*hf_f*basket_f(xf_f);
                    ac_f -= 0.5f*hc_f*basket_f(xc_f);
                }
                float Pf=(option==1)?std::fmaxf(0.0f,basket_f(xf_f)-kK_f):std::fmaxf(0.0f,af_f-kK_f);
                float Pc=(option==1)?std::fmaxf(0.0f,basket_f(xc_f)-kK_f):std::fmaxf(0.0f,ac_f-kK_f);
                dP = disc*(Pf-Pc); Pfv = disc*Pf;

            } else {
                // Odd, k>=l_star: pure-fp32 chain TWICE on the same draws.
                double dP_runs[2], Pfv_runs[2];
                for (int run = 0; run < 2; ++run) {
                    float xf_f[5], xc_f[5];
                    for(int i=0;i<kDim;i++){ xf_f[i]=kK_f; xc_f[i]=kK_f; }
                    float af_f = 0.5f*hf_f*basket_f(xf_f);
                    float ac_f = 0.5f*hc_f*basket_f(xc_f);
                    float xfc[5]={0,0,0,0,0}, xcc[5]={0,0,0,0,0};

                    for (int n = 0; n < nc; ++n) {
                        float dw1_f[5], dw2_f[5], di1_f[5], di2_f[5];
                        draw_increment_f(sf_f, dw1_f);
                        if (option == 2) draw_integral_f(si_f, di1_f);
                        draw_increment_f(sf_f, dw2_f);
                        if (option == 2) draw_integral_f(si_f, di2_f);

                        float dwc_f[5], ddw_f[5];
                        for(int i=0;i<kDim;i++){ dwc_f[i]=dw1_f[i]+dw2_f[i]; ddw_f[i]=dw1_f[i]-dw2_f[i]; }

                        float vf1_f[5]; sigma_x_f(xf_f, vf1_f);
                        milstein_f(xf_f, xfc, dw1_f, hf_f);
                        if (option==2) af_f += hf_f*basket_f(xf_f) + wdot_f(vf1_f, di1_f);

                        float vf2_f[5]; sigma_x_f(xf_f, vf2_f);
                        milstein_f(xf_f, xfc, dw2_f, hf_f);
                        if (option==2) af_f += hf_f*basket_f(xf_f) + wdot_f(vf2_f, di2_f);

                        float vc_f[5]; sigma_x_f(xc_f, vc_f);
                        milstein_f(xc_f, xcc, dwc_f, hc_f);
                        if (option == 2) {
                            float bridge_f[5];
                            for(int i=0;i<kDim;i++) bridge_f[i] = di1_f[i]+di2_f[i]+0.25f*hc_f*ddw_f[i];
                            ac_f += hc_f*basket_f(xc_f) + wdot_f(vc_f, bridge_f);
                        }
                    }
                    if (option == 2) {
                        af_f -= 0.5f*hf_f*basket_f(xf_f);
                        ac_f -= 0.5f*hc_f*basket_f(xc_f);
                    }
                    float Pf=(option==1)?std::fmaxf(0.0f,basket_f(xf_f)-kK_f):std::fmaxf(0.0f,af_f-kK_f);
                    float Pc=(option==1)?std::fmaxf(0.0f,basket_f(xc_f)-kK_f):std::fmaxf(0.0f,ac_f-kK_f);
                    dP_runs[run]  = disc*(Pf-Pc);
                    Pfv_runs[run] = disc*Pf;
                }
                // Difference between two independent fp32 evaluations of the
                // same quantity -- zero-mean, pure fp32-rounding self-noise.
                dP  = dP_runs[0] - dP_runs[1];
                Pfv = Pfv_runs[0];
            }

            sums[0] += (l == 0) ? nf : 2.0*nf;
            sums[1] += dP;       sums[2] += dP*dP;
            sums[3] += dP*dP*dP; sums[4] += dP*dP*dP*dP;
            sums[5] += Pfv;      sums[6] += Pfv*Pfv;
            dY += dP; dY2 += dP*dP; dY3 += dP*dP*dP; dY4 += dP*dP*dP*dP;
            dPfsum += Pfv; dPf2sum += Pfv*Pfv;
        }

        double meanY = dY/N, varY = dY2/N - meanY*meanY, meanPf = dPfsum/N;
        std::fprintf(diagfp, "%3d  %3d  %-6s  %6d  %12.6e  %12.6e  %12.6e\n",
                     l, k, (l % 2 == 0) ? "even" : "odd", N, meanY, varY, meanPf);
        std::fflush(diagfp);

        double meanPf2 = dPf2sum/N, varPf = meanPf2 - meanPf*meanPf;
        double s3 = dY3/N, s4 = dY4/N;
        std::fprintf(momfp, "level %d: cost=%g del1=%.10g del2=%.10g var1=%.10g var2=%.10g s3=%.10g s4=%.10g\n",
                     l, (double)nf, meanY, meanPf, varY, varPf, s3, s4);
        std::fflush(momfp);
        return;
    }

    for (int np = 0; np < N; ++np) {

        // ------------------------------------------------------------------
        // l = 0: fp16 Milstein base level, 1 fine step. Everything fp16:
        // state, increment, payoff (points 1, 5, 7). Kahan-accumulated iff
        // kahan_mode (comp[] inert otherwise).
        // ------------------------------------------------------------------
        if (l == 0) {
            half  xf_h[5]; for(int i=0;i<kDim;i++) xf_h[i]=(half)kK_f;
            half  xf_c[5] = {(half)0.0f16,(half)0.0f16,(half)0.0f16,(half)0.0f16,(half)0.0f16};
            half  af_c    = (half)0.0f16;
            float dw_f[5], di_f[5];

            draw_increment_f(sf_f, dw_f);
            if (option == 2) draw_integral_f(si_f, di_f);

            // Point 4: narrow to fp16 immediately.
            half dw_h[5], di_h[5];
            for(int i=0;i<kDim;i++) {
                dw_h[i] = (half)dw_f[i];
                if (option == 2) di_h[i] = (half)di_f[i];
            }

            half af_h = halfhf * basket_h(xf_h);
            half vf_h[5]; sigma_x_h(xf_h, vf_h);
            milstein_h(xf_h, xf_c, dw_h, hf_h);

            half Pf_h;
            if (option == 2) {
                kahan_accum(af_h, af_c, halfhf * basket_h(xf_h) + wdot_h(vf_h, di_h));
                Pf_h = af_h - (half)kK_f > (half)0.0f16 ? af_h - (half)kK_f : (half)0.0f16;
            } else {
                Pf_h = basket_h(xf_h) - (half)kK_f > (half)0.0f16 ? basket_h(xf_h) - (half)kK_f : (half)0.0f16;
            }
            half dP_h = disc_h * Pf_h;
            double dP = (double)dP_h;

            sums[0]+=nf; sums[1]+=dP; sums[2]+=dP*dP;
            sums[3]+=dP*dP*dP; sums[4]+=dP*dP*dP*dP;
            sums[5]+=dP; sums[6]+=dP*dP;
            dY += dP; dY2 += dP*dP; dY3 += dP*dP*dP; dY4 += dP*dP*dP*dP;
            dPfsum += dP; dPf2sum += dP*dP;

        // ------------------------------------------------------------------
        // l = 1: fp32-minus-fp16 precision correction at base level. Both
        // chains use the SAME Kahan on/off switch and the SAME draws (fp16
        // chain narrows immediately, per point 4).
        // ------------------------------------------------------------------
        } else if (l == 1) {
            half  xf_h[5]; for(int i=0;i<kDim;i++) xf_h[i]=(half)kK_f;
            float xf_f[5]; for(int i=0;i<kDim;i++) xf_f[i]=kK_f;
            half  xf_hc[5] = {(half)0.0f16,(half)0.0f16,(half)0.0f16,(half)0.0f16,(half)0.0f16};
            float xf_fc[5] = {0,0,0,0,0};
            half  af_hc = (half)0.0f16;
            float af_fc = 0.0f;
            float dw_f[5], di_f[5];

            draw_increment_f(sf_f, dw_f);
            if (option == 2) draw_integral_f(si_f, di_f);

            half dw_h[5], di_h[5];
            for(int i=0;i<kDim;i++) {
                dw_h[i] = (half)dw_f[i];
                if (option == 2) di_h[i] = (half)di_f[i];
            }

            half  af_h = halfhf * basket_h(xf_h);
            float af_f = 0.5f*hf_f*basket_f(xf_f);

            half  vf_h[5]; sigma_x_h(xf_h, vf_h);
            float vf_f[5]; sigma_x_f(xf_f, vf_f);

            milstein_h(xf_h, xf_hc, dw_h, hf_h);
            milstein_f(xf_f, xf_fc, dw_f, hf_f);

            half   Pf_h; double Pf_f;
            if (option == 2) {
                kahan_accum(af_h, af_hc, halfhf * basket_h(xf_h) + wdot_h(vf_h, di_h));
                kahan_accum(af_f, af_fc, 0.5f*hf_f*basket_f(xf_f) + wdot_f(vf_f, di_f));
                Pf_h = af_h - (half)kK_f > (half)0.0f16 ? af_h - (half)kK_f : (half)0.0f16;
                Pf_f = std::fmaxf(0.0f, af_f - kK_f);
            } else {
                Pf_h = basket_h(xf_h) - (half)kK_f > (half)0.0f16 ? basket_h(xf_h) - (half)kK_f : (half)0.0f16;
                Pf_f = std::fmaxf(0.0f, basket_f(xf_f) - kK_f);
            }
            double dP_h = disc * (double)Pf_h;
            double dP_f = disc * Pf_f;
            double dP = dP_f - dP_h;

            sums[0]+=2.0*nf; sums[1]+=dP; sums[2]+=dP*dP;
            sums[3]+=dP*dP*dP; sums[4]+=dP*dP*dP*dP;
            sums[5]+=dP; sums[6]+=dP*dP;
            dY += dP; dY2 += dP*dP; dY3 += dP*dP*dP; dY4 += dP*dP*dP*dP;
            dPfsum += dP_f; dPf2sum += dP_f*dP_f;

        // ------------------------------------------------------------------
        // l even >= 2: fp16 Milstein MLMC correction (nf fine, nc coarse).
        // Everything fp16: state, running sum, payoff, bracketed. Kahan-
        // accumulated iff kahan_mode.
        // ------------------------------------------------------------------
        } else if (l % 2 == 0) {
            half xf_h[5], xc_h[5];
            for(int i=0;i<kDim;i++){ xf_h[i]=(half)kK_f; xc_h[i]=(half)kK_f; }
            half xf_c[5]={(half)0.0f16,(half)0.0f16,(half)0.0f16,(half)0.0f16,(half)0.0f16};
            half xc_c[5]={(half)0.0f16,(half)0.0f16,(half)0.0f16,(half)0.0f16,(half)0.0f16};
            half af_c = (half)0.0f16, ac_c = (half)0.0f16;

            half af_h = halfhf * basket_h(xf_h);
            half ac_h = halfhc * basket_h(xc_h);

            for (int n = 0; n < nc; ++n) {
                float dw1_f[5], dw2_f[5], di1_f[5], di2_f[5];
                draw_increment_f(sf_f, dw1_f);
                if (option == 2) draw_integral_f(si_f, di1_f);
                draw_increment_f(sf_f, dw2_f);
                if (option == 2) draw_integral_f(si_f, di2_f);

                half dw1_h[5], dw2_h[5], di1_h[5], di2_h[5];
                half dwc_h[5], ddw_h[5];
                for(int i=0;i<kDim;i++){
                    dw1_h[i] = (half)dw1_f[i];
                    dw2_h[i] = (half)dw2_f[i];
                    if (option == 2) {
                        di1_h[i] = (half)di1_f[i];
                        di2_h[i] = (half)di2_f[i];
                    }
                    dwc_h[i] = dw1_h[i] + dw2_h[i];
                    ddw_h[i] = dw1_h[i] - dw2_h[i];
                }

                half vf1_h[5]; sigma_x_h(xf_h, vf1_h);
                milstein_h(xf_h, xf_c, dw1_h, hf_h);
                if (option == 2) kahan_accum(af_h, af_c, hf_h * basket_h(xf_h) + wdot_h(vf1_h, di1_h));

                half vf2_h[5]; sigma_x_h(xf_h, vf2_h);
                milstein_h(xf_h, xf_c, dw2_h, hf_h);
                if (option == 2) kahan_accum(af_h, af_c, hf_h * basket_h(xf_h) + wdot_h(vf2_h, di2_h));

                half vc_h[5]; sigma_x_h(xc_h, vc_h);
                milstein_h(xc_h, xc_c, dwc_h, hc_h);
                if (option == 2) {
                    half bridge_h[5];
                    half quarter = (half)0.25f16 * hc_h;
                    for(int i=0;i<kDim;i++) bridge_h[i] = di1_h[i] + di2_h[i] + quarter * ddw_h[i];
                    kahan_accum(ac_h, ac_c, hc_h * basket_h(xc_h) + wdot_h(vc_h, bridge_h));
                }
            }
            if (option == 2) {
                af_h = af_h - halfhf * basket_h(xf_h);
                ac_h = ac_h - halfhc * basket_h(xc_h);
            }

            half Pf_h = (option==1) ? (basket_h(xf_h)-(half)kK_f > (half)0.0f16 ? basket_h(xf_h)-(half)kK_f : (half)0.0f16)
                                    : (af_h - (half)kK_f > (half)0.0f16 ? af_h - (half)kK_f : (half)0.0f16);
            half Pc_h = (option==1) ? (basket_h(xc_h)-(half)kK_f > (half)0.0f16 ? basket_h(xc_h)-(half)kK_f : (half)0.0f16)
                                    : (ac_h - (half)kK_f > (half)0.0f16 ? ac_h - (half)kK_f : (half)0.0f16);

            half dY_h  = Pf_h - Pc_h;
            half dP_h  = disc_h * dY_h;
            half Pfv_h = disc_h * Pf_h;

            double dP  = (double)dP_h;
            double Pfv = (double)Pfv_h;

            sums[0]+=nf; sums[1]+=dP; sums[2]+=dP*dP;
            sums[3]+=dP*dP*dP; sums[4]+=dP*dP*dP*dP;
            sums[5]+=Pfv; sums[6]+=Pfv*Pfv;
            dY += dP; dY2 += dP*dP; dY3 += dP*dP*dP; dY4 += dP*dP*dP*dP;
            dPfsum += Pfv; dPf2sum += Pfv*Pfv;

        // ------------------------------------------------------------------
        // l odd >= 3: fp32-minus-fp16 precision correction. Both chains use
        // the SAME Kahan on/off switch and the SAME draws (fp16 chain
        // narrows immediately, per point 4).
        // ------------------------------------------------------------------
        } else {
            half  xf_h[5], xc_h[5];
            float xf_f[5], xc_f[5];
            for(int i=0;i<kDim;i++){
                xf_h[i]=(half)kK_f; xc_h[i]=(half)kK_f;
                xf_f[i]=kK_f;       xc_f[i]=kK_f;
            }
            half  xf_hc[5]={(half)0.0f16,(half)0.0f16,(half)0.0f16,(half)0.0f16,(half)0.0f16};
            half  xc_hc[5]={(half)0.0f16,(half)0.0f16,(half)0.0f16,(half)0.0f16,(half)0.0f16};
            float xf_fc[5]={0,0,0,0,0}, xc_fc[5]={0,0,0,0,0};
            half  af_hc = (half)0.0f16, ac_hc = (half)0.0f16;
            float af_fc = 0.0f, ac_fc = 0.0f;

            half  af_h = halfhf * basket_h(xf_h);
            half  ac_h = halfhc * basket_h(xc_h);
            float af_f = 0.5f*hf_f*basket_f(xf_f);
            float ac_f = 0.5f*hc_f*basket_f(xc_f);

            for (int n = 0; n < nc; ++n) {
                float dw1_f[5], dw2_f[5], di1_f[5], di2_f[5];
                draw_increment_f(sf_f, dw1_f);
                if (option == 2) draw_integral_f(si_f, di1_f);
                draw_increment_f(sf_f, dw2_f);
                if (option == 2) draw_integral_f(si_f, di2_f);

                half dw1_h[5], dw2_h[5], di1_h[5], di2_h[5];
                half dwc_h[5], ddw_h[5];
                float dwc_f[5], ddw_f[5];

                for(int i=0;i<kDim;i++){
                    dw1_h[i] = (half)dw1_f[i];
                    dw2_h[i] = (half)dw2_f[i];
                    if (option == 2) {
                        di1_h[i] = (half)di1_f[i];
                        di2_h[i] = (half)di2_f[i];
                    }
                    dwc_h[i] = dw1_h[i] + dw2_h[i];
                    ddw_h[i] = dw1_h[i] - dw2_h[i];
                    dwc_f[i] = dw1_f[i] + dw2_f[i];
                    ddw_f[i] = dw1_f[i] - dw2_f[i];
                }

                // ---- fp16 fine step 1 ----
                half vf1_h[5]; sigma_x_h(xf_h, vf1_h);
                milstein_h(xf_h, xf_hc, dw1_h, hf_h);
                if (option==2) kahan_accum(af_h, af_hc, hf_h * basket_h(xf_h) + wdot_h(vf1_h, di1_h));

                // ---- fp32 fine step 1 ----
                float vf1_f[5]; sigma_x_f(xf_f, vf1_f);
                milstein_f(xf_f, xf_fc, dw1_f, hf_f);
                if (option==2) kahan_accum(af_f, af_fc, hf_f*basket_f(xf_f) + wdot_f(vf1_f, di1_f));

                // ---- fp16 fine step 2 ----
                half vf2_h[5]; sigma_x_h(xf_h, vf2_h);
                milstein_h(xf_h, xf_hc, dw2_h, hf_h);
                if (option==2) kahan_accum(af_h, af_hc, hf_h * basket_h(xf_h) + wdot_h(vf2_h, di2_h));

                // ---- fp32 fine step 2 ----
                float vf2_f[5]; sigma_x_f(xf_f, vf2_f);
                milstein_f(xf_f, xf_fc, dw2_f, hf_f);
                if (option==2) kahan_accum(af_f, af_fc, hf_f*basket_f(xf_f) + wdot_f(vf2_f, di2_f));

                // ---- fp16 coarse step ----
                half vc_h[5]; sigma_x_h(xc_h, vc_h);
                milstein_h(xc_h, xc_hc, dwc_h, hc_h);

                // ---- fp32 coarse step ----
                float vc_f[5]; sigma_x_f(xc_f, vc_f);
                milstein_f(xc_f, xc_fc, dwc_f, hc_f);

                if (option == 2) {
                    half bridge_h[5];
                    float bridge_f[5];
                    half quarter_h = (half)0.25f16 * hc_h;
                    for(int i=0;i<kDim;i++){
                        bridge_h[i] = di1_h[i] + di2_h[i] + quarter_h * ddw_h[i];
                        bridge_f[i] = di1_f[i] + di2_f[i] + 0.25f*hc_f*ddw_f[i];
                    }
                    kahan_accum(ac_h, ac_hc, hc_h * basket_h(xc_h) + wdot_h(vc_h, bridge_h));
                    kahan_accum(ac_f, ac_fc, hc_f*basket_f(xc_f) + wdot_f(vc_f, bridge_f));
                }
            }

            if (option == 2) {
                af_h = af_h - halfhf * basket_h(xf_h);
                ac_h = ac_h - halfhc * basket_h(xc_h);
                af_f -= 0.5f*hf_f*basket_f(xf_f);
                ac_f -= 0.5f*hc_f*basket_f(xc_f);
            }

            half  Pf_h = (option==1)?(basket_h(xf_h)-(half)kK_f > (half)0.0f16 ? basket_h(xf_h)-(half)kK_f : (half)0.0f16):(af_h-(half)kK_f > (half)0.0f16 ? af_h-(half)kK_f : (half)0.0f16);
            half  Pc_h = (option==1)?(basket_h(xc_h)-(half)kK_f > (half)0.0f16 ? basket_h(xc_h)-(half)kK_f : (half)0.0f16):(ac_h-(half)kK_f > (half)0.0f16 ? ac_h-(half)kK_f : (half)0.0f16);
            float Pf_f = (option==1)?std::fmaxf(0.0f,basket_f(xf_f)-kK_f):std::fmaxf(0.0f,af_f-kK_f);
            float Pc_f = (option==1)?std::fmaxf(0.0f,basket_f(xc_f)-kK_f):std::fmaxf(0.0f,ac_f-kK_f);

            double dP_h = disc * ((double)Pf_h - (double)Pc_h);
            double dP_f = disc * (Pf_f - Pc_f);
            double dP   = dP_f - dP_h;

            double Pfv  = disc * Pf_f;

            sums[0]+=2.0*nf; sums[1]+=dP; sums[2]+=dP*dP;
            sums[3]+=dP*dP*dP; sums[4]+=dP*dP*dP*dP;
            sums[5]+=Pfv; sums[6]+=Pfv*Pfv;
            dY += dP; dY2 += dP*dP; dY3 += dP*dP*dP; dY4 += dP*dP*dP*dP;
            dPfsum += Pfv; dPf2sum += Pfv*Pfv;
        }
    }

    double meanY = dY / N, varY = dY2/N - meanY*meanY, meanPf = dPfsum / N;
    std::fprintf(diagfp, "%3d  %3d  %-6s  %6d  %12.6e  %12.6e  %12.6e\n",
                 l, k, (l % 2 == 0) ? "even" : "odd", N, meanY, varY, meanPf);
    std::fflush(diagfp);

    double meanPf2 = dPf2sum / N, varPf = meanPf2 - meanPf*meanPf;
    double s3 = dY3 / N, s4 = dY4 / N;
    std::fprintf(momfp, "level %d: cost=%g del1=%.10g del2=%.10g var1=%.10g var2=%.10g s3=%.10g s4=%.10g\n",
                 l, (double)nf, meanY, meanPf, varY, varPf, s3, s4);
    std::fflush(momfp);
}
