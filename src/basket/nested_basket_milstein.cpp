/*
 * Nested MLMC for basket European and Asian calls — Milstein fp32/fp64 unified
 * template. Two sweep modes (controlled by adaptive_mode global), mirroring
 * src/scalar/nested_scalar_milstein.cpp's standard/adaptive structure exactly:
 *
 *   standard: vanilla fp32/fp64 nested MLMC
 *     - Even levels (l=2k):  float (fp32) Milstein MLMC correction
 *     - Odd levels (l=2k+1): double (fp64) - float (fp32) precision correction
 *     - Both payoffs (option=1 European, option=2 Asian), l_star unused
 *
 *   adaptive: fp32/fp64 with grid-level cutoff
 *     - Below l_star (k < l_star): same as standard (fp32 fine, fp64 correction)
 *     - At/above l_star (k >= l_star):
 *       * Even levels: pure double (fp64) MLMC correction (no fp32)
 *       * Odd levels: cost counted, correction = 0 (replaced by fp64-vs-fp64 check)
 *     - Per-option cutoffs: l_star = 6 for both options (not yet independently
 *       tuned from basket's own fp32-vs-fp64 divergence data, unlike scalar's
 *       Asian=5/Lookback=6 which were picked from that file's own climb --
 *       revisit if a tighter basket-specific cutoff is needed)
 *
 * No Kahan compensation anywhere in this file: fp32 and fp64 do not suffer
 * fp16's severe per-step rounding, so plain += accumulation is sufficient at
 * both precisions (Kahan is only needed in the separate fp16 template,
 * nested_basket_milstein_fp16.cpp).
 *
 * Model: 5-asset correlated GBM (matches basket_scalar.cpp),
 *   S0 = K = 100, T = 1, r = 0.05, sigma = {0.25..0.45}, weights = 0.2,
 *   pairwise rho = 0.25, Cholesky decomposition for correlation.
 *
 * Both float and double paths are driven by the SAME float random numbers
 * so their difference (precision correction) has small variance.
 *
 * Build (from repo root):
 *   g++ -O2 -std=c++11 src/basket/nested_basket_milstein.cpp \
 *       -o build/nested_basket_milstein
 *   cd outputs && ../build/nested_basket_milstein
 *   Output files: nested_basket_standard_{1,2}.txt, nested_basket_adaptive_{1,2}.txt
 *     option 1 = European call, option 2 = Asian call
 *
 * Plot:
 *   MPLBACKEND=Agg python3 python/nested_mlmc_plot_python.py outputs/nested_basket_standard_1
 *   MPLBACKEND=Agg python3 python/nested_mlmc_plot_python.py outputs/nested_basket_adaptive_1
 */

#include "../core/nested_mlmc_test.cpp"
#include "../core/mlmc_rng.cpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <array>

// 1 = Basket European, 2 = Basket Asian
int option;

// Adaptive mode switch: false = standard (fp32/fp64), true = adaptive
// (fp32 below l_star, pure fp64 above).
static bool adaptive_mode = false;
static int  l_star = 0;
static const int l_star_by_option[] = {0, 6, 6};  // index by option 1/2

void nested_basket_l(int, int, double *);

// ============================================================
// Model constants (double and float copies)
// ============================================================
static const int    kDim = 5;
static const double kK   = 100.0, kT = 1.0, kR = 0.05;
static const double kSig[5] = {0.25, 0.30, 0.35, 0.40, 0.45};
static const double kW[5]   = {0.2, 0.2, 0.2, 0.2, 0.2};
static const double kRho    = 0.25;

static const float kK_f   = 100.0f, kT_f = 1.0f, kR_f = 0.05f;
static const float kSig_f[5] = {0.25f, 0.30f, 0.35f, 0.40f, 0.45f};
static const float kW_f[5]   = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};

// ============================================================
// Cholesky of correlation matrix (double), precomputed once.
// L_d[i][j]  for i >= j, 0 otherwise.
// ============================================================
static double L_d[5][5];
static float  L_f[5][5];

static void build_cholesky()
{
    // Correlation matrix: diag = 1, off-diag = kRho
    double a[5][5];
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            a[i][j] = (i == j) ? 1.0 : kRho;

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = a[i][j];
            for (int k = 0; k < j; ++k) s -= L_d[i][k] * L_d[j][k];
            L_d[i][j] = (i == j) ? std::sqrt(s) : s / L_d[j][j];
        }
    }
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            L_f[i][j] = (float)L_d[i][j];
}

// ============================================================
// Helper: generate kDim correlated standard normals using L.
// Draws are taken from next_normal() (returns float) and
// stored in both float and double arrays simultaneously so
// both precision paths share the same realization.
// ============================================================
static inline void draw_correlated(float z_f[5], double z_d[5])
{
    float y[5];
    for (int i = 0; i < kDim; ++i) y[i] = next_normal();

    for (int i = 0; i < kDim; ++i) {
        float  sf = 0.0f;
        double sd = 0.0;
        for (int j = 0; j <= i; ++j) {
            sf += L_f[i][j] * y[j];
            sd += L_d[i][j] * (double)y[j];
        }
        z_f[i] = sf;
        z_d[i] = sd;
    }
}

// Draw scaled increment vectors for both precisions.
// scale_f = sqrt(h) as float, scale_d = sqrt(h) as double.
static inline void draw_increment(float scale_f, double scale_d,
                                  float dw_f[5], double dw_d[5])
{
    float  z_f[5];
    double z_d[5];
    draw_correlated(z_f, z_d);
    for (int i = 0; i < kDim; ++i) {
        dw_f[i] = scale_f * z_f[i];
        dw_d[i] = scale_d * z_d[i];
    }
}

// ============================================================
// Milstein step (in-place) — float and double versions.
// x[i] += r*x[i]*h + sig[i]*x[i]*dw[i] + 0.5*sig[i]^2*x[i]*(dw[i]^2-h)
// ============================================================
static inline void milstein_f(float x[5], const float dw[5], float h)
{
    for (int i = 0; i < kDim; ++i) {
        float xi = x[i], sig = kSig_f[i];
        x[i] = xi + kR_f*xi*h + sig*xi*dw[i] + 0.5f*sig*sig*xi*(dw[i]*dw[i] - h);
    }
}

static inline void milstein_d(double x[5], const double dw[5], double h)
{
    for (int i = 0; i < kDim; ++i) {
        double xi = x[i], sig = kSig[i];
        x[i] = xi + kR*xi*h + sig*xi*dw[i] + 0.5*sig*sig*xi*(dw[i]*dw[i] - h);
    }
}

// Basket (weighted sum of assets)
static inline float  basket_f(const float x[5])
{ float s=0; for(int i=0;i<kDim;i++) s+=kW_f[i]*x[i]; return s; }
static inline double basket_d(const double x[5])
{ double s=0; for(int i=0;i<kDim;i++) s+=kW[i]*x[i]; return s; }

// sigma_i * x_i  (vf = sigma * state)
static inline void sigma_x_f(const float x[5], float v[5])
{ for(int i=0;i<kDim;i++) v[i]=kSig_f[i]*x[i]; }
static inline void sigma_x_d(const double x[5], double v[5])
{ for(int i=0;i<kDim;i++) v[i]=kSig[i]*x[i]; }

// Weighted dot: sum_i w_i * a_i * b_i  (Asian bridge correction must carry weights)
static inline float  wdot_f(const float a[5], const float b[5])
{ float s=0; for(int i=0;i<kDim;i++) s+=kW_f[i]*a[i]*b[i]; return s; }
static inline double wdot_d(const double a[5], const double b[5])
{ double s=0; for(int i=0;i<kDim;i++) s+=kW[i]*a[i]*b[i]; return s; }

// ============================================================
// main
// ============================================================
int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    build_cholesky();

    int N0   = 200;
    int Lmin = 2;
    int Lmax = 20;

    int   N    = 50000;  // match Giles reference count
    int   L    = 10;     // nested levels for convergence table
    char  filename[64];
    FILE *fp;

    float Eps[] = { 0.01f, 0.02f, 0.05f, 0.1f, 0.2f, 0.0f };  // Giles Eps

    // Two sweep modes: standard (vanilla fp32/fp64), adaptive (fp32/fp64 with cutoff)
    const bool adapt_modes[2] = { false, true };
    const char *mode_tag[2]   = { "standard", "adaptive" };

    for (int m = 0; m < 2; ++m) {
        adaptive_mode = adapt_modes[m];

        for (option = 1; option <= 2; ++option) {
            l_star = l_star_by_option[option];
            rng_initialisation();

            std::sprintf(filename, "nested_basket_%s_%d.txt", mode_tag[m], option);
            fp = std::fopen(filename, "w");
            if (!fp) { std::perror("fopen"); return EXIT_FAILURE; }

            std::printf("\n ---- Nested basket %s %s (Milstein, fp32/fp64%s) ----\n",
                        mode_tag[m],
                        option == 1 ? "European" : "Asian",
                        adaptive_mode ? ", adaptive l_star" : "");
            if (adaptive_mode) std::printf("      l_star = %d\n", l_star);

            mlmc_test(nested_basket_l, N, L, N0, Eps, Lmin, Lmax, fp);
            std::fclose(fp);
            rng_termination();
        }
    }
    return EXIT_SUCCESS;
}

/* ============================================================
 * nested_basket_l: the mlmc_l estimator for nested basket MLMC.
 *
 * Mode control (global variable adaptive_mode, set by main()):
 *   adaptive_mode = false (standard mode):
 *     All levels use vanilla fp32/fp64 nested MLMC.
 *     Even: fp32 fine/coarse MLMC correction.
 *     Odd:  fp64 - fp32 precision correction.
 *
 *   adaptive_mode = true (adaptive mode):
 *     Below l_star (grid level k < l_star): same as standard (fp32/fp64).
 *     At/above l_star (k >= l_star):
 *       Even: pure fp64 MLMC correction (no fp32 involved).
 *       Odd:  zero correction (cost still counted; replaced by fp64-vs-fp64
 *             check above cutoff).
 *     l_star = 6 for both options.
 * ============================================================ */
void nested_basket_l(int l, int N, double *sums)
{
    const int nf = 1 << (l / 2);
    const int nc = nf / 2;
    const int k  = l / 2;

    // Time steps
    const float  hf_f = kT_f / (float)nf;
    const float  hc_f = (nc > 0) ? kT_f / (float)nc : 0.0f;
    const double hf_d = kT / (double)nf;
    const double hc_d = (nc > 0) ? kT / (double)nc : 0.0;

    const float  disc_f = std::exp(-kR_f * kT_f);
    const double disc_d = std::exp(-kR  * kT);

    // Scales for increment drawing
    const float  sf_f = sqrtf(hf_f);
    const double sf_d = std::sqrt(hf_d);
    const float  si_f = sqrtf(hf_f / 12.0f) * hf_f;  // integral increment scale
    const double si_d = std::sqrt(hf_d / 12.0) * hf_d;

    for (int i = 0; i < 7; ++i) sums[i] = 0.0;

    // ----------------------------------------------------------------------
    // Adaptive mode: odd level at or above cut-off returns zero correction
    // (cost still counted, but correction = 0) -- mirrors
    // nested_scalar_milstein.cpp's adaptive convention exactly.
    // ----------------------------------------------------------------------
    if (adaptive_mode && l % 2 == 1 && k >= l_star) {
        sums[0] = (double)N * 2.0 * nf;
        return;
    }

    // ----------------------------------------------------------------------
    // Adaptive mode, at/above cut-off: pure fp64 MLMC correction (no fp32).
    // ----------------------------------------------------------------------
    if (adaptive_mode && k >= l_star && l % 2 == 0) {
        for (int np = 0; np < N; ++np) {
            double dP, Pfv;

            if (l == 0) {
                double xf[5]; for (int i=0;i<kDim;i++) xf[i]=kK;
                float  dw_f[5], di_f[5];
                double dw_d[5], di_d[5];
                draw_increment(sf_f, sf_d, dw_f, dw_d);
                if (option == 2) draw_increment(si_f, si_d, di_f, di_d);
                double af = 0.5*hf_d*basket_d(xf);
                double vf[5]; sigma_x_d(xf, vf);
                milstein_d(xf, dw_d, hf_d);
                double Pf;
                if (option == 2) { af += 0.5*hf_d*basket_d(xf)+wdot_d(vf,di_d); Pf=std::fmax(0.0,af-kK); }
                else Pf = std::fmax(0.0, basket_d(xf) - kK);
                dP = disc_d * Pf; Pfv = dP;
            } else {
                double xf[5], xc[5];
                for (int i=0;i<kDim;i++){ xf[i]=kK; xc[i]=kK; }
                double af = 0.5*hf_d*basket_d(xf), ac = 0.5*hc_d*basket_d(xc);
                for (int n = 0; n < nc; ++n) {
                    float  dw1_f[5], dw2_f[5], di1_f[5], di2_f[5];
                    double dw1_d[5], dw2_d[5], di1_d[5], di2_d[5];
                    draw_increment(sf_f, sf_d, dw1_f, dw1_d); if (option==2) draw_increment(si_f, si_d, di1_f, di1_d);
                    draw_increment(sf_f, sf_d, dw2_f, dw2_d); if (option==2) draw_increment(si_f, si_d, di2_f, di2_d);
                    double dwc[5], ddw[5];
                    for (int i=0;i<kDim;i++){ dwc[i]=dw1_d[i]+dw2_d[i]; ddw[i]=dw1_d[i]-dw2_d[i]; }
                    double vf1[5]; sigma_x_d(xf, vf1); milstein_d(xf, dw1_d, hf_d);
                    if (option==2) af += hf_d*basket_d(xf)+wdot_d(vf1,di1_d);
                    double vf2[5]; sigma_x_d(xf, vf2); milstein_d(xf, dw2_d, hf_d);
                    if (option==2) af += hf_d*basket_d(xf)+wdot_d(vf2,di2_d);
                    double vc[5]; sigma_x_d(xc, vc); milstein_d(xc, dwc, hc_d);
                    if (option==2) {
                        double bridge[5];
                        for (int i=0;i<kDim;i++) bridge[i]=di1_d[i]+di2_d[i]+0.25*hc_d*ddw[i];
                        ac += hc_d*basket_d(xc)+wdot_d(vc,bridge);
                    }
                }
                if (option == 2) { af -= 0.5*hf_d*basket_d(xf); ac -= 0.5*hc_d*basket_d(xc); }
                double Pf=(option==1)?std::fmax(0.0,basket_d(xf)-kK):std::fmax(0.0,af-kK);
                double Pc=(option==1)?std::fmax(0.0,basket_d(xc)-kK):std::fmax(0.0,ac-kK);
                dP=disc_d*(Pf-Pc); Pfv=disc_d*Pf;
            }

            sums[0] += nf;
            sums[1] += dP;       sums[2] += dP*dP;
            sums[3] += dP*dP*dP; sums[4] += dP*dP*dP*dP;
            sums[5] += Pfv;      sums[6] += Pfv*Pfv;
        }
        return;
    }

    for (int np = 0; np < N; ++np) {

        // ------------------------------------------------------------------
        // l = 0: float Milstein base level, 1 fine step, no coarse
        // (standard and adaptive-below-cutoff both use this -- adaptive's
        // cutoff never applies at k=0 since l_star >= 1)
        // ------------------------------------------------------------------
        if (l == 0) {
            float  xf_f[5], dw_f[5], di_f[5], vf_f[5];
            double xf_d[5], dw_d[5], di_d[5];   // (unused for payoff)
            for (int i=0;i<kDim;i++) xf_f[i]=kK_f;

            draw_increment(sf_f, sf_d, dw_f, dw_d);
            if (option == 2) draw_increment(si_f, si_d, di_f, di_d);

            float af_f = 0.5f * hf_f * basket_f(xf_f);
            sigma_x_f(xf_f, vf_f);
            milstein_f(xf_f, dw_f, hf_f);

            float Pf;
            if (option == 2) {
                af_f += 0.5f*hf_f*basket_f(xf_f) + wdot_f(vf_f, di_f);
                Pf = std::fmaxf(0.0f, af_f - kK_f);
            } else {
                Pf = std::fmaxf(0.0f, basket_f(xf_f) - kK_f);
            }
            double dP = disc_f * Pf;

            sums[0] += nf;
            sums[1]+=dP; sums[2]+=dP*dP; sums[3]+=dP*dP*dP; sums[4]+=dP*dP*dP*dP;
            sums[5]+=dP; sums[6]+=dP*dP;

        // ------------------------------------------------------------------
        // l = 1: precision correction at base level (1 fine step)
        // Standard mode: fp32 path vs fp64 path, same random draws.
        // Adaptive mode: same as standard (no cutoff effect at level 1).
        // ------------------------------------------------------------------
        } else if (l == 1) {
            float  xf_f[5], dw_f[5], di_f[5], vf_f[5];
            double xf_d[5], dw_d[5], di_d[5], vf_d[5];
            for (int i=0;i<kDim;i++){ xf_f[i]=kK_f; xf_d[i]=kK; }

            draw_increment(sf_f, sf_d, dw_f, dw_d);
            if (option == 2) draw_increment(si_f, si_d, di_f, di_d);

            float  af_f = 0.5f * hf_f * basket_f(xf_f);
            double af_d = 0.5  * hf_d * basket_d(xf_d);
            sigma_x_f(xf_f, vf_f);
            sigma_x_d(xf_d, vf_d);
            milstein_f(xf_f, dw_f, hf_f);
            milstein_d(xf_d, dw_d, hf_d);

            double Pf_f, Pf_d;
            if (option == 2) {
                af_f += 0.5f*hf_f*basket_f(xf_f) + wdot_f(vf_f, di_f);
                af_d += 0.5 *hf_d*basket_d(xf_d) + wdot_d(vf_d, di_d);
                Pf_f = disc_f * std::fmaxf(0.0f, af_f - kK_f);
                Pf_d = disc_d * std::fmax(0.0,   af_d - kK);
            } else {
                Pf_f = disc_f * std::fmaxf(0.0f, basket_f(xf_f) - kK_f);
                Pf_d = disc_d * std::fmax(0.0,   basket_d(xf_d) - kK);
            }
            double dP = Pf_d - Pf_f;   // precision correction of base level

            sums[0] += 2.0 * nf;
            sums[1]+=dP; sums[2]+=dP*dP; sums[3]+=dP*dP*dP; sums[4]+=dP*dP*dP*dP;
            sums[5]+=dP; sums[6]+=dP*dP;

        // ------------------------------------------------------------------
        // l even >= 2: float Milstein MLMC correction (nf fine, nc coarse)
        // Standard mode: both fine/coarse in fp32.
        // Adaptive mode (k < l_star): same as standard.
        // ------------------------------------------------------------------
        } else if (l % 2 == 0) {
            float xf_f[5], xc_f[5];
            for (int i=0;i<kDim;i++){ xf_f[i]=kK_f; xc_f[i]=kK_f; }

            float af_f = 0.5f*hf_f*basket_f(xf_f);
            float ac_f = 0.5f*hc_f*basket_f(xc_f);

            for (int n = 0; n < nc; ++n) {
                float dw1_f[5], dw2_f[5], di1_f[5], di2_f[5];
                double dw1_d[5], dw2_d[5], di1_d[5], di2_d[5];  // draw but discard double

                draw_increment(sf_f, sf_d, dw1_f, dw1_d);
                if (option == 2) draw_increment(si_f, si_d, di1_f, di1_d);
                draw_increment(sf_f, sf_d, dw2_f, dw2_d);
                if (option == 2) draw_increment(si_f, si_d, di2_f, di2_d);

                // Fine step 1
                float vf1[5]; sigma_x_f(xf_f, vf1);
                milstein_f(xf_f, dw1_f, hf_f);
                if (option == 2) af_f += hf_f*basket_f(xf_f) + wdot_f(vf1, di1_f);

                // Fine step 2
                float vf2[5]; sigma_x_f(xf_f, vf2);
                milstein_f(xf_f, dw2_f, hf_f);
                if (option == 2) af_f += hf_f*basket_f(xf_f) + wdot_f(vf2, di2_f);

                // Coarse step: dWc = dW1 + dW2
                float dwc_f[5], ddw_f[5];
                for (int i=0;i<kDim;i++){
                    dwc_f[i] = dw1_f[i]+dw2_f[i];
                    ddw_f[i] = dw1_f[i]-dw2_f[i];
                }
                float vc[5]; sigma_x_f(xc_f, vc);
                milstein_f(xc_f, dwc_f, hc_f);
                if (option == 2) {
                    float bridge[5];
                    for (int i=0;i<kDim;i++)
                        bridge[i] = di1_f[i]+di2_f[i]+0.25f*hc_f*ddw_f[i];
                    ac_f += hc_f*basket_f(xc_f) + wdot_f(vc, bridge);
                }
            }
            if (option == 2) {
                af_f -= 0.5f*hf_f*basket_f(xf_f);
                ac_f -= 0.5f*hc_f*basket_f(xc_f);
            }

            float Pf = (option==1) ? std::fmaxf(0.0f, basket_f(xf_f)-kK_f)
                                   : std::fmaxf(0.0f, af_f - kK_f);
            float Pc = (option==1) ? std::fmaxf(0.0f, basket_f(xc_f)-kK_f)
                                   : std::fmaxf(0.0f, ac_f - kK_f);

            double dP  = disc_f * (Pf - Pc);
            double Pfv = disc_f * Pf;

            sums[0]+=nf; sums[1]+=dP; sums[2]+=dP*dP;
            sums[3]+=dP*dP*dP; sums[4]+=dP*dP*dP*dP;
            sums[5]+=Pfv; sums[6]+=Pfv*Pfv;

        // ------------------------------------------------------------------
        // l odd >= 3: double-minus-float precision correction
        // Standard mode: both fine/coarse in fp64, correction = fp64(Pf-Pc) - fp32(Pf-Pc).
        // Adaptive mode (k < l_star): same as standard.
        // ------------------------------------------------------------------
        } else {
            float  xf_f[5], xc_f[5];
            double xf_d[5], xc_d[5];
            for (int i=0;i<kDim;i++){
                xf_f[i]=kK_f; xc_f[i]=kK_f;
                xf_d[i]=kK;   xc_d[i]=kK;
            }

            float  af_f = 0.5f*hf_f*basket_f(xf_f);
            float  ac_f = 0.5f*hc_f*basket_f(xc_f);
            double af_d = 0.5 *hf_d*basket_d(xf_d);
            double ac_d = 0.5 *hc_d*basket_d(xc_d);

            for (int n = 0; n < nc; ++n) {
                float  dw1_f[5], dw2_f[5], di1_f[5], di2_f[5];
                double dw1_d[5], dw2_d[5], di1_d[5], di2_d[5];

                draw_increment(sf_f, sf_d, dw1_f, dw1_d);
                if (option == 2) draw_increment(si_f, si_d, di1_f, di1_d);
                draw_increment(sf_f, sf_d, dw2_f, dw2_d);
                if (option == 2) draw_increment(si_f, si_d, di2_f, di2_d);

                // Float fine step 1
                float vf1_f[5]; sigma_x_f(xf_f, vf1_f);
                milstein_f(xf_f, dw1_f, hf_f);
                if (option==2) af_f += hf_f*basket_f(xf_f) + wdot_f(vf1_f, di1_f);

                // Double fine step 1
                double vf1_d[5]; sigma_x_d(xf_d, vf1_d);
                milstein_d(xf_d, dw1_d, hf_d);
                if (option==2) af_d += hf_d*basket_d(xf_d) + wdot_d(vf1_d, di1_d);

                // Float fine step 2
                float vf2_f[5]; sigma_x_f(xf_f, vf2_f);
                milstein_f(xf_f, dw2_f, hf_f);
                if (option==2) af_f += hf_f*basket_f(xf_f) + wdot_f(vf2_f, di2_f);

                // Double fine step 2
                double vf2_d[5]; sigma_x_d(xf_d, vf2_d);
                milstein_d(xf_d, dw2_d, hf_d);
                if (option==2) af_d += hf_d*basket_d(xf_d) + wdot_d(vf2_d, di2_d);

                // Float coarse step
                float  dwc_f[5], ddw_f[5];
                double dwc_d[5], ddw_d[5];
                for (int i=0;i<kDim;i++){
                    dwc_f[i]=dw1_f[i]+dw2_f[i]; ddw_f[i]=dw1_f[i]-dw2_f[i];
                    dwc_d[i]=dw1_d[i]+dw2_d[i]; ddw_d[i]=dw1_d[i]-dw2_d[i];
                }
                float  vc_f[5]; sigma_x_f(xc_f, vc_f);
                double vc_d[5]; sigma_x_d(xc_d, vc_d);

                milstein_f(xc_f, dwc_f, hc_f);
                milstein_d(xc_d, dwc_d, hc_d);

                if (option == 2) {
                    float  bridge_f[5];
                    double bridge_d[5];
                    for (int i=0;i<kDim;i++){
                        bridge_f[i]=di1_f[i]+di2_f[i]+0.25f*hc_f*ddw_f[i];
                        bridge_d[i]=di1_d[i]+di2_d[i]+0.25 *hc_d*ddw_d[i];
                    }
                    ac_f += hc_f*basket_f(xc_f) + wdot_f(vc_f, bridge_f);
                    ac_d += hc_d*basket_d(xc_d) + wdot_d(vc_d, bridge_d);
                }
            }

            if (option == 2) {
                af_f -= 0.5f*hf_f*basket_f(xf_f); ac_f -= 0.5f*hc_f*basket_f(xc_f);
                af_d -= 0.5 *hf_d*basket_d(xf_d); ac_d -= 0.5 *hc_d*basket_d(xc_d);
            }

            double Pff = (option==1)?std::fmaxf(0.0f,basket_f(xf_f)-kK_f):std::fmaxf(0.0f,af_f-kK_f);
            double Pcf = (option==1)?std::fmaxf(0.0f,basket_f(xc_f)-kK_f):std::fmaxf(0.0f,ac_f-kK_f);
            double Pfd = (option==1)?std::fmax(0.0, basket_d(xf_d)-kK)   :std::fmax(0.0, af_d-kK);
            double Pcd = (option==1)?std::fmax(0.0, basket_d(xc_d)-kK)   :std::fmax(0.0, ac_d-kK);

            double dP_f = disc_f*(Pff - Pcf);
            double dP_d = disc_d*(Pfd - Pcd);
            double dP   = dP_d - dP_f;

            double Pfv  = disc_d*Pfd - disc_f*Pff;   // precision corr of fine payoff

            sums[0] += 2.0*nf;
            sums[1]+=dP; sums[2]+=dP*dP; sums[3]+=dP*dP*dP; sums[4]+=dP*dP*dP*dP;
            sums[5]+=Pfv; sums[6]+=Pfv*Pfv;
        }
    }
}
