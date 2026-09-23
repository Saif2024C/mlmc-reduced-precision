/*
 * Nested MLMC for the scalar Asian call — Milstein fp32/fp64 unified template.
 *
 * Two sweep modes (controlled by adaptive_mode global):
 *   standard: vanilla fp32/fp64 nested MLMC
 *     - Even levels (l=2k):  float (fp32) Milstein MLMC correction
 *     - Odd levels (l=2k+1):  double (fp64) - float (fp32) precision correction
 *     - option=1 Asian, with l_star unused
 *
 *   adaptive: fp32/fp64 with grid-level cutoff
 *     - Below l_star (k < l_star): same as standard (fp32 fine, fp64 correction)
 *     - At/above l_star (k >= l_star):
 *       * Even levels: pure double (fp64) MLMC correction (no fp32)
 *       * Odd levels: cost counted, correction = 0 (replaced by fp64-vs-fp64 check)
 *     - Cutoff: Asian l_star=5
 *
 * Telescoping sum estimate (standard mode):
 *   E[P_L^fine] = sum of even super-levels (fp32 paths)
 *   E[P_L^double - P_L^fine] = sum of odd super-levels (precision correction)
 *   Total E[P_L^double] is the final nested MLMC estimate
 *
 * Cost model:
 *   Even levels: cost = nf  (fine timesteps = 2^k)
 *   Odd levels: cost = 2*nf (both fine and coarse paths at 2*nf cost)
 *   Empirical alpha/beta/gamma estimated via linear regression on per-level diagnostics.
 *
 * ========== BUILD & RUN COMMANDS (from repo root) ==========
 *
 * Default build (no flags):
 *   g++ -O2 -std=c++11 src/scalar/nested_scalar_milstein.cpp -o build/nested_scalar_milstein
 *
 * Run both standard and adaptive sweeps:
 *   cd outputs && ../build/nested_scalar_milstein && cd ..
 *   Output files: nested_scalar_standard_1.txt, nested_scalar_adaptive_1.txt
 *     option 1 = Asian call
 *
 * Generate convergence + complexity plots:
 *   MPLBACKEND=Agg python3 python/nested_mlmc_plot_python.py outputs/nested_scalar_standard_1
 *   MPLBACKEND=Agg python3 python/nested_mlmc_plot_python.py outputs/nested_scalar_adaptive_1
 *   Output plots: outputs/nested/scalar/standard/nested_scalar_{standard,adaptive}_1_convergence.png
 *
 * All-in-one (rebuild, run, plot):
 *   g++ -O2 -std=c++11 src/scalar/nested_scalar_milstein.cpp -o build/nested_scalar_milstein && \
 *   cd outputs && ../build/nested_scalar_milstein && cd .. && \
 *   for mode in standard adaptive; do \
 *     MPLBACKEND=Agg python3 python/nested_mlmc_plot_python.py outputs/nested_scalar_${mode}_1; \
 *   done && \
 *   mv outputs/nested_scalar_*.png outputs/nested/scalar/standard/
 */

#include "../core/nested_mlmc_test.cpp"
#include "../core/mlmc_rng.cpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

// 1 = Asian call
int option;

// Adaptive mode switch: false = standard (fp32/fp64), true = adaptive (fp32 below l_star, fp64 above)
static bool adaptive_mode = false;
static int  l_star = 0;
static const int l_star_by_option[] = {0, 5};  // index by option 1

void nested_scalar_l(int, int, double *);

int main(int argc, char **argv)
{
    (void)argc; (void)argv;

    int N0   = 200;
    int Lmin = 2;
    int Lmax = 20;   // nested super-level max (10 grid levels)

    int   N    = 20000;
    int   L    = 10;  // nested levels for convergence table (5 grid levels shown)
    char  filename[64];
    FILE *fp;

    // Two sweep modes: standard (vanilla fp32/fp64), adaptive (fp32/fp64 with cutoff)
    const bool adapt_modes[2]  = { false, true };
    const char *mode_tag[2]    = { "standard", "adaptive" };

    for (int m = 0; m < 2; ++m) {
        adaptive_mode = adapt_modes[m];

        for (option = 1; option <= 1; ++option) {
            l_star = l_star_by_option[option];
            rng_initialisation();

            std::sprintf(filename, "nested_scalar_%s_%d.txt", mode_tag[m], option);
            fp = std::fopen(filename, "w");
            if (!fp) { std::perror("fopen"); return EXIT_FAILURE; }

            float Eps[] = { 0.005f, 0.01f, 0.02f, 0.05f, 0.1f, 0.0f };

            std::printf("\n ---- Nested scalar %s %s (Milstein, fp32/fp64%s) ----\n",
                        mode_tag[m],
                        "Asian",
                        adaptive_mode ? ", adaptive l_star" : "");
            if (adaptive_mode) std::printf("      l_star = %d\n", l_star);

            mlmc_test(nested_scalar_l, N, L, N0, Eps, Lmin, Lmax, fp);
            std::fclose(fp);
            rng_termination();
        }
    }
    return EXIT_SUCCESS;
}

/* ===================================================================
 * nested_scalar_l(l, N, sums): the mlmc_l estimator for the nested MLMC driver.
 *
 * Signature: void nested_scalar_l(int l, int N, double *sums)
 *   l    = super-level index (0..2*Lmax+1; even=MLMC, odd=precision correction)
 *   N    = number of Monte Carlo paths to simulate
 *   sums = output array of 7 doubles (allocated by caller)
 *
 * Output sums[] layout (matches nested_mlmc_test.cpp expectations):
 *   [0] = sum(cost)          cost per path (proxy: number of fine timesteps)
 *   [1] = sum(Y)             sum of correction deltas (Pf on level 0, Pf-Pc on l>0)
 *   [2] = sum(Y^2)           sum of squared corrections
 *   [3] = sum(Y^3)           third moment (for kurtosis diagnostics)
 *   [4] = sum(Y^4)           fourth moment
 *   [5] = sum(Pf)            sum of fine payoff (Pf on even, precision on odd)
 *   [6] = sum(Pf^2)          sum of squared fine payoff
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
 *       Odd:  zero correction (cost still counted; replaced by fp64-vs-fp64 check above cutoff).
 *     l_star: Asian=5.
 *
 * Grid level k and timestep count:
 *   k  = l / 2                  (grid level, 0..Lmax)
 *   nf = 2^k                    (fine timesteps: 1,1,2,2,4,4,...)
 *   nc = nf/2                   (coarse timesteps: 0,0,1,1,2,2,...)
 *
 * Path simulation:
 *   GBM SDE: dX = r*X*dt + sigma*X*dW  (log-normal stock price)
 *   Milstein: O(dt) strong order, beta ≈ 2 variance decay
 *   Fine/coarse coupling: same Brownian increments, fine steps summed for coarse
 *   Asian: running sum via trapezoidal rule plus a Brownian-bridge correction
 *
 * =================================================================== */
void nested_scalar_l(int l, int N, double *sums)
{
    // Grid level k and time-step counts
    const int k  = l / 2;
    const int nf = 1 << k;    // fine steps: 1,1,2,2,4,4,...
    const int nc = nf / 2;    // coarse steps: 0,0,1,1,2,2,...

    // Float model constants
    const float K_f   = 100.0f, T_f = 1.0f, r_f = 0.05f, sig_f = 0.2f;
    const float hf_f  = T_f / (float)nf;
    const float hc_f  = (nc > 0) ? T_f / (float)nc : 0.0f;
    const float disc_f = std::exp(-r_f * T_f);

    // Double model constants (same values, double precision arithmetic)
    const double K_d   = K_f, T_d = T_f, r_d = r_f, sig_d = sig_f;
    const double hf_d  = (double)hf_f;
    const double hc_d  = (nc > 0) ? T_d / (double)nc : 0.0;
    const double disc_d = std::exp(-r_d * T_d);

    // Adaptive mode: odd level at or above cut-off returns zero correction
    // (cost still counted, but correction = 0)
    if (adaptive_mode && l % 2 == 1 && k >= l_star) {
        sums[0] = (double)N * 2.0 * nf;
        for (int i = 1; i < 7; ++i) sums[i] = 0.0;
        return;
    }

    // Adaptive mode: above cut-off, even levels only (k >= l_star)
    // Pure double (fp64) Milstein MLMC correction — no fp32 involved
    if (adaptive_mode && k >= l_star && l % 2 == 0) {
        for (int np = 0; np < N; ++np) {
            double dP, Pfv;

            if (l == 0) {
                double dW  = sqrtf(hf_d) * next_normal();
                double dI  = sqrtf(hf_d / 12.0) * hf_d * next_normal();
                double Xf0 = K_d, vf = sig_d * Xf0;
                double Xf  = Xf0 + r_d*Xf0*hf_d + sig_d*Xf0*dW
                            + 0.5*sig_d*sig_d*Xf0*(dW*dW - hf_d);
                double Af  = 0.5*hf_d*Xf0 + 0.5*hf_d*Xf + vf*dI;
                double Pf  = std::fmax(0.0, Af-K_d);
                dP = disc_d * Pf; Pfv = dP;

            } else {
                double Xf=K_d, Xc=K_d;
                double Af=0.5*hf_d*K_d, Ac=0.5*hc_d*K_d;

                for (int n = 0; n < nc; ++n) {
                    double dW0  = sqrtf(hf_d)*next_normal(), dW1 = sqrtf(hf_d)*next_normal();
                    double dI0  = sqrtf(hf_d/12.0)*hf_d*next_normal();
                    double dI1  = sqrtf(hf_d/12.0)*hf_d*next_normal();
                    double dWc  = dW0+dW1, ddW = dW0-dW1;

                    double Xf0a=Xf, vf0=sig_d*Xf0a;
                    Xf  = Xf0a+r_d*Xf0a*hf_d+sig_d*Xf0a*dW0+0.5*sig_d*sig_d*Xf0a*(dW0*dW0-hf_d);
                    Af += hf_d*Xf+vf0*dI0;

                    double Xf0b=Xf, vf1=sig_d*Xf0b;
                    Xf  = Xf0b+r_d*Xf0b*hf_d+sig_d*Xf0b*dW1+0.5*sig_d*sig_d*Xf0b*(dW1*dW1-hf_d);
                    Af += hf_d*Xf+vf1*dI1;

                    double Xc0=Xc, vc=sig_d*Xc0;
                    Xc  = Xc0+r_d*Xc0*hc_d+sig_d*Xc0*dWc+0.5*sig_d*sig_d*Xc0*(dWc*dWc-hc_d);
                    Ac += hc_d*Xc+vc*(dI0+dI1+0.25*hc_d*ddW);
                }
                Af -= 0.5*hf_d*Xf; Ac -= 0.5*hc_d*Xc;
                double Pf=std::fmax(0.0, Af-K_d);
                double Pc=std::fmax(0.0, Ac-K_d);
                dP = disc_d*(Pf-Pc); Pfv = disc_d*Pf;
            }

            sums[0] += nf;
            sums[1] += dP;       sums[2] += dP*dP;
            sums[3] += dP*dP*dP; sums[4] += dP*dP*dP*dP;
            sums[5] += Pfv;      sums[6] += Pfv*Pfv;
        }
        return;
    }

    for (int i = 0; i < 7; ++i) sums[i] = 0.0;
    for (int np = 0; np < N; ++np) {

        // ================================================================
        // l = 0 (even, k=0): fp32 Milstein base level (1 fine step, no coarse)
        // Standard mode: compute Pf in fp32
        // Adaptive mode: same as standard (no cutoff effect at level 0)
        // ================================================================
        if (l == 0) {
            // All arithmetic in float (fp32)
            float dW   = sqrtf(hf_f) * next_normal();
            float dI   = sqrtf(hf_f / 12.0f) * hf_f * next_normal();

            float Xf0  = K_f;
            float vf   = sig_f * Xf0;
            // Milstein step: X += r*X*h + sigma*X*dW + 0.5*sigma^2*X*(dW^2 - h)
            float Xf   = Xf0 + r_f*Xf0*hf_f + sig_f*Xf0*dW
                             + 0.5f*sig_f*sig_f*Xf0*(dW*dW - hf_f);

            float Af   = 0.5f*hf_f*Xf0 + 0.5f*hf_f*Xf + vf*dI;

            float Pf   = std::fmaxf(0.0f, Af - K_f);
            double dP  = disc_f * Pf;

            sums[0] += nf;
            sums[1] += dP;       sums[2] += dP*dP;
            sums[3] += dP*dP*dP; sums[4] += dP*dP*dP*dP;
            sums[5] += dP;       sums[6] += dP*dP;

        // ================================================================
        // l = 1 (odd, k=0): precision correction (double - float) at base level
        // Standard mode: fp32 path vs fp64 path, same random draws
        // Adaptive mode: same as standard (no cutoff effect at level 1)
        // ================================================================
        } else if (l == 1) {
            // Draw randomness once (as float), reused for both fp32 and fp64 paths
            float dW_f  = sqrtf(hf_f) * next_normal();
            float dI_f  = sqrtf(hf_f / 12.0f) * hf_f * next_normal();

            // --- fp32 path: single-precision Milstein ---
            float Xf0_f = K_f;
            float vf_f  = sig_f * Xf0_f;
            float Xf_f  = Xf0_f + r_f*Xf0_f*hf_f + sig_f*Xf0_f*dW_f
                              + 0.5f*sig_f*sig_f*Xf0_f*(dW_f*dW_f - hf_f);

            float Af_f  = 0.5f*hf_f*Xf0_f + 0.5f*hf_f*Xf_f + vf_f*dI_f;

            float Pf_f  = std::fmaxf(0.0f, Af_f - K_f);
            double dP_f = disc_f * Pf_f;

            // --- fp64 path: same random draws, double-precision Milstein ---
            double dW_d  = dW_f;
            double dI_d  = dI_f;

            double Xf0_d = K_d;
            double vf_d  = sig_d * Xf0_d;
            double Xf_d  = Xf0_d + r_d*Xf0_d*hf_d + sig_d*Xf0_d*dW_d
                               + 0.5*sig_d*sig_d*Xf0_d*(dW_d*dW_d - hf_d);

            double Af_d  = 0.5*hf_d*Xf0_d + 0.5*hf_d*Xf_d + vf_d*dI_d;

            double Pf_d  = std::fmax(0.0, Af_d - K_d);
            double dP_d  = disc_d * Pf_d;

            // Precision correction
            double dP  = dP_d - dP_f;
            double Pf  = dP;

            sums[0] += 2.0 * nf;
            sums[1] += dP;       sums[2] += dP*dP;
            sums[3] += dP*dP*dP; sums[4] += dP*dP*dP*dP;
            sums[5] += Pf;       sums[6] += Pf*Pf;

        // ================================================================
        // l even >= 2: fp32 MLMC correction (Pf - Pc, nf fine vs nc coarse)
        // Standard mode: both fine/coarse in fp32
        // Adaptive mode (k >= l_star): switches to pure fp64 above cutoff
        // ================================================================
        } else if (l % 2 == 0) {
            // Fine/coarse states in float (fp32), except when adaptive_mode above l_star
            float Xf   = K_f, Xc   = K_f;
            float Af   = 0.5f*hf_f*K_f, Ac = 0.5f*hc_f*K_f;

            for (int n = 0; n < nc; ++n) {
                float dW0  = sqrtf(hf_f) * next_normal();
                float dW1  = sqrtf(hf_f) * next_normal();
                float dI0  = sqrtf(hf_f / 12.0f) * hf_f * next_normal();
                float dI1  = sqrtf(hf_f / 12.0f) * hf_f * next_normal();

                float dWc  = dW0 + dW1;
                float ddW  = dW0 - dW1;

                // Fine step 0
                float Xf0a = Xf;
                float vf0  = sig_f * Xf0a;
                Xf  = Xf0a + r_f*Xf0a*hf_f + sig_f*Xf0a*dW0
                           + 0.5f*sig_f*sig_f*Xf0a*(dW0*dW0 - hf_f);
                Af += hf_f * Xf + vf0 * dI0;

                // Fine step 1
                float Xf0b = Xf;
                float vf1  = sig_f * Xf0b;
                Xf  = Xf0b + r_f*Xf0b*hf_f + sig_f*Xf0b*dW1
                           + 0.5f*sig_f*sig_f*Xf0b*(dW1*dW1 - hf_f);
                Af += hf_f * Xf + vf1 * dI1;

                // Coarse step (driven by dWc = dW0 + dW1)
                float Xc0  = Xc;
                float vc   = sig_f * Xc0;
                Xc  = Xc0 + r_f*Xc0*hc_f + sig_f*Xc0*dWc
                           + 0.5f*sig_f*sig_f*Xc0*(dWc*dWc - hc_f);
                Ac += hc_f * Xc + vc * (dI0 + dI1 + 0.25f * hc_f * ddW);
            }
            // Trapezoidal endpoint correction for Asian
            Af -= 0.5f * hf_f * Xf;
            Ac -= 0.5f * hc_f * Xc;

            float Pf = std::fmaxf(0.0f, Af - K_f);
            float Pc = std::fmaxf(0.0f, Ac - K_f);

            double dP  = disc_f * (Pf - Pc);
            double Pfv = disc_f * Pf;

            sums[0] += nf;
            sums[1] += dP;       sums[2] += dP*dP;
            sums[3] += dP*dP*dP; sums[4] += dP*dP*dP*dP;
            sums[5] += Pfv;      sums[6] += Pfv*Pfv;

        // ================================================================
        // l odd >= 3: fp64 - fp32 precision correction at grid level k = l/2
        // Standard mode: both fine/coarse in fp64, correction = fp64(Pf-Pc) - fp32(Pf-Pc)
        // Adaptive mode (k >= l_star): zero correction (cost still counted)
        // ================================================================
        } else {
            // Standard mode: double (fp64) - float (fp32) precision correction
            // Float path state
            float Xf_f  = K_f, Xc_f  = K_f;
            float Af_f  = 0.5f*hf_f*K_f, Ac_f = 0.5f*hc_f*K_f;
            // Double path state
            double Xf_d = K_d, Xc_d  = K_d;
            double Af_d = 0.5*hf_d*K_d, Ac_d = 0.5*hc_d*K_d;

            for (int n = 0; n < nc; ++n) {
                // Draw all randoms as float (shared between float and double paths)
                float dW0_f  = sqrtf(hf_f) * next_normal();
                float dW1_f  = sqrtf(hf_f) * next_normal();
                float dI0_f  = sqrtf(hf_f / 12.0f) * hf_f * next_normal();
                float dI1_f  = sqrtf(hf_f / 12.0f) * hf_f * next_normal();

                // Cast same values to double for the double path
                double dW0_d  = dW0_f,  dW1_d  = dW1_f;
                double dI0_d  = dI0_f,  dI1_d  = dI1_f;

                float  dWc_f  = dW0_f + dW1_f, ddW_f = dW0_f - dW1_f;
                double dWc_d  = dW0_d + dW1_d, ddW_d = dW0_d - dW1_d;

                // ---- Float fine step 0 ----
                float  Xf0a_f = Xf_f;
                float  vf0_f  = sig_f * Xf0a_f;
                Xf_f  = Xf0a_f + r_f*Xf0a_f*hf_f + sig_f*Xf0a_f*dW0_f
                              + 0.5f*sig_f*sig_f*Xf0a_f*(dW0_f*dW0_f - hf_f);
                Af_f += hf_f * Xf_f + vf0_f * dI0_f;

                // ---- Double fine step 0 (same dW0) ----
                double Xf0a_d = Xf_d;
                double vf0_d  = sig_d * Xf0a_d;
                Xf_d  = Xf0a_d + r_d*Xf0a_d*hf_d + sig_d*Xf0a_d*dW0_d
                              + 0.5*sig_d*sig_d*Xf0a_d*(dW0_d*dW0_d - hf_d);
                Af_d += hf_d * Xf_d + vf0_d * dI0_d;

                // ---- Float fine step 1 ----
                float  Xf0b_f = Xf_f;
                float  vf1_f  = sig_f * Xf0b_f;
                Xf_f  = Xf0b_f + r_f*Xf0b_f*hf_f + sig_f*Xf0b_f*dW1_f
                              + 0.5f*sig_f*sig_f*Xf0b_f*(dW1_f*dW1_f - hf_f);
                Af_f += hf_f * Xf_f + vf1_f * dI1_f;

                // ---- Double fine step 1 (same dW1) ----
                double Xf0b_d = Xf_d;
                double vf1_d  = sig_d * Xf0b_d;
                Xf_d  = Xf0b_d + r_d*Xf0b_d*hf_d + sig_d*Xf0b_d*dW1_d
                              + 0.5*sig_d*sig_d*Xf0b_d*(dW1_d*dW1_d - hf_d);
                Af_d += hf_d * Xf_d + vf1_d * dI1_d;

                // ---- Float coarse step ----
                float  Xc0_f  = Xc_f;
                float  vc_f   = sig_f * Xc0_f;
                Xc_f  = Xc0_f + r_f*Xc0_f*hc_f + sig_f*Xc0_f*dWc_f
                              + 0.5f*sig_f*sig_f*Xc0_f*(dWc_f*dWc_f - hc_f);
                Ac_f += hc_f * Xc_f + vc_f * (dI0_f + dI1_f + 0.25f*hc_f*ddW_f);

                // ---- Double coarse step ----
                double Xc0_d  = Xc_d;
                double vc_d   = sig_d * Xc0_d;
                Xc_d  = Xc0_d + r_d*Xc0_d*hc_d + sig_d*Xc0_d*dWc_d
                              + 0.5*sig_d*sig_d*Xc0_d*(dWc_d*dWc_d - hc_d);
                Ac_d += hc_d * Xc_d + vc_d * (dI0_d + dI1_d + 0.25*hc_d*ddW_d);
            }

            // Trapezoidal endpoint corrections
            Af_f -= 0.5f*hf_f*Xf_f;   Ac_f -= 0.5f*hc_f*Xc_f;
            Af_d -= 0.5 *hf_d*Xf_d;   Ac_d -= 0.5 *hc_d*Xc_d;

            float  Pff = std::fmaxf(0.0f,Af_f-K_f);
            float  Pcf = std::fmaxf(0.0f,Ac_f-K_f);
            double Pfd = std::fmax(0.0, Af_d-K_d);
            double Pcd = std::fmax(0.0, Ac_d-K_d);

            double dP_f_val = disc_f * (Pff - Pcf);    // float correction
            double dP_d_val = disc_d * (Pfd - Pcd);    // double correction
            double dP       = dP_d_val - dP_f_val;     // precision correction

            double Pf_float = disc_f * Pff;
            double Pf_doub  = disc_d * Pfd;
            double Pfv      = Pf_doub - Pf_float;      // precision corr of fine Pf

            sums[0] += 2.0 * nf;
            sums[1] += dP;       sums[2] += dP*dP;
            sums[3] += dP*dP*dP; sums[4] += dP*dP*dP*dP;
            sums[5] += Pfv;      sums[6] += Pfv*Pfv;
        }
    }
}
