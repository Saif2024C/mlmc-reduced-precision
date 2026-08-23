/*
% Scalar standard MLMC tests (Milstein) for
%  - Asian call
%  - Lookback call
%
% This is the stripped-down version of the MCQMC06 test code,
% keeping only the scalar Asian and scalar lookback benchmarks.
%
% Notes:
% - Uses 2^l timesteps on level l.
% - Keeps the same MLMC / RNG structure as the original code.
% - Asian has no simple closed-form exact value, so val is left as NaN.
% - Lookback uses the closed-form exact value used in the original test.
*/

#include "../core/mlmc_test_100.cpp"   // master file for 100 tests
#include "../core/mlmc_rng.cpp"        // RNG functions

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

//
// option = 1 -> Asian call
// option = 2 -> Lookback call
//
int option;

void mcqmc06_scalar_l(int, int, double *);

static inline float ncff(float x)
{
    return 0.5f + 0.5f * erff(std::sqrt(0.5f) * x);
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    int N0   = 200;  // initial samples on each level
    int Lmin = 2;    // minimum refinement level
    int Lmax = 10;   // maximum refinement level

    int N, L;
    float Eps[11];
    char filename[32];
    FILE *fp;

#ifdef _OPENMP
    double wtime = omp_get_wtime();
#endif

    for (option = 1; option <= 2; ++option) {
        // initialise generator, with separate storage for each thread
        // when compiled for OpenMP
#pragma omp parallel
        rng_initialisation();

        std::sprintf(filename, "mcqmc06_scalar_%d.txt", option);
        fp = std::fopen(filename, "w");
        if (!fp) {
            std::perror("fopen");
            return EXIT_FAILURE;
        }

        if (option == 1) {
            std::printf("\n ---- Asian call ----\n");
            N = 20000;
            L = 8;
            float Eps2[] = { 0.005f, 0.01f, 0.02f, 0.05f, 0.1f, 0.0f };
            std::memcpy(Eps, Eps2, sizeof(Eps2));
        }
        else {
            std::printf("\n ---- Lookback call ----\n");
            N = 20000;
            L = 10;
            float Eps2[] = { 0.005f, 0.01f, 0.02f, 0.05f, 0.1f, 0.0f };
            std::memcpy(Eps, Eps2, sizeof(Eps2));
        }

        // original (supervisor's reference code):
        // mlmc_test_100(mcqmc06_scalar_l, NAN, N0, Eps, Lmin, Lmax, fp);
        // changed to mlmc_test so this file produces the convergence + complexity
        // tables (alpha/beta/gamma) that mlmc_plot_python.py expects:
        mlmc_test(mcqmc06_scalar_l, N, L, N0, Eps, Lmin, Lmax, fp);
        std::fclose(fp);

#ifdef _OPENMP
        std::printf(" execution time = %f s\n", omp_get_wtime() - wtime);
        wtime = omp_get_wtime();
#endif

#pragma omp parallel
        rng_termination();

        // Exact reference value for lookback; Asian is left as NaN.
        float T, r, sig, sig2, K, val, k, d1, d2;
        T = 1.0f;
        r = 0.05f;
        sig = 0.2f;
        sig2 = sig * sig;
        K = 100.0f;

        if (option == 1) {
            val = nanf("");
        }
        else {
            d1 = (r + 0.5f * sig2) * T / (sig * std::sqrt(T));
            d2 = (r - 0.5f * sig2) * T / (sig * std::sqrt(T));
            k  = 0.5f * sig2 / r;
            val = K * (ncff(d1) - ncff(-d1) * k - std::exp(-r * T) * (ncff(d2) - ncff(d2) * k));
        }

        // now do 100 MLMC calcs
#pragma omp parallel
        rng_initialisation();

        std::sprintf(filename, "mcqmc06_scalar_%d_100.txt", option);
        fp = std::fopen(filename, "w");
        if (!fp) {
            std::perror("fopen");
            return EXIT_FAILURE;
        }

        mlmc_test_100(mcqmc06_scalar_l, val, N0, Eps, Lmin, Lmax, fp);
        std::fclose(fp);

#ifdef _OPENMP
        std::printf(" execution time = %f s\n", omp_get_wtime() - wtime);
        wtime = omp_get_wtime();
#endif

#pragma omp parallel
        rng_termination();
    }

    return EXIT_SUCCESS;
}

/*-------------------------------------------------------
%
% level l estimator
%
*/
void mcqmc06_scalar_l(int l, int N, double *sums)
{
    // model parameters
    const float K   = 100.0f;
    const float T   = 1.0f;
    const float r   = 0.05f;
    const float sig = 0.2f;
    const float B   = 0.85f * K;

    const int nf = 1 << l;
    const int nc = nf / 2;

    const float hf = T / static_cast<float>(nf);
    const float hc = T / static_cast<float>(nc);

    for (int k = 0; k < 7; ++k) sums[k] = 0.0;

#pragma omp parallel for reduction(+:sums[0:7])
    for (int np = 0; np < N; ++np) {
        float X0, Xf, Xc, Af, Ac, Mf, Mc, Bf, Bc;
        float Xf0, Xc0, Xc1, vf, vc, dWc, ddW, Pf, Pc, dP;
        float dWf[2], dIf[2], Lf[2];

        X0 = K;
        Xf = X0;
        Xc = Xf;

        Af = 0.5f * hf * Xf;
        Ac = 0.5f * hc * Xc;

        Mf = Xf;
        Mc = Xc;

        Bf = 1.0f;
        Bc = 1.0f;

        if (l == 0) {
            dWf[0] = std::sqrt(hf) * next_normal();
            Lf[0]  = - next_exponential();
            dIf[0] = std::sqrt(hf / 12.0f) * hf * next_normal();

            Xf0 = Xf;
            Xf  = Xf + r * Xf * hf + sig * Xf * dWf[0]
                     + 0.5f * sig * sig * Xf * (dWf[0] * dWf[0] - hf);
            vf  = sig * Xf0;
            Af  = Af + 0.5f * hf * Xf + vf * dIf[0];
            Mf  = fminf(Mf,
                        0.5f * (Xf0 + Xf - std::sqrt((Xf - Xf0) * (Xf - Xf0) - 2.0f * hf * vf * vf * Lf[0])));
            Bf  = Bf * (1.0f - std::exp(-2.0f * fmaxf(0.0f, (Xf0 - B) * (Xf - B) / (hf * vf * vf))));
        }
        else {
            for (int n = 0; n < nc; ++n) {
                dWf[0] = std::sqrt(hf) * next_normal();
                dWf[1] = std::sqrt(hf) * next_normal();
                Lf[0]  = - next_exponential();
                Lf[1]  = - next_exponential();
                dIf[0] = std::sqrt(hf / 12.0f) * hf * next_normal();
                dIf[1] = std::sqrt(hf / 12.0f) * hf * next_normal();

                for (int m = 0; m < 2; ++m) {
                    Xf0 = Xf;
                    Xf  = Xf + r * Xf * hf + sig * Xf * dWf[m]
                             + 0.5f * sig * sig * Xf * (dWf[m] * dWf[m] - hf);
                    vf  = sig * Xf0;
                    Af  = Af + hf * Xf + vf * dIf[m];
                    Mf  = fminf(Mf,
                                0.5f * (Xf0 + Xf - std::sqrt((Xf - Xf0) * (Xf - Xf0) - 2.0f * hf * vf * vf * Lf[m])));
                    Bf  = Bf * (1.0f - std::exp(-2.0f * fmaxf(0.0f, (Xf0 - B) * (Xf - B) / (hf * vf * vf))));
                }

                dWc = dWf[0] + dWf[1];
                ddW = dWf[0] - dWf[1];

                Xc0 = Xc;
                Xc  = Xc + r * Xc * hc + sig * Xc * dWc + 0.5f * sig * sig * Xc * (dWc * dWc - hc);

                vc  = sig * Xc0;
                Ac  = Ac + hc * Xc + vc * (dIf[0] + dIf[1] + 0.25f * hc * ddW);
                Xc1 = 0.5f * (Xc0 + Xc + vc * ddW);
                Mc  = fminf(Mc,
                            0.5f * (Xc0 + Xc1 - std::sqrt((Xc1 - Xc0) * (Xc1 - Xc0) - 2.0f * hf * vc * vc * Lf[0])));
                Mc  = fminf(Mc,
                            0.5f * (Xc1 + Xc  - std::sqrt((Xc  - Xc1) * (Xc  - Xc1) - 2.0f * hf * vc * vc * Lf[1])));
                Bc  = Bc * (1.0f - std::exp(-2.0f * fmaxf(0.0f, (Xc0 - B) * (Xc1 - B) / (hf * vc * vc))));
                Bc  = Bc * (1.0f - std::exp(-2.0f * fmaxf(0.0f, (Xc1 - B) * (Xc  - B) / (hf * vc * vc))));
            }
            Af = Af - 0.5f * hf * Xf;
            Ac = Ac - 0.5f * hc * Xc;
        }

        // Here we keep only the scalar Asian and scalar lookback payoffs.
        if (option == 1) {
            Pf = fmaxf(0.0f, Af - K);
            Pc = fmaxf(0.0f, Ac - K);
        }
        else {
            Pf = Xf - Mf;
            Pc = Xc - Mc;
        }

        dP = std::exp(-r * T) * (Pf - Pc);
        Pf = std::exp(-r * T) * Pf;

        if (l == 0) dP = Pf;

        sums[0] += nf;           // cost proxy: number of fine timesteps
        sums[1] += dP;
        sums[2] += dP * dP;
        sums[3] += dP * dP * dP;
        sums[4] += dP * dP * dP * dP;
        sums[5] += Pf;
        sums[6] += Pf * Pf;
    }
}
