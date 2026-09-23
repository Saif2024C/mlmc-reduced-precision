/*
 * std_mlmc_fp32_scalar_avx512.cpp -- STANDARD (non-nested) fp32 MLMC reference,
 * scalar Asian, AVX-512.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * A nested ladder run entirely in fp32 is not a meaningful reference: with a
 * single working precision there is nothing for its odd super-levels to
 * correct, so they are structurally present but numerically zero.  The
 * reference for a nested mixed-precision scheme is ordinary MLMC in fp32 --
 * one correction Y_k = P_k - P_{k-1} per grid level, no super-levels -- and
 * that is what this file runs.
 *
 * A genuine fp32 baseline is ordinary MLMC: one correction per grid level,
 * Y_k = P^{f32}_k - P^{f32}_{k-1}, no super-levels.  That is what this file
 * runs.  It reuses the nested estimator's own level function so the arithmetic,
 * the Philox stream and the payoff are identical -- the only difference is the
 * ladder.  Grid level k is the nested file's even super-level 2k, so the
 * wrapper below is just a level remap with adaptive_mode/l_star forced to make
 * the pure-fp32 branch fire.
 *
 * Build (mimic):
 *   source /opt/intel/oneapi/setvars.sh
 *   icpx -O3 -march=native -std=c++17 \
 *        --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13 \
 *        std_mlmc_fp32_scalar_avx512.cpp -o std_fp32_scalar
 */

// The nested estimator drags in nested_mlmc_test.cpp, which defines mlmc_test,
// mlmc and regression under the SAME names as the standard harness.  Both are
// needed here -- the nested file for its level function, the standard harness
// to drive it -- so the standard harness is renamed on the way in.  Its own
// main() is suppressed too: this file provides one.
#define mlmc_test  std_mlmc_test
#define mlmc       std_mlmc
#define regression std_regression
#include "../core/mlmc_test.cpp"
#undef mlmc_test
#undef mlmc
#undef regression

#define main nested_scalar_main_unused
#include "nested_scalar_milstein_fp16_avx512.cpp"
#undef main

// ---------------------------------------------------------------------------
// Standard MLMC level function.
//
// mlmc_test hands us grid level k; the nested estimator indexes super-levels,
// where even super-level l=2k is exactly the fp32 Milstein correction
// P_k - P_{k-1} at grid level k.  Forcing adaptive_mode with l_star=0 selects
// the pure-fp32 branch at every level, so no fp16 arithmetic is involved.
// Cost comes back as n_f fine timesteps, the same unit the nested runs report.
// ---------------------------------------------------------------------------
static void std_fp32_scalar_l(int k, int N, double *sums)
{
    adaptive_mode = true;
    kahan_mode    = false;
    l_star        = 0;
    nested_scalar_fp16_avx_l(2 * k, N, sums);
}

int main(int argc, char **argv)
{
    static const float EPS[] = { 0.005f, 0.01f, 0.02f, 0.05f, 0.1f, 0.0f };

    int opt_lo = 1, opt_hi = 1;   // thesis uses the Asian payoff only
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--option") == 0 && i + 1 < argc)
            opt_lo = opt_hi = std::atoi(argv[++i]);

    char filename[160];
    for (option = opt_lo; option <= opt_hi; ++option) {
        std::sprintf(filename, "std_mlmc_fp32_scalar_%d.txt", option);
        FILE *fp = std::fopen(filename, "w");
        if (!fp) { std::perror("fopen"); return EXIT_FAILURE; }

        std::printf("\n ---- standard fp32 MLMC, scalar Asian (AVX-512) ----\n");

        float e[16];
        int n = 0;
        while (EPS[n] != 0.0f && n < 15) { e[n] = EPS[n]; ++n; }
        e[n] = 0.0f;

        // N, L for the convergence table; Lmin/Lmax for the complexity test.
        std_mlmc_test(std_fp32_scalar_l, 20000, 10, 200, e, 2, 10, fp);
        std::fclose(fp);
    }
    return EXIT_SUCCESS;
}
