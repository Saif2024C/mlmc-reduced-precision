#ifndef MODES_MM512_H
#define MODES_MM512_H
////////////////////////////////////////////////////////////////////////////
// modes_mm512.h -- the three-mode precision sweep shared by the AVX-512 fp16
// estimators, and the runtime switches those modes set.
//
// SEPARATE FROM options_mm512.h ON PURPOSE.  options_mm512.h is the reusable
// numerics library -- Milstein steps, bridges, payoffs, Kahan summation --
// and a new payoff can be written against it alone.  This header is the
// experiment scaffolding built on top: the nokahan/kahan/adaptive
// comparison, its file naming, and its Eps[] policy.  Include it only if you
// want that specific three-mode study; ignore it entirely otherwise and call
// mlmc_test() yourself.
//
// ---------------------------------------------------------------------------
// USING IT FOR A NEW PAYOFF
// ---------------------------------------------------------------------------
//   0. Include it AFTER the test harness -- run_sweep() calls mlmc_test():
//
//          #include "../core/nested_mlmc_test.cpp"
//          #include "options_mm512.h"
//          #include "modes_mm512.h"
//
//      The other order is a hard #error, not a silent failure.
//
//   1. Write a level estimator with the usual MLMC signature:
//          void my_l(int l, int N, double *sums);
//      It reads the switches below (kahan_mode / adaptive_mode / l_star) and
//      the `option` global to decide what to compute.
//   2. Fill in a SweepConfig and call run_sweep():
//
//          SweepConfig cfg;
//          cfg.prefix       = "nested_myopt_fp16_avx512";  // output filename stem
//          cfg.family       = "myopt";                     // shown in the banner
//          cfg.opt_name[0]  = "Call";                      // option 1
//          cfg.opt_name[1]  = "Put";                       // option 2
//          cfg.estimator    = my_l;
//          cfg.l_star[1]    = 5;   cfg.l_star[2] = 6;      // per-option cutoff
//          cfg.eps          = my_eps;                      // 0-terminated
//          return run_sweep(cfg, argc, argv);
//
//      Writes <prefix>_<mode>_<option>.txt for each of the 3 modes x 2 options.
//
// ---------------------------------------------------------------------------
// THE THREE MODES
// ---------------------------------------------------------------------------
//   nokahan  : plain += accumulation.  Convergence table only -- beta is
//              often non-positive here, and mlmc_test's adaptive complexity
//              sampler never returns against a non-positive beta, so the
//              Eps[] array is replaced by a zero terminator.
//   kahan    : Kahan-compensated accumulation.  beta positive, so the real
//              complexity test runs against the caller's Eps[].
//   adaptive : Kahan below grid level l_star, pure fp32 at/above it.  Above
//              the cutoff the odd super-levels carry no precision correction
//              and are not run, so the scheme is standard non-nested MLMC
//              there.
//
// The pure-fp32 reference is NOT a mode here.  It used to be run as `adaptive`
// with l_star=0, which put the whole nested ladder in fp32 -- but a nested
// ladder with a single working precision has nothing for its odd super-levels
// to correct, so that was never the right reference.  The reference is
// ordinary MLMC in fp32, built by std_mlmc_fp32_{scalar,basket}_avx512.cpp,
// which reuses the estimators below (and so the same Philox stream and the
// same payoff) on a single-correction-per-grid-level ladder.
////////////////////////////////////////////////////////////////////////////

// run_sweep() calls mlmc_test(), so this header must be included AFTER
// ../core/nested_mlmc_test.cpp.  The wrong order would otherwise fail with a
// confusing "mlmc_test was not declared" from deep inside run_sweep.  PRINTF2
// is defined by that harness and by nothing else, so its absence means the
// harness has not been included yet.
#ifndef PRINTF2
#error "modes_mm512.h must be included AFTER ../core/nested_mlmc_test.cpp -- it calls mlmc_test(). Move the #include below it."
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Runtime switches.  The level estimator reads these; run_sweep() sets them.
// ---------------------------------------------------------------------------

// 1 or 2 -- which payoff of the family.  Defined by the estimator's file.
extern int option;

// Kahan compensation on the running state (path position, running sums).
static bool kahan_mode = false;

// At/above grid level l_star, even levels run the pure-fp32 chain instead of
// fp16, and odd levels are not run at all: with the path in fp32 throughout
// there is no precision correction to make, so the scheme is standard
// non-nested MLMC above the cutoff.  Below l_star the fp16 chain is unchanged.
// Only ever paired with kahan_mode = true.
static bool adaptive_mode = false;
static int  l_star = 0;

// ---------------------------------------------------------------------------
// Sweep configuration
// ---------------------------------------------------------------------------
struct SweepConfig {
    const char *prefix;        // output filename stem
    const char *family;        // "scalar" / "basket" / ... , for the banner
    const char *opt_name[2];   // display names for option 1 and 2
    void (*estimator)(int, int, double *);
    const float *eps;          // 0-terminated, used by the kahan/adaptive modes
    int   l_star[3];           // indexed by option (1,2); [0] unused
    void (*setup)();           // optional one-off init (e.g. Cholesky), may be null

    int N    = 20000;          // samples for the convergence table
    int L    = 10;             // levels for the convergence table
    int N0   = 200;            // initial samples per level in the complexity test
    int Lmin = 2;
    int Lmax = 20;

    SweepConfig() : prefix(""), family(""), estimator(0), eps(0), setup(0) {
        opt_name[0] = "1"; opt_name[1] = "2";
        l_star[0] = l_star[1] = l_star[2] = 0;
    }
};

// ---------------------------------------------------------------------------
// run_sweep: 3 modes x 2 options (or one option, with --option N).
// Returns EXIT_SUCCESS / EXIT_FAILURE, so main() can `return run_sweep(...)`.
// ---------------------------------------------------------------------------
static inline int run_sweep(const SweepConfig &cfg, int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    if (cfg.setup) cfg.setup();

    // --option N restricts the run to a single payoff.
    int opt_lo = 1, opt_hi = 2;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--option") == 0 && i + 1 < argc)
            opt_lo = opt_hi = std::atoi(argv[++i]);

    const bool  kahan_modes[3]   = { false, true, true };
    const bool  adapt_modes[3]   = { false, false, true };
    const char *mode_tag[3]      = { "nokahan", "kahan", "adaptive" };

    char filename[160];

    for (int m = 0; m < 3; ++m) {
        kahan_mode    = kahan_modes[m];
        adaptive_mode = adapt_modes[m];

        for (option = opt_lo; option <= opt_hi; ++option) {
            l_star = cfg.l_star[option];

            std::sprintf(filename, "%s_%s_%d.txt", cfg.prefix, mode_tag[m], option);
            FILE *fp = std::fopen(filename, "w");
            if (!fp) { std::perror("fopen"); return EXIT_FAILURE; }

            std::printf("\n ---- AVX-512 nested %s fp16 %s (pure-half Milstein, %s%s) ----\n",
                        cfg.family, cfg.opt_name[option - 1],
                        kahan_mode ? "Kahan ON" : "Kahan OFF",
                        adaptive_mode ? ", Adaptive l_star" : "");
            if (adaptive_mode) std::printf("      l_star = %d\n", l_star);

            if (kahan_mode) {
                // beta genuinely positive -> the adaptive complexity sampler
                // converges, so run the real complexity test.
                float e[16];
                int n = 0;
                while (cfg.eps && cfg.eps[n] != 0.0f && n < 15) { e[n] = cfg.eps[n]; ++n; }
                e[n] = 0.0f;
                mlmc_test(cfg.estimator, cfg.N, cfg.L, cfg.N0, e,
                          cfg.Lmin, cfg.Lmax, fp);
            } else {
                // beta often non-positive -> zero-terminated Eps skips the
                // complexity loop, leaving the convergence table only.
                float e[1] = { 0.0f };
                mlmc_test(cfg.estimator, cfg.N, cfg.L, cfg.N0, e,
                          cfg.Lmin, cfg.Lmax, fp);
            }
            std::fclose(fp);
        }
    }
    return EXIT_SUCCESS;
}

#endif // MODES_MM512_H
