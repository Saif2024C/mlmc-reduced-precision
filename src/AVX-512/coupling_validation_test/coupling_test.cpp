////////////////////////////////////////////////////////////////////////
// Verifies the fp32/fp16 coupling in avx512_rng.h.
//
// Both precisions must read the SAME word of a Philox draw -- fp32 its top
// 23 bits, fp16 its top 16 -- so that i16 == i32 >> 7.  If that fails, the
// odd-level "precision correction" is a difference of two INDEPENDENT
// Normals (O(1) variance, no decay with level) rather than fp16 error.
//
// Three inverse-CDFs are in play, and the checks play them off each other:
//
//   cdfnorminv           double, scalar, expensive   test 4's reference
//   _mm512_cdfnorminv_ps fp32, 16 lanes, SVML        the fp32 path
//   norminv_fp16         fp16, 32 lanes, spline      the fp16 path
//
// Tests 1-3 compare the two vector paths against each other, so they would
// not notice both drifting together; test 4 pins the cheap spline to the
// expensive double reference.
//
// Build and run on mimic (needs AVX-512 FP16 hardware and icpx; setvars.sh
// puts icpx on PATH and must be sourced first):

/*
source /opt/intel/oneapi/setvars.sh
icpx -O2 -march=native -std=c++17 --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13 coupling_test.cpp -o coupling_test
*/

//  ./coupling_test              # the four checks
//  ./coupling_test --selftest   # ...and verify they reject known-bad inputs
// Exits non-zero on any failure, so it works as a CI gate.
////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <mathimf.h>     // cdfnorminv, for test 4's double-precision reference
#include <immintrin.h>

#undef USE_MKL_RNG
#include "../avx512_rng.h"

static const int NDRAW = 4096;

typedef __m256h (*fp16_fn)(__m512i);
typedef void    (*coupled_fn)(avx_rng &, __m512 &, __m256h &);

// ===========================================================================
// Known-bad inputs -- --selftest aims the checks at these and requires that
// they be rejected.  A check never seen to fail proves nothing.
// ===========================================================================

// Input 1: fp16 from the LOW 16 bits (even lanes).  Shares almost nothing
// with u32, so the correction degenerates to independent Normals.
static __m256h bad_fp16_low_bits(__m512i bits) {
    const __m512i even_idx = _mm512_set_epi16(
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                30,28,26,24,22,20,18,16,14,12,10,8,6,4,2,0);
    __m512i g = _mm512_permutexvar_epi16(even_idx,
                    _mm512_castph_si512(norminv_fp16(bits)));
    return _mm512_castph512_ph256(_mm512_castsi512_ph(g));
}

// Input 2: lanes 0 and 1 read each other's word; the other 14 stay correct.
static __m256h bad_fp16_swapped_lanes(__m512i bits) {
    const __m512i idx = _mm512_set_epi16(
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                31,29,27,25,23,21,19,17,15,13,11,9,7,5,1,3);  // last two swapped
    __m512i g = _mm512_permutexvar_epi16(idx,
                    _mm512_castph_si512(norminv_fp16(bits)));
    return _mm512_castph512_ph256(_mm512_castsi512_ph(g));
}

// Input 3: fp16 from its own draw.  Both streams are individually valid,
// which is what hides it from any single-stream check.
static void bad_coupled_independent(avx_rng &g, __m512 &z32, __m256h &z16) {
    z32 = normal16_fp32_bits(philox_next(g));
    z16 = normal16_fp16_aligned(philox_next(g));
}

// ===========================================================================
// Checks.  Each returns a failure count, printing detail only when verbose.
// ===========================================================================

// Test 1: the uniforms nest.  Depends only on the shift constants, so it
// isolates "are they nested at all" from any transform or lane question.
static int test1_nesting(bool verbose) {
    int bad = 0;
    avx_rng h; rng_seed(h, 0);
    for (int d = 0; d < NDRAW; ++d) {
        alignas(64) uint32_t w[16];
        _mm512_store_si512((void *)w, philox_next(h));
        for (int j = 0; j < 16; ++j) {
            uint32_t i32 = w[j] >> 9;    // fp32: top 23 bits
            uint32_t i16 = w[j] >> 16;   // fp16: top 16 bits
            if (i16 != (i32 >> 7)) {
                if (verbose && bad < 5)
                    printf("  FAIL: draw %d lane %d: w=%08x i32>>7=%u i16=%u\n",
                           d, j, w[j], i32 >> 7, i16);
                ++bad;
            }
        }
    }
    if (verbose) printf("  checked %d lane-values\n", NDRAW * 16);
    return bad;
}

// Test 2: fp16 lane j derives from the same word fp32 lane j reads.  The only
// check that localises a permutation fault -- test 1 passes with a wrong
// odd_idx, test 3 only says "decoupled".  Case 0 is crafted (distinct top and
// low halves per lane, catching an off-by-one that random data can mask).
static int test2_lane_alignment(fp16_fn f16, bool verbose) {
    int bad = 0;
    const int NCASE = 256;
    avx_rng h; rng_seed(h, 11);

    for (int c = 0; c < NCASE; ++c) {
        alignas(64) uint32_t w[16];
        if (c == 0)
            for (int j = 0; j < 16; ++j)
                w[j] = ((uint32_t)(0x1234 + j * 0x0871) << 16)
                     | (uint32_t)(0xFFFF - j);
        else
            _mm512_store_si512((void *)w, philox_next(h));

        alignas(32) _Float16 got[16];
        _mm256_store_ph(got, f16(_mm512_load_si512((const void *)w)));

        // reference: splat word j's top half across all lanes, transform,
        // then read any lane back
        for (int j = 0; j < 16; ++j) {
            uint16_t top = (uint16_t)(w[j] >> 16);
            alignas(64) _Float16 ref[32];
            _mm512_store_ph(ref, norminv_fp16(_mm512_set1_epi16((short)top)));
            if ((float)got[j] != (float)ref[0]) {
                if (verbose && bad < 5)
                    printf("  FAIL: case %d lane %d: top=%04x got=%g want=%g\n",
                           c, j, top, (float)got[j], (float)ref[0]);
                ++bad;
            }
        }
    }
    if (verbose)
        printf("  checked %d lane-values (1 crafted + %d random cases)\n",
               NCASE * 16, NCASE - 1);
    return bad;
}

// Test 3: coupled draws are close, not independent.  Independent Normals
// differ with variance 2 (Var[X-Y] = 1+1-0); a real pair differs only by
// quantisation.  It rises ~0.125 per decoupled lane (healthy 5e-6, 1 lane
// 0.125, all 16 2.003), so 0.01 catches even a single bad lane.
//
// Tested on the raw second moment E[X^2] = Var[X] + E[X]^2 rather than the
// variance (Giles' suggestion), so one threshold bounds the spread AND a
// systematic offset: a nonzero mean can only inflate E[X^2].  Adding a
// constant to every fp16 value leaves Var[z32-z16] untouched but shifts
// E[X^2] -- a +0.001 offset lifts it by 20%.
static int test3_closeness(coupled_fn cpl, bool verbose) {
    avx_rng h; rng_seed(h, 7);
    double s = 0.0, s2 = 0.0;
    for (int d = 0; d < NDRAW; ++d) {
        __m512  z32;
        __m256h z16;
        cpl(h, z32, z16);
        alignas(64) float    a32[16];
        alignas(32) _Float16 a16[16];
        _mm512_store_ps(a32, z32);
        _mm256_store_ph(a16, z16);
        for (int j = 0; j < 16; ++j) {
            double diff = (double)a32[j] - (double)(float)a16[j];
            s += diff; s2 += diff * diff;
        }
    }
    double n    = NDRAW * 16.0;
    double mean = s / n;
    double m2   = s2 / n;            // E[X^2] -- bounds variance and mean at once

    if (verbose) {
        printf("  E[(z32-z16)^2] = %.6e   (independent Normals give ~2.0)\n", m2);
        printf("  mean           = %+.3e  (reported, bounded via E[X^2])\n", mean);
    }
    if (m2 < 0.01) return 0;

    if (verbose)
        printf("  FAIL: E[X^2] %.6e exceeds 0.01 "
               "(healthy ~5e-6, one decoupled lane ~0.125)\n", m2);
    return 1;
}

// Test 4: the spline is accurate, swept over all 65536 possible inputs.
//
// Absolute error only -- norminv crosses zero at u=1/2, so relative error is
// unbounded there however correct the value (measured 77% at i=32769).  Bulk
// and tail are reported apart because a piecewise-linear spline cannot track
// an unbounded function; one merged bound would be loose enough to hide a
// bulk regression.
//
// Symmetry is checked PAIRWISE rather than as an aggregate mean: the spline
// folds about u=1/2, so norminv_fp16(i) must be exactly -norminv_fp16(65535-i)
// -- an exact target, and per-pair it cannot hide two faults cancelling.
//
// Measured: 32767 of 32768 pairs cancel bit-exactly.  The lone exception is
// i=32768, and the cause is in my_mm512.h's fold predicate, not here:
// norminv_fp16 uses _mm512_cmplt_epu16_mask(2^15, i), a STRICT compare, so
// i=32768 alone is neither folded nor sign-flipped and is then read as a
// signed int16 (= -32768).  Giles' scalar reference (normal_dyadic2 in
// normal_test.cpp) uses i >= (1<<15) and is symmetric on all 32768 pairs;
// swapping cmplt for cmple in a local probe copy takes the AVX count to 0.
//
// Left alone deliberately.  my_mm512.h is Giles' file, and at that input the
// shipped behaviour is the MORE accurate of the two: error 1.9e-5 against
// 3.7e-4 for the symmetric version, since the true value (+1.9e-5) is far
// below what fp16 resolves there.  One input in 65536, 280x below the
// spline's own 5.4e-3 bulk error -- no measurable effect on the estimators.
static int test4_norminv_accuracy(bool verbose) {
    double max_bulk = 0.0, max_tail = 0.0;
    int    worst_bulk = 0, worst_tail = 0;

    // lanes are independent, so a splatted value hits the same code path
    for (int i = 0; i < 65536; ++i) {
        alignas(64) _Float16 got[32];
        _mm512_store_ph(got, norminv_fp16(_mm512_set1_epi16((short)i)));

        // interval [i, i+1)/65536 at its midpoint -- uniform16_fp32's convention
        double u   = (i + 0.5) / 65536.0;
        double err = std::fabs((double)(float)got[0] - cdfnorminv(u));

        if (u > 3.2e-5 && u < 1.0 - 3.2e-5) {   // |z| < 4, ~99.994% of the mass
            if (err > max_bulk) { max_bulk = err; worst_bulk = i; }
        } else {
            if (err > max_tail) { max_tail = err; worst_tail = i; }
        }
    }

    int nasym = 0;
    for (int i = 0; i < 32768; ++i) {
        alignas(64) _Float16 a[32], b[32];
        _mm512_store_ph(a, norminv_fp16(_mm512_set1_epi16((short)i)));
        _mm512_store_ph(b, norminv_fp16(_mm512_set1_epi16((short)(65535 - i))));
        if ((double)(float)a[0] + (double)(float)b[0] != 0.0) ++nasym;
    }

    if (verbose) {
        printf("  max abs error = %.5f  bulk |z|<4   (at i=%d)\n",
               max_bulk, worst_bulk);
        printf("  max abs error = %.5f  extreme tail (at i=%d)\n",
               max_tail, worst_tail);
        printf("  asymmetric pairs = %d of 32768  (0 expected)\n", nasym);
    }
    // measured 0.0054 bulk / 0.0618 tail; fp16 itself resolves only ~1e-3, so
    // these sit clear of noise yet catch a bad table entry.
    // Symmetry must be exact: norminv_fp16 folds i >= 2^15 and negates, so
    // f(i) + f(65535-i) is 0 by construction.  This was 1 of 32768 until
    // 2026-08-03, when the fold's strict cmplt_epu16 (which skipped i=32768,
    // leaving it to be read as signed -32768) became cmpge_epu16 in my_mm512.h.
    if (max_bulk < 0.01 && max_tail < 0.10 && nasym == 0) return 0;

    if (verbose)
        printf("  FAIL: bulk %.5f (limit 0.01), tail %.5f (limit 0.10), "
               "asymmetric pairs %d (limit 0)\n", max_bulk, max_tail, nasym);
    return 1;
}

// ===========================================================================
// --selftest: verifies that the lane-alignment and statistical-coupling
// checks reject representative known-bad implementations.
// ===========================================================================
static int selftest() {
    struct Row {
        const char *name;
        fp16_fn     f16;   // NULL => exercised via test 3 instead
        coupled_fn  cpl;
    };
    const Row rows[] = {
        {"fp16 from LOW 16 bits",   bad_fp16_low_bits,      NULL},
        {"two lanes swapped",       bad_fp16_swapped_lanes, NULL},
        {"fp16 from separate draw", NULL, bad_coupled_independent},
    };

    printf("Self-test: known-bad inputs must be rejected\n");
    int missed = 0;
    for (const Row &r : rows) {
        bool caught = r.f16 ? test2_lane_alignment(r.f16, false) > 0
                            : test3_closeness(r.cpl, false) > 0;
        printf("  %-26s %s\n", r.name, caught ? "rejected" : "MISSED !");
        if (!caught) ++missed;
    }
    return missed;
}

int main(int argc, char **argv) {
    int failures = 0;

    printf("Test 1: uniform nesting (i16 == i32 >> 7)\n");
    failures += test1_nesting(true);

    printf("\nTest 2: lane alignment (fp16 lane j <- word j top 16 bits)\n");
    failures += test2_lane_alignment(normal16_fp16_aligned, true);

    printf("\nTest 3: coupled draws are close, not independent\n");
    failures += test3_closeness(normal16_coupled, true);

    printf("\nTest 4: norminv_fp16 accuracy vs cdfnorminv (all 65536 inputs)\n");
    failures += test4_norminv_accuracy(true);

    printf("\n%s\n", failures ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED");

    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--selftest")) {
            printf("\n");
            failures += selftest();
        }

    return failures != 0;
}
