/*
 * avx512_rng.h — vectorised random-number generation for the AVX-512 fp16
 * nested-MLMC estimators.
 *
 * Philox-4x32-10 (counter-based, in-register, no memory traffic) is the RNG.
 * Each philox_next() call gives one 512-bit __m512i = 16 independent 32-bit
 * words, one per SIMD lane -- each word is treated as one random interval.
 *
 * fp32 and fp16 draw from the SAME 16 words, both taking the TOP bits of
 * each word, so the two uniforms are nested rather than independent:
 *
 *   fp32:  i32 = word >> 9   (bits 31..9, 23 bits)   u32 = (i32+0.5)*2^-23
 *   fp16:  i16 = word >> 16  (bits 31..16, 16 bits)  u16 = (i16+0.5)*2^-16
 *
 * Since i16 == i32 >> 7, u16 IS u32 truncated to 16 bits -- the fp16 uniform
 * is the fp32 uniform's own leading bits, which is what makes the odd-level
 * fp32-minus-fp16 difference a true precision correction (normal16_coupled).
 * Taking fp16 from the LOW 16 bits instead would share almost no information
 * with u32 and the "correction" would be a difference of two independent
 * Normals -- O(1) variance, no decay with level.
 *
 * Lane mapping matters here: norminv_fp16 reads the same 512-bit register as
 * 32 x 16-bit lanes, where lane 2j = bits 15..0 of word j and lane 2j+1 =
 * bits 31..16. The coupled path therefore keeps the ODD lanes
 * (normal16_fp16_aligned) to land on each word's top 16 bits.
 *
 * norminv_fp16 (my_mm512.h) is Giles' superdyadic piecewise-linear inverse
 * Normal CDF: symmetry + bit-derived interval index into precomputed
 * slope/intercept tables (con1/con2), one FMA to evaluate.
 *
 * normal16_fp32_bits builds the (0,1) uniform itself from the unsigned top
 * 23 bits. 23 bits (not 24) keeps (2*i32+1) inside fp32's 24-bit mantissa,
 * so u32 is exact and lands in [2^-24, 1-2^-24] by construction -- no
 * endpoint clamping needed.  my_mm512.h's norminv_fp32_approx (added by
 * Giles 2026-08-03, replacing the earlier single norminv_fp32 whose signed
 * cvtepi32 gave NaN on ~half of all inputs) is the same construction; the
 * local version is kept so the fp32 arm stays self-contained and uses an
 * unsigned convert.  Its sibling norminv_fp32_accurate keeps all 32 bits by
 * folding about 2^31 before converting -- more accurate than needed here,
 * since the odd-level correction only has to resolve the fp16 gap.
 *
 * Define USE_MKL_RNG to swap the Philox bit-source for MKL VSL
 * viRngUniformBits32 (Giles' reference statistics), same norminv transforms.
 *
 * Requires the Intel compiler (icpx): my_mm512.h pulls in <mathimf.h> / SVML.
 */

#pragma once

#include "my_mm512.h"
#include <immintrin.h>
#include <cstdint>

// ===========================================================================
// Philox-4x32-10 constants
// ===========================================================================
static const uint32_t PHILOX_M0 = 0xD2511F53u;
static const uint32_t PHILOX_M1 = 0xCD9E8D57u;
static const uint32_t PHILOX_W0 = 0x9E3779B9u;   // Weyl key bump (golden ratio)
static const uint32_t PHILOX_W1 = 0xBB67AE85u;   // Weyl key bump (sqrt 2)

// 32x32 -> high 32 bits, for all 16 lanes of a __m512i (unsigned).
static inline __m512i mulhi_epu32(__m512i a, __m512i b) {
    // even 32-bit lanes: full 64-bit products in 8 x 64-bit slots
    __m512i even = _mm512_mul_epu32(a, b);
    // odd  32-bit lanes: shift down into even position first
    __m512i ao   = _mm512_srli_epi64(a, 32);
    __m512i bo   = _mm512_srli_epi64(b, 32);
    __m512i odd  = _mm512_mul_epu32(ao, bo);
    // take the high halves and re-interleave into the original lane positions
    __m512i even_hi = _mm512_srli_epi64(even, 32);         // -> even 32-bit lanes
    __m512i odd_hi  = _mm512_slli_epi64(_mm512_srli_epi64(odd, 32), 32); // -> odd lanes
    return _mm512_mask_blend_epi32(0xAAAA, even_hi, odd_hi);
}

// ===========================================================================
// RNG state
// ===========================================================================
struct avx_rng {
    __m512i c0, c1, c2, c3;   // 128-bit counter, one independent value per lane
    __m512i k0, k1;           // 64-bit key (thread / stream id)
#ifdef USE_MKL_RNG
    // filled in the MKL specialisation below
#endif
};

// Seed a stream. `stream_id` gives each OpenMP thread an independent sequence;
// the 16 SIMD lanes are made independent by seeding c0 with the lane index.
static inline void rng_seed(avx_rng &g, uint32_t stream_id) {
    g.c0 = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    g.c1 = _mm512_setzero_si512();
    g.c2 = _mm512_setzero_si512();
    g.c3 = _mm512_setzero_si512();
    g.k0 = _mm512_set1_epi32((int)(0x1234567u ^ stream_id));
    g.k1 = _mm512_set1_epi32((int)(0x9abcdefu + stream_id * 0x9E3779B9u));
}

#ifndef USE_MKL_RNG
// ---------------------------------------------------------------------------
// Primary path: in-register Philox-4x32-10
// ---------------------------------------------------------------------------
static inline __m512i philox_next(avx_rng &g) {
    __m512i c0 = g.c0, c1 = g.c1, c2 = g.c2, c3 = g.c3;
    __m512i k0 = g.k0, k1 = g.k1;
    const __m512i m0 = _mm512_set1_epi32((int)PHILOX_M0);
    const __m512i m1 = _mm512_set1_epi32((int)PHILOX_M1);

    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        __m512i hi0 = mulhi_epu32(m0, c0);
        __m512i lo0 = _mm512_mullo_epi32(m0, c0);
        __m512i hi1 = mulhi_epu32(m1, c2);
        __m512i lo1 = _mm512_mullo_epi32(m1, c2);

        __m512i n0 = _mm512_xor_si512(_mm512_xor_si512(hi1, c1), k0);
        __m512i n1 = lo1;
        __m512i n2 = _mm512_xor_si512(_mm512_xor_si512(hi0, c3), k1);
        __m512i n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;

        k0 = _mm512_add_epi32(k0, _mm512_set1_epi32((int)PHILOX_W0));
        k1 = _mm512_add_epi32(k1, _mm512_set1_epi32((int)PHILOX_W1));
    }

    // advance the counter for the next draw (increment c1, carry into c2)
    __m512i one = _mm512_set1_epi32(1);
    __m512i nc1 = _mm512_add_epi32(g.c1, one);
    __mmask16 carry = _mm512_cmpeq_epi32_mask(nc1, _mm512_setzero_si512());
    g.c1 = nc1;
    g.c2 = _mm512_mask_add_epi32(g.c2, carry, g.c2, one);

    return c0;   // 16 x uint32 of output
}
#else
// ---------------------------------------------------------------------------
// Cross-check path: MKL VSL uniform bits (Giles' reference statistics)
// ---------------------------------------------------------------------------
#include <mkl.h>
#include <mkl_vsl.h>
static VSLStreamStatePtr mkl_stream;
#pragma omp threadprivate(mkl_stream)

static inline void mkl_rng_init(uint32_t stream_id) {
    vslNewStream(&mkl_stream, VSL_BRNG_PHILOX4X32X10, 1234u + stream_id);
}
static inline __m512i philox_next(avx_rng &) {
    alignas(64) unsigned int buf[16];
    viRngUniformBits32(VSL_RNG_METHOD_UNIFORMBITS32_STD, mkl_stream, 16, buf);
    return _mm512_load_si512((const void *)buf);
}
#endif

// ===========================================================================
// fp16 Normal transforms (Giles' norminv_fp16 spline from my_mm512.h)
// ===========================================================================

// Uncoupled fp16 draw. norminv_fp16 natively gives 32 fp16 lanes; this keeps
// the low 256 bits, i.e. lanes 0..15 = BOTH halves of words 0..7 (words 8..15
// are discarded). Fine here precisely because there is no fp32 partner to
// stay word-aligned with.
// Note this consumes a full 512-bit Philox draw to return 16 fp16 lanes; 
// The current implementation processes 16 paths per SIMD iteration to match
// the fp32 path.

// Not suitable for the coupled path because lane 'i' would not correspond to
// fp32 lane 'i'; use normal16_fp16_aligned() instead.
// path -- lane i would not correspond to fp32 lane 'i' (use _aligned below).
static inline __m256h normal16_fp16_truncated(__m512i bits) {
    return _mm512_castph512_ph256(norminv_fp16(bits));
}

// Fresh, uncoupled fp16 draw: its own Philox call, not shared with any fp32 draw.
static inline __m256h normal16_fp16(avx_rng &g) {
    return normal16_fp16_truncated(philox_next(g));
}

// ===========================================================================
// fp32 Normal / exponential transforms (accurate reference precision)
// ===========================================================================

// Builds the (0,1) uniform from a word's top 23 bits (bits 31..9), placed at
// the interval midpoint: u = 2^-23*i23 + 2^-24 = (2*i23+1)*2^-24. With 23
// bits, 2*i23+1 has at most 24 significant bits, so u is exactly representable
// in fp32 and the fmadd is exact. Range is [2^-24, 1-2^-24], symmetric about
// 0.5 and never 0 or 1, so cdfnorminv is safe with no clamping.
static inline __m512 uniform16_fp32(__m512i bits) {
    __m512i u23 = _mm512_srli_epi32(bits, 9);
    return _mm512_fmadd_ps(_mm512_cvtepu32_ps(u23),
                           _mm512_set1_ps(0x1.0p-23f),
                           _mm512_set1_ps(0x1.0p-24f));
}

static inline __m512 normal16_fp32_bits(__m512i bits) {
    return _mm512_cdfnorminv_ps(uniform16_fp32(bits));
}

static inline __m512 normal16_fp32(avx_rng &g) {
    return normal16_fp32_bits(philox_next(g));
}

// 16 exponential-ish RVs in fp32 (log-uniform bridge draws).
// Returns Lrv = log(U), U~Uniform(0,1)  (i.e. Lrv = -Exp(1), matching the
// scalar reference's Lrv = -next_exponential()).
static inline __m512 loguniform16_fp32(avx_rng &g) {
    __m512i bits = philox_next(g);
    __m512i u23 = _mm512_srli_epi32(bits, 9);
    __m512  U   = _mm512_fmadd_ps(_mm512_cvtepu32_ps(u23),
                                  _mm512_set1_ps(0x1.0p-23f),
                                  _mm512_set1_ps(0x1.0p-24f));
    return _mm512_log_ps(U);
}

// ===========================================================================
// Coupled fp32/fp16 draw (odd, precision-correction levels)
// ===========================================================================

// fp16 half of the coupled draw. norminv_fp16 sees the register as 32 x
// 16-bit lanes (lane 2j = bits 15..0 of word j, lane 2j+1 = bits 31..16), so
// the ODD lanes are the ones sitting on each word's top 16 bits. Permuting
// lanes 1,3,..,31 down into 0..15 gives a result whose lane j is word j's
// top 16 bits -- the same word fp32's lane j reads its top 23 bits from,
// hence u16 = u32 truncated to 16 bits. No masking is needed: norminv_fp16
// is per-lane, so the even lanes it also computes are simply discarded here.
static inline __m256h normal16_fp16_aligned(__m512i bits) {
    __m512h full = norminv_fp16(bits);

// _mm512_set_epi16 fills lanes from high to low; the low 16 result lanes
// are set to input lanes 1,3,5,...,31.
    const __m512i odd_idx = _mm512_set_epi16(
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                31,29,27,25,23,21,19,17,15,13,11,9,7,5,3,1); 

    __m512i gathered = _mm512_permutexvar_epi16(odd_idx, _mm512_castph_si512(full));
    return _mm512_castph512_ph256(_mm512_castsi512_ph(gathered));
}

// One Philox draw feeds both transforms: fp32 reads each word's top 23 bits,
// fp16 the same word's top 16 bits. Each precision runs its own inverse-CDF
// transform -- neither result is a narrowing of the other -- so the odd-level
// difference isolates fp16 arithmetic/quantisation error rather than adding
// independent-draw noise on top of it.
static inline void normal16_coupled(avx_rng &g, __m512 &z_f32, __m256h &z_f16) {
    __m512i bits = philox_next(g);
    z_f32 = normal16_fp32_bits(bits);
    z_f16 = normal16_fp16_aligned(bits);
}

