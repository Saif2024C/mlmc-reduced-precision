// ---------------------------------------------------------------------------
// fp_contract_test.cpp -- direct evidence that the compiler contracts a
// SEPARATELY WRITTEN vector multiply and add into a single vector FMA, and
// that -ffp-contract controls it at 512-bit width.
//
// Build both ways and diff the assembly:
//   icpx -O3 -march=native -std=c++17 -ffp-contract=fast -S fp_contract_test.cpp -o fast.s
//   icpx -O3 -march=native -std=c++17 -ffp-contract=off  -S fp_contract_test.cpp -o off.s
//   grep -c vfmadd fast.s off.s
//
// mil_incr32_contract below is the Milstein increment copied verbatim from
// options_mm512.h -- written as plain _mm512_mul_ps / _mm512_add_ps, with NO
// fmadd intrinsic anywhere. Any vfmadd appearing in its assembly was inserted
// by the compiler, not by the source.
//
// mil_incr32_explicit is the same arithmetic written with _mm512_fmadd_ps on
// purpose: it should emit vfmadd under BOTH flag settings, since there is
// nothing left to contract.
// ---------------------------------------------------------------------------

#include <immintrin.h>

struct C32 { __m512 r, sig, halfsig2, hf; };

// ---- 1. plain mul/add chain (verbatim from options_mm512.h) ----------------
// X*r*h + X*sig*dW + 0.5*sig^2*X*(dW^2 - h)
__attribute__((noinline))
__m512 mil_incr32_contract(__m512 X, __m512 dW, const C32 &c) {
    __m512 drift = _mm512_mul_ps(_mm512_mul_ps(X, c.r), c.hf);
    __m512 diff  = _mm512_mul_ps(_mm512_mul_ps(X, c.sig), dW);
    __m512 d2h   = _mm512_sub_ps(_mm512_mul_ps(dW, dW), c.hf);
    __m512 mil   = _mm512_mul_ps(_mm512_mul_ps(X, c.halfsig2), d2h);
    return _mm512_add_ps(_mm512_add_ps(drift, diff), mil);
}

// ---- 2. the minimal case: one mul feeding one add -------------------------
// The clearest possible demonstration. Two intrinsics in, one instruction out.
__attribute__((noinline))
__m512 mul_then_add(__m512 a, __m512 b, __m512 c) {
    __m512 t = _mm512_mul_ps(a, b);     // vmulps
    return _mm512_add_ps(t, c);         // vaddps   -> fused to one vfmadd?
}

// ---- 3. explicit FMA: nothing to contract ---------------------------------
__attribute__((noinline))
__m512 explicit_fma(__m512 a, __m512 b, __m512 c) {
    return _mm512_fmadd_ps(a, b, c);
}

// ---- 4. intermediate REUSED: must NOT fuse --------------------------------
// t is used twice, so the multiply result has to exist as a real value and
// the mul/add pair cannot collapse into an FMA.
__attribute__((noinline))
__m512 mul_reused(__m512 a, __m512 b, __m512 c, __m512 *out) {
    __m512 t = _mm512_mul_ps(a, b);
    *out = t;                            // second use of t
    return _mm512_add_ps(t, c);
}

// ---- 5. fp16 equivalent, 32 lanes -----------------------------------------
__attribute__((noinline))
__m512h mul_then_add_ph(__m512h a, __m512h b, __m512h c) {
    __m512h t = _mm512_mul_ph(a, b);
    return _mm512_add_ph(t, c);
}
