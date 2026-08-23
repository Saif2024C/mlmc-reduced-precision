/*

a header file defining various functions for AVX-512 vector types
 _m512 (16 x fp32) and _m512h (32 x fp16)

at the bottom are functions which can be used to generate 32-bit
and 16-bit Normal random variables

*/

#pragma once

#include <immintrin.h>
#include <mathimf.h>

#include <iostream>
#include <cmath>

//
// __m512 functions
//

inline __m512 fmadd(__m512& a, __m512& b, __m512& c) {
  return _mm512_fmadd_ps(a, b, c);
}

inline __m512 fmsub(__m512& a, __m512& b, __m512& c) {
  return _mm512_fmsub_ps(a, b, c);
}

inline __m512 fnmadd(__m512& a, __m512& b, __m512& c) {
  return _mm512_fnmadd_ps(a, b, c);
}

inline __m512 fnmsub(__m512& a, __m512& b, __m512& c) {
  return _mm512_fnmsub_ps(a, b, c);
}

inline __m512 rcp(__m512& x) {
    return _mm512_rcp14_ps(x);
}

inline __m512 sqrt(__m512& x) {
    return _mm512_sqrt_ps(x);
}

inline __m512 rsqrt(__m512& x) {
    return _mm512_rsqrt14_ps(x);
}

inline __m512 fmax(__m512& x, __m512& y) {
  return _mm512_max_ps(x,y);
}

inline __m512 relu(__m512& x) {
  return _mm512_max_ps(x,_mm512_setzero_ps());
}

inline __m512 exp(__m512& x) {
  return _mm512_exp_ps(x);
}

inline __m512 log(__m512& x) {
  return _mm512_log_ps(x);
}

inline __m512 sin(__m512& x) {
  return _mm512_sin_ps(x);
}

inline __m512 cos(__m512& x) {
  return _mm512_cos_ps(x);
}

inline float reduce_add(__m512& x) {
  return _mm512_reduce_add_ps(x);
}

inline __m512 aligned_load(float *ptr) {
    return _mm512_load_ps(ptr);
}

inline __m512 unaligned_load(float *ptr) {
    return _mm512_loadu_ps(ptr);
}

inline void aligned_store(float *ptr, __m512& x) {
    _mm512_store_ps(ptr, x);
}

inline void unaligned_store(float *ptr, __m512& x) {
    _mm512_storeu_ph(ptr, x);
}


//
// __m512h functions
//

inline __m512h fmadd(__m512h& a, __m512h& b, __m512h& c) {
  return _mm512_fmadd_ph(a, b, c);
}

inline __m512h fmsub(__m512h& a, __m512h& b, __m512h& c) {
  return _mm512_fmsub_ph(a, b, c);
}

inline __m512h fnmadd(__m512h& a, __m512h& b, __m512h& c) {
  return _mm512_fnmadd_ph(a, b, c);
}

inline __m512h fnmsub(__m512h& a, __m512h& b, __m512h& c) {
  return _mm512_fnmsub_ph(a, b, c);
}

inline __m512h rcp(__m512h& x) {
    return _mm512_rcp_ph(x);
}

inline __m512h sqrt(__m512h& x) {
    return _mm512_sqrt_ph(x);
}

inline __m512h rsqrt(__m512h& x) {
    return _mm512_rsqrt_ph(x);
}

inline __m512h fmax(__m512h& x, __m512h& y) {
  return _mm512_max_ph(x,y);
}

inline __m512h relu(__m512h& x) {
  return _mm512_max_ph(x,_mm512_setzero_ph());
}

inline __m512h exp(__m512h& x) {
  return _mm512_exp_ph(x);
}

inline __m512h log(__m512h& x) {
  return _mm512_log_ph(x);
}

inline __m512h sin(__m512h& x) {
  return _mm512_sin_ph(x);
}

inline __m512h cos(__m512h& x) {
  return _mm512_cos_ph(x);
}

inline float reduce_add(__m512h& x) {
  return (float) _mm512_reduce_add_ph(x);
}

inline __m512h aligned_loadh(_Float16 *ptr) {
    return _mm512_load_ph(ptr);
}

inline __m512h unaligned_loadh(_Float16 *ptr) {
    return _mm512_loadu_ph(ptr);
}

inline void aligned_storeh(_Float16 *ptr, __m512h& x) {
    _mm512_store_ph(ptr, x);
}

inline void unaligned_storeh(_Float16 *ptr, __m512h& x) {
    _mm512_storeu_ph(ptr, x);
}

// Debug print functions

inline void debug_print(__m512i& x) {
    alignas(64) int tmp[16];
    _mm512_store_epi32(tmp, x);
    for (int i = 0; i < 16; ++i)
      std::cout << tmp[i] << (i != 15 ? ", " : "\n");
}

inline void debug_print(__m512& x) {
    alignas(64) float tmp[16];
    _mm512_store_ps(tmp, x);
    for (int i = 0; i < 16; ++i)
      std::cout << tmp[i] << (i != 15 ? ", " : "\n");
}

inline void debug_print_short(__m512i& x) {
    alignas(64) short tmp[32];
    _mm512_storeu_epi16(tmp, x);
    for (int i = 0; i < 32; ++i)
      std::cout << static_cast<int>(tmp[i]) << (i != 31 ? ", " : "\n");
}

inline void debug_print(__m512h& x) {
    alignas(64) _Float16 tmp[32];
    _mm512_store_ph(tmp, x);
    for (int i = 0; i < 32; ++i)
      std::cout << static_cast<float>(tmp[i]) << (i != 31 ? ", " : "\n");
}


// 16-bit gather functions, assuming data is provided as an array
// of 32-bit values each duplicating the 16-bit value; this is due
// to the lack of an AVX gather instruction for 16-bit data

inline __m512i gather_epi16(const int* base, __m512i indices16) {
  __m512i even_lanes = _mm512_set_epi16(
            30,30,28,28,26,26,24,24,
            22,22,20,20,18,18,16,16,
            14,14,12,12,10,10,8,8,
             6, 6, 4, 4, 2, 2, 0, 0
        );

  __m512i odd_lanes = _mm512_set_epi16(
            31,31,29,29,27,27,25,25,
            23,23,21,21,19,19,17,17,
            15,15,13,13,11,11,9,9,
             7, 7, 5, 5, 3, 3, 1, 1
        );

  __m512i index_even = _mm512_maskz_permutexvar_epi16(0x55555555,
                                           even_lanes, indices16);
  __m512i index_odd  = _mm512_maskz_permutexvar_epi16(0x55555555,
                                            odd_lanes, indices16);

  __m512i gathered_even = _mm512_i32gather_epi32(index_even, base, 4);
  __m512i gathered_odd  = _mm512_i32gather_epi32(index_odd,  base, 4);

  return _mm512_mask_blend_epi16(0xAAAAAAAA,gathered_even,gathered_odd);
}

inline __m512h gather_fp16(const _Float16* base, __m512i indices16) {
  return _mm512_castsi512_ph(gather_epi16((int*) base,indices16));
}
//
// AVX-512 vectors with superdyadic piecewise linear spline coefs
//

__m512i con1 = _mm512_castph_si512(_mm512_setr_ph(
    0.0000000f16,   0.0000000f16,   0.0000000f16,   0.0000000f16, 
    0.0000000f16,   6.4032523f16,   4.6524331f16,   3.3429773f16, 
    2.4320530f16,   1.7896978f16,   1.2990414f16,   0.9409888f16, 
    0.6846983f16,   0.5001156f16,   0.3646161f16,   0.2660284f16, 
    0.1946923f16,   0.1428579f16,   0.1050341f16,   0.0774698f16, 
    0.0573213f16,   0.0425913f16,   0.0317966f16,   0.0238803f16, 
    0.0180653f16,   0.0137943f16,   0.0106610f16,   0.0083761f16, 
    0.0067380f16,   0.0056217f16,   0.0049986f16,   0.0000000f16
                                                     ));   // slope
__m512i con2 = _mm512_castph_si512(_mm512_setr_ph(
   -4.3875114f16,  -4.0804306f16,   0.0000000f16,  -3.9572776f16, 
   -3.8753599f16,  -4.0135121f16,  -3.9394078f16,  -3.8606159f16, 
   -3.7800080f16,  -3.7008669f16,  -3.6146951f16,  -3.5256808f16, 
   -3.4346582f16,  -3.3423427f16,  -3.2462729f16,  -3.1474375f16, 
   -3.0459985f16,  -2.9419128f16,  -2.8343864f16,  -2.7236453f16, 
   -2.6091068f16,  -2.4907230f16,  -2.3679991f16,  -2.2407355f16, 
   -2.1085159f16,  -1.9711855f16,  -1.8287113f16,  -1.6818128f16, 
   -1.5329659f16,  -1.3898027f16,  -1.2778941f16,   0.0000000f16
                                                     ));   // constant

//
// superdyadic piecewise linear mapping from 16-bit integers
// to inverse Normal CDF
//
    
inline __m512h norminv_fp16(__m512i i) {
  __m512h x, y;

  __m512i all_ones  = _mm512_ternarylogic_epi32(i,i,i,255);
  __m512i two_to_15 = _mm512_slli_epi16(all_ones,15);

  __mmask32 m = _mm512_cmpge_epu16_mask(i,two_to_15);
  i = _mm512_mask_sub_epi16(i,m,all_ones,i);

  __m512h two_to_minus_7 = _mm512_castsi512_ph(_mm512_srli_epi16(two_to_15,2));

  x = _mm512_cvt_roundepi16_ph(i,_MM_FROUND_TO_ZERO|_MM_FROUND_NO_EXC);
  x = _mm512_mul_ph(x,two_to_minus_7);
  y = _mm512_mul_ph(x,x);
  __m512i j = _mm512_srli_epi16(_mm512_castph_si512(y),10);

  __m512h c1 = _mm512_castsi512_ph(_mm512_permutexvar_epi16(j,con1));
  __m512h c2 = _mm512_castsi512_ph(_mm512_permutexvar_epi16(j,con2));
  x = _mm512_fmadd_ph(x,c1,c2);
  x = _mm512_mask_sub_ph(x,m,_mm512_setzero_ph(),x);

  return x;
}

//
// float version of norminv for 32-bit integer inputs
//

// version 1 uses leading 23 bits to achieve a symmetric distribution

inline __m512 norminv_fp32_approx(__m512i i) {
  __m512 x;
  i = _mm512_srli_epi32(i,9);
  x = _mm512_cvtepi32_ps(i);
  x = _mm512_fmadd_ps(x,_mm512_set1_ps(0x1p-23f),
                        _mm512_set1_ps(0x1p-24f));
  x = _mm512_cdfnorminv_ps(x);
  return x;
}

// version 2 uses all 32 bits for an accurate symmetric distribution

inline __m512 norminv_fp32_accurate(__m512i i) {
  __m512 x;

  __m512i all_ones  = _mm512_ternarylogic_epi32(i,i,i,255);
  __m512i two_to_31 = _mm512_slli_epi32(all_ones,31);

  __mmask32 m = _mm512_cmpge_epu32_mask(i,two_to_31);
  i = _mm512_mask_sub_epi32(i,m,all_ones,i);
  // lower latency alternative to flip all bits -- no 16-bit equivalent
  // i = _mm512_mask_xor_epi32(i,m,all_ones,i);

  x = _mm512_cvtepi32_ps(i);
  x = _mm512_fmadd_ps(x,_mm512_set1_ps(0x1p-32f),
                        _mm512_set1_ps(0x1p-33f));
  x = _mm512_cdfnorminv_ps(x);
  x = _mm512_mask_sub_ps(x,m,_mm512_setzero_ps(),x);
  // lower latency alternative to flip the sign bit -- no 16-bit equivalent
  // x = _mm512_castsi512_ph(_mm512_mask_xor_epi32(_mm512_castph_si512(x),
  //                                   m,two_to_31,_mm512_castph_si512(x)));
  return x;
}

//
// convert fp16 odd elements to fp32
// (this is for mixed-precision MLMC)
//

inline __m512 cvt_odd_ph_ps(__m512h x) {
  __m512i i = _mm512_castph_si512(x);
  i = _mm512_maskz_compress_epi16(0xAAAAAAAA,i);
  __m256i j = _mm512_castsi512_si256(i);
  __m256h y = _mm256_castsi256_ph(j);
  return _mm512_cvtxph_ps(y);
}

//
// 32-bit unsigned integer min/max
//

inline __m512i min(__m512i& i, __m512i& j) {
  return _mm512_min_epu32(i,j);
}

inline __m512i max(__m512i& i, __m512i& j) {
  return _mm512_max_epu32(i,j);
}
