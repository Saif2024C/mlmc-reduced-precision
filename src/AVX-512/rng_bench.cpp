//
// Standalone MKL VSL vs Philox-4x32-10 throughput benchmark.
//
// Philox here is self-contained and uses ALL FOUR output words per counter
// increment (4 x __m512i = 64 x 32-bit words per call), so the counter is
// advanced once per 64 words rather than once per 16. This is the fair
// comparison against MKL, which also yields every word it generates.
//
// Three stages are timed, matching the transforms in the thesis:
//   1. raw uniform bits
//   2. FP32 Normals   (uniform -> _mm512_cdfnorminv_ps)
//   3. FP16 Normals   (norminv_fp16, 32 lanes)
//
// Build on mimic:
//   source /opt/intel/oneapi/setvars.sh
//   icpx -O3 -march=native -std=c++17 -qopenmp \
//        --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13 \
//        rng_bench.cpp -o rng_bench -lmkl_rt
//
// Run:  ./rng_bench [n_vectors] [threads]
//
#include "my_mm512.h"
#include <immintrin.h>
#include <mkl.h>
#include <omp.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

static int MKL_BRNG = VSL_BRNG_PHILOX4X32X10;
static const char *MKL_NAME = "Philox";

static const uint32_t PHILOX_M0 = 0xD2511F53u;
static const uint32_t PHILOX_M1 = 0xCD9E8D57u;
static const uint32_t PHILOX_W0 = 0x9E3779B9u;
static const uint32_t PHILOX_W1 = 0xBB67AE85u;

// 32x32 -> high 32 bits, unsigned, for all 16 lanes.
static inline __m512i mulhi_epu32(__m512i a, __m512i b) {
    __m512i even    = _mm512_mul_epu32(a, b);
    __m512i ao      = _mm512_srli_epi64(a, 32);
    __m512i bo      = _mm512_srli_epi64(b, 32);
    __m512i odd     = _mm512_mul_epu32(ao, bo);
    __m512i even_hi = _mm512_srli_epi64(even, 32);
    __m512i odd_hi  = _mm512_slli_epi64(_mm512_srli_epi64(odd, 32), 32);
    return _mm512_mask_blend_epi32(0xAAAA, even_hi, odd_hi);
}

struct philox_state {
    __m512i c0, c1, c2, c3;   // 128-bit counter, 16 lanes wide
    __m512i k0, k1;           // 64-bit key
};

static inline void philox_seed(philox_state &g, uint32_t thread_id) {
    g.c0 = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    g.c1 = _mm512_setzero_si512();
    g.c2 = _mm512_setzero_si512();
    g.c3 = _mm512_setzero_si512();
    g.k0 = _mm512_set1_epi32((int)(0x9E3779B9u ^ thread_id));
    g.k1 = _mm512_set1_epi32((int)(0xBB67AE85u + thread_id));
}

// Ten rounds; returns all four output words. Counter advances once per call,
// so one call yields 64 x 32-bit words.
static inline void philox_next4(philox_state &g,
                                __m512i &x0, __m512i &x1,
                                __m512i &x2, __m512i &x3) {
    __m512i c0 = g.c0, c1 = g.c1, c2 = g.c2, c3 = g.c3;
    __m512i k0 = g.k0, k1 = g.k1;
    const __m512i m0 = _mm512_set1_epi32((int)PHILOX_M0);
    const __m512i m1 = _mm512_set1_epi32((int)PHILOX_M1);

    for (int r = 0; r < 10; ++r) {
        __m512i hi1 = mulhi_epu32(m1, c2);
        __m512i lo1 = _mm512_mullo_epi32(m1, c2);
        __m512i hi0 = mulhi_epu32(m0, c0);
        __m512i lo0 = _mm512_mullo_epi32(m0, c0);

        __m512i n0 = _mm512_xor_si512(_mm512_xor_si512(hi1, c1), k0);
        __m512i n1 = lo1;
        __m512i n2 = _mm512_xor_si512(_mm512_xor_si512(hi0, c3), k1);
        __m512i n3 = lo0;

        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        k0 = _mm512_add_epi32(k0, _mm512_set1_epi32((int)PHILOX_W0));
        k1 = _mm512_add_epi32(k1, _mm512_set1_epi32((int)PHILOX_W1));
    }
    x0 = c0; x1 = c1; x2 = c2; x3 = c3;
    g.c0 = _mm512_add_epi32(g.c0, _mm512_set1_epi32(16));   // 16 lanes consumed
}

// bits -> uniform on (0,1), 16 FP32 lanes  (midpoint construction)
static inline __m512 uniform_fp32(__m512i bits) {
    __m512i u23 = _mm512_srli_epi32(bits, 9);
    return _mm512_fmadd_ps(_mm512_cvtepu32_ps(u23),
                           _mm512_set1_ps(0x1.0p-23f),
                           _mm512_set1_ps(0x1.0p-24f));
}

static double now() { return omp_get_wtime(); }

int main(int argc, char **argv) {
    long   nvec    = (argc > 1) ? atol(argv[1]) : 134217728L; // __m512i draws
    int    threads = (argc > 2) ? atoi(argv[2]) : 1;
    if (argc > 3) {
        const char *b = argv[3];
        if      (!strcmp(b,"mt19937")) { MKL_BRNG = VSL_BRNG_MT19937;  MKL_NAME = "MT19937"; }
        else if (!strcmp(b,"mt2203"))  { MKL_BRNG = VSL_BRNG_MT2203;   MKL_NAME = "MT2203"; }
        else if (!strcmp(b,"sfmt"))    { MKL_BRNG = VSL_BRNG_SFMT19937;MKL_NAME = "SFMT19937"; }
        else if (!strcmp(b,"mcg59"))   { MKL_BRNG = VSL_BRNG_MCG59;    MKL_NAME = "MCG59"; }
    }
    omp_set_num_threads(threads);

    printf("# vectors = %ld, threads = %d, MKL BRNG = %s\n", nvec, threads, MKL_NAME);
    printf("# Philox uses all 4 output words per counter increment\n\n");
    printf("%-16s %14s %14s %8s\n", "stage", "MKL (RNG/s)", "ours (RNG/s)", "ratio");

    // ---------------- stage 1: uniform bits ----------------
    double t0, t1, t_mkl, t_phi;
    volatile uint32_t sink = 0;

    {   // MKL: generate into a per-thread buffer
        const int BUF = 65536;
        t0 = now();
        #pragma omp parallel
        {
            VSLStreamStatePtr st;
            vslNewStream(&st, MKL_BRNG, 7777 + omp_get_thread_num());
            std::vector<unsigned int> buf(BUF);
            long per = (nvec * 16) / threads;
            uint32_t acc = 0;
            for (long done = 0; done < per; done += BUF) {
                int n = (int)std::min((long)BUF, per - done);
                viRngUniformBits32(VSL_RNG_METHOD_UNIFORMBITS32_STD, st, n, buf.data());
                acc ^= buf[n-1];
            }
            vslDeleteStream(&st);
            #pragma omp atomic
            sink ^= acc;
        }
        t1 = now(); t_mkl = t1 - t0;
    }
    {   // Philox: in-register, all four words
        t0 = now();
        #pragma omp parallel
        {
            philox_state g; philox_seed(g, omp_get_thread_num());
            long calls = (nvec / 4) / threads;      // 4 words per call
            __m512i acc = _mm512_setzero_si512();
            for (long n = 0; n < calls; ++n) {
                __m512i x0,x1,x2,x3;
                philox_next4(g, x0,x1,x2,x3);
                acc = _mm512_xor_si512(acc, _mm512_xor_si512(
                          _mm512_xor_si512(x0,x1), _mm512_xor_si512(x2,x3)));
            }
            uint32_t lane0 = (uint32_t)_mm512_cvtsi512_si32(acc);
            #pragma omp atomic
            sink ^= lane0;
        }
        t1 = now(); t_phi = t1 - t0;
    }
    {
        double n = (double)nvec * 16.0;
        printf("%-16s %14.3e %14.3e %8.2f\n", "uniform bits",
               n/t_mkl, n/t_phi, t_mkl/t_phi);
    }

    // ---------------- stage 2: FP32 Normals ----------------
    {
        const int BUF = 65536;
        t0 = now();
        #pragma omp parallel
        {
            VSLStreamStatePtr st;
            vslNewStream(&st, MKL_BRNG, 4242 + omp_get_thread_num());
            std::vector<unsigned int> buf(BUF);
            long per = (nvec * 16) / threads;
            float acc = 0.f;
            for (long done = 0; done < per; done += BUF) {
                int n = (int)std::min((long)BUF, per - done);
                viRngUniformBits32(VSL_RNG_METHOD_UNIFORMBITS32_STD, st, n, buf.data());
                for (int i = 0; i + 16 <= n; i += 16) {
                    __m512i b = _mm512_loadu_si512((const void*)(buf.data()+i));
                    __m512  z = _mm512_cdfnorminv_ps(uniform_fp32(b));
                    acc += _mm512_cvtss_f32(z);
                }
            }
            vslDeleteStream(&st);
            #pragma omp atomic
            sink ^= (uint32_t)acc;
        }
        t1 = now(); t_mkl = t1 - t0;

        t0 = now();
        #pragma omp parallel
        {
            philox_state g; philox_seed(g, omp_get_thread_num());
            long calls = (nvec / 4) / threads;
            __m512 acc = _mm512_setzero_ps();
            for (long n = 0; n < calls; ++n) {
                __m512i x0,x1,x2,x3;
                philox_next4(g, x0,x1,x2,x3);
                acc = _mm512_add_ps(acc, _mm512_cdfnorminv_ps(uniform_fp32(x0)));
                acc = _mm512_add_ps(acc, _mm512_cdfnorminv_ps(uniform_fp32(x1)));
                acc = _mm512_add_ps(acc, _mm512_cdfnorminv_ps(uniform_fp32(x2)));
                acc = _mm512_add_ps(acc, _mm512_cdfnorminv_ps(uniform_fp32(x3)));
            }
            #pragma omp atomic
            sink ^= (uint32_t)_mm512_cvtss_f32(acc);
        }
        t1 = now(); t_phi = t1 - t0;

        double n = (double)nvec * 16.0;
        printf("%-16s %14.3e %14.3e %8.2f\n", "FP32 Normal",
               n/t_mkl, n/t_phi, t_mkl/t_phi);
    }

    // ---------------- stage 3: FP16 Normals ----------------
    {
        const int BUF = 65536;
        t0 = now();
        #pragma omp parallel
        {
            VSLStreamStatePtr st;
            vslNewStream(&st, MKL_BRNG, 1234 + omp_get_thread_num());
            std::vector<unsigned int> buf(BUF);
            long per = (nvec * 16) / threads;
            float acc = 0.f;
            for (long done = 0; done < per; done += BUF) {
                int n = (int)std::min((long)BUF, per - done);
                viRngUniformBits32(VSL_RNG_METHOD_UNIFORMBITS32_STD, st, n, buf.data());
                for (int i = 0; i + 16 <= n; i += 16) {
                    __m512i b = _mm512_loadu_si512((const void*)(buf.data()+i));
                    __m512h z = norminv_fp16(b);
                    acc += (float)_mm512_cvtsh_h(z);
                }
            }
            vslDeleteStream(&st);
            #pragma omp atomic
            sink ^= (uint32_t)acc;
        }
        t1 = now(); t_mkl = t1 - t0;

        t0 = now();
        #pragma omp parallel
        {
            philox_state g; philox_seed(g, omp_get_thread_num());
            long calls = (nvec / 4) / threads;
            __m512h acc = _mm512_setzero_ph();
            for (long n = 0; n < calls; ++n) {
                __m512i x0,x1,x2,x3;
                philox_next4(g, x0,x1,x2,x3);
                acc = _mm512_add_ph(acc, norminv_fp16(x0));
                acc = _mm512_add_ph(acc, norminv_fp16(x1));
                acc = _mm512_add_ph(acc, norminv_fp16(x2));
                acc = _mm512_add_ph(acc, norminv_fp16(x3));
            }
            #pragma omp atomic
            sink ^= (uint32_t)(float)_mm512_cvtsh_h(acc);
        }
        t1 = now(); t_phi = t1 - t0;

        // FP16 path yields 32 lanes per register
        double n = (double)nvec * 32.0;
        printf("%-16s %14.3e %14.3e %8.2f\n", "FP16 Normal",
               n/t_mkl, n/t_phi, t_mkl/t_phi);
    }

    printf("\n# sink = %u (prevents dead-code elimination)\n", (unsigned)sink);
    return 0;
}
