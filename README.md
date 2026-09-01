# Reduced-precision nested MLMC with the Milstein discretisation

Code and results for the MSc dissertation *Reduced Precision Hardware
Acceleration of Nested Multi-Level Monte Carlo using CPUs* (University of
Oxford, supervisor Prof. Mike Giles).

Multilevel Monte Carlo estimators for option pricing under the Milstein
discretisation, evaluated in FP16 and FP32 on AVX-512. The payoffs reported
are a scalar Asian call and European and Asian calls on a five-asset
correlated basket.

## Layout

```
src/core/       MLMC driver and test harnesses (mlmc.cpp, mlmc_test.cpp,
                nested_mlmc_test.cpp) and the scalar RNG
src/scalar/     Scalar payoff estimators, plain and nested
src/basket/     Five-asset basket estimators, plain and nested
src/AVX-512/    Vectorised FP16/FP32 estimators, the in-register Philox
                generator, kernel benchmarks and validation tests
python/         Plotting scripts for the generated .txt tables
outputs/        Generated convergence tables, timings and figures
```

## Building

There is no build system. Each program is a single translation unit: the
top-level `.cpp` includes the files below it, so only that file is compiled.

```bash
mkdir -p build
g++ -O2 -std=c++11 src/scalar/asian_lookback_scalar.cpp -o build/asian_lookback_scalar
g++ -O2 -std=c++11 src/basket/basket_scalar.cpp          -o build/basket_scalar
```

The non-AVX FP16 estimators need half-precision support, and fast-math
contraction must stay off since it would undo the explicit bracketing of the
Milstein step:

```bash
g++ -O0 -std=c++11 -mf16c -fno-fast-math src/scalar/nested_scalar_milstein_fp16.cpp -o build/nested_scalar_fp16
```

AVX-512 FP16 requires `icpx` and hardware carrying the FP16 extension:

```bash
source /opt/intel/oneapi/setvars.sh
icpx -O3 -march=native -std=c++17 src/AVX-512/nested_scalar_milstein_fp16_avx512.cpp -o build/nsk_avx
```

The FP32 reference the nested schemes are compared against is standard
non-nested MLMC, built separately by `std_mlmc_fp32_scalar_avx512.cpp` and
`std_mlmc_fp32_basket_avx512.cpp`. These reuse the nested estimators' own
level function, so the arithmetic, the Philox stream and the payoff are
identical and only the ladder differs. Their output goes in
`outputs/avx512/stdmlmc/`, which the overlay and cost scripts read.

`src/AVX-512/Makefile` builds and runs every AVX-512 variant, serial and
OpenMP, with per-payoff timing.

## Plotting

```bash
MPLBACKEND=Agg python3 python/nested_mlmc_plot_python.py outputs/avx512/basket/nested_basket_fp16_avx512_adaptive_1
MPLBACKEND=Agg python3 python/nested_overlay_plot_python.py outputs/avx512/basket basket 1
```

## Results

Fitted variance decay exponent, measured on an Intel Xeon Gold 6538Y+
(Emerald Rapids). The MLMC cost model requires this to exceed the cost growth
exponent, which is 1 for these estimators.

| Scheme | Basket European | Basket Asian | Scalar Asian |
|---|---|---|---|
| FP16, no compensation | 0.262 | -0.499 | -0.745 |
| FP16, Kahan compensated | 1.089 | 1.036 | 0.824 |
| Adaptive, FP16 below cut-off | 1.961 | 2.151 | 2.219 |
| Pure FP32 (non-nested MLMC) | 1.929 | 2.138 | 2.189 |

At matched sample counts the level-0 kernel runs 2.07x faster in FP16 with
Kahan compensation than in FP32 for the basket European payoff, and 1.84x for
the scalar Asian payoff.
