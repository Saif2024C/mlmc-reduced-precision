/*
% MLMC harness driver + numerics for the multi-asset basket benchmarks:
%  - Basket European call  (option = 1)
%  - Basket Asian   call   (option = 2)
%
% All basket SDE evolution, correlation handling, fine/coarse coupling and
% payoff calculation (namespace basket_mlmc, function basket_path) live in
% this file alongside the harness glue:
%   1. the level estimator basket_scalar_l, adapting basket_path to contract
%          void mlmc_l(int l, int N, double *sums);
%   2. fed by the SHARED global RNG stream from mlmc_rng.cpp (next_normal),
%      which is what makes mlmc_test_100's 100 repeated runs independent
%      (the stream advances across runs; it is never reseeded between them);
%   3. accumulating all seven sums[] entries the convergence / kurtosis /
%      telescoping checks in mlmc_test.cpp require.
%
% Build (single translation unit, from the repo root):
%   g++ -O2 -std=c++11 src/basket/basket_scalar.cpp -o build/basket_scalar
%   cd outputs && ../build/basket_scalar   # writes basket_scalar_{1,2}.txt and *_100.txt
%
% Neither basket payoff has a closed form, so the exact value is NaN.
*/

#include "../core/mlmc_test_100.cpp"          // master file for 100 tests (pulls in mlmc_test -> mlmc)
#include "../core/mlmc_rng.cpp"               // shared global RNG (C++11 default, or MKL/VSL under _OPENMP)

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace basket_mlmc {

constexpr int kDim = 5;
constexpr double kK = 100.0;
constexpr double kT = 1.0;
constexpr double kR = 0.05;
constexpr std::array<double, kDim> kSigma = {0.25, 0.30, 0.35, 0.40, 0.45};
constexpr std::array<double, kDim> kWeights = {0.2, 0.2, 0.2, 0.2, 0.2};

using Vec = std::array<double, kDim>;
using Mat = std::array<std::array<double, kDim>, kDim>;

Mat cholesky_lower(const Mat& a) {
  Mat l{};
  for (int i = 0; i < kDim; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = a[i][j];
      for (int k = 0; k < j; ++k) {
        sum -= l[i][k] * l[j][k];
      }
      if (i == j) {
        if (sum <= 0.0) {
          throw std::runtime_error("Correlation matrix is not positive definite");
        }
        l[i][j] = std::sqrt(sum);
      } else {
        l[i][j] = sum / l[j][j];
      }
    }
  }
  return l;
}

Mat make_correlation_cholesky() {
  Mat corr{};
  for (int i = 0; i < kDim; ++i) {
    for (int j = 0; j < kDim; ++j) {
      corr[i][j] = (i == j) ? 1.0 : 0.25;
    }
  }
  return cholesky_lower(corr);
}

// Templated on a callable `nrm()` returning one iid standard normal, so the
// same numerics can be driven by the harness's shared global RNG.
template <typename NormalFn>
Vec correlated_standard_normals(const Mat& chol, NormalFn& nrm) {
  Vec z{};
  Vec y{};
  for (int i = 0; i < kDim; ++i) {
    y[i] = nrm();
  }
  for (int i = 0; i < kDim; ++i) {
    double s = 0.0;
    for (int j = 0; j <= i; ++j) {
      s += chol[i][j] * y[j];
    }
    z[i] = s;
  }
  return z;
}

inline double basket_average(const Vec& s) {
  double acc = 0.0;
  for (int i = 0; i < kDim; ++i) {
    acc += kWeights[i] * s[i];
  }
  return acc;
}

inline void milstein_step(Vec& s, const Vec& d_w, double h) {
  const double r_h = kR * h;
  for (int i = 0; i < kDim; ++i) {
    const double sigma = kSigma[i];
    const double x = s[i];
    s[i] = x + r_h * x + sigma * x * d_w[i] + 0.5 * sigma * sigma * x * (d_w[i] * d_w[i] - h);
  }
}

template <typename NormalFn>
inline Vec scaled_correlated_increment(const Mat& chol, double scale, NormalFn& nrm) {
  Vec dw{};
  if (scale == 0.0) {
    return dw;
  }
  const Vec z = correlated_standard_normals(chol, nrm);
  for (int i = 0; i < kDim; ++i) {
    dw[i] = scale * z[i];
  }
  return dw;
}

// Basket-weighted dot product: sum_i w_i a_i b_i. The Asian time-average is on
// the *weighted* basket B = sum_i w_i S_i, so its bridge correction must carry the
// same weights -- matching Giles' `alf'*(vf.*dIf)`. (A plain unweighted dot here was
// a bug: it made the bridge term dim× too large and destroyed the low-level coupling.)
inline double weighted_dot(const Vec& a, const Vec& b) {
  double acc = 0.0;
  for (int i = 0; i < kDim; ++i) {
    acc += kWeights[i] * a[i] * b[i];
  }
  return acc;
}

inline Vec sigma_times_state(const Vec& s) {
  Vec v{};
  for (int i = 0; i < kDim; ++i) {
    v[i] = kSigma[i] * s[i];
  }
  return v;
}

// ---------------------------------------------------------------------------
// basket_path: single-path MLMC correction estimator (the numerical core).
//
// Simulates ONE coupled fine/coarse path at refinement `level` and returns the
// discounted MLMC correction y and the discounted fine payoff pf:
//
//   y  = exp(-rT) * (Pf - Pc)        for level > 0   (Pc=0 / y=disc*Pf on l=0)
//   pf = exp(-rT) *  Pf
//
// where Pf is the payoff on the fine grid (nf = 2^level steps) and Pc the
// payoff on the coarse grid (nc = nf/2 steps). The coarse step is driven by the
// SUM of the two fine Brownian increments (d_w1 + d_w2), with a bridge
// correction for the Asian time-average integral -- this shared driving
// randomness is what makes Var(Pf - Pc) decay with level.
//
// Templated on a callable `nrm()` -> one iid standard normal, so the caller
// supplies the RNG stream (the harness's global next_normal() for the MLMC
// driver). All accumulation is in double.
//
//   option = 1 -> European (payoff on terminal basket average)
//   option = 2 -> Asian    (payoff on time-averaged basket, trapezoidal + bridge)
template <typename NormalFn>
inline void basket_path(int level, int option, const Mat& chol,
                        NormalFn nrm, double& y_out, double& pf_out) {
  const int nf = 1 << level;
  const int nc = (level == 0) ? 0 : (nf >> 1);
  const double hf = kT / static_cast<double>(nf);
  const double hc = (level == 0) ? 0.0 : kT / static_cast<double>(nc);
  const double disc = std::exp(-kR * kT);

  Vec xf{};
  Vec xc{};
  xf.fill(kK);
  xc = xf;

  double af = 0.5 * hf * basket_average(xf);
  double ac = 0.5 * hc * basket_average(xc);

  double pf = 0.0;
  double pc = 0.0;

  if (level == 0) {
    // base MLMC level: a single fine-only Milstein step (nf = 1), no coarse
    // partner (Pc = 0).
    for (int n = 0; n < nf; ++n) {
      const Vec d_w = scaled_correlated_increment(chol, std::sqrt(hf), nrm);
      const Vec d_i = scaled_correlated_increment(chol, std::sqrt(hf / 12.0) * hf, nrm);

      const Vec xf0 = xf;
      const Vec vf = sigma_times_state(xf0);
      milstein_step(xf, d_w, hf);

      if (option == 2) {
        af += hf * basket_average(xf) + weighted_dot(vf, d_i);
      }
    }

    if (option == 2) {
      af -= 0.5 * hf * basket_average(xf);   // trapezoidal endpoint correction
    }

    pf = (option == 1) ? std::max(0.0, basket_average(xf) - kK)
                       : std::max(0.0, af - kK);
    pc = 0.0;
  } else {
    for (int n = 0; n < nc; ++n) {
      const Vec d_w1 = scaled_correlated_increment(chol, std::sqrt(hf), nrm);
      const Vec d_i1 = scaled_correlated_increment(chol, std::sqrt(hf / 12.0) * hf, nrm);

      const Vec xf0 = xf;
      const Vec vf1 = sigma_times_state(xf0);
      milstein_step(xf, d_w1, hf);
      if (option == 2) {
        af += hf * basket_average(xf) + weighted_dot(vf1, d_i1);
      }

      Vec d_wc = d_w1;
      Vec dd_w = d_w1;
      Vec d_ic = d_i1;

      const Vec d_w2 = scaled_correlated_increment(chol, std::sqrt(hf), nrm);
      const Vec d_i2 = scaled_correlated_increment(chol, std::sqrt(hf / 12.0) * hf, nrm);

      const Vec xf1 = xf;
      const Vec vf2 = sigma_times_state(xf1);
      milstein_step(xf, d_w2, hf);
      if (option == 2) {
        af += hf * basket_average(xf) + weighted_dot(vf2, d_i2);
      }

      for (int i = 0; i < kDim; ++i) {
        d_wc[i] += d_w2[i];
        dd_w[i] -= d_w2[i];
        d_ic[i] += d_i2[i];
      }

      const Vec xc0 = xc;
      const Vec vc = sigma_times_state(xc0);
      milstein_step(xc, d_wc, hc);

      if (option == 2) {
        Vec bridge_term{};
        for (int i = 0; i < kDim; ++i) {
          bridge_term[i] = d_ic[i] + 0.25 * hc * dd_w[i];
        }
        ac += hc * basket_average(xc) + weighted_dot(vc, bridge_term);
      }
    }

    if (option == 2) {
      af -= 0.5 * hf * basket_average(xf);
      ac -= 0.5 * hc * basket_average(xc);
    }

    pf = (option == 1) ? std::max(0.0, basket_average(xf) - kK)
                       : std::max(0.0, af - kK);
    pc = (option == 1) ? std::max(0.0, basket_average(xc) - kK)
                       : std::max(0.0, ac - kK);
  }

  double y = disc * (pf - pc);
  if (level == 0) {
    y = disc * pf;
  }

  y_out = y;
  pf_out = disc * pf;
}

}  // namespace basket_mlmc

//
// option = 1 -> Basket European call ; option = 2 -> Basket Asian call
//
int option;

// Level estimator — the mlmc_l contract: void(int, int, double*). Delegates all
// numerics to basket_mlmc::basket_path; this only loops over N paths and packs the
// seven accumulators the harness reads:
//   sums[0]=Sum cost(=nf), [1]=Sum Y, [2]=Sum Y^2, [3]=Sum Y^3, [4]=Sum Y^4,
//   [5]=Sum Pf (disc), [6]=Sum Pf^2,   with Y = disc*(Pf - Pc).
void basket_scalar_l(int l, int N, double *sums)
{
    using basket_mlmc::Mat;
    using basket_mlmc::make_correlation_cholesky;
    using basket_mlmc::basket_path;

    // Cholesky factor of the correlation matrix: built once, read-only,
    // safe to share across OpenMP threads (C++11 thread-safe static init).
    static const Mat chol = make_correlation_cholesky();

    const int nf = 1 << l;            // cost proxy: number of fine timesteps

    for (int k = 0; k < 7; ++k) sums[k] = 0.0;

#pragma omp parallel for reduction(+:sums[0:7])
    for (int np = 0; np < N; ++np) {
        // standard normals from the shared global stream (mlmc_rng.cpp)
        auto nrm = []() -> double { return (double) next_normal(); };

        double y = 0.0, pf = 0.0;
        basket_path(l, option, chol, nrm, y, pf);

        sums[0] += nf;
        sums[1] += y;
        sums[2] += y * y;
        sums[3] += y * y * y;
        sums[4] += y * y * y * y;
        sums[5] += pf;
        sums[6] += pf * pf;
    }
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

        std::sprintf(filename, "basket_scalar_%d.txt", option);
        fp = std::fopen(filename, "w");
        if (!fp) {
            std::perror("fopen");
            return EXIT_FAILURE;
        }

        if (option == 1) {
            std::printf("\n ---- Basket European call ----\n");
        }
        else {
            std::printf("\n ---- Basket Asian call ----\n");
        }
        N = 50000;   // match Giles' reference convergence-test sample count
        L = 8;
        float Eps2[] = { 0.01f, 0.02f, 0.05f, 0.1f, 0.2f, 0.0f };   // Giles' reference Eps
        std::memcpy(Eps, Eps2, sizeof(Eps2));

        // convergence + complexity tables (alpha/beta/gamma) for mlmc_plot_python.py
        mlmc_test(basket_scalar_l, N, L, N0, Eps, Lmin, Lmax, fp);
        std::fclose(fp);

#ifdef _OPENMP
        std::printf(" execution time = %f s\n", omp_get_wtime() - wtime);
        wtime = omp_get_wtime();
#endif

#pragma omp parallel
        rng_termination();

        // No closed form for either basket payoff -> exact value unknown.
        float val = nanf("");

        // now do 100 MLMC calcs
#pragma omp parallel
        rng_initialisation();

        std::sprintf(filename, "basket_scalar_%d_100.txt", option);
        fp = std::fopen(filename, "w");
        if (!fp) {
            std::perror("fopen");
            return EXIT_FAILURE;
        }

        mlmc_test_100(basket_scalar_l, val, N0, Eps, Lmin, Lmax, fp);
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
