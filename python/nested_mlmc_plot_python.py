"""
nested_mlmc_plot_python.py — convergence + complexity plots for nested MLMC output.

The nested MLMC test harness (nested_mlmc_test.cpp) produces a convergence table
with 2L+2 super-levels alternating between:
  even l=2k  : float Milstein MLMC correction at grid level k  (nf = 2^k, cost = nf)
  odd  l=2k+1: double-minus-float precision correction at grid k (cost = 10*nf)

The standard mlmc_plot_python.py plots all super-levels on the same axis, producing
a sawtooth because odd-level variances hit the 1e-10 floor. This script splits the
two streams and plots them correctly.

Usage:
  MPLBACKEND=Agg python3 python/nested_mlmc_plot_python.py outputs/nested_scalar_1
  MPLBACKEND=Agg python3 python/nested_mlmc_plot_python.py outputs/nested_basket_2
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Human-readable label for a nested-MLMC output basename.
#
# Filename conventions covered:
#   nested_scalar_<mode>_[omp_]<opt>            (scalar: 1=Asian)
#   nested_basket_<mode>_[omp_]<opt>            (basket: 1=European, 2=Asian)
#   nested_scalar_fp16_kahan_avx512_<opt>       (AVX-512 serial)
#   nested_scalar_fp16_kahan_omp_avx512_<opt>   (AVX-512 + OpenMP)
# ---------------------------------------------------------------------------

_PAYOFF_NAMES = {
    ("scalar", "1"): "Scalar Asian Call",
    ("basket", "1"): "Basket European Call",
    ("basket", "2"): "Basket Asian Call",
}

_SCHEME_LABELS = [
    ("nokahan", "fp16 no-Kahan"),
    ("fp16_kahan", "fp16+Kahan"),
    ("kahan", "fp16+Kahan"),
    ("fp16", "fp32/fp16"),
]


def describe_run(base):
    """Return (payoff_label, scheme_label, precision_label, is_avx512, is_omp) for a basename."""
    name = os.path.basename(base)

    domain = "scalar" if "scalar" in name else ("basket" if "basket" in name else None)
    opt = name.rsplit("_", 1)[-1]
    payoff = _PAYOFF_NAMES.get((domain, opt), name)

    scheme = ""
    for key, label in _SCHEME_LABELS:
        if key in name:
            scheme = label
            break

    # Detect precision and mode: standard fp32/fp64, adaptive, or fp16 variants
    if "adaptive" in name and "fp16" in name:
        # Adaptive fp16 variant (from nested_scalar_milstein_fp16.cpp)
        precision = "Adaptive fp32/fp16"
        mode = "Adaptive"
    elif "adaptive" in name:
        # Adaptive fp32/fp64 variant (from nested_scalar_milstein.cpp dual-mode)
        precision = "Adaptive fp32/fp64"
        mode = "Adaptive"
    elif "nokahan" in name:
        # Plain fp16 accumulation, no Kahan compensation
        precision = "fp32/fp16"
        mode = "fp16 no-Kahan"
    elif "fp16" in name or "kahan" in name:
        # Pure fp16 variants
        precision = "fp32/fp16"
        mode = "fp16"
    elif "standard" in name:
        # Plain standard fp32/fp64 (from nested_scalar_milstein.cpp plain mode)
        precision = "Standard fp32/fp64"
        mode = "Standard"
    else:
        # Fallback: plain nested_scalar_* or nested_basket_* without mode suffix
        precision = "Standard fp32/fp64"
        mode = "Standard"

    is_avx512 = "avx512" in name
    is_omp = "_omp_" in name or name.endswith("_omp")

    return payoff, scheme, precision, mode, is_avx512, is_omp


# ---------------------------------------------------------------------------
# File parsing (identical format to mlmc_test output)
# ---------------------------------------------------------------------------

def _read_lines(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _find_line(lines, predicate, start=0):
    for i in range(start, len(lines)):
        if predicate(lines[i]):
            return i
    raise ValueError("Required marker line not found.")


def _parse_file(base):
    lines = _read_lines(base + ".txt")

    # Version / N header
    i = _find_line(lines, lambda s: s[:4] == "*** ")
    j = _find_line(lines, lambda s: s[:9] == "*** using", start=i + 1)
    ints = [int(x) for x in lines[j].split() if x.lstrip("-").isdigit()]
    N = ints[0] if ints else 0

    # Convergence table (all super-levels)
    k = _find_line(lines, lambda s: len(s) > 0 and s[0] == "-", start=i + 1)
    row = k + 1
    cols = [[] for _ in range(7)]   # del1 del2 var1 var2 kur1 chk1 cost

    while row < len(lines) and len(lines[row]) > 10:
        data = np.fromstring(lines[row], sep=" ")
        if data.size < 8:
            break
        for c in range(7):
            cols[c].append(data[c + 1])
        row += 1

    del1, del2, var1, var2, kur1, chk1, cost = (np.array(c) for c in cols)

    # Regression estimates (grab from text)
    alpha_est = beta_est = gamma_est = np.nan
    for ln in lines[row:]:
        if "alpha" in ln and "=" in ln:
            try:
                alpha_est = float(ln.split("=")[1].split()[0])
            except Exception:
                pass
        if "beta" in ln and "=" in ln:
            try:
                beta_est = float(ln.split("=")[1].split()[0])
            except Exception:
                pass
        if "gamma" in ln and "=" in ln:
            try:
                gamma_est = float(ln.split("=")[1].split()[0])
            except Exception:
                pass

    # Complexity table -- absent when the run used mlmc_test_noconv (no
    # complexity section at all); fall back to empty arrays in that case
    # instead of erroring, since panels 1-3 don't need it.
    Eps, mlmc_cost, std_cost = [], [], []
    ls_cols, Nls_cols = [], []
    try:
        m = _find_line(lines, lambda s: len(s) > 0 and s[0] == "-", start=row)
        row = m + 1
        while row < len(lines) and len(lines[row]) > 10:
            data = np.fromstring(lines[row], sep=" ")
            if data.size < 6:
                break
            Eps.append(data[0])
            mlmc_cost.append(data[2])
            std_cost.append(data[3])
            n_lev = len(data) - 5
            ls_cols.append(np.arange(n_lev))
            Nls_cols.append(data[5:])
            row += 1
    except ValueError:
        pass

    Eps = np.array(Eps)
    mlmc_cost = np.array(mlmc_cost)
    std_cost = np.array(std_cost)

    n_eps = len(Eps)
    max_lev = max((len(c) for c in Nls_cols), default=0)
    ls_mat = np.full((max_lev, n_eps), np.nan)
    Nls_mat = np.full((max_lev, n_eps), np.nan)
    for j in range(n_eps):
        lv = len(Nls_cols[j])
        ls_mat[:lv, j] = ls_cols[j]
        Nls_mat[:lv, j] = Nls_cols[j]

    return dict(
        N=N,
        del1=del1, del2=del2, var1=var1, var2=var2,
        kur1=kur1, chk1=chk1, cost=cost,
        alpha=alpha_est, beta=beta_est, gamma=gamma_est,
        Eps=Eps, mlmc_cost=mlmc_cost, std_cost=std_cost,
        ls_mat=ls_mat, Nls_mat=Nls_mat,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def nested_mlmc_plot(base):
    d = _parse_file(base)
    payoff_label, scheme_label, _, _, _, _ = describe_run(base)
    is_adaptive_scheme = "Adaptive" in scheme_label

    del1, del2 = d["del1"], d["del2"]
    var1, var2 = d["var1"], d["var2"]
    kur1 = d["kur1"]
    cost = d["cost"]
    alpha_est, beta_est = d["alpha"], d["beta"]
    Eps = d["Eps"]
    mlmc_cost, std_cost = d["mlmc_cost"], d["std_cost"]
    ls_mat, Nls_mat = d["ls_mat"], d["Nls_mat"]

    n_super = len(var1)             # 2L+2 total super-levels
    L = (n_super - 2) // 2         # number of grid levels (k = 0..L)

    even = np.arange(0, n_super, 2)   # float MLMC corrections
    odd  = np.arange(1, n_super, 2)   # precision corrections

    k_ev = even // 2   # grid level k for each even super-level (0..L)
    k_od = odd  // 2   # grid level k for each odd super-level  (0..L)

    # Quantities on even super-levels (standard MLMC interpretation)
    ev_var1 = var1[even]
    ev_var2 = var2[even]
    ev_del1 = del1[even]
    ev_del2 = del2[even]
    ev_kur1 = kur1[even]

    # Quantities on odd super-levels (precision corrections)
    od_var1 = var1[odd]   # should be ~1e-10 (float ≈ double)

    # ----------------------------------------------------------------
    # No complexity section (e.g. mlmc_test_noconv output) -> drop panels
    # 5-6 entirely and use a 2x2 layout instead of 3x2.
    has_complexity = len(Eps) > 0
    if has_complexity:
        fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.rcParams.update({"font.size": 10})

    # ---- Panel 1: Coupling + option variance (even super-levels) ----
    ax = axes[0, 0]
    ax.plot(k_ev,      np.log2(ev_var2),      "b-o", ms=6, label=r"$\mathrm{Var}(P_\ell)$")
    ax.plot(k_ev[1:],  np.log2(ev_var1[1:]),  "r-s", ms=6, label=r"$\mathrm{Var}(P_\ell - P_{\ell-1})$")

    # Reference slope −β
    if L >= 2 and not np.isnan(beta_est):
        kk = np.array([k_ev[1], k_ev[-1]], dtype=float)
        y0 = np.log2(ev_var1[1])
        ax.plot(kk, y0 - beta_est * (kk - kk[0]), "k--", lw=1.2,
                label=rf"slope $-\hat\beta = -{beta_est:.2f}$")

    ax.set_xlabel("grid level $k$")
    ax.set_ylabel(r"$\log_2$ variance")
    ax.set_title("Variance decay — float Milstein levels")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Mean magnitude (even super-levels) ----
    ax = axes[0, 1]
    ax.plot(k_ev,     np.log2(np.abs(ev_del2)),      "b-o", ms=6, label=r"$|E[P_\ell]|$")
    ax.plot(k_ev[1:], np.log2(np.abs(ev_del1[1:])),  "r-s", ms=6,
            label=r"$|E[P_\ell - P_{\ell-1}]|$")

    if L >= 2 and not np.isnan(alpha_est):
        kk = np.array([k_ev[1], k_ev[-1]], dtype=float)
        y0 = np.log2(np.abs(ev_del1[1]))
        ax.plot(kk, y0 - alpha_est * (kk - kk[0]), "k--", lw=1.2,
                label=rf"slope $-\hat\alpha = -{alpha_est:.2f}$")

    ax.set_xlabel("grid level $k$")
    ax.set_ylabel(r"$\log_2 |\mathrm{mean}|$")
    ax.set_title("Mean decay — float Milstein levels")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Precision correction variance (odd super-levels) ----
    # Cutoff-detection (a genuine, discrete adaptive l* where the correction is
    # exactly skipped by construction) only makes sense for the Adaptive scheme.
    # Kahan-type schemes are compensated at EVERY level -- there is no cutoff,
    # and the correction sits near the harness floor throughout, subject to
    # ordinary floating-point/ULP noise. Misapplying the cutoff heuristic there
    # both hides the real (near-floor) data and draws a false "adaptive l*"
    # marker, so we only run it when the filename says this is Adaptive.
    od_del1 = del1[odd]

    ax = axes[1, 0]
    adaptive_cutoff = False
    if is_adaptive_scheme:
        zeroed = (np.abs(od_del1) < 1e-15) & (np.abs(od_var1 - 1e-10) < 1e-15)
        active = ~zeroed
        if active.any():
            ax.plot(k_od[active], np.log2(np.maximum(od_var1[active], 1e-15)),
                    "g-^", ms=6, label=r"$\mathrm{Var}(\delta_{\mathrm{prec}})$")
        if zeroed.any():
            l_star_k = int(k_od[zeroed][0])
            ax.axvline(l_star_k - 0.5, color="darkorange", ls=":", lw=1.8,
                       label=rf"adaptive $\ell^*={l_star_k}$")
            adaptive_cutoff = True
    else:
        # Kahan (or any non-adaptive scheme): always plot the real values —
        # they should sit essentially flat at the floor across every level.
        ax.plot(k_od, np.log2(np.maximum(od_var1, 1e-15)),
                "g-^", ms=6, label=r"$\mathrm{Var}(\delta_{\mathrm{prec}})$")
    ax.axhline(np.log2(1e-10), color="gray", ls="--", lw=1, label="floor 1e-10")

    ax.set_xlabel("grid level $k$")
    ax.set_ylabel(r"$\log_2$ variance")
    if adaptive_cutoff:
        ax.set_title(f"Precision corr. variance — odd super-levels\n"
                     rf"(fp16/fp32 below $\ell^*={l_star_k}$; pure fp32 above)")
    elif is_adaptive_scheme:
        ax.set_title("Precision correction variance — odd super-levels\n(near floor ⇒ float ≈ double)")
    else:
        ax.set_title("Precision correction variance — odd super-levels\n"
                      "(Kahan: real, bounded correction — decays, then floors at fp16 resolution)")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 4: Kurtosis (even super-levels, l≥1) ----
    ax = axes[1, 1]
    ax.plot(k_ev[1:], ev_kur1[1:], "b-o", ms=6)
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("grid level $k$")
    ax.set_ylabel("kurtosis")
    ax.set_title("Kurtosis — float Milstein levels")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    if has_complexity:
        # ---- Panel 5: Sample allocation (all super-levels) ----
        ax = axes[2, 0]
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        legend_done = False
        for j in range(len(Eps)):
            col = colors[j % len(colors)]
            sl = ls_mat[:, j]
            nl = Nls_mat[:, j]
            valid = ~np.isnan(nl) & (nl > 0)
            sl = sl[valid].astype(int)
            nl = nl[valid]

            ev_m = sl % 2 == 0
            od_m = sl % 2 == 1

            label_ev = f"ε={Eps[j]:.4g}  (float MLMC)" if not legend_done else ""
            label_od = f"ε={Eps[j]:.4g}  (precision corr.)" if not legend_done else ""

            ax.semilogy(sl[ev_m], nl[ev_m], "o-",  color=col, lw=1.2,
                        label=f"ε={Eps[j]:.4g}" if not legend_done else "")
            ax.semilogy(sl[od_m], nl[od_m], "s--", color=col, lw=1.0)
            legend_done = True   # only label once

        # Custom legend patches
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", ls="-",  color="gray", label="even $\\ell$: float MLMC corr."),
            Line2D([0], [0], marker="s", ls="--", color="gray", label="odd $\\ell$: precision corr."),
        ]
        for j in range(len(Eps)):
            col = colors[j % len(colors)]
            handles.append(Line2D([0], [0], color=col, lw=2, label=f"ε={Eps[j]:.4g}"))

        ax.set_xlabel(r"super-level $\ell$")
        ax.set_ylabel(r"$N_\ell$")
        ax.set_title("Sample allocation (solid=even/MLMC, dashed=odd/precision)")
        ax.legend(handles=handles, fontsize=7, loc="upper right", ncol=2)
        ax.grid(True, which="both", alpha=0.3)

        # ---- Panel 6: Cost complexity ----
        ax = axes[2, 1]
        ax.loglog(Eps, Eps**2 * std_cost,   "b-o", ms=6, label="Std MC (double)")
        ax.loglog(Eps, Eps**2 * mlmc_cost,  "r-s", ms=6, label="Nested MLMC")
        ax.set_xlabel(r"accuracy $\varepsilon$")
        ax.set_ylabel(r"$\varepsilon^2 \times \mathrm{Cost}$")
        ax.set_title(r"Cost complexity — nested MLMC vs standard MC")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    payoff, scheme, precision, mode, is_avx512, is_omp = describe_run(base)
    tag_bits = [b for b in (scheme, "AVX-512" if is_avx512 else "", "+ OpenMP" if is_omp else "") if b]
    tag = " — " + " ".join(tag_bits) if tag_bits else ""
    domain_label = "Basket" if "basket" in os.path.basename(base) else "Scalar"

    plt.suptitle(
        f"{mode} {domain_label} Nested MLMC ({precision}) — {payoff}{tag}\n"
        rf"$\hat\alpha={alpha_est:.3f}$,  $\hat\beta={beta_est:.3f}$,  $\hat\gamma={d['gamma']:.3f}$"
        "  (regression on even super-levels)",
        fontsize=11, y=1.002,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "outputs/nested_scalar_1"
    fig = nested_mlmc_plot(base)
    out = base + "_convergence.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)
    plt.show()
