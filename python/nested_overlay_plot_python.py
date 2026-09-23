"""
nested_overlay_plot_python.py — five-series fp16/fp32 precision overlay.

Reproduces the `*_overlay5_<opt>.png` figures: one axis carrying every
precision variant of a payoff, so the fp16 story is legible in a single plot
rather than spread across four separate convergence figures.

The five series, all Var[Pf-Pc] on a log axis against grid level k:

  even super-levels (l=2k, the Milstein MLMC correction)
    fp16, no Kahan      -- plain += accumulation; the baseline that breaks
    fp16, Kahan         -- compensated summation; bounded, real decay
    Pure fp32           -- the reference; classical beta ~ 2 straight line.
                           This is STANDARD (non-nested) MLMC, read from
                           ../stdmlmc/std_mlmc_fp32_<domain>_<opt>.txt, not a
                           mode of the nested sweep: with a single precision
                           there is no precision correction to make, so the
                           odd super-levels do not exist for it.

  odd super-levels (l=2k+1, the fp32-minus-fp16 precision correction)
    fp32-fp16 gap, no Kahan
    fp32-fp16 gap, Kahan

Reading it: where the two fp16 even-level curves peel away from the green
fp32 line is where fp16 rounding starts to dominate the discretisation error.
no-Kahan turns upward (variance GROWING with level -- the breakdown); Kahan
flattens to a floor instead (bounded cost, not divergence).  The dotted
vertical line is the adaptive scheme's cutoff k*, chosen to sit at the last
level before that departure matters.

Usage:
    MPLBACKEND=Agg python3 python/nested_overlay_plot_python.py <dir> <domain> <opt> [k_star]

    <dir>     directory holding the four nested_<domain>_fp16_avx512_<mode>_<opt>.txt
    <domain>  scalar | basket
    <opt>     1 | 2   (scalar: 1=Asian; basket: 1=European 2=Asian)
    [k_star]  cutoff level for the dotted line; default 5 scalar-1 / 6 otherwise

Writes <dir>/nested_<domain>_fp16_avx512_overlay5_<opt>.png
"""

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Publication styling.
#
# STIX is matplotlib's bundled Times/Computer-Modern-alike; using it for both
# text and mathtext means the axis labels and the $\mathrm{Var}[P_f-P_c]$
# expression share one typeface, which is what makes a figure read as
# typeset rather than plotted.  No LaTeX dependency (no `latex` on PATH here),
# so this renders identically on any machine.
#
# Weights are deliberately light: 0.5pt frame, 0.35pt grid.  In print, a
# figure is reduced to column width, and anything heavier than the data lines
# competes with them.  Only the left/bottom spines are kept -- the top/right
# pair adds a box that carries no information.
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset":  "stix",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     12,
    "legend.fontsize":    9.5,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "axes.linewidth":     0.5,
    "grid.linewidth":     0.35,
    "lines.linewidth":    1.3,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.minor.width":  0.4,
    "ytick.minor.width":  0.4,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          False,
    "ytick.right":        False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "legend.frameon":     False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "pdf.fonttype":       42,   # embed TrueType, not Type-3: required by most
    "ps.fonttype":        42,   # journals and keeps text selectable in the PDF
})

# adaptive-scheme cutoffs, per CLAUDE.md: scalar Asian 5, everything else 6
DEFAULT_KSTAR = {("scalar", 1): 5, ("scalar", 2): 6,
                 ("basket", 1): 6, ("basket", 2): 6}

PAYOFF_NAME = {("scalar", 1): "Scalar Asian",
               ("basket", 1): "Basket European",  ("basket", 2): "Basket Asian"}


def read_levels(path):
    """Parse the mlmc_test convergence table.

    Returns (level, var_PfPc) arrays over ALL super-levels; splitting into
    even/odd is left to the caller since the two mean different things.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]

    # table starts after the dashed rule following the column header
    k = next(i for i, l in enumerate(lines) if l and l[0] == "-")

    rows = []
    for l in lines[k + 1:]:
        if len(l) <= 10:
            break
        data = np.fromstring(l, sep=" ")
        if data.size < 8:
            break
        rows.append(data)

    rows = np.array(rows)
    # cols: l, ave(Pf-Pc), ave(Pf), var(Pf-Pc), var(Pf), kurtosis, check, cost
    return rows[:, 0].astype(int), rows[:, 3]


def sample_count(path):
    """N from the '*** using N = ... samples' banner, for the title."""
    with open(path, "r", encoding="utf-8") as f:
        for l in f:
            if l.startswith("*** using"):
                ints = [int(x) for x in l.split() if x.isdigit()]
                if ints:
                    return ints[0]
    return None


def even_series(lev, var):
    """Even super-levels l=2k -> grid level k (the Milstein correction)."""
    m = (lev % 2 == 0)
    return lev[m] // 2, var[m]


def odd_series(lev, var):
    """Odd super-levels l=2k+1 -> grid level k (the precision correction)."""
    m = (lev % 2 == 1)
    return (lev[m] - 1) // 2, var[m]


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    d, domain, opt = sys.argv[1], sys.argv[2], int(sys.argv[3])
    kstar = int(sys.argv[4]) if len(sys.argv) > 4 \
        else DEFAULT_KSTAR.get((domain, opt), 6)

    def path(mode):
        return os.path.join(
            d, f"nested_{domain}_fp16_avx512_{mode}_{opt}.txt")

    # The pure-fp32 reference is STANDARD (non-nested) MLMC, produced by
    # std_mlmc_fp32_{domain}_avx512.cpp, so its level column is already the
    # grid level k rather than a super-level index.  It lives in its own
    # directory since it is not part of the nested sweep.
    std_path = os.path.join(
        os.path.dirname(d.rstrip("/")), "stdmlmc",
        f"std_mlmc_fp32_{domain}_{opt}.txt")

    modes = ["nokahan", "kahan"]
    missing = [m for m in modes if not os.path.exists(path(m))]
    if not os.path.exists(std_path):
        missing.append(f"puref32 ({std_path})")
    if missing:
        print(f"error: missing required file(s) for mode(s): {', '.join(missing)}")
        print(f"       looked in {d}")
        sys.exit(1)

    data = {m: read_levels(path(m)) for m in modes}
    # Re-index k -> super-level 2k so even_series() recovers k unchanged, and
    # odd_series() finds nothing: non-nested MLMC has no precision correction.
    std_lev, std_var = read_levels(std_path)
    data["puref32"] = (2 * std_lev, std_var)
    modes = ["nokahan", "kahan", "puref32"]
    N = sample_count(path("kahan"))

    # single-column journal proportions (~golden ratio); scales down cleanly
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    # Okabe-Ito derived palette: distinguishable in greyscale and to the most
    # common colour-vision deficiencies, unlike matplotlib's default cycle.
    C_NOKAHAN = "#c1272d"   # red
    C_KAHAN   = "#e69f00"   # orange
    C_FP32    = "#0f7b3e"   # green
    C_GAP_NK  = "#8c564b"   # brown
    C_GAP_K   = "#7048a8"   # purple

    # --- even levels: the MLMC correction variance, one curve per precision --
    # Filled markers = the estimator's own variance.
    style = {
        "nokahan": dict(color=C_NOKAHAN, marker="o", ls="-",
                        label=r"fp16, no Kahan"),
        "kahan":   dict(color=C_KAHAN,   marker="s", ls="-",
                        label=r"fp16, Kahan"),
        "puref32": dict(color=C_FP32,    marker="D", ls="-",
                        label=r"pure fp32 (non-nested MLMC)"),
    }
    for m in modes:
        k, v = even_series(*data[m])
        ax.semilogy(k, v, ms=4.2, lw=1.3, mew=0.6, zorder=3, **style[m])

    # --- odd levels: the fp32-minus-fp16 precision-correction variance -------
    # Hollow markers + dashes: a different quantity from the curves above, so
    # it should not read as another member of the same family.
    gap_style = {
        "nokahan": dict(color=C_GAP_NK, marker="^", ls=(0, (4, 1.6)),
                        label=r"fp32$-$fp16 (precision correction), no Kahan"),
        "kahan":   dict(color=C_GAP_K,  marker="v", ls=(0, (4, 1.6)),
                        label=r"fp32$-$fp16 (precision correction), Kahan"),
    }
    for m in ["nokahan", "kahan"]:
        k, v = odd_series(*data[m])
        ax.semilogy(k, v, ms=4.2, lw=1.1, mfc="white", mew=0.9, zorder=2,
                    **gap_style[m])

    ax.axvline(kstar, color="0.35", ls=(0, (1, 2)), lw=0.9, zorder=1,
               label=rf"cutoff $k^{{*}}={kstar}$")

    ax.set_xlabel(r"grid level $k$")
    # Two different quantities share this axis: the even super-levels carry the
    # MLMC correction Var[Pf-Pc], the odd ones the precision correction
    # Var[P^fp32 - P^fp16].  Both are per-level variances, so the generic label
    # is the honest one -- the legend says which curve is which.
    ax.set_ylabel(r"level variance")

    title = (f"{PAYOFF_NAME.get((domain, opt), domain)}: "
             f"AVX-512 fp16/fp32 precision comparison")
    if N:
        title += rf"  ($N = {N:,}$)"
    ax.set_title(title, pad=8)

    # horizontal-only grid: the y decades are what a reader compares against,
    # while x is a small integer axis that needs no rules.
    ax.grid(True, axis="y", which="major", color="0.8", alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_xlim(left=-0.3)
    ax.set_xticks(np.arange(0, max(even_series(*data["puref32"])[0]) + 1, 2))

    ax.legend(loc="lower left", handlelength=2.2, labelspacing=0.3,
              borderaxespad=0.4, handletextpad=0.7)

    fig.tight_layout()
    stem = os.path.join(d, f"nested_{domain}_fp16_avx512_overlay5_{opt}")
    fig.savefig(stem + ".png")
    fig.savefig(stem + ".pdf")   # vector copy for LaTeX \includegraphics
    plt.close(fig)
    print(f"wrote {stem}.png and .pdf")


if __name__ == "__main__":
    main()
