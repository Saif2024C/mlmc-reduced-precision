"""
kernel_scaling_plot_python.py -- OpenMP scaling figures for the Level-0 kernel
benchmarks (src/AVX-512/performance_analysis/).

Serves both kernels, which share one --scaling output format:
    outputs/avx512/kernel/basket/basket_l0_scaling_sweep.txt   (basket European)
    outputs/avx512/kernel/scalar/scalar_asian_l0_scaling_sweep.txt (scalar Asian)
The figure title is derived from the input filename, so neither needs a flag.

Input is the output of `make kbench_scaling`, whose wall-clock block is keyed
by eps/N_0:

    Wall clock (ms):
      eps       N_0 | fp32 1thr  fp32 32t | fp16 1thr  fp16 32t | ratio32
    0.200      9704 |    0.7679    0.0350 |    0.2940    0.0210 |  1.67x
    ...

The block is parsed by column position after splitting on '|', so a change to
the widths in the C++ printf is safe but a change to the COLUMN ORDER or the
block header text is not.

The x-axis is N_0 (log scale), not eps: N_0 is what the kernel actually sees,
and it is what the trend is a function of.  eps is annotated on each point so
the mapping back to the driver's tolerance stays visible.

Produces two standalone figures:
  wall clock vs N_0, 1 and 32 threads, both precisions (log-log)
  fp32/fp16 ratio at 1 and 32 threads -- does threading erode the fp16
  advantage?

Usage:
  MPLBACKEND=Agg python3 python/kernel_scaling_plot_python.py \
      outputs/avx512/kernel/basket_l0_scaling_sweep.txt
"""

import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# House style, matching nested_overlay_plot_python.py so every figure in the
# dissertation reads as one set: serif/STIX text to sit with LaTeX body copy,
# hairline axes and grid, no top/right spines, no legend frame, 300 dpi with
# TrueType (not Type-3) fonts embedded for journal submission.
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
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

NTHREADS = 32

# fp32 / fp16 kept visually distinct and colour-blind safe; 1-thread dashed
# and open-faced, 32-thread solid and filled, so precision reads as colour
# and thread count as weight -- the same encoding the overlay plots use.
C32, C16 = "#0072B2", "#D55E00"
MS = 4.2          # marker size
LW = 1.3          # line width, solid (primary) series
LWD = 1.1         # line width, dashed (secondary) series


def parse(path):
    """Pull the wall-clock table out of the benchmark's text output."""
    rows = {}          # n0 -> dict of fields
    order = []         # preserve file order
    block = None

    with open(path) as fh:
        for line in fh:
            if line.startswith("  Wall clock"):
                block = "time"
                continue
            if line.startswith("  Speedup and parallel"):
                block = "skip"
                continue
            if block is None or "|" not in line:
                continue
            # header lines carry no digits in the stub
            stub = line.split("|")[0].strip()
            m = re.match(r"^([\d.]+)\s+(\d+)$", stub)
            if not m:
                continue
            eps, n0 = float(m.group(1)), int(m.group(2))
            parts = [p.strip() for p in line.split("|")]
            if n0 not in rows:
                rows[n0] = {"eps": eps, "n0": n0}
                order.append(n0)
            r = rows[n0]

            if block == "time":
                # parts[1] = "fp32_1thr fp32_32t", parts[2] = fp16 pair
                a = parts[1].split()
                b = parts[2].split()
                r["t32_1"], r["t32_p"] = float(a[0]), float(a[1])
                r["t16_1"], r["t16_p"] = float(b[0]), float(b[1])

    if not order:
        sys.exit(f"error: no data rows parsed from {path} -- has the table "
                 f"format changed?")
    return [rows[n] for n in sorted(order)]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip().splitlines()[-2].strip())
    path = sys.argv[1]
    d = parse(path)

    n0 = [r["n0"] for r in d]

    # Two standalone figures rather than one stacked block: each carries its
    # own full x-axis and title so it can be dropped into a different section
    # of the write-up on its own.  (A shared-x stacked figure labels only the
    # bottom panel, which reads badly once a panel is cropped out of context.)
    base = path.rsplit(".", 1)[0]
    XLABEL = "$N_0$  (level-0 paths requested by the MLMC driver)"
    # Derived from the input filename, not hardcoded: this script serves both
    # the basket European and scalar Asian kernel benchmarks, whose
    # --scaling output shares one format.
    _name = base.rsplit("/", 1)[-1]
    if "scalar" in _name or "asian" in _name:
        TITLE = "Level-0 scalar Asian kernel"
    elif "basket" in _name:
        TITLE = "Level-0 basket European kernel"
    else:
        TITLE = "Level-0 kernel"

    def finish(fig, a, tag, annotate_eps=None, offsets=None):
        a.set_xlabel(XLABEL)
        a.grid(True, which="both", alpha=0.3)
        # Widen the data limits before annotating: the eps labels hang below
        # and to the right of their points, so the first and last would be
        # clipped by the axes frame at the default tight limits.
        if annotate_eps is not None:
            a.set_xmargin(0.10)
            a.autoscale(axis="x")
            if a.get_yscale() == "log":
                lo, hi = a.get_ylim()
                a.set_ylim(lo / 1.6, hi)
            # linear-scale figures here fix their own ylim with headroom at the
            # bottom already, so they are left alone
        # Every point carries the tolerance it came from, on every figure, so
        # the reader never has to hold the N_0 <-> eps mapping in their head.
        #
        # Placement is per-point rather than a single fixed offset: a constant
        # below-right offset collides with the marker wherever the curve turns
        # (the eps=0.05 peak sat on top of its own marker).  `offsets` lets the caller push individual labels to the
        # side that is locally free -- a label that overlaps its own data is
        # worse than no label.
        if annotate_eps is not None:
            for i, (r, y) in enumerate(zip(d, annotate_eps)):
                dx, dy, ha = offsets[i] if offsets else (5, -12, "left")
                a.annotate(f"$\\varepsilon={r['eps']:g}$", xy=(r["n0"], y),
                           xytext=(dx, dy), textcoords="offset points",
                           fontsize=8.5, ha=ha, va="center", color="0.30")
        fig.tight_layout()
        out = f"{base}_{tag}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")

    # ---- figure 1: wall clock ------------------------------------------
    fig, a = plt.subplots(figsize=(7.2, 5.0))
    # 1-thread series drawn open-faced and dashed (secondary), 32-thread solid
    # and filled (primary) -- the reader's eye lands on the parallel result.
    a.loglog(n0, [r["t32_1"] for r in d], "o--", color=C32, ms=MS, lw=LWD,
             mfc="white", mew=0.9, zorder=2, label="fp32, 1 thread")
    a.loglog(n0, [r["t32_p"] for r in d], "o-", color=C32, ms=MS, lw=LW,
             mew=0.6, zorder=3, label=f"fp32, {NTHREADS} threads")
    a.loglog(n0, [r["t16_1"] for r in d], "s--", color=C16, ms=MS, lw=LWD,
             mfc="white", mew=0.9, zorder=2, label="fp16+Kahan, 1 thread")
    a.loglog(n0, [r["t16_p"] for r in d], "s-", color=C16, ms=MS, lw=LW,
             mew=0.6, zorder=3, label=f"fp16+Kahan, {NTHREADS} threads")
    a.set_ylabel("wall clock (ms)")
    a.set_title(f"{TITLE}: OpenMP scaling")
    a.legend(loc="upper left")
    a.grid(True, which="major", color="0.8", alpha=0.9)
    # anchored to the lowest curve; all five sit below it with room
    finish(fig, a, "time", annotate_eps=[r["t16_p"] for r in d])

    # ---- figure 2: fp16 advantage, 1 vs 32 threads ---------------------
    fig, a = plt.subplots(figsize=(7.2, 5.0))
    r1 = [r["t32_1"] / r["t16_1"] for r in d]
    rp = [r["t32_p"] / r["t16_p"] for r in d]
    a.semilogx(n0, r1, "^--", color="0.45", ms=MS, lw=LWD, mfc="white",
               mew=0.9, zorder=2, label="1 thread")
    a.semilogx(n0, rp, "v-", color="0.10", ms=MS, lw=LW, mew=0.6,
               zorder=3, label=f"{NTHREADS} threads")
    a.axhline(1.0, color="0.35", lw=0.9, ls=(0, (1, 2)), zorder=1)
    a.set_ylabel("fp32 / fp16+Kahan   ($>1$: fp16 faster)")
    a.set_title(f"{TITLE}: fp16 speedup (1 and {NTHREADS} threads)")
    a.set_ylim(0, max(max(r1), max(rp)) * 1.25)
    a.legend(loc="lower right")
    a.grid(True, axis="y", which="major", color="0.8", alpha=0.9)
    # eps=0.02 is where the two curves cross, so its label goes below-left to
    # avoid sitting in the intersection.
    finish(fig, a, "ratio", annotate_eps=rp,
           offsets=[(7, -11, "left"),      # 0.2
                    (7, -11, "left"),      # 0.1
                    (7, -11, "left"),      # 0.05
                    (-9, -12, "right"),    # 0.02  (crossover)
                    (-2, -22, "right")])   # 0.01  (last point: the incoming
                                           #        segment rises steeply, so
                                           #        this sits well below it
                                           #        rather than beside it)


if __name__ == "__main__":
    main()
