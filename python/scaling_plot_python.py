"""
scaling_plot_python.py -- strong-scaling figures for fp16-Kahan / Adaptive /
pure-fp32 AVX-512 OMP estimators (see src/AVX-512/Makefile's
`scaling_csv`/`plot` targets).

All three designs are plotted together everywhere: fp16-Kahan is the slower
precision-cost baseline (beta~1-1.4) that motivates the Adaptive cutoff, and
dropping it makes a 3-design comparison look like a 2-horse race -- the
Kahan curve IS the evidence for why Adaptive's cutoff exists.

Input: one CSV per (payoff, design) with two columns, no header:
    threads,seconds
e.g. outputs/avx512/scaling/scalar_adaptive_opt1.csv:
    1,1.34
    2,0.72
    4,0.42
    ...

Produces, per payoff, a two-panel figure:
  top:    runtime vs threads, log2 x-axis, one curve each for Kahan / Adaptive / fp32
  bottom: speedup (fp32_time / design_time, one curve per non-fp32 design) and
          OpenMP parallel efficiency E(p) = (T_1 / (p * T_p)) * 100%, one curve
          per design, computed relative to that same design's own 1-thread
          time (the standard definition -- NOT relative to another design).

Usage:
  MPLBACKEND=Agg python3 python/scaling_plot_python.py \
      outputs/avx512/scaling/scalar_kahan_opt1.csv \
      outputs/avx512/scaling/scalar_adaptive_opt1.csv \
      outputs/avx512/scaling/scalar_fp32_opt1.csv \
      "Scalar Asian (option 1)" \
      outputs/avx512/scaling/scalar_opt1

  # combined efficiency-only figure across payoffs, one curve per
  # (payoff x design) combination:
  MPLBACKEND=Agg python3 python/scaling_plot_python.py --efficiency-combo \
      outputs/avx512/scaling/scalar_kahan_opt1.csv    Scalar-Kahan \
      outputs/avx512/scaling/scalar_adaptive_opt1.csv Scalar-Adaptive \
      outputs/avx512/scaling/scalar_fp32_opt1.csv     Scalar-fp32 \
      outputs/avx512/scaling/basket_kahan_opt1.csv    Basket-Kahan \
      outputs/avx512/scaling/basket_adaptive_opt1.csv Basket-Adaptive \
      outputs/avx512/scaling/basket_fp32_opt1.csv     Basket-fp32 \
      outputs/avx512/scaling/combo_opt1
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

# dataviz skill categorical slots -- fixed assignment, not cycled, so design
# identity reads the same across every figure: Kahan=green, Adaptive=blue,
# fp32=red.
COLOR_KAHAN    = "#1baf7a"
COLOR_ADAPTIVE = "#2a78d6"
COLOR_FP32     = "#e34948"
DESIGN_STYLE = {
    "kahan":    dict(color=COLOR_KAHAN,    marker="^", linestyle="-."),
    "adaptive": dict(color=COLOR_ADAPTIVE, marker="o", linestyle="-"),
    "fp32":     dict(color=COLOR_FP32,     marker="s", linestyle="--"),
}
DESIGN_LABEL = {"kahan": "fp16 Kahan", "adaptive": "Adaptive", "fp32": "fp32"}
COMBO_COLORS = ["#1baf7a", "#2a78d6", "#e34948", "#0b7a52", "#17518f", "#a83231"]


def read_csv(path):
    threads, secs = [], []
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            threads.append(int(row[0]))
            secs.append(float(row[1]))
    order = np.argsort(threads)
    return np.array(threads)[order], np.array(secs)[order]


def efficiency(threads, secs):
    t1 = secs[threads == 1][0]
    return 100.0 * t1 / (threads * secs)


def two_panel(kahan_csv, adaptive_csv, fp32_csv, payoff_label, out_base):
    designs = [
        ("kahan", read_csv(kahan_csv)),
        ("adaptive", read_csv(adaptive_csv)),
        ("fp32", read_csv(fp32_csv)),
    ]
    t_f, s_f = designs[2][1]  # fp32 is the speedup denominator

    # Three stacked single-axis panels -- never a dual-axis plot (speedup and
    # efficiency are different quantities with different scales; sharing one
    # plot with two y-axes distorts both).
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6.5, 12.0))
    # payoff_label must name the actual payoff (e.g. "Scalar Asian"), not a
    # bare option number, which means nothing to a reader with no context.
    fig.suptitle(payoff_label, fontsize=15, fontweight="bold", y=0.995)

    # ---- panel 1: runtime vs threads, all three designs ----
    for name, (t, s) in designs:
        st = DESIGN_STYLE[name]
        ax1.plot(t, s, markersize=6, linewidth=2, label=DESIGN_LABEL[name], **st)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(t_f)
    ax1.set_xticklabels([str(t) for t in t_f])
    ax1.set_xlabel("Threads")
    ax1.set_ylabel("Runtime (s)")
    ax1.set_title("Runtime vs threads")
    ax1.grid(True, which="both", axis="both", alpha=0.3)
    ax1.legend(frameon=False)

    # ---- panel 2: speedup of Kahan/Adaptive over fp32 ----
    for name, (t, s) in designs:
        if name == "fp32":
            continue
        speedup = s_f / s  # >1 means this design is faster than fp32
        st = DESIGN_STYLE[name]
        ax2.plot(t, speedup, markersize=6, linewidth=2,
                  label=f"{DESIGN_LABEL[name]} / fp32", **st)
    ax2.axhline(1.0, color="gray", linewidth=1, linestyle=":")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(t_f)
    ax2.set_xticklabels([str(t) for t in t_f])
    ax2.set_xlabel("Threads")
    ax2.set_ylabel("Speedup over fp32")
    ax2.set_title("Speedup over fp32")
    ax2.grid(True, which="both", axis="both", alpha=0.3)
    ax2.legend(frameon=False)

    # ---- panel 3: parallel efficiency, each design vs its own 1-thread time ----
    for name, (t, s) in designs:
        eff = efficiency(t, s)
        st = DESIGN_STYLE[name]
        ax3.plot(t, eff, markersize=6, linewidth=2, label=DESIGN_LABEL[name], **st)
    ax3.set_xscale("log", base=2)
    ax3.set_xticks(t_f)
    ax3.set_xticklabels([str(t) for t in t_f])
    ax3.set_xlabel("Threads")
    ax3.set_ylabel("Parallel efficiency  E(p) = T₁/(p·Tₚ)  (%)")
    ax3.set_ylim(0, 105)
    ax3.set_title("Parallel efficiency")
    ax3.grid(True, which="both", axis="both", alpha=0.3)
    ax3.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(f"{out_base}_scaling.png", dpi=150)
    print(f"saved {out_base}_scaling.png")

    # summary table -> .txt, for an appendix table
    with open(f"{out_base}_scaling_table.txt", "w") as f:
        f.write(f"{payoff_label}\n")
        f.write(f"{'threads':>8} {'kahan(s)':>10} {'adaptive(s)':>12} {'fp32(s)':>10} "
                f"{'kahan_spd':>10} {'adapt_spd':>10} "
                f"{'eff_kahan':>10} {'eff_adapt':>10} {'eff_fp32':>10}\n")
        t_k, s_k = designs[0][1]
        t_a, s_a = designs[1][1]
        eff_k = efficiency(t_k, s_k)
        eff_a = efficiency(t_a, s_a)
        eff_f = efficiency(t_f, s_f)
        for i, t in enumerate(t_f):
            f.write(f"{t:>8} {s_k[i]:>10.3f} {s_a[i]:>12.3f} {s_f[i]:>10.3f} "
                    f"{s_f[i]/s_k[i]:>9.2f}x {s_f[i]/s_a[i]:>9.2f}x "
                    f"{eff_k[i]:>9.1f}% {eff_a[i]:>9.1f}% {eff_f[i]:>9.1f}%\n")
    print(f"saved {out_base}_scaling_table.txt")


def efficiency_combo(pairs, out_base, combo_title=None):
    # pairs: list of (csv_path, label)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for (path, label), color in zip(pairs, COMBO_COLORS):
        t, s = read_csv(path)
        eff = efficiency(t, s)
        lname = label.lower()
        if "fp32" in lname:
            linestyle, marker = "--", "s"
        elif "kahan" in lname:
            linestyle, marker = "-.", "^"
        else:
            linestyle, marker = "-", "o"
        ax.plot(t, eff, marker=marker, markersize=6, linewidth=2,
                 linestyle=linestyle, color=color, label=label)

    ax.set_xscale("log", base=2)
    all_threads, _ = read_csv(pairs[0][0])
    ax.set_xticks(all_threads)
    ax.set_xticklabels([str(t) for t in all_threads])
    ax.set_xlabel("Threads")
    ax.set_ylabel("Parallel efficiency  E(p) = T₁/(p·Tₚ)  (%)")
    ax.set_ylim(0, 105)
    ax.set_title(combo_title or "Parallel efficiency vs thread count")
    ax.grid(True, which="both", axis="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{out_base}_efficiency_combo.png", dpi=150)
    print(f"saved {out_base}_efficiency_combo.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--efficiency-combo":
        rest = args[1:]
        combo_title = None
        if "--title" in rest:
            i = rest.index("--title")
            combo_title = rest[i + 1]
            rest = rest[:i] + rest[i + 2:]
        out_base = rest[-1]
        pair_args = rest[:-1]
        pairs = [(pair_args[i], pair_args[i + 1]) for i in range(0, len(pair_args), 2)]
        efficiency_combo(pairs, out_base, combo_title)
    else:
        kahan_csv, adaptive_csv, fp32_csv, payoff_label, out_base = args
        two_panel(kahan_csv, adaptive_csv, fp32_csv, payoff_label, out_base)
