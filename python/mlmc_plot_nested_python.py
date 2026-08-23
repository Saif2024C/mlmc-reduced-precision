"""
mlmc_plot_nested_python.py — dissertation-quality plots for nested MLMC output.

Generates TWO figures per output file:

Figure 1 (_convergence.png): 2x2 diagnostic
  [0,0]  Variance decay  — even super-levels, with regression slope -beta_hat
  [0,1]  Mean decay      — even super-levels, with regression slope -alpha_hat
  [1,0]  Precision correction variance — odd super-levels, with floor annotation
  [1,1]  Kurtosis        — even super-levels

Figure 2 (_complexity.png): 1x2 complexity
  [0]    Sample allocation: solid=even (MLMC corr.), dashed=odd (precision corr.)
  [1]    eps^2 * Cost: Std MC vs Nested MLMC

Usage:
    MPLBACKEND=Agg python3 python/mlmc_plot_nested_python.py outputs/nested_scalar_fp16_kahan_1
    MPLBACKEND=Agg python3 python/mlmc_plot_nested_python.py outputs/nested_basket_milstein_1 "float Milstein"
"""

import sys
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------
# File reader
# ----------------------------------------------------------------
def read_nested_mlmc(filename):
    with open(filename + ".txt", "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]

    i = next(k for k, l in enumerate(lines) if l[:4] == "*** ")

    N = 0
    for j in range(i + 1, len(lines)):
        if lines[j][:9] == "*** using":
            ints = [int(x) for x in lines[j].split() if x.lstrip("-").isdigit()]
            if ints:
                N = ints[0]
            break

    k = next(k for k in range(i + 1, len(lines)) if lines[k] and lines[k][0] == "-")

    rows = []
    l = k + 1
    while l < len(lines) and len(lines[l]) > 10:
        data = np.fromstring(lines[l], sep=" ")
        if data.size < 8:
            break
        rows.append(data)
        l += 1
    rows = np.array(rows)
    # cols: super_l, ave(Pf-Pc), ave(Pf), var(Pf-Pc), var(Pf), kurtosis, check, cost

    m = next(k for k in range(l, len(lines)) if lines[k] and lines[k][0] == "-")

    Eps, mlmc_cost, std_cost, Nls_list = [], [], [], []
    l = m + 1
    while l < len(lines) and len(lines[l]) > 10:
        data = np.fromstring(lines[l], sep=" ")
        if data.size < 6:
            break
        Eps.append(data[0])
        mlmc_cost.append(data[2])
        std_cost.append(data[3])
        Nls_list.append(data[5:])
        l += 1

    return (rows,
            np.array(Eps), np.array(mlmc_cost), np.array(std_cost),
            Nls_list, N)


# ----------------------------------------------------------------
# Figure 1: 2x2 diagnostic (variance, mean, precision corr, kurtosis)
# ----------------------------------------------------------------
def plot_convergence(filename, label=""):
    rows, Eps, mlmc_cost, std_cost, Nls_list, N = read_nested_mlmc(filename)

    super_levels = rows[:, 0].astype(int)
    even_mask = (super_levels % 2 == 0)
    odd_mask  = ~even_mask

    even_rows = rows[even_mask]
    odd_rows  = rows[odd_mask]

    e_phys = even_rows[:, 0].astype(int) // 2
    o_phys = odd_rows[:,  0].astype(int) // 2

    e_del1 = even_rows[:, 1]
    e_del2 = even_rows[:, 2]
    e_var1 = even_rows[:, 3]
    e_var2 = even_rows[:, 4]
    e_kur1 = even_rows[:, 5]
    o_var1 = odd_rows[:, 3]

    TINY  = 1e-40
    FLOOR = 1e-15

    BLUE  = "#1f77b4"
    RED   = "#d62728"
    GREEN = "#2ca02c"
    GREY  = "grey"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title_suffix = f" — {label}" if label else ""

    # ---- [0,0]: Variance decay --------------------------------------
    ax = axes[0, 0]
    ax.plot(e_phys, np.log2(np.maximum(e_var2, TINY)),
            "o-", color=BLUE, label=r"$\mathrm{Var}(P_\ell)$")

    k_corr = e_phys[1:]
    v_corr = np.log2(np.maximum(e_var1[1:], TINY))
    ax.plot(k_corr, v_corr,
            "s-", color=RED, label=r"$\mathrm{Var}(P_\ell - P_{\ell-1})$")

    if len(k_corr) >= 2:
        coeffs = np.polyfit(k_corr, v_corr, 1)
        beta_hat = -coeffs[0]
        k_line = np.array([k_corr[0], k_corr[-1]])
        ax.plot(k_line, np.polyval(coeffs, k_line), "k--",
                label=rf"slope $-\hat{{\beta}} = {beta_hat:.2f}$")

    ax.set_xlabel("grid level $k$")
    ax.set_ylabel(r"$\log_2$ variance")
    ax.set_title("Variance decay" + title_suffix)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- [0,1]: Mean decay ------------------------------------------
    ax = axes[0, 1]
    ax.plot(e_phys, np.log2(np.maximum(np.abs(e_del2), TINY)),
            "o-", color=BLUE, label=r"$|E[P_\ell]|$")

    m_corr = np.log2(np.maximum(np.abs(e_del1[1:]), TINY))
    ax.plot(k_corr, m_corr,
            "s-", color=RED, label=r"$|E[P_\ell - P_{\ell-1}]|$")

    if len(k_corr) >= 2:
        coeffs_m = np.polyfit(k_corr, m_corr, 1)
        alpha_hat = -coeffs_m[0]
        k_line_m = np.array([k_corr[0], k_corr[-1]])
        ax.plot(k_line_m, np.polyval(coeffs_m, k_line_m), "k--",
                label=rf"slope $-\hat{{\alpha}} = {alpha_hat:.2f}$")

    ax.set_xlabel("grid level $k$")
    ax.set_ylabel(r"$\log_2 |\mathrm{mean}|$")
    ax.set_title("Mean decay" + title_suffix)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- [1,0]: Precision correction variance -----------------------
    ax = axes[1, 0]
    ax.plot(o_phys, np.log2(np.maximum(o_var1, TINY)),
            "^-", color=GREEN,
            label=r"$\mathrm{Var}(\delta_{\mathrm{prec}})$")
    floor_log2 = np.log2(FLOOR)
    ax.axhline(floor_log2, color=GREY, linestyle="--", linewidth=1,
               label=f"floor {FLOOR:.0e}")

    ax.set_xlabel("grid level $k$")
    ax.set_ylabel(r"$\log_2$ variance")
    ax.set_title("Precision correction variance — odd super-levels\n"
                 r"(near floor $\Rightarrow$ cheap $\approx$ exact)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- [1,1]: Kurtosis --------------------------------------------
    ax = axes[1, 1]
    if len(e_phys) > 1:
        ax.plot(e_phys[1:], e_kur1[1:], "o-", color=BLUE)
    ax.axhline(0, color=GREY, linestyle="--", linewidth=0.8)
    ax.set_xlabel("grid level $k$")
    ax.set_ylabel("kurtosis")
    ax.set_title("Kurtosis" + title_suffix)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = filename + "_convergence.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)
    plt.show()
    plt.close()


# ----------------------------------------------------------------
# Figure 2: 1x2 complexity (sample allocation + eps^2 cost)
# ----------------------------------------------------------------
def plot_complexity(filename, label=""):
    rows, Eps, mlmc_cost, std_cost, Nls_list, N = read_nested_mlmc(filename)

    BLUE = "#1f77b4"
    RED  = "#d62728"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    title_suffix = f" — {label}" if label else ""

    # ---- Left: sample allocation ------------------------------------
    ax = axes[0]

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(Eps)))

    # Dummy legend entries for marker style
    ax.plot([], [], "o-",  color="grey", label="even $\\ell$: MLMC corr.")
    ax.plot([], [], "s--", color="grey", label="odd $\\ell$: precision corr.")

    for eps_val, Nls, col in zip(Eps, Nls_list, colors):
        n = len(Nls)
        sl = np.arange(n)

        even_sl  = sl[0::2];  even_Nls  = Nls[0::2]
        odd_sl   = sl[1::2];  odd_Nls   = Nls[1::2]

        ax.semilogy(even_sl, even_Nls, "o-",  color=col,
                    label=f"$\\varepsilon={eps_val}$")
        if len(odd_sl) > 0:
            ax.semilogy(odd_sl, odd_Nls, "s--", color=col)

    ax.set_xlabel("super-level $\\ell$")
    ax.set_ylabel("$N_\\ell$")
    ax.set_title("Sample allocation (solid=even/MLMC, dashed=odd/precision)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    # ---- Right: cost complexity -------------------------------------
    ax = axes[1]
    ax.loglog(Eps, Eps**2 * std_cost,  "o-", color=BLUE,
              label="Std MC (double)")
    ax.loglog(Eps, Eps**2 * mlmc_cost, "s-", color=RED,
              label="Nested MLMC")
    ax.set_xlabel("accuracy $\\varepsilon$")
    ax.set_ylabel("$\\varepsilon^2 \\times$ Cost")
    ax.set_title("Cost complexity — nested MLMC vs standard MC" + title_suffix)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = filename + "_complexity.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)
    plt.show()
    plt.close()


# ----------------------------------------------------------------
# Combined 3x2 figure
# ----------------------------------------------------------------
def plot_nested_mlmc(filename, label=""):
    rows, Eps, mlmc_cost, std_cost, Nls_list, N = read_nested_mlmc(filename)

    super_levels = rows[:, 0].astype(int)
    even_mask = (super_levels % 2 == 0)
    odd_mask  = ~even_mask

    even_rows = rows[even_mask]
    odd_rows  = rows[odd_mask]

    e_phys = even_rows[:, 0].astype(int) // 2
    o_phys = odd_rows[:,  0].astype(int) // 2

    e_del1 = even_rows[:, 1]
    e_del2 = even_rows[:, 2]
    e_var1 = even_rows[:, 3]
    e_var2 = even_rows[:, 4]
    e_kur1 = even_rows[:, 5]
    o_var1 = odd_rows[:, 3]

    TINY  = 1e-40
    FLOOR = 1e-15
    BLUE  = "#1f77b4"
    RED   = "#d62728"
    GREEN = "#2ca02c"
    GREY  = "grey"

    title_suffix = f" — {label}" if label else ""
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    # ---- [0,0]: Variance decay --------------------------------------
    ax = axes[0, 0]
    ax.plot(e_phys, np.log2(np.maximum(e_var2, TINY)),
            "o-", color=BLUE, label=r"$\mathrm{Var}(P_\ell)$")
    k_corr = e_phys[1:]
    v_corr = np.log2(np.maximum(e_var1[1:], TINY))
    ax.plot(k_corr, v_corr, "s-", color=RED,
            label=r"$\mathrm{Var}(P_\ell - P_{\ell-1})$")
    if len(k_corr) >= 2:
        coeffs = np.polyfit(k_corr, v_corr, 1)
        k_line = np.array([k_corr[0], k_corr[-1]])
        ax.plot(k_line, np.polyval(coeffs, k_line), "k--",
                label=rf"slope $-\hat{{\beta}} = {-coeffs[0]:.2f}$")
    ax.set_xlabel("grid level $k$")
    ax.set_ylabel(r"$\log_2$ variance")
    ax.set_title("Variance decay" + title_suffix)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- [0,1]: Mean decay ------------------------------------------
    ax = axes[0, 1]
    ax.plot(e_phys, np.log2(np.maximum(np.abs(e_del2), TINY)),
            "o-", color=BLUE, label=r"$|E[P_\ell]|$")
    m_corr = np.log2(np.maximum(np.abs(e_del1[1:]), TINY))
    ax.plot(k_corr, m_corr, "s-", color=RED,
            label=r"$|E[P_\ell - P_{\ell-1}]|$")
    if len(k_corr) >= 2:
        coeffs_m = np.polyfit(k_corr, m_corr, 1)
        k_line_m = np.array([k_corr[0], k_corr[-1]])
        ax.plot(k_line_m, np.polyval(coeffs_m, k_line_m), "k--",
                label=rf"slope $-\hat{{\alpha}} = {-coeffs_m[0]:.2f}$")
    ax.set_xlabel("grid level $k$")
    ax.set_ylabel(r"$\log_2 |\mathrm{mean}|$")
    ax.set_title("Mean decay" + title_suffix)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- [1,0]: Precision correction variance -----------------------
    ax = axes[1, 0]
    ax.plot(o_phys, np.log2(np.maximum(o_var1, TINY)),
            "^-", color=GREEN,
            label=r"$\mathrm{Var}(\delta_{\mathrm{prec}})$")
    ax.axhline(np.log2(FLOOR), color=GREY, linestyle="--", linewidth=1,
               label=f"floor {FLOOR:.0e}")
    ax.set_xlabel("grid level $k$")
    ax.set_ylabel(r"$\log_2$ variance")
    ax.set_title("Precision correction variance — odd super-levels\n"
                 r"(near floor $\Rightarrow$ cheap $\approx$ exact)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- [1,1]: Kurtosis --------------------------------------------
    ax = axes[1, 1]
    if len(e_phys) > 1:
        ax.plot(e_phys[1:], e_kur1[1:], "o-", color=BLUE)
    ax.axhline(0, color=GREY, linestyle="--", linewidth=0.8)
    ax.set_xlabel("grid level $k$")
    ax.set_ylabel("kurtosis")
    ax.set_title("Kurtosis" + title_suffix)
    ax.grid(True, alpha=0.3)

    # ---- [2,0]: Sample allocation -----------------------------------
    ax = axes[2, 0]
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(Eps)))
    ax.plot([], [], "o-",  color="grey", label="even $\\ell$: MLMC corr.")
    ax.plot([], [], "s--", color="grey", label="odd $\\ell$: precision corr.")
    for eps_val, Nls, col in zip(Eps, Nls_list, colors):
        n   = len(Nls)
        sl  = np.arange(n)
        ax.semilogy(sl[0::2], Nls[0::2], "o-",  color=col,
                    label=f"$\\varepsilon={eps_val}$")
        if len(sl[1::2]) > 0:
            ax.semilogy(sl[1::2], Nls[1::2], "s--", color=col)
    ax.set_xlabel("super-level $\\ell$")
    ax.set_ylabel("$N_\\ell$")
    ax.set_title("Sample allocation (solid=even/MLMC, dashed=odd/precision)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    # ---- [2,1]: Cost complexity -------------------------------------
    ax = axes[2, 1]
    ax.loglog(Eps, Eps**2 * std_cost,  "o-", color=BLUE,
              label="Std MC (double)")
    ax.loglog(Eps, Eps**2 * mlmc_cost, "s-", color=RED,
              label="Nested MLMC")
    ax.set_xlabel("accuracy $\\varepsilon$")
    ax.set_ylabel("$\\varepsilon^2 \\times$ Cost")
    ax.set_title("Cost complexity — nested MLMC vs standard MC")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = filename + "_convergence.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)
    plt.show()
    plt.close()


# ----------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------
if __name__ == "__main__":
    base  = sys.argv[1] if len(sys.argv) > 1 else "outputs/nested_scalar_fp16_kahan_1"
    label = sys.argv[2] if len(sys.argv) > 2 else ""
    plot_nested_mlmc(base, label)
