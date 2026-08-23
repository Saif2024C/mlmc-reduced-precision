import numpy as np
import matplotlib.pyplot as plt


def _read_lines(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _find_line(lines, predicate, start=0):
    for i in range(start, len(lines)):
        if predicate(lines[i]):
            return i
    raise ValueError("Required marker line not found in file.")


def mlmc_plot(filename, nvert, error_bars=False):
    """
    Python translation of Giles' MATLAB mlmc_plot.m

    Parameters
    ----------
    filename : str
        Base filename without .txt
    nvert : int
        Either 1 or 3 in the MATLAB code
    error_bars : bool
        Whether to show 3-sigma error bars
    """
    if nvert not in (1, 3):
        raise ValueError("nvert must be 1 or 3")

    lines = _read_lines(filename + ".txt")

    # ------------------------------------------------------------
    # Find version line: first line beginning with '*** '
    # ------------------------------------------------------------
    i = _find_line(lines, lambda s: len(s) >= 4 and s[:4] == "*** ")

    line = lines[i]
    try:
        file_version = float(line[22:30].strip())
    except ValueError:
        file_version = 0.8

    # ------------------------------------------------------------
    # Find sample count N, if file_version >= 0.9
    # ------------------------------------------------------------
    N = 0
    if file_version >= 0.9:
        j = _find_line(lines, lambda s: len(s) >= 9 and s[:9] == "*** using", start=i + 1)
        line = lines[j]
        # MATLAB used sscanf(line(14:20),'%d')
        # safer in Python: grab all integers from the line
        ints = [int(x) for x in line.split() if x.lstrip("-").isdigit()]
        if len(ints) == 0:
            raise ValueError("Could not parse N from '*** using' line.")
        N = ints[0]
    elif error_bars:
        raise ValueError("cannot plot error bars -- no value of N in file")

    # ------------------------------------------------------------
    # Find first dashed separator line
    # ------------------------------------------------------------
    k = _find_line(lines, lambda s: len(s) > 0 and s[0] == "-", start=i + 1)

    # ------------------------------------------------------------
    # First block: level statistics
    # ------------------------------------------------------------
    l = k + 1
    del1 = []
    del2 = []
    var1 = []
    var2 = []
    kur1 = []
    chk1 = []
    cost = []

    while l < len(lines) and len(lines[l]) > 10:
        data = np.fromstring(lines[l], sep=" ")
        if data.size < 8:
            break
        del1.append(data[1])
        del2.append(data[2])
        var1.append(data[3])
        var2.append(data[4])
        kur1.append(data[5])
        chk1.append(data[6])
        cost.append(data[7])
        l += 1

    del1 = np.asarray(del1)
    del2 = np.asarray(del2)
    var1 = np.asarray(var1)
    var2 = np.asarray(var2)
    kur1 = np.asarray(kur1)
    chk1 = np.asarray(chk1)
    cost = np.asarray(cost)

    L = len(var1) - 1
    if L < 1:
        raise ValueError("Not enough level data found in file.")

    vvr1 = var1**2 * (kur1 - 1)

    # ------------------------------------------------------------
    # Find second dashed separator line
    # ------------------------------------------------------------
    m = _find_line(lines, lambda s: len(s) > 0 and s[0] == "-", start=l)

    # ------------------------------------------------------------
    # Second block: costs and sample allocations vs epsilon
    # ------------------------------------------------------------
    l = m + 1
    Eps = []
    mlmc_cost = []
    std_cost = []
    ls_cols = []
    Nls_cols = []

    while l < len(lines) and len(lines[l]) > 10:
        data = np.fromstring(lines[l], sep=" ")
        if data.size < 6:
            break

        Eps.append(data[0])
        mlmc_cost.append(data[2])
        std_cost.append(data[3])

        # MATLAB: len = length(data)-5; ls(1:len,l)=0:len-1; Nls(1:len,l)=data(6:end);
        num_levels = len(data) - 5
        ls_cols.append(np.arange(num_levels))
        Nls_cols.append(data[5:])

        l += 1

    Eps = np.asarray(Eps)
    mlmc_cost = np.asarray(mlmc_cost)
    std_cost = np.asarray(std_cost)

    max_levels = max(len(col) for col in Nls_cols)
    n_eps = len(Eps)

    # Build rectangular arrays padded with nan for plotting
    ls = np.full((max_levels, n_eps), np.nan)
    Nls = np.full((max_levels, n_eps), np.nan)
    for j in range(n_eps):
        lev = len(Nls_cols[j])
        ls[:lev, j] = ls_cols[j]
        Nls[:lev, j] = Nls_cols[j]

    # ------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------
    if nvert == 3:
        fig = plt.figure(figsize=(14, 12))
    else:
        fig = plt.figure(figsize=(14, 8))

    # Use black lines / markers similar to MATLAB defaults
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["k"])

    # ----- First panel: variance
    ax1 = plt.subplot(nvert, 2, 1)
    ax1.plot(np.arange(L + 1), np.log2(var2), "-*")
    ax1.plot(np.arange(1, L + 1), np.log2(var1[1:]), "-*")
    ax1.set_xlabel(r"level $\ell$")
    ax1.set_ylabel(r"$\log_2$ variance")
    ax1.set_xlim(0, L)
    ax1.legend([r"$P_\ell$", r"$P_\ell\!-\! P_{\ell-1}$"], loc="lower left")
    ax1.grid(True, alpha=0.3)

    if error_bars and N > 0:
        x = np.arange(1, L + 1)
        lower = np.log2(np.maximum(np.abs(var1[1:]) - 3 * np.sqrt(vvr1[1:] / N), 1e-10))
        upper = np.log2(np.abs(var1[1:]) + 3 * np.sqrt(vvr1[1:] / N))
        ax1.plot(np.vstack([x, x]), np.vstack([lower, upper]), "-r.")

    # ----- Second panel: mean magnitude
    ax2 = plt.subplot(nvert, 2, 2)
    ax2.plot(np.arange(L + 1), np.log2(np.abs(del2)), "-*")
    ax2.plot(np.arange(1, L + 1), np.log2(np.abs(del1[1:])), "-*")
    ax2.set_xlabel(r"level $\ell$")
    # MODIFIED: matplotlib >=3.x mathtext rejects the LaTeX \mbox command
    # (raises "Unknown symbol: \mbox" inside tight_layout). \mathrm gives the
    # same upright text. Original line kept below for reference:
    # ax2.set_ylabel(r"$\log_2 |\mbox{mean}|$")
    ax2.set_ylabel(r"$\log_2 |\mathrm{mean}|$")
    ax2.set_xlim(0, L)
    ax2.legend([r"$P_\ell$", r"$P_\ell\!-\! P_{\ell-1}$"], loc="lower left")
    ax2.grid(True, alpha=0.3)

    if error_bars and N > 0:
        x = np.arange(1, L + 1)
        lower = np.log2(np.maximum(np.abs(del1[1:]) - 3 * np.sqrt(var1[1:] / N), 1e-10))
        upper = np.log2(np.abs(del1[1:]) + 3 * np.sqrt(var1[1:] / N))
        ax2.plot(np.vstack([x, x]), np.vstack([lower, upper]), "-r.")

    # ----- Optional third/fourth panels
    if nvert == 3:
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(np.arange(L + 1), np.log2(cost), "--*")
        ax3.set_xlabel(r"level $\ell$")
        ax3.set_ylabel(r"$\log_2$ cost per sample")
        ax3.set_xlim(0, L)
        ax3.grid(True, alpha=0.3)

        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(np.arange(1, L + 1), kur1[1:], "--*")
        ax4.set_xlabel(r"level $\ell$")
        ax4.set_ylabel("kurtosis")
        ax4.set_xlim(0, L)
        ax4.grid(True, alpha=0.3)

    # ----- Sample counts
    ax5 = plt.subplot(nvert, 2, 2 * nvert - 1)
    for j in range(n_eps):
        ax5.semilogy(ls[:, j], Nls[:, j], marker="o")
    ax5.set_xlabel(r"level $\ell$")
    ax5.set_ylabel(r"$N_\ell$")
    ax5.set_xlim(0, Nls.shape[0] - 1)
    labels = [str(eps) for eps in Eps]
    ax5.legend(labels, loc="upper right")
    ax5.grid(True, which="both", alpha=0.3)

    # ----- Cost plot
    ax6 = plt.subplot(nvert, 2, 2 * nvert)
    ax6.loglog(Eps, Eps**2 * std_cost, "-*", label="Std MC")
    ax6.loglog(Eps, Eps**2 * mlmc_cost, "-*", label="MLMC")
    ax6.set_xlabel(r"accuracy $\varepsilon$")
    ax6.set_ylabel(r"$\varepsilon^2$ Cost")
    ax6.set_xlim(Eps[0], Eps[-1])
    ax6.legend()
    ax6.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()


# Example:
# mlmc_plot("results/mlmc", 3, error_bars=True)

# ADDED: command-line runner so this file can be executed directly, e.g.
#   python3 mlmc_plot_python.py mcqmc06_scalar_1
# The original file only defined mlmc_plot() with no entry point, so running it
# did nothing. argv[1] is the filename base (without .txt); defaults to the
# Asian convergence file. Saves a PNG (works headless) and also calls plt.show()
# which opens a window if a display (e.g. WSLg) is available.
if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "mcqmc06_scalar_1"
    mlmc_plot(base, 3, error_bars=False)
    out = base + "_convergence.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)
    plt.show()