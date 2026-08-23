import numpy as np
import matplotlib.pyplot as plt


def _read_lines(filename):
    with open(filename + ".txt", "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def mlmc_plot_100(filename):
    """
    Python translation of Giles' MATLAB mlmc_plot_100.m

    Expects the same text file format produced by mlmc_test_100.m.
    """
    lines = _read_lines(filename)

    # ------------------------------------------------------------
    # Find version line
    # ------------------------------------------------------------
    i = 0
    line = "    "
    while len(line) < 20 or line[:4] != "*** ":
        line = lines[i] + "    "
        i += 1

    try:
        file_version = float(line[26:34].strip())
    except ValueError:
        file_version = 0.8

    # MODIFIED: the original skipped a fixed 8 lines and read the reference value
    # from a fixed column offset (line[14:]), which lands on a blank line for the
    # header layout this file actually uses. Replaced with a scan for the
    # "Exact value" line. Original lines kept for reference:
    #     for _ in range(8):
    #         i += 1
    #     line = lines[i]
    #     try:
    #         value = float(line[14:].strip()); novalue = False
    #     except ValueError:
    #         value = None; novalue = True
    #     i += 1
    value = None
    novalue = True
    while i < len(lines):
        if "Exact value" in lines[i]:
            tail = lines[i].split("Exact value")[1].lstrip(": ").strip()
            try:
                value = float(tail)
                novalue = False
            except ValueError:          # e.g. "Exact value unknown"
                value = None
                novalue = True
            i += 1
            break
        i += 1

    # ------------------------------------------------------------
    # Read blocks for each epsilon
    # ------------------------------------------------------------
    Eps = []
    data_rows = []

    while i < len(lines):
        line = lines[i]
        # MODIFIED: original only treated a line as an eps block when its first
        # char was a digit/'.'/'-', but the eps lines read " eps = 5.000e-03 "
        # (leading space), so no block was ever found ("No epsilon blocks").
        # Detect by the "eps" marker and read the number after '='. Originals:
        #     if not line or not line[0].isdigit() and line[0] not in ".-":
        #         i += 1; continue
        #     eps_val = float(line[6:].strip())
        if "eps" not in line:
            i += 1
            continue
        try:
            eps_val = float(line.split("=")[1].strip())
        except (ValueError, IndexError):
            break

        Eps.append(eps_val)
        i += 1

        # skip the dashed separator line ("-----------------"), if present
        if i < len(lines) and lines[i].strip() and set(lines[i].strip()) <= {"-"}:
            i += 1

        # MODIFIED: original consumed a FIXED 20 lines (100 samples). The C++
        # driver (mlmc_test_100.cpp) now writes a variable number of runs per eps
        # (currently 10, "// changed to 10 runs"), so read consecutive numeric
        # lines until a blank line / next block / EOF instead. Original:
        #     row_vals = []
        #     for _ in range(20):
        #         if i >= len(lines): break
        #         vals = np.fromstring(lines[i], sep=" ")
        #         row_vals.extend(vals.tolist()); i += 1
        #     if len(row_vals) >= 100: data_rows.append(row_vals[:100])
        #     else: break
        row_vals = []
        while i < len(lines):
            if lines[i].strip() == "" or "eps" in lines[i]:
                break
            vals = np.fromstring(lines[i], sep=" ")
            if vals.size == 0:
                break
            row_vals.extend(vals.tolist())
            i += 1
        if len(row_vals) == 0:
            break
        data_rows.append(row_vals)

    if len(Eps) == 0:
        raise ValueError("No epsilon blocks found in file.")

    Eps = np.asarray(Eps)
    data = np.asarray(data_rows)
    # MODIFIED: original divided by a hard-coded 100; use the actual number of
    # runs per eps (n) so the statistics are correct for any run count.
    n = data.shape[1]

    if novalue:
        err = data - np.sum(data[0, :]) / n
    else:
        err = data - value

    ave = np.sum(err, axis=1) / n
    rms = np.sqrt(np.sum(err**2, axis=1) / n - ave**2)

    # ------------------------------------------------------------
    # Plot results
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute error at each epsilon
    ax = axes[0]
    ax.loglog(Eps, 3 * Eps, "--", label="3*Tol")
    ax.loglog(Eps, Eps, "-.", label="Tol")

    for l in range(len(Eps)):
        ax.loglog(np.full(err.shape[1], Eps[l]), np.abs(err[l, :]), "o")

    ax.set_xlabel(r"accuracy $\varepsilon$")
    ax.set_ylabel("Error")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    # Right: normalised RMS / MC / weak error
    ax = axes[1]
    if novalue:
        ax.semilogx(
            Eps,
            np.sqrt(rms**2 + ave**2) / Eps,
            Eps,
            rms / Eps,
            Eps,
            np.abs(ave / Eps),
        )
        ax.legend(["RMS error", "MC error", "weak error (est)"], loc="lower left")
    else:
        ax.semilogx(
            Eps,
            np.sqrt(rms**2 + ave**2) / Eps,
            Eps,
            rms / Eps,
            Eps,
            np.abs(ave / Eps),
        )
        ax.legend(["RMS error", "MC error", "weak error"], loc="lower left")

    lims = ax.axis()
    ax.set_ylim(0, max(lims[3], 1))
    ax.set_xlabel(r"accuracy $\varepsilon$")
    ax.set_ylabel(r"Error / $\varepsilon$")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()


# ADDED: command-line runner so this file can be executed directly, e.g.
#   python3 mlmc_plot_100_python.py mcqmc06_scalar_1_100
# The original file only defined mlmc_plot_100() with no entry point. argv[1] is
# the filename base (without .txt); defaults to the Asian 100-run file. Saves a
# PNG (works headless) and also calls plt.show() for an on-screen window if a
# display (e.g. WSLg) is available.
if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "mcqmc06_scalar_1_100"
    mlmc_plot_100(base)
    out = base + ".png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)
    plt.show()