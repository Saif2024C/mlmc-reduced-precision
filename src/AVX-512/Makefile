# ===========================================================================
# Makefile — AVX-512 fp16/fp32 nested-MLMC Milstein estimators
#
# Target: Oxford Maths fp16 nodes (mimic / nocturne — Emerald / Granite Rapids,
# avx512_fp16).  Requires the Intel oneAPI compiler (icpx): my_mm512.h uses
# <mathimf.h>, cdfnorminv and SVML which g++ does not provide.
#
# One unified template per payoff, built serial (*_avx) and OpenMP+AVX (*_omp):
# nsh (scalar) / nbh (basket). The four design variants (nokahan / kahan /
# adaptive / puref32) are RUNTIME MODES swept by run_sweep() in modes_mm512.h,
# not build targets — the binaries take ONLY --option N, there is no --mode.
# Isolating one mode's timing means reading the per-mode output files, not
# invoking a per-mode binary.
#
# --option N (1 or 2) restricts a run to one payoff. Use it for any timing:
# scalar Asian/Lookback and basket European/Asian do NOT cost the same per
# sample (different l_star, different N_l), so a combined time averages two
# different things together.
#
# IMPORTANT: `source /opt/intel/oneapi/setvars.sh` before BOTH building AND
# running — the OpenMP binaries need libiomp5.so at load time. Without it they
# exit 127 instantly, and since they never run, every timing target reports
# avg=0.000000s. That is not a Makefile bug.
#
# RUNS (default 5) = repetitions each timing target averages.
#
# Usage (on a compute node):
#   source /opt/intel/oneapi/setvars.sh
#   make              # both serial AVX binaries
#   make omp          # both OpenMP+AVX binaries
#   make everything   # both sets
#   make run OPTION=1                 # build serial + run both payoffs, option 1 only
#   make run_omp OPTION=2             # same, OpenMP, option 2 only
#   make scaling BIN=nsh_omp OPTION=1 # strong-scaling sweep (1..64 threads), one binary/option
#   make speedup BIN=nsh_omp OPTION=1 THREADS=16   # 1-thread vs THREADS-thread ratio, one line
#   make mkl          # serial with MKL VSL RNG cross-check (USE_MKL_RNG)
#   make coupling_test ; make clean
#
# gcc-14's dir exists on the nodes but its libstdc++ headers do not, so pin icpx
# to the complete gcc-13 toolchain (else: "C++ header location not resolved").
# ===========================================================================

CXX       := icpx
GCCDIR    := /usr/lib/gcc/x86_64-linux-gnu/13
CXXFLAGS  := -O3 -march=native -std=c++17 --gcc-install-dir=$(GCCDIR)
OMPFLAGS  := -qopenmp
OUT       := ../../outputs/avx512

ifeq ($(MAKECMDGOALS),mkl)
  CXXFLAGS += -DUSE_MKL_RNG -qmkl
endif

DEPS := avx512_rng.h my_mm512.h options_mm512.h modes_mm512.h

# OPTION selects the payoff for run/scaling/speedup/compare targets:
#   scalar: 1=Asian, 2=Lookback   |   basket: 1=European, 2=Asian
# Leave unset to run/scale/compare both options (the binaries' own default).
OPTION ?=
OPTARG := $(if $(OPTION),--option $(OPTION),)

# Number of repeated timings per thread count / payoff. The scaling and CSV
# targets average these runs so the dissertation figures are based on means
# rather than single noisy measurements.
RUNS ?= 5


BINS    := nsh_avx nbh_avx
OMPBINS := nsh_omp nbh_omp

# ---- serial AVX ----
nsh_avx: nested_scalar_milstein_fp16_avx512.cpp         $(DEPS)
	$(CXX) $(CXXFLAGS) $< -o $@
nbh_avx: nested_basket_milstein_fp16_avx512.cpp         $(DEPS)
	$(CXX) $(CXXFLAGS) $< -o $@

# ---- OpenMP + AVX ----
nsh_omp: nested_scalar_milstein_fp16_avx512.cpp           $(DEPS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@
nbh_omp: nested_basket_milstein_fp16_avx512.cpp           $(DEPS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

all:        $(BINS)
omp:        $(OMPBINS)
everything: $(BINS) $(OMPBINS)
mkl:        $(BINS)

# ---- coupling validation (coupling_validation_test/) ----
# Checks that the fp32/fp16 coupled draws really are matched -- i.e. that the
# odd-level precision correction isolates fp16 ARITHMETIC error rather than
# adding independent-draw noise on top of it.
coupling_test: coupling_validation_test/coupling_test.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $< -o $@ && ./coupling_test

# ---- Level-0 kernel benchmark (Performance_analysis.txt item 1) ----
# Profiles the real l=0 production kernels in isolation, NOT the MLMC driver.
# kb_basket_l0     : serial AVX-512, basket European, fp32 vs fp16+Kahan
# kb_basket_l0_omp : same, OpenMP (item 2, thread scaling)
kb_basket_l0: performance_analysis/kernel_bench_basket_l0.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $< -o $@
kb_basket_l0_omp: performance_analysis/kernel_bench_basket_l0.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

# Serial timing across the driver's own level-0 sample counts (one row per
# eps in {0.2,0.1,0.05,0.02,0.01}), with throughput and the correctness
# checksum.  This replaces the former single-N `basket_l0_kernel.txt` run,
# whose eps=0.01 row this sweep reproduces along with four more.
kbench: kb_basket_l0
	@mkdir -p $(OUT)/kernel
	./kb_basket_l0 --sweep $(if $(REPS),--reps $(REPS),) \
	   | tee $(OUT)/kernel/basket_l0_eps_sweep.txt

# perf counters for ONE kernel at a time (KERNEL=fp32 or fp16).
# IPC, SIMD utilisation, L1/L2/L3 misses, bandwidth -- the analysis columns.
# fp16 ops are counted by fp_arith_inst_retired2.* -- a SEPARATE event group
# from the fp32 fp_arith_inst_retired.* (note the trailing 2), not a suffix on
# the existing one. Without them the fp16+Kahan kernel reads as ~zero vector
# activity, since its work is invisible to the _single counters. The fp16
# kernel's state is __m256h (16 lanes, 256-bit), so 256b_packed_half is the
# one that fires; 512b is collected too in case a path widens. Both groups are
# collected for both kernels so the SIMD comparison is like-for-like.
#
# Collected as THREE passes, not one. All 13 events in a single `perf stat` do
# not fit the available PMU counters, so perf time-multiplexes them and every
# number becomes an extrapolation from a ~30-70% sample window (the percentage
# perf prints after each line). Three smaller groups each fit, so every pass
# reports exact counts. The kernel is deterministic and re-run per pass, so
# combining across passes is sound.
#   pass 1 (SIMD):  the fp32 and fp16 arith counters
#   pass 2 (L1/L2): L1 loads+misses, L2 misses
#   pass 3 (LLC):   LLC loads+misses
# LLC is split off because the L1/L2/LLC events together still exceeded the
# counters and multiplexed at ~85%.
#
# task-clock,cycles,instructions appear in EVERY pass. task-clock is a software
# event (no PMU counter), so it is free and does not reintroduce multiplexing;
# perf divides cycles by it to print the effective clock, which is what makes
# the AVX-512 downclocking visible (~2.7 GHz here, well under base) and lets
# you rule out frequency as the explanation for a cycle-count difference. The
# cycles/instructions repetition is a cross-check: if IPC disagrees across
# passes, the runs were not comparable and the rest is suspect.
KERNEL ?= fp32
kbench_perf: kb_basket_l0
	@mkdir -p $(OUT)/kernel
	@{ echo "=== pass 1: SIMD / instruction mix ==="; \
	perf stat -e task-clock,cycles,instructions,\
fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.256b_packed_single,\
fp_arith_inst_retired2.512b_packed_half,fp_arith_inst_retired2.256b_packed_half \
	  ./kb_basket_l0 --kernel $(KERNEL) $(if $(NPATHS),--n $(NPATHS),) 2>&1; \
	echo; echo "=== pass 2: L1 / L2 ==="; \
	perf stat -e task-clock,cycles,instructions,\
L1-dcache-loads,L1-dcache-load-misses,l2_rqsts.miss \
	  ./kb_basket_l0 --kernel $(KERNEL) $(if $(NPATHS),--n $(NPATHS),) 2>&1; \
	echo; echo "=== pass 3: LLC ==="; \
	perf stat -e task-clock,cycles,instructions,LLC-loads,LLC-load-misses \
	  ./kb_basket_l0 --kernel $(KERNEL) $(if $(NPATHS),--n $(NPATHS),) 2>&1; \
	} | tee $(OUT)/kernel/basket_l0_perf_$(KERNEL).txt

# VTune uarch-exploration for one kernel (richer than perf: port utilisation,
# retiring/bad-spec/backend breakdown).
kbench_vtune: kb_basket_l0
	@mkdir -p $(OUT)/kernel
	vtune -collect uarch-exploration -r $(OUT)/kernel/vt_$(KERNEL) \
	  -- ./kb_basket_l0 --kernel $(KERNEL) $(if $(NPATHS),--n $(NPATHS),)

# Thread scaling of the l=0 kernels (analysis item 2).
# 1 and 32 threads only (Giles' decision).  mimic is 2 sockets x 32 physical
# cores x 2 HT, so 32 threads is exactly ONE socket's physical cores: no
# hyperthread contention and no NUMA crossing, which makes the efficiency
# number clean -- any loss is the kernel's own, not an interconnect artifact.
#
# Same row-per-eps layout as the serial `kbench` sweep, with speedup and
# parallel efficiency columns added.  Medians over repeated interleaved trials
# (see the --scaling block's comment) rather than best-of, because the earlier
# best-of version produced a physically impossible row at eps=0.02.
# Thread counts are set INSIDE the program, so OMP_NUM_THREADS is deliberately
# not exported here -- one process, one warm-up, identical N per row.
kbench_scaling: kb_basket_l0_omp
	@mkdir -p $(OUT)/kernel
	OMP_PLACES=cores OMP_PROC_BIND=close \
	  ./kb_basket_l0_omp --scaling $(if $(REPS),--reps $(REPS),) \
	  2>&1 | tee $(OUT)/kernel/basket_l0_scaling_sweep.txt

run: $(BINS)
	mkdir -p $(OUT)
	cd $(OUT) && $(CURDIR)/nsh_avx $(OPTARG) && $(CURDIR)/nbh_avx $(OPTARG)

run_omp: $(OMPBINS)
	mkdir -p $(OUT)
	cd $(OUT) && $(CURDIR)/nsh_omp $(OPTARG) && $(CURDIR)/nbh_omp $(OPTARG)
# Strong-scaling sweep, one OMP binary/option: mean wall-time and sample SD
# over RUNS repetitions at each thread count (RUNS=1 -> SD 0).
# e.g. make scaling BIN=nsh_omp OPTION=1 RUNS=5
BIN ?= nsh_omp
scaling: $(BIN)
	@mkdir -p $(OUT); cd $(OUT); \
	for t in 1 2 4 8 16 32 64; do \
	  sum=0; sumsq=0; \
	  for r in $$(seq 1 $(RUNS)); do \
	    OMP_NUM_THREADS=$$t /usr/bin/time -f "%e" -o /tmp/.mlmc_t $(CURDIR)/$(BIN) $(OPTARG) >/dev/null 2>/dev/null; \
	    tt=$$(cat /tmp/.mlmc_t); \
	    sum=$$(awk -v s="$$sum" -v x="$$tt" 'BEGIN{printf "%.10f", s + x}'); \
	    sumsq=$$(awk -v s="$$sumsq" -v x="$$tt" 'BEGIN{printf "%.10f", s + x*x}'); \
	  done; \
	  awk -v s="$$sum" -v sq="$$sumsq" -v n=$(RUNS) -v t=$$t \
	    'BEGIN{ m=s/n; \
	            sd=(n>1)?sqrt((sq-s*s/n)/(n-1)):0; \
	            printf "threads=%-3d  avg=%.6fs  sd=%.6fs  (n=%d)\n", t, m, sd, n }'; \
	done

# One-line speedup: mean 1-thread vs THREADS-thread wall-time and the ratio.
# e.g. make speedup BIN=nsh_omp OPTION=1 THREADS=16
THREADS ?= 16
speedup: $(BIN)
	@mkdir -p $(OUT); cd $(OUT); \
	t1sum=0; tnsum=0; \
	for r in $$(seq 1 $(RUNS)); do \
	  OMP_NUM_THREADS=1 /usr/bin/time -f "%e" -o /tmp/.mlmc_t $(CURDIR)/$(BIN) $(OPTARG) >/dev/null 2>/dev/null; t1=$$(cat /tmp/.mlmc_t); \
	  OMP_NUM_THREADS=$(THREADS) /usr/bin/time -f "%e" -o /tmp/.mlmc_t $(CURDIR)/$(BIN) $(OPTARG) >/dev/null 2>/dev/null; tn=$$(cat /tmp/.mlmc_t); \
	  t1sum=$$(awk -v s="$$t1sum" -v x="$$t1" 'BEGIN{printf "%.10f", s + x}'); \
	  tnsum=$$(awk -v s="$$tnsum" -v x="$$tn" 'BEGIN{printf "%.10f", s + x}'); \
	done; \
	t1avg=$$(awk -v s="$$t1sum" -v n=$(RUNS) 'BEGIN{printf "%.6f", s / n}'); \
	tnavg=$$(awk -v s="$$tnsum" -v n=$(RUNS) 'BEGIN{printf "%.6f", s / n}'); \
	awk -v t1=$$t1avg -v tn=$$tnavg -v bin=$(BIN) -v th=$(THREADS) -v opt="$(OPTION)" -v n=$(RUNS) \
	  'BEGIN{printf "%s option=%s  1-thread(avg %d)=%.3fs  %s-thread(avg %d)=%.3fs  speedup=%.2fx\n", bin, (opt==""?"both":opt), n, t1, th, n, tn, t1/tn}'

# REMOVED: `compare` (side-by-side timing of the three designs at one option).
# It drove the six per-design binaries nsk/nsa/nsf/nbk/nba/nbf, which no longer
# exist — the designs are runtime modes inside nsh/nbh now, so "time one
# design" is not expressible as a binary invocation. Reinstating it needs
# either a --mode flag on run_sweep() or per-mode times read from the outputs.

# Strong-scaling sweep to CSV (threads,mean_seconds), RUNS averaged per point.
# e.g. make scaling_csv BIN=nsh_omp OPTION=1 CSV=/tmp/x.csv
CSV ?= /tmp/scaling.csv
scaling_csv: $(BIN)
	@mkdir -p $(OUT); cd $(OUT); \
	rm -f $(CSV); \
	for t in 1 2 4 8 16 32 64; do \
	  sum=0; \
	  for r in $$(seq 1 $(RUNS)); do \
	    OMP_NUM_THREADS=$$t /usr/bin/time -f "%e" -o /tmp/.mlmc_t $(CURDIR)/$(BIN) $(OPTARG) >/dev/null 2>/dev/null; tt=$$(cat /tmp/.mlmc_t); \
	    sum=$$(awk -v s="$$sum" -v x="$$tt" 'BEGIN{printf "%.10f", s + x}'); \
	  done; \
	  avg=$$(awk -v s="$$sum" -v n=$(RUNS) 'BEGIN{printf "%.6f", s / n}'); \
	  echo "$$t,$$avg" >> $(CSV); \
	done

# NOTE: a `plot` target used to live here, producing the dissertation
# runtime/speedup/efficiency figures via scaling_csv once per design. Same
# reason as `compare`: it drove the six per-design binaries. The plotting
# script python/scaling_plot_python.py is untouched and still works — what
# needs rebuilding is the per-design timing that feeds it.
PLOTDIR := $(OUT)/scaling

clean:
	rm -f $(BINS) $(OMPBINS) coupling_test kb_basket_l0 kb_basket_l0_omp

# push / pull / kpull run on the LOCAL machine, not on mimic (every other
# target above runs on the compute node). Requires an ssh alias `mimic`
# (~/.ssh/config) resolving the gate proxy-jump. From src/AVX-512/ locally:
#   make push
#   ssh mimic 'source /opt/intel/oneapi/setvars.sh && cd ~/mlmc_reference/src/AVX-512 && make run_omp OPTION=1'
#   make pull      # or: make kpull, for the level-0 kernel-benchmark results
REMOTE     := mimic
REMOTEROOT := ~/mlmc_reference
LOCALROOT  := $(CURDIR)/../..

push:
	ssh $(REMOTE) 'mkdir -p $(REMOTEROOT)/src/AVX-512/performance_analysis \
	                        $(REMOTEROOT)/src/AVX-512/coupling_validation_test \
	                        $(REMOTEROOT)/src/core'
	scp $(LOCALROOT)/src/core/nested_mlmc_test.cpp $(REMOTE):$(REMOTEROOT)/src/core/
	scp $(LOCALROOT)/src/core/nested_mlmc.cpp $(REMOTE):$(REMOTEROOT)/src/core/
	scp $(CURDIR)/nested_scalar_milstein_fp16_avx512.cpp $(REMOTE):$(REMOTEROOT)/src/AVX-512/
	scp $(CURDIR)/nested_basket_milstein_fp16_avx512.cpp $(REMOTE):$(REMOTEROOT)/src/AVX-512/
	scp $(CURDIR)/Makefile $(REMOTE):$(REMOTEROOT)/src/AVX-512/Makefile
	scp $(CURDIR)/options_mm512.h $(REMOTE):$(REMOTEROOT)/src/AVX-512/
	scp $(CURDIR)/modes_mm512.h $(REMOTE):$(REMOTEROOT)/src/AVX-512/
	scp $(CURDIR)/performance_analysis/kernel_bench_basket_l0.cpp $(REMOTE):$(REMOTEROOT)/src/AVX-512/performance_analysis/
	scp $(CURDIR)/coupling_validation_test/coupling_test.cpp $(REMOTE):$(REMOTEROOT)/src/AVX-512/coupling_validation_test/
	scp $(LOCALROOT)/python/scaling_plot_python.py $(REMOTE):$(REMOTEROOT)/python/scaling_plot_python.py

# Pull the Level-0 kernel-benchmark results back from mimic.  These are
# produced on the cluster (no AVX-512 FP16 hardware locally), so without this
# they exist ONLY on the remote and are invisible in the local tree.
kpull:
	@mkdir -p $(LOCALROOT)/outputs/avx512/kernel
	scp '$(REMOTE):$(REMOTEROOT)/outputs/avx512/kernel/*' \
	    $(LOCALROOT)/outputs/avx512/kernel/
	@echo "--- pulled into outputs/avx512/kernel/ ---"
	@ls -la $(LOCALROOT)/outputs/avx512/kernel/

pull:
	@mkdir -p $(LOCALROOT)/outputs/avx512/scaling
	@for pat in '*.csv' '*_scaling.png' '*_efficiency_combo.png' '*_scaling_table.txt'; do \
	  scp "$(REMOTE):$(REMOTEROOT)/outputs/avx512/scaling/$$pat" $(LOCALROOT)/outputs/avx512/scaling/; \
	done

.PHONY: all omp everything mkl run run_omp scaling scaling_csv speedup push pull kpull clean \
        kbench kbench_perf kbench_vtune kbench_scaling coupling_test
