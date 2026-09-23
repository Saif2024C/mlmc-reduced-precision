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
# scalar Asian and basket European/Asian do NOT cost the same per
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
#   scalar: 1=Asian   |   basket: 1=European, 2=Asian
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
kb_scalar_asian_l0: performance_analysis/kernel_bench_scalar_asian_l0.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $< -o $@

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

# Store-side counters (Giles, 2026-08-24). kbench_perf collects L1-dcache-loads
# but NO store event, so Table 4.4 shows 157 load uops/path against 62 arith
# instructions/path in fp32 without being able to say whether the excess loads
# are spill reloads. A spill is a store followed by a reload, so if the excess
# is spillage the store count should be comparable in magnitude to it; if
# stores come back near zero the loads are read-only data (Philox constants,
# the Cholesky matrix, SVML tables) and the spill reading is wrong.
#
# Store bandwidth is half load bandwidth on this core (2 loads/cycle vs 1
# store/cycle), so counts must be converted to port pressure before comparing:
# loads/2 against stores/1. Stores can dominate while being fewer.
#
# resource_stalls.sb (store-buffer full) is the direct test of whether stores
# actually cost cycles rather than just being issued. mem_inst_retired.all_stores
# is collected alongside L1-dcache-stores as a cross-check, since the two count
# slightly different things (retired store uops vs L1 store accesses) and a
# large disagreement would mean one of them is not measuring what we think.
#
# One pass: five events fit the PMU counters without multiplexing. Same
# task-clock,cycles,instructions preamble as the other passes so the run can be
# checked for comparability against them.
kbench_stores: kb_basket_l0
	@mkdir -p $(OUT)/kernel
	@{ echo "=== store-side counters ==="; \
	perf stat -e task-clock,cycles,instructions,\
L1-dcache-loads,L1-dcache-stores,mem_inst_retired.all_stores,resource_stalls.sb \
	  ./kb_basket_l0 --kernel $(KERNEL) $(if $(NPATHS),--n $(NPATHS),) 2>&1; \
	} | tee $(OUT)/kernel/basket_l0_stores_$(KERNEL).txt

# Non-memory IPC (Giles, 2026-08-25): "can you restrict IPC to non-memory
# instructions?".  Subtracting L1-dcache-loads/stores from `instructions`
# mixes units -- those are uop counters, `instructions` is an instruction
# counter, so a fused load+op is 1 instruction but 1 load uop and the
# subtraction over-removes.  mem_inst_retired.all_loads/.all_stores are
# INSTRUCTION-retired counters, so (instructions - all_loads - all_stores)
# is unit-consistent and non-memory IPC is quotable without a caveat.
#
# L1-dcache-* are collected alongside in the same pass so the two definitions
# can be compared directly on identical runs: all_stores already matched
# L1-dcache-stores exactly (231,147,044) in the kbench_stores run, so if
# all_loads likewise matches L1-dcache-loads the earlier subtraction needs no
# correction.  Six events, one pass, no multiplexing.
kbench_nonmem: kb_basket_l0
	@mkdir -p $(OUT)/kernel
	@{ echo "=== non-memory IPC counters ==="; \
	perf stat -e task-clock,cycles,instructions,\
mem_inst_retired.all_loads,mem_inst_retired.all_stores,\
L1-dcache-loads,L1-dcache-stores \
	  ./kb_basket_l0 --kernel $(KERNEL) $(if $(NPATHS),--n $(NPATHS),) 2>&1; \
	} | tee $(OUT)/kernel/basket_l0_nonmem_$(KERNEL).txt

# OpenMP counters (Giles, 2026-09-23): "have you tried using perf with OpenMP
# multi-threaded applications?".  Same three passes and the same event set as
# kbench_perf, but on the -qopenmp binary under OMP_NUM_THREADS, so the serial
# and threaded numbers are directly comparable row for row.
#
# perf stat follows the whole process tree, so the counts it reports are summed
# over ALL threads, not per thread.  Two consequences when reading the output:
#   * `instructions` and the fp_arith_* events should be essentially UNCHANGED
#     from the 1-thread run -- the same total work is divided between threads,
#     not duplicated.  A rise means the parallel build is doing extra work
#     (per-thread setup, a different code path); a fall means threads are
#     idling in the timed region.
#   * `task-clock` is CPU-time summed over threads, so it scales UP with the
#     thread count while elapsed time falls.  IPC stays honest (both numerator
#     and denominator are process-wide totals), but "cycles" is no longer a
#     wall-clock proxy -- read the elapsed line for that.
# OMP_PLACES/OMP_PROC_BIND pin threads as in kbench_scaling, so the counts are
# not polluted by migration.
#
# make kbench_perf_omp KERNEL=fp16 OMPTHREADS=32
OMPTHREADS ?= 32
kbench_perf_omp: kb_basket_l0_omp
	@mkdir -p $(OUT)/kernel
	@{ echo "=== OpenMP counters: $(OMPTHREADS) threads, kernel $(KERNEL) ==="; \
	echo "=== pass 1: SIMD / instruction mix ==="; \
	OMP_NUM_THREADS=$(OMPTHREADS) OMP_PLACES=cores OMP_PROC_BIND=close \
	perf stat -e task-clock,cycles,instructions,\
fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.256b_packed_single,\
fp_arith_inst_retired2.512b_packed_half,fp_arith_inst_retired2.256b_packed_half \
	  ./kb_basket_l0_omp --kernel $(KERNEL) $(if $(NPATHS),--n $(NPATHS),) 2>&1; \
	echo; echo "=== pass 2: L1 / L2 ==="; \
	OMP_NUM_THREADS=$(OMPTHREADS) OMP_PLACES=cores OMP_PROC_BIND=close \
	perf stat -e task-clock,cycles,instructions,\
L1-dcache-loads,L1-dcache-load-misses,l2_rqsts.miss \
	  ./kb_basket_l0_omp --kernel $(KERNEL) $(if $(NPATHS),--n $(NPATHS),) 2>&1; \
	echo; echo "=== pass 3: stores / non-memory ==="; \
	OMP_NUM_THREADS=$(OMPTHREADS) OMP_PLACES=cores OMP_PROC_BIND=close \
	perf stat -e task-clock,cycles,instructions,\
mem_inst_retired.all_loads,mem_inst_retired.all_stores,\
L1-dcache-stores,resource_stalls.sb \
	  ./kb_basket_l0_omp --kernel $(KERNEL) $(if $(NPATHS),--n $(NPATHS),) 2>&1; \
	} | tee $(OUT)/kernel/basket_l0_perf_omp$(OMPTHREADS)_$(KERNEL).txt

# Per-thread counters (Giles, 2026-09-23): shows that the fixed total work is
# DIVIDED between threads rather than duplicated.  perf's --per-thread mode only
# works when attaching to an existing process (-p), not when perf launches the
# workload itself, so the binary is started in the background with enough reps
# to outlive the attach, and perf is pointed at its pid.
#
# NOT -A: that is per-CPU and requires system-wide mode, which would also count
# every other user's work on this shared node.
#
# PERTHREAD_REPS is deliberately large: the attach costs a few hundred ms, so a
# short run would be over before perf starts counting.
PERTHREAD_REPS ?= 200
kbench_perf_omp_percpu: kb_basket_l0_omp
	@mkdir -p $(OUT)/kernel
	@{ echo "=== per-thread breakdown: $(OMPTHREADS) threads, kernel $(KERNEL) ==="; \
	echo "=== (counts are per OpenMP thread; sum = process total) ==="; \
	OMP_NUM_THREADS=$(OMPTHREADS) OMP_PLACES=cores OMP_PROC_BIND=close \
	  ./kb_basket_l0_omp --kernel $(KERNEL) --reps $(PERTHREAD_REPS) \
	  $(if $(NPATHS),--n $(NPATHS),) & \
	pid=$$!; sleep 1; \
	perf stat --per-thread -e cycles,instructions -p $$pid 2>&1; \
	wait $$pid; \
	} | tee $(OUT)/kernel/basket_l0_perthread_omp$(OMPTHREADS)_$(KERNEL).txt

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

# Every counter pass, both kernels, one invocation, into ONE file (2026-08-28).
# The passes were previously run on separate occasions, so Table 4.4's clock
# (~2.70 GHz) and the non-memory IPC pass's clock (~2.00 GHz) came from
# differently loaded machines and the two tables' IPC figures did not agree.
# Counts are ratios and survive that; IPC is a rate and does not.  Taking every
# pass back to back makes the absolute frequency irrelevant, since all rows
# then share it.  Ordering is fp32 then fp16 within each pass rather than all
# passes per kernel, so any drift over the session hits both sides of a
# comparison equally.  Every pass prints task-clock/cycles/instructions, so the
# single output file carries its own comparability check.
# BENCH selects which level-0 kernel binary to profile, so the scalar Asian
# appendix data comes from exactly the passes Chapter 4's basket table does.
BENCH ?= kb_basket_l0
kbench_allcounters: $(BENCH)
	@mkdir -p $(OUT)/kernel
	@{ echo "AVX-512 kernel counters -- $(BENCH) level 0"; \
	   echo "all passes, both kernels, one session"; \
	   date; uptime; echo; \
	   for k in fp32 fp16; do \
	     echo "################ SIMD / instruction mix -- $$k"; \
	     perf stat -e task-clock,cycles,instructions,\
fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.256b_packed_single,\
fp_arith_inst_retired2.512b_packed_half,fp_arith_inst_retired2.256b_packed_half \
	       ./$(BENCH) --kernel $$k $(if $(NPATHS),--n $(NPATHS),) 2>&1; echo; \
	   done; \
	   for k in fp32 fp16; do \
	     echo "################ L1 / L2 -- $$k"; \
	     perf stat -e task-clock,cycles,instructions,\
L1-dcache-loads,L1-dcache-load-misses,l2_rqsts.miss \
	       ./$(BENCH) --kernel $$k $(if $(NPATHS),--n $(NPATHS),) 2>&1; echo; \
	   done; \
	   for k in fp32 fp16; do \
	     echo "################ store side -- $$k"; \
	     perf stat -e task-clock,cycles,instructions,\
L1-dcache-loads,L1-dcache-stores,mem_inst_retired.all_stores,resource_stalls.sb \
	       ./$(BENCH) --kernel $$k $(if $(NPATHS),--n $(NPATHS),) 2>&1; echo; \
	   done; \
	   for k in fp32 fp16; do \
	     echo "################ non-memory IPC -- $$k"; \
	     perf stat -e task-clock,cycles,instructions,\
mem_inst_retired.all_loads,mem_inst_retired.all_stores,\
L1-dcache-loads,L1-dcache-stores \
	       ./$(BENCH) --kernel $$k $(if $(NPATHS),--n $(NPATHS),) 2>&1; echo; \
	   done; \
	   echo "################ timing"; \
	   ./$(BENCH) --sweep $(if $(REPS),--reps $(REPS),) 2>&1; \
	 } | tee $(OUT)/kernel/$(BENCH)_allcounters.txt

.PHONY: all omp everything mkl run run_omp scaling scaling_csv speedup push pull kpull clean \
        kbench kbench_perf kbench_perf_omp kbench_perf_omp_percpu kbench_vtune \
        kbench_scaling kbench_allcounters coupling_test
