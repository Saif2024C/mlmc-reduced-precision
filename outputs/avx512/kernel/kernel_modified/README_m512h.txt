32-lane (__m512h) fp16 kernel vs original 16-lane (__m256h)
basket European level-0, N = 4,194,304, serial, mimic. 2026-08-25.

WHY: norminv_fp16 already computes 32 fp16 lanes from ONE Philox draw;
normal16_fp16_truncated() discarded the upper 16. perf saw 32.87 M
512b-packed-half instrs in the 16-lane build = the spline running
32-wide with half its output thrown away. This build consumes both
halves. No extra RNG call: 512 bits = 32 independent 16-bit uniforms.

                                  16-lane        32-lane    change
instructions                    896624138      442928735     2.02x
FP-arith instr                  134111232       68370432     1.96x
  512b packed-half               32870400       65740800     0.50x
  256b packed-half              101240832        2629632    38.50x
loads                           126558637       57337355     2.21x
stores                           25878285       10635323     2.43x
ns/path                             12.23           6.09     2.01x

Speedup vs fp32 (28.79 ns/path):
  16-lane fp16 : 2.35x
  32-lane fp16 : 4.73x

Advisor (FLOP + INTOP, --flop):
                                  16-lane        32-lane
GFLOPS                               9.90          16.66
GINTOPS                             19.16          19.38
combined GOPS                       29.06          36.04

fp32 (unchanged): memory 40.6% of stream, non-mem instr 1304249755
fp16 32-lane: memory 15.3% of stream, non-mem instr 374956057

IPC (see clock caveat below -- compare within a run, not across)
                                     fp32   fp16 16-lane   fp16 32-lane
IPC (all)                            1.64           1.58           1.56
IPC (non-memory)                     0.98           1.31           1.32
cycles (M)                        1334.68         569.13         284.30

  non-memory IPC = (instructions - all_loads - all_stores) / cycles.
  fp32 leads on raw IPC but trails once memory instructions are
  removed: 40.6% of its stream is loads/stores vs 15.3% for 32-lane
  fp16. Going 16->32 lanes barely moves fp16 IPC (1.58->1.56 all,
  1.31->1.32 non-mem) because instructions AND cycles both roughly
  halve -- the gain shows up as fewer cycles, not a higher rate.

FINDINGS
  1. ns/path 12.23 -> 6.09, almost exactly 2x, for the same RNG cost.
  2. Width shift confirms the mechanism: 512b-packed-half 32.9M -> 65.7M,
     256b-packed-half 101.2M -> 2.6M. The kernel is now genuinely 32-wide.
  3. Instructions halve (896.6M -> 442.9M); FP-arith 134.1M -> 68.4M.
  4. Correctness holds: chk 10.949784 (32-lane) vs 10.948546 (16-lane)
     vs 10.957052 (fp32).
  5. Speedup vs fp32 rises 2.36x -> 4.73x.

NOTE -- NOT a like-for-like replacement for Table 4.4/4.5.
  This kernel retires 32 paths per iteration against fp32's 16, so the
  RNG is amortised over twice as many paths. ns/path no longer isolates
  arithmetic the way the 16-lane benchmark did. The 4.73x is a real
  end-to-end speedup but mixes packing with RNG amortisation; the 16-lane
  2.36x remains the number that isolates precision alone.

  Clock in these passes was 1.49-1.54 GHz (fp16) -- mimic was running
  well below the ~2.7 GHz of the original Table 4.4 collection, so
  absolute IPC/ns are not comparable across those runs. Counts are.

Source: performance_analysis/kernel_bench_basket_l0_m512h.cpp
options_mm512.h deliberately untouched; 32-lane helpers are local to
that file, so the production estimators are unaffected.
