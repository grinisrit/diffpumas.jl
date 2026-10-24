# Exact raw-flux cache microbenchmark (preliminary, no production deployment)

Frozen dry and accepted material-corrected contexts, 600 cells/1003 covered rays, single CPU20/5GiB. Two timing orders after warmup; no objective, tolerance or solver changes. All 24 raw predictions were bit-identical and their residual/profile normalization/objective outputs identical. A representative on-path sparse gradient was unchanged before/after cache calls.

Only 31 rays cross no supported cell. Whole-field transitions between completed dry0 and dry.15 reused 251/1003 rays: median raw/cache seconds 5.275/3.708 (dry), 5.362/3.602 (corrected). These are callback timings, not end-to-end solver speedups. Single-cell corner perturbations reuse 994 rays and are deliberately NOT representative of complete GN directions. An exact real trial/accepted-step replay and guarded threaded implementation remain separate required gates.

The serial prototype's artifact-string context token is permitted only because each isolated context has an immutable closure lifetime. It is insufficient as a general production context guard.

The first attempt failed at the final Julia file guard before physics work. Its original helper, log and clean failed end audit are retained. The corrected helper ran in a fresh output directory. All completed output and source/input hashes were independently rechecked when this archive was created.
