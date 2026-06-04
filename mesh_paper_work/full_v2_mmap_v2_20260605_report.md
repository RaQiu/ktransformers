# full-v2 / mmap-v2 Benchmark Report

Date: 2026-06-05 Asia/Shanghai

This report records the current `full-v2` rerun and the follow-up `mmap-v2` fair-memory experiment. All rows use the standard five-domain prompt file:

`/mnt/data3/work/mesh_standard_5domain_prompts_20260602.json`

Formal performance rows use 5 prompts and `max_new_tokens=768`. The local raw row manifest is:

`/Users/qr/Documents/ktransformers/mesh_paper_work/full_mmap_v2_rows_20260605.jsonl`

## Method

- `full-v2` uses the current latest active KTransformers/MESH tree, backend `FULL`.
- `mmap-v2` uses only the old isolated legacy mmap checkout, `code_version_policy=legacy_old_checkout`.
- CUDA graph was not disabled. For 35B rows, server logs contain decode `cuda graph: True` events; for 397B rows, summaries record `disable_cuda_graph=false`.
- `mmap-v2` fair memory follows the same-cap rule: for each valid MESH cap, use the corresponding MESH memory envelope and round up to a 16 GiB multiple.
- When a post-fix mmap row OOMs under the same-cap budget, the OOM row remains a same-cap diagnostic. A separate 16 GiB budget ladder is reported as `minimum-runnable` data and is not mixed into the same-cap effective table.

## Fair Memory Derivation

| Target | Source MESH summary | MESH cap | MESH peak GiB | mmap-v2 MemoryMax |
|---|---|---:|---:|---:|
| 35B AMXINT4 TP2 | `/mnt/data3/work/mesh_paper_35b_runs/fresh_matrix_20260603_35b_768/35b_amxint4_mesh_tp2/summary.json` | 128 | 19.509 | 32G |
| 35B AMXINT4 TP4 | `/mnt/data3/work/mesh_paper_35b_runs/fresh_matrix_20260603_35b_768/35b_amxint4_mesh_tp4/summary.json` | 128 | 24.474 | 32G |
| 35B BF16 TP2 | `/mnt/data3/work/mesh_paper_35b_runs/continuation_20260604_bf16_mesh_statsfix_5p768/35b_bf16_mesh_tp2/summary.json` | 128 | 56.971 | 64G |
| 35B BF16 TP4 | `/mnt/data3/work/mesh_paper_35b_runs/continuation_20260604_bf16_mesh_statsfix_5p768/35b_bf16_mesh_tp4/summary.json` | 128 | 62.076 | 64G |
| 397B AMXINT4 TP2 | `/mnt/data3/work/mesh_paper_397b_runs/fresh_matrix_20260604_amxint4_tp2_mesh_ge16_cap192_fill64_pool4096_graphon_after_bf16fix_5p768/mesh_cap192_layer/summary.json` | 192 | 92.647 | 96G |
| 397B AMXINT4 TP4 | `/mnt/data3/work/mesh_paper_397b_runs/fresh_matrix_20260604_amxint4_tp4_mesh_ge16_cap192_fill64_pool4096_graphon_after_bf16fix_5p768/mesh_cap192_layer/summary.json` | 192 | 97.791 | 112G |

## Effective Performance Rows

| Model | Precision | Mode | Policy | TP | GE | Cgroup | ok/exp | Decode tok/s | Prefill tok/s | Peak GiB | File GiB | Summary |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 35B | AMXINT4 | full-v2 | current_latest | 2 | 32 | 768G | 5/5 | 53.851 | 682.323 | 21.981 | 0.238 | `/mnt/data3/work/mesh_paper_35b_runs/full_v2_20260605_35b_5p768/35b_amxint4_full_tp2/summary.json` |
| 35B | AMXINT4 | full-v2 | current_latest | 4 | 32 | 768G | 5/5 | 100.857 | 604.636 | 26.970 | 0.586 | `/mnt/data3/work/mesh_paper_35b_runs/full_v2_20260605_35b_5p768/35b_amxint4_full_tp4/summary.json` |
| 35B | BF16 | full-v2 | current_latest | 2 | 32 | 768G | 5/5 | 61.490 | 613.274 | 69.502 | 2.929 | `/mnt/data3/work/mesh_paper_35b_runs/full_v2_20260605_35b_5p768/35b_bf16_full_tp2/summary.json` |
| 35B | BF16 | full-v2 | current_latest | 4 | 32 | 768G | 5/5 | 65.454 | 600.927 | 71.810 | 0.580 | `/mnt/data3/work/mesh_paper_35b_runs/full_v2_20260605_35b_5p768/35b_bf16_full_tp4/summary.json` |
| 397B | AMXINT4 | full-v2 | current_latest | 2 | 16 | 768G | 5/5 | 31.068 | 248.000 | 206.058 | 17.013 | `/mnt/data3/work/mesh_paper_397b_runs/full_v2_20260605_amxint4_tp2_5p768/full/summary.json` |
| 397B | AMXINT4 | full-v2 | current_latest | 4 | 16 | 768G | 5/5 | 32.713 | 239.400 | 194.529 | 0.823 | `/mnt/data3/work/mesh_paper_397b_runs/full_v2_20260605_amxint4_tp4_5p768/full/summary.json` |
| 397B | BF16 | full-v2 | current_latest | 2 | 16 | 768G | 5/5 | 18.100 | 124.261 | 768.000 | 115.296 | `/mnt/data3/work/mesh_paper_397b_runs/full_v2_20260605_bf16_tp2_5p768/full/summary.json` |
| 397B | BF16 | full-v2 | current_latest | 4 | 16 | 768G | 5/5 | 19.055 | 122.377 | 768.000 | 178.078 | `/mnt/data3/work/mesh_paper_397b_runs/full_v2_20260605_bf16_tp4_5p768/full/summary.json` |
| 35B | AMXINT4 | mmap-v2 | legacy_old_checkout | 2 | 32 | 32G | 5/5 | 93.557 | 880.779 | 28.945 | 7.025 | `/mnt/data3/work/mesh_paper_35b_runs/mmap_v2_20260605_35b_amxint4_cap128_mem32_5p768/35b_amxint4_mmap_tp2/summary.json` |
| 35B | AMXINT4 | mmap-v2 | legacy_old_checkout | 4 | 32 | 32G | 5/5 | 101.097 | 875.092 | 27.280 | 0.596 | `/mnt/data3/work/mesh_paper_35b_runs/mmap_v2_20260605_35b_amxint4_cap128_mem32_5p768/35b_amxint4_mmap_tp4/summary.json` |

## Minimum Runnable Ladder Rows

These rows are follow-up runs after a same-cap mmap-v2 OOM. They answer "what budget lets legacy mmap run?" and should not be read as same-cap fair-memory wins.

| Model | Precision | Mode | Policy | TP | GE | Cgroup | ok/exp | Decode tok/s | Prefill tok/s | Peak GiB | File GiB | Ladder note | Summary |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 35B | BF16 | mmap-v2 | legacy_old_checkout | 2 | 32 | 80G | 5/5 | 65.419 | 638.640 | 75.252 | 8.269 | 64G same-cap OOM; 80G passes | `/mnt/data3/work/mesh_paper_35b_runs/mmap_v2_ladder_after_oom_fix_20260605_35b_bf16_5p768/35b_bf16_mmap_tp2_mem80/summary.json` |
| 35B | BF16 | mmap-v2 | legacy_old_checkout | 4 | 32 | 80G | 5/5 | 65.297 | 606.494 | 72.406 | 0.594 | 64G same-cap OOM; 80G passes | `/mnt/data3/work/mesh_paper_35b_runs/mmap_v2_ladder_after_oom_fix_20260605_35b_bf16_5p768/35b_bf16_mmap_tp4_mem80/summary.json` |
| 397B | AMXINT4 | mmap-v2 | legacy_old_checkout | 2 | 16 | 192G | 5/5 | 33.519 | 250.258 | 192.000 | 19.657 | 96G same-cap OOM; 192G passes | `/mnt/data3/work/mesh_paper_397b_runs/mmap_v2_ladder_after_samecap_oom_20260605_amxint4_5p768/397b_amxint4_mmap_tp2_mem192/summary.json` |
| 397B | AMXINT4 | mmap-v2 | legacy_old_checkout | 4 | 16 | 208G | 5/5 | 37.436 | 246.552 | 208.000 | 14.365 | 112G same-cap OOM; 208G passes | `/mnt/data3/work/mesh_paper_397b_runs/mmap_v2_ladder_after_samecap_oom_20260605_amxint4_5p768/397b_amxint4_mmap_tp4_mem208/summary.json` |
| 397B | BF16 | mmap-v2 | legacy_old_checkout | 2 | 16 | 768G | 5/5 | 19.389 | 133.558 | 768.000 | 58.924 | 512G ladder OOM; 768G passes | `/mnt/data3/work/mesh_paper_397b_runs/mmap_v2_ladder_after_samecap_oom_20260605_bf16_5p768/397b_bf16_mmap_tp2_mem768/summary.json` |
| 397B | BF16 | mmap-v2 | legacy_old_checkout | 4 | 16 | 768G | 5/5 | 18.761 | 132.948 | 768.000 | 114.680 | 512G ladder OOM; 768G passes | `/mnt/data3/work/mesh_paper_397b_runs/mmap_v2_ladder_after_samecap_oom_20260605_bf16_5p768/397b_bf16_mmap_tp4_mem768/summary.json` |

## Fix Applied After Initial Diagnostics

The initial BF16 legacy mmap diagnostics failed in the old loader before real model loading, with `No experts found for key model.language_model.layers.0.mlp.experts`. The legacy mmap environment had two `BF16SafeTensorLoader` definitions in the generated loader file: the first already supported Qwen3.5 packed `gate_up_proj/down_proj`, while a later unpacked-only class with the same name overwrote it.

Applied fix:

- Remote source: `/mnt/data3/work/ktransformers_mmap_d7b5b49/kt-kernel/python/utils/loader.py`
- Remote runtime build: `/mnt/data3/work/ktransformers_mmap_d7b5b49/kt-kernel/build/lib.linux-x86_64-cpython-311/kt_kernel/utils/loader.py`
- Local patch record: `/Users/qr/Documents/ktransformers/mesh_paper_work/legacy_mmap_loader_fix/legacy_mmap_bf16_packed_loader_fix.patch`

After the fix, BF16 legacy mmap logs show `Detected format: packed (Qwen3.5 MoE style)`. The remaining BF16 failures are cgroup-limit failures under the same-cap diagnostic budget, not loader key-layout failures.

## Diagnostic / Blocked Rows

| Model | Precision | Mode | Policy | TP | Cgroup | Status | Peak GiB | File GiB | OOM events | Root cause | Summary |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|---|
| 397B | AMXINT4 | mmap-v2 | legacy_old_checkout | 2 | 96G | fail | 96.000 | 4.188 | 1 | Same-cap budget reaches cgroup limit during model loading; remaining TP ranks report likely OOM/slow rank. | `/mnt/data3/work/mesh_paper_397b_runs/mmap_v2_20260605_amxint4_tp2_cap192_mem96_5p768/mmap/summary.json` |
| 397B | AMXINT4 | mmap-v2 | legacy_old_checkout | 4 | 112G | fail | 112.000 | 8.785 | 1 | Same-cap budget reaches cgroup limit during model loading; ranks fail before serving. | `/mnt/data3/work/mesh_paper_397b_runs/mmap_v2_20260605_amxint4_tp4_cap192_mem112_5p768/mmap/summary.json` |
| 35B | BF16 | mmap-v2 after loader fix | legacy_old_checkout | 2 | 64G | fail | 64.000 | 0.166 | 1 | Loader now detects packed Qwen3.5 BF16; same-cap 64G budget reaches cgroup limit before serving. | `/mnt/data3/work/mesh_paper_35b_runs/mmap_v2_after_bf16_loader_fix_20260605_35b_bf16_cap128_mem64_5p768/35b_bf16_mmap_tp2/summary.json` |
| 35B | BF16 | mmap-v2 after loader fix | legacy_old_checkout | 4 | 64G | fail | 64.000 | 0.454 | 1 | Loader now detects packed Qwen3.5 BF16; same-cap 64G budget reaches cgroup limit before serving. | `/mnt/data3/work/mesh_paper_35b_runs/mmap_v2_after_bf16_loader_fix_20260605_35b_bf16_cap128_mem64_5p768/35b_bf16_mmap_tp4/summary.json` |
| 397B | BF16 | mmap-v2 after loader fix | legacy_old_checkout | 2 | 64G | fail | 64.000 | 17.984 | - | Loader now detects packed Qwen3.5 BF16; 64G diagnostic budget reaches cgroup limit during model loading. | `/mnt/data3/work/mesh_paper_397b_runs/mmap_v2_after_bf16_loader_fix_20260605_bf16_tp2_mem64_1tok/mmap/summary.json` |
| 397B | BF16 | mmap-v2 after loader fix | legacy_old_checkout | 4 | 64G | fail | 64.000 | 8.906 | - | Loader now detects packed Qwen3.5 BF16; 64G diagnostic budget reaches cgroup limit during model loading. | `/mnt/data3/work/mesh_paper_397b_runs/mmap_v2_after_bf16_loader_fix_20260605_bf16_tp4_mem64_1tok/mmap/summary.json` |
| 397B | BF16 | mmap-v2 ladder | legacy_old_checkout | 2 | 512G | fail | 512.000 | 75.658 | 1 | 512G ladder budget reaches cgroup limit during expert materialization; 768G passes. | `/mnt/data3/work/mesh_paper_397b_runs/mmap_v2_ladder_after_samecap_oom_20260605_bf16_5p768/397b_bf16_mmap_tp2_mem512/summary.json` |
| 397B | BF16 | mmap-v2 ladder | legacy_old_checkout | 4 | 512G | fail | 512.000 | 133.989 | 1 | 512G ladder budget reaches cgroup limit during expert materialization; 768G passes. | `/mnt/data3/work/mesh_paper_397b_runs/mmap_v2_ladder_after_samecap_oom_20260605_bf16_5p768/397b_bf16_mmap_tp4_mem512/summary.json` |

## Readout

The `full-v2` rerun is complete for all eight requested full rows. The corrected full path now gives 35B AMXINT4 TP4 around 100.9 tok/s, while 35B AMXINT4 TP2 remains much slower at 53.9 tok/s. For BF16, 35B TP2/TP4 are close, while 397B BF16 reaches the 768G cgroup ceiling on both TP settings.

Under the same-cap fair memory rule, 35B AMXINT4 legacy mmap still runs at 32G and reaches roughly 94-101 tok/s. In contrast, 397B AMXINT4 legacy mmap cannot start under the cap192-equivalent budgets of 96G and 112G; both runs hit the cgroup limit during model loading before any prompt completes. The minimum-runnable ladder shows that the same legacy mmap path runs once raised to 192G for TP2 and 208G for TP4, with throughput close to the prior high-budget mmap measurements.

The BF16 legacy mmap loader bug was fixed after the initial diagnostics. Post-fix BF16 mmap no longer fails on missing expert keys; it recognizes packed Qwen3.5 BF16 weights. For 35B BF16, the same-cap 64G rows OOM, while 80G completes both TP2 and TP4 at about 65 tok/s. For 397B BF16, 512G still OOMs during expert materialization, while 768G completes both TP2 and TP4 at about 19 tok/s.
