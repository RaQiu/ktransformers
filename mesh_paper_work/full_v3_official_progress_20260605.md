# full-v3 official KT rerun progress, 2026-06-05

Updated 2026-06-06: true `full-v3` rows now use the clean upstream `kvcache-ai/ktransformers` checkout. Older rows from the pre-kvcache/RaQiu-style attempts remain listed only as invalid diagnostics.

## Correct Definition

`full-v3` means: rerun a fresh isolated checkout of upstream official KTransformers from `https://github.com/kvcache-ai/ktransformers.git`. It is not a mode inside our current MESH plugin tree, and it is not allowed to inherit MESH runtime flags, MESHIO/file-slot paths, MESH cap, pool cap, prefill window, or current-tree plugin code.

If the official checkout cannot run a requested model/precision/TP point without those plugin-only paths, the row is `blocked` or `diagnostic`, not `full-v3`.

## Valid effective rows

These rows were launched from `/mnt/data2/tmp/qujing_full_v3/ktransformers_kvcache_official_clean_20260606` at upstream commit `c1cb22311bffb402c80fd3758f07d916cd148948`, with SGLang submodule commit `51032b71279d9038058563f8d2e758d99b278ef4`.

| Model | Precision | TP | OK/Expected | Max new tokens | API total tok/s | Log decode tok/s | Log prefill tok/s | Peak GiB | Anon GiB | File GiB | Contamination | Summary |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5-35B-A3B | AMXINT4 | 2 | 5/5 | 768 | 87.423 | 87.978 | 16.957 | 33.458 | 21.649 | 11.587 | 0 MESH markers audited in log | `/mnt/data3/work/mesh_paper_35b_runs/full_v3_kvcache_official_20260606/formal_35b_amxint4_tp2_5p768/35b_amxint4_full-v3_tp2/summary.json` |
| Qwen3.5-35B-A3B | AMXINT4 | 4 | 5/5 | 768 | 94.959 | 95.784 | 18.466 | 28.091 | 27.179 | 0.588 | 0 MESH markers audited in log | `/mnt/data3/work/mesh_paper_35b_runs/full_v3_kvcache_official_20260606/formal_35b_amxint4_tp4_5p768/35b_amxint4_full-v3_tp4/summary.json` |
| Qwen3.5-397B-A17B | AMXINT4 | 2 | 5/5 | 768 | 31.722 | 31.859 | 6.271 | 189.180 | 188.067 | 0.254 | 0 | `/mnt/data3/work/mesh_paper_397b_runs/full_v3_kvcache_official_20260606/formal_397b_amxint4_tp2_5p768_after_visionpatch/397b_amxint4_amxint4_full-v3_tp2/summary.json` |
| Qwen3.5-397B-A17B | AMXINT4 | 4 | 5/5 | 768 | 35.965 | 36.087 | 6.945 | 195.286 | 193.711 | 0.596 | 0 | `/mnt/data3/work/mesh_paper_397b_runs/full_v3_kvcache_official_20260606/formal_397b_amxint4_tp4_5p768_after_visionpatch/397b_amxint4_amxint4_full-v3_tp4/summary.json` |

397B sanity note: the valid official full-v3 rows peak at 189.180 GiB and 195.286 GiB. The older 22--29 GiB rows below are invalid because their logs prove MESH/file-slot execution.

## Invalidated rows that were previously mislabeled

| Claimed row | TP | OK/Expected | Max new tokens | Total tok/s (API) | Decode tok/s (log) | Peak GiB | Why invalid | Summary |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Qwen3.5-35B-A3B AMXINT4 claimed full-v3 | 2 | 5/5 | 768 | 1.680 | 1.767 | 9.983 | Pre-kvcache/RaQiu-style official-control attempt superseded by the clean `kvcache-ai` rows above. Do not use this row for `full-v3`. | `/mnt/data3/work/mesh_paper_35b_runs/full_v3_official_20260605/formal_amxint4_tp2_tp4_5p768_ignoreeos_reqpool2_hybridpatch_b826763/35b_amxint4_full-v3_tp2/summary.json` |
| Qwen3.5-35B-A3B AMXINT4 claimed full-v3 | 4 | 5/5 | 768 | 1.879 | 1.931 | 16.070 | Pre-kvcache/RaQiu-style official-control attempt superseded by the clean `kvcache-ai` rows above. Do not use this row for `full-v3`. | `/mnt/data3/work/mesh_paper_35b_runs/full_v3_official_20260605/formal_amxint4_tp2_tp4_5p768_ignoreeos_reqpool2_hybridpatch_b826763/35b_amxint4_full-v3_tp4/summary.json` |
| Qwen3.5-397B-A17B AMXINT4 claimed full-v3 | 2 | 5/5 | 768 | 0.564 | 0.568 | 22.850 | Invalid by log evidence: the server log contains `MESHIO` 120 times, `file_slots` 180 times, and `io_uring` 60 times, including `TP Load from io_uring file slots`. This is a MESH/file-slot row, not official KT full-v3. | `/mnt/data3/work/mesh_paper_397b_runs/full_v3_official_20260605/formal_amxint4_tp2_5p768_processorfallback_9d8bbf8/397b_amxint4_full-v3_tp2/summary.json` |
| Qwen3.5-397B-A17B AMXINT4 claimed full-v3 | 4 | 5/5 | 768 | 0.620 | 0.624 | 28.641 | Invalid by log evidence: the server log contains `MESHIO` 120 times, `file_slots` 180 times, and `io_uring` 60 times, including `TP Load from io_uring file slots`. This is a MESH/file-slot row, not official KT full-v3. | `/mnt/data3/work/mesh_paper_397b_runs/full_v3_official_20260605/formal_amxint4_tp4_5p768_processorfallback_9d8bbf8/397b_amxint4_full-v3_tp4/summary.json` |

## Other diagnostic invalid rows

- early-stop run without `ignore_eos=true`, not fixed 768: `/mnt/data3/work/mesh_paper_35b_runs/full_v3_official_20260605/formal_amxint4_tp2_tp4_5p768_reqpool2_hybridpatch_af21430`
- req pool hang with `--max-running-requests 1`: `/mnt/data3/work/mesh_paper_35b_runs/full_v3_official_20260605/smoke_amxint4_tp1_8tok_clean_official_ktfix_b35dc82/35b_amxint4_full-v3_tp1/summary.json`
- CUDA graph `out_cache_loc` interface failure before hybrid patch: `/mnt/data3/work/mesh_paper_35b_runs/full_v3_official_20260605/smoke_amxint4_tp1_8tok_reqpool2_4c5b4a4/35b_amxint4_full-v3_tp1/summary.json`

## Required cleanup

- Use only the clean `kvcache-ai` official checkout for new `full-v3` rows: `/mnt/data2/tmp/qujing_full_v3/ktransformers_kvcache_official_clean_20260606`.
- The run record must include repository URL, commit, clone path, venv, SGLang path, exact command, cgroup samples, and the standard 5-domain prompt file.
- Add a validity gate: any `full-v3` command or log containing `MESHIO`, `file_slots`, `io_uring` resident-cache loading, MESH cap/pool/window envs, or current MESH plugin checkout paths is automatically invalid.
