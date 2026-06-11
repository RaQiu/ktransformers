# SPDX-License-Identifier: Apache-2.0
"""Frequency-based GPU expert placement.

Decides, per layer, which experts should live in the GPU weight buffers,
based on routing statistics collected during a profiling prefill run (the
JSONL dump written by operators/mesh when KT_MESH_PREFILL_EXPERT_FREQ_PATH
is set).

Two input formats are accepted by select_gpu_experts_for_layer:

1. Raw frequency dump (JSONL): one record per (layer, prefill call), each
   carrying full-expert ``router_counts`` and ``cpu_counts`` histograms.
   Records of the same layer are summed, so multi-chunk / multi-request
   profiling runs accumulate naturally.
2. Compiled placement file (single JSON object with a ``layers`` map),
   built offline by scripts/mesh/build_gpu_expert_placement.py. Preferred
   for serving: explicit, reviewable, and validated at build time.

Metric choice ("router" vs "cpu"):
- ``router_counts`` counts every routed token regardless of where the expert
  currently runs. It is placement-independent, so placements can be iterated
  (profiling with a placement active does not make GPU experts disappear
  from the statistics). This is the default.
- ``cpu_counts`` is exactly the per-prefill increment that operators/mesh
  accumulates into expert_frequency_score_ (the eviction score): CPU-routed
  tokens only. Identical to router_counts when profiling with zero GPU
  experts; under a GE>0 profiling run, GPU-resident experts score zero here.

This module is pure stdlib so offline scripts can load it without the
compiled kt_kernel extension.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

VALID_PLACEMENT_METRICS = ("router", "cpu")

_METRIC_FIELD = {"router": "router_counts", "cpu": "cpu_counts"}

# Parsed placement sources keyed by (path, metric). Value is one of:
#   {"kind": "placement", "layers": {layer: [ids]}, "counts": {layer: [counts]}}
#   {"kind": "freq", "layers": {layer: [counts]}}
_source_cache: Dict[Tuple[str, str], dict] = {}
_cache_lock = threading.Lock()


def aggregate_frequency_dump(path: str, metric: str = "router") -> Dict[int, List[int]]:
    """Sum per-layer expert counts across all records of a frequency dump."""
    field = _METRIC_FIELD[metric]
    counts_by_layer: Dict[int, List[int]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            layer = record.get("layer")
            counts = record.get(field)
            if not isinstance(layer, int) or not isinstance(counts, list):
                continue
            acc = counts_by_layer.get(layer)
            if acc is None:
                counts_by_layer[layer] = [int(v) for v in counts]
            elif len(acc) == len(counts):
                for i, v in enumerate(counts):
                    acc[i] += int(v)
    return counts_by_layer


def select_top_experts(counts: List[int], num_gpu_experts: int) -> List[int]:
    """Rank experts by count desc / expert id asc and return the sorted top set.

    The tie-break mirrors cpu_managed_experts_by_frequency in
    operators/mesh/prefill_policy.hpp so offline selection and any future
    in-run selection agree on equal scores.
    """
    ranked = sorted(range(len(counts)), key=lambda e: (-counts[e], e))
    return sorted(ranked[: max(0, num_gpu_experts)])


def _parse_source(path: str, metric: str) -> dict:
    """Detect and parse the placement source file (compiled or raw dump)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            whole = fh.read()
    except OSError as exc:
        logger.warning("GPU expert placement: cannot read %s (%s)", path, exc)
        return {"kind": "freq", "layers": {}}

    try:
        obj = json.loads(whole)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict) and isinstance(obj.get("layers"), dict):
        layers: Dict[int, List[int]] = {}
        for key, ids in obj["layers"].items():
            try:
                layers[int(key)] = [int(e) for e in ids]
            except (TypeError, ValueError):
                continue
        counts: Dict[int, List[int]] = {}
        raw_counts = obj.get("counts")
        if isinstance(raw_counts, dict):
            for key, values in raw_counts.items():
                try:
                    counts[int(key)] = [int(v) for v in values]
                except (TypeError, ValueError):
                    continue
        logger.info(
            "GPU expert placement: loaded compiled placement %s "
            "(layers=%d, num_gpu_experts=%s, metric=%s)",
            path,
            len(layers),
            obj.get("num_gpu_experts"),
            obj.get("metric"),
        )
        return {"kind": "placement", "layers": layers, "counts": counts}

    return {"kind": "freq", "layers": aggregate_frequency_dump(path, metric)}


def _get_source(path: str, metric: str) -> dict:
    key = (path, metric)
    with _cache_lock:
        cached = _source_cache.get(key)
        if cached is None:
            cached = _parse_source(path, metric)
            _source_cache[key] = cached
        return cached


def select_gpu_experts_for_layer(
    path: str,
    layer_idx: int,
    num_experts: int,
    num_gpu_experts: int,
    metric: str = "router",
) -> Optional[List[int]]:
    """Return the sorted global expert IDs to place on GPU for one layer.

    Returns None when the source cannot answer for this layer (missing file,
    missing layer, count mismatch) so the caller can fall back to legacy
    prefix placement.
    """
    if num_gpu_experts <= 0:
        return None
    if metric not in VALID_PLACEMENT_METRICS:
        logger.warning(
            "GPU expert placement: unknown metric %r, using 'router'", metric
        )
        metric = "router"

    source = _get_source(path, metric)
    if source["kind"] == "placement":
        selected = source["layers"].get(layer_idx)
        if (
            selected is None
            or len(selected) != num_gpu_experts
            or any(e < 0 or e >= num_experts for e in selected)
            or len(set(selected)) != len(selected)
        ):
            logger.warning(
                "GPU expert placement: compiled file %s has no valid entry for "
                "layer %d (need %d unique ids < %d)",
                path,
                layer_idx,
                num_gpu_experts,
                num_experts,
            )
            return None
        return sorted(selected)

    counts = source["layers"].get(layer_idx)
    if counts is None or len(counts) != num_experts:
        logger.warning(
            "GPU expert placement: layer %d missing or expert count mismatch in %s",
            layer_idx,
            path,
        )
        return None
    selected = select_top_experts(counts, num_gpu_experts)
    total = sum(counts)
    covered = sum(counts[e] for e in selected)
    logger.info(
        "GPU expert placement: layer %d selected %d/%d experts by '%s' frequency, "
        "routed-token coverage %.1f%%",
        layer_idx,
        len(selected),
        num_experts,
        metric,
        100.0 * covered / total if total > 0 else 0.0,
    )
    return selected


def layer_counts_from_source(
    path: str, layer_idx: int, num_experts: int, metric: str = "router"
) -> Optional[List[int]]:
    """Per-expert counts for one layer, or None when unavailable.

    Raw dumps are aggregated with the requested metric. Compiled placement
    files answer from their embedded ``counts`` histograms (aggregated with
    the build-time metric; the metric argument does not re-derive them).
    Compiled files built before counts embedding return None.
    """
    if not path:
        return None
    if metric not in VALID_PLACEMENT_METRICS:
        metric = "router"
    source = _get_source(path, metric)
    if source["kind"] == "freq":
        counts = source["layers"].get(layer_idx)
    else:
        counts = source.get("counts", {}).get(layer_idx)
    if counts is None or len(counts) != num_experts:
        return None
    return counts


def partition_cpu_experts_for_rank(
    cpu_expert_ids: List[int],
    rank: int,
    world_size: int,
    counts: Optional[List[int]] = None,
) -> List[int]:
    """Deterministically partition CPU-managed experts across moe-TP ranks.

    Every rank computes the same assignment independently (pure function of
    the inputs), so the rank partitions are mutually exclusive and jointly
    cover cpu_expert_ids — the invariant that makes the post-MoE all-reduce
    sum each CPU expert's output exactly once.

    With per-expert counts, experts are ranked hot-to-cold (count desc, id asc
    — same tie-break as select_top_experts) and dealt snake-wise
    (0,1,..,w-1,w-1,..,1,0,...) so each rank receives a balanced share of hot
    experts. Without counts, plain round-robin over ascending expert ids.
    """
    if world_size <= 1:
        return sorted(cpu_expert_ids)
    if rank < 0 or rank >= world_size:
        return []

    if counts is not None:
        ordered = sorted(cpu_expert_ids, key=lambda e: (-counts[e], e))
    else:
        ordered = sorted(cpu_expert_ids)

    mine: List[int] = []
    for i, expert_id in enumerate(ordered):
        block, j = divmod(i, world_size)
        owner = j if (counts is None or block % 2 == 0) else world_size - 1 - j
        if owner == rank:
            mine.append(expert_id)
    return sorted(mine)


def split_capacity_for_rank(total: int, rank: int, world_size: int) -> int:
    """Split a resident-slot budget across moe-TP ranks.

    Used for caps that were sized for a single CPU backend process
    (KT_MESH_GLOBAL_POOL_CAPACITY, KT_MAX_RESIDENT_EXPERTS,
    KT_MAX_TIER0_EXPERTS) when CPU experts are partitioned across ranks:
    each rank keeps total/world_size slots, remainder to the lowest ranks,
    so the machine-wide total stays equal to the single-backend setting
    instead of multiplying by world_size.

    Out-of-range ranks get 0 (caller bug guard). Valid ranks never get less
    than 1: several consumers treat 0 as "unset" (the C++ pool falls back to
    its default capacity, KT_MAX_RESIDENT_EXPERTS=0 means unlimited), which
    would silently blow the budget instead of shrinking it.
    """
    if world_size <= 1:
        return total
    if rank < 0 or rank >= world_size:
        return 0
    base, rem = divmod(total, world_size)
    return max(1, base + (1 if rank < rem else 0))
