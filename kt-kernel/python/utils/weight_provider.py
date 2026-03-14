# Weight provider abstraction for tiered MoE weight management
# SPDX-License-Identifier: Apache-2.0

"""
Tiered weight provider for MoE expert weights.

Implements a three-tier caching strategy:
  Tier 0: NUMA-local malloc buffers (hottest experts, ~80ns access)
          Managed by C++ promote_expert/demote_expert on the MOE object.
  Tier 1: mmap pages resident in RAM (warm experts, ~80-150ns, OS-managed)
  Tier 2: mmap pages on disk (cold experts, ~100us on page fault)

This is essential when model_size >= physical_ram to avoid swap thrashing.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Try to import madvise support
_libc = None
MADV_WILLNEED = 3
TIERED_BACKENDS = frozenset({"LLAMAFILE", "AMXINT4", "AMXINT8", "MOE_INT4", "MOE_INT8", "BF16"})

try:
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
except OSError:
    try:
        _libc = ctypes.CDLL("libc.dylib", use_errno=True)
    except OSError:
        pass


def _madvise_willneed(addr: int, length: int):
    """Hint to the OS that the given address range will be needed soon."""
    if _libc is None or addr == 0 or length <= 0:
        return

    page_size = os.sysconf("SC_PAGESIZE")
    aligned_addr = addr & ~(page_size - 1)
    aligned_end = (addr + length + page_size - 1) & ~(page_size - 1)
    aligned_length = aligned_end - aligned_addr
    _libc.madvise(ctypes.c_void_p(aligned_addr), ctypes.c_size_t(aligned_length), ctypes.c_int(MADV_WILLNEED))


def get_available_ram_bytes() -> int:
    """Get available physical RAM in bytes."""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    # Fallback: read from /proc/meminfo on Linux
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB to bytes
    except (FileNotFoundError, ValueError):
        pass
    # Last resort: assume 64GB
    return 64 * 1024**3


def get_total_ram_bytes() -> int:
    """Get total physical RAM in bytes."""
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        pass
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ValueError):
        pass
    return 64 * 1024**3


def estimate_model_weight_bytes(
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    bytes_per_element: float = 0.5,  # Q4 = 0.5 bytes/element
) -> int:
    """Estimate total MoE expert weight size in bytes."""
    # Each expert has 3 projections: gate, up (hidden×intermediate), down (intermediate×hidden)
    elements_per_expert = 2 * hidden_size * intermediate_size + intermediate_size * hidden_size
    total_elements = num_layers * num_experts * elements_per_expert
    return int(total_elements * bytes_per_element)


def method_bytes_per_element(method: Optional[str]) -> float:
    """Return an approximate bytes/element ratio for a backend's expert weights."""
    if method is None:
        return 0.5

    normalized = method.upper()
    if normalized in {"LLAMAFILE", "AMXINT4", "MOE_INT4", "RAWINT4"}:
        return 0.5
    if normalized in {"AMXINT8", "MOE_INT8", "FP8", "FP8_PERCHANNEL"}:
        return 1.0
    if normalized == "BF16":
        return 2.0
    return 0.5


def should_use_tiered_strategy(
    model_bytes: int,
    total_ram: Optional[int] = None,
    available_ram: Optional[int] = None,
    threshold: float = 0.7,
    safety_bytes: int = 4 * 1024**3,
) -> bool:
    """Determine whether to use tiered (mmap) or legacy (malloc) strategy."""
    if available_ram is None:
        available_ram = get_available_ram_bytes()
    if model_bytes > max(0, available_ram - safety_bytes):
        return True
    if total_ram is None:
        total_ram = get_total_ram_bytes()
    return model_bytes >= total_ram * threshold


def backend_supports_tiered_strategy(method: Optional[str]) -> bool:
    """Return whether a backend supports mmap-backed expert residency control."""
    if method is None:
        return False
    return method.upper() in TIERED_BACKENDS


def resolve_weight_strategy(
    requested_strategy: Optional[str],
    *,
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    bytes_per_element: float = 0.5,
    total_ram_bytes: Optional[int] = None,
    available_ram_bytes: Optional[int] = None,
    threshold: float = 0.7,
    safety_bytes: int = 4 * 1024**3,
) -> Tuple[str, int, int]:
    """Resolve "auto" into a concrete strategy using estimated model size."""
    strategy = requested_strategy or "tiered"
    if strategy != "auto":
        total_ram = get_total_ram_bytes() if total_ram_bytes is None else total_ram_bytes
        return strategy, 0, total_ram

    total_ram = get_total_ram_bytes() if total_ram_bytes is None else total_ram_bytes
    available_ram = get_available_ram_bytes() if available_ram_bytes is None else available_ram_bytes
    model_bytes = estimate_model_weight_bytes(
        num_layers=num_layers,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bytes_per_element=bytes_per_element,
    )
    resolved = (
        "tiered"
        if should_use_tiered_strategy(
            model_bytes,
            total_ram=total_ram,
            available_ram=available_ram,
            threshold=threshold,
            safety_bytes=safety_bytes,
        )
        else "legacy"
    )
    return resolved, model_bytes, total_ram


def resolve_backend_weight_strategy(
    method: Optional[str],
    requested_strategy: Optional[str],
    *,
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    total_ram_bytes: Optional[int] = None,
    available_ram_bytes: Optional[int] = None,
    threshold: float = 0.7,
    safety_bytes: int = 4 * 1024**3,
) -> Tuple[str, int, int]:
    """Resolve a requested strategy while respecting backend residency capabilities."""
    strategy = requested_strategy or "auto"
    total_ram = get_total_ram_bytes() if total_ram_bytes is None else total_ram_bytes
    if not backend_supports_tiered_strategy(method):
        if strategy in {"auto", "tiered"}:
            return "legacy", 0, total_ram
        return strategy, 0, total_ram

    return resolve_weight_strategy(
        strategy,
        num_layers=num_layers,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bytes_per_element=method_bytes_per_element(method),
        total_ram_bytes=total_ram,
        available_ram_bytes=available_ram_bytes,
        threshold=threshold,
        safety_bytes=safety_bytes,
    )


def resolve_auto_tier0_budget_bytes(
    *,
    model_bytes: int,
    total_ram_bytes: Optional[int] = None,
    available_ram_bytes: Optional[int] = None,
    safety_bytes: int = 4 * 1024**3,
) -> int:
    """
    Pick a Tier0 NUMA budget automatically.

    The baseline assumption is:
      - Tier1/Tier2 live in mmap
      - Tier0 is a bounded NUMA-local hotset

    More free headroom allows a larger hotset; higher model pressure shrinks it.
    """
    total_ram = get_total_ram_bytes() if total_ram_bytes is None else total_ram_bytes
    available_ram = get_available_ram_bytes() if available_ram_bytes is None else available_ram_bytes
    headroom_bytes = max(0, available_ram - safety_bytes)
    if headroom_bytes <= 0:
        return 0

    pressure = model_bytes / max(float(total_ram), 1.0)
    if pressure <= 0.70:
        tier0_fraction = 1.0
    elif pressure <= 1.00:
        tier0_fraction = 0.50
    elif pressure <= 1.50:
        tier0_fraction = 0.25
    else:
        tier0_fraction = 0.10

    return int(max(0.0, min(float(headroom_bytes), headroom_bytes * tier0_fraction)))


class ExpertHotnessTracker:
    """
    Track expert activation frequency to identify hot experts for Tier 0 promotion.

    Uses an exponential moving average to adapt to changing access patterns.
    Thread-safe for concurrent recording from inference threads.
    """

    def __init__(self, num_experts: int, ema_alpha: float = 0.01):
        self.num_experts = num_experts
        self.ema_alpha = ema_alpha
        self.counts = np.zeros(num_experts, dtype=np.float64)
        self._lock = threading.Lock()

    def record(self, expert_ids: np.ndarray):
        """
        Record expert activations from a forward pass (vectorized).

        expert_ids: flat array of activated expert IDs (may contain duplicates).
        Uses np.add.at for O(n) vectorized accumulation instead of Python for-loop.
        """
        if expert_ids.size == 0:
            return
        # Filter valid IDs
        valid = expert_ids[(expert_ids >= 0) & (expert_ids < self.num_experts)]
        if valid.size == 0:
            return

        # Build per-expert hit counts in one vectorized pass
        hits = np.zeros(self.num_experts, dtype=np.float64)
        np.add.at(hits, valid, 1.0)
        # Normalize: each activated expert gets alpha, regardless of how many tokens hit it
        mask = hits > 0

        with self._lock:
            # EMA update: decay all, then boost activated ones
            self.counts *= (1 - self.ema_alpha)
            self.counts[mask] += self.ema_alpha

    def get_top_k(self, k: int) -> List[int]:
        """Return indices of the top-k hottest experts."""
        with self._lock:
            if k <= 0:
                return []
            k = min(k, self.num_experts)
            return np.argsort(self.counts)[-k:][::-1].tolist()

    def decay(self):
        """Decay all counts (call periodically to let cold experts fade)."""
        with self._lock:
            self.counts *= (1 - self.ema_alpha)


class MmapWeightRegion:
    """
    Manages a zero-copy mmap view into a GGUF weight file.

    Keeps the numpy mmap handle alive and provides raw pointer access.
    """

    def __init__(self, mmap_data: np.ndarray, offset: int, n_bytes: int):
        # Create a view into the mmap region (no copy)
        self._view = np.frombuffer(mmap_data[offset : offset + n_bytes], dtype=np.uint8)
        self.ptr = self._view.ctypes.data
        self.n_bytes = n_bytes

    def prefetch(self):
        """Hint to OS that this region will be needed soon."""
        _madvise_willneed(self.ptr, self.n_bytes)


def compute_max_tier0_experts(
    tier0_memory_bytes: int,
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    bytes_per_element: float = 0.5,  # Q4_K ≈ 0.5 bytes/element
) -> int:
    """
    Compute how many experts can fit in the Tier 0 NUMA-local memory budget.

    Tier 0 promotes the same expert IDs across ALL layers, so total memory is:
        max_tier0_experts * num_layers * per_expert_per_layer_bytes

    Args:
        tier0_memory_bytes: Total memory budget for Tier 0 in bytes
        num_layers: Number of MoE layers in the model
        num_experts: Total experts per layer (for clamping)
        hidden_size: Model hidden size (e.g., 7168)
        intermediate_size: Expert intermediate size (e.g., 2048)
        bytes_per_element: Quantization ratio (Q4_K≈0.5, Q8≈1.0, BF16≈2.0)

    Returns:
        Maximum number of experts that can be promoted to Tier 0
    """
    # Each expert has 3 projections: gate(H×I), up(H×I), down(I×H)
    per_expert_per_layer = int(3 * hidden_size * intermediate_size * bytes_per_element)
    if per_expert_per_layer <= 0 or num_layers <= 0:
        return 0
    if tier0_memory_bytes <= 0:
        return 0
    per_expert_total = per_expert_per_layer * num_layers
    max_experts = int(tier0_memory_bytes / per_expert_total)
    return max(0, min(max_experts, num_experts))


class TieredWeightProvider:
    """
    Three-tier weight manager for MoE experts.

    Tier 0 is managed entirely by C++ (NUMA-local malloc + pointer swap).
    This Python class orchestrates which experts to promote/demote by calling
    moe.promote_expert(eid) / moe.demote_expert(eid) on the C++ MOE objects.

    Usage:
        provider = TieredWeightProvider(num_experts=256, num_layers=60)
        # After creating MOE object in load_weights():
        provider.register_moe(layer_idx, moe_object)
        # During inference:
        provider.prefetch_layer(layer_idx, topk_ids)
        provider.record_activations(layer_idx, topk_ids)
    """

    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        max_tier0_experts: int = 30,
        promotion_interval_sec: float = 5.0,
    ):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.max_tier0_experts = max_tier0_experts
        self.promotion_interval_sec = promotion_interval_sec

        # Per-layer mmap regions: layer_idx → proj_name → list[expert_id] → list[MmapWeightRegion]
        self.mmap_regions: Dict[int, Dict[str, List[List[MmapWeightRegion]]]] = {}

        # C++ MOE object references: layer_idx → MOE object (has promote/demote methods)
        self.moe_refs: Dict[int, object] = {}
        self.layer_gpu_expert_masks: Dict[int, np.ndarray] = {}

        # Hotness tracker (global across layers — experts that are hot in one layer
        # tend to be hot in others due to token routing)
        self.hotness = ExpertHotnessTracker(num_experts)

        # Background promotion thread
        self._running = False
        self._promotion_thread: Optional[threading.Thread] = None

    def register_moe(self, layer_idx: int, moe: object, gpu_experts_mask: Optional[np.ndarray] = None):
        """
        Register a C++ MOE object for a layer.

        The MOE object must have promote_expert(eid), demote_expert(eid),
        and is_expert_promoted(eid) methods (exposed via pybind11).
        """
        self.moe_refs[layer_idx] = moe
        if gpu_experts_mask is None:
            self.layer_gpu_expert_masks[layer_idx] = np.zeros(self.num_experts, dtype=np.bool_)
        else:
            mask = np.asarray(gpu_experts_mask, dtype=np.bool_).reshape(-1)
            if mask.size != self.num_experts:
                raise ValueError(
                    f"gpu_experts_mask for layer {layer_idx} has size {mask.size}, expected {self.num_experts}"
                )
            self.layer_gpu_expert_masks[layer_idx] = mask.copy()

        # Lazy start: only launch promotion thread once at least one MOE is registered.
        # This avoids running the thread when no backends support promote/demote (e.g., AMX).
        if self.max_tier0_experts > 0 and not self._running:
            self.start_promotion_thread()

    def unregister_moe(self, layer_idx: int):
        """Remove all state associated with a layer's MOE object and mmap regions."""
        self.moe_refs.pop(layer_idx, None)
        self.layer_gpu_expert_masks.pop(layer_idx, None)
        self.mmap_regions.pop(layer_idx, None)
        if not self.moe_refs:
            self.stop_promotion_thread()
            self.hotness.counts.fill(0.0)

    def clear_layer_regions(self, layer_idx: int):
        """Drop mmap-region metadata for a layer before re-registering fresh slices."""
        self.mmap_regions.pop(layer_idx, None)

    def register_mmap_region(
        self, layer_idx: int, proj_name: str, expert_id: int, region: MmapWeightRegion
    ):
        """Register an mmap region for a specific expert weight."""
        if layer_idx not in self.mmap_regions:
            self.mmap_regions[layer_idx] = {}
        if proj_name not in self.mmap_regions[layer_idx]:
            self.mmap_regions[layer_idx][proj_name] = [[] for _ in range(self.num_experts)]
        self.mmap_regions[layer_idx][proj_name][expert_id].append(region)

    def _filter_cpu_expert_ids(self, layer_idx: int, expert_ids: np.ndarray) -> np.ndarray:
        """Remove invalid IDs and experts that are assigned to GPU on this layer."""
        valid = expert_ids[(expert_ids >= 0) & (expert_ids < self.num_experts)]
        if valid.size == 0:
            return valid
        gpu_mask = self.layer_gpu_expert_masks.get(layer_idx)
        if gpu_mask is None:
            return valid
        return valid[~gpu_mask[valid]]

    def _is_gpu_expert(self, layer_idx: int, expert_id: int) -> bool:
        """Check whether an expert is assigned to GPU for the given layer."""
        gpu_mask = self.layer_gpu_expert_masks.get(layer_idx)
        if gpu_mask is None or expert_id < 0 or expert_id >= self.num_experts:
            return False
        return bool(gpu_mask[expert_id])

    def record_activations(self, layer_idx: int, topk_ids: np.ndarray):
        """Record expert activations for hotness tracking."""
        if self.max_tier0_experts <= 0:
            return
        flat = self._filter_cpu_expert_ids(layer_idx, topk_ids.flatten())
        self.hotness.record(flat)

    def prefetch_layer(self, layer_idx: int, topk_ids: np.ndarray):
        """
        Issue madvise(MADV_WILLNEED) for a specific layer's upcoming experts.

        Skips experts already promoted to Tier 0 (they're in NUMA-local malloc,
        no page fault possible). This avoids the madvise syscall storm (PERF-3).
        """
        if layer_idx not in self.mmap_regions:
            return

        moe = self.moe_refs.get(layer_idx)
        unique_ids = np.unique(self._filter_cpu_expert_ids(layer_idx, topk_ids.flatten()))
        layer_regions = self.mmap_regions[layer_idx]

        for eid in unique_ids:
            if eid < 0:
                continue
            # Skip Tier 0 experts — already in NUMA-local malloc, no prefetch needed
            if moe is not None and moe.is_expert_promoted(int(eid)):
                continue
            for proj_name, regions in layer_regions.items():
                del proj_name
                for region in regions[eid]:
                    region.prefetch()

    def start_promotion_thread(self):
        """Start background thread that promotes hot experts to Tier 0."""
        if self._running or self.max_tier0_experts <= 0:
            return
        self._running = True
        self._promotion_thread = threading.Thread(target=self._promotion_loop, daemon=True)
        self._promotion_thread.start()

    def stop_promotion_thread(self):
        """Stop the background promotion thread."""
        self._running = False
        if self._promotion_thread is not None:
            self._promotion_thread.join(timeout=10)
            self._promotion_thread = None

    def _promotion_loop(self):
        """Background loop: periodically promote hot experts to Tier 0."""
        while self._running:
            time.sleep(self.promotion_interval_sec)
            try:
                self._maybe_promote()
                self.hotness.decay()
            except Exception as e:
                print(f"[TieredWeightProvider] promotion error: {e}")

    def _maybe_promote(self):
        """
        Promote/demote experts via C++ MOE objects.

        Calls moe.promote_expert(eid) which allocates NUMA-local buffers,
        copies weight data, and atomically swaps the live pointers.
        Calls moe.demote_expert(eid) to restore baseline pointers and free NUMA buffers.
        """
        if self.max_tier0_experts <= 0 or not self.moe_refs:
            return
        hot_ids = set(self.hotness.get_top_k(self.max_tier0_experts))

        for layer_idx, moe in list(self.moe_refs.items()):
            # Promote hot experts not yet in Tier 0
            for eid in hot_ids:
                if self._is_gpu_expert(layer_idx, eid):
                    continue
                if not moe.is_expert_promoted(eid):
                    moe.promote_expert(eid)

            # Demote cold experts back to baseline (mmap or legacy)
            for eid in range(self.num_experts):
                if self._is_gpu_expert(layer_idx, eid):
                    if moe.is_expert_promoted(eid):
                        moe.demote_expert(eid)
                    continue
                if eid not in hot_ids and moe.is_expert_promoted(eid):
                    moe.demote_expert(eid)
