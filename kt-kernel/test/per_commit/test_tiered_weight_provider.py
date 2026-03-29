"""Pure-Python tests for tiered MoE weight management helpers."""

import importlib.util
import os
import tempfile
from pathlib import Path

import numpy as np


MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "python", "utils", "weight_provider.py")
SPEC = importlib.util.spec_from_file_location("weight_provider", MODULE_PATH)
weight_provider = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(weight_provider)

TieredWeightProvider = weight_provider.TieredWeightProvider
backend_supports_tiered_strategy = weight_provider.backend_supports_tiered_strategy
resolve_backend_weight_strategy = weight_provider.resolve_backend_weight_strategy
resolve_weight_strategy = weight_provider.resolve_weight_strategy
compute_max_tier0_experts = weight_provider.compute_max_tier0_experts
resolve_auto_tier0_budget_bytes = weight_provider.resolve_auto_tier0_budget_bytes
get_cgroup_memory_limit_current_bytes = weight_provider.get_cgroup_memory_limit_current_bytes
get_available_ram_bytes = weight_provider.get_available_ram_bytes
get_total_ram_bytes = weight_provider.get_total_ram_bytes
constrain_tier0_memory_bytes = weight_provider.constrain_tier0_memory_bytes


class DummyMoe:
    """Minimal MOE stub for promotion/demotion tests."""

    def __init__(self):
        self.promoted = set()
        self.promote_calls = []
        self.demote_calls = []

    def promote_expert(self, expert_id: int):
        self.promoted.add(expert_id)
        self.promote_calls.append(expert_id)

    def demote_expert(self, expert_id: int):
        self.promoted.discard(expert_id)
        self.demote_calls.append(expert_id)

    def is_expert_promoted(self, expert_id: int) -> bool:
        return expert_id in self.promoted


class DummyRegion:
    """Simple mmap-region stub that records prefetch requests."""

    def __init__(self):
        self.prefetch_calls = 0

    def prefetch(self):
        self.prefetch_calls += 1


def test_resolve_weight_strategy_auto_switches_between_legacy_and_tiered():
    """Auto mode should choose a concrete strategy from the RAM estimate."""
    tiered, model_bytes, total_ram = resolve_weight_strategy(
        "auto",
        num_layers=60,
        num_experts=256,
        hidden_size=7168,
        intermediate_size=2048,
        total_ram_bytes=64 * 1024**3,
        available_ram_bytes=32 * 1024**3,
    )
    assert tiered == "tiered"
    assert model_bytes > 0
    assert total_ram == 64 * 1024**3

    legacy, _, _ = resolve_weight_strategy(
        "auto",
        num_layers=60,
        num_experts=256,
        hidden_size=7168,
        intermediate_size=2048,
        total_ram_bytes=512 * 1024**3,
        available_ram_bytes=512 * 1024**3,
    )
    assert legacy == "legacy"


def test_backend_strategy_support_is_backend_specific():
    """Only backends with mmap support should expose tiered semantics."""
    assert backend_supports_tiered_strategy("LLAMAFILE") is True
    assert backend_supports_tiered_strategy("AMXINT4") is True
    assert backend_supports_tiered_strategy("AMXINT8") is True
    assert backend_supports_tiered_strategy("MOE_INT4") is True
    assert backend_supports_tiered_strategy("MOE_INT8") is True
    assert backend_supports_tiered_strategy("BF16") is True
    assert backend_supports_tiered_strategy("RAWINT4") is False


def test_amx_backends_can_resolve_into_tiered_mode():
    """AMX mmap-capable backends should resolve auto into tiered mode under memory pressure."""
    resolved, model_bytes, total_ram = resolve_backend_weight_strategy(
        "AMXINT4",
        "auto",
        num_layers=60,
        num_experts=256,
        hidden_size=7168,
        intermediate_size=2048,
        total_ram_bytes=64 * 1024**3,
        available_ram_bytes=8 * 1024**3,
    )
    assert resolved == "tiered"
    assert model_bytes > 0
    assert total_ram == 64 * 1024**3

    moe_resolved, moe_model_bytes, _ = resolve_backend_weight_strategy(
        "MOE_INT8",
        "auto",
        num_layers=60,
        num_experts=256,
        hidden_size=7168,
        intermediate_size=2048,
        total_ram_bytes=64 * 1024**3,
        available_ram_bytes=8 * 1024**3,
    )
    assert moe_resolved == "tiered"
    assert moe_model_bytes > 0

    bf16_resolved, bf16_model_bytes, _ = resolve_backend_weight_strategy(
        "BF16",
        "auto",
        num_layers=60,
        num_experts=256,
        hidden_size=7168,
        intermediate_size=2048,
        total_ram_bytes=256 * 1024**3,
        available_ram_bytes=64 * 1024**3,
    )
    assert bf16_resolved == "tiered"
    assert bf16_model_bytes > moe_model_bytes


def test_non_mmap_backends_still_fall_back_to_legacy():
    """Backends without file-backed weight support must stay on resident loading."""
    forced, _, _ = resolve_backend_weight_strategy(
        "RAWINT4",
        "tiered",
        num_layers=60,
        num_experts=256,
        hidden_size=7168,
        intermediate_size=2048,
        total_ram_bytes=64 * 1024**3,
        available_ram_bytes=8 * 1024**3,
    )
    assert forced == "legacy"


def test_auto_tier0_budget_shrinks_as_memory_pressure_rises():
    """Auto Tier0 budget should reserve less NUMA hotset when the model pressure is high."""
    low_pressure = resolve_auto_tier0_budget_bytes(
        model_bytes=40 * 1024**3,
        total_ram_bytes=128 * 1024**3,
        available_ram_bytes=96 * 1024**3,
    )
    high_pressure = resolve_auto_tier0_budget_bytes(
        model_bytes=200 * 1024**3,
        total_ram_bytes=128 * 1024**3,
        available_ram_bytes=96 * 1024**3,
    )

    assert low_pressure > high_pressure > 0


def test_cgroup_v2_memory_limit_is_detected_from_process_scope():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        scoped = root / "system.slice" / "run-test.scope"
        scoped.mkdir(parents=True)
        (scoped / "memory.max").write_text(str(64 * 1024**3))
        (scoped / "memory.current").write_text(str(12 * 1024**3))

        limit, current = get_cgroup_memory_limit_current_bytes(
            proc_self_cgroup_text="0::/system.slice/run-test.scope\n",
            mountinfo_text="29 23 0:28 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n",
            mount_root_override=root,
        )

    assert limit == 64 * 1024**3
    assert current == 12 * 1024**3


def test_ram_queries_prefer_cgroup_over_host_memory():
    original = weight_provider.get_cgroup_memory_limit_current_bytes
    try:
        weight_provider.get_cgroup_memory_limit_current_bytes = lambda **_: (
            64 * 1024**3,
            12 * 1024**3,
        )
        assert get_total_ram_bytes() == 64 * 1024**3
        assert get_available_ram_bytes() == 52 * 1024**3
    finally:
        weight_provider.get_cgroup_memory_limit_current_bytes = original


def test_explicit_tier0_budget_is_clamped_to_effective_scope():
    constrained = constrain_tier0_memory_bytes(
        60 * 1024**3,
        available_ram_bytes=32 * 1024**3,
    )
    assert constrained == 28 * 1024**3


def test_provider_skips_gpu_experts_for_prefetch_and_promotion():
    """GPU-routed experts must not consume Tier0 promotion or mmap prefetch budget."""
    provider = TieredWeightProvider(num_experts=4, num_layers=2, max_tier0_experts=2)
    provider.start_promotion_thread = lambda: None

    moe0 = DummyMoe()
    moe1 = DummyMoe()
    provider.register_moe(0, moe0, gpu_experts_mask=np.array([False, True, False, False], dtype=np.bool_))
    provider.register_moe(1, moe1, gpu_experts_mask=np.array([True, False, False, False], dtype=np.bool_))

    regions = {}
    for expert_id in range(4):
        region = DummyRegion()
        regions[expert_id] = region
        provider.register_mmap_region(0, "gate", expert_id, region)

    provider.prefetch_layer(0, np.array([[0, 1, 2, -1]], dtype=np.int64))
    assert regions[0].prefetch_calls == 1
    assert regions[1].prefetch_calls == 0
    assert regions[2].prefetch_calls == 1
    assert regions[3].prefetch_calls == 0

    provider.record_activations(0, np.array([[1, 2, 2, 3]], dtype=np.int64))
    assert provider.hotness.counts[1] == 0.0
    assert provider.hotness.counts[2] > 0.0
    assert provider.hotness.counts[3] > 0.0

    provider.hotness.counts[:] = np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float64)
    provider._maybe_promote()

    assert moe0.promote_calls == [0]
    assert moe1.promote_calls == [1]


def test_provider_prefetches_all_registered_regions_for_an_expert():
    """TP-aware registration may attach multiple mmap slices to the same expert."""
    provider = TieredWeightProvider(num_experts=2, num_layers=1, max_tier0_experts=1)

    region_a = DummyRegion()
    region_b = DummyRegion()
    provider.register_mmap_region(0, "gate", 0, region_a)
    provider.register_mmap_region(0, "gate", 0, region_b)

    provider.prefetch_layer(0, np.array([[0]], dtype=np.int64))

    assert region_a.prefetch_calls == 1
    assert region_b.prefetch_calls == 1


def test_zero_tier0_budget_disables_background_promotion():
    """A zero Tier0 budget must disable provider-managed promotion entirely."""
    assert (
        compute_max_tier0_experts(
            tier0_memory_bytes=0,
            num_layers=60,
            num_experts=256,
            hidden_size=7168,
            intermediate_size=2048,
        )
        == 0
    )

    provider = TieredWeightProvider(num_experts=4, num_layers=1, max_tier0_experts=0)
    start_calls = []
    provider.start_promotion_thread = lambda: start_calls.append(True)

    moe = DummyMoe()
    provider.register_moe(0, moe, gpu_experts_mask=np.zeros(4, dtype=np.bool_))
    provider.hotness.counts[:] = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float64)
    provider._maybe_promote()

    assert start_calls == []
    assert moe.promote_calls == []
    assert moe.demote_calls == []
