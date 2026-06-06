"""Pure-Python tests for decode previous-topk MESH prefetch scheduling."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[3] / "python" / "utils" / "mesh" / "runtime_helpers.py"
SPEC = importlib.util.spec_from_file_location("runtime_helpers", MODULE_PATH)
runtime_helpers = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(runtime_helpers)


class _FakeBaseMoEWrapper:
    _layer_has_pending_deferred = {}
    _mesh_decode_transition_done = True
    _prev_topk_ids_by_layer = {}
    _wrappers_by_layer = {}


class _FakeCPUBuffer:
    buffer_depth = 2

    @classmethod
    def get_buffer(cls, hidden_states: torch.Tensor, num_experts_per_tok: int):
        batch_size, hidden_size = hidden_states.shape
        input_tensor_cpu = [torch.zeros((batch_size, hidden_size), dtype=torch.bfloat16) for _ in range(2)]
        immediate_experts_ids_cpu = [
            torch.zeros((batch_size, num_experts_per_tok), dtype=torch.long) for _ in range(2)
        ]
        deferred_experts_ids_cpu = [
            torch.full((batch_size, num_experts_per_tok), -1, dtype=torch.long) for _ in range(2)
        ]
        weights_cpu = [torch.zeros((batch_size, num_experts_per_tok), dtype=torch.float32) for _ in range(2)]
        output_cpu = [torch.zeros((batch_size, hidden_size), dtype=torch.bfloat16) for _ in range(2)]
        bsz_tensor_cpu = [torch.zeros((1,), dtype=torch.int32) for _ in range(2)]
        output_gpu = [torch.zeros((batch_size, hidden_size), dtype=hidden_states.dtype) for _ in range(2)]
        return (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            output_gpu,
        )


runtime_helpers.install_base_moe_helpers(_FakeBaseMoEWrapper, _FakeCPUBuffer, False)


class _FakeMoe:
    def __init__(self) -> None:
        self.prefetch_calls = []
        self.forward_calls = 0

    def prefetch_experts_task(
        self,
        ids_ptr: int,
        count: int,
        protect_ptr: int,
        protect_count: int,
        max_to_submit: int,
        prefetch_kind: int = 0,
    ):
        self.prefetch_calls.append((count, protect_count, max_to_submit, prefetch_kind))
        return ("prefetch", count)

    def forward_task(self, *args):
        self.forward_calls += 1
        return ("forward", self.forward_calls)


class _FakeWrapper(_FakeBaseMoEWrapper):
    def __init__(self, layer_idx: int = 3) -> None:
        self.layer_idx = layer_idx
        self.io_backend = "IOURING"
        self.moe = _FakeMoe()
        self.num_experts_per_tok = 4
        self.max_deferred_experts_per_token = 0
        self._provider = None
        self.tasks = []

    def _submit_cpuinfer_task(self, task, cuda_stream=None):
        self.tasks.append(task)

    def _debug_log_moe_once(self, *args, **kwargs):
        pass

    def _mesh_full_gate_batched_enabled(self):
        return False

    def _maybe_mesh_prepare_prefill_layer_window(self, qlen: int):
        pass

    def _maybe_mesh_transition_to_decode_cache(self, qlen: int):
        pass

    def _prepare_router_scores_for_forward(self, *args, **kwargs):
        return (0, 0, 0, 0)

    def _cuda_graph_capture_active(self):
        return False

    def close(self):
        pass

    def __del__(self):
        pass


def test_previous_topk_prefetch_submits_after_sync_not_before_submit(monkeypatch):
    monkeypatch.setenv("KT_MESH_PREV_TOPK_PREFETCH", "1")
    wrapper = _FakeWrapper()
    _FakeBaseMoEWrapper._wrappers_by_layer = {wrapper.layer_idx: wrapper}
    hidden_states = torch.zeros((1, 8), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    topk_weights = torch.ones((1, 4), dtype=torch.float32)

    wrapper._mesh_submit_forward_impl(hidden_states, topk_ids, topk_weights, None)
    assert wrapper.moe.prefetch_calls == []

    wrapper._mesh_after_sync_forward(
        hidden_states,
        torch.zeros((1, 4), dtype=torch.long),
        topk_weights,
        torch.zeros((1, 8), dtype=torch.bfloat16),
        topk_ids,
        None,
    )
    assert wrapper.moe.prefetch_calls == [(4, 0, 4, 0)]


def test_previous_topk_prefetch_skips_prefill_batches(monkeypatch):
    monkeypatch.setenv("KT_MESH_PREV_TOPK_PREFETCH", "1")
    wrapper = _FakeWrapper()
    _FakeBaseMoEWrapper._wrappers_by_layer = {wrapper.layer_idx: wrapper}
    hidden_states = torch.zeros((2, 8), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    topk_weights = torch.ones((2, 4), dtype=torch.float32)

    wrapper._mesh_after_sync_forward(
        hidden_states,
        torch.zeros((2, 4), dtype=torch.long),
        topk_weights,
        torch.zeros((2, 8), dtype=torch.bfloat16),
        topk_ids,
        None,
    )
    assert wrapper.moe.prefetch_calls == []


def test_previous_topk_prefetch_targets_next_layer(monkeypatch):
    monkeypatch.setenv("KT_MESH_PREV_TOPK_PREFETCH", "1")
    current_wrapper = _FakeWrapper(layer_idx=3)
    next_wrapper = _FakeWrapper(layer_idx=4)
    _FakeBaseMoEWrapper._wrappers_by_layer = {
        current_wrapper.layer_idx: current_wrapper,
        next_wrapper.layer_idx: next_wrapper,
    }
    _FakeBaseMoEWrapper._prev_topk_ids_by_layer = {
        next_wrapper.layer_idx: torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    }
    hidden_states = torch.zeros((1, 8), dtype=torch.bfloat16)
    current_topk_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    topk_weights = torch.ones((1, 4), dtype=torch.float32)

    current_wrapper._mesh_after_sync_forward(
        hidden_states,
        torch.zeros((1, 4), dtype=torch.long),
        topk_weights,
        torch.zeros((1, 8), dtype=torch.bfloat16),
        current_topk_ids,
        None,
    )

    assert current_wrapper.moe.prefetch_calls == []
    assert next_wrapper.moe.prefetch_calls == [(4, 0, 4, 0)]
