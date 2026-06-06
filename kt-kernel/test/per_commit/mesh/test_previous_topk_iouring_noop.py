"""Regression guard for disabled IOURING previous-topk prefetch."""

from __future__ import annotations

import importlib.util
from pathlib import Path


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


runtime_helpers.install_base_moe_helpers(_FakeBaseMoEWrapper, _FakeCPUBuffer, False)


class _FakeMoe:
    def __init__(self) -> None:
        self.prefetch_calls = 0

    def prefetch_experts_task(self, *args, **kwargs):
        self.prefetch_calls += 1
        return ("prefetch", args, kwargs)


class _FakeWrapper(_FakeBaseMoEWrapper):
    def __init__(self) -> None:
        self.layer_idx = 0
        self.io_backend = "IOURING"
        self.moe = _FakeMoe()
        self.tasks = []

    def _submit_cpuinfer_task(self, task, cuda_stream=None):
        self.tasks.append(task)

    def close(self):
        pass

    def __del__(self):
        pass


def test_previous_topk_does_not_submit_iouring_prefetch_task():
    wrapper = _FakeWrapper()

    wrapper._mesh_prefetch_previous_topk()

    assert wrapper.moe.prefetch_calls == 0
    assert wrapper.tasks == []
