# SPDX-License-Identifier: Apache-2.0
"""
KT Expert Parallelism Wrapper for MoE layers.

This module provides a generic wrapper that enables CPU-GPU expert parallelism
for any MoE quantization method. It coordinates parallel execution of GPU experts
(using any quantization method) and CPU experts (using AMX/AVX instructions).
"""

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.utils import get_compiler_backend

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


def _select_gpu_expert_ids(
    layer_idx: int, num_experts: int, num_gpu_experts: int
) -> Optional[List[int]]:
    """Pick which experts live on GPU for this layer.

    The selection logic lives in kt-kernel
    (kt_kernel/python/utils/mesh/gpu_expert_placement.py); this adapter only
    reads the env configuration and falls back to legacy prefix placement
    (experts 0..num_gpu_experts-1) when placement is not configured or the
    source cannot answer for this layer.

    KT_GPU_EXPERT_PLACEMENT_PATH accepts either the raw frequency dump
    (JSONL produced via KT_MESH_PREFILL_EXPERT_FREQ_PATH during a profiling
    run) or the compiled placement file built by
    kt-kernel/scripts/mesh/build_gpu_expert_placement.py.
    KT_GPU_EXPERT_PLACEMENT_METRIC selects the ranking histogram:
    "router" (all routed tokens, default) or "cpu" (eviction-score
    increments only).
    """
    path = os.environ.get("KT_GPU_EXPERT_PLACEMENT_PATH", "")
    if not path or num_gpu_experts <= 0:
        return None
    try:
        from kt_kernel.utils.mesh.gpu_expert_placement import (
            select_gpu_experts_for_layer,
        )
    except ImportError:
        logger.warning(
            "KT_GPU_EXPERT_PLACEMENT_PATH is set but the installed kt_kernel "
            "does not provide gpu_expert_placement (too old); using prefix placement"
        )
        return None
    metric = os.environ.get("KT_GPU_EXPERT_PLACEMENT_METRIC", "router")
    return select_gpu_experts_for_layer(
        path, layer_idx, num_experts, num_gpu_experts, metric=metric
    )


def _partition_cpu_experts(
    layer_idx: int,
    num_experts: int,
    cpu_expert_ids: List[int],
    moe_tp_rank: int,
    moe_tp_size: int,
) -> List[int]:
    """Decide which CPU-managed experts this moe-TP rank owns.

    Default (KT_CPU_EXPERT_PARALLEL=1): experts are partitioned disjointly
    across all moe-TP ranks so every rank runs its own KT/MESH backend on its
    own expert subset. KT_CPU_EXPERT_PARALLEL=0 restores the legacy layout
    (rank 0 owns every CPU expert, other ranks contribute GPU shards only).

    The partitioning math lives in kt-kernel
    (gpu_expert_placement.partition_cpu_experts_for_rank); when the GPU
    placement source is a raw frequency dump, per-expert counts are used to
    snake-balance hot experts across ranks.
    """
    if moe_tp_size <= 1:
        return list(cpu_expert_ids)
    if os.environ.get("KT_CPU_EXPERT_PARALLEL", "1") in ("0", "false", "False"):
        return list(cpu_expert_ids) if moe_tp_rank == 0 else []
    try:
        from kt_kernel.utils.mesh.gpu_expert_placement import (
            layer_counts_from_source,
            partition_cpu_experts_for_rank,
        )
    except ImportError:
        logger.warning(
            "KT CPU expert parallel requested but the installed kt_kernel does "
            "not provide gpu_expert_placement (too old); falling back to the "
            "legacy rank-0-only CPU backend"
        )
        return list(cpu_expert_ids) if moe_tp_rank == 0 else []

    path = os.environ.get("KT_GPU_EXPERT_PLACEMENT_PATH", "")
    metric = os.environ.get("KT_GPU_EXPERT_PLACEMENT_METRIC", "router")
    counts = (
        layer_counts_from_source(path, layer_idx, num_experts, metric)
        if path
        else None
    )
    return partition_cpu_experts_for_rank(
        cpu_expert_ids, moe_tp_rank, moe_tp_size, counts
    )


_worker_core_offset_adjusted = False


def _maybe_shift_worker_cores(moe_tp_rank: int, moe_tp_size: int, kt_config) -> None:
    """Spread each rank's CPUInfer workers onto disjoint core ranges.

    kt-kernel's worker_pool.cpp binds subpool thread i to core
    (KT_WORKER_CPU_CORE_OFFSET + i) WITHIN its NUMA node and reads the env
    once per process. With CPU expert parallelism every moe-TP rank process
    creates its own pool; without a per-rank shift all ranks would pin to the
    same cores and oversubscribe them moe_tp_size-fold.

    Shift = moe_tp_rank * stride. Stride defaults to the per-NUMA thread
    count (cpuinfer_threads // threadpool_count); override with
    KT_CPU_EXPERT_PARALLEL_CORE_STRIDE, or set it to 0 to disable shifting
    entirely (e.g. when ranks are pinned externally). Must run before this
    process constructs its first KTMoEWrapper, i.e. before the worker pool
    exists.
    """
    global _worker_core_offset_adjusted
    if _worker_core_offset_adjusted:
        return
    _worker_core_offset_adjusted = True
    if moe_tp_size <= 1 or moe_tp_rank == 0:
        return

    stride_env = os.environ.get("KT_CPU_EXPERT_PARALLEL_CORE_STRIDE", "")
    if stride_env:
        try:
            stride = int(stride_env)
        except ValueError:
            logger.warning(
                "Invalid KT_CPU_EXPERT_PARALLEL_CORE_STRIDE=%r; "
                "worker core offsets left unshifted",
                stride_env,
            )
            return
    else:
        pools = max(1, int(kt_config.threadpool_count or 1))
        stride = max(1, int(kt_config.cpuinfer_threads) // pools)
    if stride <= 0:
        return

    base_env = os.environ.get("KT_WORKER_CPU_CORE_OFFSET", "0")
    try:
        base = int(base_env) if base_env else 0
    except ValueError:
        base = 0
    shifted = base + moe_tp_rank * stride
    os.environ["KT_WORKER_CPU_CORE_OFFSET"] = str(shifted)
    logger.info(
        "KT CPU expert parallel: moe-TP rank %d shifted worker core offset "
        "%d -> %d (stride %d)",
        moe_tp_rank,
        base,
        shifted,
        stride,
    )


_resident_caps_adjusted = False


def _maybe_split_resident_caps(moe_tp_rank: int, moe_tp_size: int) -> None:
    """Divide process-wide resident-expert caps across moe-TP ranks.

    KT_MESH_GLOBAL_POOL_CAPACITY, KT_MAX_RESIDENT_EXPERTS and
    KT_MAX_TIER0_EXPERTS are read once per process and were sized assuming a
    single CPU backend. With CPU expert parallelism every rank runs its own
    backend; left untouched, each rank would claim the full budget, so the
    machine-wide footprint (and any cap-sweep semantics) would multiply by
    moe_tp_size. Splitting keeps "cap N" meaning N resident slots in total.

    Only vars set to a positive integer are rewritten; unset vars keep their
    backend defaults. Opt out with KT_CPU_EXPERT_PARALLEL_SPLIT_CAPS=0. Must
    run before this process constructs its first KTMoEWrapper.
    """
    global _resident_caps_adjusted
    if _resident_caps_adjusted:
        return
    _resident_caps_adjusted = True
    if moe_tp_size <= 1:
        return
    if os.environ.get("KT_CPU_EXPERT_PARALLEL", "1") in ("0", "false", "False"):
        return
    if os.environ.get("KT_CPU_EXPERT_PARALLEL_SPLIT_CAPS", "1") in ("0", "false", "False"):
        return

    try:
        from kt_kernel.utils.mesh.gpu_expert_placement import split_capacity_for_rank
    except ImportError:

        def split_capacity_for_rank(total: int, rank: int, world_size: int) -> int:
            base, rem = divmod(total, world_size)
            return max(1, base + (1 if rank < rem else 0))

    for name in (
        "KT_MESH_GLOBAL_POOL_CAPACITY",
        "KT_MAX_RESIDENT_EXPERTS",
        "KT_MAX_TIER0_EXPERTS",
    ):
        raw = os.environ.get(name, "")
        try:
            total = int(raw)
        except ValueError:
            continue
        if total <= 0:
            continue
        share = split_capacity_for_rank(total, moe_tp_rank, moe_tp_size)
        os.environ[name] = str(share)
        logger.info(
            "KT CPU expert parallel: moe-TP rank %d/%d takes %d of %d for %s",
            moe_tp_rank,
            moe_tp_size,
            share,
            total,
            name,
        )


@dataclass
class KTConfig:
    """Configuration for KTransformers heterogeneous computing CPU part.

    Args:
        layer_idx: Layer index in the model
        num_gpu_experts: Number of experts to run on GPU
        cpuinfer_threads: Number of CPU inference threads
        threadpool_count: Number of thread pools for CPU computation
        weight_path: Path to CPU quantized weights
        chunked_prefill_size: Chunk size for prefill computation
        method: CPU computation method (e.g., "int4")
        num_layers: Total number of layers in the model (optional)
    """

    layer_idx: int
    num_gpu_experts: int
    cpuinfer_threads: int
    threadpool_count: int
    weight_path: str
    chunked_prefill_size: int
    max_deferred_experts_per_token: int
    method: str
    num_layers: Optional[int] = None


def create_kt_config_from_server_args(
    server_args: "ServerArgs", layer_idx: int
) -> Optional[KTConfig]:
    """Create KTConfig from ServerArgs if KT is configured.

    Args:
        server_args: Global server arguments
        layer_idx: Layer index in the model

    Returns:
        KTConfig if KT is configured, None otherwise
    """
    if server_args.kt_weight_path is None:
        return None

    # Try to get num_layers from model config
    num_layers = None
    try:
        hf_config = server_args.get_hf_config()
        num_layers = getattr(hf_config, "num_hidden_layers", None)
    except Exception:
        # If we can't get the config, num_layers will be None
        pass

    return KTConfig(
        layer_idx=layer_idx,
        num_gpu_experts=server_args.kt_num_gpu_experts,
        cpuinfer_threads=server_args.kt_cpuinfer,
        threadpool_count=server_args.kt_threadpool_count,
        weight_path=server_args.kt_weight_path,
        chunked_prefill_size=server_args.chunked_prefill_size,
        method=server_args.kt_method,
        max_deferred_experts_per_token=server_args.kt_max_deferred_experts_per_token,
        num_layers=num_layers,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_cpu_expert_ids(topk_ids: torch.Tensor, num_gpu_experts: int) -> torch.Tensor:
    """Mask CPU expert IDs by setting them to -1.

    This function masks expert IDs that should be computed on CPU (IDs >= num_gpu_experts)
    so they won't be computed on GPU. The masked IDs are set to -1, which causes the
    GPU MoE kernel to skip those experts.

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing expert IDs
        num_gpu_experts: Number of experts that should run on GPU (experts 0 to num_gpu_experts-1)

    Returns:
        Modified topk_ids tensor with CPU expert IDs masked as -1
    """
    masked_topk_ids = topk_ids.clone()
    masked_topk_ids[masked_topk_ids >= num_gpu_experts] = -1
    return masked_topk_ids


@torch.compile(dynamic=True, backend=get_compiler_backend())
def remap_gpu_expert_ids(
    topk_ids: torch.Tensor, gpu_local_id_table: torch.Tensor
) -> torch.Tensor:
    """Translate global expert IDs into GPU-local weight slots via a lookup table.

    Used instead of mask_cpu_expert_ids when frequency-based placement is
    active: GPU-resident experts are no longer the contiguous prefix, so a
    plain comparison cannot decide GPU membership. The table maps each global
    expert ID to its slot in the GPU weight buffers, and CPU-managed experts
    map to -1 so the GPU MoE kernel skips them. Out-of-range IDs (including a
    pre-existing -1 sentinel) become -1, matching the old comparison-based
    masking which also left them skipped. The table is read at every call
    (pure gather), so its contents may be swapped between forwards without
    recompiling or recapturing CUDA graphs.
    """
    num_experts = gpu_local_id_table.shape[0]
    valid = (topk_ids >= 0) & (topk_ids < num_experts)
    safe_ids = topk_ids.clamp(0, num_experts - 1)
    return torch.where(valid, gpu_local_id_table[safe_ids], -1)


class KTEPWrapperMethod(FusedMoEMethodBase):
    """Wrapper for any MoE quantization method to enable CPU-GPU expert parallelism.

    This wrapper coordinates parallel execution of:
    - GPU experts: the placement-selected subset (prefix by default), run as
      intermediate-dim TP shards on every rank using any quantization method
    - CPU experts: the remaining experts, partitioned disjointly across
      moe-TP ranks; each rank runs its own KT/MESH backend (AMX/AVX) over its
      subset at full intermediate size, and the post-MoE all-reduce sums each
      expert's output exactly once (KT_CPU_EXPERT_PARALLEL=0 restores the
      legacy layout where rank 0 owns all CPU experts)

    The wrapper implements the submit-compute-sync pattern:
    1. Submit CPU expert computation (non-blocking)
    2. Execute GPU expert computation in parallel
    3. Synchronize and merge CPU+GPU results

    Example:
        # Wrap any GPU method with AMX/AVX CPU expert support
        gpu_method = CompressedTensorsWNA16MoE(quant_config, prefix)
        kt_config = KTConfig(layer_idx=0, num_gpu_experts=4, ...)
        method = KTEPWrapperMethod(gpu_method, kt_config)
    """

    def __init__(
        self,
        gpu_method: FusedMoEMethodBase,
        kt_config: KTConfig,
    ):
        """Initialize the KT EP wrapper.

        Args:
            gpu_method: The quantization method to use for GPU experts
            kt_config: Configuration for KT CPU expert computation
        """
        if not KTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "kt_kernel is not installed. To use KTransformers EP wrapper, please install kt_kernel."
            )

        self.gpu_method = gpu_method
        self.kt_config = kt_config
        self.num_gpu_experts = kt_config.num_gpu_experts
        self.override_num_local_experts = True
        self.gpu_method.num_gpu_experts = self.num_gpu_experts
        self.tp_rank = get_tensor_model_parallel_rank()

        # KT wrapper will be initialized in create_weights
        self.wrapper: Optional[KTMoEWrapper] = None

        # Frequency-based GPU expert placement (None = legacy prefix placement).
        # gpu_expert_global_ids: sorted global IDs of the experts resident on
        # GPU for this layer; local weight slot = rank within this list.
        self.gpu_expert_global_ids: Optional[List[int]] = None
        self._gpu_local_slot_by_global: Optional[Dict[int, int]] = None
        # Lazy per-(device, dtype) lookup tables for remap_gpu_expert_ids.
        self._gpu_local_id_tables: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}

        # Store parameters needed for KT initialization
        self._layer_params = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for both GPU and CPU experts.

        Args:
            layer: The MoE layer module
            num_experts: Total number of experts (GPU + CPU)
            hidden_size: Hidden dimension size
            intermediate_size_per_partition: Intermediate size per TP partition
            params_dtype: Data type for parameters
            **extra_weight_attrs: Additional weight attributes
        """
        self.global_num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size_per_partition

        # Decide GPU expert placement on every rank (the weight loader and the
        # topk-id masking in apply() run on all TP ranks and must agree); the
        # selection is a deterministic function of the placement file.
        selected = _select_gpu_expert_ids(
            self.kt_config.layer_idx, num_experts, self.num_gpu_experts
        )
        if selected is not None:
            self.gpu_expert_global_ids = selected
            self._gpu_local_slot_by_global = {
                global_id: local_slot for local_slot, global_id in enumerate(selected)
            }

        # Get required parameters from layer object
        # top_k: number of experts selected per token
        num_experts_per_tok = layer.top_k

        # intermediate_size_full: full intermediate size before TP partitioning
        intermediate_size_full = (
            layer.intermediate_size_per_partition * layer.moe_tp_size
        )

        layer_max_deferred = self.kt_config.max_deferred_experts_per_token or 0
        if (
            self.kt_config.max_deferred_experts_per_token is not None
            and self.kt_config.num_layers is not None
            and self.kt_config.layer_idx == self.kt_config.num_layers - 1
        ):
            layer_max_deferred = 0

        # 1. Create weights for GPU experts using the wrapped method
        # GPU experts: 0 to num_gpu_experts-1
        self.gpu_method.create_weights(
            layer=layer,
            num_experts=self.num_gpu_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

        # 2. Initialize the KT/MESH CPU expert backend.
        # GPU experts run as intermediate-dim TP shards on every rank; CPU
        # experts are instead partitioned ACROSS moe-TP ranks: each rank owns
        # a disjoint expert subset and computes those experts' FULL FFN output
        # (full intermediate size), so the existing post-MoE all-reduce sums
        # every CPU expert exactly once. This reuses the AMX weight blobs
        # unchanged — they are packed for the full intermediate size, so a
        # true intermediate-dim CPU shard would require repacking, while an
        # expert partition only changes which blobs each rank touches.
        moe_tp_rank = int(getattr(layer, "moe_tp_rank", 0) or 0)
        moe_tp_size = int(getattr(layer, "moe_tp_size", 1) or 1)
        self.moe_tp_rank = moe_tp_rank

        gpu_experts_mask = torch.zeros(num_experts, dtype=torch.bool, device="cpu")
        if self.gpu_expert_global_ids is not None:
            gpu_experts_mask[
                torch.tensor(self.gpu_expert_global_ids, dtype=torch.long)
            ] = True
        elif self.num_gpu_experts > 0:
            gpu_experts_mask[: self.num_gpu_experts] = True

        cpu_expert_ids = [
            expert_id
            for expert_id in range(num_experts)
            if not bool(gpu_experts_mask[expert_id])
        ]
        my_cpu_experts = _partition_cpu_experts(
            self.kt_config.layer_idx,
            num_experts,
            cpu_expert_ids,
            moe_tp_rank,
            moe_tp_size,
        )

        if my_cpu_experts:
            # The mask handed to kt-kernel means "skip this expert": GPU
            # experts plus every CPU expert owned by another rank.
            skip_mask = gpu_experts_mask.clone()
            not_mine = sorted(set(cpu_expert_ids) - set(my_cpu_experts))
            if not_mine:
                skip_mask[torch.tensor(not_mine, dtype=torch.long)] = True
            _maybe_shift_worker_cores(moe_tp_rank, moe_tp_size, self.kt_config)
            _maybe_split_resident_caps(moe_tp_rank, moe_tp_size)
            if self.kt_config.layer_idx == 0:
                logger.info(
                    "KT CPU expert parallel: moe-TP rank %d/%d owns %d of %d "
                    "CPU experts per layer",
                    moe_tp_rank,
                    moe_tp_size,
                    len(my_cpu_experts),
                    len(cpu_expert_ids),
                )
            self.wrapper = KTMoEWrapper(
                layer_idx=self.kt_config.layer_idx,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                moe_intermediate_size=intermediate_size_full,
                gpu_experts_mask=skip_mask,
                cpuinfer_threads=self.kt_config.cpuinfer_threads,
                threadpool_count=self.kt_config.threadpool_count,
                weight_path=self.kt_config.weight_path,
                chunked_prefill_size=self.kt_config.chunked_prefill_size,
                method=self.kt_config.method,
                max_deferred_experts_per_token=layer_max_deferred,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after loading from checkpoint.

        Args:
            layer: The MoE layer module
        """
        # 1. Process GPU weights
        if hasattr(self.gpu_method, "process_weights_after_loading"):
            self.gpu_method.process_weights_after_loading(layer)

        # 2. Load CPU weights using KT wrapper (every rank that owns CPU
        # experts loads its own subset; kt-kernel skips masked experts)
        if self.wrapper is not None:
            torch.cuda.synchronize()

            # Get expert location metadata for CPU expert mapping
            from sglang.srt.eplb.expert_location_dispatch import (
                get_global_expert_location_metadata,
            )

            expert_location_metadata = get_global_expert_location_metadata()
            if expert_location_metadata is None:
                physical_to_logical_map_cpu = torch.arange(
                    self.global_num_experts, dtype=torch.int64, device="cpu"
                )
            else:
                physical_to_logical_map_cpu = (
                    expert_location_metadata.physical_to_logical_map_cpu[
                        self.kt_config.layer_idx
                    ]
                    .to(dtype=torch.int64, device="cpu")
                    .contiguous()
                )
            self.wrapper.load_weights(physical_to_logical_map_cpu)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        """Create MoE runner for computation.

        Args:
            layer: The MoE layer module
            moe_runner_config: Configuration for MoE runner
        """
        self.moe_runner_config = moe_runner_config
        if self.override_num_local_experts:
            moe_runner_config.num_local_experts = self.num_gpu_experts
        # Delegate to GPU method to create its runner
        self.gpu_method.create_moe_runner(layer, moe_runner_config)

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Submit CPU expert computation asynchronously (non-blocking).

        This method submits the CPU expert computation to AMX/AVX without waiting
        for completion, allowing GPU computation to proceed in parallel.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
        """
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        if self.wrapper is None:
            return

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Submit forward task to CPU (non-blocking)
        self.wrapper.submit_forward(
            x, topk_ids, topk_weights, torch.cuda.current_stream(x.device).cuda_stream
        )

    def sync(self, x: torch.Tensor) -> torch.Tensor:
        """Synchronize and retrieve CPU expert computation results.

        This method waits for the CPU computation to complete and returns the results.

        Args:
            x: Reference tensor for shape and device information

        Returns:
            CPU expert computation results
        """
        if self.wrapper is None:
            return torch.zeros_like(x)

        # Wait for CPU computation and retrieve results
        return self.wrapper.sync_forward(
            x, torch.cuda.current_stream(x.device).cuda_stream
        )

    def gpu_expert_local_slot(self, expert_id: int) -> int:
        """Map a global expert ID to its GPU-local weight slot, or -1 if CPU-managed.

        With frequency-based placement the slot is the expert's rank within
        the sorted GPU expert list; with legacy prefix placement the mapping
        is identity for IDs below num_gpu_experts. The weight loader uses this
        to route checkpoint experts into the GE-sized GPU buffers.
        """
        if self._gpu_local_slot_by_global is not None:
            return self._gpu_local_slot_by_global.get(expert_id, -1)
        if 0 <= expert_id < self.num_gpu_experts:
            return expert_id
        return -1

    def _gpu_local_id_table(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Lookup table for remap_gpu_expert_ids, built lazily per (device, dtype).

        First use happens during warmup forwards, i.e. before any CUDA graph
        capture, so the table address is stable across captured replays.
        """
        key = (topk_ids.device, topk_ids.dtype)
        table = self._gpu_local_id_tables.get(key)
        if table is None:
            table = torch.full(
                (self.global_num_experts,), -1, dtype=topk_ids.dtype, device="cpu"
            )
            for global_id, local_slot in self._gpu_local_slot_by_global.items():
                table[global_id] = local_slot
            table = table.to(topk_ids.device)
            self._gpu_local_id_tables[key] = table
        return table

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Execute hybrid CPU+GPU MoE forward pass with parallelism.

        This is the main computation method that coordinates:
        1. Submit CPU expert computation (non-blocking)
        2. Execute GPU expert computation in parallel
        3. Synchronize CPU results and merge with GPU results

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information

        Returns:
            Combined computation results from CPU and GPU experts
        """
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        # Step 1: Submit this rank's CPU expert computation (non-blocking).
        # Each moe-TP rank owns a disjoint CPU expert subset; ranks without a
        # wrapper (no CPU experts assigned) contribute GPU shards only.
        if self.wrapper is not None:
            self.submit(layer, dispatch_output)

        # Step 2: Prepare GPU computation by translating topk ids for the GPU kernel
        # CPU-managed expert IDs are set to -1 so the GPU kernel skips them; with
        # frequency-based placement, GPU-resident IDs are additionally remapped
        # from global expert IDs to their local weight slots.
        topk_ids = topk_output.topk_ids
        if self._gpu_local_slot_by_global is not None:
            masked_topk_ids = remap_gpu_expert_ids(
                topk_ids, self._gpu_local_id_table(topk_ids)
            )
        else:
            masked_topk_ids = mask_cpu_expert_ids(topk_ids, self.num_gpu_experts)

        # Create modified dispatch output for GPU computation
        masked_topk_output = topk_output._replace(topk_ids=masked_topk_ids)
        masked_dispatch_output = dispatch_output._replace(
            topk_output=masked_topk_output
        )

        # Step 3: Execute GPU expert computation (any quantization method)
        # This runs in parallel with CPU computation
        gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)

        # Step 4: Synchronize this rank's CPU experts and merge with the GPU
        # partial output. Expert subsets are disjoint across ranks, so the
        # post-MoE all-reduce sums each CPU expert's output exactly once.
        output = gpu_combine_input.hidden_states
        if self.wrapper is not None:
            cpu_output = self.sync(x)
            output = output + cpu_output

        return StandardCombineInput(hidden_states=output)

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped GPU method.

        This allows the wrapper to transparently expose attributes and methods
        from the wrapped GPU quantization method.

        Args:
            name: Attribute name

        Returns:
            Attribute value from gpu_method
        """
        # Avoid infinite recursion for internal attributes
        if name in ("gpu_method", "wrapper", "kt_config"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        return getattr(self.gpu_method, name)
