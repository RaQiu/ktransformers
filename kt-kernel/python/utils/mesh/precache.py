from __future__ import annotations

import os
from pathlib import Path


def precache_mesh_experts(weight_path: Path | str, kt_method: str, tensor_parallel_size: int, env: dict[str, str]):
    if kt_method.upper() != "BF16":
        return {
            "skipped": True,
            "reason": f"--precache currently applies to BF16 MESH expert cache only; method={kt_method}",
        }

    false_values = ("0", "false", "False", "FALSE", "no", "No", "NO")
    use_direct_io = env.get("KT_IOURING_DIRECT", "1") not in false_values

    from kt_kernel.utils.loader import BF16SafeTensorLoader
    from kt_kernel.utils.mesh import loader as mesh_loader

    old_env = os.environ.copy()
    os.environ.update(env)
    loader = None
    try:
        loader = BF16SafeTensorLoader(str(weight_path))
        loader._ensure_mesh_index()
        mesh_loader.bf16_detect_format(loader)
        result = mesh_loader.precache_bf16_experts_iouring(
            loader,
            tp_count=tensor_parallel_size,
            use_direct_io=use_direct_io,
        )
        result["skipped"] = False
        return result
    finally:
        if loader is not None:
            loader.close_all_handles()
        os.environ.clear()
        os.environ.update(old_env)
