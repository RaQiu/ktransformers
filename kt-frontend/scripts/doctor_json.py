#!/usr/bin/env python3
"""
Doctor JSON output helper for KTransformers Electron frontend.
Runs all environment checks and outputs results as JSON to stdout.
Usage: python doctor_json.py
"""

import json
import platform
import sys

def run_checks() -> list:
    checks = []

    # 1. Python version
    python_version = platform.python_version()
    parts = python_version.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
        python_ok = major >= 3 and minor >= 10
    except Exception:
        python_ok = False

    checks.append({
        "name": "Python",
        "status": "ok" if python_ok else "error",
        "value": python_version,
        "hint": "Python 3.10+ required" if not python_ok else None,
    })

    # 2. CUDA
    try:
        from kt_kernel.cli.utils.environment import detect_cuda_version
        cuda_version = detect_cuda_version()
    except Exception:
        cuda_version = None
    checks.append({
        "name": "CUDA",
        "status": "ok" if cuda_version else "warning",
        "value": cuda_version or "Not found",
        "hint": "CUDA optional but recommended for GPU acceleration" if not cuda_version else None,
    })

    # 3. GPU detection
    try:
        from kt_kernel.cli.utils.environment import detect_gpus
        gpus = detect_gpus()
        if gpus:
            gpu_names = ", ".join(g.name for g in gpus)
            total_vram = sum(g.vram_gb for g in gpus)
            checks.append({
                "name": "GPU",
                "status": "ok",
                "value": f"{len(gpus)} GPU(s): {gpu_names}",
                "hint": f"Total VRAM: {total_vram}GB",
            })
        else:
            checks.append({
                "name": "GPU",
                "status": "warning",
                "value": "No GPU detected",
                "hint": "GPU recommended for best performance",
            })
    except Exception as e:
        checks.append({"name": "GPU", "status": "warning", "value": f"Detection failed: {e}", "hint": None})

    # 4. CPU info
    try:
        from kt_kernel.cli.utils.environment import detect_cpu_info
        cpu_info = detect_cpu_info()
        checks.append({
            "name": "CPU",
            "status": "ok",
            "value": f"{cpu_info.name}, {cpu_info.cores} cores / {cpu_info.threads} threads",
            "hint": None,
        })

        # 5. CPU ISA
        isa_list = cpu_info.instruction_sets
        has_amx = any(isa.startswith("AMX") for isa in isa_list)
        has_avx512 = any(isa.startswith("AVX512") for isa in isa_list)
        has_avx2 = "AVX2" in isa_list
        if has_amx:
            isa_status, isa_hint = "ok", "AMX available - best performance"
        elif has_avx512:
            isa_status, isa_hint = "ok", "AVX512 available - good performance"
        elif has_avx2:
            isa_status, isa_hint = "warning", "AVX2 only - upgrade CPU for better performance"
        else:
            isa_status, isa_hint = "error", "AVX2 required for kt-kernel"
        display_isa = isa_list[:8] if len(isa_list) > 8 else isa_list
        isa_display = ", ".join(display_isa)
        if len(isa_list) > 8:
            isa_display += f" (+{len(isa_list) - 8} more)"
        checks.append({
            "name": "CPU ISA",
            "status": isa_status,
            "value": isa_display or "None detected",
            "hint": isa_hint,
        })

        # 6. NUMA
        numa_nodes = cpu_info.numa_nodes
        checks.append({
            "name": "NUMA",
            "status": "ok",
            "value": f"{numa_nodes} node(s)",
            "hint": f"{cpu_info.threads // numa_nodes} threads/node" if numa_nodes > 1 else None,
        })
    except Exception as e:
        checks.append({"name": "CPU", "status": "warning", "value": f"Detection failed: {e}", "hint": None})

    # 7. kt-kernel
    try:
        import glob, os
        import kt_kernel
        version = getattr(kt_kernel, "__version__", "unknown")
        variant = getattr(kt_kernel, "__cpu_variant__", "unknown")
        kt_dir = os.path.dirname(kt_kernel.__file__)
        so_files = glob.glob(os.path.join(kt_dir, "_kt_kernel_ext_*.so"))
        available = sorted({os.path.basename(f).split("_")[3] for f in so_files})
        if variant == "amx":
            kt_status, kt_hint = "ok", "AMX variant - optimal"
        elif variant.startswith("avx512"):
            kt_status, kt_hint = "ok", "AVX512 variant - good"
        elif variant == "avx2":
            kt_status, kt_hint = "warning", "AVX2 variant - upgrade CPU for AMX"
        else:
            kt_status, kt_hint = "warning", f"Variant: {variant}"
        checks.append({
            "name": "kt-kernel",
            "status": kt_status,
            "value": f"v{version} ({variant.upper()})",
            "hint": kt_hint,
        })
    except ImportError:
        checks.append({
            "name": "kt-kernel",
            "status": "error",
            "value": "Not installed",
            "hint": "pip install kt-kernel",
        })
    except Exception as e:
        checks.append({"name": "kt-kernel", "status": "error", "value": str(e), "hint": None})

    # 8. RAM
    try:
        from kt_kernel.cli.utils.environment import detect_memory_info
        mem = detect_memory_info()
        ram_ok = mem.total_gb >= 32
        mem_value = f"{mem.available_gb}GB available / {mem.total_gb}GB total"
        if mem.frequency_mhz and mem.type:
            mem_value += f" ({mem.type} @ {mem.frequency_mhz}MHz)"
        checks.append({
            "name": "RAM",
            "status": "ok" if ram_ok else "warning",
            "value": mem_value,
            "hint": "32GB+ recommended for large models" if not ram_ok else None,
        })
    except Exception as e:
        checks.append({"name": "RAM", "status": "warning", "value": f"Detection failed: {e}", "hint": None})

    # 9. Disk
    try:
        from kt_kernel.cli.config.settings import get_settings
        from kt_kernel.cli.utils.environment import detect_disk_space_gb
        settings = get_settings()
        model_paths = settings.get_model_paths()
        for i, p in enumerate(model_paths):
            avail, total = detect_disk_space_gb(str(p))
            label = f"Disk (path {i+1})" if len(model_paths) > 1 else "Disk"
            checks.append({
                "name": label,
                "status": "ok" if avail >= 100 else "warning",
                "value": f"{avail}GB free at {p}",
                "hint": "100GB+ free recommended" if avail < 100 else None,
            })
    except Exception as e:
        checks.append({"name": "Disk", "status": "warning", "value": f"Detection failed: {e}", "hint": None})

    # 10. Key packages
    try:
        from kt_kernel.cli.utils.environment import get_installed_package_version
        for pkg, required in [("torch", True), ("transformers", True), ("sglang", False)]:
            ver = get_installed_package_version(pkg)
            checks.append({
                "name": pkg,
                "status": "ok" if ver else ("error" if required else "warning"),
                "value": ver or "Not installed",
                "hint": f"pip install {pkg}" if not ver else None,
            })
    except Exception:
        pass

    # 11. SGLang kt-kernel support
    try:
        from kt_kernel.cli.utils.sglang_checker import check_sglang_installation, check_sglang_kt_kernel_support
        sglang_info = check_sglang_installation()
        if sglang_info["installed"]:
            kt_support = check_sglang_kt_kernel_support(use_cache=False, silent=True)
            checks.append({
                "name": "SGLang kt-kernel",
                "status": "ok" if kt_support["supported"] else "error",
                "value": "Supported" if kt_support["supported"] else "Not supported",
                "hint": None if kt_support["supported"] else "Reinstall SGLang from kvcache-ai/sglang",
            })
    except Exception:
        pass

    return checks


if __name__ == "__main__":
    try:
        checks = run_checks()
        print(json.dumps({"checks": checks, "success": True}))
    except Exception as e:
        print(json.dumps({"checks": [], "success": False, "error": str(e)}))
