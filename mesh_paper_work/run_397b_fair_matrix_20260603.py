#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import argparse
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import requests


MODEL_PATH = "/mnt/data2/models/Qwen3.5-397B-A17B-TEXTONLY"
KT_WEIGHT_PATH = "/mnt/data2/models/Qwen3.5-397B-A17B-AMXINT4-NUMA2-MESH-FIXED"
MODEL_LABEL = "Qwen3.5-397B-A17B"
KT_METHOD = "AMXINT4"
CURRENT_VENV = "/mnt/data3/work/venv-ktransformers-raqiu/bin"
MMAP_PY = "/mnt/data3/work/venvs/qr-sglang-clean/bin/python"
OLD_KT_LIB = "/mnt/data3/work/ktransformers_mmap_d7b5b49/kt-kernel/build/lib.linux-x86_64-cpython-311"
COMPAT_SGLANG = "/mnt/data3/work/sglang_mmap_d7b5b49_compat"

PROMPT_FILE = Path("/mnt/data3/work/mesh_standard_5domain_prompts_20260602.json")
DEFAULT_OUT_ROOT = Path("/mnt/data3/work/mesh_paper_397b_runs/fair_5domain_20260602")
PORT = 31720
CUDA = "0,1,2,3"
TP_SIZE = 4
GPU_EXPERTS = 16
DEFERRED_EXPERTS = 3
KT_CPUINFER = 88
MEMORY_MAX = "768G"
MEM_FRACTION_STATIC = "0.98"
REQUEST_TIMEOUT_S = 300
MESH_LAYER_MODE = False
MESH_GLOBAL_POOL_CAPACITY = 3072
MESH_PREFILL_LAYER_WINDOW = 0
MESH_PREFILL_STATIC_EXPERTS: int | None = None
MESH_EARLY_LAYER_EXPERTS = "global"
MESH_DECODE_TRANSITION_SYNC = True
MESH_DECODE_TRANSITION_ON_COLD_Q1 = False
MESH_DECODE_TRANSITION_FILL_LIMIT: int | None = None
MESH_BOOTSTRAP_PREFETCH = True
MESH_CPU_BUFFER_PIN_MEMORY = True
MESH_FULL_GATE = True
MESH_POOL_TRACE = False
MESH_PREFILL_STREAM_TRACE = False
MESH_DECODE_TRANSITION_TRACE = False
IOURING_DIRECT = "1"
BF16_EXPERT_CACHE: str | None = None
BF16_EXPERT_CACHE_DIR = ""
DISABLE_CUDA_GRAPH = False
DISABLE_TORCH_DYNAMO = False

FAIL_MARKERS = (
    "Scheduler hit an exception",
    "Received sigquit from a child process",
    "Fatal Python error",
    "Segmentation fault",
    "OOM killer",
    "oom-kill",
    "out of memory",
    "A process of this unit has been killed by the OOM killer",
    "AssertionError:",
    "CUDA out of memory",
    "unrecognized arguments:",
    "Traceback (most recent call last)",
    "RuntimeError:",
    "invalid device ordinal",
    "io_uring direct I/O requires",
    "Broken pipe",
    "Force exiting",
)

PREFILL_RE = re.compile(
    r"Prefill batch, .*?#new-token: (?P<tokens>\d+), .*?input throughput \(token/s\): (?P<tps>[0-9.]+)"
)
DECODE_RE = re.compile(r"Decode batch, .*?gen throughput \(token/s\): (?P<tps>[0-9.]+)")


def read_text(path: Path, max_bytes: int | None = None) -> str:
    try:
        if max_bytes is None:
            return path.read_text(encoding="utf-8", errors="replace")
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def code_version_policy(mode: str) -> dict[str, Any]:
    if mode == "mmap":
        return {
            "code_version_policy": "legacy_old_checkout",
            "python": MMAP_PY,
            "ktransformers_lib": OLD_KT_LIB,
            "sglang_compat": COMPAT_SGLANG,
        }
    return {
        "code_version_policy": "current_latest",
        "python": f"{CURRENT_VENV}/python3",
        "ktransformers_tree": "current active KTransformers/KTransformers-MESH checkout",
    }


def shell(cmd: str, timeout: float | None = None) -> str:
    return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=timeout)


def pids_for_port(port: int) -> list[int]:
    out = shell(f"pgrep -f 'sglang.launch_server.*{port}' || true", timeout=5).strip()
    return [int(x) for x in out.split() if x.isdigit()]


def cleanup(port: int, name: str) -> None:
    pat = f"sglang.launch_server.*{port}|{re.escape(name)}"
    subprocess.run(f"pkill -TERM -f '{pat}'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)
    subprocess.run(f"pkill -KILL -f '{pat}'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)


def cgroup_path_for_pid(pid: int) -> Path | None:
    try:
        lines = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
    except OSError:
        return None
    if not lines:
        return None
    rel = lines[-1].split(":", 2)[-1].lstrip("/")
    path = Path("/sys/fs/cgroup") / rel
    return path if path.exists() else None


def read_memory_snapshot(cg: Path | None) -> dict[str, Any]:
    if cg is None or not cg.exists():
        return {}
    out: dict[str, Any] = {"cgroup": str(cg)}
    for name in ("memory.current", "memory.peak", "memory.max", "memory.swap.max"):
        p = cg / name
        if p.exists():
            value = p.read_text().strip()
            try:
                out[name.replace(".", "_")] = int(value)
            except ValueError:
                out[name.replace(".", "_")] = value
    stat = cg / "memory.stat"
    if stat.exists():
        for line in stat.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    out[parts[0]] = int(parts[1])
                except ValueError:
                    pass
    events = cg / "memory.events"
    if events.exists():
        for line in events.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    out[f"event_{parts[0]}"] = int(parts[1])
                except ValueError:
                    pass
    return out


class MemorySampler:
    def __init__(self, port: int, path: Path, interval_s: float = 1.0):
        self.port = port
        self.path = path
        self.interval_s = interval_s
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.cgroup: Path | None = None

    def __enter__(self) -> "MemorySampler":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)

    def _run(self) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            while not self.stop.is_set():
                if self.cgroup is None:
                    pids = pids_for_port(self.port)
                    if pids:
                        self.cgroup = cgroup_path_for_pid(pids[0])
                snap = read_memory_snapshot(self.cgroup)
                snap["t"] = time.time()
                f.write(json.dumps(snap, ensure_ascii=False) + "\n")
                f.flush()
                time.sleep(self.interval_s)


def parse_memory_samples(path: Path) -> dict[str, Any]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj:
                rows.append(obj)
    if not rows:
        return {"sample_count": 0}
    peak = max(rows, key=lambda r: int(r.get("memory_current", 0) or 0))
    last = rows[-1]
    cur = int(peak.get("memory_current", 0) or 0)
    out: dict[str, Any] = {
        "sample_count": len(rows),
        "oom_kill_events_max": max(int(r.get("event_oom_kill", 0) or 0) for r in rows),
        "peak_sample": {k: peak.get(k) for k in ("memory_current", "anon", "file", "file_mapped", "active_file", "inactive_file", "kernel", "pagetables", "slab") if k in peak},
        "last_sample": {k: last.get(k) for k in ("memory_current", "anon", "file", "file_mapped", "active_file", "inactive_file", "kernel", "pagetables", "slab") if k in last},
    }
    if cur:
        out["peak_gib"] = cur / 1024**3
        for k in ("anon", "file", "file_mapped", "active_file", "inactive_file", "kernel", "pagetables", "slab"):
            if k in peak:
                out[f"peak_{k}_gib"] = int(peak[k]) / 1024**3
                out[f"peak_{k}_pct_current"] = int(peak[k]) / cur
    return out


def parse_runtime_log(log_path: Path) -> dict[str, Any]:
    text = read_text(log_path)
    pre = []
    dec = []
    for line in text.splitlines():
        m = PREFILL_RE.search(line)
        if m:
            pre.append((int(m.group("tokens")), float(m.group("tps"))))
        m = DECODE_RE.search(line)
        if m:
            dec.append(float(m.group("tps")))
    pre_hmean = None
    if pre:
        toks = sum(t for t, _ in pre if t > 0)
        secs = sum(t / s for t, s in pre if t > 0 and s > 0)
        pre_hmean = toks / secs if secs else None
    dec_vals = [x for x in dec if x > 0]
    return {
        "prefill_event_count": len(pre),
        "decode_event_count": len(dec),
        "prefill_tps_hmean_log": pre_hmean,
        "decode_tps_avg_log": sum(dec_vals) / len(dec_vals) if dec_vals else None,
        "decode_tps_last_log": dec_vals[-1] if dec_vals else None,
        "decode_tps_min_log": min(dec_vals) if dec_vals else None,
        "decode_tps_max_log": max(dec_vals) if dec_vals else None,
    }


def parse_expert_stats(path: Path) -> dict[str, Any]:
    latest: dict[int, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            layer = obj.get("layer")
            if layer is not None:
                latest[int(layer)] = obj
    if not latest:
        return {"exists": path.exists(), "layers": 0, "hit_rate": None}
    scalar_keys = (
        "hit_count",
        "miss_count",
        "cold_miss_count",
        "in_flight_miss_count",
        "promote_count",
        "demote_count",
        "eviction_count",
        "prefetch_count",
        "async_prefetch_count",
        "prefetch_hit_count",
        "iouring_read_request_count",
        "iouring_read_bytes",
        "state_defer_token_count",
        "state_defer_cpu_topk_count",
        "state_defer_nonready_count",
        "state_defer_deferred_count",
        "state_defer_overflow_immediate_count",
    )
    totals: dict[str, int] = {}
    for obj in latest.values():
        for key in scalar_keys:
            totals[key] = totals.get(key, 0) + int(obj.get(key, 0) or 0)
    hit = totals.get("hit_count", 0)
    miss = totals.get("miss_count", 0)
    requests = totals.get("iouring_read_request_count", 0)
    read_bytes = totals.get("iouring_read_bytes", 0)
    return {
        "exists": True,
        "layers": len(latest),
        "total_hit": hit,
        "total_miss": miss,
        "hit_rate": hit / (hit + miss) if hit + miss else None,
        "iouring_read_gib": read_bytes / 1024**3,
        "iouring_read_request_count": requests,
        "avg_read_mib_per_request": read_bytes / 1048576.0 / requests if requests else None,
        "totals": totals,
    }


def load_prompts() -> tuple[list[dict[str, str]], dict[str, Any]]:
    data = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    return data["prompts"], data


def make_env(mode: str, expert_stats_path: Path, mesh_cap: int) -> tuple[str, dict[str, str]]:
    env = os.environ.copy()
    env.update(
        {
            "TMPDIR": "/mnt/data3/work/tmp",
            "XDG_CACHE_HOME": "/mnt/data3/work/xdg_cache",
            "TRITON_CACHE_DIR": "/mnt/data3/work/triton_cache",
            "HF_HOME": "/mnt/data3/work/hf_home",
            "TRANSFORMERS_CACHE": "/mnt/data3/work/hf_cache",
            "CUDA_VISIBLE_DEVICES": CUDA,
            "SGLANG_DISABLE_CUDNN_CHECK": "1",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "KT_LANG": "en",
        }
    )
    if DISABLE_TORCH_DYNAMO:
        env["TORCHDYNAMO_DISABLE"] = "1"
    for key in list(env):
        if key.startswith("KT_") and key != "KT_LANG":
            del env[key]
    if mode == "mmap":
        env["PATH"] = f"{Path(MMAP_PY).parent}:" + env.get("PATH", "")
        env["PYTHONPATH"] = f"{COMPAT_SGLANG}:{OLD_KT_LIB}" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return MMAP_PY, env
    env["PATH"] = f"{CURRENT_VENV}:" + env.get("PATH", "")
    py = f"{CURRENT_VENV}/python3"
    if mode == "full":
        env["KT_IO_BACKEND"] = "FULL"
    elif mode == "mesh":
        env.update(
            {
                "KT_IO_BACKEND": "IOURING",
                "KT_IOURING_DIRECT": str(IOURING_DIRECT),
                "KT_MAX_RESIDENT_EXPERTS": str(mesh_cap),
                "KT_MAX_TIER0_EXPERTS": str(mesh_cap),
                "KT_RESIDENCY_POLICY": "sieve",
                "KT_MESH_STATE_DEFER": "1",
                "KT_MESH_DEFER_PREFETCH": "1",
                "KT_MESH_BOOTSTRAP_PREFETCH": "1" if MESH_BOOTSTRAP_PREFETCH else "0",
                "KT_ENABLE_CACHE_STATS": "1",
                "KT_EXPERT_STATS_PATH": str(expert_stats_path),
                "KT_EXPERT_STATS_DUMP_EVERY": "128",
                "KT_MESH_VERBOSE": "1",
                "KT_CPU_BUFFER_PIN_MEMORY": "1" if MESH_CPU_BUFFER_PIN_MEMORY else "0",
                "KT_MESH_FULL_GATE": "1" if MESH_FULL_GATE else "0",
            }
        )
        if MESH_POOL_TRACE:
            env["KT_MESH_POOL_TRACE"] = "1"
        if MESH_PREFILL_STREAM_TRACE:
            env["KT_MESH_PREFILL_STREAM_TRACE"] = "1"
        if MESH_DECODE_TRANSITION_TRACE:
            env["KT_MESH_DECODE_TRANSITION_TRACE"] = "1"
        if KT_METHOD.upper() == "BF16":
            env["KT_ENABLE_BF16_WAVE_RESIDENT"] = "1"
            if BF16_EXPERT_CACHE is not None:
                env["KT_MESH_BF16_EXPERT_CACHE"] = str(BF16_EXPERT_CACHE)
            if BF16_EXPERT_CACHE_DIR:
                env["KT_MESH_BF16_EXPERT_CACHE_DIR"] = str(BF16_EXPERT_CACHE_DIR)
        if MESH_LAYER_MODE:
            env.update(
                {
                    "KT_MESH_PREFILL_LAYER_MODE": "1",
                    "KT_MESH_PREFILL_LAYER_WINDOW": str(MESH_PREFILL_LAYER_WINDOW),
                    "KT_MESH_GLOBAL_POOL_CAPACITY": str(MESH_GLOBAL_POOL_CAPACITY),
                    "KT_MESH_EARLY_LAYER_EXPERTS": str(MESH_EARLY_LAYER_EXPERTS),
                    "KT_MESH_DECODE_TRANSITION_SYNC": "1" if MESH_DECODE_TRANSITION_SYNC else "0",
                    "KT_MESH_DECODE_TRANSITION_ON_COLD_Q1": "1"
                    if MESH_DECODE_TRANSITION_ON_COLD_Q1
                    else "0",
                }
            )
            if MESH_DECODE_TRANSITION_FILL_LIMIT is not None:
                env["KT_MESH_DECODE_TRANSITION_FILL_LIMIT"] = str(MESH_DECODE_TRANSITION_FILL_LIMIT)
            if MESH_PREFILL_STATIC_EXPERTS is not None:
                env["KT_MESH_PREFILL_STATIC_EXPERTS"] = str(MESH_PREFILL_STATIC_EXPERTS)
    else:
        raise ValueError(mode)
    return py, env


def launch(mode: str, run_dir: Path, name: str, log_path: Path, expert_stats_path: Path, mesh_cap: int) -> subprocess.Popen:
    py, env = make_env(mode, expert_stats_path, mesh_cap)
    cmd = [
        "systemd-run",
        "--user",
        "--scope",
        "-p",
        f"MemoryMax={MEMORY_MAX}",
        "-p",
        "MemorySwapMax=0",
        py,
        "-m",
        "sglang.launch_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--model",
        MODEL_PATH,
        "--kt-weight-path",
        KT_WEIGHT_PATH,
        "--kt-cpuinfer",
        str(KT_CPUINFER),
        "--kt-threadpool-count",
        "2",
        "--kt-num-gpu-experts",
        str(GPU_EXPERTS),
        "--kt-max-deferred-experts-per-token",
        str(DEFERRED_EXPERTS),
        "--kt-method",
        KT_METHOD,
        "--attention-backend",
        "flashinfer",
        "--trust-remote-code",
        "--mem-fraction-static",
        str(MEM_FRACTION_STATIC),
        "--chunked-prefill-size",
        "2048",
        "--max-running-requests",
        "1",
        "--max-total-tokens",
        "4096",
        "--watchdog-timeout",
        "3000",
        "--enable-mixed-chunk",
        "--tensor-parallel-size",
        str(TP_SIZE),
        "--enable-p2p-check",
        "--served-model-name",
        name,
        "--disable-shared-experts-fusion",
        "--skip-server-warmup",
    ]
    if DISABLE_CUDA_GRAPH:
        cmd.append("--disable-cuda-graph")
    config = {
        "mode": mode,
        **code_version_policy(mode),
        "cmd": cmd,
        "env_subset": {
            k: env[k]
            for k in sorted(env)
            if k.startswith("KT_")
            or k
            in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONPATH",
                "PATH",
                "TMPDIR",
                "XDG_CACHE_HOME",
                "TRITON_CACHE_DIR",
                "TORCHDYNAMO_DISABLE",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "TORCHINDUCTOR_COMPILE_THREADS",
                "MAX_JOBS",
            )
        },
        "model_label": MODEL_LABEL,
        "kt_method": KT_METHOD,
        "model": MODEL_LABEL,
        "model_path": MODEL_PATH,
        "kt_weight_path": KT_WEIGHT_PATH,
        "prompt_file": str(PROMPT_FILE),
        "tp": TP_SIZE,
        "gpu_experts": GPU_EXPERTS,
        "defer": DEFERRED_EXPERTS,
        "kt_cpuinfer": KT_CPUINFER,
        "memory_max": MEMORY_MAX,
        "mem_fraction_static": MEM_FRACTION_STATIC,
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "mesh_cap": mesh_cap if mode == "mesh" else None,
        "mesh_layer_mode": MESH_LAYER_MODE if mode == "mesh" else None,
        "mesh_global_pool_capacity": MESH_GLOBAL_POOL_CAPACITY if mode == "mesh" and MESH_LAYER_MODE else None,
        "mesh_prefill_layer_window": MESH_PREFILL_LAYER_WINDOW if mode == "mesh" and MESH_LAYER_MODE else None,
        "mesh_prefill_static_experts": MESH_PREFILL_STATIC_EXPERTS if mode == "mesh" and MESH_LAYER_MODE else None,
        "mesh_early_layer_experts": MESH_EARLY_LAYER_EXPERTS if mode == "mesh" and MESH_LAYER_MODE else None,
        "mesh_decode_transition_sync": MESH_DECODE_TRANSITION_SYNC if mode == "mesh" and MESH_LAYER_MODE else None,
        "mesh_decode_transition_on_cold_q1": MESH_DECODE_TRANSITION_ON_COLD_Q1
        if mode == "mesh" and MESH_LAYER_MODE
        else None,
        "mesh_decode_transition_fill_limit": MESH_DECODE_TRANSITION_FILL_LIMIT
        if mode == "mesh" and MESH_LAYER_MODE
        else None,
        "mesh_bootstrap_prefetch": MESH_BOOTSTRAP_PREFETCH if mode == "mesh" else None,
        "mesh_cpu_buffer_pin_memory": MESH_CPU_BUFFER_PIN_MEMORY if mode == "mesh" else None,
        "mesh_full_gate": MESH_FULL_GATE if mode == "mesh" else None,
        "mesh_pool_trace": MESH_POOL_TRACE if mode == "mesh" else None,
        "mesh_prefill_stream_trace": MESH_PREFILL_STREAM_TRACE if mode == "mesh" else None,
        "mesh_decode_transition_trace": MESH_DECODE_TRANSITION_TRACE if mode == "mesh" else None,
        "iouring_direct": IOURING_DIRECT if mode == "mesh" else None,
        "bf16_expert_cache": BF16_EXPERT_CACHE if mode == "mesh" and KT_METHOD.upper() == "BF16" else None,
        "bf16_expert_cache_dir": BF16_EXPERT_CACHE_DIR if mode == "mesh" and KT_METHOD.upper() == "BF16" else None,
        "disable_cuda_graph": DISABLE_CUDA_GRAPH,
        "disable_torch_dynamo": DISABLE_TORCH_DYNAMO,
    }
    write_json(run_dir / "config.json", config)
    logf = log_path.open("w", encoding="utf-8", errors="replace")
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)


def wait_ready(log_path: Path, timeout_s: int = 3600) -> tuple[str, str]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if r.status_code == 200:
                return "ready", ""
        except Exception:
            pass
        tail = read_text(log_path, max_bytes=80000)
        if any(marker in tail for marker in FAIL_MARKERS):
            return "fail", tail[-12000:]
        if log_path.exists() and not pids_for_port(PORT) and "Running as unit" in tail:
            return "dead", tail[-12000:]
        time.sleep(5)
    return "timeout", read_text(log_path, max_bytes=12000)


def bench_one(name: str, prompt_obj: dict[str, str], max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": name,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "messages": [{"role": "user", "content": prompt_obj["input_prompt"]}],
    }
    t0 = time.time()
    first = None
    usage = None
    chunks = 0
    output_parts: list[str] = []
    try:
        with requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT_S,
        ) as resp:
            if resp.status_code != 200:
                return {
                    "status": "http_error",
                    "id": prompt_obj["id"],
                    "domain": prompt_obj["domain"],
                    "code": resp.status_code,
                    "body": resp.text[:4000],
                }
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                data = raw[6:]
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        chunks += 1
                        output_parts.append(content)
                        if first is None:
                            first = time.time()
    except requests.RequestException as exc:
        t1 = time.time()
        text = "".join(output_parts)
        return {
            "status": "stream_error",
            "id": prompt_obj["id"],
            "domain": prompt_obj["domain"],
            "error": repr(exc),
            "total_s": t1 - t0,
            "chunks": chunks,
            "output_text": text,
            "output_chars": len(text),
            "output_words": len(re.findall(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)?", text)),
            "numbered_line_count": len(re.findall(r"(?m)^\s*\d+[\).]", text)),
        }
    t1 = time.time()
    text = "".join(output_parts)
    prompt_tokens = int((usage or {}).get("prompt_tokens", 0) or 0)
    completion_tokens = int((usage or {}).get("completion_tokens", 0) or 0)
    ttft = None if first is None else first - t0
    decode_s = None if first is None else max(t1 - first, 1e-9)
    return {
        "status": "ok",
        "id": prompt_obj["id"],
        "domain": prompt_obj["domain"],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "decode_s": decode_s,
        "total_s": t1 - t0,
        "api_prefill_tok_s": prompt_tokens / ttft if ttft and prompt_tokens else None,
        "api_decode_tok_s": completion_tokens / decode_s if decode_s and completion_tokens else None,
        "chunks": chunks,
        "output_text": text,
        "output_chars": len(text),
        "output_words": len(re.findall(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)?", text)),
        "numbered_line_count": len(re.findall(r"(?m)^\s*\d+[\).]", text)),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    dec = [float(r["api_decode_tok_s"]) for r in ok if r.get("api_decode_tok_s")]
    pre = [float(r["api_prefill_tok_s"]) for r in ok if r.get("api_prefill_tok_s")]
    return {
        "row_count": len(rows),
        "ok_count": len(ok),
        "non_ok_count": len(rows) - len(ok),
        "decode_tok_s_avg_api": sum(dec) / len(dec) if dec else None,
        "decode_tok_s_min_api": min(dec) if dec else None,
        "decode_tok_s_max_api": max(dec) if dec else None,
        "prefill_tok_s_avg_api": sum(pre) / len(pre) if pre else None,
        "completion_tokens_total": sum(int(r.get("completion_tokens", 0) or 0) for r in ok),
        "prompt_tokens_total": sum(int(r.get("prompt_tokens", 0) or 0) for r in ok),
        "output_words_total": sum(int(r.get("output_words", 0) or 0) for r in ok),
        "numbered_lines_total": sum(int(r.get("numbered_line_count", 0) or 0) for r in ok),
    }


def run_mode(mode: str, prompt_data: dict[str, Any], prompts: list[dict[str, str]], out_root: Path, mesh_cap: int) -> dict[str, Any]:
    label = mode if mode != "mesh" else f"mesh_cap{mesh_cap}" + ("_layer" if MESH_LAYER_MODE else "")
    if mode == "mesh" and not MESH_CPU_BUFFER_PIN_MEMORY:
        label += "_nopin"
    name = f"Q397-fair-{label}-tp{TP_SIZE}-ge{GPU_EXPERTS}-defer{DEFERRED_EXPERTS}"
    run_dir = out_root / label
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "server.log"
    mem_path = run_dir / "memory_samples.jsonl"
    expert_stats_path = run_dir / "expert_stats.jsonl"
    summary_path = run_dir / "summary.json"
    for path in (log_path, mem_path, expert_stats_path, summary_path):
        if path.exists():
            path.unlink()
    cleanup(PORT, name)
    start = time.time()
    with MemorySampler(PORT, mem_path):
        proc = launch(mode, run_dir, name, log_path, expert_stats_path, mesh_cap)
        status, fail_tail = wait_ready(log_path)
        entry: dict[str, Any] = {
            "mode": mode,
            **code_version_policy(mode),
            "name": name,
            "status": status,
            "launcher_pid": proc.pid,
            "ready_after_s": time.time() - start,
            "model_label": MODEL_LABEL,
            "kt_method": KT_METHOD,
            "model": MODEL_LABEL,
            "model_path": MODEL_PATH,
            "kt_weight_path": KT_WEIGHT_PATH,
            "tp": TP_SIZE,
            "cuda_visible_devices": CUDA,
            "gpu_experts": GPU_EXPERTS,
            "defer": DEFERRED_EXPERTS,
            "mesh_cap": mesh_cap if mode == "mesh" else None,
            "mesh_layer_mode": MESH_LAYER_MODE if mode == "mesh" else None,
            "mesh_global_pool_capacity": MESH_GLOBAL_POOL_CAPACITY if mode == "mesh" and MESH_LAYER_MODE else None,
            "mesh_prefill_layer_window": MESH_PREFILL_LAYER_WINDOW if mode == "mesh" and MESH_LAYER_MODE else None,
            "mesh_prefill_static_experts": MESH_PREFILL_STATIC_EXPERTS if mode == "mesh" and MESH_LAYER_MODE else None,
            "mesh_early_layer_experts": MESH_EARLY_LAYER_EXPERTS if mode == "mesh" and MESH_LAYER_MODE else None,
            "mesh_decode_transition_sync": MESH_DECODE_TRANSITION_SYNC if mode == "mesh" and MESH_LAYER_MODE else None,
            "mesh_decode_transition_on_cold_q1": MESH_DECODE_TRANSITION_ON_COLD_Q1
            if mode == "mesh" and MESH_LAYER_MODE
            else None,
            "mesh_decode_transition_fill_limit": MESH_DECODE_TRANSITION_FILL_LIMIT
            if mode == "mesh" and MESH_LAYER_MODE
            else None,
            "mesh_bootstrap_prefetch": MESH_BOOTSTRAP_PREFETCH if mode == "mesh" else None,
            "mesh_cpu_buffer_pin_memory": MESH_CPU_BUFFER_PIN_MEMORY if mode == "mesh" else None,
            "mesh_full_gate": MESH_FULL_GATE if mode == "mesh" else None,
            "iouring_direct": IOURING_DIRECT if mode == "mesh" else None,
            "bf16_expert_cache": BF16_EXPERT_CACHE if mode == "mesh" and KT_METHOD.upper() == "BF16" else None,
            "bf16_expert_cache_dir": BF16_EXPERT_CACHE_DIR if mode == "mesh" and KT_METHOD.upper() == "BF16" else None,
            "disable_cuda_graph": DISABLE_CUDA_GRAPH,
            "disable_torch_dynamo": DISABLE_TORCH_DYNAMO,
            "memory_max": MEMORY_MAX,
            "mem_fraction_static": MEM_FRACTION_STATIC,
            "request_timeout_s": REQUEST_TIMEOUT_S,
            "prompt_file": str(PROMPT_FILE),
            "prompt_set": {
                "name": prompt_data.get("name"),
                "version": prompt_data.get("version"),
                "num_prompts": len(prompts),
                "generation_defaults": prompt_data.get("generation_defaults"),
                "output_contract": prompt_data.get("output_contract"),
            },
            "log": str(log_path),
            "memory_samples": str(mem_path),
            "expert_stats": str(expert_stats_path) if mode == "mesh" else "not_applicable",
        }
        if status == "ready":
            max_tokens = int(prompt_data["generation_defaults"]["max_new_tokens"])
            rows = []
            for prompt in prompts:
                try:
                    row = bench_one(name, prompt, max_tokens=max_tokens)
                except Exception as exc:
                    row = {
                        "id": prompt.get("id"),
                        "domain": prompt.get("domain"),
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback_tail": traceback.format_exc()[-4000:],
                    }
                    rows.append(row)
                    row_summary = summarize_rows(rows)
                    write_json(
                        summary_path,
                        {
                            **entry,
                            "status": "partial",
                            "expected_prompt_count": len(prompts),
                            "rows": rows,
                            "row_summary": row_summary,
                            "fail_tail": f"Prompt {prompt.get('id')} failed: {type(exc).__name__}: {exc}",
                        },
                    )
                    break
                rows.append(row)
                row_summary = summarize_rows(rows)
                interim_status = (
                    "ready"
                    if row_summary["ok_count"] == len(prompts)
                    else "partial"
                )
                write_json(
                    summary_path,
                    {
                        **entry,
                        "status": interim_status,
                        "expected_prompt_count": len(prompts),
                        "rows": rows,
                        "row_summary": row_summary,
                    },
                )
            entry["rows"] = rows
            entry["row_summary"] = summarize_rows(rows)
            entry["expected_prompt_count"] = len(prompts)
            if entry["row_summary"]["ok_count"] != len(prompts):
                entry["status"] = "partial"
                entry["fail_tail"] = (
                    f"Only {entry['row_summary']['ok_count']}/{len(prompts)} prompts completed successfully"
                )
        else:
            entry["fail_tail"] = fail_tail
        entry["elapsed_s"] = time.time() - start
        entry["runtime_log_summary"] = parse_runtime_log(log_path)
        entry["memory_summary"] = parse_memory_samples(mem_path)
        entry["expert_summary"] = parse_expert_stats(expert_stats_path) if mode == "mesh" else {"hit_rate": "not_applicable"}
        write_json(summary_path, entry)
    cleanup(PORT, name)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["full", "mmap", "mesh"], choices=["full", "mmap", "mesh"])
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--kt-weight-path", default="")
    parser.add_argument("--kt-method", default=KT_METHOD)
    parser.add_argument("--model-label", default=MODEL_LABEL)
    parser.add_argument("--mesh-cap", type=int, default=512)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--cuda", default=CUDA)
    parser.add_argument("--tp", type=int, default=TP_SIZE)
    parser.add_argument("--gpu-experts", type=int, default=GPU_EXPERTS)
    parser.add_argument("--defer", type=int, default=DEFERRED_EXPERTS)
    parser.add_argument("--kt-cpuinfer", type=int, default=KT_CPUINFER)
    parser.add_argument("--memory-max", default=MEMORY_MAX)
    parser.add_argument("--mem-fraction-static", default=MEM_FRACTION_STATIC)
    parser.add_argument("--request-timeout-s", type=int, default=REQUEST_TIMEOUT_S)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--mesh-layer-mode", action="store_true")
    parser.add_argument("--mesh-global-pool-capacity", type=int, default=MESH_GLOBAL_POOL_CAPACITY)
    parser.add_argument("--mesh-prefill-layer-window", type=int, default=MESH_PREFILL_LAYER_WINDOW)
    parser.add_argument("--mesh-prefill-static-experts", type=int)
    parser.add_argument("--mesh-early-layer-experts", default=MESH_EARLY_LAYER_EXPERTS)
    parser.add_argument("--mesh-decode-transition-sync", choices=["0", "1"], default="1")
    parser.add_argument("--mesh-decode-transition-on-cold-q1", choices=["0", "1"], default="0")
    parser.add_argument("--mesh-decode-transition-fill-limit", type=int)
    parser.add_argument("--mesh-bootstrap-prefetch", choices=["0", "1"], default="1")
    parser.add_argument("--mesh-cpu-buffer-pin-memory", choices=["0", "1"], default="1")
    parser.add_argument("--mesh-full-gate", choices=["0", "1"], default="1")
    parser.add_argument("--mesh-pool-trace", action="store_true")
    parser.add_argument("--mesh-prefill-stream-trace", action="store_true")
    parser.add_argument("--mesh-decode-transition-trace", action="store_true")
    parser.add_argument("--iouring-direct", choices=["0", "1"], default="1")
    parser.add_argument("--bf16-expert-cache", choices=["0", "1"])
    parser.add_argument("--bf16-expert-cache-dir", default="")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--disable-torch-dynamo", action="store_true")
    return parser.parse_args()


def main() -> int:
    global MODEL_PATH, KT_WEIGHT_PATH, MODEL_LABEL, KT_METHOD
    global CUDA, TP_SIZE, GPU_EXPERTS, DEFERRED_EXPERTS, KT_CPUINFER, MEMORY_MAX, MEM_FRACTION_STATIC, REQUEST_TIMEOUT_S
    global MESH_LAYER_MODE, MESH_GLOBAL_POOL_CAPACITY, MESH_PREFILL_LAYER_WINDOW, MESH_PREFILL_STATIC_EXPERTS
    global MESH_EARLY_LAYER_EXPERTS, MESH_DECODE_TRANSITION_SYNC, MESH_DECODE_TRANSITION_ON_COLD_Q1
    global MESH_DECODE_TRANSITION_FILL_LIMIT
    global MESH_BOOTSTRAP_PREFETCH, MESH_CPU_BUFFER_PIN_MEMORY, MESH_FULL_GATE, MESH_POOL_TRACE
    global MESH_PREFILL_STREAM_TRACE, MESH_DECODE_TRANSITION_TRACE
    global IOURING_DIRECT, BF16_EXPERT_CACHE, BF16_EXPERT_CACHE_DIR
    global DISABLE_CUDA_GRAPH, DISABLE_TORCH_DYNAMO
    args = parse_args()
    MODEL_PATH = str(args.model_path)
    KT_WEIGHT_PATH = str(args.kt_weight_path or args.model_path)
    MODEL_LABEL = str(args.model_label)
    KT_METHOD = str(args.kt_method).upper()
    CUDA = args.cuda
    TP_SIZE = int(args.tp)
    GPU_EXPERTS = int(args.gpu_experts)
    DEFERRED_EXPERTS = int(args.defer)
    KT_CPUINFER = int(args.kt_cpuinfer)
    MEMORY_MAX = args.memory_max
    MEM_FRACTION_STATIC = str(args.mem_fraction_static)
    REQUEST_TIMEOUT_S = int(args.request_timeout_s)
    MESH_LAYER_MODE = bool(args.mesh_layer_mode)
    MESH_GLOBAL_POOL_CAPACITY = int(args.mesh_global_pool_capacity)
    MESH_PREFILL_LAYER_WINDOW = int(args.mesh_prefill_layer_window)
    MESH_PREFILL_STATIC_EXPERTS = None if args.mesh_prefill_static_experts is None else int(args.mesh_prefill_static_experts)
    MESH_EARLY_LAYER_EXPERTS = str(args.mesh_early_layer_experts)
    MESH_DECODE_TRANSITION_SYNC = args.mesh_decode_transition_sync == "1"
    MESH_DECODE_TRANSITION_ON_COLD_Q1 = args.mesh_decode_transition_on_cold_q1 == "1"
    MESH_DECODE_TRANSITION_FILL_LIMIT = (
        None if args.mesh_decode_transition_fill_limit is None else int(args.mesh_decode_transition_fill_limit)
    )
    MESH_BOOTSTRAP_PREFETCH = args.mesh_bootstrap_prefetch == "1"
    MESH_CPU_BUFFER_PIN_MEMORY = args.mesh_cpu_buffer_pin_memory == "1"
    MESH_FULL_GATE = args.mesh_full_gate == "1"
    MESH_POOL_TRACE = bool(args.mesh_pool_trace)
    MESH_PREFILL_STREAM_TRACE = bool(args.mesh_prefill_stream_trace)
    MESH_DECODE_TRANSITION_TRACE = bool(args.mesh_decode_transition_trace)
    IOURING_DIRECT = str(args.iouring_direct)
    BF16_EXPERT_CACHE = args.bf16_expert_cache
    BF16_EXPERT_CACHE_DIR = str(args.bf16_expert_cache_dir or "")
    DISABLE_CUDA_GRAPH = bool(args.disable_cuda_graph)
    DISABLE_TORCH_DYNAMO = bool(args.disable_torch_dynamo)
    prompts, prompt_data = load_prompts()
    if args.prompt_offset:
        prompts = prompts[int(args.prompt_offset) :]
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    if args.max_new_tokens is not None:
        prompt_data = dict(prompt_data)
        prompt_data["generation_defaults"] = dict(prompt_data["generation_defaults"])
        prompt_data["generation_defaults"]["max_new_tokens"] = args.max_new_tokens
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "prompt_snapshot.json", prompt_data)
    write_json(
        out_root / "run_matrix.json",
        {
            "model": MODEL_LABEL,
            "kt_method": KT_METHOD,
            "modes": args.modes,
            "model_path": MODEL_PATH,
            "kt_weight_path": KT_WEIGHT_PATH,
            "cuda": CUDA,
            "cgroup_memory_max": MEMORY_MAX,
            "tp": TP_SIZE,
            "gpu_experts": GPU_EXPERTS,
            "defer": DEFERRED_EXPERTS,
            "kt_cpuinfer": KT_CPUINFER,
            "mem_fraction_static": MEM_FRACTION_STATIC,
            "prompt_file": str(PROMPT_FILE),
            "prompt_offset": int(args.prompt_offset),
            "prompt_count": len(prompts),
            "max_new_tokens": prompt_data.get("generation_defaults", {}).get("max_new_tokens"),
            "code_version_policy": {
                "full": "current_latest",
                "mesh": "current_latest",
                "mmap": "legacy_old_checkout",
                "fill_alias": "full",
            },
            "mesh_cpu_buffer_pin_memory": MESH_CPU_BUFFER_PIN_MEMORY,
            "mesh_full_gate": MESH_FULL_GATE,
            "mesh_pool_trace": MESH_POOL_TRACE,
            "mesh_prefill_static_experts": MESH_PREFILL_STATIC_EXPERTS,
            "mesh_decode_transition_fill_limit": MESH_DECODE_TRANSITION_FILL_LIMIT,
            "disable_cuda_graph": DISABLE_CUDA_GRAPH,
            "disable_torch_dynamo": DISABLE_TORCH_DYNAMO,
        },
    )
    modes = args.modes
    results = []
    aggregate_path = out_root / "aggregate_summary.json"
    if aggregate_path.exists():
        try:
            results = json.loads(aggregate_path.read_text(encoding="utf-8")).get("results", [])
        except Exception:
            results = []
    for mode in modes:
        result = run_mode(mode, prompt_data, prompts, out_root=out_root, mesh_cap=args.mesh_cap)
        results.append(result)
        write_json(aggregate_path, {"results": results})
    compact = []
    for result in results:
        compact.append(
            {
                "mode": result.get("mode"),
                "mesh_cap": result.get("mesh_cap"),
                "mesh_layer_mode": result.get("mesh_layer_mode"),
                "mesh_global_pool_capacity": result.get("mesh_global_pool_capacity"),
                "mesh_prefill_layer_window": result.get("mesh_prefill_layer_window"),
                "mesh_prefill_static_experts": result.get("mesh_prefill_static_experts"),
                "mesh_early_layer_experts": result.get("mesh_early_layer_experts"),
                "mesh_decode_transition_fill_limit": result.get("mesh_decode_transition_fill_limit"),
                "mesh_bootstrap_prefetch": result.get("mesh_bootstrap_prefetch"),
                "mesh_cpu_buffer_pin_memory": result.get("mesh_cpu_buffer_pin_memory"),
                "mesh_full_gate": result.get("mesh_full_gate"),
                "disable_cuda_graph": result.get("disable_cuda_graph"),
                "disable_torch_dynamo": result.get("disable_torch_dynamo"),
                "mem_fraction_static": result.get("mem_fraction_static"),
                "status": result.get("status"),
                "ok_count": (result.get("row_summary") or {}).get("ok_count"),
                "expected_prompt_count": result.get("expected_prompt_count"),
                "decode_tok_s_avg_api": (result.get("row_summary") or {}).get("decode_tok_s_avg_api"),
                "prefill_tok_s_avg_api": (result.get("row_summary") or {}).get("prefill_tok_s_avg_api"),
                "completion_tokens_total": (result.get("row_summary") or {}).get("completion_tokens_total"),
                "peak_gib": (result.get("memory_summary") or {}).get("peak_gib"),
                "peak_anon_gib": (result.get("memory_summary") or {}).get("peak_anon_gib"),
                "peak_file_gib": (result.get("memory_summary") or {}).get("peak_file_gib"),
                "hit_rate": (result.get("expert_summary") or {}).get("hit_rate"),
                "iouring_read_gib": (result.get("expert_summary") or {}).get("iouring_read_gib"),
            }
        )
    write_json(out_root / "compact_summary.json", compact)
    print(json.dumps({"out_root": str(out_root), "compact": compact}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
