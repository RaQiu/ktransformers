#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import requests


SERVER_PY = "/mnt/data3/work/venvs/qr-sglang-clean/bin/python"
OLD_KT_LIB = "/mnt/data3/work/ktransformers_mmap_d7b5b49/kt-kernel/build/lib.linux-x86_64-cpython-311"
COMPAT_SGLANG = "/mnt/data3/work/sglang_mmap_d7b5b49_compat"
DEFAULT_PROMPT_FILE = "/mnt/data3/work/mesh_standard_5domain_prompts_20260602.json"

MODEL_MATRIX: dict[tuple[str, str], dict[str, Any]] = {
    ("35b", "amxint4"): {
        "model_name": "Qwen3.5-35B-A3B",
        "model_path": "/mnt/data3/models/Qwen3.5-35B-A3B",
        "kt_weight_path": "/mnt/data2/models/Qwen3.5-35B-A3B-AMXINT4-NUMA2-MESH",
        "kt_method": "AMXINT4",
        "ge": 32,
        "kt_cpuinfer": 96,
        "mem_fraction_static": "0.85",
        "chunked_prefill_size": "4096",
        "attention_backend": "triton",
    },
    ("35b", "bf16"): {
        "model_name": "Qwen3.5-35B-A3B",
        "model_path": "/mnt/data3/models/Qwen3.5-35B-A3B",
        "kt_weight_path": "/mnt/data3/models/Qwen3.5-35B-A3B",
        "kt_method": "BF16",
        "ge": 32,
        "kt_cpuinfer": 96,
        "mem_fraction_static": "0.85",
        "chunked_prefill_size": "4096",
        "attention_backend": "triton",
    },
    ("397b", "amxint4"): {
        "model_name": "Qwen3.5-397B-A17B",
        "model_path": "/mnt/data2/models/Qwen3.5-397B-A17B-TEXTONLY",
        "kt_weight_path": "/mnt/data2/models/Qwen3.5-397B-A17B-AMXINT4-NUMA2-MESH-FIXED",
        "kt_method": "AMXINT4",
        "ge": 16,
        "kt_cpuinfer": 88,
        "mem_fraction_static": "0.70",
        "chunked_prefill_size": "2048",
        "attention_backend": "flashinfer",
    },
    ("397b", "bf16"): {
        "model_name": "Qwen3.5-397B-A17B",
        "model_path": "/mnt/data2/models/Qwen3.5-397B-A17B",
        "kt_weight_path": "/mnt/data2/models/Qwen3.5-397B-A17B",
        "kt_method": "BF16",
        "ge": 16,
        "kt_cpuinfer": 88,
        "mem_fraction_static": "0.70",
        "chunked_prefill_size": "2048",
        "attention_backend": "flashinfer",
    },
}

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
)

PREFILL_RE = re.compile(
    r"Prefill batch, .*?#new-token: (?P<tokens>\d+), .*?input throughput \(token/s\): (?P<tps>[0-9.]+)"
)
DECODE_RE = re.compile(r"Decode batch, .*?gen throughput \(token/s\): (?P<tps>[0-9.]+)")


def shell(cmd: str, timeout: float | None = None) -> str:
    return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=timeout)


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


def load_prompts(path: Path) -> list[dict[str, str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [{"id": f"prompt_{i}", "domain": "unknown", "input_prompt": str(p)} for i, p in enumerate(obj)]
    prompts = obj.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError(f"Unsupported prompt file schema: {path}")
    out = []
    for i, item in enumerate(prompts):
        if not isinstance(item, dict) or "input_prompt" not in item:
            raise ValueError(f"Prompt {i} lacks input_prompt in {path}")
        out.append(
            {
                "id": str(item.get("id", f"prompt_{i}")),
                "domain": str(item.get("domain", "unknown")),
                "input_prompt": str(item["input_prompt"]),
            }
        )
    return out


def gpu_snapshot() -> list[dict[str, int]]:
    text = shell(
        "nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits",
        timeout=15,
    )
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows.append(
                {
                    "index": int(parts[0]),
                    "memory_used_mib": int(parts[1]),
                    "memory_free_mib": int(parts[2]),
                    "util_gpu_pct": int(parts[3]),
                }
            )
    return rows


def choose_gpus(tp: int) -> list[int]:
    rows = gpu_snapshot()
    candidates = [r for r in rows if r["memory_free_mib"] >= 35000 and r["util_gpu_pct"] <= 30]
    candidates.sort(key=lambda r: (r["memory_free_mib"], -r["util_gpu_pct"]), reverse=True)
    if len(candidates) < tp:
        raise RuntimeError(f"Need {tp} free GPUs, found {len(candidates)} candidates: {rows}")
    # Keep physical order after selecting the cleanest devices. NCCL logs are easier to compare this way.
    return sorted(r["index"] for r in candidates[:tp])


def cleanup(port: int, name: str) -> None:
    pat = f"sglang.launch_server.*{port}|{re.escape(name)}"
    subprocess.run(f"pkill -TERM -f '{pat}'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)
    subprocess.run(f"pkill -KILL -f '{pat}'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)


def find_server_pids(port: int) -> list[int]:
    text = shell(f"pgrep -af 'sglang.launch_server.*{port}' || true", timeout=5)
    out = []
    for line in text.splitlines():
        if "sglang.launch_server" in line:
            try:
                out.append(int(line.split(None, 1)[0]))
            except (IndexError, ValueError):
                pass
    return out


def cgroup_path_for_pid(pid: int) -> Path | None:
    try:
        rel = Path(f"/proc/{pid}/cgroup").read_text().splitlines()[-1].split(":", 2)[-1].lstrip("/")
    except OSError:
        return None
    path = Path("/sys/fs/cgroup") / rel
    return path if path.exists() else None


def read_memory_snapshot(cg: Path | None) -> dict[str, Any]:
    if cg is None or not cg.exists():
        return {}
    out: dict[str, Any] = {"cgroup": str(cg)}
    for name in ("memory.current", "memory.peak", "memory.max", "memory.swap.max"):
        p = cg / name
        if p.exists():
            raw = p.read_text().strip()
            out[name.replace(".", "_")] = int(raw) if raw.isdigit() else raw
    stat = cg / "memory.stat"
    if stat.exists():
        for line in stat.read_text().splitlines():
            k, v = line.split()
            out[k] = int(v)
    events = cg / "memory.events"
    if events.exists():
        for line in events.read_text().splitlines():
            k, v = line.split()
            out[f"event_{k}"] = int(v)
    return out


class MemorySampler:
    def __init__(self, port: int, path: Path):
        self.port = port
        self.path = path
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.cgroup: Path | None = None

    def __enter__(self) -> "MemorySampler":
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
                    pids = find_server_pids(self.port)
                    if pids:
                        self.cgroup = cgroup_path_for_pid(pids[0])
                row = read_memory_snapshot(self.cgroup)
                row["t"] = time.time()
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                time.sleep(1)


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
    cur = int(peak.get("memory_current", 0) or 0)
    out: dict[str, Any] = {"sample_count": len(rows), "peak_sample": peak}
    if cur:
        out["peak_gib"] = cur / 1024**3
        for key in ("anon", "file", "file_mapped", "active_file", "inactive_file", "slab", "kernel", "pagetables"):
            if key in peak:
                out[f"peak_{key}_gib"] = int(peak[key]) / 1024**3
                out[f"peak_{key}_pct_current"] = int(peak[key]) / cur
    out["oom_kill_events_max"] = max(int(r.get("event_oom_kill", 0) or 0) for r in rows)
    out["oom_events_max"] = max(int(r.get("event_oom", 0) or 0) for r in rows)
    return out


def parse_runtime_log(path: Path) -> dict[str, Any]:
    text = read_text(path)
    prefill = []
    decode = []
    cuda_graph_true = 0
    cuda_graph_false = 0
    for line in text.splitlines():
        if "cuda graph: True" in line or "Cuda graph captured" in line:
            cuda_graph_true += 1
        if "cuda graph: False" in line or "disable_cuda_graph=True" in line:
            cuda_graph_false += 1
        m = PREFILL_RE.search(line)
        if m:
            prefill.append((int(m.group("tokens")), float(m.group("tps"))))
        m = DECODE_RE.search(line)
        if m:
            decode.append(float(m.group("tps")))
    pairs = [(t, s) for t, s in prefill if t > 0 and s > 0]
    prefill_tps = sum(t for t, _ in pairs) / sum(t / s for t, s in pairs) if pairs else None
    return {
        "prefill_event_count": len(prefill),
        "decode_event_count": len(decode),
        "prefill_tps_hmean": prefill_tps,
        "decode_tps_avg_log": sum(decode) / len(decode) if decode else None,
        "decode_tps_min_log": min(decode) if decode else None,
        "decode_tps_max_log": max(decode) if decode else None,
        "cuda_graph_true_events": cuda_graph_true,
        "cuda_graph_false_events": cuda_graph_false,
    }


def wait_ready(port: int, log_path: Path, timeout_s: int) -> tuple[str, str]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
            if r.status_code == 200:
                return "ready", ""
        except Exception:
            pass
        tail = read_text(log_path, max_bytes=80000)
        if any(marker in tail for marker in FAIL_MARKERS):
            return "fail", tail[-16000:]
        time.sleep(3)
    return "timeout", read_text(log_path, max_bytes=16000)


def bench_one(port: int, model_name: str, prompt: dict[str, str], max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "messages": [{"role": "user", "content": prompt["input_prompt"]}],
    }
    t0 = time.time()
    first = None
    usage = None
    chunks = 0
    output_parts: list[str] = []
    with requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=max(1800, max_tokens * 25),
    ) as resp:
        if resp.status_code != 200:
            return {
                "status": "http_error",
                "id": prompt["id"],
                "domain": prompt["domain"],
                "code": resp.status_code,
                "body": resp.text[:4000],
            }
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            data = raw[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    output_parts.append(content)
                    chunks += 1
                    if first is None:
                        first = time.time()
    t1 = time.time()
    prompt_tokens = int((usage or {}).get("prompt_tokens", 0) or 0)
    completion_tokens = int((usage or {}).get("completion_tokens", 0) or 0)
    ttft = None if first is None else first - t0
    decode_s = None if first is None else max(t1 - first, 1e-9)
    output = "".join(output_parts)
    return {
        "status": "ok",
        "id": prompt["id"],
        "domain": prompt["domain"],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "decode_s": decode_s,
        "total_s": t1 - t0,
        "api_prefill_tok_s": prompt_tokens / ttft if ttft and prompt_tokens else None,
        "api_decode_tok_s": completion_tokens / decode_s if decode_s and completion_tokens else None,
        "chunks": chunks,
        "output_prefix": output[:240],
        "output_suffix": output[-240:],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    dec = [float(r["api_decode_tok_s"]) for r in ok if r.get("api_decode_tok_s")]
    pre = [float(r["api_prefill_tok_s"]) for r in ok if r.get("api_prefill_tok_s")]
    return {
        "ok_count": len(ok),
        "decode_tok_s_avg_api": sum(dec) / len(dec) if dec else None,
        "decode_tok_s_min_api": min(dec) if dec else None,
        "decode_tok_s_max_api": max(dec) if dec else None,
        "prefill_tok_s_avg_api": sum(pre) / len(pre) if pre else None,
        "completion_tokens_total": sum(int(r.get("completion_tokens", 0) or 0) for r in ok),
        "prompt_tokens_total": sum(int(r.get("prompt_tokens", 0) or 0) for r in ok),
    }


def build_command(
    cfg: dict[str, Any],
    port: int,
    name: str,
    memory_gb: int,
    tp: int,
    mem_fraction_static: str,
) -> list[str]:
    cmd = [
        "systemd-run",
        "--user",
        "--scope",
        "-p",
        f"MemoryMax={memory_gb}G",
        "-p",
        "MemorySwapMax=0",
        SERVER_PY,
        "-m",
        "sglang.launch_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        cfg["model_path"],
        "--kt-weight-path",
        cfg["kt_weight_path"],
        "--kt-cpuinfer",
        str(cfg["kt_cpuinfer"]),
        "--kt-threadpool-count",
        "2",
        "--kt-num-gpu-experts",
        str(cfg["ge"]),
        "--kt-max-deferred-experts-per-token",
        "3",
        "--kt-method",
        cfg["kt_method"],
        "--attention-backend",
        cfg["attention_backend"],
        "--trust-remote-code",
        "--mem-fraction-static",
        mem_fraction_static,
        "--chunked-prefill-size",
        str(cfg["chunked_prefill_size"]),
        "--max-running-requests",
        "1",
        "--max-total-tokens",
        "4096",
        "--watchdog-timeout",
        "3000",
        "--enable-mixed-chunk",
        "--tensor-parallel-size",
        str(tp),
        "--enable-p2p-check",
        "--served-model-name",
        name,
        "--disable-shared-experts-fusion",
        "--skip-server-warmup",
    ]
    if cfg["attention_backend"] == "triton":
        # 35B legacy rows did not use P2P check. Removing it keeps parity with existing v2 rows.
        i = cmd.index("--enable-p2p-check")
        del cmd[i]
    return cmd


def run_one(args: argparse.Namespace, cfg: dict[str, Any], memory_gb: int, prompts: list[dict[str, str]]) -> dict[str, Any]:
    label = f"{args.model}_{args.precision}_mmap_tp{args.tp}_mem{memory_gb}"
    run_dir = Path(args.out_root) / label
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "server.log"
    mem_path = run_dir / "memory_samples.jsonl"
    rows_path = run_dir / "rows.partial.json"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    gpus = choose_gpus(args.tp) if args.cuda == "auto" else [int(x) for x in args.cuda.split(",") if x.strip()]
    if len(gpus) != args.tp:
        raise ValueError(f"CUDA device count {len(gpus)} does not match TP {args.tp}: {gpus}")
    cuda = ",".join(str(x) for x in gpus)
    name = f"mmap-v2-{args.model}-{args.precision}-tp{args.tp}-mem{memory_gb}-{time.strftime('%Y%m%d_%H%M%S')}"
    cleanup(args.port, name)
    for p in (log_path, mem_path, rows_path):
        if p.exists():
            p.unlink()

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{Path(SERVER_PY).parent}:" + env.get("PATH", ""),
            "PYTHONPATH": f"{COMPAT_SGLANG}:{OLD_KT_LIB}" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
            "TMPDIR": "/mnt/data3/work/tmp",
            "HF_HOME": "/mnt/data3/work/hf_home",
            "TRANSFORMERS_CACHE": "/mnt/data3/work/hf_cache",
            "TRITON_CACHE_DIR": "/mnt/data3/work/triton_cache",
            "XDG_CACHE_HOME": "/mnt/data3/work/xdg_cache",
            "CUDA_VISIBLE_DEVICES": cuda,
            "KT_LANG": "en",
            "SGLANG_DISABLE_CUDNN_CHECK": "1",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    for key in list(env):
        if key.startswith("KT_") and key != "KT_LANG":
            del env[key]

    cmd = build_command(cfg, args.port, name, memory_gb, args.tp, args.mem_fraction_static or cfg["mem_fraction_static"])
    config = {
        "label": label,
        "mode": "mmap",
        "code_version_policy": "legacy_old_checkout",
        "model": cfg["model_name"],
        "model_size": args.model,
        "precision": cfg["kt_method"],
        "model_path": cfg["model_path"],
        "kt_weight_path": cfg["kt_weight_path"],
        "runner": str(Path(__file__)),
        "cuda": cuda,
        "cgroup": f"{memory_gb}G",
        "memory_max": f"{memory_gb}G",
        "tp": args.tp,
        "ge": cfg["ge"],
        "defer": 3,
        "mesh_cap": None,
        "mesh_global_pool_capacity": None,
        "mesh_prefill_layer_window": None,
        "mem_fraction_static": args.mem_fraction_static or cfg["mem_fraction_static"],
        "max_new_tokens": args.max_new_tokens,
        "prompt_file": args.prompt_file,
        "expected_prompt_count": len(prompts),
        "port": args.port,
        "name": name,
        "cmd": cmd,
        "env_subset": {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "TMPDIR", "HF_HOME", "TRANSFORMERS_CACHE", "TRITON_CACHE_DIR", "XDG_CACHE_HOME", "KT_LANG")},
        "server_log": str(log_path),
        "memory_samples": str(mem_path),
        "started_at": time.strftime("%Y%m%d_%H%M%S"),
        "gpu_before": gpu_snapshot(),
    }
    write_json(config_path, config)

    start = time.time()
    proc: subprocess.Popen | None = None
    entry: dict[str, Any] = dict(config)
    with log_path.open("w", encoding="utf-8", errors="replace") as logf, MemorySampler(args.port, mem_path):
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
        status, fail_tail = wait_ready(args.port, log_path, args.ready_timeout_s)
        entry["status"] = status
        entry["ready"] = status
        entry["ready_after_s"] = time.time() - start
        entry["launcher_pid"] = proc.pid
        if status == "ready":
            rows = []
            for prompt in prompts:
                try:
                    row = bench_one(args.port, name, prompt, args.max_new_tokens)
                except Exception as exc:
                    row = {"status": "exception", "id": prompt["id"], "domain": prompt["domain"], "error": repr(exc)}
                rows.append(row)
                write_json(rows_path, rows)
            entry["rows"] = rows
            entry["row_summary"] = summarize_rows(rows)
        else:
            entry["fail_tail"] = fail_tail
        time.sleep(2)

    entry["elapsed_s"] = time.time() - start
    entry["runtime_log_summary"] = parse_runtime_log(log_path)
    entry["memory_summary"] = parse_memory_samples(mem_path)
    entry["gpu_after"] = gpu_snapshot()
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    cleanup(args.port, name)
    write_json(summary_path, entry)
    print(json.dumps({"event": "run_done", "label": label, "status": entry["status"], "summary": str(summary_path)}, ensure_ascii=False), flush=True)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["35b", "397b"], required=True)
    parser.add_argument("--precision", choices=["amxint4", "bf16"], required=True)
    parser.add_argument("--tp", type=int, choices=[2, 4], required=True)
    parser.add_argument("--budgets", nargs="+", type=int, required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--port", type=int, default=31880)
    parser.add_argument("--cuda", default="auto")
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--ready-timeout-s", type=int, default=2400)
    parser.add_argument("--mem-fraction-static", default=None)
    parser.add_argument("--stop-after-ready", action="store_true")
    args = parser.parse_args()

    cfg = MODEL_MATRIX[(args.model, args.precision)]
    prompts = load_prompts(Path(args.prompt_file))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for memory_gb in args.budgets:
        result = run_one(args, cfg, memory_gb, prompts)
        results.append(result)
        if args.stop_after_ready and result.get("status") == "ready" and result.get("row_summary", {}).get("ok_count") == len(prompts):
            break

    write_json(
        out_root / f"aggregate_{args.model}_{args.precision}_tp{args.tp}.json",
        {
            "model": args.model,
            "precision": args.precision,
            "tp": args.tp,
            "budgets": args.budgets,
            "prompt_file": args.prompt_file,
            "max_new_tokens": args.max_new_tokens,
            "results": results,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
