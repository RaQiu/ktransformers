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


OFFICIAL_REPO_URL = "https://github.com/RaQiu/ktransformers.git"
OFFICIAL_COMMIT = "97c5dcac8863dc28fe28e8f5d389c45be090dee8"
OFFICIAL_SGLANG_COMMIT = "43ed1ec77a7fc37e6eeedbf9191e23abd418ae85"
OFFICIAL_WORK_ROOT = "/mnt/data2/tmp/qujing_full_v3"
OFFICIAL_ROOT = f"{OFFICIAL_WORK_ROOT}/ktransformers_raqiu_official_full_v3_clean_20260605"
OFFICIAL_SGLANG_PY = f"{OFFICIAL_ROOT}/third_party/sglang/python"
OFFICIAL_KT_LIB = f"{OFFICIAL_ROOT}/kt-kernel/build/lib.linux-x86_64-cpython-310"
OFFICIAL_KT_SRC = f"{OFFICIAL_ROOT}/kt-kernel"
OFFICIAL_KT_PY = f"{OFFICIAL_ROOT}/kt-kernel/python"
OFFICIAL_VENV_ROOT = f"{OFFICIAL_WORK_ROOT}/venvs/ktransformers-official-full-v3-20260605"
OFFICIAL_VENV = f"{OFFICIAL_VENV_ROOT}/bin"

PROMPT_FILE = Path("/mnt/data3/work/mesh_standard_5domain_prompts_20260602.json")
DEFAULT_OUT_ROOT = Path("/mnt/data3/work/mesh_paper_35b_runs/full_v3_official_20260605")

MODEL_SPECS: dict[str, dict[str, str]] = {
    "BF16": {
        "model_path": "/mnt/data3/models/Qwen3.5-35B-A3B",
        "kt_weight_path": "/mnt/data3/models/Qwen3.5-35B-A3B",
        "kt_method": "BF16",
    },
    "AMXINT4": {
        "model_path": "/mnt/data3/models/Qwen3.5-35B-A3B",
        "kt_weight_path": "/mnt/data2/models/Qwen3.5-35B-A3B-AMXINT4-NUMA2-MESH",
        "kt_method": "AMXINT4",
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
    "invalid device ordinal",
    "io_uring direct I/O requires",
    "BF16 lazy-pack promotion failed",
    "[BF16_PROMOTION_FAIL]",
)

PREFILL_RE = re.compile(
    r"Prefill batch, .*?#new-token: (?P<tokens>\d+), .*?input throughput \(token/s\): (?P<tps>[0-9.]+)"
)
DECODE_RE = re.compile(r"Decode batch, .*?gen throughput \(token/s\): (?P<tps>[0-9.]+)")


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


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


def shell(cmd: str, timeout: float | None = None) -> str:
    return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=timeout)


def load_prompt_bundle(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_prompts = data["prompts"] if isinstance(data, dict) and "prompts" in data else data
    prompts: list[dict[str, str]] = []
    for i, item in enumerate(raw_prompts):
        if isinstance(item, str):
            prompt = item
            prompt_id = f"prompt_{i + 1}"
            domain = ""
        else:
            prompt = str(item.get("input_prompt") or item.get("prompt") or item.get("content") or item.get("text") or "")
            prompt_id = str(item.get("id") or f"prompt_{i + 1}")
            domain = str(item.get("domain") or "")
        if not prompt:
            raise ValueError(f"empty prompt at index {i}")
        prompts.append({"id": prompt_id, "domain": domain, "input_prompt": prompt})
    if len(prompts) != 5:
        raise ValueError(f"expected 5 standard prompts, got {len(prompts)}")
    short = [(p["id"], len(p["input_prompt"].split())) for p in prompts if len(p["input_prompt"].split()) < 100]
    if short:
        raise ValueError(f"standard prompt shorter than 100 words: {short}")
    return prompts, data if isinstance(data, dict) else {"prompts": prompts}


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


def choose_gpus(tp: int, min_free_mib: int, wait_s: int) -> str:
    deadline = time.time() + wait_s
    while True:
        rows = gpu_snapshot()
        free = [r for r in rows if r["memory_free_mib"] >= min_free_mib and r["util_gpu_pct"] <= 20]
        free.sort(key=lambda r: r["index"])
        if len(free) >= tp:
            return ",".join(str(r["index"]) for r in free[:tp])
        if time.time() > deadline:
            raise RuntimeError(f"need {tp} GPUs with >= {min_free_mib} MiB free; last snapshot={rows}")
        print(json.dumps({"event": "wait_gpu", "tp": tp, "snapshot": rows, "time": time.time()}), flush=True)
        time.sleep(60)


def pids_for_port(port: int) -> list[int]:
    text = shell(f"pgrep -af 'sglang.launch_server.*{port}' || true", timeout=5)
    pids: list[int] = []
    for line in text.splitlines():
        if "sglang.launch_server" not in line:
            continue
        try:
            pids.append(int(line.split(None, 1)[0]))
        except (ValueError, IndexError):
            pass
    return pids


def pids_for_name_or_port(name: str, port: int) -> list[int]:
    pat = f"sglang.launch_server.*{port}|{re.escape(name)}"
    text = shell(f"pgrep -af '{pat}' || true", timeout=5)
    pids: list[int] = []
    me = os.getpid()
    for line in text.splitlines():
        try:
            pid = int(line.split(None, 1)[0])
        except (ValueError, IndexError):
            continue
        if pid != me:
            pids.append(pid)
    return pids


def cleanup(name: str, port: int) -> None:
    pids = pids_for_name_or_port(name, port)
    if pids:
        subprocess.run(["kill", "-TERM", *map(str, pids)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(4)
    pids = pids_for_name_or_port(name, port)
    if pids:
        subprocess.run(["kill", "-KILL", *map(str, pids)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)


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
            raw = p.read_text().strip()
            out[name.replace(".", "_")] = int(raw) if raw.isdigit() else raw
    stat = cg / "memory.stat"
    if stat.exists():
        for line in stat.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                out[parts[0]] = int(parts[1])
    events = cg / "memory.events"
    if events.exists():
        for line in events.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                out[f"event_{parts[0]}"] = int(parts[1])
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
                row = read_memory_snapshot(self.cgroup)
                row["t"] = time.time()
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
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
        "peak_sample": peak,
        "last_sample": last,
        "oom_kill_events_max": max(int(r.get("event_oom_kill", 0) or 0) for r in rows),
    }
    if cur:
        out["peak_gib"] = cur / 1024**3
        if "memory_peak" in peak:
            out["memory_peak_gib_from_cgroup"] = int(peak["memory_peak"]) / 1024**3
        for key in ("anon", "file", "file_mapped", "active_file", "inactive_file", "slab", "kernel"):
            if key in peak:
                out[f"peak_{key}_gib"] = int(peak[key]) / 1024**3
                out[f"peak_{key}_pct_current"] = int(peak[key]) / cur
    return out


def parse_runtime_log(path: Path) -> dict[str, Any]:
    text = read_text(path)
    prefill: list[tuple[int, float]] = []
    decode: list[float] = []
    for line in text.splitlines():
        m = PREFILL_RE.search(line)
        if m:
            prefill.append((int(m.group("tokens")), float(m.group("tps"))))
        m = DECODE_RE.search(line)
        if m:
            decode.append(float(m.group("tps")))
    prefill_tps = None
    valid_prefill = [(t, s) for t, s in prefill if t > 0 and s > 0]
    if valid_prefill:
        prefill_tps = sum(t for t, _ in valid_prefill) / sum(t / s for t, s in valid_prefill)
    return {
        "prefill_event_count": len(prefill),
        "decode_event_count": len(decode),
        "prefill_tps_hmean": prefill_tps,
        "decode_tps_avg_log": sum(decode) / len(decode) if decode else None,
        "decode_tps_min_log": min(decode) if decode else None,
        "decode_tps_max_log": max(decode) if decode else None,
    }


def parse_expert_stats(path: Path) -> dict[str, Any]:
    latest: dict[int, dict[str, Any]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                layer = obj.get("layer")
                if layer is not None:
                    latest[int(layer)] = obj
    if not latest:
        return {"exists": path.exists(), "layers": 0}
    scalar_keys = (
        "hit_count",
        "miss_count",
        "cold_miss_count",
        "in_flight_miss_count",
        "promote_count",
        "demote_count",
        "eviction_count",
        "total_access_count",
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


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    decode = [float(r["api_decode_tok_s"]) for r in ok if r.get("api_decode_tok_s")]
    prefill = [float(r["api_prefill_tok_s"]) for r in ok if r.get("api_prefill_tok_s")]
    total = [float(r["api_total_tok_s"]) for r in ok if r.get("api_total_tok_s")]
    return {
        "ok_count": len(ok),
        "decode_tok_s_avg_api": sum(decode) / len(decode) if decode else None,
        "decode_tok_s_min_api": min(decode) if decode else None,
        "decode_tok_s_max_api": max(decode) if decode else None,
        "prefill_tok_s_avg_api": sum(prefill) / len(prefill) if prefill else None,
        "total_tok_s_avg_api": sum(total) / len(total) if total else None,
        "completion_tokens_total": sum(int(r.get("completion_tokens", 0) or 0) for r in ok),
        "prompt_tokens_total": sum(int(r.get("prompt_tokens", 0) or 0) for r in ok),
    }


def wait_ready(port: int, log_path: Path, timeout_s: int, health_timeout_s: int) -> tuple[str, str]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        tail = read_text(log_path, max_bytes=80000)
        if "The server is fired up and ready to roll!" in tail:
            return "ready", ""
        if any(marker in tail for marker in FAIL_MARKERS):
            return "fail", tail[-16000:]
        if log_path.exists() and "Running as unit" in tail and not pids_for_port(port):
            return "dead", tail[-16000:]
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=health_timeout_s)
            if r.status_code == 200:
                return "ready", ""
        except Exception:
            pass
        time.sleep(5)
    return "timeout", read_text(log_path, max_bytes=16000)


def _base_row(prompt_obj: dict[str, str]) -> dict[str, Any]:
    return {
        "prompt_id": prompt_obj["id"],
        "domain": prompt_obj["domain"],
        "prompt_words": len(prompt_obj["input_prompt"].split()),
    }


def _finish_nonstream_row(
    prompt_obj: dict[str, str],
    output: str,
    prompt_tokens: int,
    completion_tokens: int,
    elapsed_s: float,
    request_mode_effective: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        **_base_row(prompt_obj),
        "status": "ok" if completion_tokens > 0 and output else "bad_output",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_s": elapsed_s,
        "ttft_s": None,
        "api_prefill_tok_s": None,
        "api_decode_tok_s": None,
        "api_total_tok_s": completion_tokens / elapsed_s if elapsed_s and completion_tokens else None,
        "chunks": 0,
        "output_chars": len(output),
        "output_preview": output[:600],
        "request_mode_effective": request_mode_effective,
    }
    if extra:
        row.update(extra)
    return row


def bench_one_chat_nonstream(
    port: int,
    model_name: str,
    prompt_obj: dict[str, str],
    max_tokens: int,
    request_timeout_s: int,
) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "messages": [{"role": "user", "content": prompt_obj["input_prompt"]}],
    }
    t0 = time.time()
    try:
        resp = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json=payload,
            timeout=(10, request_timeout_s),
        )
        if resp.status_code != 200:
            return {
                **_base_row(prompt_obj),
                "status": "http_error",
                "request_mode_effective": "chat-nonstream",
                "code": resp.status_code,
                "body": resp.text[:4000],
            }
        obj = resp.json()
    except Exception as exc:
        return {
            **_base_row(prompt_obj),
            "status": "exception",
            "request_mode_effective": "chat-nonstream",
            "error": repr(exc),
        }
    t1 = time.time()
    usage = obj.get("usage") or {}
    parts: list[str] = []
    for choice in obj.get("choices") or []:
        msg = choice.get("message") or {}
        text = msg.get("content") or choice.get("text") or ""
        if text:
            parts.append(text)
    return _finish_nonstream_row(
        prompt_obj,
        "".join(parts),
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
        t1 - t0,
        "chat-nonstream",
        {"raw_finish_reason": (obj.get("choices") or [{}])[0].get("finish_reason")},
    )


def bench_one_generate(
    port: int,
    prompt_obj: dict[str, str],
    max_tokens: int,
    request_timeout_s: int,
) -> dict[str, Any]:
    payload = {
        "text": prompt_obj["input_prompt"],
        "stream": False,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    }
    t0 = time.time()
    try:
        resp = requests.post(
            f"http://127.0.0.1:{port}/generate",
            json=payload,
            timeout=(10, request_timeout_s),
        )
        if resp.status_code != 200:
            return {
                **_base_row(prompt_obj),
                "status": "http_error",
                "request_mode_effective": "generate",
                "code": resp.status_code,
                "body": resp.text[:4000],
            }
        obj = resp.json()
    except Exception as exc:
        return {
            **_base_row(prompt_obj),
            "status": "exception",
            "request_mode_effective": "generate",
            "error": repr(exc),
        }
    t1 = time.time()
    meta = obj.get("meta_info") or {}
    output = str(obj.get("text") or obj.get("output") or "")
    return _finish_nonstream_row(
        prompt_obj,
        output,
        int(meta.get("prompt_tokens") or 0),
        int(meta.get("completion_tokens") or 0),
        t1 - t0,
        "generate",
        {"raw_finish_reason": meta.get("finish_reason")},
    )


def bench_one_chat_stream(
    port: int,
    model_name: str,
    prompt_obj: dict[str, str],
    max_tokens: int,
    stream_read_timeout_s: int,
) -> dict[str, Any]:
    payload = {
        "model": model_name,
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
    output_parts: list[str] = []
    chunks = 0
    try:
        with requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=(10, stream_read_timeout_s),
        ) as resp:
            if resp.status_code != 200:
                return {
                    **_base_row(prompt_obj),
                    "status": "http_error",
                    "request_mode_effective": "chat-stream",
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
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    if text:
                        output_parts.append(text)
                        chunks += 1
                        if first is None:
                            first = time.time()
    except Exception as exc:
        return {
            **_base_row(prompt_obj),
            "status": "exception",
            "request_mode_effective": "chat-stream",
            "error": repr(exc),
        }
    t1 = time.time()
    output = "".join(output_parts)
    prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    completion_tokens = int((usage or {}).get("completion_tokens") or 0)
    ttft = first - t0 if first else None
    decode_window = t1 - first if first else None
    return {
        **_base_row(prompt_obj),
        "status": "ok" if completion_tokens > 0 and output else "bad_output",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_s": t1 - t0,
        "ttft_s": ttft,
        "api_prefill_tok_s": prompt_tokens / ttft if ttft and prompt_tokens else None,
        "api_decode_tok_s": completion_tokens / decode_window if decode_window and completion_tokens else None,
        "api_total_tok_s": completion_tokens / (t1 - t0) if t1 > t0 and completion_tokens else None,
        "chunks": chunks,
        "output_chars": len(output),
        "output_preview": output[:600],
        "request_mode_effective": "chat-stream",
    }


def bench_one(
    port: int,
    model_name: str,
    prompt_obj: dict[str, str],
    max_tokens: int,
    request_mode: str,
    request_timeout_s: int,
    stream_read_timeout_s: int,
) -> dict[str, Any]:
    if request_mode == "chat-stream":
        return bench_one_chat_stream(port, model_name, prompt_obj, max_tokens, stream_read_timeout_s)
    if request_mode == "chat-nonstream":
        return bench_one_chat_nonstream(port, model_name, prompt_obj, max_tokens, request_timeout_s)
    if request_mode == "generate":
        return bench_one_generate(port, prompt_obj, max_tokens, request_timeout_s)
    if request_mode != "auto":
        raise ValueError(f"unknown request mode {request_mode}")

    first = bench_one_chat_nonstream(port, model_name, prompt_obj, max_tokens, request_timeout_s)
    if first.get("status") == "ok":
        first["request_mode_requested"] = "auto"
        return first
    second = bench_one_generate(port, prompt_obj, max_tokens, request_timeout_s)
    second["request_mode_requested"] = "auto"
    second["fallback_from"] = first.get("request_mode_effective")
    second["fallback_reason"] = {
        "status": first.get("status"),
        "error": first.get("error"),
        "code": first.get("code"),
        "body": first.get("body"),
        "output_chars": first.get("output_chars"),
        "completion_tokens": first.get("completion_tokens"),
    }
    return second


def mode_policy(mode: str) -> str:
    if mode != "full-v3":
        raise ValueError(f"official runner only supports full-v3, got {mode}")
    return "official_clean_checkout"


def build_env(mode: str, precision: str, cuda: str, expert_stats_path: Path, args: argparse.Namespace) -> tuple[str, dict[str, str]]:
    if mode != "full-v3":
        raise ValueError(f"official runner only supports full-v3, got {mode}")
    env = os.environ.copy()
    env.update(
        {
            "TMPDIR": "/mnt/data3/work/tmp",
            "HF_HOME": "/mnt/data3/work/hf_home",
            "TRANSFORMERS_CACHE": "/mnt/data3/work/hf_cache",
            "CUDA_VISIBLE_DEVICES": cuda,
            "SGLANG_DISABLE_CUDNN_CHECK": "1",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "PYTHONNOUSERSITE": "1",
            "VIRTUAL_ENV": OFFICIAL_VENV_ROOT,
            "KT_LANG": "en",
        }
    )
    for key in list(env):
        if key.startswith("KT_") and key != "KT_LANG":
            del env[key]

    env["PATH"] = f"{OFFICIAL_VENV}:" + env.get("PATH", "")
    pythonpath = f"{OFFICIAL_SGLANG_PY}:{OFFICIAL_KT_PY}:{OFFICIAL_KT_LIB}:{OFFICIAL_KT_SRC}:{OFFICIAL_ROOT}"
    env["PYTHONPATH"] = pythonpath
    return f"{OFFICIAL_VENV}/python3", env


def build_command(
    run: dict[str, Any],
    py: str,
    name: str,
    port: int,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        "systemd-run",
        "--user",
        "--scope",
        "-p",
        f"MemoryMax={args.memory_max}",
        "-p",
        "MemorySwapMax=0",
        py,
        "-m",
        "sglang.launch_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        run["model_path"],
        "--kt-weight-path",
        run["kt_weight_path"],
        "--kt-cpuinfer",
        str(args.cpuinfer),
        "--kt-threadpool-count",
        str(args.threadpool_count),
        "--kt-num-gpu-experts",
        str(args.gpu_experts),
        "--kt-max-deferred-experts-per-token",
        str(args.defer),
        "--kt-method",
        run["kt_method"],
        "--attention-backend",
        args.attention_backend,
        "--trust-remote-code",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--max-running-requests",
        "1",
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--watchdog-timeout",
        "3000",
        "--enable-mixed-chunk",
        "--tensor-parallel-size",
        str(run["tp"]),
        "--served-model-name",
        name,
        "--disable-shared-experts-fusion",
        "--skip-server-warmup",
    ]
    cmd[cmd.index("--kt-method") : cmd.index("--kt-method")] = [
        "--kt-gpu-prefill-token-threshold",
        str(args.gpu_prefill_token_threshold),
    ]
    if not args.disable_dynamic_expert_update:
        cmd.insert(cmd.index("--attention-backend"), "--kt-enable-dynamic-expert-update")
    if args.enable_p2p_check:
        cmd.append("--enable-p2p-check")
    return cmd


def make_plan(args: argparse.Namespace, prompts: list[dict[str, str]]) -> list[dict[str, Any]]:
    modes = []
    for mode in args.modes:
        if mode in ("full", "fill", "full-v3"):
            modes.append("full-v3")
        else:
            raise ValueError(f"official runner only supports full-v3/full/fill aliases, got {mode}")
    plan = []
    for precision in args.precisions:
        if precision not in MODEL_SPECS:
            raise ValueError(f"unknown precision {precision}")
        for tp in args.tps:
            for mode in modes:
                spec = dict(MODEL_SPECS[precision])
                if precision == "BF16" and args.bf16_model_path:
                    spec["model_path"] = args.bf16_model_path
                    spec["kt_weight_path"] = args.bf16_kt_weight_path or args.bf16_model_path
                label = f"35b_{precision.lower()}_{mode}_tp{tp}"
                plan.append(
                    {
                        "label": label,
                        "model": "Qwen3.5-35B-A3B",
                        "precision": precision,
                        "mode": mode,
                        "code_version_policy": mode_policy(mode),
                        "official_repo_url": OFFICIAL_REPO_URL,
                        "official_commit": OFFICIAL_COMMIT,
                        "official_sglang_commit": OFFICIAL_SGLANG_COMMIT,
                        "official_checkout": OFFICIAL_ROOT,
                        "official_sglang_py": OFFICIAL_SGLANG_PY,
                        "official_kt_py": OFFICIAL_KT_PY,
                        "official_kt_lib": OFFICIAL_KT_LIB,
                        "official_venv_root": OFFICIAL_VENV_ROOT,
                        "official_python": f"{OFFICIAL_VENV}/python3",
                        "model_path": spec["model_path"],
                        "kt_weight_path": spec["kt_weight_path"],
                        "kt_method": spec["kt_method"],
                        "tp": int(tp),
                        "gpu_experts": args.gpu_experts,
                        "defer": args.defer,
                        "mesh_cap": None,
                        "mesh_global_pool_capacity": None,
                        "mesh_prefill_layer_window": None,
                        "mesh_policy": None,
                        "bf16_expert_cache": None,
                        "bf16_expert_cache_dir": None,
                        "memory_max": args.memory_max,
                        "mem_fraction_static": args.mem_fraction_static,
                        "request_mode": args.request_mode,
                        "prompt_file": str(args.prompts_file),
                        "expected_prompt_count": len(prompts),
                        "max_new_tokens": args.max_tokens,
                    }
                )
    if args.only:
        keep = set(args.only)
        plan = [r for r in plan if r["label"] in keep]
    return plan


def write_matrix_markdown(out_root: Path, plan: list[dict[str, Any]]) -> None:
    lines = [
        "# 35B official full-v3 benchmark matrix",
        "",
        "Policy: official KTransformers checkout default path. No current-tree FULL backend env is set.",
        "",
        f"Official repo: `{OFFICIAL_REPO_URL}`",
        f"Official commit: `{OFFICIAL_COMMIT}`",
        f"Official SGLang commit: `{OFFICIAL_SGLANG_COMMIT}`",
        f"Official checkout: `{OFFICIAL_ROOT}`",
        f"Official venv: `{OFFICIAL_VENV_ROOT}`",
        "",
        "| model | precision | mode | code_version_policy | model_path | kt_weight_path | runner | cuda | cgroup | TP | GE | defer | mesh_cap | pool_cap | prefill_window | mem_fraction_static | prompt_file | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in plan:
        runner = "official checkout via venv"
        lines.append(
            "| {model} | {precision} | {mode} | {policy} | {mp} | {wp} | {runner} | assigned at runtime | {mem} | {tp} | {ge} | {defer} | {cap} | {pool} | {window} | {mf} | {pf} | pending |".format(
                model=r["model"],
                precision=r["precision"],
                mode=r["mode"],
                policy=r["code_version_policy"],
                mp=r["model_path"],
                wp=r["kt_weight_path"],
                runner=runner,
                mem=r["memory_max"],
                tp=r["tp"],
                ge=r["gpu_experts"],
                defer=r["defer"],
                cap="" if r["mesh_cap"] is None else r["mesh_cap"],
                pool="" if r["mesh_global_pool_capacity"] is None else r["mesh_global_pool_capacity"],
                window="" if r["mesh_prefill_layer_window"] is None else r["mesh_prefill_layer_window"],
                mf=r["mem_fraction_static"],
                pf=r["prompt_file"],
            )
        )
    (out_root / "run_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_record(run_dir: Path, entry: dict[str, Any]) -> None:
    row = entry.get("row_summary") or {}
    mem = entry.get("memory_summary") or {}
    expert = entry.get("expert_summary") or {}
    lines = [
        f"# Run record: {entry.get('label')}",
        "",
        f"- status: {entry.get('status')}",
        f"- model: {entry.get('model')}",
        f"- precision: {entry.get('precision')}",
        f"- mode: {entry.get('mode')}",
        f"- code_version_policy: {entry.get('code_version_policy')}",
        f"- official_repo_url: {entry.get('official_repo_url')}",
        f"- official_commit: {entry.get('official_commit')}",
        f"- official_sglang_commit: {entry.get('official_sglang_commit')}",
        f"- official_checkout: {entry.get('official_checkout')}",
        f"- official_sglang_py: {entry.get('official_sglang_py')}",
        f"- official_kt_py: {entry.get('official_kt_py')}",
        f"- official_kt_lib: {entry.get('official_kt_lib')}",
        f"- official_venv_root: {entry.get('official_venv_root')}",
        f"- official_python: {entry.get('official_python')}",
        f"- model_path: {entry.get('model_path')}",
        f"- kt_weight_path: {entry.get('kt_weight_path')}",
        f"- prompt_file: {entry.get('prompt_file')}",
        f"- ok/expected: {row.get('ok_count')}/{entry.get('expected_prompt_count')}",
        f"- TP/GE/defer: {entry.get('tp')}/{entry.get('gpu_experts')}/{entry.get('defer')}",
        f"- CUDA_VISIBLE_DEVICES: {entry.get('cuda')}",
        f"- MemoryMax: {entry.get('memory_max')}",
        f"- peak_gib: {mem.get('peak_gib')}",
        f"- memory_peak_gib_from_cgroup: {mem.get('memory_peak_gib_from_cgroup')}",
        f"- anon_peak_gib: {mem.get('peak_anon_gib')}",
        f"- file_peak_gib: {mem.get('peak_file_gib')}",
        f"- file_mapped_peak_gib: {mem.get('peak_file_mapped_gib')}",
        f"- oom_kill_events_max: {mem.get('oom_kill_events_max')}",
        f"- api_decode_tok_s_avg: {row.get('decode_tok_s_avg_api')}",
        f"- api_prefill_tok_s_avg: {row.get('prefill_tok_s_avg_api')}",
        f"- api_total_tok_s_avg: {row.get('total_tok_s_avg_api')}",
        f"- hit_rate: {expert.get('hit_rate')}",
        f"- iouring_read_gib: {expert.get('iouring_read_gib')}",
        f"- request_mode: {entry.get('request_mode')}",
        f"- bf16_expert_cache: {entry.get('bf16_expert_cache')}",
        f"- bf16_expert_cache_dir: {entry.get('bf16_expert_cache_dir')}",
        "",
        "## Command",
        "",
        "```bash",
        " ".join(map(str, entry.get("cmd") or [])),
        "```",
    ]
    (run_dir / "run_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pitfalls(run_dir: Path, entry: dict[str, Any]) -> None:
    lines = [f"# Pitfalls: {entry.get('label')}", ""]
    if entry.get("status") == "ready":
        lines.append("No blocking pitfall observed for this run.")
    elif entry.get("status") == "partial":
        lines.append("Run reached server ready state, but not all prompts completed successfully.")
    else:
        lines.append(f"Run did not produce a complete performance row. status={entry.get('status')}")
    if entry.get("fail_tail"):
        lines.extend(["", "## Failure Tail", "", "```text", str(entry["fail_tail"])[-12000:], "```"])
    (run_dir / "pitfalls.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def launch_one(run: dict[str, Any], out_root: Path, prompts: list[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    port = int(args.port)
    cuda = choose_gpus(int(run["tp"]), args.min_gpu_free_mib, args.wait_gpu_s)
    run_dir = out_root / run["label"]
    run_dir.mkdir(parents=True, exist_ok=True)
    name = f"official-full-v3-35b-{run['precision'].lower()}-tp{run['tp']}-{now_stamp()}"
    log_path = run_dir / "server.log"
    mem_path = run_dir / "memory_samples.jsonl"
    expert_stats_path = run_dir / "expert_stats.jsonl"
    for p in (log_path, mem_path, expert_stats_path):
        if p.exists():
            p.unlink()

    py, env = build_env(run["mode"], run["precision"], cuda, expert_stats_path, args)
    cmd = build_command(run, py, name, port, args)
    cleanup(name, port)

    entry: dict[str, Any] = {
        **run,
        "name": name,
        "port": port,
        "cuda": cuda,
        "run_dir": str(run_dir),
        "server_log": str(log_path),
        "memory_samples": str(mem_path),
        "expert_stats": str(expert_stats_path),
        "cmd": cmd,
        "env_subset": {
            k: env[k]
            for k in sorted(env)
            if k.startswith("KT_") or k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "PATH", "TMPDIR", "HF_HOME", "TRANSFORMERS_CACHE")
        },
        "gpu_before": gpu_snapshot(),
        "started_at": now_stamp(),
    }
    write_json(run_dir / "config.json", entry)

    start = time.time()
    logf = log_path.open("w", encoding="utf-8", errors="replace")
    try:
        with MemorySampler(port, mem_path):
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
            entry["launcher_pid"] = proc.pid
            status, fail_tail = wait_ready(port, log_path, args.ready_timeout_s, args.health_timeout_s)
            entry["ready_status"] = status
            entry["ready_after_s"] = time.time() - start
            print(
                json.dumps(
                    {
                        "event": "server_ready_check",
                        "label": run["label"],
                        "ready_status": status,
                        "ready_after_s": entry["ready_after_s"],
                        "time": time.time(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if status == "ready":
                rows = []
                for prompt in prompts:
                    print(
                        json.dumps(
                            {
                                "event": "prompt_start",
                                "label": run["label"],
                                "prompt_id": prompt["id"],
                                "request_mode": args.request_mode,
                                "time": time.time(),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    row = bench_one(
                        port,
                        name,
                        prompt,
                        args.max_tokens,
                        args.request_mode,
                        args.request_timeout_s,
                        args.stream_read_timeout_s,
                    )
                    rows.append(row)
                    write_json(run_dir / "rows.partial.json", rows)
                    print(
                        json.dumps(
                            {
                                "event": "prompt_done",
                                "label": run["label"],
                                "prompt_id": prompt["id"],
                                "status": row.get("status"),
                                "request_mode_effective": row.get("request_mode_effective"),
                                "completion_tokens": row.get("completion_tokens"),
                                "elapsed_s": row.get("elapsed_s"),
                                "time": time.time(),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                entry["rows"] = rows
                entry["row_summary"] = summarize_rows(rows)
                if entry["row_summary"]["ok_count"] == len(prompts):
                    entry["status"] = "ready"
                else:
                    entry["status"] = "partial"
                    entry["fail_tail"] = f"Only {entry['row_summary']['ok_count']}/{len(prompts)} prompts completed successfully"
            else:
                entry["status"] = status
                entry["fail_tail"] = fail_tail
            time.sleep(2)
    finally:
        logf.close()
        entry["elapsed_s"] = time.time() - start
        entry["runtime_log_summary"] = parse_runtime_log(log_path)
        entry["memory_summary"] = parse_memory_samples(mem_path)
        entry["expert_summary"] = parse_expert_stats(expert_stats_path)
        entry["gpu_after"] = gpu_snapshot()
        entry["finished_at"] = now_stamp()
        write_json(run_dir / "summary.json", entry)
        write_run_record(run_dir, entry)
        write_pitfalls(run_dir, entry)
        cleanup(name, port)

    print(
        json.dumps(
            {
                "event": "run_done",
                "label": run["label"],
                "status": entry["status"],
                "summary": str(run_dir / "summary.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return entry


def aggregate(out_root: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for r in results:
        row = r.get("row_summary") or {}
        mem = r.get("memory_summary") or {}
        expert = r.get("expert_summary") or {}
        rows.append(
            {
                "label": r.get("label"),
                "model": r.get("model"),
                "precision": r.get("precision"),
                "mode": r.get("mode"),
                "code_version_policy": r.get("code_version_policy"),
                "official_commit": r.get("official_commit"),
                "official_sglang_commit": r.get("official_sglang_commit"),
                "status": r.get("status"),
                "tp": r.get("tp"),
                "gpu_experts": r.get("gpu_experts"),
                "defer": r.get("defer"),
                "memory_max": r.get("memory_max"),
                "mesh_cap": r.get("mesh_cap"),
                "mesh_global_pool_capacity": r.get("mesh_global_pool_capacity"),
                "mesh_prefill_layer_window": r.get("mesh_prefill_layer_window"),
                "bf16_expert_cache": r.get("bf16_expert_cache"),
                "bf16_expert_cache_dir": r.get("bf16_expert_cache_dir"),
                "ok_count": row.get("ok_count"),
                "expected_prompt_count": r.get("expected_prompt_count"),
                "api_decode_tok_s": row.get("decode_tok_s_avg_api"),
                "api_prefill_tok_s": row.get("prefill_tok_s_avg_api"),
                "api_total_tok_s": row.get("total_tok_s_avg_api"),
                "log_decode_tok_s": (r.get("runtime_log_summary") or {}).get("decode_tps_avg_log"),
                "log_prefill_tok_s": (r.get("runtime_log_summary") or {}).get("prefill_tps_hmean"),
                "peak_gib": mem.get("peak_gib"),
                "memory_peak_gib_from_cgroup": mem.get("memory_peak_gib_from_cgroup"),
                "file_peak_gib": mem.get("peak_file_gib"),
                "file_mapped_peak_gib": mem.get("peak_file_mapped_gib"),
                "anon_peak_gib": mem.get("peak_anon_gib"),
                "oom_kill_events_max": mem.get("oom_kill_events_max"),
                "hit_rate": expert.get("hit_rate"),
                "iouring_read_gib": expert.get("iouring_read_gib"),
                "summary": str(Path(r.get("run_dir", "")) / "summary.json") if r.get("run_dir") else None,
                "request_mode": r.get("request_mode"),
            }
        )
    agg = {"generated_at": now_stamp(), "out_root": str(out_root), "runs": rows}
    write_json(out_root / "aggregate_summary.json", agg)

    md = [
        "# 35B official full-v3 aggregate",
        "",
        "Effective rows require status=ready and ok_count=expected_prompt_count.",
        "",
        f"Official repo: `{OFFICIAL_REPO_URL}`",
        f"Official commit: `{OFFICIAL_COMMIT}`",
        f"Official SGLang commit: `{OFFICIAL_SGLANG_COMMIT}`",
        f"Official checkout: `{OFFICIAL_ROOT}`",
        f"Official venv: `{OFFICIAL_VENV_ROOT}`",
        "",
        "| label | precision | mode | policy | request | TP | GE | defer | status | ok/expected | API decode tok/s | API total tok/s | log decode tok/s | log prefill tok/s | peak GiB | memory.peak GiB | anon GiB | file GiB | file_mapped GiB | OOM | summary |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        ok = "" if r["ok_count"] is None else f"{r['ok_count']}/{r['expected_prompt_count']}"
        md.append(
            "| {label} | {precision} | {mode} | {policy} | {request} | {tp} | {ge} | {defer} | {status} | {ok} | {dec} | {total} | {logdec} | {logpre} | {peak} | {mempeak} | {anon} | {file} | {mapped} | {oom} | {summary} |".format(
                label=r["label"],
                precision=r["precision"],
                mode=r["mode"],
                policy=r["code_version_policy"],
                request=r["request_mode"],
                tp=r["tp"],
                ge=r["gpu_experts"],
                defer=r["defer"],
                status=r["status"],
                ok=ok,
                dec=f"{r['api_decode_tok_s']:.3f}" if r["api_decode_tok_s"] is not None else "",
                total=f"{r['api_total_tok_s']:.3f}" if r["api_total_tok_s"] is not None else "",
                logdec=f"{r['log_decode_tok_s']:.3f}" if r["log_decode_tok_s"] is not None else "",
                logpre=f"{r['log_prefill_tok_s']:.3f}" if r["log_prefill_tok_s"] is not None else "",
                peak=f"{r['peak_gib']:.3f}" if r["peak_gib"] is not None else "",
                mempeak=f"{r['memory_peak_gib_from_cgroup']:.3f}" if r["memory_peak_gib_from_cgroup"] is not None else "",
                anon=f"{r['anon_peak_gib']:.3f}" if r["anon_peak_gib"] is not None else "",
                file=f"{r['file_peak_gib']:.3f}" if r["file_peak_gib"] is not None else "",
                mapped=f"{r['file_mapped_peak_gib']:.3f}" if r["file_mapped_peak_gib"] is not None else "",
                oom="" if r["oom_kill_events_max"] is None else r["oom_kill_events_max"],
                summary=r["summary"] or "",
            )
        )
    (out_root / "aggregate_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return agg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT / f"run_{now_stamp()}"))
    parser.add_argument("--prompts-file", type=Path, default=PROMPT_FILE)
    parser.add_argument("--precisions", nargs="+", default=["AMXINT4"])
    parser.add_argument("--modes", nargs="+", default=["full-v3"])
    parser.add_argument("--tps", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--bf16-model-path", default="")
    parser.add_argument("--bf16-kt-weight-path", default="")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--port", type=int, default=31850)
    parser.add_argument("--memory-max", default="768G")
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--ready-timeout-s", type=int, default=1800)
    parser.add_argument("--health-timeout-s", type=int, default=10)
    parser.add_argument("--request-mode", choices=["auto", "chat-nonstream", "chat-stream", "generate"], default="auto")
    parser.add_argument("--request-timeout-s", type=int, default=1800)
    parser.add_argument("--stream-read-timeout-s", type=int, default=120)
    parser.add_argument("--wait-gpu-s", type=int, default=7200)
    parser.add_argument("--min-gpu-free-mib", type=int, default=40000)
    parser.add_argument("--gpu-experts", type=int, default=32)
    parser.add_argument("--defer", type=int, default=3)
    parser.add_argument("--cpuinfer", type=int, default=96)
    parser.add_argument("--threadpool-count", type=int, default=2)
    parser.add_argument("--gpu-prefill-token-threshold", type=int, default=4096)
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--mem-fraction-static", default="0.85")
    parser.add_argument("--chunked-prefill-size", type=int, default=4096)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument("--disable-dynamic-expert-update", action="store_true")
    parser.add_argument("--enable-p2p-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompts, prompt_bundle = load_prompt_bundle(args.prompts_file)
    if not args.max_tokens:
        args.max_tokens = int((prompt_bundle.get("generation_defaults") or {}).get("max_new_tokens") or 384)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    plan = make_plan(args, prompts)
    write_json(out_root / "plan.json", plan)
    write_json(out_root / "prompt_bundle.json", prompt_bundle)
    write_matrix_markdown(out_root, plan)

    print(json.dumps({"event": "matrix_ready", "out_root": str(out_root), "runs": len(plan), "dry_run": args.dry_run}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0

    results: list[dict[str, Any]] = []
    for run in plan:
        print(json.dumps({"event": "run_start", "label": run["label"], "run": run, "time": time.time()}, ensure_ascii=False), flush=True)
        try:
            result = launch_one(run, out_root, prompts, args)
        except Exception as exc:
            result = {**run, "status": "runner_exception", "error": repr(exc), "run_dir": str(out_root / run["label"])}
            run_dir = out_root / run["label"]
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(run_dir / "summary.json", result)
            write_pitfalls(run_dir, {**result, "fail_tail": repr(exc)})
            print(json.dumps({"event": "run_exception", "label": run["label"], "error": repr(exc)}, ensure_ascii=False), flush=True)
        results.append(result)
        write_json(out_root / "partial_results.json", results)
        aggregate(out_root, results)
    aggregate(out_root, results)
    print(json.dumps({"event": "all_done", "out_root": str(out_root), "runs": len(results)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
