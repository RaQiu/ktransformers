#!/usr/bin/env python3
"""
KV Cache size calculator JSON output helper for KTransformers Electron frontend.
Usage: python kv_cache_json.py <model_path> <max_total_tokens> [tp] [dtype]
"""

import json
import sys


def calc_kv_cache(model_path: str, max_total_tokens: int, tp: int = 1, dtype: str = "auto") -> dict:
    try:
        from transformers import AutoConfig
        import os

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Extract model dimensions
        num_heads = getattr(config, 'num_key_value_heads', None) or getattr(config, 'num_attention_heads', 32)
        num_layers = getattr(config, 'num_hidden_layers', 32)
        head_dim = getattr(config, 'head_dim', None)
        if head_dim is None:
            hidden = getattr(config, 'hidden_size', 4096)
            num_attn = getattr(config, 'num_attention_heads', 32)
            head_dim = hidden // num_attn

        dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2,
                       "float8_e4m3fn": 1, "float8_e5m2": 1, "auto": 2}.get(dtype, 2)

        # KV cache: 2 (K+V) * num_layers * num_heads/tp * head_dim * max_tokens * dtype_bytes
        heads_per_tp = num_heads / tp
        kv_bytes = 2 * num_layers * heads_per_tp * head_dim * max_total_tokens * dtype_bytes
        kv_gb = kv_bytes / (1024 ** 3)

        return {
            "success": True,
            "kv_cache_gb": round(kv_gb, 3),
            "details": {
                "num_layers": num_layers,
                "num_kv_heads": num_heads,
                "head_dim": head_dim,
                "max_total_tokens": max_total_tokens,
                "tp": tp,
                "dtype": dtype,
                "dtype_bytes": dtype_bytes,
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"success": False, "error": "Usage: kv_cache_json.py <model_path> <max_tokens> [tp] [dtype]"}))
        sys.exit(1)

    model_path = sys.argv[1]
    max_tokens = int(sys.argv[2])
    tp = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    dtype = sys.argv[4] if len(sys.argv) > 4 else "auto"

    result = calc_kv_cache(model_path, max_tokens, tp, dtype)
    print(json.dumps(result))
