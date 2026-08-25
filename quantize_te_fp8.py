"""Quantize a bf16/fp16 text encoder to ComfyUI's fp8 "mixed" layout,
cloning the per-layer choices of a reference repack.

    python quantize_te_fp8.py --input  models/text_encoders/flux2-klein-9b-uncensored-text-encoder.safetensors \
                              --like   models/text_encoders/qwen_3_8b_fp8mixed.safetensors \
                              --output models/text_encoders/flux2-klein-9b-uncensored-text-encoder_fp8mixed.safetensors

Why: a full-precision Qwen3-8B encoder is ~16GB and, with Klein 9B fp8
resident, overflows a 32GB card (ComfyUI 0.32 keeps the TE resident after
encoding). The Comfy-Org `qwen_3_8b_fp8mixed` repack is ~8.7GB: every
attention projection and MOST MLP linears stored as float8_e4m3fn with a
per-tensor fp32 scale + a `comfy_quant` JSON descriptor (85 of them are
NVFP4 in the repack - emitted as fp8 here), while a chosen subset of MLP
linears (26/108 - presumably outlier-sensitive) and all norms/embeddings
stay bf16. This tool reproduces that layout exactly by
copying the reference file's per-tensor dtype decisions, so any same-
architecture encoder (e.g. an abliterated Qwen3-8B) drops into the same
VRAM budget. Tensors absent from the reference (lm_head - unused by
ComfyUI) are dropped.

Quantization: scale = amax / 448 (e4m3 max), w8 = (w / scale).to(e4m3).
Progress prints per tensor (rule: anything long shows progress).
"""
import argparse
import json
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
E4M3_MAX = 448.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="bf16/fp16 safetensors to quantize")
    ap.add_argument("--like", required=True, help="reference fp8mixed repack (same architecture)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ref = safe_open(args.like, "pt")
    plan = {}                       # key -> "fp8" | "keep"
    for k in ref.keys():
        if k.endswith((".weight_scale", ".weight_scale_2", ".comfy_quant")):
            continue
        # the repack is fp8 for most linears and NVFP4 (packed U8) for ~85 of
        # them; both become plain fp8 here (NVFP4 packing is not worth
        # reimplementing - fp8 is strictly higher precision, ~1GB larger)
        plan[k] = "fp8" if ref.get_slice(k).get_dtype() in ("F8_E4M3", "U8") else "keep"
    src = safe_open(args.input, "pt")
    src_keys = set(src.keys())
    missing = [k for k in plan if k not in src_keys]
    if missing:
        sys.exit(f"input lacks {len(missing)} tensors the reference has, e.g. {missing[:3]}")
    dropped = sorted(src_keys - set(plan))
    print(f"{len(plan)} tensors ({sum(v == 'fp8' for v in plan.values())} -> fp8, "
          f"{sum(v == 'keep' for v in plan.values())} kept bf16); dropping {dropped}")

    out, t0 = {}, time.time()
    for n, (k, mode) in enumerate(plan.items(), 1):
        w = src.get_tensor(k)
        if mode == "fp8":
            w = w.to(torch.float32)
            scale = (w.abs().amax() / E4M3_MAX).clamp(min=1e-12)
            out[k] = (w / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
            out[k[:-len(".weight")] + ".weight_scale"] = scale.to(torch.float32)
            out[k[:-len(".weight")] + ".comfy_quant"] = torch.tensor(
                list(json.dumps({"format": "float8_e4m3fn"}).encode("utf-8")), dtype=torch.uint8)
        else:
            out[k] = w.to(torch.bfloat16)
        print(f"\r  {n}/{len(plan)}  {k[:60]:<60} {int(time.time() - t0)}s\x1b[K", end="")
    print()
    print("writing", args.output)
    save_file(out, args.output)
    print(f"done in {int(time.time() - t0)}s")


if __name__ == "__main__":
    main()
