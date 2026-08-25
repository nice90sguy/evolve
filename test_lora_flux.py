"""Likeness testing for FLUX.2 Klein character LoRAs (A/B/C prompt probes).

Usage:  python test_lora_flux.py --lora <name|path> --prompt "..." [options]

Renders on the DISTILLED klein (flux-2-klein-9b-fp8, 4 steps, cfg 1.0) -
never on base; train-on-base -> render-on-distilled is the doctrine.
ComfyUI must be up (launch it yourself).

--lora: either a .safetensors path, or a name resolved to the NEWEST
version in .\\loras\\<name>\\ (CWD-relative - run from your project root).
Musubi-native files are auto-converted to ComfyUI format once (a sibling
*_comfy.safetensors, cached) via musubi's convert_lora.

Output follows the house convention: --out-dir (default .) as
<prefix>NNNNN.png, numbering continues, never overwrites; seed printed
for reuse (pin --seed to compare prompts on identical noise).
"""
import argparse
import random
import subprocess
import sys
from pathlib import Path

from pose_from_char import (SCRATCH_PREFIX, TEMPLATE, next_index, queue,
                            save_render, snap16, wait)
from char_lora_zimage import MUSUBI, MUSUBI_PY

import json

INFER_DIT = "flux-2-klein-9b-fp8.safetensors"     # DISTILLED - never base


def resolve_lora(spec):
    p = Path(spec)
    if p.suffix == ".safetensors":
        if not p.is_file():
            sys.exit(f"LoRA file not found: {p}")
        return p.resolve()
    d = Path("loras") / spec
    cands = sorted(f for f in d.glob(f"{spec}_v*.safetensors")
                   if not f.name.endswith("_comfy.safetensors"))
    if not cands:
        sys.exit(f"no LoRA versions found in {d.resolve()}")
    return cands[-1].resolve()


def ensure_comfy_format(native):
    """Convert musubi-native -> ComfyUI format once; cached beside it."""
    comfy = native.with_name(native.stem + "_comfy.safetensors")
    if comfy.exists():
        return comfy
    print(f"converting {native.name} -> ComfyUI format ...")
    r = subprocess.run(
        [str(MUSUBI_PY), str(MUSUBI / "src/musubi_tuner/convert_lora.py"),
         "--input", str(native), "--output", str(comfy), "--target", "other"],
        capture_output=True, text=True, cwd=str(MUSUBI))
    if r.returncode != 0 or not comfy.exists():
        sys.exit("LoRA conversion failed:\n" + (r.stdout or "")[-1500:]
                 + (r.stderr or "")[-1500:])
    fix_flux2_keys(comfy)
    return comfy


# musubi's "other" target names Flux2 double-block attention Flux.1-style
# (img_attn_qkv); ComfyUI's Flux2 Klein weights are img_attn.qkv. ComfyUI
# silently skips the mismatched keys ("lora key not loaded") - 64 of 224,
# ALL the double-block attention - found 2026-08-21. Rename in place.
FLUX2_KEY_FIX = {"img_attn_qkv": "img_attn.qkv", "img_attn_proj": "img_attn.proj",
                 "txt_attn_qkv": "txt_attn.qkv", "txt_attn_proj": "txt_attn.proj"}


def fix_flux2_keys(comfy):
    """Rewrite a converted LoRA so every key matches ComfyUI's Flux2 naming.
    Idempotent; keeps a .bak of the original the first time."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    with safe_open(str(comfy), "pt") as f:
        meta = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    fixed, n = {}, 0
    for k, v in tensors.items():
        nk = k
        for a, b in FLUX2_KEY_FIX.items():
            if "." + a + "." in nk:
                nk = nk.replace("." + a + ".", "." + b + ".")
                n += 1
        fixed[nk] = v
    if n:
        bak = comfy.with_suffix(comfy.suffix + ".bak")
        if not bak.exists():
            comfy.rename(bak)
        save_file(fixed, str(comfy), metadata=meta)
        print(f"fixed {n} Flux2 LoRA key names in {comfy.name}")
    return n


def build_graph(prompt, seed, width, height, batch, lora_path, strength):
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    p = payload["prompt"]
    for old in ("img_ref", "enc_ref", "pos_ref", "neg_ref", "ift"):
        p.pop(old, None)
    p["unet"]["inputs"]["unet_name"] = INFER_DIT
    p["lora"] = {"class_type": "ApplyTrainedLora",
                 "inputs": {"strength": strength, "model": ["unet", 0],
                            "lora_path": str(lora_path)}}
    p["guider"]["inputs"]["model"] = ["lora", 0]
    p["guider"]["inputs"]["positive"] = ["txt", 0]
    p["guider"]["inputs"]["negative"] = ["zero", 0]
    p["txt"]["inputs"]["text"] = prompt
    p["noise"]["inputs"]["noise_seed"] = seed
    p["sched"]["inputs"].update(width=width, height=height)
    p["latent"]["inputs"].update(width=width, height=height, batch_size=batch)
    p["save"]["inputs"]["filename_prefix"] = f"{SCRATCH_PREFIX}/loratest"
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", required=True,
                    help="LoRA name (newest version in .\\loras\\<name>) or path")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="pin this to compare prompts on identical noise")
    ap.add_argument("--num-images", type=int, default=1)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--out-prefix", default="img")
    args = ap.parse_args()

    lora = ensure_comfy_format(resolve_lora(args.lora))
    seed = args.seed if args.seed is not None else random.randint(1, 999_999_999_999)
    out_dir = Path(args.out_dir).expanduser().resolve()
    w, h = snap16(args.width), snap16(args.height)

    pid = queue(build_graph(args.prompt, seed, w, h, args.num_images,
                            lora, args.strength))
    print(f"queued {lora.name} -> {pid} (seed {seed}, strength {args.strength})")
    srcs = wait(pid, timeout=600)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = next_index(out_dir, args.out_prefix)
    for s in srcs:
        dest = save_render(s, out_dir, args.out_prefix, n)
        n += 1
        print(f"rendered -> {dest}")


if __name__ == "__main__":
    main()
