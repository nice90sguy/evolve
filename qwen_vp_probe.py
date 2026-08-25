"""Qwen-Image-Edit-2511 + Multiple-Angles LoRA viewpoint probe.

Capability probe, not a production tool: given ANY source image, re-shoot it
from a set of trained camera viewpoints so we can find where the LoRA's
envelope ends. Run it down a difficulty ladder (environment-coupled scene ->
plain-backdrop full body -> bust on plain grey) and the rung where results
start holding up IS the capability boundary.

Viewpoint grammar (fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA, trained on
3000+ Gaussian-splat orbit renders, 96 poses):

    <sks> <azimuth> <elevation> <distance>

`distance` scales the ORBIT RADIUS (x0.6 / x1.0 / x1.8) - it is a camera
move, not a crop. That is why this belongs in evolve as a BUDDING operator
(single parent, parent 0 carries continuity) rather than a model family.

The `[VP: right, low, medium]` tag form is the seed of the eventual evolve
integration: a closed 8x4x3 enum, so it validates cleanly and expands to the
exact trained token string.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from pose_from_char import COMFY, free_vram, queue, snap16, wait

HERE = Path(__file__).resolve().parent
SCRATCH_PREFIX = "akasutils_scratch"

# --- trained vocabulary -------------------------------------------------
# Keys are the short aliases accepted in a [VP: ...] tag; values are the
# exact strings the LoRA was trained on. Do not paraphrase the values.
AZIMUTH = {
    "front":       "front view",                  # 0
    "front-right": "front-right quarter view",    # 45
    "right":       "right side view",             # 90
    "back-right":  "back-right quarter view",     # 135
    "back":        "back view",                   # 180
    "back-left":   "back-left quarter view",      # 225
    "left":        "left side view",              # 270
    "front-left":  "front-left quarter view",     # 315
}
ELEVATION = {
    "low":      "low-angle shot",    # -30
    "eye":      "eye-level shot",    #   0
    "elevated": "elevated shot",     #  30
    "high":     "high-angle shot",   #  60
}
DISTANCE = {
    "close":  "close-up",      # x0.6
    "medium": "medium shot",   # x1.0
    "wide":   "wide shot",     # x1.8
}
ALIASES = {"fr": "front-right", "br": "back-right", "bl": "back-left",
           "fl": "front-left", "closeup": "close", "close-up": "close",
           "med": "medium", "eye-level": "eye"}

MODEL_GGUF = "qwen-image-edit-2511-Q5_0.gguf"
LORA_ANGLES = "qwen\\qwen-image-edit-2511-multiple-angles-lora.safetensors"
# 4-step distillation, stacked AFTER the angle LoRA. Roughly 5x faster, but
# it is a second intervention on the same weights - never introduce it in
# the same run as the question you are actually asking.
LORA_LIGHTNING = "qwen\\Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
CLIP_NAME = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE_NAME = "qwen_image_vae.safetensors"

# Qwen-Image-Edit-2511 works around 1 megapixel; anything much smaller is
# out of distribution and would confound a negative result with an input
# problem (ComfyUI_01376_ is 512x512 = 0.26MP).
MIN_MP = 0.9
TARGET_MP = 1.05


def parse_vp(spec):
    """'[VP: right, low, medium]' or 'right,low,medium' -> (prompt, tag).

    Missing fields default to eye-level / medium shot, so 'right' alone is a
    valid viewpoint.
    """
    body = spec.strip()
    m = re.fullmatch(r"\[\s*VP\s*:\s*(.*?)\s*\]", body, re.I)
    if m:
        body = m.group(1)
    parts = [p.strip().lower() for p in re.split(r"[,\s]+", body) if p.strip()]
    parts = [ALIASES.get(p, p) for p in parts]
    az, el, di = None, "eye", "medium"
    for p in parts:
        if p in AZIMUTH:
            az = p
        elif p in ELEVATION:
            el = p
        elif p in DISTANCE:
            di = p
        else:
            sys.exit(f"unknown viewpoint token {p!r} in {spec!r}\n"
                     f"  azimuth:   {', '.join(AZIMUTH)}\n"
                     f"  elevation: {', '.join(ELEVATION)}\n"
                     f"  distance:  {', '.join(DISTANCE)}")
    if az is None:
        sys.exit(f"no azimuth in {spec!r} (one of: {', '.join(AZIMUTH)})")
    prompt = f"<sks> {AZIMUTH[az]} {ELEVATION[el]} {DISTANCE[di]}"
    return prompt, f"{az}-{el}-{di}"


def stage(src):
    """Copy the source into ComfyUI's input dir, flattening alpha onto white
    and upscaling undersized inputs to ~1MP.

    Alpha must never reach LoadImage: it drops the channel, the model sees a
    cutout and paints checkerboard 'transparency' around the subject.
    """
    img = Image.open(src)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        flat = Image.new("RGB", img.size, (255, 255, 255))
        flat.paste(img, mask=img.split()[-1])
        img = flat
    else:
        img = img.convert("RGB")

    w, h = img.size
    mp = w * h / 1e6
    if mp < MIN_MP:
        s = (TARGET_MP / mp) ** 0.5
        w, h = snap16(round(w * s)), snap16(round(h * s))
        img = img.resize((w, h), Image.LANCZOS)
        print(f"  upscaled {src.name}: {mp:.2f}MP -> {w}x{h} ({w*h/1e6:.2f}MP)")
    elif w % 16 or h % 16:
        w, h = snap16(w), snap16(h)
        img = img.resize((w, h), Image.LANCZOS)
        print(f"  snapped {src.name} to /16: {w}x{h}")

    name = f"vp_src_{src.stem}.png"
    img.save(COMFY / "input" / name)
    return name, (w, h)


def build_graph(image_name, prompt, negative, seed, steps, cfg, strength,
                out_prefix, lightning=False):
    """Qwen-Image-Edit-2511 edit graph with the multi-angle LoRA on the model.

    The source image goes to BOTH text encoders (reference conditioning) and
    to VAEEncode (the latent, denoise 1.0) - so output dims follow the staged
    source and the viewpoint is the only variable across a sweep.
    """
    g = {
        "unet": {"class_type": "UnetLoaderGGUF",
                 "inputs": {"unet_name": MODEL_GGUF}},
        "gene": {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["unet", 0], "lora_name": LORA_ANGLES,
                            "strength_model": strength}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": CLIP_NAME, "type": "qwen_image"}},
        "vae":  {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "img":  {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "pos":  {"class_type": "TextEncodeQwenImageEditPlus",
                 "inputs": {"clip": ["clip", 0], "prompt": prompt,
                            "vae": ["vae", 0], "image1": ["img", 0]}},
        "neg":  {"class_type": "TextEncodeQwenImageEditPlus",
                 "inputs": {"clip": ["clip", 0], "prompt": negative,
                            "vae": ["vae", 0], "image1": ["img", 0]}},
        "lat":  {"class_type": "VAEEncode",
                 "inputs": {"pixels": ["img", 0], "vae": ["vae", 0]}},
        "samp": {"class_type": "KSampler",
                 "inputs": {"model": ["gene", 0], "seed": seed, "steps": steps,
                            "cfg": cfg, "sampler_name": "euler",
                            "scheduler": "simple", "positive": ["pos", 0],
                            "negative": ["neg", 0], "latent_image": ["lat", 0],
                            "denoise": 1.0}},
        "dec":  {"class_type": "VAEDecode",
                 "inputs": {"samples": ["samp", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["dec", 0],
                            "filename_prefix": f"{SCRATCH_PREFIX}/{out_prefix}"}},
    }
    if lightning:
        g["fast"] = {"class_type": "LoraLoaderModelOnly",
                     "inputs": {"model": ["gene", 0],
                                "lora_name": LORA_LIGHTNING,
                                "strength_model": 1.0}}
        g["samp"]["inputs"]["model"] = ["fast", 0]
    return g


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", action="append", required=True,
                    help="source image (repeatable) - the difficulty ladder")
    ap.add_argument("--vp", action="append",
                    help="viewpoint, e.g. 'right' or '[VP: back, low, wide]' "
                         "(repeatable; default: 4-azimuth eye-level sweep)")
    ap.add_argument("--out-dir", required=True,
                    help="where finished PNGs are copied (never the scratchpad)")
    ap.add_argument("--steps", type=int, default=None,
                    help="default 20, or 4 with --lightning")
    ap.add_argument("--cfg", type=float, default=None,
                    help="default 2.5, or 1.0 with --lightning")
    ap.add_argument("--lightning", action="store_true",
                    help="stack the 4-step Lightning LoRA (~5x faster)")
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames, so passes don't collide")
    ap.add_argument("--attempts", type=int, default=1,
                    help="renders per viewpoint, seed incrementing from --seed. "
                         "Seed varies POSE AND DETAIL ONLY - it does not move "
                         "the viewpoint (measured, 8x4 sweep). Use it for "
                         "variation within a viewpoint, never to hunt an angle")
    ap.add_argument("--subdirs", action="store_true",
                    help="write each viewpoint into its own subdirectory")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="LoRA strength (fal recommends 0.8-1.0)")
    ap.add_argument("--seed", type=int, default=42,
                    help="held fixed across the sweep so viewpoint is the "
                         "only variable")
    ap.add_argument("--negative", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the payload, queue nothing, render nothing")
    a = ap.parse_args()

    steps = a.steps if a.steps is not None else (4 if a.lightning else 20)
    cfg = a.cfg if a.cfg is not None else (1.0 if a.lightning else 2.5)

    vps = a.vp or ["front-right", "right", "back-right", "back"]
    views = [parse_vp(v) for v in vps]
    out_dir = Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    srcs = []
    for s in a.image:
        p = Path(s).resolve()
        if not p.exists():
            sys.exit(f"no such image: {p}")
        srcs.append(p)

    total = len(srcs) * len(views) * a.attempts
    print(f"{total} renders: {len(srcs)} image(s) x {len(views)} viewpoint(s) "
          f"x {a.attempts} attempt(s)")
    print(f"out: {out_dir}")
    print(f"steps={steps} cfg={cfg} strength={a.strength} seed={a.seed} "
          f"lightning={a.lightning}")
    for pr, tag in views:
        print(f"  {tag:28s} {pr}")

    if a.dry_run:
        for src in srcs:
            name, size = stage(src)
            g = build_graph(name, views[0][0], a.negative, a.seed, steps,
                            cfg, a.strength, "dryrun", a.lightning)
            (HERE / "last_payload.json").write_text(
                json.dumps({"prompt": g}, indent=1), encoding="utf-8")
            print(f"  staged {name} at {size[0]}x{size[1]}")
        print("dry run - nothing queued, nothing rendered")
        return

    # Klein / Z-Image / Wan may be resident from an earlier session and the
    # Qwen stack (Q5 GGUF + 7B VL encoder) wants the headroom. free_memory
    # also resets the node cache, which is the only thing that clears
    # compounded LoRA deltas (ComfyUI #11021).
    print("freeing VRAM before the Qwen stack loads ...")
    free_vram()

    done = 0
    for src in srcs:
        name, size = stage(src)
        for prompt, tag in views:
            dest_dir = out_dir / tag if a.subdirs else out_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            for i in range(a.attempts):
                seed = a.seed + i
                done += 1
                out_prefix = f"vp_{src.stem}_{tag}{a.tag}_{seed}"
                print(f"\n[{done}/{total}] {src.name}  {tag}  seed {seed}"
                      f"\n    {prompt}")
                g = build_graph(name, prompt, a.negative, seed, steps,
                                cfg, a.strength, out_prefix, a.lightning)
                pid = queue({"client_id": "akasutils", "prompt": g})
                for got in wait(pid, timeout=900):
                    dest = dest_dir / f"{src.stem}__{tag}{a.tag}_seed{seed}.png"
                    Image.open(got).save(dest)
                    print(f"    -> {dest}")

    print(f"\ndone: {total} renders in {out_dir}")


if __name__ == "__main__":
    main()
