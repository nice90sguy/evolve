"""Viewpoint probe for the Qwen-Image-Edit-2511 multiple-angles LoRA: any
source image in, re-shot from the trained viewpoints (the engine behind
evolve's Camera tab; what it established is in CLAUDE.md, findings 1-14).

    python tools/camera_probe.py --image src.png --vp right --vp "[VP: back, low, wide]" ^
        --out-dir D:/probe --lightning --attempts 4

Sources are staged with alpha flattened onto white and anything under
0.9MP upscaled to ~1MP (the model wants ~1MP). --attempts sweeps seeds per
viewpoint - seed varies POSE AND DETAIL ONLY, it never moves the viewpoint.
--lightning stacks the 4-step LoRA (4 steps / cfg 1.0): faster AND better.
"""
import argparse
import sys
from pathlib import Path

import _cli  # noqa: F401
from PIL import Image

from build_payload import qwen_edit
from camera import parse_vp
from comfy_client import INPUT_DIR, free_vram, queue, wait
from image_utils import fit_megapixels, flatten_alpha


def stage(src):
    """Copy the source into ComfyUI's input dir, flattened + fitted."""
    img, size = fit_megapixels(flatten_alpha(Image.open(src)), log=print)
    name = f"vp_src_{src.stem}.png"
    img.save(INPUT_DIR / name)
    return name, size


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", action="append", required=True, help="source image (repeatable)")
    ap.add_argument("--vp", action="append",
                    help="viewpoint, e.g. 'right' or '[VP: back, low, wide]' "
                         "(repeatable; default: 4-azimuth eye-level sweep)")
    ap.add_argument("--out-dir", required=True, help="where finished PNGs are copied")
    ap.add_argument("--steps", type=int, default=None, help="default 20, or 4 with --lightning")
    ap.add_argument("--cfg", type=float, default=None, help="default 2.5, or 1.0 with --lightning")
    ap.add_argument("--lightning", action="store_true", help="stack the 4-step Lightning LoRA")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--attempts", type=int, default=1, help="renders per viewpoint (seed += 1)")
    ap.add_argument("--subdirs", action="store_true", help="one subdirectory per viewpoint")
    ap.add_argument("--strength", type=float, default=1.0, help="LoRA strength (0.8-1.0)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negative", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="stage + write the payload, queue nothing")
    a = ap.parse_args()

    steps = a.steps if a.steps is not None else (
        qwen_edit.DEFAULT_STEPS if a.lightning else qwen_edit.SLOW_STEPS)
    cfg = a.cfg if a.cfg is not None else (
        qwen_edit.DEFAULT_CFG if a.lightning else qwen_edit.SLOW_CFG)
    try:
        views = [parse_vp(v) for v in (a.vp or ["front-right", "right", "back-right", "back"])]
    except ValueError as e:
        sys.exit(str(e))
    out_dir = Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    srcs = [Path(s).resolve() for s in a.image]
    for p in srcs:
        if not p.exists():
            sys.exit(f"no such image: {p}")

    total = len(srcs) * len(views) * a.attempts
    print(f"{total} renders: {len(srcs)} image(s) x {len(views)} viewpoint(s) "
          f"x {a.attempts} attempt(s)\nout: {out_dir}\n"
          f"steps={steps} cfg={cfg} strength={a.strength} seed={a.seed} lightning={a.lightning}")
    for pr, tag in views:
        print(f"  {tag:28s} {pr}")
    if a.dry_run:
        for src in srcs:
            name, size = stage(src)
            print(f"  staged {name} at {size[0]}x{size[1]}")
        print("dry run - nothing queued, nothing rendered")
        return

    print("freeing VRAM before the Qwen stack loads ...")
    free_vram()          # also resets the node cache (compounded LoRA deltas, #11021)
    done = 0
    for src in srcs:
        name, size = stage(src)
        for prompt, tag in views:
            dest_dir = out_dir / tag if a.subdirs else out_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            for i in range(a.attempts):
                seed = a.seed + i
                done += 1
                print(f"\n[{done}/{total}] {src.name}  {tag}  seed {seed}\n    {prompt}")
                pid = queue(qwen_edit.build(name, prompt, a.negative, seed, steps, cfg,
                                            a.strength, a.lightning,
                                            f"vp_{src.stem}_{tag}{a.tag}_{seed}"))
                for got in wait(pid, timeout=900):
                    dest = dest_dir / f"{src.stem}__{tag}{a.tag}_seed{seed}.png"
                    Image.open(got).save(dest)
                    print(f"    -> {dest}")
    print(f"\ndone: {total} renders in {out_dir}")


if __name__ == "__main__":
    _cli.exit_on_comfy_error(main)
