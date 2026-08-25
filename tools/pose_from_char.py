"""Render pose/expression variants of a character from reference images via
the ComfyUI API: single- or multi-reference identity transfer (FLUX.2 Klein +
IdentityFeatureTransferFinal) plus matting to a transparent webp.

Examples:
  python tools/pose_from_char.py --ref char.webp --prompt "..." --out-dir chars/anita
  python tools/pose_from_char.py --ref "primary.webp,hand.webp" --identity-refs 0 ^
      --locks SOFT_LOCK,MID_LOCK --seed 960 --prompt "..." --num-images 4

Defaults are tuned for making drop-in variants of an existing sprite:
- Refs with an alpha channel are flattened onto solid WHITE at staging, and
  the prompt is prefixed with a white-background instruction (a transparent
  sprite fed raw makes the model paint checkerboard blocks).
  --no-flatten-alpha disables both.
- --width/--height default to the primary ref's dimensions; the render runs
  at /16-snapped dims and is resized back to the exact target.
- Matting runs by default (--no-finalize for the raw render only), in place.

Renders go into --out-dir as <prefix>NNNNN.png (5-digit, numbering continues
past existing files, nothing is ever overwritten); multiple --locks tag the
prefix (img_soft00001.png). The FIRST --ref is the primary identity
reference (reference_index 0). The submitted graph is written to
_debug/last_payload.json.
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

import _cli  # noqa: F401
from PIL import Image

from build_payload import flux_klein
from build_payload.flux_klein import PRESETS, WHITE_BG_PREFIX
from comfy_client import INPUT_DIR, queue, wait
from image_file import save_renders
from image_utils import flatten_alpha, has_alpha, snap16


def stage_refs(ref_arg, flatten):
    """Stage refs into ComfyUI's input dir, flattening any alpha channel onto
    solid white (unless disabled). Returns (staged basenames, primary WxH)."""
    names, primary_size = [], None
    for raw in ref_arg.split(","):
        p = Path(raw.strip()).resolve()
        img = Image.open(p)
        if primary_size is None:
            primary_size = img.size
        if flatten and has_alpha(img):
            name = p.stem + "_white.png"
            flatten_alpha(img).save(INPUT_DIR / name)
        else:
            name = p.name
            if p != INPUT_DIR / name:
                shutil.copy2(p, INPUT_DIR / name)
        names.append(name)
    return names, primary_size


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True,
                    help="reference image path(s), comma-separated; first = primary identity")
    ap.add_argument("--prompt", required=True, help="full positive prompt text")
    ap.add_argument("--out-dir", default=".", help="directory renders are written into")
    ap.add_argument("--out-prefix", default="img",
                    help='render filename prefix; files are <prefix>NNNNN.png (default "img")')
    ap.add_argument("--num-images", type=int, default=1, help="latent batch size")
    ap.add_argument("--seed", type=int, default=None, help="default: random, printed for reuse")
    ap.add_argument("--width", type=int, default=None, help="default: primary ref's width")
    ap.add_argument("--height", type=int, default=None, help="default: primary ref's height")
    ap.add_argument("--locks", default="SOFT_LOCK",
                    help="comma-separated identity presets: SOFT_LOCK,MID_LOCK,HARD_LOCK")
    ap.add_argument("--identity-refs", default="all",
                    help='which refs drive identity: "all", "0", "0,2", "0-1"')
    ap.add_argument("--flatten-alpha", default=True, action=argparse.BooleanOptionalAction,
                    help="flatten transparent refs onto white + white-background prompt prefix")
    ap.add_argument("--finalize", default=True, action=argparse.BooleanOptionalAction,
                    help="matte each render to a transparent webp at the same dims")
    args = ap.parse_args()

    locks = [lk.strip().upper() for lk in args.locks.split(",") if lk.strip()]
    for lock in locks:
        if lock not in PRESETS:
            sys.exit(f"unknown lock preset '{lock}' - valid: {', '.join(PRESETS)}")

    refs, (ref_w, ref_h) = stage_refs(args.ref, args.flatten_alpha)
    width = args.width if args.width is not None else ref_w
    height = args.height if args.height is not None else ref_h
    prompt = (WHITE_BG_PREFIX + args.prompt) if args.flatten_alpha else args.prompt
    seed = args.seed if args.seed is not None else random.randint(1, 999_999_999_999)
    out_dir = Path(args.out_dir).expanduser().resolve()

    for lock in locks:
        prefix = args.out_prefix if len(locks) == 1 \
            else f"{args.out_prefix}_{lock.lower().replace('_lock', '')}"
        payload = flux_klein.build(prompt, seed, snap16(width), snap16(height), refs, lock,
                                   identity_refs=args.identity_refs,
                                   batch_size=args.num_images, out_prefix=prefix)
        pid = queue(payload)
        print(f"queued {lock} -> {pid} (seed {seed}, {width}x{height}, "
              f"{args.num_images} image(s))")
        srcs = wait(pid)
        for png in save_renders(srcs, out_dir, prefix, (width, height)):
            if args.finalize:
                from image_utils import matte
                print("matted ->", matte(str(png), png.stem))


if __name__ == "__main__":
    _cli.exit_on_comfy_error(main)
