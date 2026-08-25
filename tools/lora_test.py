"""Render a prompt with a trained LoRA - the likeness check after a bake.

    python tools/lora_test.py --family klein  --lora D:/root/loras/julie/julie_v003_x_comfy.safetensors ^
        --prompt "julie, portrait" --seed 7 --num-images 4 --out-dir D:/checks
    python tools/lora_test.py --family zimage --lora <native or _comfy file> ...

A musubi-native file is converted (and, for Klein, key-fixed) on the fly;
the _comfy file is cached beside it. Renders on the DISTILLED model
(Klein 9B fp8 / Z-Image Turbo) - never on the training base.
"""
import argparse
import random
import sys
from pathlib import Path

import _cli  # noqa: F401

import lora_train
from build_payload import flux_klein, zimage
from comfy_client import queue, wait
from image_file import save_renders
from image_utils import snap16
from lora_train.common import TrainError


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="klein", choices=["klein", "zimage"])
    ap.add_argument("--lora", required=True, help="LoRA file (native or _comfy)")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--seed", type=int, default=None, help="pin to compare on identical noise")
    ap.add_argument("--num-images", type=int, default=1)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--out-prefix", default="img")
    args = ap.parse_args()

    lora = Path(args.lora).expanduser().resolve()
    if not lora.is_file():
        sys.exit(f"LoRA file not found: {lora}")
    if not lora.name.endswith("_comfy.safetensors"):
        try:
            lora = lora_train.get_trainer(args.family).to_comfy(lora)
        except TrainError as e:
            sys.exit(str(e))
    seed = args.seed if args.seed is not None else random.randint(1, 999_999_999_999)
    w, h = snap16(args.width), snap16(args.height)
    if args.family == "zimage":
        payload = zimage.build(args.prompt, args.negative, seed, w, h,
                               args.steps or zimage.DEFAULT_STEPS, args.cfg or zimage.DEFAULT_CFG,
                               lora, args.strength, args.num_images, "loratest")
    else:
        payload = flux_klein.build(args.prompt, seed, w, h, lora_path=lora,
                                   lora_strength=args.strength, steps=args.steps, cfg=args.cfg,
                                   batch_size=args.num_images, out_prefix="loratest")
    pid = queue(payload)
    print(f"queued {lora.name} -> {pid} (seed {seed}, strength {args.strength})")
    srcs = wait(pid, timeout=600)
    save_renders(srcs, Path(args.out_dir).expanduser().resolve(), args.out_prefix)


if __name__ == "__main__":
    _cli.exit_on_comfy_error(main)
