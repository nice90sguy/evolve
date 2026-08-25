"""Matte a rendered PNG -> transparent webp + dark-bg check.

Usage:  python finalize.py <render.png> <out_name> [--normalize]
Writes <out_name>.webp and <out_name>_darkcheck.png next to the source render.

Default: matte IN PLACE — the webp keeps the render's exact dimensions and
the character stays exactly where it sits in the render (the pose_from_char
workflow anchors position to the ref sprite). --normalize is the legacy
sprite-deliverable path: trim to alpha bbox -> scale to 1080 height ->
center on a 1920x1080 canvas.

Pipeline: rembg isnet-anime matte -> harden alpha (<=40 -> 0, >=220 -> 255,
stretch between; raw mattes leave interiors ~240-254, which ghosts over dark
scenes). Generate on a WHITE background — dark clothing on black won't matte.
"""
import os
import sys

import numpy as np
from PIL import Image
from rembg import new_session, remove

_session = None


def finalize(src: str, out_name: str, normalize: bool = False) -> str:
    global _session
    if _session is None:
        _session = new_session("isnet-anime")
    out_dir = os.path.dirname(os.path.abspath(src))

    matted = remove(Image.open(src), session=_session)
    arr = np.array(matted.convert("RGBA")).astype(np.float32)
    a = arr[..., 3]
    arr[..., 3] = np.clip((a - 40.0) * (255.0 / (220.0 - 40.0)), 0, 255)
    matted = Image.fromarray(arr.astype(np.uint8), "RGBA")

    if normalize:
        matted = matted.crop(matted.getchannel("A").getbbox())
        scale = 1080.0 / matted.height
        matted = matted.resize((round(matted.width * scale), 1080), Image.LANCZOS)
        canvas = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
        canvas.paste(matted, ((1920 - matted.width) // 2, 0), matted)
    else:
        canvas = matted

    out_webp = os.path.join(out_dir, out_name + ".webp")
    canvas.save(out_webp, "WEBP", lossless=False, quality=95)

    dark = Image.new("RGBA", canvas.size, (24, 22, 28, 255))
    dark.alpha_composite(canvas)
    dark.convert("RGB").save(os.path.join(out_dir, out_name + "_darkcheck.png"))
    return out_webp


if __name__ == "__main__":
    cli = [a for a in sys.argv[1:] if a != "--normalize"]
    if len(cli) != 2:
        sys.exit(__doc__)
    print("wrote", finalize(cli[0], cli[1], normalize="--normalize" in sys.argv))
