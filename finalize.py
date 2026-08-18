"""Matte a rendered PNG -> game-ready 1920x1080 transparent webp + dark-bg check.

Usage:  python finalize.py <render.png> <out_name>
Writes <out_name>.webp and <out_name>_darkcheck.png next to the source render.

Pipeline: rembg isnet-anime matte -> harden alpha (<=40 -> 0, >=220 -> 255,
stretch between; raw mattes leave interiors ~240-254, which ghosts over dark
scenes) -> trim -> scale to 1080 height -> center on a 1920x1080 transparent
canvas. Generate on a WHITE background — dark clothing on black won't matte.
"""
import os
import sys

import numpy as np
from PIL import Image
from rembg import new_session, remove

_session = None


def finalize(src: str, out_name: str) -> str:
    global _session
    if _session is None:
        _session = new_session("isnet-anime")
    out_dir = os.path.dirname(os.path.abspath(src))

    matted = remove(Image.open(src), session=_session)
    arr = np.array(matted.convert("RGBA")).astype(np.float32)
    a = arr[..., 3]
    arr[..., 3] = np.clip((a - 40.0) * (255.0 / (220.0 - 40.0)), 0, 255)
    matted = Image.fromarray(arr.astype(np.uint8), "RGBA")

    matted = matted.crop(matted.getchannel("A").getbbox())
    scale = 1080.0 / matted.height
    matted = matted.resize((round(matted.width * scale), 1080), Image.LANCZOS)
    canvas = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    canvas.paste(matted, ((1920 - matted.width) // 2, 0), matted)

    out_webp = os.path.join(out_dir, out_name + ".webp")
    canvas.save(out_webp, "WEBP", lossless=False, quality=95)

    dark = Image.new("RGBA", (1920, 1080), (24, 22, 28, 255))
    dark.alpha_composite(canvas)
    dark.convert("RGB").save(os.path.join(out_dir, out_name + "_darkcheck.png"))
    return out_webp


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print("wrote", finalize(sys.argv[1], sys.argv[2]))
