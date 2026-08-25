"""image_utils.py - pixel-level helpers shared by every driver.

flatten / snap / fit are cheap PIL ops; matte() is the finalize step
(rembg isnet-anime, imported lazily - it is slow to load and only the
sprite-deliverable path needs it).

    python image_utils.py <render.png> <out_name> [--normalize]   (matte CLI)
"""
import os

from PIL import Image


def snap16(v):
    """Nearest multiple of 16 (Flux2/Qwen latents want /16)."""
    return max(16, round(v / 16) * 16)


def has_alpha(img):
    return img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)


def flatten_alpha(img, bg=(255, 255, 255)):
    """RGB copy with any alpha flattened onto solid WHITE. Alpha must never
    reach LoadImage: it drops the channel, the model sees a cutout and
    paints checkerboard 'transparency' around the subject (CLAUDE.md)."""
    if has_alpha(img):
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, bg)
        flat.paste(rgba, mask=rgba.getchannel("A"))
        return flat
    return img.convert("RGB")


def fit_megapixels(img, min_mp=0.9, target_mp=1.05, log=None):
    """Upscale an undersized image to ~target_mp (Qwen-Image-Edit-2511 wants
    ~1MP - a 512x512 source would confound a negative result with an input
    problem), and snap dims to /16 either way. Returns (img, (w, h))."""
    w, h = img.size
    mp = w * h / 1e6
    if mp < min_mp:
        s = (target_mp / mp) ** 0.5
        w, h = snap16(round(w * s)), snap16(round(h * s))
        img = img.resize((w, h), Image.LANCZOS)
        if log:
            log(f"  upscaled: {mp:.2f}MP -> {w}x{h} ({w * h / 1e6:.2f}MP)")
    elif w % 16 or h % 16:
        w, h = snap16(w), snap16(h)
        img = img.resize((w, h), Image.LANCZOS)
        if log:
            log(f"  snapped to /16: {w}x{h}")
    return img, (w, h)


# ---------- matting (the old finalize.py) ----------

_session = None


def matte(src, out_name, normalize=False):
    """Matte a rendered PNG -> transparent webp + dark-bg check png, written
    next to the source. rembg isnet-anime (user-verified better than
    InSPyReNet) -> harden alpha (<=40 -> 0, >=220 -> 255, stretch between;
    raw mattes leave interiors ~240-254, which ghosts over dark scenes).

    Default: IN PLACE - the webp keeps the render's exact dims and position.
    normalize=True is the legacy sprite deliverable: trim to alpha bbox ->
    scale to 1080 high -> center on a 1920x1080 canvas.
    Generate on a WHITE background - dark clothing on black won't matte.
    ALWAYS inspect the _darkcheck.png: halo/pocket/edge failures are
    invisible on light backgrounds."""
    global _session
    import numpy as np
    from rembg import new_session, remove
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
    import sys
    cli = [a for a in sys.argv[1:] if a != "--normalize"]
    if len(cli) != 2:
        sys.exit("usage: python image_utils.py <render.png> <out_name> [--normalize]\n"
                 "matte a render to <out_name>.webp + <out_name>_darkcheck.png")
    print("wrote", matte(cli[0], cli[1], normalize="--normalize" in sys.argv))
