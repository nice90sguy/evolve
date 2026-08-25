"""depth_warp.py — geometry-steered viewpoint change (aesthetic, not metric).

    stage 1  depth      ComfyUI DepthAnythingV2Preprocessor (vitl, already cached)
    stage 2  warp       unproject -> move camera -> reproject (splat + z-buffer)
    stage 3  refine     Qwen-Image-Edit at PARTIAL denoise (optional, --refine)

WHY THIS EXISTS. The Qwen multi-angle LoRA fails on scenes because it has no
geometry: it cannot express a camera move except through background parallax
it is unable to track, so it nominates a subject and rotates that instead
(CLAUDE.md, "Viewpoint control" campaign). The warp HANDS it the geometry —
real parallax, and an unambiguous chirality because WE pick the sign. So:
the warp supplies the evidence the model lacks; the model supplies the
plausibility the warp lacks. They are complementary failures.

The target is Ren'Py-game aesthetics, NOT metric correctness (user, 2026-08-22),
so relative depth plus an assumed FOV is enough. `--depth-scale` is a
by-eye parallax-strength knob, not a measurement.

Keep rotations modest. The campaign's usable envelope is about a quarter
turn; beyond that disocclusion holes stop being holes and become most of
the picture.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from pose_from_char import COMFY, queue, wait
from qwen_vp_probe import CLIP_NAME, LORA_LIGHTNING, MODEL_GGUF, VAE_NAME

HERE = Path(__file__).resolve().parent
SCRATCH_PREFIX = "akasutils_scratch"
DEPTH_CKPT = "depth_anything_v2_vitl.pth"   # cached in controlnet_aux/ckpts


# ---------------------------------------------------------------- stage 1

def depth_graph(image_name, resolution):
    return {
        "img": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "dep": {"class_type": "DepthAnythingV2Preprocessor",
                "inputs": {"image": ["img", 0], "ckpt_name": DEPTH_CKPT,
                           "resolution": resolution}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["dep", 0],
                            "filename_prefix": f"{SCRATCH_PREFIX}/warp_depth"}},
    }


def get_depth(image_name, size, resolution=1024):
    """Depth Anything v2 via the running ComfyUI. Returns float array in
    [0,1] at the source image size, where 1.0 = NEAREST (it emits disparity,
    bright = close)."""
    pid = queue({"client_id": "akasutils", "prompt": depth_graph(image_name, resolution)})
    out = wait(pid, timeout=300)
    if not out:
        sys.exit("depth pass produced no image")
    d = Image.open(out[0]).convert("L")
    if d.size != size:
        d = d.resize(size, Image.LANCZOS)
    return np.asarray(d).astype(np.float32) / 255.0


# ---------------------------------------------------------------- stage 2

def warp(rgb, disp, yaw_deg, pitch_deg, dolly, fov_deg, depth_scale,
         pivot=None, splat=2, ss=2):
    """Supersampling wrapper around _warp.

    Forward splatting scatters one source pixel per target pixel, so wherever
    the warp MAGNIFIES, the target grid is sampled sparsely and you get a
    regular lattice of gaps — visible as cross-hatched dots and contour-
    following streaks (user crops, 2026-08-22). Those are a rasterisation
    artefact, not missing information: the pixels exist, they are just landing
    too far apart. Warping at ss x resolution puts ss^2 more source points
    into the same target area and closes them; what survives downsampling is
    genuine disocclusion.
    """
    if ss <= 1:
        return _warp(rgb, disp, yaw_deg, pitch_deg, dolly, fov_deg,
                     depth_scale, pivot, splat)

    H, W = disp.shape
    big_rgb = np.asarray(Image.fromarray(rgb).resize((W * ss, H * ss), Image.LANCZOS))
    big_disp = np.asarray(Image.fromarray(disp).resize((W * ss, H * ss), Image.BILINEAR))
    w, m = _warp(big_rgb, big_disp, yaw_deg, pitch_deg, dolly, fov_deg,
                 depth_scale, pivot, splat)
    # Area-average back down. A target pixel counts as a hole only if it is
    # mostly hole, so surviving speckle averages away instead of dilating.
    w = np.asarray(Image.fromarray(w).resize((W, H), Image.BOX))
    m = np.asarray(Image.fromarray(m).resize((W, H), Image.BOX))
    return w, (m > 127).astype(np.uint8) * 255


def _warp(rgb, disp, yaw_deg, pitch_deg, dolly, fov_deg, depth_scale,
          pivot=None, splat=2):
    """Move the camera and re-render by forward splatting.

    Convention: +yaw orbits the camera to the RIGHT, +pitch orbits it UP,
    +dolly pushes it toward the subject. Chirality is ours to choose, which
    is exactly what the LoRA could not do -- if a sign reads backwards for
    a given plate, negate it.

    Returns (warped uint8 HxWx3, hole mask uint8 HxW where 255 = no source).
    """
    H, W = disp.shape
    f = (W * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
    cx, cy = W * 0.5, H * 0.5

    # disparity -> depth. Near plane at 1.0, far at 1+depth_scale, so
    # depth_scale IS the parallax-strength knob.
    near, far = 1.0, 1.0 + max(depth_scale, 1e-3)
    D = np.clip(disp, 0.0, 1.0)
    z = 1.0 / (D * (1.0 / near - 1.0 / far) + 1.0 / far)

    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    pts = np.stack([(u - cx) * z / f, (v - cy) * z / f, z], -1).reshape(-1, 3)

    # Orbit about a pivot on the optical axis: rotating the SCENE about it
    # is the same as orbiting the CAMERA the other way.
    piv = np.array([0.0, 0.0,
                    float(pivot) if pivot else float(np.median(z))], np.float32)
    ya, pa = math.radians(-yaw_deg), math.radians(-pitch_deg)
    Ry = np.array([[math.cos(ya), 0, math.sin(ya)],
                   [0, 1, 0],
                   [-math.sin(ya), 0, math.cos(ya)]], np.float32)
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pa), -math.sin(pa)],
                   [0, math.sin(pa), math.cos(pa)]], np.float32)
    q = (pts - piv) @ (Rx @ Ry).T + piv
    q[:, 2] -= dolly

    zc = q[:, 2]
    ok = zc > 1e-3
    un = f * q[:, 0] / np.where(ok, zc, 1.0) + cx
    vn = f * q[:, 1] / np.where(ok, zc, 1.0) + cy

    src = rgb.reshape(-1, 3)
    out = np.zeros((H * W, 3), np.uint8)
    filled = np.zeros(H * W, bool)

    # Painter's algorithm: far first, so nearer points overwrite them.
    # Splatting a small block closes the speckle gaps that appear wherever
    # the warp magnifies.
    for dy in range(splat):
        for dx in range(splat):
            ui = np.floor(un).astype(np.int64) + dx
            vi = np.floor(vn).astype(np.int64) + dy
            m = ok & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if not m.any():
                continue
            idx = np.nonzero(m)[0]
            idx = idx[np.argsort(-zc[idx])]          # descending z = far first
            tgt = vi[idx] * W + ui[idx]
            out[tgt] = src[idx]
            filled[tgt] = True

    return out.reshape(H, W, 3), (~filled).reshape(H, W).astype(np.uint8) * 255


def fill_holes(warped, mask):
    """Telea-inpaint the disocclusion holes.

    NOTE: Telea is a SMEAR filter — user verdict 2026-08-22 was "the invented
    edge is just a smudgy blob", and that is all it can ever be. This is now
    only a base layer for the masked-inpaint route (a smear VAE-encodes more
    sanely than a black void, and gets overwritten inside the mask anyway).
    Do not ship its output as the result."""
    try:
        import cv2
    except ImportError:
        return warped
    return cv2.inpaint(warped, mask, 3, cv2.INPAINT_TELEA)


def feather_mask(mask, grow=8, blur=5):
    """Dilate then soften the hole mask. Dilation matters twice over: splat
    edges are unreliable for a pixel or two, and the mask must cover the
    smeared boundary, not just the void."""
    try:
        import cv2
    except ImportError:
        return mask
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
        mask = cv2.dilate(mask, k)
    if blur > 0:
        b = 2 * blur + 1
        mask = cv2.GaussianBlur(mask, (b, b), 0)
    return mask


def inpaint_graph(base_name, mask_name, prompt, negative, seed, steps, cfg,
                  lightning, out_prefix):
    """Regenerate ONLY the disocclusion holes: SetLatentNoiseMask, denoise
    1.0 inside the mask, untouched outside. This is the right shape for the
    job — the warp's geometry is correct and must be preserved, while the
    holes have no source pixels and must genuinely be invented. Global
    partial denoise degrades the good pixels to fix the bad ones."""
    g = {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": MODEL_GGUF}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": CLIP_NAME, "type": "qwen_image"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "img": {"class_type": "LoadImage", "inputs": {"image": base_name}},
        "msk": {"class_type": "LoadImageMask",
                "inputs": {"image": mask_name, "channel": "red"}},
        "pos": {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["clip", 0], "prompt": prompt,
                           "vae": ["vae", 0], "image1": ["img", 0]}},
        "neg": {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["clip", 0], "prompt": negative,
                           "vae": ["vae", 0], "image1": ["img", 0]}},
        "enc": {"class_type": "VAEEncode",
                "inputs": {"pixels": ["img", 0], "vae": ["vae", 0]}},
        "lat": {"class_type": "SetLatentNoiseMask",
                "inputs": {"samples": ["enc", 0], "mask": ["msk", 0]}},
        "samp": {"class_type": "KSampler",
                 "inputs": {"model": ["unet", 0], "seed": seed, "steps": steps,
                            "cfg": cfg, "sampler_name": "euler",
                            "scheduler": "simple", "positive": ["pos", 0],
                            "negative": ["neg", 0], "latent_image": ["lat", 0],
                            "denoise": 1.0}},
        "dec": {"class_type": "VAEDecode",
                "inputs": {"samples": ["samp", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["dec", 0],
                            "filename_prefix": f"{SCRATCH_PREFIX}/{out_prefix}"}},
    }
    if lightning:
        g["fast"] = {"class_type": "LoraLoaderModelOnly",
                     "inputs": {"model": ["unet", 0], "lora_name": LORA_LIGHTNING,
                                "strength_model": 1.0}}
        g["samp"]["inputs"]["model"] = ["fast", 0]
    return g


# ---------------------------------------------------------------- stage 3

def refine_graph(image_name, prompt, negative, seed, steps, cfg, denoise,
                 lightning, out_prefix):
    """Plain Qwen-Image-Edit cleanup at partial denoise. NO angle LoRA — the
    warp already did the camera move; the model is only fixing artifacts.
    Too much denoise and it re-invents the shot and throws your camera move
    away (same over-invention that made 20 steps worse than 4)."""
    g = {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": MODEL_GGUF}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": CLIP_NAME, "type": "qwen_image"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "img": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "pos": {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["clip", 0], "prompt": prompt,
                           "vae": ["vae", 0], "image1": ["img", 0]}},
        "neg": {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["clip", 0], "prompt": negative,
                           "vae": ["vae", 0], "image1": ["img", 0]}},
        "lat": {"class_type": "VAEEncode",
                "inputs": {"pixels": ["img", 0], "vae": ["vae", 0]}},
        "samp": {"class_type": "KSampler",
                 "inputs": {"model": ["unet", 0], "seed": seed, "steps": steps,
                            "cfg": cfg, "sampler_name": "euler",
                            "scheduler": "simple", "positive": ["pos", 0],
                            "negative": ["neg", 0], "latent_image": ["lat", 0],
                            "denoise": denoise}},
        "dec": {"class_type": "VAEDecode",
                "inputs": {"samples": ["samp", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["dec", 0],
                            "filename_prefix": f"{SCRATCH_PREFIX}/{out_prefix}"}},
    }
    if lightning:
        g["fast"] = {"class_type": "LoraLoaderModelOnly",
                     "inputs": {"model": ["unet", 0], "lora_name": LORA_LIGHTNING,
                                "strength_model": 1.0}}
        g["samp"]["inputs"]["model"] = ["fast", 0]
    return g


# ---------------------------------------------------------------- driver

def stage_source(src):
    img = Image.open(src)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        flat = Image.new("RGB", img.size, (255, 255, 255))
        flat.paste(img, mask=img.split()[-1])
        img = flat
    else:
        img = img.convert("RGB")
    name = f"warp_src_{src.stem}.png"
    img.save(COMFY / "input" / name)
    return name, img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--yaw", type=float, default=0.0, help="+ orbits camera right")
    ap.add_argument("--pitch", type=float, default=0.0, help="+ orbits camera up")
    ap.add_argument("--dolly", type=float, default=0.0, help="+ moves camera in")
    ap.add_argument("--fov", type=float, default=55.0, help="assumed horizontal FOV")
    ap.add_argument("--depth-scale", type=float, default=3.0,
                    help="parallax strength (by eye, not a measurement)")
    ap.add_argument("--pivot", type=float, default=None,
                    help="orbit distance; default = median scene depth")
    ap.add_argument("--splat", type=int, default=2)
    ap.add_argument("--ss", type=int, default=2,
                    help="supersample factor for the warp; kills the "
                         "magnification speckle. 1 = off")
    ap.add_argument("--fill", choices=("color", "telea", "none"), default="color",
                    help="what to put in the holes BEFORE refining. 'color' "
                         "(default) marks them with a flat out-of-gamut colour "
                         "the model can see and be told about; 'telea' smears "
                         "neighbouring pixels in, which MISINFORMS it (a tear "
                         "inside the stool gets smeared with bar, and the model "
                         "duly inpaints bar); 'none' leaves them black, which "
                         "reads as real dark scene content in a dim room.")
    ap.add_argument("--hole-color", default="#00ff00",
                    help="marker colour for --fill color")
    ap.add_argument("--no-fill", action="store_true",
                    help="deprecated alias for --fill none")
    ap.add_argument("--depth-resolution", type=int, default=1024)
    # stage 3
    ap.add_argument("--refine", action="store_true",
                    help="run the Qwen cleanup (GPU render)")
    ap.add_argument("--refine-mode", choices=("mask", "global"), default="mask",
                    help="mask = regenerate ONLY the disocclusion holes "
                         "(preserves the warp's geometry; the right default). "
                         "global = partial denoise over the whole frame")
    ap.add_argument("--mask-grow", type=int, default=8)
    ap.add_argument("--mask-blur", type=int, default=5)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--negative", default="")
    ap.add_argument("--denoise", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=1.0,
                    help="cfg 1.0 = NO classifier-free guidance, so the prompt "
                         "barely steers and the reference image wins. Measured "
                         "2026-08-22: marker colour surviving the inpaint went "
                         "3.10%% at cfg 1/4-step -> 0.69%% at cfg 4/20-step. "
                         "Low cfg when you want the REFERENCE to win (viewpoint "
                         "edits); high cfg when you need the PROMPT to win "
                         "(repairing marked holes). Lightning forces cfg ~1, so "
                         "raising cfg means --no-lightning and ~6x the time.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-lightning", action="store_true")
    a = ap.parse_args()

    src = Path(a.image).resolve()
    if not src.exists():
        sys.exit(f"no such image: {src}")
    out_dir = Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    name, img = stage_source(src)
    rgb = np.asarray(img)
    print(f"{src.name}  {img.size[0]}x{img.size[1]}")
    print(f"yaw {a.yaw:+g}  pitch {a.pitch:+g}  dolly {a.dolly:+g}  "
          f"fov {a.fov:g}  depth-scale {a.depth_scale:g}")

    print("stage 1: depth (Depth Anything v2 via ComfyUI) ...")
    disp = get_depth(name, img.size, a.depth_resolution)
    tag = f"y{a.yaw:+g}_p{a.pitch:+g}_d{a.dolly:+g}".replace("+", "p").replace("-", "m")
    Image.fromarray((disp * 255).astype(np.uint8)).save(
        out_dir / f"{src.stem}__depth.png")

    print("stage 2: warp ...")
    warped, mask = warp(rgb, disp, a.yaw, a.pitch, a.dolly, a.fov,
                        a.depth_scale, a.pivot, a.splat, a.ss)
    hole_pct = 100.0 * (mask > 0).mean()
    print(f"    disocclusion holes: {hole_pct:.1f}% of frame")
    Image.fromarray(mask).save(out_dir / f"{src.stem}__{tag}_mask.png")
    Image.fromarray(warped).save(out_dir / f"{src.stem}__{tag}_warp_raw.png")
    mode = "none" if a.no_fill else a.fill
    if mode == "telea":
        final = fill_holes(warped, mask)
    elif mode == "color":
        # A flat marker beats a smear: the smear tells the model that a tear
        # inside the stool is made of bar, and it believes it. A colour the
        # scene cannot contain says only "this is missing" - and can be named
        # in the prompt. Mask grow must exceed blur so no marker pixel ends up
        # only partially masked, or the colour bleeds into the result.
        c = a.hole_color.lstrip("#")
        rgbc = np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)
        final = warped.copy()
        final[mask > 127] = rgbc
    else:
        final = warped
    warp_path = out_dir / f"{src.stem}__{tag}_warp.png"
    Image.fromarray(final).save(warp_path)
    print(f"    -> {warp_path}")

    if not a.refine:
        print("stage 3 skipped (pass --refine to run the GPU cleanup)")
        return

    rname = f"warp_refine_{src.stem}.png"
    Image.fromarray(final).save(COMFY / "input" / rname)

    if a.refine_mode == "mask":
        print(f"stage 3: masked inpaint of holes "
              f"(grow {a.mask_grow}, blur {a.mask_blur}) ...")
        if mode == "color" and a.mask_grow < a.mask_blur:
            print(f"    WARNING: mask-grow {a.mask_grow} < mask-blur "
                  f"{a.mask_blur}; marker colour may sit in partially-masked "
                  f"pixels and bleed into the result")
        fm = feather_mask(mask, a.mask_grow, a.mask_blur)
        mname = f"warp_mask_{src.stem}.png"
        Image.fromarray(fm).convert("RGB").save(COMFY / "input" / mname)
        Image.fromarray(fm).save(out_dir / f"{src.stem}__{tag}_mask_feathered.png")
        print(f"    masked area: {100.0 * (fm > 0).mean():.1f}% of frame")
        g = inpaint_graph(rname, mname, a.prompt, a.negative, a.seed,
                          a.steps, a.cfg, not a.no_lightning,
                          f"warp_ref_{src.stem}")
        # Every knob that changes the result belongs in the name. Runs
        # differing only in fill, or only in cfg/steps, silently overwrote
        # each other twice before this was fixed.
        suffix = (f"inpaint_{mode}_g{a.mask_grow}_cfg{a.cfg:g}_s{a.steps}"
                  + ("" if a.no_lightning else "_lt"))
    else:
        print(f"stage 3: global refine, denoise {a.denoise} ...")
        g = refine_graph(rname, a.prompt, a.negative, a.seed, a.steps, a.cfg,
                         a.denoise, not a.no_lightning, f"warp_ref_{src.stem}")
        suffix = f"global_dn{a.denoise:g}"

    pid = queue({"client_id": "akasutils", "prompt": g})
    for got in wait(pid, timeout=900):
        dest = out_dir / f"{src.stem}__{tag}_{suffix}.png"
        Image.open(got).save(dest)
        print(f"    -> {dest}")


if __name__ == "__main__":
    main()
