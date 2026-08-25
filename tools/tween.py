"""Tween two keyframe images into a video (Wan 2.2 FLF2V via ComfyUI).

Usage:  python tween.py --start a.png --end b.png --prompt "..." [options]

The two keyframes are enforced as ground truth at t=0 and t=1 inside the
diffusion (WanFirstLastFrameToVideo); the prompt describes the MOTION
between them, not the destination - the end frame IS the destination.
Chain segments by reusing a segment's end frame as the next one's start:
shared keyframes make the joins pixel-perfect by construction.

Models (native ComfyUI Wan 2.2 graph, two-expert MoE sampling):
  diffusion_models\\wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
  diffusion_models\\wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
  text_encoders\\umt5_xxl_fp8_e4m3fn_scaled.safetensors  (type "wan")
  vae\\wan_2.1_vae.safetensors
High-noise expert samples the first half of the schedule (motion
planning), low-noise the second (refinement). The experts run as TWO
API submissions with a SaveLatent/LoadLatent handoff and a /free VRAM
flush between: both resident at once is 13.6+13.6+6.4GB TE > 32GB, and
ComfyUI 0.32's dynamic VRAM loader does not evict expert 1 at an
in-graph swap (OOMed twice, 2026-08-20). Output video via
VHS_VideoCombine (h264 mp4, proven installed).

Conventions: ComfyUI must be UP (launch it yourself); keyframes are
staged into ComfyUI's input dir; renders land in --out-dir (default .)
as <prefix>NNNNN.mp4, numbering continues, never overwrites; seed
printed for reuse. Width/height default to the START keyframe's dims
snapped to /16 - keep keyframes same-size (evolve siblings/lineage
images already are). Length must be 4n+1 frames (auto-adjusted).

evolve.py integration (parent/child keyframe warnings) comes later.
"""
import argparse
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path

from PIL import Image

import _cli  # noqa: F401  (puts the evolve package dir on sys.path)
from comfy_client import COMFY_DIR as COMFY, SCRATCH_PREFIX, free_vram, queue, wait_entry
from image_utils import snap16

HIGH = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
TE = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "wan_2.1_vae.safetensors"
DEFAULT_NEGATIVE = ("bad quality video, blurry, static frame, still image, "
                    "jpeg artifacts, deformed, disfigured, malformed limbs, "
                    "extra fingers, messy background, subtitles, watermark")


# The two experts CANNOT be in one graph: 13.6GB + 13.6GB + 6.4GB TE > 32GB,
# and ComfyUI 0.32's dynamic VRAM loader does NOT evict expert 1 at the swap
# ("0 models unloaded" -> hostbuf copy OOM, twice, 2026-08-20). So the tween
# is TWO submissions: high expert samples its half and saves the latent;
# /free flushes everything; low expert loads the latent and finishes. One
# expert resident at a time, ever.

def common_nodes(prompt, negative, width, height, length, start_name, end_name):
    """Nodes both stages share: conditioning + keyframe anchoring. flf is
    deterministic (VAE encode of the keyframes), so re-running it in stage 2
    reproduces stage 1's conditioning exactly."""
    return {
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": TE, "type": "wan", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "pos": {"class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode",
                "inputs": {"text": negative, "clip": ["clip", 0]}},
        "img_s": {"class_type": "LoadImage", "inputs": {"image": start_name}},
        "img_e": {"class_type": "LoadImage", "inputs": {"image": end_name}},
        "flf": {"class_type": "WanFirstLastFrameToVideo",
                "inputs": {"positive": ["pos", 0], "negative": ["neg", 0],
                           "vae": ["vae", 0], "width": width, "height": height,
                           "length": length, "batch_size": 1,
                           "start_image": ["img_s", 0],
                           "end_image": ["img_e", 0]}},
    }


def build_high(common, seed, steps, cfg, shift):
    """Stage 1: high-noise expert samples the first half of the schedule
    (motion planning) and saves the leftover-noise latent for handoff."""
    g = dict(common)
    g["high"] = {"class_type": "UNETLoader",
                 "inputs": {"unet_name": HIGH, "weight_dtype": "default"}}
    g["mh"] = {"class_type": "ModelSamplingSD3",
               "inputs": {"model": ["high", 0], "shift": shift}}
    g["ks_h"] = {"class_type": "KSamplerAdvanced",
                 "inputs": {"add_noise": "enable", "noise_seed": seed,
                            "steps": steps, "cfg": cfg,
                            "sampler_name": "euler", "scheduler": "simple",
                            "start_at_step": 0, "end_at_step": steps // 2,
                            "return_with_leftover_noise": "enable",
                            "model": ["mh", 0], "positive": ["flf", 0],
                            "negative": ["flf", 1], "latent_image": ["flf", 2]}}
    g["sl"] = {"class_type": "SaveLatent",
               "inputs": {"samples": ["ks_h", 0],
                          "filename_prefix": f"{SCRATCH_PREFIX}/tween_hi"}}
    return {"client_id": "tween", "prompt": g}


def build_low(common, latent_name, seed, steps, cfg, shift, fps, provenance):
    """Stage 2: low-noise expert continues from the staged latent
    (add_noise disabled - the leftover noise came with the latent) and
    decodes to video. `provenance` (original keyframe paths, argv, ...)
    rides along as extra_pnginfo: VHS_VideoCombine(save_metadata) embeds
    it in the mp4 next to the graph, so a video is self-describing even
    after the staged keyframe copies are overwritten by the next run."""
    g = dict(common)
    g["low"] = {"class_type": "UNETLoader",
                "inputs": {"unet_name": LOW, "weight_dtype": "default"}}
    g["ml"] = {"class_type": "ModelSamplingSD3",
               "inputs": {"model": ["low", 0], "shift": shift}}
    g["ll"] = {"class_type": "LoadLatent", "inputs": {"latent": latent_name}}
    g["ks_l"] = {"class_type": "KSamplerAdvanced",
                 "inputs": {"add_noise": "disable", "noise_seed": seed,
                            "steps": steps, "cfg": cfg,
                            "sampler_name": "euler", "scheduler": "simple",
                            "start_at_step": steps // 2, "end_at_step": 10000,
                            "return_with_leftover_noise": "disable",
                            "model": ["ml", 0], "positive": ["flf", 0],
                            "negative": ["flf", 1], "latent_image": ["ll", 0]}}
    g["dec"] = {"class_type": "VAEDecode",
                "inputs": {"samples": ["ks_l", 0], "vae": ["vae", 0]}}
    g["vid"] = {"class_type": "VHS_VideoCombine",
                "inputs": {"frame_rate": fps, "loop_count": 0,
                           "filename_prefix": f"{SCRATCH_PREFIX}/tween",
                           "format": "video/h264-mp4", "pix_fmt": "yuv420p",
                           "crf": 19, "save_metadata": True,
                           "trim_to_audio": False, "pingpong": False,
                           "save_output": True, "images": ["dec", 0]}}
    return {"client_id": "tween", "prompt": g,
            "extra_data": {"extra_pnginfo": {"evolve_tween": provenance}}}


def wait_latent(prompt_id, timeout=3600):
    """Wait for stage 1 (live progress); return the saved latent's path."""
    entry = wait_entry(prompt_id, timeout, client_id="tween")
    for out in entry["outputs"].values():
        for item in out.get("latents", []):
            return COMFY / "output" / item["subfolder"] / item["filename"]
    sys.exit("no latent in stage-1 outputs")


def wait_video(prompt_id, timeout=3600):
    """Wait for the tween (live per-step progress); return the mp4 path."""
    entry = wait_entry(prompt_id, timeout, client_id="tween")
    for out in entry["outputs"].values():
        for key in ("gifs", "videos", "images"):
            for item in out.get(key, []):
                if item["filename"].endswith(".mp4"):
                    return COMFY / "output" / item["subfolder"] / item["filename"]
    sys.exit("no mp4 in workflow outputs")


def next_mp4(out_dir, prefix):
    pat = re.compile(re.escape(prefix) + r"(\d{5})\.mp4$")
    taken = [int(m.group(1)) for p in out_dir.glob(f"{prefix}*.mp4")
             if (m := pat.fullmatch(p.name))] if out_dir.is_dir() else []
    return max(taken, default=0) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="first keyframe image")
    ap.add_argument("--end", required=True, help="last keyframe image")
    ap.add_argument("--prompt", required=True,
                    help="describe the MOTION between the keyframes")
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--frames", type=int, default=81,
                    help="video length; forced to 4n+1 (81 ~= 5s at 16fps)")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--width", type=int, default=None,
                    help="default: start keyframe's width (snapped /16)")
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--steps", type=int, default=20,
                    help="total steps, split evenly across the two experts")
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--shift", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--out-prefix", default="tween")
    args = ap.parse_args()

    start = Path(args.start).resolve()
    end = Path(args.end).resolve()
    for p in (start, end):
        if not p.is_file():
            sys.exit(f"keyframe not found: {p}")
    with Image.open(start) as im:
        sw, sh = im.size
    width = snap16(args.width if args.width else sw)
    height = snap16(args.height if args.height else sh)
    frames = max(5, ((args.frames - 1) // 4) * 4 + 1)
    seed = args.seed if args.seed is not None else random.randint(1, 999_999_999_999)

    shutil.copy2(start, COMFY / "input" / "tween_start.png")
    shutil.copy2(end, COMFY / "input" / "tween_end.png")

    common = common_nodes(args.prompt, args.negative, width, height, frames,
                          "tween_start.png", "tween_end.png")

    free_vram()   # evict any resident Klein model - the expert needs the room
    pid = queue(build_high(common, seed, args.steps, args.cfg, args.shift))
    print(f"stage 1/2 (high-noise expert, motion) -> {pid} (seed {seed}, "
          f"{frames} frames @ {args.fps}fps, {width}x{height}, {args.steps} steps)")
    latent = wait_latent(pid)

    # hand the half-sampled latent to stage 2 via the input dir, and flush
    # the high expert out of VRAM before the low one loads
    shutil.copy2(latent, COMFY / "input" / "tween_handoff.latent")
    free_vram()
    provenance = {"start": start.as_posix(), "end": end.as_posix(),
                  "prompt": args.prompt, "negative": args.negative,
                  "seed": seed, "width": width, "height": height,
                  "frames": frames, "fps": args.fps, "steps": args.steps,
                  "cfg": args.cfg, "shift": args.shift,
                  "cwd": Path.cwd().as_posix(), "argv": sys.argv[1:],
                  "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    pid = queue(build_low(common, "tween_handoff.latent", seed,
                          args.steps, args.cfg, args.shift, args.fps,
                          provenance))
    print(f"stage 2/2 (low-noise expert, refine + decode) -> {pid}")
    src = wait_video(pid)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{args.out_prefix}{next_mp4(out_dir, args.out_prefix):05d}.mp4"
    shutil.copy2(src, dest)
    # sidecar: the full recipe in plain text beside the video (the mp4
    # carries the same data in its metadata, but grep beats ffprobe)
    dest.with_suffix(".json").write_text(
        json.dumps(provenance, indent=1), encoding="utf-8")
    print(f"tweened -> {dest}  (recipe: {dest.with_suffix('.json').name})")


if __name__ == "__main__":
    main()
