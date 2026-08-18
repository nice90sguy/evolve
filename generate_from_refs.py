"""Render character variants from reference images via the ComfyUI API.

Python port of charmed_pipeline\\generate.ps1: single- or multi-reference
identity transfer (FLUX.2 Klein + IdentityFeatureTransferFinal), optional
matting to a game-ready transparent webp.

Assumes the ComfyUI service is up (http://127.0.0.1:8188) and runs in the
ComfyUI venv (D:\\projects\\ComfyUI\\.venv) — see requirements.txt.

Examples:
  python generate_from_refs.py --ref base.webp --prompt "..." --out mychar_pose
  python generate_from_refs.py --ref "primary.webp,hand.webp" --identity-refs 0 ^
      --locks SOFT_LOCK,MID_LOCK --seed 960 --prompt "..." --out pose --finalize

The FIRST --ref image is the primary identity reference (reference_index 0);
later refs add supporting views or gesture/pose steering. Multiple --locks
render once per preset on the same seed, tagged _soft/_mid/_hard.
Every run writes the exact submitted graph to last_payload.json.
"""
import argparse
import json
import random
import shutil
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMFY = HERE.parent                      # akasutils lives inside the ComfyUI dir
API = "http://127.0.0.1:8188"
TEMPLATE = HERE / "template_single_ref.json"
OUT_DIR = COMFY / "output" / "charmed2"

# preset -> (similarity_floor, softmax_temperature), from
# IdentityFeatureTransferFinal.PRESETS in ComfyUI-Flux2Klein-Enhancer.
PRESETS = {
    "SOFT_LOCK": (0.5, 0.07),
    "MID_LOCK": (0.2, 0.07),
    "HARD_LOCK": (0.04, 0.025),
}


def stage_refs(ref_arg):
    """Copy refs into ComfyUI's input dir; return their basenames in order."""
    names = []
    for raw in ref_arg.split(","):
        p = Path(raw.strip()).resolve()
        dest = COMFY / "input" / p.name
        if p != dest:
            shutil.copy2(p, dest)
        names.append(p.name)
    return names


def build_payload(refs, prompt, seed, width, height, lock, identity_refs, prefix):
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    p = payload["prompt"]

    # Replace the template's single-ref nodes with a chain: one
    # LoadImage+VAEEncode per ref, ReferenceLatent chained through positive
    # and negative conditioning in --ref order.
    for old in ("img_ref", "enc_ref", "pos_ref", "neg_ref"):
        p.pop(old, None)
    for i, name in enumerate(refs):
        p[f"img_{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        p[f"enc_{i}"] = {"class_type": "VAEEncode",
                         "inputs": {"pixels": [f"img_{i}", 0], "vae": ["vae", 0]}}
        pos_src = "txt" if i == 0 else f"pos_{i - 1}"
        neg_src = "zero" if i == 0 else f"neg_{i - 1}"
        p[f"pos_{i}"] = {"class_type": "ReferenceLatent",
                         "inputs": {"conditioning": [pos_src, 0], "latent": [f"enc_{i}", 0]}}
        p[f"neg_{i}"] = {"class_type": "ReferenceLatent",
                         "inputs": {"conditioning": [neg_src, 0], "latent": [f"enc_{i}", 0]}}
    last = len(refs) - 1
    p["guider"]["inputs"]["positive"] = [f"pos_{last}", 0]
    p["guider"]["inputs"]["negative"] = [f"neg_{last}", 0]

    floor, temp = PRESETS[lock]
    p["ift"]["inputs"].update(preset=lock, similarity_floor=floor,
                              softmax_temperature=temp,
                              reference_index=0, reference_indices=identity_refs)
    p["txt"]["inputs"]["text"] = prompt
    p["noise"]["inputs"]["noise_seed"] = seed
    p["sched"]["inputs"].update(width=width, height=height)
    p["latent"]["inputs"].update(width=width, height=height)
    p["save"]["inputs"]["filename_prefix"] = f"charmed2/{prefix}"
    return payload


def queue(payload):
    (HERE / "last_payload.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    req = urllib.request.Request(API + "/prompt", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)["prompt_id"]


def wait(prompt_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(f"{API}/history/{prompt_id}") as r:
            h = json.load(r)
        if prompt_id in h:
            entry = h[prompt_id]
            if entry["status"]["status_str"] != "success":
                sys.exit(f"render failed: {json.dumps(entry['status'], indent=1)}")
            img = next(o for out in entry["outputs"].values() for o in out.get("images", []))
            return COMFY / "output" / img["subfolder"] / img["filename"]
        time.sleep(5)
    sys.exit(f"timed out waiting for {prompt_id}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True,
                    help="reference image path(s), comma-separated; first = primary identity")
    ap.add_argument("--prompt", required=True, help="full positive prompt text")
    ap.add_argument("--out", required=True, help="output name (no extension)")
    ap.add_argument("--seed", type=int, default=None,
                    help="noise seed (default: random, printed for reuse)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1536)
    ap.add_argument("--locks", default="SOFT_LOCK",
                    help="comma-separated identity presets: SOFT_LOCK,MID_LOCK,HARD_LOCK")
    ap.add_argument("--identity-refs", default="all",
                    help='which refs drive identity: "all", "0", "0,2", "0-1"')
    ap.add_argument("--finalize", action="store_true",
                    help="also matte each render to a 1920x1080 transparent webp")
    args = ap.parse_args()

    locks = [l.strip().upper() for l in args.locks.split(",") if l.strip()]
    for lock in locks:
        if lock not in PRESETS:
            sys.exit(f"unknown lock preset '{lock}' - valid: {', '.join(PRESETS)}")

    refs = stage_refs(args.ref)
    seed = args.seed if args.seed is not None else random.randint(1, 999_999_999_999)

    for lock in locks:
        # A single lock keeps the plain name; multiple locks tag filenames.
        name = args.out if len(locks) == 1 else f"{args.out}_{lock.lower().replace('_lock', '')}"
        pid = queue(build_payload(refs, args.prompt, seed, args.width, args.height,
                                  lock, args.identity_refs, name))
        print(f"queued {lock} -> {pid} (seed {seed})")
        png = wait(pid)
        print(f"{lock} rendered -> {png}")
        if args.finalize:
            from finalize import finalize
            print("matted ->", finalize(str(png), name))


if __name__ == "__main__":
    main()
