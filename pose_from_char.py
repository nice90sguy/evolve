"""Render pose/expression variants of a character from reference images via
the ComfyUI API: single- or multi-reference identity transfer (FLUX.2 Klein +
IdentityFeatureTransferFinal) plus matting to a transparent webp.

Assumes the ComfyUI service is up (http://127.0.0.1:8188) and runs in the
ComfyUI venv (D:\\projects\\ComfyUI\\.venv) — see requirements.txt.

Examples:
  python pose_from_char.py --ref char.webp --prompt "..." --out-dir chars/anita
  python pose_from_char.py --ref "primary.webp,hand.webp" --identity-refs 0 ^
      --locks SOFT_LOCK,MID_LOCK --seed 960 --prompt "..." --num-images 4

Defaults are tuned for making drop-in variants of an existing sprite:
- Refs with an alpha channel are flattened onto solid WHITE at staging, and
  the prompt is prefixed with a white-background instruction. Feeding a
  transparent sprite raw makes the model paint checkerboard "transparency"
  blocks around the character. --no-flatten-alpha disables both.
- --width/--height default to the primary ref's dimensions. The Flux2 latent
  needs multiples of 16, so the render is done at snapped dims and resized
  back to the exact target, keeping render and finalized dims identical.
- Matting runs by default (--no-finalize for the raw render only) and keeps
  the render's dims: no trim/rescale/recenter, so the variant lands exactly
  where the character sits in the ref.

Renders go into --out-dir (default: current directory — launch from your
game's project root), named <prefix>NNNNN.png: canonical 5-digit zero-pad,
--out-prefix default "img" (img00032.png). Numbering continues past existing
files, NOTHING is ever overwritten. --num-images renders a batch in one call
(latent batch size); multiple --locks tag the prefix (img_soft00001.png).
Missing directories are created. ComfyUI renders into a scratch prefix
(output\\akasutils_scratch) and the finished files are copied out (copied,
not moved: a repeat-seed rerun is a ComfyUI cache hit that points back at
the earlier scratch file).

The FIRST --ref image is the primary identity reference (reference_index 0);
later refs add supporting views or gesture/pose steering. Multiple --locks
render once per preset on the same seed, tagged _soft/_mid/_hard.
Every run writes the exact submitted graph to last_payload.json.
"""
import argparse
import json
import random
import re
import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path

from PIL import Image

# utf-8 + line-buffered stdout so redirects and `| tee` get lines immediately
# (Windows pipes default to cp1252 and block-buffering).
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

HERE = Path(__file__).resolve().parent
COMFY = HERE.parent                      # akasutils lives inside the ComfyUI dir
API = "http://127.0.0.1:8188"
TEMPLATE = HERE / "template_klein.json"
SCRATCH_PREFIX = "akasutils_scratch"     # subfolder of ComfyUI's output dir; copied out after render
WHITE_BG_PREFIX = "Solid plain white background. "

# preset -> (similarity_floor, softmax_temperature), from
# IdentityFeatureTransferFinal.PRESETS in ComfyUI-Flux2Klein-Enhancer.
PRESETS = {
    "SOFT_LOCK": (0.5, 0.07),
    "MID_LOCK": (0.2, 0.07),
    "HARD_LOCK": (0.04, 0.025),
}


def stage_refs(ref_arg, flatten_alpha):
    """Stage refs into ComfyUI's input dir, flattening any alpha channel onto
    solid white (unless disabled). Returns (staged basenames, primary WxH)."""
    names, primary_size = [], None
    for raw in ref_arg.split(","):
        p = Path(raw.strip()).resolve()
        img = Image.open(p)
        if primary_size is None:
            primary_size = img.size
        has_alpha = img.mode in ("RGBA", "LA") or \
            (img.mode == "P" and "transparency" in img.info)
        if flatten_alpha and has_alpha:
            rgba = img.convert("RGBA")
            flat = Image.new("RGB", rgba.size, (255, 255, 255))
            flat.paste(rgba, mask=rgba.getchannel("A"))
            name = p.stem + "_white.png"
            flat.save(COMFY / "input" / name)
        else:
            name = p.name
            dest = COMFY / "input" / name
            if p != dest:
                shutil.copy2(p, dest)
        names.append(name)
    return names, primary_size


def snap16(v):
    """Nearest multiple of 16 (EmptyFlux2LatentImage floors height//16)."""
    return max(16, round(v / 16) * 16)


def build_payload(refs, prompt, seed, width, height, lock, identity_refs, prefix,
                  batch_size=1):
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
    p["latent"]["inputs"].update(width=width, height=height, batch_size=batch_size)
    p["save"]["inputs"]["filename_prefix"] = f"{SCRATCH_PREFIX}/{prefix}"
    return payload


def free_vram():
    """Ask ComfyUI to unload cached models and free VRAM. Preflight for
    heavyweight jobs (tween's two 14B experts): a Klein model left resident
    from an earlier evolve/pose session eats the headroom the expert swap
    needs. Klein-only scripts should NOT call this - the cached model is
    what makes their re-renders take seconds."""
    req = urllib.request.Request(
        API + "/free",
        json.dumps({"unload_models": True, "free_memory": True}).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        r.read()


def queue(payload):
    (HERE / "last_payload.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    req = urllib.request.Request(API + "/prompt", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)["prompt_id"]
    except urllib.error.HTTPError as e:
        # surface ComfyUI's validation report instead of a bare 400
        body = e.read().decode("utf-8", "replace")
        try:
            body = json.dumps(json.loads(body), indent=1)
        except Exception:
            pass
        sys.exit(f"ComfyUI rejected the graph (HTTP {e.code}):\n{body[:3000]}")


def _ws_progress(prompt_id, client_id, state, start):
    """Daemon thread: stream ComfyUI's websocket events for prompt_id and
    paint an in-place per-step progress bar. Progress events route to the
    client_id that queued the prompt. Best-effort - any failure just means
    no bar; the poll heartbeat in wait_entry still shows life."""
    try:
        import asyncio
        import aiohttp             # ComfyUI's own server dep, present in its venv
    except ImportError:
        return

    def bar(v, m, node):
        fill = "#" * (24 * v // m)
        print(f"\r  {node or 'step'} {v}/{m} [{fill:<24}] "
              f"{int(time.time() - start)}s\x1b[K", end="", flush=True)
        state["last"], state["bar"] = time.time(), True

    async def run():
        async with aiohttp.ClientSession() as sess:
            url = API.replace("http", "ws", 1) + f"/ws?clientId={client_id}"
            async with sess.ws_connect(url, heartbeat=30) as ws:
                while not state["done"]:
                    try:
                        msg = await ws.receive(timeout=5)
                    except asyncio.TimeoutError:
                        continue
                    if msg.type in (aiohttp.WSMsgType.CLOSED,
                                    aiohttp.WSMsgType.CLOSING,
                                    aiohttp.WSMsgType.ERROR):
                        return
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    ev = json.loads(msg.data)
                    d = ev.get("data", {})
                    # prompt_id None = watch everything (see watch.py)
                    if prompt_id is not None and \
                            d.get("prompt_id") not in (None, prompt_id):
                        continue
                    if ev.get("type") == "progress" and d.get("max"):
                        bar(d["value"], d["max"], d.get("node"))
                    elif ev.get("type") == "progress_state":   # newer protocol
                        for nid, nd in d.get("nodes", {}).items():
                            if nd.get("state") == "running" and nd.get("max", 0) > 1:
                                bar(nd.get("value", 0), nd["max"], nid)

    try:
        asyncio.run(run())
    except Exception:
        pass


def wait_entry(prompt_id, timeout=300, client_id="akasutils"):
    """Poll history until the prompt completes, showing live progress the
    whole time: per-step sampler bars from the websocket, plus an elapsed
    heartbeat whenever nothing has printed for 15s (model loads emit no
    events). Returns the raw history entry."""
    start = time.time()
    state = {"last": start, "bar": False, "done": False}
    threading.Thread(target=_ws_progress,
                     args=(prompt_id, client_id, state, start),
                     daemon=True).start()
    try:
        deadline = start + timeout
        while time.time() < deadline:
            with urllib.request.urlopen(f"{API}/history/{prompt_id}") as r:
                h = json.load(r)
            if prompt_id in h:
                entry = h[prompt_id]
                if state["bar"]:
                    print(flush=True)
                if entry["status"]["status_str"] != "success":
                    sys.exit(f"render failed: {json.dumps(entry['status'], indent=1)}")
                return entry
            if time.time() - state["last"] > 15:
                print(f"\r  ... {int(time.time() - start)}s elapsed "
                      f"(loading models / queued)\x1b[K", end="", flush=True)
                state["bar"] = True
            time.sleep(3)
        if state["bar"]:
            print(flush=True)
        sys.exit(f"timed out waiting for {prompt_id}")
    finally:
        state["done"] = True


def wait(prompt_id, timeout=300, client_id="akasutils"):
    """Wait for a render (with live progress); return output image paths."""
    entry = wait_entry(prompt_id, timeout, client_id)
    imgs = [o for out in entry["outputs"].values() for o in out.get("images", [])]
    return [COMFY / "output" / o["subfolder"] / o["filename"] for o in imgs]


def next_index(out_dir, prefix):
    """First free canonical index: existing <prefix>NNNNN.png files are never
    overwritten, numbering continues after the highest one."""
    pat = re.compile(re.escape(prefix) + r"(\d{5})\.png$")
    taken = [int(m.group(1)) for p in out_dir.glob(f"{prefix}*.png")
             if (m := pat.fullmatch(p.name))]
    return max(taken, default=0) + 1


def save_render(src, out_dir, prefix, index, size=None):
    """Copy a scratch render to out_dir/<prefix><index:05d>.png. Copy, not
    move: resubmitting an identical graph (repeat seed + prompt) is a full
    ComfyUI cache hit whose history points at the PREVIOUS run's scratch
    file — it must still be there. If the render was snapped to /16, resize
    the copy back to the exact target size."""
    dest = out_dir / f"{prefix}{index:05d}.png"
    img = Image.open(src)
    if size and img.size != size:
        img.resize(size, Image.LANCZOS).save(dest)
    else:
        shutil.copy2(str(src), str(dest))
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True,
                    help="reference image path(s), comma-separated; first = primary identity")
    ap.add_argument("--prompt", required=True, help="full positive prompt text")
    ap.add_argument("--out-dir", default=".",
                    help="directory renders are written into (default: current directory)")
    ap.add_argument("--out-prefix", default="img",
                    help='render filename prefix; files are <prefix>NNNNN.png (default "img"); '
                         "multiple --locks tag the prefix (img_soft00001.png)")
    ap.add_argument("--num-images", type=int, default=1,
                    help="images per render call (latent batch size)")
    ap.add_argument("--seed", type=int, default=None,
                    help="noise seed (default: random, printed for reuse)")
    ap.add_argument("--width", type=int, default=None,
                    help="render width (default: primary ref's width)")
    ap.add_argument("--height", type=int, default=None,
                    help="render height (default: primary ref's height)")
    ap.add_argument("--locks", default="SOFT_LOCK",
                    help="comma-separated identity presets: SOFT_LOCK,MID_LOCK,HARD_LOCK")
    ap.add_argument("--identity-refs", default="all",
                    help='which refs drive identity: "all", "0", "0,2", "0-1"')
    ap.add_argument("--flatten-alpha", default=True, action=argparse.BooleanOptionalAction,
                    help="flatten transparent refs onto white and prefix the prompt "
                         "with a white-background instruction")
    ap.add_argument("--finalize", default=True, action=argparse.BooleanOptionalAction,
                    help="matte each render to a transparent webp at the same dims")
    args = ap.parse_args()

    locks = [l.strip().upper() for l in args.locks.split(",") if l.strip()]
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
        # A single lock keeps the plain prefix; multiple locks tag it.
        prefix = args.out_prefix if len(locks) == 1 \
            else f"{args.out_prefix}_{lock.lower().replace('_lock', '')}"
        pid = queue(build_payload(refs, prompt, seed, snap16(width), snap16(height),
                                  lock, args.identity_refs, prefix, args.num_images))
        print(f"queued {lock} -> {pid} (seed {seed}, {width}x{height}, "
              f"{args.num_images} image(s))")
        srcs = wait(pid)
        out_dir.mkdir(parents=True, exist_ok=True)
        n = next_index(out_dir, prefix)
        for src in srcs:
            png = save_render(src, out_dir, prefix, n, (width, height))
            n += 1
            print(f"{lock} rendered -> {png}")
            if args.finalize:
                from finalize import finalize
                print("matted ->", finalize(str(png), png.stem))


if __name__ == "__main__":
    main()
