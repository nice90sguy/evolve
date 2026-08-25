"""evolve_v0.py - the ROWS-AS-GENERATIONS prototype, retired 2026-08-21
(superseded by evolve.py, the Evolver). Kept runnable for browsing the old
tree roots (tierney/, matilda/).

Evolutionary character design: generate -> human-select -> refine, with
genealogy as the directory tree.

Usage:  python evolve.py [--root evolve] [--port 8189]

Serves a single-page web UI (aiohttp, no build step) and drives ComfyUI
(must be UP at 127.0.0.1:8188 - launch it yourself; this script never
does). Generation happens ONLY on user clicks.

Disk model:
- A row/"generation" = one directory of siblings. Its path IS the
  genealogy: /4/37/2 = children of image 2.png inside /4/37. Images are
  bare numbers; numbering per directory continues forever.
- VERSIONS: re-mothering an image whose offspring dir is already taken
  auto-mints the next sibling dir with a ".N" suffix (4/37 -> 4/37.2;
  a new rootless generation -> .2 at the root). No version UI - the
  path shows it.
- Prompt is always EDITABLE; "1 row = 1 prompt" is kept by the re-roll
  rule: generating with a CHANGED prompt silently clears the row's
  stale images first (refused if any image has descendant generations),
  while an unchanged prompt appends variants. Numbering is never
  reused, even across a wipe (meta history keeps the high-water mark).
  The mother and extras stay fixed by the row's position/first batch.
- meta.jsonl per dir: one line per batch (full recipe, per-image seeds).

Engine: FLUX.2 Klein. User's axes: VERTICAL = "the reference image, but
<prompt delta>" - the mother is conditioned at EVERY step (ReferenceLatent
+ IdentityFeatureTransferFinal, reference_indices="0" so identity always
comes from the mother; lock dropdown, NONE = refs without identity
transfer). HORIZONTAL = the `vary` slider: per-sibling seeded randn
injected into the reference LATENT (KJNodes InjectNoiseToLatent) - each
sibling sees a slightly different mother. Gen 0 = prompt -> empty latent.
Each image is its own submission with seed base+i (tooltip-reproducible).
cfg stays 1.0 (distilled Klein burns above it).
Failed-approach history (do not re-invent): same-prompt ref spawns clone
(2.5/255); img2img drift anchors harder; windowed (timestep-ranged) refs
break vertical - a ref-free early phase + delta prompt renders the
delta's genre prior.

UI: rows top-to-bottom in first-spawn order; each row = control box
(path, mother thumbnail + reserved extra-ref slots, prompt, "+ generate",
row-delete) and a horizontal carousel. The bottom row is the WORKBENCH -
the next, not-yet-generated generation: double-click ANY image to make it
the workbench's mother (silently replacing), write a prompt, "+ generate".
Single-click selects an image (sole yellow border): the right PREVIEW
panel (draggable divider) locks to it - unselected, it live-previews
whatever is hovered. "Highlight Family Tree" dims everything except the
selected image's bloodline (minimal scrolling, only if a lineage image is
partially hidden). "Hide non-relatives" filters rows to ancestors + own
row + descendants. Esc deselects. The only per-image op is delete, shown
on the selected image.
"""
import argparse
import asyncio
import hashlib
import io
import json
import random
import re
import shutil
import time
from pathlib import Path

from aiohttp import web
from PIL import Image

from pose_from_char import (COMFY, PRESETS, SCRATCH_PREFIX, TEMPLATE,
                            WHITE_BG_PREFIX, queue, snap16, wait)

ROOT = Path("evolve")
ASSETS = Path("assets")               # dropped/pasted extra refs, project-root-relative
REF_STAGE = "evolve_ref.png"          # staged mother in ComfyUI's input dir
VER_RE = re.compile(r"^(.*?)\.(\d+)$")   # "37.2" -> base "37", version 2
NUM_PNG = re.compile(r"^\d+\.png$")


# ---------- tree helpers ----------

def safe_rel(rel):
    """Resolve a root-relative path, refusing traversal outside ROOT."""
    p = (ROOT / rel).resolve() if rel else ROOT
    if not str(p).startswith(str(ROOT)):
        raise web.HTTPForbidden(text="path outside root")
    return p


def split_version(dir_rel):
    """'4/37.2' -> ('4/37', 2); '4/37' -> ('4/37', 1); '.2' -> ('', 2)."""
    parts = dir_rel.split("/") if dir_rel else [""]
    m = VER_RE.match(parts[-1])
    if m and (m.group(1) or parts[-1].startswith(".")):
        base = m.group(1)
        pos = "/".join(parts[:-1] + [base]) if base else "/".join(parts[:-1])
        return pos, int(m.group(2))
    return dir_rel, 1


def version_dir(base, ver):
    if ver == 1:
        return base
    if not base:
        return f".{ver}"
    parts = base.split("/")
    parts[-1] = f"{parts[-1]}.{ver}"
    return "/".join(parts)


def parent_image(vdir):
    """The mother image a generation dir descends from; None for gen 0."""
    pos, _ = split_version(vdir)
    return pos + ".png" if pos else None


def read_version(vdir):
    d = safe_rel(vdir)
    images = []
    if d.is_dir():
        images = sorted((p.name for p in d.iterdir()
                         if p.is_file() and NUM_PNG.match(p.name)),
                        key=lambda n: int(n[:-4]))
    batches = []
    mf = d / "meta.jsonl"
    if mf.exists():
        batches = [json.loads(l) for l in
                   mf.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {"dir": vdir, "images": images, "batches": batches,
            "mother": parent_image(vdir),
            "prompt": batches[-1]["prompt"] if batches else None,
            "whitebg": batches[-1]["whitebg"] if batches else None,
            "extras": (batches[-1].get("extras") or []) if batches else [],
            "lora": batches[-1].get("lora") if batches else None,
            "lora_strength": batches[-1].get("lora_strength") if batches else None,
            "ts": batches[0]["ts"] if batches else None}


def scan_log():
    """Every generation dir as a row, root first, then first-spawn order."""
    dirs = [""]
    for p in sorted(safe_rel("").rglob("*")):
        if p.is_dir():
            dirs.append(str(p.relative_to(ROOT)).replace("\\", "/"))
    rows = [read_version(d) for d in dirs]
    root = [r for r in rows if r["dir"] == ""]
    rest = sorted((r for r in rows if r["dir"] != ""),
                  key=lambda r: r["ts"] or "9999")
    return root + rest


def next_free_vdir(base):
    """First version dir of `base` that has no images and no history."""
    v = 1
    while True:
        vd = version_dir(base, v)
        st = read_version(vd)
        if not st["images"] and not st["batches"]:
            return vd
        v += 1


def next_num(d):
    nums = [int(p.name[:-4]) for p in d.iterdir()
            if p.is_file() and NUM_PNG.match(p.name)] if d.is_dir() else []
    return max(nums, default=0) + 1


# ---------- generation ----------

def build_graph(prompt, seed, width, height, batch, refs, lock, vary,
                lora_path=None, lora_strength=1.0):
    """refs: staged input-dir filenames; refs[0] = mother (identity source,
    jittered by vary), later refs = extras (composition/gesture steering).
    lora_path: optional ComfyUI-format LoRA (a GENETIC IDENTITY) applied to
    the model before identity transfer; may be None (the empty adaptor)."""
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    p = payload["prompt"]
    for old in ("img_ref", "enc_ref", "pos_ref", "neg_ref"):
        p.pop(old, None)

    model_src = "unet"
    if lora_path:
        p["gene"] = {"class_type": "ApplyTrainedLora",
                     "inputs": {"strength": lora_strength, "model": ["unet", 0],
                                "lora_path": str(lora_path)}}
        model_src = "gene"

    pos_src, neg_src = "txt", "zero"
    for i, name in enumerate(refs):
        p[f"img_{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        p[f"enc_{i}"] = {"class_type": "VAEEncode",
                         "inputs": {"pixels": [f"img_{i}", 0], "vae": ["vae", 0]}}
        lat = f"enc_{i}"
        if i == 0 and vary > 0:
            # HORIZONTAL: jitter the MOTHER's latent per sibling (enc_0
            # doubles as the strength-0 noise input for shape safety).
            p["jit"] = {"class_type": "InjectNoiseToLatent",
                        "inputs": {"latents": ["enc_0", 0], "strength": 0.0,
                                   "noise": ["enc_0", 0], "normalize": False,
                                   "average": False, "mix_randn_amount": vary,
                                   "seed": seed}}
            lat = "jit"
        p[f"pos_{i}"] = {"class_type": "ReferenceLatent",
                         "inputs": {"conditioning": [pos_src, 0], "latent": [lat, 0]}}
        p[f"neg_{i}"] = {"class_type": "ReferenceLatent",
                         "inputs": {"conditioning": [neg_src, 0], "latent": [lat, 0]}}
        pos_src, neg_src = f"pos_{i}", f"neg_{i}"
    p["guider"]["inputs"]["positive"] = [pos_src, 0]
    p["guider"]["inputs"]["negative"] = [neg_src, 0]

    if refs and lock in PRESETS:
        floor, temp = PRESETS[lock]
        p["ift"]["inputs"].update(model=[model_src, 0], preset=lock,
                                  similarity_floor=floor,
                                  softmax_temperature=temp,
                                  reference_index=0, reference_indices="0")
    else:
        p.pop("ift", None)
        p["guider"]["inputs"]["model"] = [model_src, 0]

    p["sched"]["inputs"].update(width=width, height=height)
    p["latent"]["inputs"].update(width=width, height=height, batch_size=batch)
    p["txt"]["inputs"]["text"] = prompt
    p["noise"]["inputs"]["noise_seed"] = seed
    p["save"]["inputs"]["filename_prefix"] = f"{SCRATCH_PREFIX}/evolve"
    return payload


def do_spawn(vdir, prompt, n, width, height, seed, lock, vary, whitebg, extras,
             lora=None, lora_strength=1.0):
    """Blocking: render one batch into a generation dir (created if needed)."""
    existing = read_version(vdir)
    if existing["images"] and existing["prompt"] is not None:
        if prompt.strip() == existing["prompt"].strip():
            prompt = existing["prompt"]       # unchanged: append variants
        else:
            # DIRTY prompt: the row is being re-rolled. Silently clear its
            # stale images (1 row = 1 prompt invariant) - unless any image
            # has descendant generations, which must be pruned explicitly.
            d = safe_rel(vdir)
            blockers = []
            for name in existing["images"]:
                stem = name[:-len(".png")]
                kids = [k for k in [d / stem] if k.is_dir()]
                kids += [k for k in d.glob(stem + ".*") if k.is_dir()]
                if kids:
                    blockers.append(name)
            if blockers:
                raise ValueError(
                    "prompt changed, but these images have descendant "
                    "generations: " + ", ".join(blockers)
                    + " - delete those lineages first (or breed a new version)")
            for name in existing["images"]:
                (d / name).unlink(missing_ok=True)
        if extras is None:
            extras = existing["extras"]
    elif extras is None:                      # emptied row rerun keeps recipe
        extras = existing["extras"]
    # NB lora, lock, vary, seed AND whitebg are per-batch knobs, recorded
    # in meta but applied fresh each "+" (whitebg's default is inherited
    # from the mother's generation client-side).
    if not prompt.strip():
        raise ValueError("prompt is empty")

    lora_abs = None
    if lora:
        lroot = Path("loras").resolve()
        lp = (lroot / lora).resolve()
        if not (str(lp).startswith(str(lroot)) and lp.is_file()):
            raise FileNotFoundError(f"LoRA not found under .\\loras: {lora}")
        lora_abs = lp

    mother = parent_image(vdir)
    refs = []
    if mother:
        src = safe_rel(mother)
        if not src.is_file():
            raise FileNotFoundError(f"mother image missing: {mother}")
        shutil.copy2(src, COMFY / "input" / REF_STAGE)
        refs.append(REF_STAGE)
        for k, aname in enumerate((extras or [])[:3]):
            af = ASSETS / Path(aname).name
            if not af.is_file():
                raise FileNotFoundError(f"asset missing: {aname}")
            staged = f"evolve_extra{k + 1}.png"
            shutil.copy2(af, COMFY / "input" / staged)
            refs.append(staged)
        vary = min(1.0, max(0.0, vary))
    else:
        vary, extras = 0.0, []

    out_dir = safe_rel(vdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = snap16(width), snap16(height)
    full_prompt = (WHITE_BG_PREFIX + prompt) if whitebg else prompt

    seeds = [seed + i for i in range(n)]
    pids = [queue(build_graph(full_prompt, s, width, height, 1,
                              refs, lock, vary, lora_abs, lora_strength))
            for s in seeds]
    srcs = [wait(pid, timeout=600)[0] for pid in pids]

    # numbering is never reused, even across a dirty-prompt wipe: meta
    # history keeps the high-water mark
    hist = max((int(im[:-4]) for b in existing["batches"]
                for im in b["images"]), default=0)
    i = max(next_num(out_dir), hist + 1)
    names = []
    for s in srcs:
        name = f"{i}.png"
        shutil.copy2(s, out_dir / name)
        names.append(name)
        i += 1
    batch = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "prompt": prompt,
             "whitebg": whitebg, "seed": seed, "seeds": seeds,
             "lock": lock if refs else None,
             "vary": vary if refs else None,
             "extras": [Path(a).name for a in (extras or [])],
             "lora": lora if lora_abs else None,
             "lora_strength": lora_strength if lora_abs else None,
             "width": width, "height": height, "n": n,
             "ancestor": mother, "images": names}
    with open(out_dir / "meta.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(batch) + "\n")
    return {**batch, "vdir": vdir}


# ---------- http ----------

async def api_log(request):
    return web.json_response(scan_log())


async def api_loras(request):
    """GENETIC IDENTITIES available to the generate graph: ComfyUI-format
    LoRAs under .\\loras (the char_lora_* output convention)."""
    root = Path("loras")
    found = set()
    if root.is_dir():
        for p in root.rglob("*_comfy.safetensors"):
            found.add(str(p.relative_to(root)).replace("\\", "/"))
        for p in root.glob("*.safetensors"):
            found.add(p.name)
    return web.json_response(sorted(found))


async def api_spawn(request):
    b = await request.json()
    seed = b.get("seed") or random.randint(1, 999_999_999_999)
    vdir = b.get("vdir")
    if vdir is None:      # workbench: derive target from the mother image
        mother = b.get("mother")
        base = mother[:-len(".png")] if mother else ""
        vdir = next_free_vdir(base)
    try:
        batch = await asyncio.get_event_loop().run_in_executor(
            None, do_spawn, vdir, b.get("prompt", ""),
            int(b.get("n", 8)), int(b.get("width", 1024)),
            int(b.get("height", 1024)), int(seed),
            b.get("lock", "SOFT_LOCK"), float(b.get("vary", 0.0)),
            bool(b.get("whitebg", True)),
            b.get("extras"),
            b.get("lora") or None,
            float(b.get("lora_strength", 1.0)))
    except SystemExit as e:        # queue()/wait() sys.exit on ComfyUI errors
        return web.json_response({"error": str(e)}, status=500)
    except Exception as e:
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)
    return web.json_response(batch)


async def api_delete(request):
    b = await request.json()
    rel = b["path"]
    f = safe_rel(rel)
    if not (f.is_file() and NUM_PNG.match(f.name)):
        return web.json_response({"error": "not a candidate image"}, status=400)
    stem = rel[:-len(".png")]
    kids = [d for d in [safe_rel(stem)] if d.is_dir()]
    kids += [d for d in f.parent.glob(f.name[:-4] + ".*") if d.is_dir()]
    if kids:
        return web.json_response(
            {"error": "image has descendant generations - delete those first"},
            status=409)
    f.unlink()
    return web.json_response({"deleted": rel})


async def api_delete_row(request):
    b = await request.json()
    vdir = b["vdir"]
    if not vdir:
        return web.json_response({"error": "cannot delete the root row"}, status=400)
    d = safe_rel(vdir)
    if not d.is_dir():
        return web.json_response({"error": "no such generation"}, status=404)
    shutil.rmtree(d)
    return web.json_response({"deleted": vdir})


async def serve_image(request):
    f = safe_rel(request.match_info["path"])
    if not (f.is_file() and f.suffix == ".png"):
        raise web.HTTPNotFound()
    return web.FileResponse(f)


async def api_upload(request):
    """Persist a dropped/pasted image into assets/ (deduped by content hash,
    flattened onto white if it carries alpha - raw transparency teaches the
    model checkerboards; see CLAUDE.md matting lore)."""
    data = await request.read()
    if not data:
        return web.json_response({"error": "empty upload"}, status=400)
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return web.json_response({"error": "not a readable image"}, status=400)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
        img = flat
    else:
        img = img.convert("RGB")
    name = f"{hashlib.sha1(data).hexdigest()[:12]}.png"
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / name
    if not out.exists():
        img.save(out, "PNG")
    return web.json_response({"asset": name})


async def serve_asset(request):
    name = Path(request.match_info["name"]).name    # basename only
    f = (ASSETS / name).resolve()
    if not (str(f).startswith(str(ASSETS.resolve())) and f.is_file()
            and f.suffix == ".png"):
        raise web.HTTPNotFound()
    return web.FileResponse(f)


async def index(request):
    return web.Response(text=PAGE, content_type="text/html")


PAGE = r"""<!doctype html>
<meta charset="utf-8"><title>evolve</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; font:14px system-ui,sans-serif; background:#16161a; color:#ddd;
         height:100vh; display:flex; flex-direction:column; overflow:hidden; }
  #bar { background:#202028; padding:8px 14px; display:flex; gap:14px;
         align-items:center; flex-wrap:wrap; border-bottom:1px solid #333; }
  .ctl { display:flex; flex-direction:column; font-size:11px; color:#999; gap:2px; }
  .ctl input, .ctl select { background:#16161a; color:#ddd; border:1px solid #444;
         border-radius:4px; padding:3px 6px; width:80px; font:inherit; }
  .chk { font-size:12px; color:#aaa; display:flex; gap:5px; align-items:center; }
  #vary { width:110px; }
  #status { flex:1; font-size:12px; color:#8fa; min-height:15px; text-align:right; }
  #main { flex:1; display:flex; min-height:0; }
  #rows { flex:1; overflow-y:auto; min-width:0; }
  #divider { flex:0 0 6px; cursor:col-resize; background:#26262c; }
  #divider:hover { background:#3b6ea5; }
  #preview { flex:0 0 var(--pw, 360px); background:#101014; display:flex;
             align-items:center; justify-content:center; overflow:hidden; }
  #preview img { max-width:100%; max-height:100%; object-fit:contain; }
  .row { display:flex; gap:12px; padding:10px 14px; border-bottom:1px solid #26262c;
         align-items:stretch; }
  .row.bench { background:#1b1b22; }
  .row.bench.dropping { outline:2px dashed #3b6ea5; outline-offset:-2px; }
  .info { flex:0 0 250px; display:flex; flex-direction:column; gap:7px; padding:6px; }
  .pathrow { display:flex; align-items:center; gap:6px; }
  .path { flex:1; font:13px ui-monospace,monospace; color:#7ab; word-break:break-all; }
  .rowdel { background:none; border:0; color:#666; cursor:pointer; font-size:13px;
            padding:2px 5px; border-radius:4px; }
  .rowdel:hover { color:#fff; background:#a53b3b; }
  .refslots { display:flex; gap:5px; }
  .slot { width:52px; height:52px; border:1px dashed #3a3a42; border-radius:6px;
          display:flex; align-items:center; justify-content:center; overflow:hidden;
          color:#444; font-size:10px; }
  .slot img { width:100%; height:100%; object-fit:cover; }
  .slot.reserved { opacity:.35; }
  .prompt { flex:1; font-size:12.5px; color:#bbb; overflow-y:auto; max-height:120px;
            white-space:pre-wrap; }
  .prompt textarea { width:100%; height:100%; min-height:64px; background:#16161a;
          color:#ddd; border:1px solid #555; border-radius:6px; padding:6px;
          font:inherit; }
  .plusbtn { background:#3b6ea5; color:#fff; border:0; border-radius:6px;
          padding:6px 0; font:inherit; cursor:pointer; }
  .plusbtn:disabled { background:#333; color:#777; cursor:default; }
  .stripwrap { flex:1; display:flex; align-items:center; gap:4px; min-width:0;
           min-height:212px; }
  .edge { flex:0 0 26px; height:70px; background:#26262c; color:#888; border:0;
          border-radius:6px; cursor:pointer; font-size:16px; }
  .edge:hover { color:#fff; }
  .strip { display:flex; gap:8px; overflow-x:auto; scroll-behavior:smooth;
           scrollbar-width:none; flex:1; align-items:flex-start;
           min-height:204px; scroll-snap-type:x mandatory; }
  .strip::-webkit-scrollbar { display:none; }
  .cell { position:relative; flex:0 0 auto; scroll-snap-align:start; }
  .cell img { height:200px; border-radius:6px; cursor:pointer; display:block;
              transition:opacity .18s; }
  .cell.dim img { opacity:.15; }
  .cell.sel img { outline:3px solid #e8b339; outline-offset:-3px; }
  .ops { position:absolute; top:4px; right:4px; display:none; }
  .cell.sel .ops { display:flex; }
  .ops button { border:0; border-radius:4px; padding:3px 8px; cursor:pointer;
          font-size:12px; background:#000a; color:#fff; }
  .ops button:hover { background:#a53b3b; }
  .hint { color:#555; font-size:12px; font-style:italic; align-self:center; }
</style>
<div id="bar">
  <label class="ctl">N <input id="n" type="number" value="8" min="1" max="32"></label>
  <label class="ctl">W <input id="w" type="number" value="1024" step="16"></label>
  <label class="ctl">H <input id="h" type="number" value="1024" step="16"></label>
  <label class="ctl">seed <input id="seed" placeholder="random"></label>
  <label class="ctl">lock <select id="lock">
    <option>SOFT_LOCK</option><option>MID_LOCK</option>
    <option>HARD_LOCK</option><option>NONE</option></select></label>
  <label class="ctl" title="per-sibling jitter of the mother's latent: 0 = exact reference, higher = more sibling variety, softer identity">
    vary <span id="varyv">0</span>
    <input id="vary" type="range" min="0" max="0.5" step="0.02" value="0"></label>
  <label class="ctl">lora <select id="lora" style="width:150px"><option value="">(none)</option></select></label>
  <label class="ctl">lora str <input id="lorastr" type="number" value="1.0" step="0.05" style="width:60px"></label>
  <label class="chk"><input id="hltree" type="checkbox" checked> highlight family tree</label>
  <label class="chk"><input id="hideothers" type="checkbox"> hide non-relatives</label>
  <div id="status"></div>
</div>
<div id="main">
  <div id="rows"></div>
  <div id="divider"></div>
  <div id="preview"><img></div>
</div>
<script>
const $ = id => document.getElementById(id);
$("vary").oninput = () => $("varyv").textContent = (+$("vary").value).toFixed(2);

let log = [];                                   // rows from server
let bench = {mother: null, prompt: "", extras: []};   // the workbench (next gen)
let selImg = null;                              // selected image rel path
let busy = false;

const disp = d => "/" + d;
const dirOf = rel => rel.includes("/") ? rel.slice(0, rel.lastIndexOf("/")) : "";
const stemOf = rel => rel.slice(0, -4);
const parseSeg = s => {
  const m = s.match(/^(.*?)\.(\d+)$/);
  const versioned = m && (m[1] || s.startsWith("."));
  return versioned ? m[1] : s;
};
function parentImage(vdir) {                    // mirrors the server
  if (!vdir) return null;
  const parts = vdir.split("/");
  const base = parseSeg(parts[parts.length-1]);
  if (base === "") return null;                 // root version (.N)
  return [...parts.slice(0,-1), base].join("/") + ".png";
}
function versionDir(base, v) {
  if (v === 1) return base;
  if (!base) return "." + v;
  const parts = base.split("/"); parts[parts.length-1] += "." + v;
  return parts.join("/");
}
function benchTarget() {                        // live auto-versioned path
  const base = bench.mother ? stemOf(bench.mother) : "";
  const taken = new Set(log.filter(r => r.images.length || r.batches.length)
                           .map(r => r.dir));
  for (let v = 1;; v++) {
    const vd = versionDir(base, v);
    if (!taken.has(vd)) return vd;
  }
}
function lineageOf(rel) {                       // image rel -> {dir: imageName}
  const out = {};
  let img = rel;
  while (img) {
    out[dirOf(img)] = img.slice(img.lastIndexOf("/") + 1);
    img = parentImage(dirOf(img));
  }
  return out;
}

async function refresh(keepScroll) {
  const y = $("rows").scrollTop;
  log = await (await fetch("/api/log")).json();
  try {
    const loras = await (await fetch("/api/loras")).json();
    const sel = $("lora"), cur = sel.value;
    sel.innerHTML = '<option value="">(none)</option>' +
      loras.map(l => `<option${l === cur ? " selected" : ""}>${l}</option>`).join("");
  } catch (e) {}
  render();
  if (keepScroll) $("rows").scrollTop = y;
}

function render() {
  const rows = $("rows");
  // preserve the container's vertical scroll AND every carousel's
  // horizontal position across rebuilds - a reflow must never move the
  // view out from under the user
  const yTop = rows.scrollTop;
  const scrolls = {};
  rows.querySelectorAll(".row").forEach(el => {
    const s = el.querySelector(".strip");
    if (s) scrolls[el.dataset.dir] = s.scrollLeft;
  });
  rows.innerHTML = "";
  const line = selImg ? lineageOf(selImg) : null;
  const hide = $("hideothers").checked && selImg;
  const selDir = selImg ? dirOf(selImg) : null;
  for (const r of log) {
    if (hide) {
      const stem = stemOf(selImg);
      const isAncestor = line[r.dir] !== undefined;   // incl. own row
      const isDesc = selDir === "" || r.dir.startsWith(selDir + "/") ||
                     r.dir === stem || r.dir.startsWith(stem + "/") ||
                     r.dir.startsWith(stem + ".");    // versions of offspring
      if (!isAncestor && !isDesc) continue;
    }
    rows.appendChild(renderRow(r, line));
  }
  rows.appendChild(renderBench());
  rows.querySelectorAll(".row").forEach(el => {
    const s = el.querySelector(".strip");
    if (s && scrolls[el.dataset.dir] != null) {
      s.style.scrollBehavior = "auto";
      s.scrollLeft = scrolls[el.dataset.dir];
      s.style.scrollBehavior = "";
    }
  });
  rows.scrollTop = yTop;
  if (line && $("hltree").checked) revealLineage(line);
  updatePreview();
}

function updateSelection() {
  // in-place: selection/dim/preview change WITHOUT re-rendering, so no
  // reflow and no scroll loss on click
  const hl = $("hltree").checked;
  const line = selImg ? lineageOf(selImg) : null;
  document.querySelectorAll("#rows .cell").forEach(c => {
    const rel = c.dataset.rel;
    const dir = dirOf(rel), name = rel.slice(rel.lastIndexOf("/") + 1);
    c.classList.toggle("sel", rel === selImg);
    c.classList.toggle("dim",
      !!(selImg && hl && line && rel !== selImg && line[dir] !== name));
  });
  if (line && hl) revealLineage(line);
  updatePreview();
}

function selectImage(rel) {
  selImg = (selImg === rel) ? null : rel;
  // hide-non-relatives changes the ROW SET on selection change - the one
  // case that must reflow; scroll preservation + reveal keep it in view
  if ($("hideothers").checked) {
    render();
    const t = document.querySelector(".cell.sel");
    if (t) t.scrollIntoView({block: "nearest", inline: "nearest"});
  } else updateSelection();
}

function replaceBench() {
  const be = document.querySelector('[data-dir="~bench"]');
  if (be) be.replaceWith(renderBench());
}

function renderRow(r, line) {
  const div = document.createElement("div"); div.className = "row";
  div.dataset.dir = r.dir;

  const info = document.createElement("div"); info.className = "info";
  const pathrow = document.createElement("div"); pathrow.className = "pathrow";
  const path = document.createElement("div"); path.className = "path";
  path.textContent = disp(r.dir);
  pathrow.appendChild(path);
  if (r.dir !== "") {
    const rx = document.createElement("button"); rx.className = "rowdel";
    rx.textContent = "✕"; rx.title = "delete this generation (and its descendants)";
    rx.onclick = async () => {
      if (!confirm("Delete generation " + disp(r.dir) +
          " and EVERYTHING descended from it?\nThis cannot be undone.")) return;
      const res = await (await fetch("/api/delete_row", {method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({vdir: r.dir})})).json();
      $("status").textContent = res.error || ("deleted " + disp(r.dir));
      if (!res.error) { selImg = null; refresh(true); }
    };
    pathrow.appendChild(rx);
  }
  info.appendChild(pathrow);
  info.appendChild(refSlots(r.mother, r.extras, false));
  if (r.lora) {
    const g = document.createElement("div");
    g.style.cssText = "font-size:11px;color:#8a8";
    g.textContent = "gene (last batch): " + r.lora + " @ " + r.lora_strength;
    info.appendChild(g);
  }

  const pr = document.createElement("div"); pr.className = "prompt";
  const ta = document.createElement("textarea");
  ta.placeholder = "prompt";
  ta.title = "editable: generating with a CHANGED prompt silently re-rolls "
           + "this row (clears its stale images); unchanged = adds variants";
  ta.value = r.prompt2 ?? r.prompt ?? "";
  ta.oninput = () => { r.prompt2 = ta.value; plus.disabled = busy || !ok(); };
  pr.appendChild(ta);
  info.appendChild(pr);
  const wb = whitebgCheck(false,            // per-batch knob: never frozen
    r.whitebg2 ?? r.whitebg ?? true,
    v => { r.whitebg2 = v; });
  info.appendChild(wb.el);

  const plus = document.createElement("button"); plus.className = "plusbtn";
  plus.textContent = "+ generate";
  const getPrompt = () => (r.prompt2 ?? r.prompt ?? "");
  const ok = () => (getPrompt() || "").trim().length > 0;
  plus.disabled = busy || !ok();
  plus.onclick = () => {
    if (!busy && ok()) spawn({vdir: r.dir}, getPrompt(), false, wb.get());
  };
  info.appendChild(plus);
  div.appendChild(info);

  div.appendChild(carousel(r, line));
  return div;
}

function renderBench() {
  const div = document.createElement("div"); div.className = "row bench";
  div.dataset.dir = "~bench";
  const info = document.createElement("div"); info.className = "info";

  const pathrow = document.createElement("div"); pathrow.className = "pathrow";
  const path = document.createElement("div"); path.className = "path";
  path.textContent = disp(benchTarget()) + "  (next)";
  pathrow.appendChild(path);
  info.appendChild(pathrow);
  info.appendChild(refSlots(bench.mother, bench.extras, true));

  const pr = document.createElement("div"); pr.className = "prompt";
  const ta = document.createElement("textarea");
  ta.placeholder = bench.mother
    ? "prompt: the mother image, but ..."
    : "prompt for a new gen-0 generation (no mother), or double-click an image to set one";
  ta.value = bench.prompt;
  ta.oninput = () => { bench.prompt = ta.value; plus.disabled = busy || !ok(); };
  pr.appendChild(ta);
  info.appendChild(pr);
  const wb = whitebgCheck(false, bench.whitebg ?? true,
    v => { bench.whitebg = v; });
  info.appendChild(wb.el);

  const plus = document.createElement("button"); plus.className = "plusbtn";
  plus.textContent = "+ generate";
  const ok = () => bench.prompt.trim().length > 0;
  plus.disabled = busy || !ok();
  plus.onclick = () => {
    if (busy || !ok()) return;
    spawn({mother: bench.mother, extras: bench.extras}, bench.prompt, true, wb.get());
  };
  info.appendChild(plus);
  div.appendChild(info);

  const wrap = document.createElement("div"); wrap.className = "stripwrap";
  const hint = document.createElement("div"); hint.className = "hint";
  hint.textContent = "double-click any image above to make it this generation's " +
    "mother - drop or paste image files for extra refs";
  wrap.appendChild(hint);
  div.appendChild(wrap);

  div.ondragover = e => { e.preventDefault(); div.classList.add("dropping"); };
  div.ondragleave = () => div.classList.remove("dropping");
  div.ondrop = e => {
    e.preventDefault(); div.classList.remove("dropping");
    addAssets([...e.dataTransfer.files]);
  };
  return div;
}

function whitebgCheck(frozen, initial, onChange) {
  const lab = document.createElement("label"); lab.className = "chk";
  const cb = document.createElement("input"); cb.type = "checkbox";
  cb.checked = !!initial; cb.disabled = frozen;
  cb.onchange = () => onChange(cb.checked);
  lab.append(cb, document.createTextNode(" white bg"));
  return {el: lab, get: () => cb.checked};
}

function refSlots(mother, extras, editable) {
  const slots = document.createElement("div"); slots.className = "refslots";
  const m = document.createElement("div"); m.className = "slot";
  if (mother) {
    const im = document.createElement("img");
    im.src = "/img/" + mother; im.title = "mother: " + disp(mother)
      + (editable ? "\ndouble-click to clear" : "");
    im.onmouseenter = () => previewSrc(im.src);
    im.onmouseleave = () => updatePreview();
    if (editable) im.ondblclick = () => { bench.mother = null; replaceBench(); };
    m.appendChild(im);
  } else m.textContent = editable ? "mother" : "gen 0";
  slots.appendChild(m);
  for (let i = 0; i < 3; i++) {
    const s = document.createElement("div"); s.className = "slot";
    const aname = (extras || [])[i];
    if (aname) {
      const im = document.createElement("img");
      im.src = "/asset/" + aname;
      im.title = "extra ref: " + aname + (editable ? "\ndouble-click to remove" : "");
      im.onmouseenter = () => previewSrc(im.src);
      im.onmouseleave = () => updatePreview();
      if (editable) im.ondblclick = () => {
        bench.extras.splice(i, 1); replaceBench();
      };
      s.appendChild(im);
    } else if (editable) { s.textContent = "drop"; }
    else s.classList.add("reserved");
    slots.appendChild(s);
  }
  return slots;
}

function beep() {
  try {
    const a = new (window.AudioContext || window.webkitAudioContext)();
    const o = a.createOscillator(); o.frequency.value = 520;
    o.connect(a.destination); o.start(); o.stop(a.currentTime + 0.12);
  } catch (e) {}
}

async function addAssets(files) {
  for (const f of files) {
    if (f.type && !f.type.startsWith("image/")) continue;
    if (bench.extras.length >= 3) {
      $("status").textContent = "all 3 extra ref slots are full"; beep(); break;
    }
    const r = await (await fetch("/api/upload?name=" +
      encodeURIComponent(f.name || "pasted.png"),
      {method: "POST", body: f})).json();
    if (r.error) { $("status").textContent = r.error; continue; }
    if (!bench.extras.includes(r.asset)) bench.extras.push(r.asset);
    $("status").textContent = "asset " + r.asset + " -> extra ref slot";
  }
  replaceBench();
}

document.addEventListener("paste", e => {
  const files = [...(e.clipboardData?.files || [])];
  if (!files.length) return;      // text pastes pass through untouched
  e.preventDefault();
  addAssets(files);
});

function carousel(r, line) {
  const wrap = document.createElement("div"); wrap.className = "stripwrap";
  const strip = document.createElement("div"); strip.className = "strip";
  const hl = $("hltree").checked && line;
  for (const name of r.images) {
    const rel = (r.dir ? r.dir + "/" : "") + name;
    const c = document.createElement("div"); c.className = "cell";
    c.dataset.rel = rel;
    if (rel === selImg) c.classList.add("sel");
    else if (hl && line[r.dir] !== name) c.classList.add("dim");
    if (hl && line[r.dir] === name) c.classList.add("lineage");
    const img = document.createElement("img");
    img.src = "/img/" + rel; img.loading = "lazy";
    const b = (r.batches || []).find(b => b.images.includes(name));
    if (b) {
      const s = b.seeds ? b.seeds[b.images.indexOf(name)] : b.seed;
      const v = b.vary ?? b.drift;
      img.title = name + "  seed " + s + (b.lock ? " " + b.lock : "")
        + (v ? " vary " + v : "") + "\n" + b.prompt;
    }
    // hover always previews transiently; mouse-out falls back to the
    // selected image (or keeps the last hover when nothing is selected)
    img.onmouseenter = () => setPreview(rel);
    img.onmouseleave = () => updatePreview();
    // click = in-place selection (no re-render, so the node survives and
    // dblclick fires naturally; a dblclick toggles selection twice = no-op)
    img.onclick = () => selectImage(rel);
    img.ondblclick = () => {
      bench.mother = rel;
      // new generations inherit white-bg from the mother's generation
      const mrow = log.find(x => x.dir === dirOf(rel));
      bench.whitebg = (mrow && mrow.whitebg != null) ? mrow.whitebg : true;
      replaceBench();
      const be = document.querySelector('[data-dir="~bench"]');
      if (be) be.scrollIntoView({behavior:"smooth", block:"nearest"});
    };
    const ops = document.createElement("div"); ops.className = "ops";
    const del = document.createElement("button"); del.textContent = "✕ delete";
    del.onclick = async e => {
      e.stopPropagation();
      const res = await (await fetch("/api/delete", {method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({path: rel})})).json();
      $("status").textContent = res.error || ("deleted " + disp(rel));
      if (!res.error) { selImg = null; refresh(true); }
    };
    ops.appendChild(del);
    c.append(img, ops);
    strip.appendChild(c);
  }
  const eL = document.createElement("button"); eL.className = "edge"; eL.textContent = "‹";
  const eR = document.createElement("button"); eR.className = "edge"; eR.textContent = "›";
  eL.onclick = () => strip.scrollBy({left: -strip.clientWidth});
  eR.onclick = () => strip.scrollBy({left:  strip.clientWidth});
  wrap.append(eL, strip, eR);
  return wrap;
}

function revealLineage(line) {
  // bring each lineage image into ITS OWN carousel's view - horizontal
  // scrolling only, and only if actually hidden. NEVER scroll the page
  // vertically here (scrollIntoView would walk the view up to the root).
  for (const [dir, name] of Object.entries(line)) {
    const rowEl = document.querySelector(`[data-dir="${CSS.escape(dir)}"]`);
    if (!rowEl) continue;
    const rel = (dir ? dir + "/" : "") + name;
    const cell = rowEl.querySelector(`[data-rel="${CSS.escape(rel)}"]`);
    if (!cell) continue;
    const strip = cell.closest(".strip");
    const sr = strip.getBoundingClientRect(), cr = cell.getBoundingClientRect();
    if (cr.left < sr.left) strip.scrollLeft += cr.left - sr.left;
    else if (cr.right > sr.right) strip.scrollLeft += cr.right - sr.right;
  }
}

function previewSrc(src) { $("preview").firstElementChild.src = src; }
function setPreview(rel) { previewSrc(rel ? "/img/" + rel : ""); }
function updatePreview() { if (selImg) setPreview(selImg); }

$("hltree").onchange = () => updateSelection();
$("hideothers").onchange = () => render();

document.addEventListener("keydown", e => {
  if (e.target.closest("textarea, input, select")) return;
  if (e.key === "Escape" && selImg) {
    selImg = null;
    if ($("hideothers").checked) render(); else updateSelection();
    return;
  }
  if (!selImg) return;
  const moves = {ArrowLeft: "L", ArrowRight: "R", ArrowUp: "U", ArrowDown: "D"};
  const dirn = moves[e.key];
  if (!dirn) return;
  e.preventDefault();
  const cur = document.querySelector(".cell.sel");
  if (!cur) return;
  let target = null;
  if (dirn === "L" || dirn === "R") {
    target = dirn === "L" ? cur.previousElementSibling : cur.nextElementSibling;
  } else {
    const rowsWithCells = [...document.querySelectorAll("#rows .row")]
      .filter(el => el.querySelector(".cell"));
    const idx = rowsWithCells.indexOf(cur.closest(".row"));
    const nextRow = rowsWithCells[idx + (dirn === "D" ? 1 : -1)];
    if (nextRow) {   // nearest column by screen position = grid feel
      const cx = cur.getBoundingClientRect();
      const mid = cx.left + cx.width / 2;
      let best = null, bestD = Infinity;
      nextRow.querySelectorAll(".cell").forEach(c => {
        const r = c.getBoundingClientRect();
        const d = Math.abs(r.left + r.width / 2 - mid);
        if (d < bestD) { bestD = d; best = c; }
      });
      target = best;
    }
  }
  if (target && target.dataset.rel) {
    selImg = target.dataset.rel;
    if ($("hideothers").checked) render(); else updateSelection();
    const t = document.querySelector(".cell.sel");
    if (t) t.scrollIntoView({block: "nearest", inline: "nearest"});
  }
});

// draggable divider
(() => {
  const dv = $("divider");
  const stored = localStorage.getItem("evolve_pw");
  if (stored) document.documentElement.style.setProperty("--pw", stored + "px");
  let drag = false;
  dv.onmousedown = () => { drag = true; document.body.style.userSelect = "none"; };
  document.addEventListener("mousemove", e => {
    if (!drag) return;
    const w = Math.max(160, Math.min(window.innerWidth - 300,
                                     window.innerWidth - e.clientX));
    document.documentElement.style.setProperty("--pw", w + "px");
    localStorage.setItem("evolve_pw", w);
  });
  document.addEventListener("mouseup", () => { drag = false;
    document.body.style.userSelect = ""; });
})();

async function spawn(target, prompt, isBench, whitebg) {
  busy = true;
  document.querySelectorAll(".plusbtn").forEach(b => b.disabled = true);
  $("status").textContent = "rendering ...";
  const t0 = Date.now();
  try {
    const r = await fetch("/api/spawn", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        ...target, prompt,
        n: +$("n").value, width: +$("w").value, height: +$("h").value,
        seed: $("seed").value ? +$("seed").value : null,
        lock: $("lock").value, vary: +$("vary").value,
        lora: $("lora").value, lora_strength: +$("lorastr").value,
        whitebg: !!whitebg })});
    const res = await r.json();
    if (res.error) $("status").textContent = res.error;
    else {
      $("status").textContent = "done in " + ((Date.now()-t0)/1000).toFixed(0)
        + "s -> " + disp(res.vdir) + " (seed " + res.seed + ")";
      if (isBench) bench = {mother: null, prompt: "", extras: []};
      busy = false;
      await refresh(true);
      return;
    }
  } catch (e) { $("status").textContent = "spawn failed: " + e; }
  busy = false; render();
}

refresh();
</script>
"""


def main():
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="evolve",
                    help="genealogy root directory (default: ./evolve, CWD-relative)")
    ap.add_argument("--port", type=int, default=8189)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    global ASSETS
    ROOT = Path(args.root).expanduser().resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    ASSETS = (Path.cwd() / "assets").resolve()      # project-root-relative
    ASSETS.mkdir(parents=True, exist_ok=True)

    app = web.Application(client_max_size=64 * 1024 ** 2)
    app.router.add_get("/", index)
    app.router.add_get("/api/log", api_log)
    app.router.add_get("/api/loras", api_loras)
    app.router.add_post("/api/spawn", api_spawn)
    app.router.add_post("/api/upload", api_upload)
    app.router.add_post("/api/delete", api_delete)
    app.router.add_post("/api/delete_row", api_delete_row)
    app.router.add_get("/img/{path:.+}", serve_image)
    app.router.add_get("/asset/{name}", serve_asset)

    print(f"evolve root: {ROOT}")
    print(f"open:        http://{args.host}:{args.port}/")
    print("(ComfyUI must be running at 127.0.0.1:8188 before you generate)")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
