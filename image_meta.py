"""image_meta.py - what an image file SAYS about itself.

    python image_meta.py IMAGE [IMAGE ...]        all chunks, one JSON object
    python image_meta.py IMAGE -k prompt          just that chunk
    python image_meta.py IMAGE --keys             list chunk names
    python image_meta.py IMAGE | jq '.prompt."68"'

Separation of concerns (provenance spec): the PNG = how to remake THIS
image (ComfyUI's `prompt` chunk, optional `workflow` geometry, our
versioned `evolve` chunk: project + id + recipe + staged-name map, NO
ancestry); the project journal = lineage. glean_recipe() maps foreign
metadata onto UI fields at import (layer-1-lite, fails silently per step).
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

EVOLVE_CHUNK_VERSION = 1
EXIF_TEXT_TAGS = {270: "ImageDescription", 37510: "UserComment"}


def evolve_chunk(project, image_id, source, recipe, inputs):
    """The `evolve` png chunk for a stored image (v1)."""
    return json.dumps({"v": EVOLVE_CHUNK_VERSION, "project": project, "id": image_id,
                       "source": source, "recipe": recipe, "inputs": inputs or {}})


def harvest_chunks(img, payload, embed_workflow=False):
    """Text chunks for the stored png: everything ComfyUI embedded in the
    scratch render (`prompt`, POSSIBLY augmented - LoadImage gains
    is_changed = sha256 of the loaded file), plus optional `workflow`
    geometry so the png drags into the ComfyUI frontend editable
    (API-format alone opens an EMPTY canvas). Geometry needs the live
    server's /object_info; it just rendered, so it is up - but never let a
    conversion hiccup lose the render."""
    chunks = {k: v for k, v in img.info.items() if isinstance(v, str)}
    if embed_workflow and "workflow" not in chunks:
        try:
            from add_geometry import convert_payload
            chunks["workflow"] = json.dumps(convert_payload(payload["prompt"]))
        except Exception as e:
            print(f"workflow geometry skipped: {type(e).__name__}: {e}")
    return chunks


def follow_text(payload, ref, depth=0):
    """Follow a conditioning link upstream to its text, skipping
    pass-throughs (ReferenceLatent chains etc.). None = lost the trail;
    "" = ConditioningZeroOut, i.e. no real -ive prompt."""
    while isinstance(ref, list) and len(ref) == 2 and depth < 25:
        node = payload.get(str(ref[0])) or {}
        cls = node.get("class_type", "")
        ins = node.get("inputs", {})
        if "TextEncode" in cls:
            for f in ("text", "prompt"):
                if isinstance(ins.get(f), str):
                    return ins[f]
            return None
        if "ZeroOut" in cls:
            return ""
        nxt = None
        for f in ("conditioning", "positive", "cond"):
            if f in ins:
                nxt = ins[f]
                break
        ref, depth = nxt, depth + 1
    return None


def glean_recipe(chunks):
    """Layer-1-lite import mapping (2026-08-24). The full graph interrogator
    (class-role registry) is DEFERRED - wild graph shapes vary too much; this
    gleans in priority order prompt/-ive -> sampler (cfg/steps/seed) -> LoRA
    -> family, treats the import as a fiat image (op=import, no parents), and
    fails SILENTLY per step (logged to console, no UI warnings - user
    decision). An image bred HERE carries the `evolve` marker: its own recipe
    wins outright."""
    try:
        if chunks.get("evolve"):
            r = (json.loads(chunks["evolve"]) or {}).get("recipe")
            if r:
                r = dict(r)
                r["op"] = "import"
                print("import glean: evolve marker, recipe adopted")
                return r
        payload = json.loads(chunks["prompt"])
        if not isinstance(payload, dict):
            return None
    except Exception:
        return None
    r = {"op": "import"}
    try:  # 1. prompts, via the sampler's conditioning edges
        samp = next((n for n in payload.values()
                     if "KSampler" in n.get("class_type", "")
                     or n.get("class_type") == "SamplerCustomAdvanced"), None)
        guider = next((n for n in payload.values()
                       if n.get("class_type") == "CFGGuider"), None)
        ins = ((guider or samp) or {}).get("inputs", {})
        pos = follow_text(payload, ins.get("positive"))
        neg = follow_text(payload, ins.get("negative"))
        if pos is None:  # fallback: exactly one text node in the whole graph
            texts = [n["inputs"].get("text", n["inputs"].get("prompt"))
                     for n in payload.values()
                     if "TextEncode" in n.get("class_type", "")]
            texts = [t for t in texts if isinstance(t, str)]
            if len(texts) == 1:
                pos = texts[0]
        if pos:
            r["prompt"] = pos
        if neg:
            r["negative"] = neg
    except Exception as e:
        print(f"import glean (prompt) failed: {type(e).__name__}: {e}")
    try:  # 2. sampler settings, wherever the graph shape keeps them
        for n in payload.values():
            cls, ins = n.get("class_type", ""), n.get("inputs", {})
            if not ("KSampler" in cls or cls in
                    ("CFGGuider", "RandomNoise", "Flux2Scheduler", "BasicScheduler")):
                continue
            for f, dst in (("seed", "seed"), ("noise_seed", "seed"),
                           ("steps", "steps"), ("cfg", "cfg")):
                v = ins.get(f)
                if isinstance(v, (int, float)) and dst not in r:
                    r[dst] = v
    except Exception as e:
        print(f"import glean (sampler) failed: {type(e).__name__}: {e}")
    try:  # 3. LoRA (the first on the model chain)
        for n in payload.values():
            if "Lora" not in n.get("class_type", ""):
                continue
            ins = n.get("inputs", {})
            name = ins.get("lora_name") or ins.get("lora_path")
            if isinstance(name, str):
                r["lora"] = name
                st = ins.get("strength_model", ins.get("strength"))
                if isinstance(st, (int, float)):
                    r["lora_strength"] = st
                break
    except Exception as e:
        print(f"import glean (lora) failed: {type(e).__name__}: {e}")
    try:  # family, from the model filename
        for n in payload.values():
            ins = n.get("inputs", {})
            name = str(ins.get("ckpt_name") or ins.get("unet_name") or "")
            if "flux-2-klein" in name:
                r["family"] = "klein"
            elif "z_image" in name:
                r["family"] = "zimage"
            elif "Illustrious" in name:
                r["family"] = "illustrious"
    except Exception as e:
        print(f"import glean (family) failed: {type(e).__name__}: {e}")
    got = sorted(k for k in r if k != "op")
    if not got:
        return None
    print(f"import glean: {', '.join(got)}")
    return r


# ---------- CLI: dump an image's metadata as JSON (the old imgmeta.py) ----------

def _coerce(v, raw=False):
    """One metadata value -> something JSON can hold, preferring structure."""
    if isinstance(v, (bytes, bytearray)):
        return {"_bytes": len(v)}
    if isinstance(v, str):
        if not raw and v.lstrip()[:1] in "{[":
            try:
                return json.loads(v)
            except ValueError:
                pass
        return v
    if isinstance(v, tuple):
        return list(v)
    try:
        json.dumps(v)
        return v
    except TypeError:
        return repr(v)


def read_meta(path, raw=False):
    """All embedded metadata, JSON-nested where it parses. webp/jpeg keep
    generator metadata in EXIF, so those tags are checked too."""
    out = {}
    with Image.open(path) as im:
        for k, v in im.info.items():
            out[k] = _coerce(v, raw)
        try:
            ex = im.getexif()
        except Exception:
            ex = None
        for tag, name in EXIF_TEXT_TAGS.items():
            if ex and tag in ex and name not in out:
                v = ex[tag]
                if isinstance(v, bytes):
                    v = v.split(b"\x00")[-1].decode("utf-8", "replace")
                out[name] = _coerce(v, raw)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("-k", "--key", help="emit only this chunk")
    ap.add_argument("--keys", action="store_true", help="list chunk names only")
    ap.add_argument("--raw", action="store_true",
                    help="keep JSON-looking strings as strings")
    args = ap.parse_args()
    results = {}
    for p in args.images:
        try:
            m = read_meta(p, args.raw)
        except Exception as e:
            m = {"_error": f"{type(e).__name__}: {e}"}
        if args.keys:
            m = sorted(m)
        elif args.key:
            m = m.get(args.key)
        results[str(Path(p))] = m
    out = next(iter(results.values())) if len(results) == 1 else results
    json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
