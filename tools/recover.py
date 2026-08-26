"""Recover store images whose FILES are gone (Shift+Del in Explorer, a
botched move) from the copies evolve keeps without saying so. Strategies
are tried in turn, best evidence first, per missing image:

  1. staged hardlink   ComfyUI/input/evolve/<root-name>/<id>.png   (exact store bytes)
  2. dataset copy      <root>/_train/<lora>/<id>.png                (exact store bytes)
  3. migrated original <root>/_migrated/**/*.png, sha1 == the record's (exact original)
  4. scratch render    ComfyUI/output/evolve_scratch/*.png whose `prompt` chunk
                       carries the record's seed                    (pixel-identical)

    python tools/recover.py --root D:/evolve_root            # report the plan
    python tools/recover.py --root D:/evolve_root --apply    # write the files
    python tools/recover.py --root D:/evolve_root --ids 40 41 --apply

Restored files land at the record's journaled path; because identity is
the filename, the record simply has its file again (lineage, words, place
intact). Byte-exact strategies copy; the others re-wrap the pixels with
the record's `evolve` chunk. The Windows Recycle Bin is NOT searched -
restore from there by hand if Shift wasn't held. Nothing is journaled:
recovery is a filesystem repair, and a rescan/check afterwards clears
`missing`.
"""
import argparse
import json
import sys
from pathlib import Path

import _cli  # noqa: F401
from PIL import Image

import comfy_client
import project
from image_file import sha1_of, text_chunks, write_png
from image_meta import evolve_chunk
from store import Store


def index_sources(store, comfy_dir):
    """Build the lookup tables once (the migrated/scratch scans hash or
    parse every file, so they run a single time)."""
    src = {"staged": {}, "train": {}, "migrated": {}, "scratch": {}}
    for p in (comfy_dir / "input" / "evolve" / store.name).glob("*.png"):
        if p.stem.isdigit():
            src["staged"][int(p.stem)] = p
    for p in (store.dir / "_train").rglob("*.png"):
        if p.stem.isdigit():
            src["train"].setdefault(int(p.stem), p)
    for p in (store.dir / "_migrated").rglob("*.png"):
        try:
            src["migrated"].setdefault(sha1_of(p.read_bytes()), p)
        except OSError:
            pass
    scratch = comfy_dir / "output" / comfy_client.SCRATCH_PREFIX
    for p in sorted(scratch.glob("*.png"), key=lambda q: q.stat().st_mtime) if scratch.is_dir() else []:
        try:
            with Image.open(p) as im:
                pr = json.loads(im.info.get("prompt") or "{}")
        except Exception:
            continue
        for n in pr.values():
            ins = n.get("inputs", {}) if isinstance(n, dict) else {}
            for k in ("noise_seed", "seed"):
                v = ins.get(k)
                if isinstance(v, int):
                    src["scratch"][v] = p            # newest render for a seed wins
    return src


def plan_for(store, i, src):
    """(strategy, source path) for image i, or (None, None)."""
    r = store.images[i]
    if i in src["staged"]:
        return "staged hardlink (exact)", src["staged"][i]
    if i in src["train"]:
        return "_train copy (exact)", src["train"][i]
    if r.get("sha1") in src["migrated"]:
        return "_migrated original (sha1 match)", src["migrated"][r["sha1"]]
    seed = (r.get("recipe") or {}).get("seed")
    if seed in src["scratch"]:
        return "scratch render (seed match, pixel-identical)", src["scratch"][seed]
    return None, None


def restore(store, i, how, src):
    r = store.images[i]
    dest = store.path(i)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if how.endswith("(exact)"):
        dest.write_bytes(src.read_bytes())
    else:
        with Image.open(src) as im:
            im.load()
            chunks = {k: v for k, v in text_chunks(im).items() if k != "evolve"}
            chunks["evolve"] = evolve_chunk(store.name, i, r["source"], r.get("recipe"), {})
            write_png(im.convert("RGB"), dest, chunks)
    return dest


def recover(root, comfy_dir, ids=None, apply=False, log=print):
    project.set_root(root)
    store = Store(project.root())
    missing = [i for i in store.alive_ids() if not store.path(i).is_file()]
    if ids:
        missing = [i for i in missing if i in set(ids)]
    log(f"missing files: {len(missing)}")
    if not missing:
        return {"missing": 0, "matched": 0, "restored": 0}
    src = index_sources(store, Path(comfy_dir))
    log(f"sources: staged {len(src['staged'])}, _train {len(src['train'])}, "
        f"_migrated {len(src['migrated'])}, scratch seeds {len(src['scratch'])}")
    plans = {i: plan_for(store, i, src) for i in missing}
    by_how = {}
    for i in missing:
        how, p = plans[i]
        by_how[how or "NOT FOUND"] = by_how.get(how or "NOT FOUND", 0) + 1
        rel = p.relative_to(Path(comfy_dir)) if p and str(p).startswith(str(Path(comfy_dir))) else \
            (p.relative_to(store.dir) if p else "-")
        log(f"  #{i:<4} {store.images[i]['dir']}/{store.images[i]['file']:<10} <- {how or 'NOT FOUND'}: {rel}")
    log("summary: " + ", ".join(f"{k}: {v}" for k, v in sorted(by_how.items())))
    restored = 0
    if apply:
        for i, (how, p) in plans.items():
            if how:
                restore(store, i, how, p)
                restored += 1
        store.check_files()
        log(f"restored {restored}; still missing: {sorted(store.missing)}")
    else:
        log("dry run - add --apply to write the files")
    return {"missing": len(missing), "matched": sum(1 for h, _ in plans.values() if h),
            "restored": restored}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--comfy", default=str(comfy_client.COMFY_DIR),
                    help="ComfyUI dir (staged input links + scratch renders)")
    ap.add_argument("--ids", nargs="*", type=int, help="only these image ids")
    ap.add_argument("--apply", action="store_true", help="write the files (default: report only)")
    a = ap.parse_args()
    recover(a.root, a.comfy, a.ids, a.apply)


if __name__ == "__main__":
    main()
