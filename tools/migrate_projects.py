"""Migrate a pre-tags root (project subdirs) into ONE store with tags, and
SCAN a directory of alien images into the store (same code, on demand).

    python tools/migrate_projects.py --root D:/evolve_root            # migrate
    python tools/migrate_projects.py --root D:/evolve_root --scan D:/pics [--tag word ...]

Migration: every `<root>/<project>/images/*.png` (and `archive/`) becomes
`<root>/images/<new id>.png`; the project name becomes a word on the image,
the archived BIT for archived ones, `pinned` from the old state, `lora_dataset_
<asset>` from the old assets.json datasets (descriptions kept, the leading
trigger stripped), recipes/parents/history from the old journals. Files
with no journal record are absorbed from their own metadata: recipe from
the `evolve` chunk, parents ONLY where verifiable (the parent migrated here
AND its sha256 matches the LoadImage `is_changed` in the `prompt` chunk).
Duplicates (same bytes) collapse into one image carrying both projects'
words. The old project dirs are MOVED to `<root>/_migrated/` untouched.
Refuses to run if the root already has a journal.

Scan: alien files -> absorbed the same way (png kept as the store file;
jpg/webp/alpha get a flattened png copy; originals are left in place and
reported, never deleted). Duplicates re-attach to their existing record.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import _cli  # noqa: F401
from PIL import Image

import project
from lora import family_of_path
from image_file import IMAGE_EXTS, flattened_rgb, sha1_of, text_chunks
from image_meta import glean_recipe
from image_utils import has_alpha
from store import PINNED, Store


def strip_trigger(desc, name):
    d = (desc or "").strip()
    if d.startswith(name):
        d = d[len(name):].lstrip(" ,:-")
    return d


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def chunk_parents(chunks, origin_map, origin_sha, store):
    """Parents from an image's own metadata, VERIFIED: the `evolve` chunk's
    inputs map staged names -> origin ids; the `prompt` chunk's LoadImage
    nodes carry is_changed = sha256 of the file that was actually loaded.
    A parent is recorded only if it migrated here and the ORIGIN file's bytes
    (what ComfyUI loaded; the store re-encodes on absorb) match."""
    try:
        ev = json.loads(chunks.get("evolve") or "{}")
        inputs = ev.get("inputs") or {}
        payload = json.loads(chunks.get("prompt") or "{}")
    except Exception:
        return [], 0
    if not inputs:
        return [], 0
    loaded = {}
    for n in payload.values():
        if n.get("class_type") == "LoadImage":
            loaded[n["inputs"].get("image")] = n["inputs"].get("is_changed")
    parents, unverified = [], 0
    for staged, oid in inputs.items():
        key = (ev.get("project"), oid)
        new = origin_map.get(key)
        want = loaded.get(staged)
        if new is None or not want or not store.alive(new) or origin_sha.get(key) != want:
            unverified += 1
            continue
        parents.append(new)
    return parents, unverified


def absorb(store, path, tags, origin_map, origin_sha, record=None, origin=None, report=None):
    """One file -> one store image (or a re-attach on duplicate bytes).
    record = the old journal record when there is one. Returns the new id."""
    data = Path(path).read_bytes()
    sha1 = sha1_of(data)
    existing = store.find_sha1(sha1)
    if existing is not None:
        if tags:
            store.tag([existing], add=tags)
        report and report.append(f"  dup  {path} -> #{existing} (+{','.join(tags)})")
        return existing
    with Image.open(path) as im:
        im.load()
        chunks = text_chunks(im)
        alpha = has_alpha(im) or im.mode != "RGB" or Path(path).suffix.lower() != ".png"
        rgb = flattened_rgb(im) if alpha else im.copy()
    recipe = record["recipe"] if record else glean_recipe(chunks)
    parents, unverified = [], 0
    if record:
        # journal parents are trustworthy within a project: map old -> new ids
        parents = [origin_map.get((origin, p)) for p in (record.get("parents") or [])]
        parents = [p for p in parents if p is not None and store.alive(p)]
    else:
        parents, unverified = chunk_parents(chunks, origin_map, origin_sha, store)
    try:
        ev = json.loads(chunks.get("evolve") or "{}")
        born = {"project": ev.get("project"), "id": ev.get("id")} if ev.get("id") else None
    except Exception:
        born = None
    keep = {k: v for k, v in chunks.items() if k != "evolve"}
    i = store.add_image(rgb, (record or {}).get("source") or "scan", recipe=recipe,
                        parents=parents, sha1=sha1, chunks=keep, tags=tags,
                        ts=(record or {}).get("ts"))
    if born:
        store.images[i]["origin"] = born          # provenance note, in memory only
    note = "" if not unverified else f"  ({unverified} parent(s) unverifiable)"
    report and report.append(f"  {'rec ' if record else 'meta'} {path} -> #{i} "
                             f"[{','.join(tags)}]{note}")
    return i


def migrate(root):
    root = Path(root)
    if (root / "journal.jsonl").exists():
        sys.exit(f"{root} already has a journal - nothing to migrate")
    projects = sorted(d for d in root.iterdir()
                      if d.is_dir() and (d / "images").is_dir() and d.name not in
                      ("images", "_migrated", "_train", "_debug", "loras"))
    if not projects:
        sys.exit("no project subdirs found")
    # journaled projects FIRST: their records are the trustworthy provenance,
    # and chunk-only files elsewhere can only verify parents that already
    # migrated (hand-copied dirs usually descend from journaled ones)
    projects.sort(key=lambda d: (not (d / "journal.jsonl").exists(), d.name))
    old_assets = project.read_json(root / "assets.json", [])
    store = Store(root)
    origin_map = {}          # (project, old id) -> new id
    origin_sha = {}          # (project, old id) -> sha256 of the ORIGINAL file
    report = []
    hist_events = []
    for pd in projects:
        name = pd.name
        records, gone, hist, pins = {}, set(), [], set()
        j = pd / "journal.jsonl"
        if j.exists():
            for line in j.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                ev = json.loads(line)
                if ev["t"] == "image":
                    records[ev["id"]] = ev
                elif ev["t"] == "gc":
                    gone.update(ev["ids"])
                elif ev["t"] in ("hist", "working") and ev.get("id") is not None:
                    hist.append(ev["id"])
        st = project.read_json(pd / "state.json", {})
        pins = set(st.get("pins") or [])
        files = {}
        for f in sorted((pd / "images").glob("*.png"), key=lambda p: p.stem):
            files[f] = []
        for f in sorted((pd / "archive").glob("*.png") if (pd / "archive").is_dir() else []):
            files[f] = ["__archived__"]
        report.append(f"{name}: {len(files)} files, {len(records)} records")
        for f, extra in files.items():
            try:
                oid = int(f.stem)
            except ValueError:
                oid = None
            rec = records.get(oid)
            archived = "__archived__" in extra or oid in gone
            tags = [name] + ([PINNED] if oid in pins and not archived else [])
            i = absorb(store, f, tags, origin_map, origin_sha, rec, name, report)
            if archived:
                store.archive([i])
            if oid is not None:
                origin_map[(name, oid)] = i
                origin_sha[(name, oid)] = sha256_file(f)
        for h in hist:
            if (name, h) in origin_map:
                hist_events.append(origin_map[(name, h)])
    # assets: datasets -> words + descriptions; loras kept
    new_assets = []
    for a in old_assets:
        new_assets.append({"name": a["name"], "files": list(a.get("loras") or [])})
        for e in a.get("dataset") or []:
            parts = str(e.get("path", "")).split("/")
            if len(parts) == 3 and parts[1] == "images" and parts[2].endswith(".png"):
                key = (parts[0], int(parts[2][:-4]))
                i = origin_map.get(key)
                if i is None:
                    report.append(f"  asset {a['name']}: {e['path']} not migrated")
                    continue
                store.tag([i], add=[f"lora_dataset_{a['name']}"])
                d = strip_trigger(e.get("description"), a["name"])
                cur = store.images[i]["description"]
                seeded = ((store.images[i].get("recipe") or {}).get("prompt") or "").strip()
                if d and (not cur or cur == seeded):      # a written caption beats a seed
                    store.describe(i, d)
                elif d and cur != d:
                    report.append(f"  #{i}: kept description {store.images[i]['description']!r}, "
                                  f"asset {a['name']} said {d!r}")
    project.write_json(root / "loras.json", new_assets)
    (root / "assets.json").unlink(missing_ok=True)
    report.extend(migrate_loras(root))
    for h in hist_events:
        store.hist_append(h)
    store.save_state()
    mig = root / "_migrated"
    mig.mkdir(exist_ok=True)
    for pd in projects:
        shutil.move(str(pd), str(mig / pd.name))
    report.append(f"done: {len(store.alive_ids())} images, words: "
                  + ", ".join(f"{w} x{n}" for w, n in sorted(store.words().items())))
    report.append(f"old project dirs moved to {mig}")
    return report


def migrate_loras(root):
    """loras/<asset>/*.safetensors -> loras/<asset>/<family>/ (family read
    from each file's own metadata), and assets.json entries -> {path,
    family}. Idempotent: files already under a family dir are left alone;
    .bak files travel with their file; logs/ stays."""
    from lora_train.common import detect_family
    root = Path(root)
    report = []
    src = root / "loras.json" if (root / "loras.json").exists() else root / "assets.json"
    raw = project.read_json(src, [])
    moved = {}
    lr = root / "loras"
    for adir in sorted(lr.iterdir()) if lr.is_dir() else []:
        if not adir.is_dir():
            continue
        for f in sorted(adir.glob("*.safetensors")):
            fam = detect_family(f)
            if fam is None:
                report.append(f"  ?? {f.relative_to(root).as_posix()}: family unknown, left in place")
                continue
            dest = adir / fam.value / f.name
            dest.parent.mkdir(exist_ok=True)
            shutil.move(str(f), str(dest))
            bak = f.with_suffix(f.suffix + ".bak")
            if bak.exists():
                shutil.move(str(bak), str(dest.with_suffix(dest.suffix + ".bak")))
            moved[f.relative_to(root).as_posix()] = (dest.relative_to(root).as_posix(), fam.value)
            report.append(f"  {f.name} -> {adir.name}/{fam.value}/")
    out = []
    for a in raw:
        entries = []
        for e in a.get("files") or a.get("loras") or []:
            if isinstance(e, dict):
                entries.append(e)
                continue
            rel = str(e).replace("\\", "/")
            if rel in moved:
                path, fam = moved[rel]
            else:
                path = rel
                famv = family_of_path(rel)
                if famv is None:
                    q = root / rel
                    famd = detect_family(q) if q.is_file() else None
                    if famd is None:
                        report.append(f"  asset {a['name']}: {rel} dropped (file/family unknown)")
                        continue
                    fam = famd.value
                else:
                    fam = famv.value
            entries.append({"path": path, "family": fam})
        out.append({"name": a["name"], "files": entries})
    project.write_json(root / "loras.json", out)
    if src.name == "assets.json":
        src.unlink(missing_ok=True)
    report.append(f"loras.json: {len(out)} LoRA(s), per-family file entries")
    return report


def scan(root, folder, tags):
    root, folder = Path(root), Path(folder)
    store = Store(root)
    report = []
    for f in sorted(q for q in folder.rglob("*") if q.is_file() and q.suffix.lower() in IMAGE_EXTS):
        try:
            absorb(store, f, list(tags), {}, {}, None, None, report)
        except Exception as e:
            report.append(f"  FAIL {f}: {type(e).__name__}: {e}")
    report.append("originals left in place (delete them yourself once happy)")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--scan", help="folder of alien images to absorb (instead of migrating)")
    ap.add_argument("--tag", action="append", default=[], help="word(s) to put on scanned images")
    ap.add_argument("--loras", action="store_true",
                    help="only move loras/<asset>/*.safetensors into per-family subdirs "
                         "and rewrite the entries as loras.json {path, family}")
    a = ap.parse_args()
    project.set_root(a.root)
    if a.loras:
        report = migrate_loras(a.root)
    elif a.scan:
        report = scan(a.root, a.scan, a.tag)
    else:
        report = migrate(a.root)
    print("\n".join(report))


if __name__ == "__main__":
    main()
