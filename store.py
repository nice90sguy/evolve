"""store.py - THE image store, now over a real directory tree (Places,
2026-08-26): every image lives at exactly ONE path under the root
("one image, one place"; Explorer is a legitimate second UI), plus the
append-only journal.jsonl and state.json.

Three orthogonal axes: PLACE (the tree; `.trash/` is a place - archived ==
lives under .trash), SETS (tags: views, datasets - "show me", never
"where is it"), BITS (none left; `pinned` is a word). The journal is the
sole authority for identity, recipe and lineage; the FILESYSTEM is the
authority for place - the app journals its own moves, and rescan()
reconciles external ones (id-in-filename first, sha1 fallback; unknown
files are imported in place; a journaled image whose file is gone shows as
missing - the journal never forgets). Bytes immutable; location mutable.

Tags are COLLECTIONS the user curates; the code reads two (`pinned`,
`lora_dataset_*`) and owns none. Rule: pinned => not archived (pinning
restores; archiving skips pinned unless forced, which unpins). At birth a
child copies its MOTHER's words minus `pinned` plus the default tags, and
lands in its mother's FOLDER (fiat: the current folder). Cascade is one
explicit operation, one journal event. `lock` guards all access.
"""
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

from comfy_client import INPUT_DIR
from controls import CONTROLS, sanitize_controls
from image_file import (IMAGE_EXTS, flattened_rgb, link_or_copy, open_bytes,
                        sha1_of, text_chunks, write_png)
from image_meta import evolve_chunk, glean_recipe
from PIL import Image
from project import clean_tags, load_settings

TABS = ("create", "derive", "camera")
PINNED = "pinned"
NOT_INHERITED = {PINNED}
LEGACY_ARCHIVED_WORD = "archived"
TRASH = ".trash"
USE_RECYCLE_BIN = os.environ.get("EVOLVE_HARD_DELETE") != "1"   # tests hard-delete
                                                                # (no polluting the bin)
DEFAULT_DIR = "images"              # where the pre-Places flat store becomes a folder
RESERVED_DIRS = {"loras", "_train", "_debug", "_migrated"}   # never part of the tree
_ID_NAME = re.compile(r"(\d+)(?:[_-][^.]*)?\.png$", re.I)


def valid_dir(d):
    """A tree path: posix, relative, no empty/dot segments, top segment not
    reserved ('' = the root itself is not a place; TRASH is)."""
    d = str(d).replace("\\", "/").strip("/")
    if not d:
        return None
    parts = d.split("/")
    if any(p in ("", ".", "..") or p.startswith(".") and p != TRASH for p in parts):
        return None
    if parts[0] in RESERVED_DIRS:
        return None
    if TRASH in parts and parts[0] != TRASH:
        return None
    return d


def empty_candidates():
    return {t: [] for t in TABS}


class Store:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.name = self.dir.name
        self.journal = self.dir / "journal.jsonl"
        self.state_file = self.dir / "state.json"
        self.lock = threading.RLock()
        self.images = {}          # id -> record (dir, file, tags, description, purged?)
        self.missing = set()      # journaled ids whose file was not found by rescan
        self.last_rescan = None
        self.scan_busy = None     # {done, total, phase} while a rescan runs
        self.history = []
        self.next_id = 1
        self._load()
        (self.dir / DEFAULT_DIR).mkdir(parents=True, exist_ok=True)

    # ---------- persistence ----------

    def _load(self):
        if self.journal.exists():
            for line in self.journal.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                ev = json.loads(line)
                t = ev["t"]
                if t == "image":
                    tags = ev.get("tags") or []
                    ev["tags"] = clean_tags(tags)
                    ev["description"] = ev.get("description") or ""
                    ev["dir"] = ev.get("dir") or DEFAULT_DIR
                    ev["file"] = ev.get("file") or f"{ev['id']}.png"
                    ev["home"] = ev.get("home") or ev["dir"]
                    ev["fresh"] = bool(ev.get("fresh"))
                    ev.setdefault("source_sha1", None)
                    if ev.pop("archived", False) or LEGACY_ARCHIVED_WORD in tags:
                        ev["dir"] = TRASH        # legacy bit/word -> the trash place
                    self.images[ev["id"]] = ev
                    self.next_id = max(self.next_id, ev["id"] + 1)
                elif t == "touch":
                    for i in ev["ids"]:
                        if i in self.images:
                            self.images[i]["fresh"] = False
                elif t == "rehash":
                    r = self.images.get(ev["id"])
                    if r:
                        r["sha1"] = ev.get("sha1", r.get("sha1"))
                        r["source_sha1"] = ev.get("source_sha1", r.get("source_sha1"))
                elif t == "tag":
                    add, rm = ev.get("add") or [], ev.get("remove") or []
                    for i in ev.get("ids", []):
                        r = self.images.get(i)
                        if r:
                            r["fresh"] = False
                            self._apply_tags(r, add, rm)
                            if LEGACY_ARCHIVED_WORD in add:
                                r["dir"] = TRASH
                            elif LEGACY_ARCHIVED_WORD in rm:
                                r["dir"] = r["home"]
                elif t in ("archive", "gc"):        # legacy events: archive = a bit then
                    for i in ev["ids"]:
                        if i in self.images:
                            self.images[i]["dir"] = TRASH
                elif t == "restore":
                    for i in ev["ids"]:
                        if i in self.images:
                            self.images[i]["dir"] = self.images[i]["home"]
                elif t == "move":
                    for m in ev["moves"]:
                        r = self.images.get(m["id"])
                        if r:
                            if ev.get("source") != "rescan":
                                r["fresh"] = False
                            r["dir"] = m["to"]
                            if "file" in m:
                                r["file"] = m["file"]
                            if m["to"] != TRASH:
                                r["home"] = m["to"]
                elif t == "describe":
                    r = self.images.get(ev["id"])
                    if r:
                        r["fresh"] = False
                        r["description"] = ev.get("description") or ""
                elif t == "working" and ev["id"] is not None:
                    if ev["id"] not in self.history:      # legacy (pre-v2)
                        self.history.append(ev["id"])
                elif t == "hist" and ev["id"] is not None:
                    if ev["id"] in self.images:
                        self.images[ev["id"]]["fresh"] = False     # bred from = touched
                    if not self.history or self.history[-1] != ev["id"]:
                        self.history.append(ev["id"])
                elif t == "purge":
                    for i in ev["ids"]:
                        if i in self.images:
                            self.images[i]["purged"] = True
                elif t == "revive":            # an id-named file reappeared
                    for i in ev["ids"]:
                        if i in self.images:
                            self.images[i]["purged"] = False
        if self.state_file.exists():
            self.state = json.loads(self.state_file.read_text(encoding="utf-8"))
        else:
            self.state = {"working": None, "slots": 6, "candidates": empty_candidates(),
                          "controls": dict(CONTROLS)}
        s = self.state
        s["controls"] = sanitize_controls(s.get("controls") or {}, self.alive)
        if isinstance(s.get("candidates"), list):          # v3.1 migration
            b = empty_candidates()
            for i in s["candidates"]:
                r = (self.images.get(i) or {}).get("recipe") or {}
                op = r.get("op") or ("derive" if (self.images.get(i) or {}).get("parents")
                                     else "create")
                b["camera" if op == "pov" else (op if op in b else "derive")].append(i)
            s["candidates"] = b
        s["candidates"] = {t: [i for i in s["candidates"].get(t, []) if self.alive(i)]
                           for t in TABS}
        for i in s.pop("pins", []):                         # legacy pins -> the word
            if self.alive(i) and PINNED not in self.images[i]["tags"]:
                self._apply_tags(self.images[i], [PINNED], [])
        if not self.alive(s.get("working")):
            s["working"] = None
        if not valid_dir(s.get("cwd") or ""):
            s["cwd"] = DEFAULT_DIR
        nav = s.get("nav") or {}
        stack = [i for i in nav.get("stack") or [] if self.alive(i)]
        s["nav"] = {"stack": stack, "pos": min(max(int(nav.get("pos", len(stack) - 1)), -1),
                                               len(stack) - 1)}

    @staticmethod
    def _apply_tags(rec, add, remove):
        tags = [w for w in rec["tags"] if w not in remove]
        for w in add:
            if w not in tags:
                tags.append(w)
        rec["tags"] = tags

    def _append(self, ev):
        ev["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.journal.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")

    def journal_event(self, ev):
        self._append(ev)

    def save_state(self):
        self.state_file.write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    # ---------- identity & place ----------

    def alive(self, i):
        """Exists and not purged (archived and missing images ARE alive)."""
        return i is not None and i in self.images and not self.images[i].get("purged")

    def alive_ids(self):
        return sorted(i for i in self.images if self.alive(i))

    def image_dir(self, i):
        return self.images[i]["dir"]

    def path(self, i):
        r = self.images[i]
        return self.dir / r["dir"] / r["file"]

    def is_archived(self, i):
        return self.alive(i) and self.images[i]["dir"].split("/")[0] == TRASH

    def archived_ids(self):
        return [i for i in self.alive_ids() if self.is_archived(i)]

    def cwd(self):
        """The open folder - falling back to images/ when it no longer
        exists on disk (deleted in Explorer): a folder that is gone is not
        a place, so it is never listed, never born into."""
        d = self.state.get("cwd") or DEFAULT_DIR
        return d if (self.dir / d).is_dir() else DEFAULT_DIR

    def set_cwd(self, d):
        d = valid_dir(d)
        if d is None:
            return False
        with self.lock:
            self.state["cwd"] = d
            (self.dir / d).mkdir(parents=True, exist_ok=True)
            self.save_state()
            return True

    def dirs(self):
        """Every directory in the tree = the REAL directories on disk (plus
        the cwd, which set_cwd creates). Records naming a folder that no
        longer exists do not conjure it: the filesystem is the authority
        for place; such records are `missing`. TRASH always listed last."""
        out = {DEFAULT_DIR, self.cwd()}
        for p in self.dir.rglob("*"):
            if p.is_dir():
                d = valid_dir(p.relative_to(self.dir).as_posix())
                if d and d != TRASH:
                    out.add(d)
        # parents of every dir are dirs too
        for d in list(out):
            parts = d.split("/")
            for k in range(1, len(parts)):
                out.add("/".join(parts[:k]))
        return sorted(out) + [TRASH]

    def in_dir(self, d):
        return [i for i in self.alive_ids() if self.images[i]["dir"] == d]

    def move(self, ids, to):
        """Move images to another folder (the app's own moves; Explorer
        moves are reconciled by rescan). Returns the ids moved."""
        to = valid_dir(to)
        if to is None:
            return []
        with self.lock:
            (self.dir / to).mkdir(parents=True, exist_ok=True)
            moves = []
            for i in ids:
                if not self.alive(i) or self.images[i]["dir"] == to:
                    continue
                r = self.images[i]
                src, dst = self.path(i), self.dir / to / r["file"]
                try:
                    if src.is_file():
                        shutil.move(str(src), str(dst))
                except OSError as e:
                    print(f"move #{i} failed: {e}")
                    continue
                r["dir"] = to
                if to != TRASH:
                    r["home"] = to
                self.missing.discard(i)
                moves.append({"id": i, "to": to})
            if moves:
                self._append({"t": "move", "moves": moves})
                self._unfresh([m["id"] for m in moves])
            return [m["id"] for m in moves]

    # ---------- trash (a place) ----------

    def archive(self, ids, force=False):
        """Move to .trash. Pinned images are SKIPPED unless force, which
        unpins them first (pinned => not archived)."""
        with self.lock:
            todo = [i for i in ids if self.alive(i) and not self.is_archived(i)]
            if not force:
                todo = [i for i in todo if not self.has(i, PINNED)]
            else:
                pinned = [i for i in todo if self.has(i, PINNED)]
                if pinned:
                    self.tag(pinned, remove=[PINNED])
            return self.move(todo, TRASH)

    def restore(self, ids):
        with self.lock:
            done = []
            for i in ids:
                if self.is_archived(i):
                    done += self.move([i], self.images[i]["home"])
            return done

    def load_bearing(self):
        """Trashed images some LIVE image descends from. Integrity is a
        WARNING, not a gate (user rule): emptying takes these too, and the
        UI shows missing-ancestor placeholders afterwards."""
        live = [i for i in self.alive_ids() if not self.is_archived(i)]
        keep = self.ancestors_closure(live)
        return [i for i in self.alive_ids() if self.is_archived(i) and i in keep]

    def empty_trash_plan(self):
        """The impact report for the confirm dialog - counts and samples,
        never an itemised ream (the trash may hold thousands)."""
        with self.lock:
            ids = self.archived_ids()
            bearing = self.load_bearing()
            bearing_set = set(bearing)
            orphaned = sorted({j for j in self.alive_ids() if not self.is_archived(j)
                               and (set(self.ancestors_closure([j])) - {j}) & bearing_set})
            size = 0
            for i in ids:
                try:
                    size += self.path(i).stat().st_size
                except OSError:
                    pass
            return {"count": len(ids), "bytes": size,
                    "load_bearing": len(bearing), "sample_bearing": bearing[:8],
                    "orphaned": len(orphaned), "sample_orphaned": orphaned[:8]}

    def empty_trash(self):
        """EVERYTHING in .trash -> the OS recycle bin (never a hard delete
        - Windows is the final undo), staged hardlinks unlinked so no
        bytes survive in the ComfyUI input dir. The journal keeps every
        record (purged); descendants show missing-ancestor placeholders."""
        with self.lock:
            ids = self.archived_ids()
            binned = []
            for i in ids:
                p = self.path(i)
                try:
                    if p.is_file():
                        if USE_RECYCLE_BIN:
                            import send2trash
                            send2trash.send2trash(str(p))
                        else:
                            p.unlink()
                except OSError as e:
                    print(f"empty trash: #{i} {e}")
                    continue
                self.staged_path(i).unlink(missing_ok=True)
                self.images[i]["purged"] = True
                binned.append(i)
            if binned:
                self._append({"t": "purge", "ids": binned, "to": "recycle-bin"})
            self.forget(binned)
            self.save_state()
            return binned

    # ---------- tags & words ----------

    def is_fresh(self, i):
        """A spawn nobody has touched yet: born of a round and not since
        made the WI, referenced, pinned, tagged, described, moved, trashed
        by hand or bred from. Fresh candidates are what a round throws
        away before it runs."""
        return self.alive(i) and bool(self.images[i].get("fresh"))

    def _unfresh(self, ids):
        """In-memory clear for mutations whose own journal event implies
        the touch on replay (tag / describe / move)."""
        for i in ids:
            r = self.images.get(i)
            if r:
                r["fresh"] = False

    def touch(self, ids):
        """Clear `fresh` (journaled once per image). Mutations that already
        journal (tag/describe/move/hist) clear it on replay; this is for
        the state-only ones: becoming the WI, landing in ref0/a ref."""
        with self.lock:
            todo = [i for i in ids if i is not None and self.is_fresh(i)]
            for i in todo:
                self.images[i]["fresh"] = False
            if todo:
                self._append({"t": "touch", "ids": todo})
            return todo

    def tags(self, i):
        return self.images[i]["tags"] if self.alive(i) else []

    def has(self, i, word):
        return word in self.tags(i)

    def with_word(self, word):
        return [i for i in self.alive_ids() if word in self.images[i]["tags"]]

    def pins(self):
        return self.with_word(PINNED)

    def words(self):
        out = {}
        for i in self.alive_ids():
            for w in self.images[i]["tags"]:
                out[w] = out.get(w, 0) + 1
        return out

    def has_children(self, i):
        """Any tracked image that used i as an input (any parent slot)."""
        return any(self.alive(j) and i in (r.get("parents") or [])
                   for j, r in self.images.items())

    def descendants(self, i):
        """The mother-line subtree under i (parent-0 edges), excluding i."""
        kids = {}
        for j, r in self.images.items():
            if not self.alive(j):
                continue
            ps = r.get("parents") or []
            if ps:
                kids.setdefault(ps[0], []).append(j)
        out, todo, seen = [], list(kids.get(i, [])), {i}
        while todo:
            x = todo.pop()
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
            todo.extend(kids.get(x, []))
        return sorted(out)

    def tag(self, ids, add=(), remove=(), cascade=False):
        """Add/remove words; with cascade also on mother-line descendants
        (ADD skips archived descendants, REMOVE does not). One journal
        event listing every id touched."""
        add, remove = clean_tags(add), clean_tags(remove)
        with self.lock:
            base = [i for i in ids if self.alive(i)]
            add_ids, rm_ids = set(base), set(base)
            if cascade:
                for i in base:
                    for d in self.descendants(i):
                        rm_ids.add(d)
                        if not self.is_archived(d):
                            add_ids.add(d)
            touched = []
            for i in sorted(add_ids | rm_ids):
                r = self.images[i]
                before = list(r["tags"])
                self._apply_tags(r, add if i in add_ids else [], remove if i in rm_ids else [])
                if r["tags"] != before:
                    touched.append(i)
            if touched:
                self._append({"t": "tag", "ids": touched, "add": add, "remove": remove,
                              "cascade": bool(cascade), "from": base})
                self._unfresh(touched)
            return touched

    def describe(self, i, text):
        with self.lock:
            if not self.alive(i):
                return False
            text = (text or "").strip()
            if self.images[i]["description"] != text:
                self.images[i]["description"] = text
                self._append({"t": "describe", "id": i, "description": text})
                self._unfresh([i])
            return True

    def ancestors_closure(self, ids):
        keep, todo = set(), list(ids)
        while todo:
            i = todo.pop()
            if i in keep or not self.alive(i):
                continue
            keep.add(i)
            todo.extend(self.images[i].get("parents") or [])
        return keep

    # ---------- creation ----------

    def staged_path(self, i):
        return INPUT_DIR / "evolve" / self.name / f"{i}.png"

    def stage_ref(self, i):
        """One hardlink per image at input/evolve/<root-name>/<id>.png -
        durable, survives moves within the volume (same inode)."""
        link_or_copy(self.path(i), self.staged_path(i))
        return f"evolve/{self.name}/{i}.png"

    def birth_tags(self, mother=None):
        inherited = [w for w in self.tags(mother) if w not in NOT_INHERITED] if mother else []
        return clean_tags(inherited + load_settings()["default_tags"])

    def birth_dir(self, mother=None):
        """A bred image lands in its MOTHER's folder; fiat in the cwd. A new
        image is never BORN into the trash (the cwd may be .trash while the
        user inspects it - reveal opens it there)."""
        if mother is not None and self.alive(mother) and not self.is_archived(mother):
            return self.images[mother]["dir"]
        d = self.cwd()
        return DEFAULT_DIR if d.split("/")[0] == TRASH else d

    def add_image(self, img, source, recipe=None, parents=None, sha1=None,
                  chunks=None, inputs=None, tags=None, description=None, ts=None,
                  dir=None, fresh=False, source_sha1=None):
        """Persist a PIL image (already RGB) under a fresh id at ONE place.
        `sha1` is the STORE FILE's bytes (always computed - rescan's
        identity fallback); `source_sha1` is the ORIGINAL's bytes when this
        is a conversion (import-dedupe; "already converted" for rescan).
        fresh=True marks a round's spawn (see is_fresh)."""
        with self.lock:
            i = self.next_id
            self.next_id += 1
            d = valid_dir(dir) if dir else None
            if d is None:                        # defaulting to the cwd: never
                d = self.cwd()                   # birth INTO the trash
                if d.split("/")[0] == TRASH:
                    d = DEFAULT_DIR
            (self.dir / d).mkdir(parents=True, exist_ok=True)
            all_chunks = dict(chunks or {})
            all_chunks["evolve"] = evolve_chunk(self.name, i, source, recipe, inputs)
            file = f"{i}.png"
            write_png(img, self.dir / d / file, all_chunks)
            if sha1 is not None and source_sha1 is None:
                source_sha1 = sha1                 # legacy call shape: the given hash was the original's
            sha1 = sha1_of((self.dir / d / file).read_bytes())
            if description is None:
                description = ((recipe or {}).get("prompt") or "").strip()
            rec = {"t": "image", "id": i, "dir": d, "file": file, "home": d,
                   "source": source, "w": img.width, "h": img.height,
                   "sha1": sha1, "source_sha1": source_sha1, "fresh": bool(fresh),
                   "recipe": recipe, "parents": parents or [],
                   "tags": clean_tags(tags or []), "description": description}
            if ts:
                rec["ts_origin"] = ts
            self._append(rec)
            self.images[i] = rec
            return i

    def find_sha1(self, sha1):
        """An alive record whose store file OR whose original had these bytes."""
        for i, r in self.images.items():
            if self.alive(i) and (r.get("sha1") == sha1 or r.get("source_sha1") == sha1):
                return i
        return None

    def import_bytes(self, data, source="import", tags=None, dir=None):
        """Dedupe by content hash, flatten alpha onto white, PRESERVE every
        text chunk (layer 0), glean a recipe. Lands in `dir` (default cwd).
        Returns (id, was_new)."""
        sha1 = sha1_of(data)
        existing = self.find_sha1(sha1)
        if existing is not None:
            if tags:
                self.tag([existing], add=tags)
            return existing, False
        img, raw = open_bytes(data)
        recipe = glean_recipe(raw)
        keep = {k: v for k, v in raw.items() if k != "evolve"}
        i = self.add_image(flattened_rgb(img), source, recipe=recipe, chunks=keep,
                           tags=self.birth_tags() + list(tags or []), dir=dir,
                           source_sha1=sha1)
        return i, True

    # ---------- rescan: reconcile the tree with the journal ----------

    def rescan(self, progress=None):
        """Walk the tree and reconcile it with the journal. Known ids found
        elsewhere -> journaled moves (Explorer moves become history);
        files the app has never seen -> imported IN PLACE ("add files
        freely, evolve will import them so you can track them properly");
        a purged record whose id-named file reappears -> revived; a
        journaled image with no file -> missing (placeholders; the journal
        never forgets). The walk and the hashing run WITHOUT the lock (a
        big drop must not freeze the UI); only the bookkeeping takes it.
        progress(done, total, phase) is called as it goes."""
        self.scan_busy = {"done": 0, "total": 0, "phase": "walking"}

        def tick(done, total, phase):
            self.scan_busy = {"done": done, "total": total, "phase": phase}
            if progress:
                progress(done, total, phase)

        try:
            report = {"moved": 0, "imported": 0, "missing": 0, "revived": 0, "skipped": []}
            with self.lock:
                known = set(self.images)
                by_sha = {r["sha1"]: j for j, r in self.images.items()
                          if r.get("sha1") and self.alive(j)}
                by_source = {r["source_sha1"]: j for j, r in self.images.items()
                             if r.get("source_sha1") and self.alive(j)}
            files = []
            for q in self.dir.rglob("*"):
                if q.is_file() and q.suffix.lower() in IMAGE_EXTS:
                    rel = q.relative_to(self.dir)
                    d = valid_dir(rel.parent.as_posix()) if rel.parent.as_posix() != "." else None
                    if d is not None:
                        files.append((q, d))
            total = len(files)
            found, unknown, originals = {}, [], []
            rest = []
            for q, d in files:                 # pass 1: id-named pngs are identity, by design
                m = _ID_NAME.fullmatch(q.name)
                i = int(m.group(1)) if m else None
                if i is not None and i in known and i not in found:
                    found[i] = (d, q.name)
                else:
                    rest.append((q, d))
            for n, (q, d) in enumerate(rest, 1):   # pass 2: everything else, by content
                h = sha1_of(q.read_bytes())
                j = by_sha.get(h)
                if j is not None and j not in found and q.suffix.lower() == ".png":
                    found[j] = (d, q.name)     # a renamed store png
                elif h in by_source or (j is not None and q.suffix.lower() != ".png"):
                    originals.append((q, d, by_source.get(h, j)))   # already converted: skip
                else:
                    unknown.append((q, d))
                if n % 25 == 0 or n == len(rest):
                    tick(n, len(rest), "matching")
            with self.lock:
                moves, revived = [], []
                for i, (d, file) in found.items():
                    r = self.images.get(i)
                    if r is None:
                        continue
                    if r.get("purged"):        # its file is back (or a stand-in
                        r["purged"] = False    # wearing its name): the record
                        revived.append(i)      # revives, lineage and words intact
                    if (r["dir"], r["file"]) != (d, file):
                        r["dir"], r["file"] = d, file
                        if d != TRASH:
                            r["home"] = d
                        moves.append({"id": i, "to": d, "file": file})
                    self.missing.discard(i)
                if moves:
                    self._append({"t": "move", "moves": moves, "source": "rescan"})
                    report["moved"] = len(moves)
                if revived:
                    self._append({"t": "revive", "ids": sorted(revived)})
                    report["revived"] = len(revived)
                # legacy repair: the one-hash era let an ORIGINAL (webp/jpg)
                # claim its converted record. Records must point at pngs.
                for q, d, j in originals:
                    r = self.images.get(j)
                    if r is None or not self.alive(j):
                        continue
                    if not r.get("source_sha1"):
                        r["source_sha1"] = r["sha1"] if r["file"].lower().endswith(".png") else None
                    if not r["file"].lower().endswith(".png"):
                        twin = self.dir / r["dir"] / f"{j}.png"
                        if not twin.is_file():
                            twin = self.dir / d / f"{j}.png"
                        if not twin.is_file():
                            with Image.open(q) as im:
                                im.load()
                                ch = {k: v for k, v in text_chunks(im).items() if k != "evolve"}
                                ch["evolve"] = evolve_chunk(self.name, j, r["source"], r.get("recipe"), {})
                                twin.parent.mkdir(parents=True, exist_ok=True)
                                write_png(flattened_rgb(im), twin, ch)
                            report["skipped"].append(f"#{j}: re-converted from {q.name}")
                        r["source_sha1"] = sha1_of(q.read_bytes())
                        r["dir"], r["file"] = twin.parent.relative_to(self.dir).as_posix(), twin.name
                        r["sha1"] = sha1_of(twin.read_bytes())
                        self._append({"t": "move", "moves": [{"id": j, "to": r["dir"], "file": r["file"]}],
                                      "source": "repair"})
                        self._append({"t": "rehash", "id": j, "sha1": r["sha1"],
                                      "source_sha1": r["source_sha1"]})
                        report["moved"] += 1
                    elif r.get("sha1") == sha1_of(q.read_bytes()):
                        # legacy: sha1 held the original's bytes; rehash to the store file
                        r["source_sha1"] = r["sha1"]
                        r["sha1"] = sha1_of(self.path(j).read_bytes()) if self.path(j).is_file() else r["sha1"]
                        self._append({"t": "rehash", "id": j, "sha1": r["sha1"],
                                      "source_sha1": r["source_sha1"]})
            for n, (q, d) in enumerate(unknown, 1):
                try:
                    data = q.read_bytes()
                    if q.suffix.lower() == ".png":
                        img, raw = open_bytes(data)
                        if img.mode == "RGB":   # adopt IN PLACE, no rewrite
                            with self.lock:
                                i = self.next_id
                                self.next_id += 1
                                recipe = glean_recipe(raw)
                                rec = {"t": "image", "id": i, "dir": d, "file": q.name,
                                       "home": d if d != TRASH else DEFAULT_DIR,
                                       "source": "scan", "w": img.width, "h": img.height,
                                       "sha1": sha1_of(data), "source_sha1": None, "fresh": False,
                                       "recipe": recipe,
                                       "parents": [], "tags": self.birth_tags(),
                                       "description": ((recipe or {}).get("prompt") or "").strip()}
                                self._append(rec)
                                self.images[i] = rec
                            report["imported"] += 1
                            continue
                    # non-png / alpha: a flattened png copy; original reported
                    _, new = self.import_bytes(data, source="scan", dir=d)
                    if new:
                        report["imported"] += 1
                    report["skipped"].append(f"{q.relative_to(self.dir).as_posix()} "
                                             "(converted copy made; original left in place)")
                except Exception as e:
                    report["skipped"].append(f"{q.relative_to(self.dir).as_posix()}: "
                                             f"{type(e).__name__}: {e}")
                if n % 10 == 0 or n == len(unknown):
                    tick(n, len(unknown), "importing")
            with self.lock:
                self.missing = {i for i in self.alive_ids() if i not in found
                                and not self.path(i).is_file()}
                report["missing"] = len(self.missing)
                report["ts"] = time.strftime("%H:%M:%S")
                self.last_rescan = report
            return report
        finally:
            self.scan_busy = None

    def check_files(self):
        """A stat pass over known files - the cheap way to notice Explorer
        deletions without a rescan. Updates `missing`; journals nothing."""
        with self.lock:
            self.missing = {i for i in self.alive_ids() if not self.path(i).is_file()}
            return len(self.missing)

    def forget_missing(self, ids=None):
        """Tombstone records whose files are gone (Shift+Del in Explorer,
        a dropped-then-deleted import): purged, no file touched. The
        journal keeps the lines; the app stops showing placeholders."""
        with self.lock:
            self.check_files()
            todo = sorted(self.missing if ids is None else (set(ids) & self.missing))
            for i in todo:
                self.images[i]["purged"] = True
                self.missing.discard(i)
            if todo:
                self._append({"t": "purge", "ids": todo, "to": "forgotten"})
            self.forget(todo)
            self.save_state()
            return todo

    # ---------- live state ----------

    def set_working(self, i, push=True):
        """push=True (any normal pick) records browser-back navigation: the
        forward branch is truncated, i lands on top. Scrubbing (nav_step)
        moves the pointer WITHOUT pushing - the stack is browsing state,
        persisted in state.json, never journaled (History stays the sparse
        bred-from work log; this is the browse log)."""
        with self.lock:
            if i is not None and not self.alive(i):
                raise ValueError(f"no such image {i}")
            self.state["working"] = i
            self.touch([i])
            if push and i is not None:
                nav = self.state["nav"]
                st, pos = nav["stack"], nav["pos"]
                if not (0 <= pos < len(st) and st[pos] == i):
                    del st[pos + 1:]
                    st.append(i)
                    del st[:-100]                     # cap; oldest browsing falls off
                    nav["pos"] = len(st) - 1
            self.save_state()

    def nav_step(self, direction):
        """Browser back (-1) / forward (+1) for the WI. Skips dead entries;
        returns the new WI id or None at the end of the stack."""
        with self.lock:
            nav = self.state["nav"]
            st, pos = nav["stack"], nav["pos"]
            while True:
                pos += 1 if direction > 0 else -1
                if not (0 <= pos < len(st)):
                    return None
                if self.alive(st[pos]):
                    break
            nav["pos"] = pos
            self.set_working(st[pos], push=False)
            return st[pos]

    def nav_neighbors(self, span=3):
        """The next few alive ids either side of the nav pointer (prefetch)."""
        nav = self.state["nav"]
        st, pos = nav["stack"], nav["pos"]
        out = []
        for d in (-1, 1):
            k, took = pos, 0
            while took < span:
                k += d
                if not (0 <= k < len(st)):
                    break
                if self.alive(st[k]) and st[k] not in self.missing:
                    out.append(st[k])       # missing files would prefetch a 404
                    took += 1
        return out

    def hist_append(self, i):
        with self.lock:
            if i is None or not self.alive(i):
                return
            if self.history and self.history[-1] == i:
                return
            self.history.append(i)
            self._append({"t": "hist", "id": i})

    def active_tab(self):
        t = self.state["controls"].get("tab") or "create"
        return t if t in TABS else "create"

    def cands(self, tab=None):
        return self.state["candidates"][tab if tab in TABS else self.active_tab()]

    def live_set(self):
        c = self.state["controls"]
        s = {self.state["working"], c.get("ref0"), *c["refs"]}
        s.discard(None)
        return s

    def forget(self, ids):
        ids = set(ids)
        s = self.state
        for lst in s["candidates"].values():
            lst[:] = [q for q in lst if q not in ids]
        if s["working"] in ids:
            s["working"] = None
        c = s["controls"]
        if c.get("ref0") in ids:
            c["ref0"] = None
        c["refs"] = [None if r in ids else r for r in c["refs"]]
        self.history = [h for h in self.history if h not in ids]

    def pin(self, i, on):
        """Pinning is ONE thing: the word `pinned` (plus the standing rule
        pinned => not in the trash, so it restores). No eviction from
        Output, no exclusivity - a pinned image may sit in a slot."""
        with self.lock:
            if not self.alive(i):
                return
            if on:
                self.restore([i])
                self.tag([i], add=[PINNED])
            else:
                self.tag([i], remove=[PINNED])

    def set_slots(self, n, tab=None):
        """The Output count of `tab` (default: the active tab). Candidates
        that fall off the end are not merely dropped: the FRESH ones are
        thrown away (a fresh image exists only as an output; off the sheet
        it is trash), touched ones are just displaced. Returns the ids
        thrown away."""
        with self.lock:
            self.state["slots"] = max(1, min(64, int(n)))
            lst = self.cands(tab)
            cut = lst[self.state["slots"]:]
            del lst[self.state["slots"]:]
            doomed = [q for q in cut if self.is_fresh(q) and not self.is_archived(q)]
            if doomed:
                self.archive(doomed)
                self.forget(doomed)
            self.save_state()
            return doomed

    def orphan_fresh(self):
        """INVARIANT: a fresh image is in some tab's Output list. Any fresh
        image that is not (a truncated list, an aborted round, a crash
        between landing and listing) is an orphan and must be thrown away
        quietly - it exists nowhere the user can see it."""
        listed = {q for lst in self.state["candidates"].values() for q in lst}
        return [i for i in self.alive_ids()
                if self.is_fresh(i) and not self.is_archived(i) and i not in listed]

    def begin_round(self, tab):
        with self.lock:
            self.state["candidates"][tab] = []
            self.save_state()
            return self.state["slots"]

    def add_candidate(self, tab, i):
        # no cap: a round lists EVERY image it lands (the cap used to drop
        # the tail silently when the count changed mid-round -> orphans)
        with self.lock:
            self.state["candidates"][tab].append(i)
            self.save_state()

    def note_round(self, base_seed, used_lora):
        with self.lock:
            self.state["last_base_seed"] = base_seed
            self.state["last_round_lora"] = bool(used_lora)
            self.save_state()

    def meta(self, i):
        r = self.images[i]
        return {"id": i, "w": r["w"], "h": r["h"], "source": r["source"], "fresh": self.is_fresh(i),
                "ts": r.get("ts"), "recipe": r.get("recipe"),
                "parents": r.get("parents"), "tags": list(r["tags"]),
                "description": r.get("description") or "",
                "dir": r["dir"], "missing": i in self.missing,
                "archived": self.is_archived(i), "purged": bool(r.get("purged")),
                "path": str(self.path(i))}

    def snapshot(self):
        with self.lock:
            s = self.state
            alive = self.alive_ids()
            pins = [i for i in alive if PINNED in self.images[i]["tags"]]
            ids = set().union(*s["candidates"].values()) | set(self.history) | \
                set(pins) | self.live_set()
            meta = {}
            for i in ids:
                r = self.images[i]
                meta[i] = {"w": r["w"], "h": r["h"], "source": r["source"],
                           "ts": r.get("ts"),
                           "recipe": r.get("recipe"), "parents": r.get("parents"),
                           "path": str(self.path(i))}
            return {"root_name": self.name,
                    "all_ids": alive,
                    "archived": self.archived_ids(),
                    "fresh": [i for i in alive if self.images[i].get("fresh")],
                    "missing": sorted(self.missing),
                    "rescan": self.last_rescan, "scan_busy": self.scan_busy,
                    "tags": {i: self.images[i]["tags"] for i in alive},
                    "descriptions": {i: self.images[i]["description"] for i in alive
                                     if self.images[i]["description"]},
                    "words": self.words(),
                    "cwd": self.cwd(), "dirs": self.dirs(),
                    "paths": {i: self.images[i]["dir"] for i in alive},
                    "working": s["working"], "slots": s["slots"],
                    "candidates": s["candidates"], "pins": pins,
                    "controls": s["controls"],
                    "history": [h for h in reversed(self.history) if self.alive(h)],
                    "meta": meta, "last_base_seed": s.get("last_base_seed")}
