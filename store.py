"""store.py - THE image store: flat images/<id>.png (ids never reused), an
append-only journal.jsonl and state.json (the live UI state), all under
the root.

Images are IMMUTABLE; records carry mutable TAGS (words the user puts on
images) and a DESCRIPTION (1-1 with the image). Lineage doctrine: a tree
with a DAG overlay - parent 0 is the continuity carrier; co-parents are
journal edges only. The journal is the sole lineage authority.

Tags (2026-08-25) are COLLECTIONS the user curates; the code reads two of
them (`pinned`, `lora_dataset_*`) because they are collections with
consequences, and owns none. ARCHIVED IS A BIT, NOT A WORD (decision
2026-08-25: "don't go crazy with tags unless they make the code more
maintainable; keep them as a layer on top of some fixed rules") - a
record flag with `archive` / `restore` journal events, like `purged`.
Rule: pinned => not archived (pinning restores; archiving unpins only when
forced). At birth a child copies its MOTHER's words minus `pinned`, plus
the settings' default tags. Cascade (add to / remove from descendants
along parent-0 edges) is one explicit operation, one journal event.
Single-threaded access is guaranteed by `lock`.
"""
import json
import threading
import time
from pathlib import Path

from comfy_client import INPUT_DIR
from controls import CONTROLS, sanitize_controls
from image_file import flattened_rgb, link_or_copy, open_bytes, sha1_of, write_png
from image_meta import evolve_chunk, glean_recipe
from project import clean_tags, load_settings

TABS = ("create", "derive", "camera")
PINNED = "pinned"
NOT_INHERITED = {PINNED}
LEGACY_ARCHIVED_WORD = "archived"     # pre-bit journals carried it as a word


def empty_candidates():
    return {t: [] for t in TABS}


class Store:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.name = self.dir.name
        self.images_dir = self.dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.journal = self.dir / "journal.jsonl"
        self.state_file = self.dir / "state.json"
        self.lock = threading.RLock()
        self.images = {}          # id -> record (tags: list, description: str, purged?)
        self.history = []         # ids, bred-from order (see hist_append)
        self.next_id = 1
        self._load()

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
                    ev["archived"] = bool(ev.get("archived")) or LEGACY_ARCHIVED_WORD in tags
                    ev["tags"] = clean_tags(tags)
                    ev["description"] = ev.get("description") or ""
                    self.images[ev["id"]] = ev
                    self.next_id = max(self.next_id, ev["id"] + 1)
                elif t == "tag":
                    add, rm = ev.get("add") or [], ev.get("remove") or []
                    for i in ev.get("ids", []):
                        r = self.images.get(i)
                        if r:
                            self._apply_tags(r, add, rm)
                            if LEGACY_ARCHIVED_WORD in add:       # pre-bit journals
                                r["archived"] = True
                            elif LEGACY_ARCHIVED_WORD in rm:
                                r["archived"] = False
                elif t in ("archive", "restore"):
                    for i in ev["ids"]:
                        if i in self.images:
                            self.images[i]["archived"] = t == "archive"
                elif t == "describe":
                    r = self.images.get(ev["id"])
                    if r:
                        r["description"] = ev.get("description") or ""
                elif t == "working" and ev["id"] is not None:
                    if ev["id"] not in self.history:      # legacy (pre-v2)
                        self.history.append(ev["id"])
                elif t == "hist" and ev["id"] is not None:
                    if not self.history or self.history[-1] != ev["id"]:
                        self.history.append(ev["id"])
                elif t == "gc":                            # legacy archive event
                    for i in ev["ids"]:
                        if i in self.images:
                            self.images[i]["archived"] = True
                elif t == "purge":
                    for i in ev["ids"]:
                        if i in self.images:
                            self.images[i]["purged"] = True
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

    # ---------- images ----------

    def alive(self, i):
        """Exists and not purged (archived images ARE alive - hidden by a flag)."""
        return i is not None and i in self.images and not self.images[i].get("purged")

    def alive_ids(self):
        return sorted(i for i in self.images if self.alive(i))

    def tags(self, i):
        return self.images[i]["tags"] if self.alive(i) else []

    def has(self, i, word):
        return word in self.tags(i)

    def is_archived(self, i):
        return self.alive(i) and bool(self.images[i].get("archived"))

    def archived_ids(self):
        return [i for i in self.alive_ids() if self.images[i].get("archived")]

    def with_word(self, word):
        return [i for i in self.alive_ids() if word in self.images[i]["tags"]]

    def pins(self):
        return self.with_word(PINNED)

    def words(self):
        """{word: count} over alive images."""
        out = {}
        for i in self.alive_ids():
            for w in self.images[i]["tags"]:
                out[w] = out.get(w, 0) + 1
        return out

    def path(self, i):
        return self.images_dir / f"{i}.png"

    def staged_path(self, i):
        return INPUT_DIR / "evolve" / self.name / f"{i}.png"

    def stage_ref(self, i):
        """Make image `i` loadable by ComfyUI under a DURABLE name: one
        hardlink per image at input/evolve/<root-name>/<id>.png, created
        once and left alone. Nothing ever WRITES through the staged path."""
        link_or_copy(self.path(i), self.staged_path(i))
        return f"evolve/{self.name}/{i}.png"

    def birth_tags(self, mother=None):
        """A new image's words: the mother's minus pinned, plus the
        settings' default tags (fiat/import: default tags only)."""
        inherited = [w for w in self.tags(mother) if w not in NOT_INHERITED] if mother else []
        return clean_tags(inherited + load_settings()["default_tags"])

    def add_image(self, img, source, recipe=None, parents=None, sha1=None,
                  chunks=None, inputs=None, tags=None, description=None, ts=None):
        """Persist a PIL image (already RGB) under a fresh id, with its
        words. chunks: text chunks to PRESERVE; inputs: staged-name ->
        store-id map. The `evolve` chunk carries UI state + that map, NO
        ancestry - the journal holds the parents. description defaults to
        the recipe prompt (seeded, never cascaded)."""
        with self.lock:
            i = self.next_id
            self.next_id += 1
            all_chunks = dict(chunks or {})
            all_chunks["evolve"] = evolve_chunk(self.name, i, source, recipe, inputs)
            write_png(img, self.path(i), all_chunks)
            if description is None:
                description = ((recipe or {}).get("prompt") or "").strip()
            rec = {"t": "image", "id": i, "file": f"{i}.png", "source": source,
                   "w": img.width, "h": img.height, "sha1": sha1,
                   "recipe": recipe, "parents": parents or [],
                   "tags": clean_tags(tags or []), "description": description,
                   "archived": False}
            if ts:
                rec["ts_origin"] = ts
            self._append(rec)
            self.images[i] = rec
            return i

    def find_sha1(self, sha1):
        for i, r in self.images.items():
            if r.get("sha1") == sha1 and self.alive(i):
                return i
        return None

    def import_bytes(self, data, source="import", tags=None):
        """Persist pasted/dropped/fetched image bytes: dedupe by content
        hash, flatten alpha onto white, PRESERVE every text chunk (layer 0)
        except a foreign `evolve` chunk, glean a recipe. Words = default
        tags (+ any given). Returns (id, was_new)."""
        sha1 = sha1_of(data)
        existing = self.find_sha1(sha1)
        if existing is not None:
            if tags:
                self.tag([existing], add=tags)
            return existing, False
        img, raw = open_bytes(data)
        recipe = glean_recipe(raw)
        keep = {k: v for k, v in raw.items() if k != "evolve"}
        i = self.add_image(flattened_rgb(img), source, recipe=recipe, sha1=sha1, chunks=keep,
                           tags=self.birth_tags() + list(tags or []))
        return i, True

    # ---------- tags & descriptions ----------

    def descendants(self, i):
        """The mother-line subtree under i (parent-0 edges only), excluding
        i, alive only."""
        kids = {}
        for j, r in self.images.items():
            if not self.alive(j):
                continue
            ps = r.get("parents") or []
            if ps:
                kids.setdefault(ps[0], []).append(j)
        out, todo = [], list(kids.get(i, []))
        seen = {i}
        while todo:
            x = todo.pop()
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
            todo.extend(kids.get(x, []))
        return sorted(out)

    def tag(self, ids, add=(), remove=(), cascade=False):
        """Add/remove words on ids; with cascade also on their descendants
        (ADD skips archived descendants, REMOVE does not). One journal
        event listing every id touched. Returns the touched ids."""
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
                a = add if i in add_ids else []
                rm = remove if i in rm_ids else []
                before = list(r["tags"])
                self._apply_tags(r, a, rm)
                if r["tags"] != before:
                    touched.append(i)
            if touched:
                self._append({"t": "tag", "ids": touched, "add": add, "remove": remove,
                              "cascade": bool(cascade), "from": base})
            return touched

    def describe(self, i, text):
        with self.lock:
            if not self.alive(i):
                return False
            text = (text or "").strip()
            if self.images[i]["description"] != text:
                self.images[i]["description"] = text
                self._append({"t": "describe", "id": i, "description": text})
            return True

    # ---------- archived: a bit, not a word ----------

    def archive(self, ids, force=False):
        """Set the archived flag. Pinned images are SKIPPED unless force,
        which unpins them first (pinned => not archived). Returns the ids
        actually archived; one journal event."""
        with self.lock:
            todo = [i for i in ids if self.alive(i) and not self.images[i].get("archived")]
            if not force:
                todo = [i for i in todo if not self.has(i, PINNED)]
            else:
                pinned = [i for i in todo if self.has(i, PINNED)]
                if pinned:
                    self.tag(pinned, remove=[PINNED])
            for i in todo:
                self.images[i]["archived"] = True
            if todo:
                self._append({"t": "archive", "ids": todo})
            return todo

    def restore(self, ids):
        with self.lock:
            todo = [i for i in ids if self.is_archived(i)]
            for i in todo:
                self.images[i]["archived"] = False
            if todo:
                self._append({"t": "restore", "ids": todo})
            return todo

    def ancestors_closure(self, ids):
        keep, todo = set(), list(ids)
        while todo:
            i = todo.pop()
            if i in keep or not self.alive(i):
                continue
            keep.add(i)
            todo.extend(self.images[i].get("parents") or [])
        return keep

    def garbage(self):
        """Archived AND not an ancestor of any unarchived image - the only
        integrity rule that survives: provenance closure of what is live."""
        live = [i for i in self.alive_ids() if not self.is_archived(i)]
        keep = self.ancestors_closure(live)
        return [i for i in self.alive_ids() if self.is_archived(i) and i not in keep]

    def purge(self):
        """DELETE the files of garbage images (+ staged links). Explicit,
        irreversible; the journal keeps their records (purged)."""
        with self.lock:
            ids = self.garbage()
            for i in ids:
                self.path(i).unlink(missing_ok=True)
                self.staged_path(i).unlink(missing_ok=True)
                self.images[i]["purged"] = True
                self.images[i]["archived"] = False
            if ids:
                self._append({"t": "purge", "ids": ids})
            self.forget(ids)
            self.save_state()
            return ids

    # ---------- live state ----------

    def set_working(self, i):
        """Set the WI. NO history side effect (v2): an image enters History
        only by being BRED FROM."""
        with self.lock:
            if i is not None and not self.alive(i):
                raise ValueError(f"no such image {i}")
            self.state["working"] = i
            self.save_state()

    def hist_append(self, i):
        """History = log of images actually consumed by a generator:
        Generate appends its ref0, Camera its source; consecutive dupes
        collapse."""
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
        """Images in use right now: WI, ref0, refs (never swept)."""
        c = self.state["controls"]
        s = {self.state["working"], c.get("ref0"), *c["refs"]}
        s.discard(None)
        return s

    def forget(self, ids):
        """Drop ids from every live slot (candidates, WI, ref0, refs,
        history). Caller saves."""
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

    def place_slot(self, i, index):
        """Put image i into the ACTIVE tab's candidate slot `index` (moving
        it if it is already a candidate anywhere; unpinning it)."""
        with self.lock:
            c = self.cands()
            for lst in self.state["candidates"].values():
                if i in lst:
                    lst.remove(i)
            if self.has(i, PINNED):
                self.tag([i], remove=[PINNED])
            if index < len(c):
                c[index] = i
            else:
                c.append(i)
            del c[self.state["slots"]:]
            self.save_state()

    def pin(self, i, on):
        """P is THE keep decision: the word `pinned`. Pinning MOVES an image
        off the candidate sheet; unpinning drops the word."""
        with self.lock:
            if not self.alive(i):
                return
            if on:
                for lst in self.state["candidates"].values():
                    if i in lst:
                        lst.remove(i)
                self.restore([i])                       # pinned => not archived
                self.tag([i], add=[PINNED])
                self.save_state()
            else:
                self.tag([i], remove=[PINNED])

    def set_slots(self, n):
        with self.lock:
            self.state["slots"] = max(1, min(64, int(n)))
            del self.cands()[self.state["slots"]:]
            self.save_state()

    def begin_round(self, tab):
        with self.lock:
            self.state["candidates"][tab] = []
            self.save_state()
            return self.state["slots"]

    def add_candidate(self, tab, i):
        with self.lock:
            lst = self.state["candidates"][tab]
            if len(lst) < self.state["slots"]:
                lst.append(i)
            self.save_state()

    def note_round(self, base_seed, used_lora):
        with self.lock:
            self.state["last_base_seed"] = base_seed
            self.state["last_round_lora"] = bool(used_lora)
            self.save_state()

    def meta(self, i):
        r = self.images[i]
        return {"id": i, "w": r["w"], "h": r["h"], "source": r["source"],
                "ts": r.get("ts"), "recipe": r.get("recipe"),
                "parents": r.get("parents"), "tags": list(r["tags"]),
                "description": r.get("description") or "",
                "archived": bool(r.get("archived")), "purged": bool(r.get("purged")),
                "path": str(self.path(i))}

    def snapshot(self):
        """This store's part of the UI state snapshot."""
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
                    "archived": [i for i in alive if self.images[i].get("archived")],
                    "tags": {i: self.images[i]["tags"] for i in alive},
                    "descriptions": {i: self.images[i]["description"] for i in alive
                                     if self.images[i]["description"]},
                    "words": self.words(),
                    "working": s["working"], "slots": s["slots"],
                    "candidates": s["candidates"], "pins": pins,
                    "controls": s["controls"],
                    "history": [h for h in reversed(self.history) if self.alive(h)],
                    "meta": meta, "last_base_seed": s.get("last_base_seed")}
