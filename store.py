"""store.py - one project's image store: flat images/<id>.png (ids never
reused), an append-only journal.jsonl (image records with full recipe +
parent ids, hist / gc / prune events) and state.json (the live UI state:
working image, slots, per-tab candidates, pins, controls).

Images are IMMUTABLE. Lineage doctrine: a tree with a DAG overlay - parent 0
is the continuity carrier; co-parents are journal edges only. The journal is
the sole lineage authority (the png's `evolve` chunk carries NO ancestry).
Single-threaded access is guaranteed by `lock` (generation runs in a thread).
"""
import json
import threading
import time
from pathlib import Path

from comfy_client import INPUT_DIR
from controls import CONTROLS, sanitize_controls
from image_file import flattened_rgb, link_or_copy, open_bytes, sha1_of, write_png
from image_meta import evolve_chunk, glean_recipe
from project import project_color, project_dir, save_config

TABS = ("create", "derive", "camera")


def empty_candidates():
    return {t: [] for t in TABS}


class Store:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.name = self.dir.name                 # the project name
        self.images_dir = self.dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.journal = self.dir / "journal.jsonl"
        self.state_file = self.dir / "state.json"
        self.lock = threading.RLock()
        self.images = {}          # id -> record (incl. "gone" after gc)
        self.history = []         # ids, bred-from order (see hist_append)
        self.next_id = 1
        self.color = None
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
                    self.images[ev["id"]] = ev
                    self.next_id = max(self.next_id, ev["id"] + 1)
                elif t == "working" and ev["id"] is not None:
                    # legacy (pre-v2): every WI change, deduped each-once
                    if ev["id"] not in self.history:
                        self.history.append(ev["id"])
                elif t == "hist" and ev["id"] is not None:
                    if not self.history or self.history[-1] != ev["id"]:
                        self.history.append(ev["id"])
                elif t == "gc":
                    for i in ev["ids"]:
                        if i in self.images:
                            self.images[i]["gone"] = True
        if self.state_file.exists():
            self.state = json.loads(self.state_file.read_text(encoding="utf-8"))
        else:
            self.state = {"working": None, "slots": 6, "candidates": empty_candidates(),
                          "pins": [], "controls": dict(CONTROLS)}
        s = self.state
        s["controls"] = sanitize_controls(s.get("controls") or {}, self.alive)
        if isinstance(s.get("candidates"), list):
            # v3.1 migration: candidates become per-tab, routed by recipe op
            b = empty_candidates()
            for i in s["candidates"]:
                r = (self.images.get(i) or {}).get("recipe") or {}
                op = r.get("op") or ("derive" if (self.images.get(i) or {}).get("parents")
                                     else "create")
                b["camera" if op == "pov" else (op if op in b else "derive")].append(i)
            s["candidates"] = b
        s["candidates"] = {t: [i for i in s["candidates"].get(t, []) if self.alive(i)]
                           for t in TABS}
        s["pins"] = [i for i in s.get("pins", []) if self.alive(i)]
        # pinned images are never also candidates, in any tab
        s["candidates"] = {t: [i for i in v if i not in s["pins"]]
                           for t, v in s["candidates"].items()}
        if not self.alive(s.get("working")):
            s["working"] = None

    def _append(self, ev):
        ev["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.journal.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")

    def save_state(self):
        self.state_file.write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    # ---------- images ----------

    def alive(self, i):
        return i is not None and i in self.images and not self.images[i].get("gone")

    def alive_ids(self):
        return sorted(i for i in self.images if self.alive(i))

    def path(self, i):
        return self.images_dir / f"{i}.png"

    def rel(self, i):
        """The global image name '<project>/images/<id>.png'."""
        return f"{self.name}/images/{i}.png"

    def staged_path(self, i):
        return INPUT_DIR / "evolve" / self.name / f"{i}.png"

    def stage_ref(self, i):
        """Make store image `i` loadable by ComfyUI under a DURABLE name.

        LoadImage is sandboxed to the input dir, so the bytes must appear
        inside it - as one hardlink per image, addressed
        input/evolve/<project>/<id>.png, created once and left alone (the
        old overwritten slot names made every embedded payload
        unreproducible one round later). Nothing ever WRITES through the
        staged path; archive() removes the link with the image."""
        link_or_copy(self.path(i), self.staged_path(i))
        return f"evolve/{self.name}/{i}.png"

    def add_image(self, img, source, recipe=None, parents=None, sha1=None,
                  chunks=None, inputs=None):
        """Persist a PIL image (already RGB) under a fresh id.

        chunks: text chunks to PRESERVE from the render (ComfyUI's `prompt`
        payload - which the server augments with is_changed sha256s - and
        optionally `workflow` geometry). inputs: staged-name -> store-id map
        for the render's LoadImage refs. The `evolve` chunk carries UI state
        + that map; NO ancestry - the journal holds the parents."""
        with self.lock:
            i = self.next_id
            self.next_id += 1
            all_chunks = dict(chunks or {})
            all_chunks["evolve"] = evolve_chunk(self.name, i, source, recipe, inputs)
            write_png(img, self.path(i), all_chunks)
            rec = {"t": "image", "id": i, "file": f"{i}.png", "source": source,
                   "w": img.width, "h": img.height, "sha1": sha1,
                   "recipe": recipe, "parents": parents or []}
            self._append(rec)
            self.images[i] = rec
            return i

    def find_sha1(self, sha1):
        for i, r in self.images.items():
            if r.get("sha1") == sha1 and self.alive(i):
                return i
        return None

    def import_bytes(self, data, source="import"):
        """Persist pasted/dropped/fetched image bytes: dedupe by content
        hash, flatten alpha onto white. LAYER 0 (user rule 2026-08-24: NEVER
        destroy image metadata on import): every incoming text chunk is
        re-embedded verbatim (except a foreign `evolve` chunk - ours must
        win); glean_recipe() maps what it can onto UI fields, so picking an
        import restores at least its prompt. Returns (id, was_new)."""
        sha1 = sha1_of(data)
        existing = self.find_sha1(sha1)
        if existing is not None:
            return existing, False
        img, raw = open_bytes(data)
        recipe = glean_recipe(raw)
        keep = {k: v for k, v in raw.items() if k != "evolve"}
        i = self.add_image(flattened_rgb(img), source, recipe=recipe, sha1=sha1, chunks=keep)
        return i, True

    def archive(self, ids):
        """ARCHIVE, never delete: files move to <project>/archive/,
        restorable by hand. The staged hardlink must go too or the bytes
        never free. One journal event."""
        ids = [i for i in ids if self.alive(i)]
        if not ids:
            return 0
        arch = self.dir / "archive"
        arch.mkdir(exist_ok=True)
        for i in ids:
            try:
                self.path(i).rename(arch / f"{i}.png")
            except OSError:
                self.path(i).unlink(missing_ok=True)
            self.staged_path(i).unlink(missing_ok=True)
            self.images[i]["gone"] = True
        self._append({"t": "gc", "ids": ids})
        return len(ids)

    def journal_event(self, ev):
        self._append(ev)

    # ---------- live state ----------

    def set_working(self, i):
        """Set the WI. NO history side effect (v2): the WI is a cheap
        browsing target; an image enters History only by being BRED FROM."""
        with self.lock:
            if i is not None and not self.alive(i):
                raise ValueError(f"no such image {i}")
            self.state["working"] = i
            self.save_state()

    def hist_append(self, i):
        """History = log of images actually consumed by a generator:
        Generate appends its ref0, Camera its source. Consecutive duplicates
        collapse; a later re-use appends again."""
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
        """The candidates list for one tab (default: the active tab).
        Outputs belong to the tab that generated them."""
        return self.state["candidates"][tab if tab in TABS else self.active_tab()]

    def live_set(self):
        """Images in use right now: WI, ref0, refs (never swept)."""
        c = self.state["controls"]
        s = {self.state["working"], c.get("ref0"), *c["refs"]}
        s.discard(None)
        return s

    def forget(self, ids):
        """Drop ids from every live slot (candidates, WI, ref0, refs,
        history) - the cadence every archive path needs. Caller saves."""
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
        it if it is already a candidate anywhere or pinned)."""
        with self.lock:
            c = self.cands()
            for lst in self.state["candidates"].values():
                if i in lst:
                    lst.remove(i)
            if i in self.state["pins"]:
                self.state["pins"].remove(i)
            if index < len(c):
                c[index] = i
            else:
                c.append(i)
            del c[self.state["slots"]:]
            self.save_state()

    def pin(self, i, on, index=None):
        """P is THE keep decision. Pinning MOVES an image off the candidate
        sheet onto the Pinned sheet; unpinning drops it from the sheet."""
        with self.lock:
            pins = self.state["pins"]
            if on:
                if not self.alive(i):
                    return
                if i in pins:
                    pins.remove(i)
                for lst in self.state["candidates"].values():
                    if i in lst:
                        lst.remove(i)
                if index is None or index >= len(pins):
                    pins.append(i)
                else:
                    pins.insert(index, i)
            elif i in pins:
                pins.remove(i)
            self.save_state()

    def set_slots(self, n):
        with self.lock:
            self.state["slots"] = max(1, min(64, int(n)))
            del self.cands()[self.state["slots"]:]
            self.save_state()

    def begin_round(self, tab, controls=None):
        """Start of a generator round: (optionally) adopt the controls,
        clear the tab's outputs (after the caller swept them). Returns the
        number of slots to fill."""
        with self.lock:
            if controls is not None:
                self.state["controls"] = controls
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
        """The per-image record the UI needs (no gc verdict - see lineage)."""
        r = self.images[i]
        return {"id": i, "w": r["w"], "h": r["h"], "source": r["source"],
                "ts": r.get("ts"), "recipe": r.get("recipe"),
                "parents": r.get("parents"), "gone": bool(r.get("gone")),
                "path": str(self.path(i))}

    def snapshot(self):
        """This store's part of the UI state snapshot."""
        with self.lock:
            s = self.state
            ids = set().union(*s["candidates"].values()) | set(self.history) | \
                set(s["pins"]) | self.live_set()
            meta = {}
            for i in ids:
                r = self.images[i]
                meta[i] = {"w": r["w"], "h": r["h"], "source": r["source"],
                           "ts": r.get("ts"),
                           "recipe": r.get("recipe"), "parents": r.get("parents"),
                           "path": str(self.path(i))}
            return {"project": self.name, "color": self.color,
                    "all_ids": self.alive_ids(),
                    "working": s["working"], "slots": s["slots"],
                    "candidates": s["candidates"], "pins": s["pins"],
                    "controls": s["controls"],
                    "history": [h for h in reversed(self.history) if self.alive(h)],
                    "meta": meta, "last_base_seed": s.get("last_base_seed")}


def open_project(name):
    """Create-or-open a project. Its total and single effect is the path
    context: <root>/<name>/images/NNN.png etc.; state/journal swap because
    those files live under the path (v4 doctrine)."""
    d = project_dir(name)                      # validates the name
    (d / "images").mkdir(parents=True, exist_ok=True)
    store = Store(d)
    store.color = project_color(name)
    save_config(last_project=name)
    return store
