"""evolve.py - the Evolver: an always-on image editor for iterative generation.

    python evolve.py --root softlock_store --port 8189   ->  http://127.0.0.1:8189/

The model (user's design, 2026-08-21):
  * ONE preview image (or none) - "working image" in the internal state and
    the older notes; renamed in the UI 2026-08-22. You improve it; I keep
    its history.
  * N candidate SLOTS (N = whatever you type). "Generate" clears every
    unpinned slot and refills them from the controls AS THEY ARE NOW:
    prompt, references, LoRA + strength, lock, vary, seed. ref0 is a real
    fourth slot and the continuity carrier; empty = fiat.
  * TWO generators, each with its own button (2026-08-22): "Generate", and
    "Adjust POV" - a camera move (elevation + distance) budded off the
    preview image. Adjust POV shares NO inputs with the left panel, not
    even the prompt: fixed recipe, see POV_ELEV/POV_DIST below. Both panels
    disable while either is running. Candidate slots are still shared;
    per-generator strips are deferred pending wider layout work.
  * Picking a candidate (double-click / Enter / drag onto the stage) makes
    it the preview image. Candidates stay until the next Generate.
  * PINS survive Generate. Unpinned candidates are RETIRED, never deleted:
    the Retired drawer lists them, drag one back into a slot to recall.
  * HISTORY is a timeline of every image that was ever the working image,
    each once, in first-seen order; click any thumb to make it working
    again - generating from there appends, nothing forward is lost.
  * GC is a button: purges images that are not history, not pinned, not in
    a slot, not a reference, and not an ancestor of any of those.
  * Clipboard / drag-drop are first class: drop or paste from Explorer,
    browsers, Paint, Photoshop into any image box; Ctrl-C copies the
    focused image as PNG (+ its path as text); Ctrl-X copies and clears;
    Del clears. Within the app every thumbnail drags into every box.

Storage (mine, not yours): <root>/images/<id>.png (ids never reused),
<root>/journal.jsonl (append-only: image records with full recipe +
parents, working-image events, gc events), <root>/state.json (the live
UI state: working, slots, pins, controls). Provenance is complete - each
candidate records its mother/extras ids - the UI just never draws a tree.

Generation is the proven Klein graph (pose_from_char/template_klein):
ReferenceLatent chains + IdentityFeatureTransferFinal on reference 0,
per-candidate seed = base + i, `vary` = latent jitter on ref 0, optional
LoRA "gene". Each candidate is its own submission; per-step progress
streams on this console (pose_from_char.wait) and the slot fills in the
UI as each lands. Generation ONLY on your click. ComfyUI must be up -
never launched from here.
"""
import argparse
import asyncio
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path

from aiohttp import web
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from api_to_ui import convert_payload

HERE = Path(__file__).resolve().parent   # the akasutils dir (trainer scripts live here)

from qwen_vp_probe import AZIMUTH as VP_AZ
from qwen_vp_probe import DISTANCE as VP_DI
from qwen_vp_probe import ELEVATION as VP_EL
from qwen_vp_probe import build_graph as build_vp_graph
from pose_from_char import (COMFY, PRESETS, SCRATCH_PREFIX, TEMPLATE,
                            WHITE_BG_PREFIX, free_vram, queue, snap16, wait)

LOCKS = ["SOFT_LOCK", "MID_LOCK", "HARD_LOCK"]
# MODEL FAMILIES. Klein is the only one with references (ReferenceLatent +
# IFT); the others are FIAT-only: the family dropdown is live only while the
# stage and the ref slots are empty (server-enforced). Illustrious and
# Z-Image checkpoints are hard-coded per the user (2026-08-21).
FAMILIES = {
    "klein": {"label": "Flux2 Klein", "steps": 4, "cfg": 1.0},
    "zimage": {"label": "Z-Image Turbo", "steps": 20, "cfg": 2.0,
               "dit": "z_image_turbo_bf16.safetensors", "te": "qwen_3_4b.safetensors",
               "vae": "ae.safetensors"},
    "illustrious": {"label": "Illustrious (WAI v16)", "steps": 28, "cfg": 6.0,
                    "ckpt": "waiIllustriousSDXL_v160.safetensors",
                    "sampler": "euler_ancestral", "scheduler": "normal"},
}
DEFAULT_CONTROLS = {"prompt": "", "negative": "", "family": "klein",
                    "ref0": None, "refs": [None, None, None],
                    "lora": "", "lora_strength": 1.0, "lock": "SOFT_LOCK",
                    "vary": 0.0, "seed": 0, "whitebg": True,
                    "width": 1024, "height": 1024, "fresh_model": False,
                    "steps": 0, "cfg": 0,          # 0 = the family's default
                    "tab": "create",               # active action-panel tab (v3)
                    "seed_create": 0, "seed_derive": 0, "seed_camera": 0,
                    "outputs_create": 6, "outputs_derive": 6, "outputs_camera": 4,
                    # camera axes: None = axis unchecked, its token is OMITTED
                    "pov_azim": None, "pov_elev": None, "pov_dist": None}

# CAMERA tab — a SECOND generator with a FIXED recipe (qwen-image-edit-2511
# + multiple-angles LoRA + Lightning, 4 steps, cfg 1.0) that ignores the
# other tabs' controls including the prompt. Budding on the WI.
#
# SEMANTICS CORRECTED 2026-08-23 (user's chained experiments): the tokens
# SET an absolute camera spec in the subject's frame - they do not delta it.
# "close-up" on an already-close image is a NO-OP. So the trained names are
# the labels (the earlier comparative Lower/Closer labels were wrong), and
# each axis has a CHECKBOX: unchecked = token omitted from the prompt.
# Whether the LoRA degrades on partial grammars is UNTESTED - the first
# partial-axis run doubles as the probe.
POV_AZIM = [("front", "front"), ("front-right", "front-right"),
            ("right", "right"), ("back-right", "back-right"),
            ("back", "back"), ("back-left", "back-left"),
            ("left", "left"), ("front-left", "front-left")]
POV_ELEV = [("low", "low-angle"), ("eye", "eye-level"),
            ("elevated", "elevated"), ("high", "high-angle")]
POV_DIST = [("close", "close-up"), ("medium", "medium"), ("wide", "wide")]
POV_STEPS, POV_CFG, POV_STRENGTH = 4, 1.0, 1.0


# ---------- store ----------

class Store:
    """Flat image store + append-only journal + live state. Single-threaded
    access is guaranteed by `lock` (the generate job runs in a thread)."""

    def __init__(self, root):
        self.root = root
        self.images_dir = root / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.journal = root / "journal.jsonl"
        self.state_file = root / "state.json"
        self.lock = threading.RLock()
        self.images = {}          # id -> record (incl. "gone" after gc)
        self.history = []         # ids, first-seen order
        self.next_id = 1
        self.busy = None          # {"total": n, "done": k, "tab": t} while generating
        self.abort = False        # set by /api/abort; checked between candidates
        self._load()

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
            self.state = {"working": None, "slots": 6,
                          "candidates": {"create": [], "derive": [], "camera": []},
                          "pins": [], "controls": dict(DEFAULT_CONTROLS)}
        c = dict(DEFAULT_CONTROLS)
        c.update({k: v for k, v in self.state.get("controls", {}).items()
                  if k in DEFAULT_CONTROLS})
        self.state["controls"] = c
        if not self.alive(c.get("ref0")):
            c["ref0"] = None
        # drop references to images that no longer exist
        if isinstance(self.state.get("candidates"), list):
            # v3.1 migration: candidates become per-tab, routed by recipe op
            b = {"create": [], "derive": [], "camera": []}
            for i in self.state["candidates"]:
                r = (self.images.get(i) or {}).get("recipe") or {}
                op = r.get("op") or ("derive" if (self.images.get(i) or {}).get("parents")
                                     else "create")
                b["camera" if op == "pov" else (op if op in b else "derive")].append(i)
            self.state["candidates"] = b
        self.state["candidates"] = {t: [i for i in v if self.alive(i)]
                                    for t, v in self.state["candidates"].items()}
        self.state["pins"] = [i for i in self.state["pins"] if self.alive(i)]
        # pinned images are never also candidates, in any tab
        self.state["candidates"] = {t: [i for i in v if i not in self.state["pins"]]
                                    for t, v in self.state["candidates"].items()}
        if not self.alive(self.state["working"]):
            self.state["working"] = None
        c["refs"] = [i if self.alive(i) else None for i in (c.get("refs") or [])][:3]
        c["refs"] += [None] * (3 - len(c["refs"]))

    def _append(self, ev):
        ev["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.journal.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")

    def save_state(self):
        self.state_file.write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    def alive(self, i):
        return i is not None and i in self.images and not self.images[i].get("gone")

    def path(self, i):
        return self.images_dir / f"{i}.png"

    def stage_ref(self, i):
        """Make store image `i` loadable by ComfyUI under a DURABLE name.

        LoadImage is sandboxed to the input dir (get_annotated_filepath
        containment), so the bytes must appear inside it - but the old
        overwritten slot names (evolver_ref0.png) made every embedded
        payload unreproducible one round later. Instead: one hardlink per
        image, addressed input/evolve/<store-uid>/<id>.png, created once and
        left alone. os.link passes the containment check honestly (realpath
        is a no-op on hardlinks - verified) and costs no disk; copy2 is the
        cross-volume / exotic-fs fallback. Nothing ever WRITES through the
        staged path. GC removes the link with the image."""
        name = f"evolve/{self.root.name}/{i}.png"
        dest = COMFY / "input" / "evolve" / self.root.name / f"{i}.png"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(self.path(i), dest)
            except OSError:
                shutil.copy2(self.path(i), dest)
        return name

    def staged_path(self, i):
        return COMFY / "input" / "evolve" / self.root.name / f"{i}.png"

    def add_image(self, img, source, recipe=None, parents=None, sha1=None,
                  chunks=None, inputs=None):
        """Persist a PIL image (flattened to RGB) under a fresh id.

        chunks: text chunks to PRESERVE from the render (ComfyUI's `prompt`
        payload - which the server augments with is_changed sha256s - and
        optionally `workflow` geometry). A bare img.save() used to DESTROY
        these; PngInfo re-embeds them explicitly.
        inputs: staged-name -> store-id map for the render's LoadImage refs.
        The `evolve` chunk carries UI state + that map; NO ancestry - the
        journal is the sole lineage authority (up to 4 parents there)."""
        with self.lock:
            i = self.next_id
            self.next_id += 1
            info = PngInfo()
            for k, v in (chunks or {}).items():
                if isinstance(v, str):
                    info.add_text(k, v)
            info.add_text("evolve", json.dumps(
                {"v": 1, "project": self.root.name, "id": i,
                 "source": source, "recipe": recipe, "inputs": inputs or {}}))
            img.save(self.path(i), "PNG", pnginfo=info)
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

    def set_working(self, i):
        """Set the WI. NO history side effect (v2, 2026-08-23): the WI is a
        cheap browsing target; an image enters History only by being BRED
        FROM - see hist_append()."""
        with self.lock:
            if i is not None and not self.alive(i):
                raise ValueError(f"no such image {i}")
            self.state["working"] = i
            self.save_state()

    def hist_append(self, i):
        """History = reverse-chron log of images actually consumed by a
        generator: Generate appends its ref0, Adjust POV its source
        (settled 2026-08-23 - the user often has ref0 and the WI showing
        different images). Consecutive duplicates collapse; a later re-use
        appends again."""
        with self.lock:
            if i is None or not self.alive(i):
                return
            if self.history and self.history[-1] == i:
                return
            self.history.append(i)
            self._append({"t": "hist", "id": i})

    def cands(self, tab=None):
        """The candidates list for one tab (default: the active tab).
        Outputs belong to the tab that generated them - switching tabs must
        never show another run's candidates."""
        c = self.state["candidates"]
        tab = tab or self.state["controls"].get("tab") or "create"
        return c[tab if tab in c else "create"]

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
        """Board = working memory. Pinning MOVES an image off the sheet onto
        the board (slot freed); unpinning drops it from the board (it shows
        up in Discarded unless held elsewhere)."""
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

    def protected(self):
        """GC roots (v2, 2026-08-23): pins (+ future story-asset promotions)
        and the LIVE WORKING SET (WI, ref0, refs, candidates). History is
        NOT a root. Every root locks its reference images TRANSITIVELY -
        provenance (user mandate)."""
        roots = set(self.state["pins"]) | \
            set().union(*self.state["candidates"].values()) | \
            {self.state["working"], self.state["controls"].get("ref0")} | \
            set(self.state["controls"]["refs"])
        roots.discard(None)
        # v4: asset-dataset entries that live in THIS project are roots too
        # (assets.json is root-global; every project roots its own images
        # whenever its GC runs)
        pref = self.root.name + "/images/"
        for a in load_assets():
            for e in a.get("dataset", []):
                q = str(e.get("path", ""))
                if q.startswith(pref) and q.endswith(".png"):
                    try:
                        roots.add(int(q[len(pref):-4]))
                    except ValueError:
                        pass
        keep, todo = set(), list(roots)
        while todo:
            i = todo.pop()
            if i in keep or not self.alive(i):
                continue
            keep.add(i)
            todo.extend(self.images[i].get("parents") or [])
        return keep

    def archive(self, ids):
        """ARCHIVE, never delete: files move to <project>/archive/, restorable
        by hand; referenced spots render placeholder tiles. The staged
        hardlink must go too or the bytes never free. One journal event."""
        ids = [i for i in ids if self.alive(i)]
        if not ids:
            return 0
        arch = self.root / "archive"
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

    def gc(self):
        """Manual sweep of everything unreachable from the roots."""
        with self.lock:
            keep = self.protected()
            doomed = [i for i in self.images if self.alive(i) and i not in keep]
            n = self.archive(doomed)
            return {"removed": n, "kept": len(keep)}

    def root_reason(self, i):
        """Why image i must NOT be archived - or None if it may. Only DURABLE
        roots count (pins, asset datasets; gids later) and their ancestor
        chains: that is referential integrity. The live working set is NOT a
        reason - discarding the WI just clears the stage."""
        why = {}
        for q in self.state["pins"]:
            why[q] = "pinned"
        pref = self.root.name + "/images/"
        for a in load_assets():
            for e in a.get("dataset", []):
                path = str(e.get("path", ""))
                if path.startswith(pref) and path.endswith(".png"):
                    try:
                        why.setdefault(int(path[len(pref):-4]), f"in asset {a['name']}")
                    except ValueError:
                        pass
        if i in why:
            return why[i]
        seen = set()
        for r0, label in list(why.items()):
            todo = [r0]
            while todo:
                x = todo.pop()
                if x in seen or not self.alive(x):
                    continue
                seen.add(x)
                for par in (self.images[x].get("parents") or []):
                    if par == i:
                        return f"ancestor of #{r0} ({label})"
                    todo.append(par)
        return None

    def discard(self, i):
        """Shift+Del: this image is trash, now. Quietly archives it and clears
        it from every live slot; refuses ONLY when referential integrity
        would break (returns the reason)."""
        with self.lock:
            if not self.alive(i):
                return "not found"
            why = self.root_reason(i)
            if why:
                return why
            s = self.state
            for lst in s["candidates"].values():
                if i in lst:
                    lst.remove(i)
            if s["working"] == i:
                s["working"] = None
            c = s["controls"]
            if c.get("ref0") == i:
                c["ref0"] = None
            c["refs"] = [None if r == i else r for r in c["refs"]]
            self.history = [h for h in self.history if h != i]
            self.archive([i])
            self.save_state()
            return None

    def prune_plan(self, root_id, force=False):
        """PRUNE (user doctrine 2026-08-24): cleanup is per BRANCH. The branch
        is the mother-line subtree under root_id (parent 0 - the tree of the
        lineage doctrine; co-parent edges are overlay and NOT followed).
        SAFE (default): archive every branch member not on a path to a
        durable root (pin / asset entry) - "dead twigs"; the rooted ones and
        their spine stay. FORCE: durable roots INSIDE the branch are un-marked
        (unpinned, asset entries removed) and go too; only images that
        rooted things OUTSIDE the branch still reference are kept - integrity
        for rooted images is never broken. Returns the plan; apply=False."""
        s = self.state
        if not self.alive(root_id):
            return None
        kids0 = {}
        for j, r in self.images.items():
            if not self.alive(j):
                continue
            ps = r.get("parents") or []
            if ps:
                kids0.setdefault(ps[0], []).append(j)
        branch, todo = set(), [root_id]
        while todo:
            x = todo.pop()
            if x in branch:
                continue
            branch.add(x)
            todo.extend(kids0.get(x, []))
        pins = set(s["pins"])
        pref = self.root.name + "/images/"
        aset = {}
        for a in load_assets():
            for e in a.get("dataset", []):
                q = str(e.get("path", ""))
                if q.startswith(pref) and q.endswith(".png"):
                    try:
                        aset.setdefault(int(q[len(pref):-4]), []).append((a["name"], q))
                    except ValueError:
                        pass
        durable = pins | set(aset)
        roots = (durable - branch) if force else durable
        keep, todo = set(), list(roots)
        while todo:
            x = todo.pop()
            if x in keep or not self.alive(x):
                continue
            keep.add(x)
            todo.extend(self.images[x].get("parents") or [])
        archive = sorted(branch - keep)
        kept = sorted(branch & keep)
        why = {}
        for k in kept:
            if k in pins and not force:
                why[k] = "pinned"
            elif k in aset and not force:
                why[k] = "in asset " + aset[k][0][0]
            else:
                why[k] = ("referenced by a rooted image outside the branch"
                          if force else "spine to a rooted image")
        outside_refs = sum(1 for j, r in self.images.items()
                           if self.alive(j) and j not in branch
                           and any(q in branch for q in (r.get("parents") or [])))
        c = s["controls"]
        return {"root": root_id, "branch": len(branch), "archive": archive,
                "keep": [{"id": k, "why": why[k]} for k in kept],
                "unpin": [i for i in archive if i in pins],
                "asset_removals": [{"asset": a, "path": q}
                                   for i in archive for a, q in aset.get(i, [])],
                "live": {"working": s["working"] in archive,
                         "ref0": c.get("ref0") in archive,
                         "refs": sum(1 for r in c["refs"] if r in archive)},
                "outside_refs": outside_refs, "force": force}

    def prune_apply(self, root_id, force=False):
        with self.lock:
            plan = self.prune_plan(root_id, force)
            if not plan or not plan["archive"]:
                return plan
            ids = set(plan["archive"])
            s = self.state
            for i in plan["unpin"]:
                if i in s["pins"]:
                    s["pins"].remove(i)
            if plan["asset_removals"]:
                assets = load_assets()
                gone_paths = {r["path"] for r in plan["asset_removals"]}
                for a in assets:
                    a["dataset"] = [e for e in a["dataset"] if e["path"] not in gone_paths]
                save_assets(assets)
            for lst in s["candidates"].values():
                lst[:] = [q for q in lst if q not in ids]
            if s["working"] in ids:
                s["working"] = None
            c = s["controls"]
            if c.get("ref0") in ids:
                c["ref0"] = None
            c["refs"] = [None if r in ids else r for r in c["refs"]]
            self.history = [h for h in self.history if h not in ids]
            self.archive(plan["archive"])
            self._append({"t": "prune", "root": root_id, "ids": plan["archive"],
                          "unpinned": plan["unpin"],
                          "assets": plan["asset_removals"], "force": force})
            self.save_state()
            print(f"pruned #{root_id}: {len(ids)} archived"
                  + (f", {len(plan['unpin'])} unpinned" if plan["unpin"] else "")
                  + (f", {len(plan['asset_removals'])} asset entries removed"
                     if plan["asset_removals"] else ""))
            return plan

    def sweep(self, tab):
        """TRASH DOCTRINE (user, 2026-08-24): P is THE keep decision. Before a
        round, that tab's candidates the user did not pin are silently
        archived (usable-to-crap is ~1:100). Never those in use (WI, ref0,
        refs), never anything integrity depends on (root_reason)."""
        s = self.state
        live = {s["working"], s["controls"].get("ref0"), *s["controls"]["refs"]}
        doomed = [q for q in s["candidates"].get(tab, [])
                  if q not in live and self.root_reason(q) is None]
        n = self.archive(doomed)
        if n:
            print(f"swept {n} unkept candidate(s) from {tab}")
        return n

    def snapshot(self):
        with self.lock:
            s = self.state
            ids = set().union(*s["candidates"].values()) | set(self.history) | set(s["pins"]) | \
                {s["working"], s["controls"].get("ref0")} | \
                set(s["controls"]["refs"])
            ids.discard(None)
            meta = {}
            for i in ids:
                r = self.images[i]
                meta[i] = {"w": r["w"], "h": r["h"], "source": r["source"],
                           "ts": r.get("ts"),
                           "recipe": r.get("recipe"), "parents": r.get("parents"),
                           "path": str(self.path(i))}
            return {"project": self.root.name, "projects": list_projects(),
                    "color": getattr(self, "color", None),
                    "all_ids": sorted(i for i in self.images if self.alive(i)),
                    "assets": load_assets(),
                    "train": (None if not TRAIN else
                              {k: TRAIN[k] for k in
                               ("asset", "family", "running", "error", "log")}
                              | {"elapsed": int(time.time() - TRAIN["started"])}),
                    "working": s["working"], "slots": s["slots"],
                    "candidates": s["candidates"], "pins": s["pins"],
                    "controls": s["controls"],
                    "history": [h for h in reversed(self.history) if self.alive(h)],
                    "comfy_ok": comfy_ok(), "meta": meta, "busy": self.busy,
                    "last_base_seed": s.get("last_base_seed"),
                    "families": {k: {"label": v["label"], "steps": v["steps"], "cfg": v["cfg"]}
                                 for k, v in FAMILIES.items()},
                    "pov_azim": POV_AZIM, "pov_elev": POV_ELEV, "pov_dist": POV_DIST,
                    "loras": list_loras()}


STORE = None
ROOT = None                       # the global root; projects are subdirs
RESERVED = {"assets", "loras", "_train"}   # never project names


def load_config():
    try:
        return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(**kw):
    c = load_config()
    c.update(kw)
    (ROOT / "config.json").write_text(json.dumps(c, indent=1), encoding="utf-8")


def load_assets():
    """<root>/assets.json: the asset IS this data (v4) - a list of
    {name, loras: [paths], dataset: [{path, description}]}. Paths are
    root-relative posix. The list is the sole authority; no physical
    asset directories exist."""
    try:
        return json.loads((ROOT / "assets.json").read_text(encoding="utf-8"))
    except Exception:
        return []


def save_assets(assets):
    (ROOT / "assets.json").write_text(json.dumps(assets, indent=1),
                                      encoding="utf-8")


def root_rel(p):
    """Canonical root-relative posix path; ValueError if outside the root."""
    q = Path(p)
    q = (q if q.is_absolute() else ROOT / q).resolve()
    return q.relative_to(ROOT).as_posix()


def list_projects():
    """A project is any root subdir that looks like one (has images/ or a
    journal). Reserved names and dot-dirs are never projects."""
    out = []
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir() or d.name in RESERVED or d.name.startswith("."):
            continue
        if (d / "images").is_dir() or (d / "journal.jsonl").exists():
            out.append(d.name)
    return out


PALETTE_SIZE = 8


def project_color(name):
    """Stable tint per project: assigned on first sight (next free slot of
    the 8-hue palette), persisted in config.json so a project keeps its
    colour for life - never roster-order, which would reshuffle everyone's
    colour when a project is added. shared_assets is reserved: always the
    same amber, outside the cycle."""
    if name == "shared_assets":
        return "shared"
    colors = load_config().get("colors") or {}
    if name not in colors:
        used = set(colors.values())
        colors[name] = next((i for i in range(PALETTE_SIZE) if i not in used),
                            len(colors) % PALETTE_SIZE)
        save_config(colors=colors)
    return colors[name]


def open_project(name):
    """Create-or-open a project. Its total and single effect is the path
    context: <root>/<name>/images/NNN.png etc.; state/journal swap because
    those files live under the path (v4 doctrine)."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) or name in RESERVED:
        raise ValueError(f"bad project name {name!r}")
    d = ROOT / name
    (d / "images").mkdir(parents=True, exist_ok=True)
    store = Store(d)
    store.color = project_color(name)
    save_config(last_project=name)
    return store


def list_loras():
    """Dropdown entries (v4). Canonical ASSET names first - picking "julie"
    resolves to the NEWEST file in that asset's loras[]; recipes always
    record the RESOLVED path (lora_file), never the alias, so a retrain
    changes the config (and the sibling key) instead of silently changing
    old recipes' meaning. Then loose files under <root>/loras, then the
    legacy CWD ./loras convention (pre-v4 stores)."""
    assets = load_assets()
    out = [a["name"] for a in assets if a.get("loras")]
    owned = {q for a in assets for q in a.get("loras", [])}
    found = set()
    rl = ROOT / "loras" if ROOT else None
    if rl and rl.is_dir():
        for q in rl.rglob("*_comfy.safetensors"):
            found.add(q.relative_to(ROOT).as_posix())
        for q in rl.glob("*.safetensors"):
            found.add("loras/" + q.name)
    legacy = Path("loras")
    if legacy.is_dir() and (not rl or legacy.resolve() != rl.resolve()):
        for q in legacy.rglob("*_comfy.safetensors"):
            found.add(str(q).replace("\\", "/"))
    return out + sorted(found - owned)   # asset-owned files show as the alias only


def resolve_lora(name):
    """Dropdown value -> absolute file, or None. Asset alias -> newest of
    its loras[]; else a ROOT-relative path; else legacy CWD-relative."""
    if not name:
        return None
    for a in load_assets():
        if a.get("name") == name and a.get("loras"):
            q = ROOT / a["loras"][-1]
            return q.resolve() if q.is_file() else None
    for q in (ROOT / name if ROOT else None, Path("loras") / name, Path(name)):
        if q is not None and q.is_file():
            return q.resolve()
    return None


def _follow_text(payload, ref, depth=0):
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
        pos = _follow_text(payload, ins.get("positive"))
        neg = _follow_text(payload, ins.get("negative"))
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


def import_bytes(data, source="import"):
    """Persist pasted/dropped/fetched image bytes: dedupe by content hash,
    flatten alpha onto white (raw transparency teaches the model
    checkerboards - CLAUDE.md matting lore). LAYER 0 (user rule 2026-08-24:
    NEVER destroy image metadata on import): every incoming text chunk is
    re-embedded verbatim (except a foreign `evolve` chunk - ours must win);
    glean_recipe() then maps what it can onto UI-exposed fields, so picking
    an import restores at least its prompt. Returns the image id."""
    sha1 = hashlib.sha1(data).hexdigest()
    existing = STORE.find_sha1(sha1)
    if existing is not None:
        return existing
    img = Image.open(io.BytesIO(data))
    img.load()
    raw = {k: v for k, v in img.info.items() if isinstance(v, str)}
    recipe = glean_recipe(raw)
    keep = {k: v for k, v in raw.items() if k != "evolve"}
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
        img = flat
    else:
        img = img.convert("RGB")
    return STORE.add_image(img, source, recipe=recipe, sha1=sha1, chunks=keep)


# ---------- generation ----------

def build_graph(prompt, seed, width, height, refs, lock, vary,
                lora_path=None, lora_strength=1.0):
    """refs: staged input-dir filenames; refs[0] = identity source (jittered
    by vary), later refs = extras. No refs = fiat (prompt only, IFT removed).
    lora_path: optional ComfyUI-format LoRA applied before identity transfer."""
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    p = payload["prompt"]
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
                                  similarity_floor=floor, softmax_temperature=temp,
                                  reference_index=0, reference_indices="0")
    else:
        p.pop("ift", None)
        p["guider"]["inputs"]["model"] = [model_src, 0]
    p["sched"]["inputs"].update(width=width, height=height)
    p["latent"]["inputs"].update(width=width, height=height, batch_size=1)
    p["txt"]["inputs"]["text"] = prompt
    p["noise"]["inputs"]["noise_seed"] = seed
    p["save"]["inputs"]["filename_prefix"] = f"{SCRATCH_PREFIX}/evolve"
    return payload


def build_zimage(prompt, negative, seed, width, height, steps, cfg,
                 lora_path=None, lora_strength=1.0):
    """Z-Image Turbo fiat graph - the settled char_lora_zimage inference
    recipe (euler/sgm_uniform, 20 steps, cfg 2)."""
    f = FAMILIES["zimage"]
    g = {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": f["dit"], "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": f["te"], "type": "lumina2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": f["vae"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["clip", 0]}},
        "latent": {"class_type": "EmptySD3LatentImage",
                   "inputs": {"width": width, "height": height, "batch_size": 1}},
        "samp": {"class_type": "KSampler",
                 "inputs": {"seed": seed, "steps": steps, "cfg": cfg,
                            "sampler_name": "euler", "scheduler": "sgm_uniform", "denoise": 1,
                            "model": ["unet", 0], "positive": ["pos", 0], "negative": ["neg", 0],
                            "latent_image": ["latent", 0]}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["samp", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["dec", 0], "filename_prefix": f"{SCRATCH_PREFIX}/evolve"}},
    }
    if lora_path:
        g["gene"] = {"class_type": "ApplyTrainedLora",
                     "inputs": {"strength": lora_strength, "model": ["unet", 0],
                                "lora_path": str(lora_path)}}
        g["samp"]["inputs"]["model"] = ["gene", 0]
    return {"client_id": "akasutils", "prompt": g}


def build_sdxl(prompt, negative, seed, width, height, steps, cfg):
    """Illustrious (SDXL) fiat graph: the standard checkpoint -> KSampler path."""
    f = FAMILIES["illustrious"]
    g = {
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": f["ckpt"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["ckpt", 1]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["ckpt", 1]}},
        "latent": {"class_type": "EmptyLatentImage",
                   "inputs": {"width": width, "height": height, "batch_size": 1}},
        "samp": {"class_type": "KSampler",
                 "inputs": {"seed": seed, "steps": steps, "cfg": cfg,
                            "sampler_name": f["sampler"], "scheduler": f["scheduler"], "denoise": 1,
                            "model": ["ckpt", 0], "positive": ["pos", 0], "negative": ["neg", 0],
                            "latent_image": ["latent", 0]}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["samp", 0], "vae": ["ckpt", 2]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["dec", 0], "filename_prefix": f"{SCRATCH_PREFIX}/evolve"}},
    }
    return {"client_id": "akasutils", "prompt": g}


EMBED_WORKFLOW = False       # --embed-workflow: add UI geometry to output pngs

_COMFY_PING = {"t": 0.0, "ok": False, "fails": 0}


def comfy_ok():
    """Cached ComfyUI liveness for the Status bar dot. A round in flight is
    live proof it is up (never ping a busy worker - the event loop stalls
    under GPU load and the dot flickered red, user-reported). One slow
    answer is not death: only 3 consecutive failures turn it red."""
    if STORE is not None and STORE.busy:
        _COMFY_PING["ok"], _COMFY_PING["fails"] = True, 0
        return True
    now = time.time()
    if now - _COMFY_PING["t"] > 5:
        _COMFY_PING["t"] = now
        try:
            urllib.request.urlopen("http://127.0.0.1:8188/system_stats",
                                   timeout=2).read()
            _COMFY_PING["ok"], _COMFY_PING["fails"] = True, 0
        except Exception:
            _COMFY_PING["fails"] += 1
            if _COMFY_PING["fails"] >= 3:
                _COMFY_PING["ok"] = False
    return _COMFY_PING["ok"]


def harvest_chunks(im, payload):
    """Text chunks for the stored png: everything ComfyUI embedded in the
    scratch render (`prompt`, POSSIBLY augmented - LoadImage gains
    is_changed = sha256 of the loaded file), plus optional `workflow`
    geometry so the png drags into the ComfyUI frontend editable
    (API-format alone opens an EMPTY canvas). Geometry conversion needs the
    live server's /object_info; it just rendered, so it is up - but never
    let a conversion hiccup lose the render."""
    chunks = {k: v for k, v in im.info.items() if isinstance(v, str)}
    if EMBED_WORKFLOW and "workflow" not in chunks:
        try:
            chunks["workflow"] = json.dumps(convert_payload(payload["prompt"]))
        except Exception as e:
            print(f"workflow geometry skipped: {type(e).__name__}: {e}")
    return chunks


def do_generate(controls):
    """Rule (a): clear every unpinned slot, then fill all free slots from
    the controls as they are now. Runs in a worker thread; the UI polls."""
    s = STORE.state
    rtab = controls.get("op") or "derive"
    if rtab not in ("create", "derive"):
        rtab = "derive"
    with STORE.lock:
        s["controls"] = controls
        STORE.sweep(rtab)               # unkept candidates -> archive, quietly
        s["candidates"][rtab] = []      # a round refills only ITS OWN tab's outputs
        STORE.save_state()
        n = s["slots"]
        working = s["working"]
    if n <= 0:
        return
    ref0 = controls.get("ref0")
    ref_ids = ([ref0] if STORE.alive(ref0) else []) + \
        [i for i in controls["refs"] if STORE.alive(i)]
    family = controls.get("family") or "klein"
    if family not in FAMILIES or ref_ids:
        family = "klein"                 # references exist only in the Klein graph
    fam = FAMILIES[family]
    steps = int(controls.get("steps") or 0) or fam["steps"]
    cfg = float(controls.get("cfg") or 0) or fam["cfg"]
    negative = controls.get("negative") or ""
    refs, inputs = [], {}
    for i in ref_ids:
        name = STORE.stage_ref(i)
        refs.append(name)
        inputs[name] = i
    prompt = (WHITE_BG_PREFIX + controls["prompt"]) if controls["whitebg"] else controls["prompt"]
    width, height = snap16(int(controls["width"])), snap16(int(controls["height"]))
    base = int(controls["seed"]) or random.randint(1, 999_999_999_999)
    lora = controls.get("lora") or ""
    if family == "illustrious":
        lora = ""                        # SDXL LoRAs: later, with unified training
    lora_abs = resolve_lora(lora) if lora else None
    if lora and lora_abs is None:
        raise FileNotFoundError(f"LoRA not found: {lora!r}")
    strength = float(controls.get("lora_strength", 1.0))
    lock = controls["lock"] if refs else None
    # additive randn on the reference latent (KJ InjectNoiseToLatent):
    # calibrated band 0.1-0.3, v0 slider max 0.5; 0.55+ just degrades
    vary = min(0.5, max(0.0, float(controls["vary"]))) if refs else 0.0

    # ComfyUI bug #11021: LoRA deltas compound on the shared base weights
    # across runs. The only clean reset is /free with free_memory (drops the
    # node cache so UNETLoader reloads pristine weights). Do it automatically
    # after any round that used a LoRA, or every round if asked.
    if controls.get("fresh_model") or STORE.state.get("last_round_lora"):
        print("reloading models (fresh weights: "
              + ("forced" if controls.get("fresh_model") else "previous round used a LoRA") + ")")
        free_vram()
    with STORE.lock:
        STORE.state["last_round_lora"] = bool(lora)
        STORE.state["last_base_seed"] = base
        STORE.save_state()
    STORE.hist_append(ref0)          # v2 history: bred-from only, co-parents excluded

    try:
        for k in range(n):
            if STORE.abort:
                print(f"round aborted at {k}/{n}")
                break
            seed = base + k
            if family == "zimage":
                payload = build_zimage(controls["prompt"], negative, seed, width, height,
                                       steps, cfg, lora_abs, strength)
            elif family == "illustrious":
                payload = build_sdxl(controls["prompt"], negative, seed, width, height, steps, cfg)
            else:
                payload = build_graph(prompt, seed, width, height, refs, lock, vary,
                                      lora_abs, strength)
                if steps != fam["steps"]:
                    payload["prompt"]["sched"]["inputs"]["steps"] = steps
                if cfg != fam["cfg"]:
                    payload["prompt"]["guider"]["inputs"]["cfg"] = cfg
            pid = queue(payload)
            print(f"candidate {k + 1}/{n} [{fam['label']}] -> {pid} (seed {seed})")
            src = wait(pid, timeout=600)[0]
            recipe = {"op": controls.get("op") or ("derive" if ref_ids else "create"),
                      "prompt": controls["prompt"], "negative": negative,
                      "family": family, "steps": steps, "cfg": cfg,
                      "whitebg": controls["whitebg"],
                      "seed": seed, "width": width, "height": height,
                      "lock": lock, "vary": vary, "lora": lora or None,
                      "lora_file": str(lora_abs) if lora_abs else None,
                      "lora_strength": strength if lora else None,
                      "ref0": STORE.alive(ref0)}
            with Image.open(src) as im:
                i = STORE.add_image(im.convert("RGB"), "gen", recipe, list(ref_ids),
                                    chunks=harvest_chunks(im, payload),
                                    inputs=inputs)
            with STORE.lock:
                lst = STORE.state["candidates"][rtab]
                if len(lst) < STORE.state["slots"]:
                    lst.append(i)
                STORE.save_state()
            STORE.busy = {"total": n, "done": k + 1, "tab": rtab}
    finally:
        STORE.busy = None


def do_pov(controls):
    """Camera tab: re-shoot the WI at an ABSOLUTE camera spec (the tokens SET
    the pov, they do not delta it - user, 2026-08-23). Only CHECKED axes emit
    tokens. Budding: single parent = the WI, which is what History records.
    N slots = N takes of the SAME camera (seed never moves the angle)."""
    s = STORE.state
    az, el, di = controls.get("azim"), controls.get("elev"), controls.get("dist")
    with STORE.lock:
        c = s["controls"]
        c["pov_azim"], c["pov_elev"], c["pov_dist"] = az, el, di
        rtab = "camera"
        STORE.sweep(rtab)
        s["candidates"][rtab] = []
        STORE.save_state()
        n = s["slots"]
        src_id = s["working"]
    if n <= 0 or not STORE.alive(src_id) or not (az or el or di):
        return
    toks = [VP_AZ[az] if az else None, VP_EL[el] if el else None,
            VP_DI[di] if di else None]
    prompt = "<sks> " + " ".join(t for t in toks if t)
    name = STORE.stage_ref(src_id)
    inputs = {name: src_id}
    base = int(controls.get("seed") or 0) or random.randint(1, 999_999_999_999)
    # always a LoRA round, so the #11021 reset rule applies on the way in
    if STORE.state.get("last_round_lora"):
        free_vram()
    with STORE.lock:
        STORE.state["last_round_lora"] = True
        STORE.state["last_base_seed"] = base
        STORE.save_state()
    STORE.hist_append(src_id)        # v2 history: Camera appends its true source
    try:
        for k in range(n):
            if STORE.abort:
                print(f"round aborted at {k}/{n}")
                break
            seed = base + k
            payload = {"client_id": "evolve",
                       "prompt": build_vp_graph(name, prompt, "", seed, POV_STEPS,
                                                POV_CFG, POV_STRENGTH, "pov", True)}
            pid = queue(payload)
            print(f"camera {k + 1}/{n} [{az}/{el}/{di}] -> {pid} (seed {seed})")
            src = wait(pid, timeout=600)[0]
            recipe = {"op": "pov", "pov_azim": az, "pov_elev": el, "pov_dist": di,
                      "prompt": prompt, "negative": "", "family": "qwen_edit",
                      "steps": POV_STEPS, "cfg": POV_CFG, "seed": seed,
                      "lora": "qwen multiple-angles + lightning", "ref0": True}
            with Image.open(src) as im:
                i = STORE.add_image(im.convert("RGB"), "pov", recipe, [src_id],
                                    chunks=harvest_chunks(im, payload),
                                    inputs=inputs)
            with STORE.lock:
                lst = STORE.state["candidates"][rtab]
                if len(lst) < STORE.state["slots"]:
                    lst.append(i)
                STORE.save_state()
            STORE.busy = {"total": n, "done": k + 1, "tab": rtab}
    finally:
        STORE.busy = None


# ---------- http ----------

def ok(**kw):
    return web.json_response(kw)


async def api_state(request):
    return web.json_response(STORE.snapshot())


async def api_generate(request):
    if STORE.busy:
        return web.json_response({"error": "already generating"}, status=409)
    if TRAIN and TRAIN.get("running"):
        return web.json_response({"error": "training in progress"}, status=409)
    controls = await request.json()
    c = dict(DEFAULT_CONTROLS)
    c.update(controls)
    c["refs"] = [(i if STORE.alive(i) else None) for i in (c.get("refs") or [])][:3]
    c["refs"] += [None] * (3 - len(c["refs"]))
    if not STORE.alive(c.get("ref0")):
        c["ref0"] = None
    op = controls.get("op") or "derive"
    c["op"] = op
    if op == "create":               # fiat: the tab IS the no-refs gate
        c["ref0"] = None
        c["refs"] = [None, None, None]
    else:                            # derive: Klein-only, per spec
        c["family"] = "klein"
    with STORE.lock:                 # per-tab Outputs count drives the round
        STORE.state["slots"] = max(1, min(64, int(controls.get("outputs")
                                                  or STORE.state["slots"])))
    # mark busy NOW, before the thread starts: the client refreshes right
    # after this returns and must see a busy state or it stops polling
    with STORE.lock:
        STORE.abort = False
        STORE.busy = {"total": STORE.state["slots"], "done": 0,
                      "tab": "create" if op == "create" else "derive"}
    loop = asyncio.get_event_loop()

    async def run():
        try:
            await loop.run_in_executor(None, do_generate, c)
        except SystemExit as e:
            print(f"generate failed: {e}")
        except Exception as e:
            print(f"generate failed: {type(e).__name__}: {e}")
        finally:
            STORE.busy = None
    asyncio.ensure_future(run())
    return ok(started=True)


async def api_pov(request):
    """Camera generator. Same busy lock as Generate; both panels disable off
    one flag."""
    if STORE.busy:
        return web.json_response({"error": "already generating"}, status=409)
    if TRAIN and TRAIN.get("running"):
        return web.json_response({"error": "training in progress"}, status=409)
    q = await request.json()
    az = q.get("azim") or None
    el = q.get("elev") or None
    di = q.get("dist") or None
    if (az and az not in VP_AZ) or (el and el not in VP_EL) or (di and di not in VP_DI):
        return web.json_response({"error": "bad camera token"}, status=400)
    if not (az or el or di):
        return web.json_response({"error": "no camera axis selected"}, status=400)
    if not STORE.alive(STORE.state["working"]):
        # budding needs a parent: nothing on the stage = nothing to re-shoot
        return web.json_response({"error": "no working image to re-shoot"}, status=400)
    with STORE.lock:
        STORE.state["slots"] = max(1, min(64, int(q.get("outputs")
                                                  or STORE.state["slots"])))
        STORE.abort = False
        STORE.busy = {"total": STORE.state["slots"], "done": 0, "tab": "camera"}
    loop = asyncio.get_event_loop()

    async def run():
        try:
            await loop.run_in_executor(None, do_pov,
                                       {"azim": az, "elev": el, "dist": di,
                                        "seed": q.get("seed") or 0})
        except SystemExit as e:
            print(f"camera failed: {e}")
        except Exception as e:
            print(f"camera failed: {type(e).__name__}: {e}")
        finally:
            STORE.busy = None
    asyncio.ensure_future(run())
    return ok(started=True)


def sibling_key(rec):
    """SIBLINGS (v2, 2026-08-23): identical generation config MODULO SEED -
    the recipe minus seed, plus the parent id set. Captures lock, vary,
    lora+strength, steps/cfg, family, w/h, white-bg: change any knob and it
    is no longer a sibling; re-rolls of an unchanged config merge. Imports
    have no config: singleton sets."""
    r = rec.get("recipe")
    if not r:
        return ("import", rec["id"])
    r = dict(r)
    r.pop("seed", None)
    return (tuple(sorted(rec.get("parents") or [])),
            json.dumps(r, sort_keys=True))


async def api_family(request):
    """Parents / siblings / children of ONE image - the Genealogy sheets,
    anchored to the WI. Flat, id-ordered; archived members carry gone=true
    and render as placeholder tiles."""
    q = await request.json()
    i = q.get("id")
    with STORE.lock:
        rec = STORE.images.get(i)
        if not rec:
            return web.json_response({"error": f"unknown image {i}"}, status=404)

        def tile(j):
            return {"id": j}

        key = sibling_key(rec)
        alive = STORE.alive
        return web.json_response({
            "id": i,
            "parents": [tile(p) for p in (rec.get("parents") or []) if alive(p)],
            "siblings": [tile(j) for j, r in sorted(STORE.images.items())
                         if alive(j) and sibling_key(r) == key],
            "children": [tile(j) for j, r in sorted(STORE.images.items())
                         if alive(j) and i in (r.get("parents") or [])]})


# ---------- Make LoRA (stage 5): sync -> train -> append ----------
TRAIN = None          # {"asset","family","log","started","error","running"}
TRAIN_PROC = None


def sync_dataset(asset):
    """5a: dataset list -> <root>/_train/<name>/ (copy-new, delete-dropped,
    skip-unchanged); descriptions written as .txt sidecars AT SYNC TIME.
    File names are the source path mangled ('/' -> '__'): deterministic, so
    reordering the list never churns the dir. The trigger-prefix rule is
    ENFORCED here - fail fast, before anything expensive."""
    name = asset["name"]
    bad = [e["path"] for e in asset["dataset"]
           if not (e.get("description") or "").startswith(name)]
    if bad:
        raise ValueError("captions must START with the trigger "
                         f"{name!r}; fix: " + ", ".join(bad[:8]))
    missing = [e["path"] for e in asset["dataset"] if not (ROOT / e["path"]).is_file()]
    if missing:
        raise ValueError("missing files: " + ", ".join(missing[:8]))
    if not asset["dataset"]:
        raise ValueError("empty dataset")
    ds = ROOT / "_train" / name
    ds.mkdir(parents=True, exist_ok=True)
    want = {}
    for e in asset["dataset"]:
        stem = e["path"].replace("/", "__")
        stem = stem[:-4] if stem.endswith(".png") else stem
        want[stem] = e
    for f in ds.iterdir():                    # delete-dropped
        if f.stem not in want:
            f.unlink()
    n_copied = 0
    for stem, e in want.items():
        src, dst = ROOT / e["path"], ds / (stem + ".png")
        st = src.stat()
        if not (dst.exists() and dst.stat().st_size == st.st_size
                and int(dst.stat().st_mtime) == int(st.st_mtime)):
            shutil.copy2(src, dst)
            n_copied += 1
        (ds / (stem + ".txt")).write_text(e["description"], encoding="utf-8")
    print(f"sync {name}: {len(want)} images ({n_copied} copied)")
    return ds


def run_training(name, family):
    """5b/5c worker thread: launch the existing trainer wrapper as a
    subprocess (it owns the musubi venv, /free and streaming), mirror its
    output to a tailable log, and on success append the newest _comfy file
    to the asset's loras[]."""
    global TRAIN, TRAIN_PROC
    import subprocess
    log = ROOT / "_train" / f"{name}.log"
    started = time.time()
    try:
        ds = ROOT / "_train" / name
        if family == "zimage":
            cmd = [sys.executable, str(HERE / "char_lora_zimage.py"), name,
                   "--images", str(ds), "--trigger", name]
        else:
            cmd = [sys.executable, str(HERE / "char_lora_flux.py"),
                   "--name", name, "--dataset", str(ds)]
        with log.open("w", encoding="utf-8", errors="replace") as lf:
            lf.write(" ".join(cmd) + chr(10))
            lf.flush()
            TRAIN_PROC = subprocess.Popen(cmd, cwd=ROOT, stdout=lf,
                                          stderr=subprocess.STDOUT)
            rc = TRAIN_PROC.wait()
        if rc != 0:
            TRAIN["error"] = f"trainer exited {rc} - see {log}"
            return
        out = ROOT / "loras" / name
        if family == "klein":
            # the Klein pipeline is train (char_lora_flux, musubi-native)
            # THEN convert (test_lora_flux.ensure_comfy_format: musubi
            # convert_lora + fix_flux2_keys - the akastierney path)
            natives = [q for q in out.glob("*.safetensors")
                       if not q.name.endswith("_comfy.safetensors")
                       and q.stat().st_mtime >= started - 5] if out.is_dir() else []
            if natives:
                from test_lora_flux import ensure_comfy_format
                ensure_comfy_format(max(natives, key=lambda q: q.stat().st_mtime))
        fresh = [q for q in out.glob("*_comfy.safetensors")
                 if q.stat().st_mtime >= started - 5] if out.is_dir() else []
        if not fresh:
            TRAIN["error"] = f"trained, but no _comfy output found under {out}"
            return
        newest = max(fresh, key=lambda q: q.stat().st_mtime)
        with STORE.lock:
            assets = load_assets()
            a = next((x for x in assets if x.get("name") == name), None)
            if a is not None:
                rel = newest.relative_to(ROOT).as_posix()
                if rel not in a["loras"]:
                    a["loras"].append(rel)
                save_assets(assets)
        print(f"training done: {newest.name} -> asset {name}")
    except BaseException as e:      # ensure_comfy_format sys.exits on failure
        TRAIN["error"] = f"{type(e).__name__}: {e}"
    finally:
        TRAIN["running"] = False
        TRAIN_PROC = None


async def api_train(request):
    """Start a training job. The button click IS the user's say-so (red
    line); one GPU: mutually exclusive with Generate in both directions."""
    global TRAIN
    b = await request.json()
    name = (b.get("name") or "").strip()
    family = b.get("family") or "zimage"
    if family not in ("zimage", "klein"):
        return web.json_response({"error": f"bad family {family!r}"}, status=400)
    if STORE.busy:
        return web.json_response({"error": "generating - wait or Stop first"}, status=409)
    if TRAIN and TRAIN.get("running"):
        return web.json_response({"error": "a training job is already running"}, status=409)
    a = next((x for x in load_assets() if x.get("name") == name), None)
    if a is None:
        return web.json_response({"error": f"no asset {name!r}"}, status=404)
    try:
        sync_dataset(a)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    TRAIN = {"asset": name, "family": family, "running": True, "error": None,
             "started": time.time(),
             "log": str(ROOT / "_train" / f"{name}.log")}
    threading.Thread(target=run_training, args=(name, family), daemon=True).start()
    return ok(started=True, log=TRAIN["log"])


async def api_train_abort(request):
    """Kill the training TREE (taskkill /T): the wrapper spawns the musubi
    venv underneath, and killing only the parent would orphan the GPU
    process. Never kill by exe path - uv venvs all share one interpreter
    image (CLAUDE.md lesson)."""
    import subprocess
    if not (TRAIN and TRAIN.get("running") and TRAIN_PROC):
        return ok(aborted=False)
    subprocess.run(["taskkill", "/PID", str(TRAIN_PROC.pid), "/T", "/F"],
                   capture_output=True)
    TRAIN["error"] = "aborted by user"
    return ok(aborted=True)


async def api_asset(request):
    """Asset CRUD (v4). ops: create / delete / add / remove / describe /
    add_lora. The server stores descriptions EXACTLY as given - the
    trigger-prefix rule is validated at edit time (UI) and enforced at
    train time, never auto-repaired (the akasakas incident)."""
    b = await request.json()
    op = b.get("op")
    name = (b.get("name") or "").strip()
    with STORE.lock:
        assets = load_assets()
        a = next((x for x in assets if x.get("name") == name), None)
        if op == "create":
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
                return web.json_response({"error": f"bad asset name {name!r} "
                                          "(it is also the LoRA trigger: letters, "
                                          "digits, - _, no spaces)"}, status=400)
            if a:
                return web.json_response({"error": f"asset {name!r} exists"}, status=400)
            assets.append({"name": name, "loras": [], "dataset": []})
        elif a is None:
            return web.json_response({"error": f"no asset {name!r}"}, status=404)
        elif op == "delete":
            assets.remove(a)
        elif op in ("add", "remove", "describe", "add_lora"):
            try:
                path = root_rel(b["path"])
            except Exception:
                return web.json_response({"error": "path outside the root"}, status=400)
            if op == "add":
                if not any(e["path"] == path for e in a["dataset"]):
                    a["dataset"].append({"path": path,
                                         "description": b.get("description")
                                         or name})
            elif op == "remove":
                a["dataset"] = [e for e in a["dataset"] if e["path"] != path]
            elif op == "describe":
                for e in a["dataset"]:
                    if e["path"] == path:
                        e["description"] = b.get("description") or ""
            elif op == "add_lora":
                if path not in a["loras"]:
                    a["loras"].append(path)
        else:
            return web.json_response({"error": f"bad op {op!r}"}, status=400)
        save_assets(assets)
    return ok(assets=assets)


async def api_import_folder(request):
    """Bulk-import every image in a DISK folder (recursive) - the fast path
    for big datasets: no browser upload, the server reads straight from
    disk. Optionally lands each image in an asset, caption seeded from its
    gleaned prompt. Returns a summary (added / duplicates / skipped)."""
    b = await request.json()
    folder = Path(str(b.get("path") or "").strip().strip('"')).expanduser()
    if not folder.is_dir():
        return web.json_response({"error": f"not a folder: {folder}"}, status=400)
    asset = (b.get("asset") or "").strip() or None

    def work():
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        added = dups = skipped = 0
        files = sorted(q for q in folder.rglob("*")
                       if q.is_file() and q.suffix.lower() in exts)
        for q in files:
            try:
                data = q.read_bytes()
            except OSError:
                skipped += 1
                continue
            existed = STORE.find_sha1(hashlib.sha1(data).hexdigest()) is not None
            i = import_bytes(data)
            if existed:
                dups += 1
            else:
                added += 1
            if asset:
                rec = STORE.images.get(i) or {}
                pr = ((rec.get("recipe") or {}).get("prompt") or "").strip()
                with STORE.lock:
                    assets = load_assets()
                    a = next((x for x in assets if x.get("name") == asset), None)
                    if a is not None:
                        path = f"{STORE.root.name}/images/{i}.png"
                        if not any(e["path"] == path for e in a["dataset"]):
                            a["dataset"].append({"path": path,
                                                 "description": (asset + ", " + pr)
                                                 if pr else asset})
                        save_assets(assets)
        return {"added": added, "duplicates": dups, "skipped": skipped,
                "total": len(files)}

    r = await asyncio.get_event_loop().run_in_executor(None, work)
    return ok(**r)


async def serve_file(request):
    """Serve any image under the root by root-relative path (the asset
    browser shows images from ANY project). Strictly contained: resolve
    must stay inside the root."""
    try:
        f = (ROOT / request.match_info["path"]).resolve()
        f.relative_to(ROOT)
    except Exception:
        raise web.HTTPNotFound()
    if not (f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")):
        raise web.HTTPNotFound()
    return web.FileResponse(f, headers={"Cache-Control": "no-cache"})


async def api_project(request):
    """Switch or create a project (singleton: refuses while generating)."""
    global STORE
    if STORE.busy:
        return web.json_response({"error": "busy - wait or Stop first"}, status=409)
    b = await request.json()
    name = (b.get("name") or "").strip()
    try:
        STORE = open_project(name)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    print(f"project: {name}")
    return ok(project=name)


async def api_abort(request):
    """Stop the current round NOW: no further candidates are queued and the
    in-flight ComfyUI job is interrupted. Finished candidates stay; the
    interrupted partial is discarded. Aborts by choice only - /interrupt
    cannot un-wedge a CUDA OOM."""
    if not STORE.busy:
        return ok(aborted=False)
    STORE.abort = True
    try:
        urllib.request.urlopen("http://127.0.0.1:8188/interrupt",
                               data=b"", timeout=2).read()
    except Exception as e:
        print(f"interrupt failed: {type(e).__name__}: {e}")
    return ok(aborted=True)


async def api_prune(request):
    """{id, force, apply}: apply=false returns the impact plan for the
    dialog; apply=true executes it (recomputed server-side)."""
    b = await request.json()
    i, force = b.get("id"), bool(b.get("force"))
    if not STORE.alive(i):
        return web.json_response({"error": f"no such image {i}"}, status=404)
    plan = STORE.prune_apply(i, force) if b.get("apply") else STORE.prune_plan(i, force)
    return web.json_response(plan)


async def api_discard(request):
    b = await request.json()
    why = STORE.discard(b.get("id"))
    if why:
        return web.json_response({"error": f"kept: {why}"}, status=409)
    return ok()


def foreign_meta(rel):
    """Metadata for an image in ANOTHER project (asset datasets span
    projects): read that project's journal for the record. Read-only; no
    gc verdict - that is the other project's business when it is open."""
    parts = rel.split("/")
    if len(parts) != 3 or parts[1] != "images" or not parts[2].endswith(".png"):
        return None
    proj, i = parts[0], int(parts[2][:-4])
    j = ROOT / proj / "journal.jsonl"
    rec, gone = None, False
    if j.exists():
        for line in j.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            if ev.get("t") == "image" and ev.get("id") == i:
                rec = ev
            elif ev.get("t") == "gc" and i in ev.get("ids", []):
                gone = True
    if rec is None:
        return None
    return {"id": i, "w": rec.get("w"), "h": rec.get("h"), "source": rec.get("source"),
            "ts": rec.get("ts"), "recipe": rec.get("recipe"),
            "parents": rec.get("parents"), "gone": gone,
            "gc": "archived" if gone else f"in project {proj}",
            "path": str(ROOT / proj / "images" / f"{i}.png")}


async def api_meta(request):
    """Full metadata for ONE image (Info Window): by id in the current
    project, or by root-relative PATH for any project's image."""
    q = await request.json()
    i = q.get("id")
    if i is None and q.get("path"):
        try:
            rel = root_rel(q["path"])
        except Exception:
            return web.json_response({"error": "path outside the root"}, status=400)
        if rel.split("/")[0] == STORE.root.name:
            try:
                i = int(rel.split("/")[-1][:-4])
            except ValueError:
                return web.json_response({"error": "bad path"}, status=400)
        else:
            m = foreign_meta(rel)
            if m is None:
                return web.json_response({"error": f"no record for {rel}"}, status=404)
            return web.json_response(m)
    with STORE.lock:
        r = STORE.images.get(i)
        if not r:
            return web.json_response({"error": f"unknown image {i}"}, status=404)
        return web.json_response({
            "id": i, "w": r["w"], "h": r["h"], "source": r["source"],
            "ts": r.get("ts"), "recipe": r.get("recipe"),
            "parents": r.get("parents"), "gone": bool(r.get("gone")),
            "gc": (STORE.root_reason(i) or "collectable") if not r.get("gone") else "archived",
            "path": str(STORE.path(i))})


async def api_controls(request):
    """Persist control edits (so a reload keeps your prompt etc.)."""
    c = await request.json()
    with STORE.lock:
        STORE.state["controls"].update({k: v for k, v in c.items() if k in DEFAULT_CONTROLS})
        STORE.save_state()
    return ok()


def restore_controls(i):
    """Picking an image restores the run state that GENERATED it: every
    widget, ref0 = its PARENT (empty for fiat - so Generate re-rolls that
    round with new seeds), extras from its recorded co-parents. Imports have
    no recipe: ref0 = none (click the stage to set ref0 = the import)."""
    rec = STORE.images[i]
    r = rec.get("recipe")
    c = STORE.state["controls"]
    if not r:
        c["ref0"] = None
        return
    c["prompt"] = r.get("prompt", c["prompt"])
    c["negative"] = r.get("negative", "")
    c["family"] = r.get("family") or "klein"
    c["steps"] = int(r.get("steps") or 0)
    c["cfg"] = float(r.get("cfg") or 0)
    c["seed"] = int(r.get("seed") or 0)
    c["whitebg"] = bool(r.get("whitebg", c["whitebg"]))
    c["width"], c["height"] = int(r.get("width", c["width"])), int(r.get("height", c["height"]))
    if r.get("lock"):
        c["lock"] = r["lock"]
    c["vary"] = float(r.get("vary") or 0.0)
    if r.get("op") == "pov":
        # a Camera candidate restores its tab + axis positions (checked =
        # token was present); ref0 = its single parent
        c["tab"] = "camera"
        c["pov_azim"] = r.get("pov_azim")
        c["pov_elev"] = r.get("pov_elev")
        c["pov_dist"] = r.get("pov_dist")
        parents = [q for q in (rec.get("parents") or []) if STORE.alive(q)]
        c["ref0"] = parents[0] if parents else None
        c["refs"] = [None, None, None]
        return
    c["lora"] = r.get("lora") or ""
    if r.get("lora_strength") is not None:
        c["lora_strength"] = float(r["lora_strength"])
    parents = [q for q in (rec.get("parents") or []) if STORE.alive(q)]
    had_ref0 = bool(r.get("ref0", r.get("use_working")))   # old recipes: use_working
    c["ref0"] = parents[0] if (had_ref0 and parents) else None
    extras = parents[1:] if had_ref0 else parents
    c["refs"] = (extras + [None, None, None])[:3]
    # tab-switch-on-pick (v3, user-confirmed): the operator that made the
    # image becomes the active tab, and its seed lands in that tab's field
    c["tab"] = "derive" if (c["ref0"] is not None
                            or any(x is not None for x in c["refs"])) else "create"
    c["seed_" + c["tab"]] = int(r.get("seed") or 0)


async def api_place(request):
    b = await request.json()
    i, target, index = b.get("id"), b["target"], int(b.get("index", 0))
    if not STORE.alive(i):
        return web.json_response({"error": f"no such image {i}"}, status=404)
    with STORE.lock:
        if target == "working":
            STORE.set_working(i)
            restore_controls(i)
            STORE.save_state()
        elif target == "ref":
            STORE.state["controls"]["refs"][index] = i
            STORE.save_state()
        elif target == "ref0":
            STORE.state["controls"]["ref0"] = i
            STORE.save_state()
        elif target == "slot":
            STORE.place_slot(i, index)
        elif target == "pin":
            STORE.pin(i, True, index)
        else:
            return web.json_response({"error": "bad target"}, status=400)
    return ok()


async def api_clear(request):
    b = await request.json()
    target, index = b["target"], int(b.get("index", 0))
    with STORE.lock:
        if target == "working":
            STORE.set_working(None)
        elif target == "ref":
            STORE.state["controls"]["refs"][index] = None
        elif target == "ref0":
            STORE.state["controls"]["ref0"] = None
        elif target == "slot":
            c = STORE.cands()
            if index < len(c):
                c.pop(index)
        elif target == "pin":
            if index < len(STORE.state["pins"]):
                STORE.state["pins"].pop(index)
        STORE.save_state()
    return ok()


async def api_pin(request):
    b = await request.json()
    i, on = b["id"], bool(b["on"])
    STORE.pin(i, on)
    return ok()


async def api_slots(request):
    b = await request.json()
    n = max(1, min(64, int(b["slots"])))
    with STORE.lock:
        STORE.state["slots"] = n
        c = STORE.cands()
        del c[n:]                    # shrink the ACTIVE tab's list from the end
        STORE.save_state()
    return ok()


async def api_import(request):
    """Raw image bytes (drop/paste/file) -> stored image id."""
    data = await request.read()
    if not data:
        return web.json_response({"error": "empty upload"}, status=400)
    try:
        i = import_bytes(data)
    except Exception as e:
        return web.json_response({"error": f"not a readable image: {e}"}, status=400)
    return ok(id=i)


async def api_import_url(request):
    """A URL dropped from another browser: fetched server-side (no CORS)."""
    b = await request.json()
    url = b["url"]
    if not re.match(r"^https?://", url):
        return web.json_response({"error": "only http(s) urls"}, status=400)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 evolver"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read(64 * 1024 ** 2)
        i = import_bytes(data)
    except Exception as e:
        return web.json_response({"error": f"fetch failed: {e}"}, status=400)
    return ok(id=i)


async def api_gc(request):
    return web.json_response(STORE.gc())


async def serve_image(request):
    """/img/<project>/<id>: project-qualified, because /img/<id> alone is
    ambiguous across projects - after a switch the same URL named a
    different image and the browser's memory cache served the stale one
    (user-caught). File existence is the truth (archived = gone = 404)."""
    i = int(request.match_info["id"])
    proj = request.match_info.get("project")
    if proj:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", proj) or proj in RESERVED:
            raise web.HTTPNotFound()
        f = ROOT / proj / "images" / f"{i}.png"
    else:                       # legacy unqualified URL: current project
        f = STORE.path(i)
    if not f.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(f, headers={
        "Content-Disposition": f'inline; filename="{i}.png"',
        "Cache-Control": "no-cache"})


async def index(request):
    """Serve the UI fresh from THIS file's source on every request, so page
    edits appear on browser refresh without a server restart (PAGE is baked
    at import time). Falls back to the baked copy if the source is
    unreadable mid-edit. no-store kills Edge's HTML caching."""
    page = PAGE
    try:
        src = Path(__file__).read_text(encoding="utf-8")
        marker = "PAGE = r" + '"' * 3
        i = src.index(marker) + len(marker)
        page = src[i:src.index('"' * 3, i)]
    except Exception as e:
        print(f"live page reload failed ({e}); serving baked copy")
    return web.Response(text=page, content_type="text/html",
                        headers={"Cache-Control": "no-store"})


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Evolver</title>
<style>
:root{--bg:#1b1b1f;--panel:#242429;--line:#3a3a42;--fg:#ddd;--dim:#888;--acc:#ffd04a;--focus:#5ab0ff}
*{box-sizing:border-box}
html,body{height:100%;margin:0;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif;overflow:hidden}
#app{display:grid;grid-template-columns:150px 1fr;height:100%}
nav{background:var(--panel);border-right:1px solid var(--line);display:flex;flex-direction:column;padding:8px 0}
nav a{padding:7px 14px;color:var(--dim);text-decoration:none;cursor:default}
nav a.on{color:var(--fg);background:#2e2e36;border-left:3px solid var(--acc)}
nav .spacer{flex:1}
#proj{display:flex;gap:4px;padding:4px 8px 10px;border-bottom:1px solid var(--line);margin-bottom:6px}
#proj select{flex:1;min-width:0;background:#111;color:var(--fg);border:1px solid var(--line);padding:3px}
#proj .sw{width:11px;height:11px;border-radius:50%;flex:none;align-self:center;border:1px solid rgba(255,255,255,.25)}
#proj button{background:#333;color:var(--fg);border:1px solid var(--line);cursor:pointer;padding:0 8px}
nav button{margin:6px 10px;background:#333;color:var(--fg);border:1px solid var(--line);padding:5px;cursor:pointer}
main{display:grid;grid-template-rows:auto 1fr auto;height:100vh;min-width:0}
#top{border-bottom:1px solid var(--line);min-width:0}
.car{padding:2px 10px;min-width:0;max-width:100%}
.car summary{cursor:pointer;color:var(--dim);font-size:12px;user-select:none;padding:3px 6px;display:flex;align-items:center;gap:6px}
.car summary::before{content:'▸';color:var(--dim);font-size:10px;flex:none}
.car[open] summary::before{content:'▾'}
.car summary:hover{background:#26262c}
#genea>summary:hover{background:#26262c}
.car summary b{color:var(--fg);font-weight:500}
.car .row{display:flex;align-items:center;gap:6px;padding:2px 0 6px}
.car .arrow{background:#333;border:1px solid var(--line);color:var(--fg);width:26px;height:60px;cursor:pointer;flex:none}
.car .strip{display:flex;gap:6px;overflow-x:auto;flex:1;height:72px;align-items:center;scrollbar-width:none;min-width:0}
.car .strip::-webkit-scrollbar{display:none}
.car img{height:64px;width:64px;object-fit:cover;border:2px solid transparent;border-radius:3px;cursor:pointer;flex:none}
#work{display:grid;grid-template-columns:minmax(220px,46%) 6px minmax(220px,1fr);min-height:0}
#split{cursor:col-resize;background:var(--line);position:relative}
#split:hover,#split.drag{background:var(--acc)}
#split::after{content:"";position:absolute;inset:0 -4px}
#stage{display:flex;flex-direction:column;min-height:0}
/* transparent border, same 2px geometry as a thumb's: selection outlines
   (offset -2) overlay it exactly instead of sitting inset behind a grey
   frame; drop-over still lights it via border-color */
#stagebox{flex:1;display:flex;align-items:center;justify-content:center;margin:10px;border:2px solid transparent;border-radius:6px;min-height:0;position:relative;overflow:hidden}
#stagebox img{max-width:100%;max-height:100%;object-fit:contain}
/* scale-down = contain but NEVER above natural size - the same sizing rule
   as the working image's max-width/height:100%, so Space A/B compares
   identically-framed images whatever the panel's aspect ratio */
#stagebox #peek{position:absolute;inset:0;width:100%;height:100%;object-fit:scale-down;background:var(--bg);display:none}
#stagebox .hint,.slot .hint,.ref .hint{color:var(--dim);font-size:12px;text-align:center;padding:6px;pointer-events:none}
#recipe{padding:0 12px 8px;color:var(--dim);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#genpanel{display:grid;grid-template-rows:auto 6px minmax(80px,1fr);min-height:0;min-width:0}
#hsplit{cursor:row-resize;background:var(--line);position:relative}
#hsplit:hover,#hsplit.drag{background:var(--acc)}
#hsplit::after{content:"";position:absolute;inset:-4px 0}
#output{display:flex;flex-direction:column;min-height:0;min-width:0}
#output .ghead{display:flex;align-items:center;gap:8px;padding:4px 10px 0;color:var(--dim);font-size:12px;user-select:none}
#output .ghead b{color:var(--fg);font-weight:500}
#output .grid{display:grid;gap:8px;padding:8px 10px;align-content:start;justify-content:center;overflow:auto;flex:1}
#bottom{border-top:1px solid var(--line);display:flex;gap:16px;align-items:center;padding:6px 12px;font-size:12px;color:var(--dim)}
#bottom .spacer{flex:1}
#msg{color:var(--fg);font-size:12px;min-height:14px}
#msg.notice{color:#f77;font-weight:600}
@keyframes barflash{0%,100%{background:transparent}25%,75%{background:#5a1c1c}}
#bottom.alert{animation:barflash 1.2s ease-in-out 2}
#bottom .lab{text-transform:uppercase;letter-spacing:.05em;font-size:10px}
#bottom label{display:flex;gap:4px;align-items:center}
#bottom input[type=number]{width:64px;background:#111;color:var(--fg);border:1px solid var(--line);padding:3px 4px}
#bottom button{background:#333;color:var(--fg);border:1px solid var(--line);padding:4px 10px;cursor:pointer}
#bottom .svc{display:flex;gap:5px;align-items:center}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#666}
.dot.ok{background:#4c4}
.dot.bad{background:#c44}
#prog{width:150px}
#genea{border-bottom:1px solid var(--line)}
#genea>summary{padding:3px 10px;color:var(--dim);font-size:12px;cursor:pointer;user-select:none}
#genea>summary b{color:var(--fg);font-weight:500}
#genea .car{margin-left:16px}
.gone{width:64px;height:64px;flex:none;display:flex;align-items:center;justify-content:center;border:2px dashed var(--line);border-radius:3px;color:var(--dim);font-size:10px;text-align:center;cursor:default}
.slot{position:relative;border:2px dashed var(--line);border-radius:6px;display:flex;align-items:center;justify-content:center;background:#202025;overflow:hidden}
.slot img{max-width:100%;max-height:100%;object-fit:contain;display:block}
.slot.pinned{border-style:solid;border-color:#7a6}
.slot.pinned img{cursor:grab}
.slot .pin{position:absolute;top:4px;right:4px;background:rgba(0,0,0,.55);border:none;color:#fff;border-radius:3px;cursor:pointer;padding:2px 5px;font-size:13px}
.slot .busy{color:var(--acc)}
.drop.over{border-color:var(--acc)!important;background:#2c2a20}
.focus{outline:2px solid var(--focus);outline-offset:-2px}
.alias{outline:2px dashed var(--focus);outline-offset:-2px}
.car.empty summary{pointer-events:none;opacity:.55}
.car summary .sz{background:none;border:1px solid var(--line);color:var(--dim);border-radius:3px;cursor:pointer;padding:0 6px;font-size:11px;line-height:16px;flex:none}
.car summary .sz:hover{color:var(--fg)}
.car:not([open]) summary .sz{display:none}
.car.big .strip{height:150px}
.car.big img{height:138px;width:138px}
.car.big .arrow{height:120px}
.car.big .gone{width:138px;height:138px}
#infowin{position:fixed;z-index:99;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px 12px;font-size:12px;max-width:480px;box-shadow:0 8px 28px rgba(0,0,0,.6)}
#infowin .iw-row{display:flex;gap:8px;align-items:baseline;margin:3px 0}
#infowin .iw-k{color:var(--dim);flex:none;width:56px}
#infowin .iw-v{overflow-wrap:anywhere;white-space:pre-line;max-height:180px;overflow-y:auto}
#infowin .iw-c{background:none;border:none;color:var(--dim);cursor:pointer;flex:none;margin-left:auto;padding:2px;display:flex;align-items:center}
#infowin .iw-c:hover{color:var(--fg)}
#infowin .iw-c.done{color:#8c7}
#controls{padding:6px 10px 8px;display:flex;flex-direction:column;gap:8px;overflow-y:auto;min-height:0}
/* rules below were casualties of the v3 CSS splice (the ref boxes rendered
   their images at NATURAL size - user-caught); restored verbatim */
#prompts{display:flex;gap:6px;flex:1;min-height:64px}
#prompt{flex:1;min-height:58px;background:#111;color:var(--fg);border:1px solid var(--line);padding:6px;font:13px system-ui;resize:vertical}
#negative{width:32%;min-height:58px;background:#2a1214;color:#f0c8c8;border:1px solid #5a2a2e;padding:6px;font:12px system-ui;resize:vertical}
#negative::placeholder{color:#8a5a5e}
#refs{display:flex;gap:6px;align-items:center}
.ref{width:58px;height:58px;border:2px dashed var(--line);border-radius:4px;display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden;flex:none}
.ref[data-target="ref0"]{border-color:#666;margin-right:12px}
.ref img{max-width:100%;max-height:100%;object-fit:contain}
/* ---- toasts + prune dialog ---- */
#toasts{position:fixed;top:12px;right:12px;z-index:90;display:flex;flex-direction:column;gap:8px;max-width:460px}
.toast{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--focus);color:var(--fg);padding:10px 14px;font-size:12px;border-radius:4px;box-shadow:0 8px 24px rgba(0,0,0,.6);cursor:pointer;white-space:pre-line}
.toast.warn{border-left-color:#e55}
#prunedlg{position:fixed;inset:0;z-index:95;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center}
#prunedlg[hidden]{display:none}
#prunedlg .pbox{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:16px 18px;width:520px;max-width:90vw;font-size:13px;box-shadow:0 12px 40px rgba(0,0,0,.7)}
#prunedlg .ptitle{font-weight:600;margin-bottom:10px}
#prunedlg .pbody{white-space:pre-line;color:var(--fg);line-height:1.5}
#prunedlg .pforce{display:flex;gap:8px;align-items:flex-start;margin:12px 0;color:var(--dim);font-size:12px;cursor:pointer}
#prunedlg .pbtns{display:flex;gap:8px;justify-content:flex-end}
#prunedlg button{background:#333;color:var(--fg);border:1px solid var(--line);padding:6px 14px;cursor:pointer;border-radius:3px}
#prunedlg #pgo{background:#c55;color:#fff;border-color:#c55;font-weight:600}
/* ---- full-workspace grid view (opened from any carousel) ---- */
#gridview{position:fixed;inset:0;z-index:60;background:var(--bg);display:flex;flex-direction:column}
#gridview[hidden]{display:none}
#gridview .ghead2{display:flex;gap:10px;align-items:center;padding:8px 14px;border-bottom:1px solid var(--line);color:var(--dim);font-size:12px}
#gridview .ghead2 b{color:var(--fg);font-weight:500}
#gridview .ghead2 .sz{background:none;border:1px solid var(--line);color:var(--dim);border-radius:3px;cursor:pointer;padding:0 6px;font-size:11px;line-height:16px}
#gridbody{flex:1;overflow-y:auto;display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:8px;padding:12px;align-content:start}
#gridview.big #gridbody{grid-template-columns:repeat(auto-fill,minmax(210px,1fr))}
#gridbody img{width:100%;aspect-ratio:1;object-fit:cover;border:2px solid var(--line);border-radius:4px;cursor:pointer}
#gridpeek{position:fixed;z-index:70;background:var(--bg);border:2px solid var(--focus);border-radius:4px;box-shadow:0 10px 36px rgba(0,0,0,.75);overflow:hidden;pointer-events:none}
#gridpeek[hidden]{display:none}
#gridpeek img{display:block}
.iconb{background:none;border:none;color:var(--dim);cursor:pointer;display:flex;align-items:center;padding:4px}
.iconb:hover{color:var(--fg)}
.car summary .sz svg{vertical-align:middle}
/* ---- asset browser (v4): swaps in for the Generate panel region ---- */
#assets{grid-column:3;grid-row:1;display:flex;flex-direction:column;min-height:0;min-width:0;background:var(--bg)}
#assets[hidden]{display:none}
#assets .ahead{display:flex;gap:8px;align-items:center;padding:6px 10px;border-bottom:1px solid var(--line);font-size:12px;color:var(--dim);flex-wrap:wrap}
#assets .ahead select{background:#111;color:var(--fg);border:1px solid var(--line);padding:3px;min-width:120px}
#assets .ahead button{background:#333;color:var(--fg);border:1px solid var(--line);cursor:pointer;padding:3px 10px}
#agrid{flex:1;overflow-y:auto;display:grid;grid-template-columns:repeat(auto-fill,150px);gap:12px;padding:12px;align-content:start;justify-content:start}
.atile{width:150px;display:flex;flex-direction:column;gap:4px;position:relative}
.atile img{width:150px;height:150px;object-fit:cover;border:2px solid var(--line);border-radius:4px;cursor:pointer}
.atile .gonebox{width:150px;height:150px;display:flex;align-items:center;justify-content:center;border:2px dashed var(--line);border-radius:4px;color:var(--dim);font-size:10px;text-align:center;white-space:pre-line}
.atile .frn{position:absolute;top:4px;left:4px;background:rgba(0,0,0,.6);color:var(--dim);font-size:9px;padding:1px 5px;border-radius:3px;pointer-events:none}
.adesc{width:100%;height:50px;background:#111;border:1px solid var(--line);color:var(--fg);font:11px system-ui;padding:3px;resize:vertical}
.adesc.bad{border-color:#c55}
.atile .ax{position:absolute;top:2px;right:2px;background:rgba(0,0,0,.55);border:none;color:#fff;border-radius:3px;cursor:pointer;padding:1px 6px;display:none}
.atile:hover .ax{display:block}
/* ---- the tabbed action panel (v3): Create | Derive | Camera | Tween ---- */
#tabbar{display:flex;gap:2px;border-bottom:1px solid var(--line)}
.tabb{background:none;border:none;border-bottom:2px solid transparent;color:var(--dim);padding:5px 14px;cursor:pointer;font-size:13px}
.tabb.on{color:var(--fg);border-bottom-color:var(--acc)}
.tpage{display:none}
#shared{display:flex;flex-direction:column;gap:8px;flex:1;min-height:0}
.knobs{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;font-size:12px;color:var(--dim)}
.knobs input[type=number],.knobs select{background:#111;color:var(--fg);border:1px solid var(--line);padding:3px 4px}
.knobs input[type=number]{width:64px}
.knobs input[type=range]{width:90px;vertical-align:middle}
.knobs .off{opacity:.4;pointer-events:none}
#controls.busy #shared,#controls.busy .tpage{opacity:.4;pointer-events:none}
/* camera sections: checkbox reveals the axis control; unchecked = token omitted */
#camrows{display:flex;gap:26px;flex-wrap:wrap;align-items:flex-start}
.csec .chead{display:flex;gap:6px;align-items:center;color:var(--fg);font-size:12px;cursor:pointer;user-select:none}
.csec .cbody{display:none;padding:6px 2px 0}
.csec.on .cbody{display:block}
.csec input[type=range]{width:150px}
.stops{display:flex;justify-content:space-between;width:150px;font-size:10px;color:var(--dim)}
.stops span{cursor:pointer;text-align:center}
.stops span.on{color:var(--acc)}
#dial circle.ring{fill:none;stroke:var(--line);stroke-width:1.5}
#dial line.needle{stroke:var(--focus);stroke-width:2;stroke-linecap:round}
#dial circle.pt{fill:#333;stroke:var(--line);stroke-width:1;cursor:pointer}
#dial circle.pt:hover{stroke:var(--fg)}
#dial circle.pt.on{fill:var(--focus);stroke:var(--focus)}
#dial text{fill:var(--dim);font-size:9px;text-anchor:middle;dominant-baseline:middle;pointer-events:none}
#dialval{color:var(--acc);font-size:11px;text-align:center;padding-top:2px;min-height:14px}
/* one Generate for all tabs, fixed size, pinned bottom-right */
#actionfoot{margin-top:auto;display:flex;gap:12px;align-items:center;justify-content:flex-end;padding-top:6px}
#status{color:var(--dim);font-size:12px;flex:1;min-height:14px}
#seedused{color:var(--dim);font-size:11px}
#gen{background:var(--acc);color:#000;font-weight:600;border:none;border-radius:4px;width:170px;height:44px;font-size:14px;cursor:pointer;flex:none}
#gen:disabled{background:#665;color:#aaa;cursor:not-allowed}
#gen.stop{background:#c55;color:#fff}
</style></style></head><body>
<div id="app">
<nav>
  <div id="proj" title="project = a subdir of the root; its only effect is the path context for images. Each project has its own background tint"><span id="projsw" class="sw"></span><select id="projsel"></select><button id="projnew" title="create a new project">+</button></div>
  <a id="nav-evolver" class="on">Evolver</a><a id="nav-assets">Assets</a>
  <div class="spacer"></div>
</nav>
<main>
  <div id="top">
    <details class="car" id="car-hist" open><summary><b>History</b> <span class="n">0</span> — images bred from: the ref0 of each Generate, the source of each POV; newest first</summary>
      <div class="row"><button class="arrow" data-dir="-1">‹</button><div class="strip"></div><button class="arrow" data-dir="1">›</button></div></details>
    <details class="car drop" id="car-pin" data-target="pin" data-index="999"><summary><b>Pinned</b> <span class="n">0</span> — P pins the selection; Del here unpins; drop / paste to pin</summary>
      <div class="row"><button class="arrow" data-dir="-1">‹</button><div class="strip"></div><button class="arrow" data-dir="1">›</button></div></details>
    <details class="car" id="car-all"><summary><b>All</b> <span class="n">0</span> — every image in this project, by number</summary>
      <div class="row"><button class="arrow" data-dir="-1">‹</button><div class="strip"></div><button class="arrow" data-dir="1">›</button></div></details>
    <details id="genea"><summary><b>Genealogy</b> — of the Working Image</summary>
      <details class="car" id="car-gpar" open><summary><b>Parents</b> <span class="n">0</span> — its reference images</summary>
        <div class="row"><button class="arrow" data-dir="-1">‹</button><div class="strip"></div><button class="arrow" data-dir="1">›</button></div></details>
      <details class="car" id="car-gsib" open><summary><b>Siblings</b> <span class="n">0</span> — identical config, seeds aside</summary>
        <div class="row"><button class="arrow" data-dir="-1">‹</button><div class="strip"></div><button class="arrow" data-dir="1">›</button></div></details>
      <details class="car" id="car-gkid" open><summary><b>Children</b> <span class="n">0</span> — images that used it as an input</summary>
        <div class="row"><button class="arrow" data-dir="-1">‹</button><div class="strip"></div><button class="arrow" data-dir="1">›</button></div></details>
    </details>
  </div>
  <div id="work">
    <div id="stage">
      <div id="stagebox" class="drop" data-target="working" tabindex="0"><div class="hint">working image<br>drop · paste · double-click a candidate</div></div>
      <div id="recipe"></div>
    </div>
    <div id="split" title="drag to resize"></div>
    <div id="genpanel">
      <div id="controls">
        <div id="tabbar">
          <button class="tabb" data-tab="create">Create</button>
          <button class="tabb" data-tab="derive">Derive</button>
          <button class="tabb" data-tab="camera">Camera</button>
          <button class="tabb" data-tab="tween">Tween</button>
        </div>
        <div id="shared">
          <div id="prompts">
            <textarea id="prompt" placeholder="+ive prompt"></textarea>
            <textarea id="negative" placeholder="-ive prompt"></textarea>
          </div>
          <div class="knobs">
            <label>steps <input type="number" id="steps" min="0" max="100" style="width:48px" title="0 = family default"></label>
            <label>cfg <input type="number" id="cfg" min="0" max="20" step="0.5" style="width:52px" title="0 = family default"></label>
            <label>lora <select id="lora"><option value="">(none)</option></select></label>
            <label>str <input type="number" id="lstr" step="0.1" min="0" max="4" value="1"></label>
            <label><input type="checkbox" id="whitebg" checked> white bg</label>
          </div>
        </div>
        <div class="tpage" data-tab="create">
          <div class="knobs">
            <label>model <select id="family"></select></label>
            <label>Outputs: <input type="number" id="outputs_create" min="1" max="64" value="6"></label>
            <label>base seed <input type="number" id="seed_create" value="0" title="0 = random base; candidate k gets base+k"></label>
          </div>
        </div>
        <div class="tpage" data-tab="derive">
          <div class="knobs">
            <span id="refs">
              <div class="ref drop" data-target="ref0" data-index="0" tabindex="0" title="reference 0 — the identity/continuity carrier. Click the working image to put it here (click while holding Space = its parent); picking an image auto-sets its parent"><span class="hint">ref0</span></div>
              <div class="ref drop" data-target="ref" data-index="0" tabindex="0"><span class="hint">+</span></div>
              <div class="ref drop" data-target="ref" data-index="1" tabindex="0"><span class="hint">+</span></div>
              <div class="ref drop" data-target="ref" data-index="2" tabindex="0"><span class="hint">+</span></div>
            </span>
            <label>lock <select id="lock"><option>SOFT_LOCK</option><option>MID_LOCK</option><option>HARD_LOCK</option></select></label>
            <label>vary <input type="range" id="vary" min="0" max="0.5" step="0.02" value="0" title="sibling variety: seeded noise added to the reference latent. 0.1-0.3 useful; above ~0.45 the reference is mostly noise and output degrades"> <span id="varyv">0</span></label>
            <label>Outputs: <input type="number" id="outputs_derive" min="1" max="64" value="6"></label>
            <label>base seed <input type="number" id="seed_derive" value="0" title="0 = random base; candidate k gets base+k"></label>
          </div>
        </div>
        <div class="tpage" data-tab="camera">
          <div id="camrows">
            <div class="csec">
              <label class="chead"><input type="checkbox" id="chk_dist"> Distance</label>
              <div class="cbody" id="body_dist"><input type="range" id="cam_dist" min="0" max="2" step="1" value="1"><div class="stops" id="stops_dist"></div></div>
            </div>
            <div class="csec">
              <label class="chead"><input type="checkbox" id="chk_elev"> Elevation</label>
              <div class="cbody" id="body_elev"><input type="range" id="cam_elev" min="0" max="3" step="1" value="1"><div class="stops" id="stops_elev"></div></div>
            </div>
            <div class="csec">
              <label class="chead"><input type="checkbox" id="chk_azim"> Orbit</label>
              <div class="cbody" id="body_azim"><svg id="dial" viewBox="0 0 132 132" width="132" height="132"></svg><div id="dialval"></div></div>
            </div>
          </div>
          <div class="knobs">
            <label>Outputs: <input type="number" id="outputs_camera" min="1" max="64" value="4"></label>
            <label>base seed <input type="number" id="seed_camera" value="0" title="0 = random base"></label>
          </div>
          <div class="hint" style="padding:2px 0">re-shoots the working image at exactly these settings — unchecked axes are left to the model</div>
        </div>
        <div class="tpage" data-tab="tween">
          <div class="hint" style="padding:12px 4px">Tween — two keyframes to motion (Wan FLF2V). Coming soon.</div>
        </div>
        <div id="actionfoot">
          <span id="status"></span>
          <span id="seedused" title="base used last round"></span>
          <button id="gen">Generate</button>
        </div>
      </div>
      <div id="hsplit" title="drag to resize"></div>
      <div id="output">
        <div class="ghead"><b>Output</b> <span class="n">0</span> <span>— P keeps; anything unpinned is archived when the next round runs · Shift+Del discards any image now</span></div>
        <div class="grid" id="sheet"></div>
      </div>
    </div>
  <div id="assets" hidden>
    <div class="ahead">
      <select id="assetsel" title="assets are root-global: shared by every project"></select>
      <button id="assetnew" title="create a new asset (its name is also the LoRA trigger)">+ new</button>
      <button id="assetdel" title="delete this asset — the images themselves are untouched">delete</button>
      <button id="assetfolder" title="import every image in a disk folder (recursive, server-side — fast for big datasets) into this asset">add folder…</button>
      <span class="hint">drag any image in to add it · Del removes an entry · the description IS the training caption and must start with the asset name</span>
      <span style="flex:1"></span>
      <select id="trainfam" title="training family"><option value="zimage">Z-Image</option><option value="klein">Flux Klein (experimental)</option></select>
      <button id="maketrain" title="sync the dataset to _train/ and train a LoRA — this IS the explicit say-so">Make LoRA</button>
      <button id="trainstop" hidden>Stop training</button>
      <span id="trainstat"></span>
    </div>
    <div id="agrid" class="drop" data-target="agrid" data-index="0" tabindex="0" title="click to focus, then Ctrl-V pastes a clipboard image straight into this dataset"></div>
  </div>
  </div>
  <div id="bottom">
    <span class="lab">session</span>
    <label>w <input type="number" id="width" step="16" value="1024"></label>
    <label>h <input type="number" id="height" step="16" value="1024"></label>
    <label title="reload pristine model weights before every round (~15-30s); otherwise only after a round that used a LoRA (ComfyUI #11021)"><input type="checkbox" id="fresh"> reload model on Generate</label>
    <button id="gc" title="archive every image that is not pinned, not in the live working set (WI / refs / candidates), and not an ancestor of those">Collect garbage</button>
    <span id="msg"></span>
    <span class="spacer"></span>
    <span id="progwrap"><progress id="prog" max="1" value="0"></progress> <span id="progtxt"></span></span>
    <span class="svc" title="ComfyUI at 127.0.0.1:8188">ComfyUI <i id="comfy-dot" class="dot"></i></span>
    <span class="svc" title="this evolve server">evolve <i id="evolve-dot" class="dot"></i></span>
  </div>
</main>
<div id="toasts"></div>
<div id="prunedlg" hidden>
  <div class="pbox">
    <div class="ptitle"></div>
    <div class="pbody"></div>
    <label class="pforce"><input type="checkbox" id="pforce"> also unpin / remove from assets, and archive everything in the branch</label>
    <div class="pbtns"><button id="pcancel">Cancel</button><button id="pgo">Prune</button></div>
  </div>
</div>
<div id="gridview" hidden>
  <div class="ghead2"><b id="gridtitle"></b> <span class="n" id="gridcount"></span>
    <button class="sz" id="gridsz" title="toggle thumbnail size">+</button>
    <span class="hint">click selects · dbl-click sends to the WI and closes · hold Space = full-size preview · Esc closes</span>
    <span style="flex:1"></span>
    <button class="iconb" id="gridclose" title="close (Esc)"></button></div>
  <div id="gridbody"></div>
  <div id="gridpeek" hidden><img alt=""></div>
</div>
</div>
<script>
const $ = s => document.querySelector(s);
let S = null;                      // last server snapshot
// ---------- selection: THE singleton ----------
// INVARIANT (user mandate 2026-08-23): there is exactly ZERO or ONE selected
// Image, app-wide, forever - enforced by construction, not convention:
//  - the selection state lives in a closure nothing else can reach;
//  - Sel.apply() is the ONLY code that writes the .focus / .alias classes,
//    and it always clears every instance before applying exactly one;
//  - a new panel/widget joins selection by tagging elements with
//    data-target/data-index (+ data-id on Images) and calling Sel.set() -
//    never by touching classes or state itself.
// Sel is frozen so its methods cannot be replaced; target/index are getters.
const Sel = (() => {
  let cur = {target: 'working', index: 0};
  const get = () => ({target: cur.target, index: cur.index});
  const el = () => cur.target === 'none' ? null
    : cur.target === 'working' ? $('#stagebox')
    : document.querySelector(`[data-target="${cur.target}"][data-index="${cur.index}"]`);
  const id = () => {
    if (!S || cur.target === 'none') return null;
    if (cur.target === 'working') return S.working;
    if (cur.target === 'ref') return S.controls.refs[cur.index];
    if (cur.target === 'ref0') return S.controls.ref0;
    if (cur.target === 'slot') return (S.candidates[activeTab()] || [])[cur.index] ?? null;
    if (cur.target === 'pin') return S.pins[cur.index] ?? null;
    const l = listFor(cur.target); if (l) return l[cur.index] ?? null;
    return null;
  };
  function apply() {
    document.querySelectorAll('.focus, .alias').forEach(e => e.classList.remove('focus', 'alias'));
    const e0 = el();
    if (e0) { e0.classList.add('focus'); if (e0.tagName === 'IMG') e0.scrollIntoView({inline: 'nearest', block: 'nearest'}); }
    const i = id();
    if (i != null) {    // SELECTED_ALIAS: every other Image showing the same underlying image.
      // Style the enclosing [data-target] box, not the bare img: an img that
      // doesn't fill its container would otherwise show its own outline
      // inside the container's grey border - a double frame (user-reported).
      const seen = new Set();
      document.querySelectorAll(`[data-id="${i}"]`).forEach(x => {
        const box = x.closest('[data-target]') || x;
        if (box === e0 || (e0 && (e0.contains(box) || box.contains(e0))) || seen.has(box)) return;
        seen.add(box);
        box.classList.add('alias');
      });
    }
    if (spaceHeld) peek(cur.target === 'working' ? null : i);
    syncHistory();
  }
  const set = (target, index) => { cur = {target, index: +index || 0}; apply(); };
  const clear = () => { cur = {target: 'none', index: 0}; apply(); };
  return Object.freeze({get, set, clear, id, el, apply,
    get target() { return cur.target; }, get index() { return cur.index; }});
})();
// where imports land when nothing is selected
const selTarget = () => Sel.target === 'none' ? {target: 'working', index: 0} : Sel.get();
let pollTimer = null;

const api = (path, body) => fetch('/api/' + path, body === undefined ? {} :
  {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}).then(r => r.json());
// the standard mutate-and-refresh tail (was hand-repeated at every call site)
const act = (path, body) => api(path, body).then(r => { if (r && r.error) notice(r.error); return refresh(); });
// MUI Material Icons (mui.com/material-ui/material-icons, Apache 2.0),
// inlined as path data; fill-based, colored via currentColor
const I = {
  copy: 'M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z',
  done: 'M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z',
  close: 'M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z',
  grid: 'M3 3v8h8V3H3zm6 6H5V5h4v4zm-6 4v8h8v-8H3zm6 6H5v-4h4v4zm4-16v8h8V3h-8zm6 6h-4V5h4v4zm-6 4v8h8v-8h-8zm6 6h-4v-4h4v4z'
};
const icon = (d, sz) => `<svg viewBox="0 0 24 24" width="${sz || 18}" height="${sz || 18}" fill="currentColor" aria-hidden="true"><path d="${d}"/></svg>`;
const imgURL = id => location.origin + '/img/' + (S ? S.project + '/' : '') + id;

// ---------- per-project background tint ----------
// 8 hues 45deg apart at low saturation: same lightness as the base theme, so
// only the cast changes - text, borders, thumbs and the accent stay put.
const HUES = [210, 255, 300, 345, 30, 75, 120, 165];
let appliedTint = undefined;
function applyTint(c) {
  const r = document.documentElement.style;
  let h, sat = 14, satPanel = 11, dot;
  if (c === 'shared') { h = 38; sat = 28; satPanel = 22; dot = 'hsl(38,70%,50%)'; }
  else if (typeof c === 'number') { h = HUES[c % HUES.length]; dot = `hsl(${h},45%,50%)`; }
  else { r.removeProperty('--bg'); r.removeProperty('--panel'); $('#projsw').style.background = '#666'; return; }
  r.setProperty('--bg', `hsl(${h},${sat}%,11%)`);
  r.setProperty('--panel', `hsl(${h},${satPanel}%,15%)`);
  $('#projsw').style.background = dot;
}

// ---------- named strips: ONE carousel that works anywhere ----------
// Every thumbnail row is a `.car` strip registered here. listFor() feeds the
// generic focus/keyboard machinery (arrows, Space-peek, Enter), so a new
// strip only has to appear in this table to get the full behavior.
let famData = null, famKey = null; // Genealogy sheets cache (anchored to the WI)
const RW = ['working', 'ref', 'ref0', 'slot', 'pin'];   // the only WRITABLE containers
// The strip registry: name -> getter. FLAT by design - a getter carries its
// own availability check, so there is no ordering to get wrong (the old
// if-chain let a genealogy guard swallow the grid/asset lookups). Contract:
// null = "not a list" (unknown name, or its source isn't loaded); an array
// is a list even when empty; entries may be null ids (foreign asset paths).
const LISTS = {
  hist:  () => S.history,
  pin:   () => S.pins,
  all:   () => S.all_ids || [],
  gpar:  () => famData && famData.parents.map(x => x.id),
  gsib:  () => famData && famData.siblings.map(x => x.id),
  gkid:  () => famData && famData.children.map(x => x.id),
  grid:  () => (gridName && gridName !== 'grid') ? listFor(gridName) : null,
  asset: () => { const a = curAssetObj(); return a ? a.dataset.map(e => pathToLocalId(e.path)) : null; },
};
function listFor(name) {
  if (!S) return null;
  const get = LISTS[name];
  const l = get ? get() : null;
  return Array.isArray(l) ? l : null;
}
// ---- grid view: any carousel, full workspace ----
let gridOn = false, gridName = null;
function openGrid(name) {
  gridName = name;
  gridOn = true;
  $('#gridview').hidden = false;
  renderGrid();
}
let gridPeekOn = false;
function gridPeek(start) {
  // hold-Space preview: the selection at ACTUAL size (never upscaled,
  // capped to the viewport), centred over its tile when there is room and
  // pushed inside the viewport at edges/corners; live while arrows move it
  if (start === true) gridPeekOn = true;
  if (!gridPeekOn) return;
  const gp = $('#gridpeek');
  const gid = Sel.target === 'grid' ? Sel.id() : null;
  if (gid == null) { gp.hidden = true; return; }
  const im = gp.firstElementChild, url = imgURL(gid);
  if (im.src !== url) {
    gp.hidden = true;
    im.onload = gridPeekPlace;
    im.src = url;
    if (im.complete && im.naturalWidth) gridPeekPlace();
  } else gridPeekPlace();
}
function gridPeekPlace() {
  if (!gridPeekOn) return;
  const gp = $('#gridpeek'), im = gp.firstElementChild;
  if (!im.naturalWidth) return;
  const M = 8, vw = innerWidth, vh = innerHeight;
  const sc = Math.min(1, (vw - 2 * M) / im.naturalWidth, (vh - 2 * M) / im.naturalHeight);
  const w = Math.round(im.naturalWidth * sc), h = Math.round(im.naturalHeight * sc);
  const el = Sel.el();
  const r = el ? el.getBoundingClientRect() : {left: vw / 2, top: vh / 2, width: 0, height: 0};
  const x = Math.max(M, Math.min(vw - w - M, r.left + r.width / 2 - w / 2));
  const y = Math.max(M, Math.min(vh - h - M, r.top + r.height / 2 - h / 2));
  im.style.width = w + 'px'; im.style.height = h + 'px';
  Object.assign(gp.style, {left: x + 'px', top: y + 'px'});
  gp.hidden = false;
}
function gridPeekEnd() {
  gridPeekOn = false;
  const gp = $('#gridpeek');
  if (gp) gp.hidden = true;
}
function closeGrid() {
  gridPeekEnd();
  gridOn = false;
  $('#gridview').hidden = true;
  if (S) render();
}
function renderGrid() {
  if (!gridOn || !S) return;
  const ids = (listFor(gridName) || []).filter(x => x != null);
  const src = document.querySelector('#car-' + gridName + ' summary b');
  $('#gridtitle').textContent = src ? src.textContent : gridName;
  $('#gridcount').textContent = ids.length;
  $('#gridview').classList.toggle('big', localStorage.getItem('size:grid') === '1');
  $('#gridsz').textContent = localStorage.getItem('size:grid') === '1' ? '-' : '+';
  const g = $('#gridbody');
  g.innerHTML = '';
  ids.forEach((id, k) => {
    const im = thumb(id);
    im.dataset.target = 'grid';
    im.dataset.index = k;
    im.addEventListener('dblclick', () => { closeGrid(); act('place', {id, target: 'working'}); });
    g.appendChild(im);
  });
}
function setOpen(car, open) {   // programmatic toggle: not a user preference
  car._forced = open;
  car.open = open;
}
function fillStrip(name, els, empty) {
  const car = $('#car-' + name);
  car.querySelector('.n').textContent = els.length;
  // STRICT rule (user, 2026-08-24): zero items = cannot expand. No
  // empty-hint exception - that loophole let sheets open onto nothing.
  // Forcing shut must not clobber the sticky preference, hence setOpen().
  const openable = els.length > 0;
  car.classList.toggle('empty', !openable);
  if (!openable && car.open) setOpen(car, false);
  else if (openable && car.dataset.wasEmpty === '1' && localStorage.getItem('open:' + car.id) !== '0') setOpen(car, true);
  car.dataset.wasEmpty = openable ? '0' : '1';
  const strip = car.querySelector('.strip');
  strip.innerHTML = '';
  if (!car.open) return;
  if (!els.length) { if (empty) strip.innerHTML = '<span class="hint">' + empty + '</span>'; return; }
  els.forEach((el, k) => { el.dataset.target = name; el.dataset.index = k; strip.appendChild(el); });
}

// ---------- rendering ----------
async function refresh() {
  try {
    S = await api('state');
    $('#evolve-dot').className = 'dot ok';
  } catch (err) {
    $('#evolve-dot').className = 'dot bad';
    clearTimeout(pollTimer);
    pollTimer = setTimeout(refresh, 3000);
    return;
  }
  render();
  clearTimeout(pollTimer);
  // 1s while generating (slots fill in as candidates land), 3s idle (cheap;
  // picks up anything another tab or the server did)
  pollTimer = setTimeout(refresh, S.busy ? 1000 : 3000);
}

function thumb(id, cls) {
  const im = document.createElement('img');
  im.src = imgURL(id); im.dataset.id = id; im.draggable = true; im.className = cls || '';
  im.loading = 'lazy';
  im.title = tip(id);
  im.addEventListener('dragstart', e => dragStart(e, id));
  return im;
}
function tip(id) {
  const m = S.meta[id]; if (!m) return '#' + id;
  const r = m.recipe;
  return `#${id}  ${m.w}x${m.h}` + (r ? `\n${r.family || 'klein'}  seed ${r.seed}  ${r.lock || 'fiat'}  vary ${r.vary}` +
    (r.lora ? `  lora ${r.lora}@${r.lora_strength}` : '') + `\n${r.prompt}` : `\n${m.source}`) + `\n${m.path}`;
}

function render() {
  if (!S) return;
  renderCarousel('hist', S.history);
  renderCarousel('pin', S.pins);
  renderCarousel('all', S.all_ids || []);
  // stage
  const box = $('#stagebox'); box.innerHTML = '';
  if (S.working != null) {
    const im = thumb(S.working); box.appendChild(im);
    $('#recipe').textContent = tip(S.working).replace(/\n/g, '  ·  ');
  } else {
    box.innerHTML = '<div class="hint">working image<br>drop · paste · double-click a candidate</div>';
    $('#recipe').textContent = '';
  }
  if (peekId != null) peek(peekId);     // survive a re-render mid-hold
  renderSlots();
  // controls: never overwrite a widget you are editing (idle polls re-render)
  const c = S.controls;
  if (!$('#controls').contains(document.activeElement) && !$('#bottom').contains(document.activeElement)) {
    $('#prompt').value = c.prompt; $('#negative').value = c.negative || '';
    const fs = $('#family');
    fs.innerHTML = Object.entries(S.families).map(([k, f]) => `<option value="${k}">${f.label}</option>`).join('');
    fs.value = S.families[c.family] ? c.family : 'klein';
    $('#steps').value = c.steps || ''; $('#cfg').value = c.cfg || '';
    const lsel = $('#lora');
    lsel.innerHTML = '<option value="">(none)</option>' + S.loras.map(l => `<option>${l}</option>`).join('');
    lsel.value = S.loras.includes(c.lora) ? c.lora : '';
    $('#lstr').value = c.lora_strength; $('#lock').value = c.lock;
    $('#vary').value = c.vary; $('#varyv').textContent = c.vary;
    $('#width').value = c.width; $('#height').value = c.height;
    $('#whitebg').checked = c.whitebg;
    $('#fresh').checked = !!c.fresh_model;
    ['create', 'derive', 'camera'].forEach(t => {
      $('#outputs_' + t).value = c['outputs_' + t] || 6;
      $('#seed_' + t).value = c['seed_' + t] || 0;
    });
    camSync(c);
    if (c.tab && c.tab !== activeTab()) setTab(c.tab, false);
  }
  $('#seedused').textContent = S.last_base_seed ? `(last used ${S.last_base_seed})` : '';
  if (S.color !== appliedTint) { applyTint(S.color); appliedTint = S.color; }
  const ps = $('#projsel');
  if (document.activeElement !== ps) {
    ps.innerHTML = (S.projects || []).map(n => `<option>${n}</option>`).join('');
    ps.value = S.project;
  }
  familyUI();
  document.querySelectorAll('.ref[data-target="ref"]').forEach((r, i) => {
    r.innerHTML = '';
    const id = c.refs[i];
    if (id != null) r.appendChild(thumb(id)); else r.innerHTML = '<span class="hint">+</span>';
  });
  const r0 = document.querySelector('.ref[data-target="ref0"]');
  r0.innerHTML = '';
  if (c.ref0 != null) r0.appendChild(thumb(c.ref0)); else r0.innerHTML = '<span class="hint">ref0</span>';
  $('#status').textContent = S.busy ? 'per-step progress is on the evolve.py console' : '';   // #msg is separate: never clobbered by polls
  genUI();
  statusBar();
  renderGenealogy();
  renderAssets();
  trainUI();
  renderGrid();
  Sel.apply();
}

// ---------- asset browser (v4): a view over root/assets.json ----------
let mode = localStorage.getItem('mode') || 'evolver';
let curAsset = localStorage.getItem('asset:last') || null;
function setMode(m) {
  mode = m;
  localStorage.setItem('mode', m);
  $('#nav-evolver').classList.toggle('on', m === 'evolver');
  $('#nav-assets').classList.toggle('on', m === 'assets');
  $('#genpanel').style.display = m === 'assets' ? 'none' : '';
  $('#assets').hidden = m !== 'assets';
  if (S) render();
}
$('#nav-evolver').addEventListener('click', () => setMode('evolver'));
$('#nav-assets').addEventListener('click', () => setMode('assets'));

function curAssetObj() {
  return (S.assets || []).find(a => a.name === curAsset) || (S.assets || [])[0] || null;
}
function pathToLocalId(path) {
  const pre = S.project + '/images/';
  if (path.startsWith(pre) && path.endsWith('.png')) {
    const n = +path.slice(pre.length, -4);
    if (Number.isInteger(n)) return n;
  }
  return null;
}
function renderAssets() {
  if (mode !== 'assets' || !S) return;
  const sel = $('#assetsel');
  const a = curAssetObj();
  curAsset = a ? a.name : null;
  if (document.activeElement !== sel) {
    sel.innerHTML = (S.assets || []).map(x => `<option>${x.name}</option>`).join('');
    if (a) sel.value = a.name;
  }
  // never rebuild the grid under an in-progress caption edit
  if ($('#agrid').contains(document.activeElement)) return;
  const g = $('#agrid');
  g.innerHTML = '';
  if (!a) { g.innerHTML = '<span class="hint">no assets yet — “+ new” creates one</span>'; return; }
  a.dataset.forEach((e, k) => {
    const t = document.createElement('div');
    t.className = 'atile';
    t.dataset.target = 'asset';
    t.dataset.index = k;
    t.tabIndex = 0;
    const lid = pathToLocalId(e.path);
    const im = document.createElement('img');
    im.src = '/file/' + e.path;
    im.title = e.path;
    im.dataset.path = e.path;   // Info popup by path (works for foreign projects)
    if (lid != null) {
      im.dataset.id = lid;
      im.addEventListener('dblclick', () => act('place', {id: lid, target: 'working'}));
    } else {
      const b = document.createElement('span');
      b.className = 'frn';
      b.textContent = e.path.split('/')[0];   // which project it lives in
      t.appendChild(b);
    }
    im.addEventListener('error', () => {
      const d = document.createElement('div');
      d.className = 'gonebox';
      d.textContent = e.path + String.fromCharCode(10) + '(missing)';
      im.replaceWith(d);
    });
    const x = document.createElement('button');
    x.className = 'ax'; x.innerHTML = icon(I.close, 13);
    x.title = 'remove from this asset (the image itself is untouched)';
    x.addEventListener('click', () => act('asset', {op: 'remove', name: a.name, path: e.path}));
    const d = document.createElement('textarea');
    d.className = 'adesc';
    d.value = e.description || '';
    const mark = () => d.classList.toggle('bad', !d.value.startsWith(a.name));
    mark();
    d.title = 'training caption — must START with "' + a.name + '" (the trigger); red border = it does not';
    d.addEventListener('input', mark);
    d.addEventListener('change', () => api('asset', {op: 'describe', name: a.name, path: e.path, description: d.value}).then(refresh));
    t.append(im, x, d);
    g.appendChild(t);
  });
}
function trainUI() {
  const t = S && S.train;
  const running = !!(t && t.running);
  $('#maketrain').disabled = running || !!(S && S.busy);
  $('#trainstop').hidden = !running;
  $('#trainstat').textContent = !t ? '' :
    t.running ? `training ${t.asset} (${t.family}) · ${Math.floor(t.elapsed / 60)}m${t.elapsed % 60}s · tail -f ${t.log}` :
    t.error ? `${t.asset}: ${t.error}` : `${t.asset}: done ✓`;
}
$('#maketrain').addEventListener('click', async () => {
  const a = curAssetObj();
  if (!a) { flash('create an asset first'); return; }
  const fam = $('#trainfam').value;
  if (!confirm(`Train "${a.name}" (${fam}) on ${a.dataset.length} image(s)?` +
      String.fromCharCode(10) + 'This runs on the GPU and blocks generation until done.')) return;
  const r = await api('train', {name: a.name, family: fam});
  if (r.error) alert(r.error);
  refresh();
});
$('#trainstop').addEventListener('click', async () => {
  if (!confirm('Abort training? Progress in this run is lost.')) return;
  await api('train_abort', {});
  refresh();
});
$('#assetsel').addEventListener('change', () => {
  curAsset = $('#assetsel').value;
  localStorage.setItem('asset:last', curAsset);
  renderAssets();
});
$('#assetfolder').addEventListener('click', async () => {
  const a = curAssetObj();
  if (!a) { flash('create an asset first'); return; }
  const path = prompt('folder to import into "' + a.name + '" (recursive):');
  if (!path) return;
  flash('importing folder…');
  const r = await api('import_folder', {path: path.trim(), asset: a.name});
  if (r.error) { alert(r.error); return; }
  flash(`${r.added} added, ${r.duplicates} duplicates, ${r.skipped} skipped (of ${r.total})`);
  refresh();
});
$('#assetnew').addEventListener('click', () => {
  const name = prompt('new asset name (it is also the LoRA trigger — letters, digits, - _):');
  if (!name) return;
  api('asset', {op: 'create', name: name.trim()}).then(r => {
    if (r.error) { flash(r.error); return; }
    curAsset = name.trim();
    localStorage.setItem('asset:last', curAsset);
    refresh();
  });
});
$('#assetdel').addEventListener('click', () => {
  const a = curAssetObj();
  if (!a) return;
  if (!confirm(`delete asset "${a.name}"? Images are untouched; only the grouping goes.`)) return;
  act('asset', {op: 'delete', name: a.name});
});
async function walkEntry(en) {
  // recursive directory walk of a dropped Explorer folder. Entries must be
  // grabbed synchronously at drop time (they go inert after an await).
  if (en.isFile) {
    return new Promise(res => en.file(f => res(f.type.startsWith('image/') ? [f] : []),
                                      () => res([])));
  }
  if (en.isDirectory) {
    const rd = en.createReader();
    let out = [], batch;
    do {
      batch = await new Promise(res => rd.readEntries(res, () => res([])));
      for (const c2 of batch) out = out.concat(await walkEntry(c2));
    } while (batch.length);
    return out;
  }
  return [];
}
async function assetAddId(a, id) {
  // add a just-imported image to the asset, caption seeded from its
  // gleaned/recorded prompt (shared by drop, folder-walk and paste)
  const m = await api('meta', {id});
  const pr = m && m.recipe && m.recipe.prompt;
  await api('asset', {op: 'add', name: a.name,
                      path: S.project + '/images/' + id + '.png',
                      description: pr ? a.name + ', ' + pr : a.name});
}
async function assetDrop(dt) {
  const a = curAssetObj();
  if (!a) { flash('create an asset first'); return; }
  const entries = [...(dt.items || [])]
    .map(it => it.webkitGetAsEntry && it.webkitGetAsEntry())
    .filter(Boolean);
  if (entries.some(en => en.isDirectory)) {
    flash('reading folder…');
    let files = [];
    for (const en of entries) files = files.concat(await walkEntry(en));
    let added = 0;
    for (const f of files) {
      const r = await fetch('/api/import', {method: 'POST', body: f}).then(x => x.json());
      if (r.error) continue;
      await assetAddId(a, r.id);
      added++;
    }
    flash(`${added} of ${files.length} images added from folder`);
    refresh();
    return;
  }
  const add = async (path, id) => {
    let desc = a.name;
    if (id != null) {   // seed the caption from the image's own prompt
      const m = await api('meta', {id});
      if (m && m.recipe && m.recipe.prompt) desc = a.name + ', ' + m.recipe.prompt;
    }
    await api('asset', {op: 'add', name: a.name, path, description: desc});
  };
  const own = dt.getData('application/x-evolver');
  if (own) { await add(S.project + '/images/' + own + '.png', +own); refresh(); return; }
  const uri = (dt.getData('text/uri-list') || '').split(String.fromCharCode(10)).map(x => x.trim()).find(x => x && !x.startsWith('#'));
  if (uri && uri.startsWith(location.origin + '/img/')) {
    const parts = uri.split('/');
    const id = +parts.pop(), proj = parts.pop();
    await add(proj + '/images/' + id + '.png', proj === S.project ? id : null);
    refresh(); return;
  }
  const files = [...(dt.files || [])].filter(f => f.type.startsWith('image/'));
  for (const f of files) {   // external file: import to the project, then add
    const r = await fetch('/api/import', {method: 'POST', body: f}).then(x => x.json());
    if (r.error) { flash(r.error); continue; }
    await add(S.project + '/images/' + r.id + '.png', r.id);
  }
  if (files.length) refresh();
  else if (!own && !uri) flash('drop an image (or drag one from any strip)');
}

// ---------- prune dialog ----------
let pruneId = null, prunePlan = null;
function pruneText(p) {
  const n = p.archive.length;
  let t = `Branch under #${p.root}: ${p.branch} image${p.branch === 1 ? '' : 's'} (mother-line).` + String.fromCharCode(10);
  t += `Archive ${n}.`;
  if (p.keep.length) {
    t += ` Keep ${p.keep.length}:` + String.fromCharCode(10) +
      p.keep.map(k => `   #${k.id} — ${k.why}`).join(String.fromCharCode(10));
  }
  if (p.unpin.length) t += String.fromCharCode(10) + `Unpin: ${p.unpin.map(i => '#' + i).join(', ')}.`;
  if (p.asset_removals.length) {
    const by = {};
    p.asset_removals.forEach(r => { by[r.asset] = (by[r.asset] || 0) + 1; });
    t += String.fromCharCode(10) + 'Remove from assets: ' +
      Object.entries(by).map(([a, c]) => `${a} ×${c}`).join(', ') + '.';
  }
  const live = [];
  if (p.live.working) live.push('the working image');
  if (p.live.ref0) live.push('ref0');
  if (p.live.refs) live.push(`${p.live.refs} reference slot(s)`);
  if (live.length) t += String.fromCharCode(10) + 'Clears: ' + live.join(', ') + '.';
  if (p.outside_refs) t += String.fromCharCode(10) + `Also referenced by ${p.outside_refs} image(s) outside the branch (not touched).`;
  if (!n) t += String.fromCharCode(10) + 'Nothing to archive under this plan.';
  return t;
}
async function openPrune(id, plan) {
  pruneId = id; prunePlan = plan;
  $('#pforce').checked = false;
  $('#prunedlg .ptitle').textContent = `Prune #${id} and its branch`;
  $('#prunedlg .pbody').textContent = pruneText(plan);
  $('#pgo').disabled = !plan.archive.length;
  $('#prunedlg').hidden = false;
}
function closePrune() { $('#prunedlg').hidden = true; pruneId = null; }
$('#pforce').addEventListener('change', async () => {
  const plan = await api('prune', {id: pruneId, force: $('#pforce').checked});
  if (plan.error) { notice(plan.error); return; }
  prunePlan = plan;
  $('#prunedlg .pbody').textContent = pruneText(plan);
  $('#pgo').disabled = !plan.archive.length;
});
$('#pcancel').addEventListener('click', closePrune);
$('#pgo').addEventListener('click', async () => {
  const r = await api('prune', {id: pruneId, force: $('#pforce').checked, apply: true});
  closePrune();
  if (r.error) notice(r.error);
  else flash(`pruned: ${r.archive.length} archived`);
  refresh();
});

// ---------- the tabbed action panel: Create | Derive | Camera | Tween ----------
// One Generate button for all tabs (fixed, bottom-right), dispatching on the
// active tab. The tab IS the mode: Create = fiat (no refs, model dropdown
// live), Derive = Klein refs+IFT, Camera = absolute re-shoot of the WI.
function activeTab() { const b = document.querySelector('.tabb.on'); return b ? b.dataset.tab : 'create'; }
function setTab(t, user) {
  document.querySelectorAll('.tabb').forEach(b => b.classList.toggle('on', b.dataset.tab === t));
  document.querySelectorAll('.tpage').forEach(pg => { pg.style.display = pg.dataset.tab === t ? 'block' : 'none'; });
  $('#shared').style.display = (t === 'create' || t === 'derive') ? '' : 'none';
  if (S) {
    familyUI(); genUI(); renderSlots();   // the grid shows THIS tab's outputs
    if (user) {   // the Output grid previews the ACTIVE tab's outputs count
      const o = $('#outputs_' + t);
      if (o) act('slots', {slots: +o.value || 1});
      saveControls();
      // USER-switch to Derive with an empty ref0: seed it from the WI
      // (w/h follow, like the Space-click gesture). Occupied ref0 is never
      // touched - a pick's restored parent outranks the WI. Del + return
      // refills; changing the WI while ON the tab does not.
      if (t === 'derive' && S.controls.ref0 == null && S.working != null) {
        api('place', {id: S.working, target: 'ref0'}).then(() => {
          const m = S.meta[S.working];
          if (m) { $('#width').value = m.w; $('#height').value = m.h; saveControls(); }
          refresh();
        });
      }
    }
  }
}
document.querySelectorAll('.tabb').forEach(b => b.addEventListener('click', () => setTab(b.dataset.tab, true)));

// ---- camera axes: checkbox reveals the control; unchecked = token omitted ----
function camVal(axis) {
  if (!S || !$('#chk_' + axis).checked) return null;
  if (axis === 'azim') return dialSel;
  const list = axis === 'dist' ? S.pov_dist : S.pov_elev;
  const k = +$('#cam_' + axis).value;
  return (list[k] || list[0])[0];
}
function setAxis(axis, list, key, dflt) {
  const on = key != null;
  $('#chk_' + axis).checked = on;
  $('#body_' + axis).parentElement.classList.toggle('on', on);
  const k = list.findIndex(x => x[0] === key);
  if (k >= 0) $('#cam_' + axis).value = k;
  else if (!on) $('#cam_' + axis).value = dflt;
}
function stopsFill(axis, list) {
  const box = $('#stops_' + axis);
  if (box.dataset.built !== '1') {
    box.dataset.built = '1';
    box.innerHTML = list.map((x, i) => `<span data-i="${i}">${x[1]}</span>`).join('');
    box.addEventListener('click', e => {
      const sp = e.target.closest('span'); if (!sp) return;
      $('#cam_' + axis).value = sp.dataset.i;
      $('#chk_' + axis).checked = true;
      $('#body_' + axis).parentElement.classList.add('on');
      camPaint(); saveControls(); genUI();
    });
  }
  [...box.children].forEach((sp, i) => sp.classList.toggle('on', +$('#cam_' + axis).value === i && $('#chk_' + axis).checked));
}
function camPaint() {
  if (!S) return;
  stopsFill('dist', S.pov_dist);
  stopsFill('elev', S.pov_elev);
  dialSet(dialSel);
}
function camSync(c) {
  if (!S) return;
  setAxis('dist', S.pov_dist, c.pov_dist, 1);
  setAxis('elev', S.pov_elev, c.pov_elev, 1);
  $('#chk_azim').checked = c.pov_azim != null;
  $('#body_azim').parentElement.classList.toggle('on', c.pov_azim != null);
  if (c.pov_azim != null) dialSel = c.pov_azim;
  camPaint();
}
['dist', 'elev', 'azim'].forEach(a => $('#chk_' + a).addEventListener('change', () => {
  $('#body_' + a).parentElement.classList.toggle('on', $('#chk_' + a).checked);
  camPaint(); saveControls(); genUI();
}));
['dist', 'elev'].forEach(a => $('#cam_' + a).addEventListener('input', () => { camPaint(); saveControls(); }));

// ---- the Orbit compass dial: front at 12 o'clock, clockwise ----
const AZ_ORDER = ['front', 'front-right', 'right', 'back-right', 'back', 'back-left', 'left', 'front-left'];
const AZ_SHORT = {front: 'F', 'front-right': 'FR', right: 'R', 'back-right': 'BR',
                  back: 'B', 'back-left': 'BL', left: 'L', 'front-left': 'FL'};
let dialSel = 'front';
function dialBuild() {
  const svg = $('#dial'), C = 66, R = 44, LR = 58;
  let h = `<circle class="ring" cx="${C}" cy="${C}" r="${R}"></circle>` +
          `<line class="needle" x1="${C}" y1="${C}" x2="${C}" y2="${C - R + 10}"></line>`;
  AZ_ORDER.forEach((az, k) => {
    const a = k * Math.PI / 4;
    h += `<circle class="pt" data-az="${az}" cx="${C + R * Math.sin(a)}" cy="${C - R * Math.cos(a)}" r="7"><title>${az} view</title></circle>`;
    h += `<text x="${C + LR * Math.sin(a)}" y="${C - LR * Math.cos(a)}">${AZ_SHORT[az]}</text>`;
  });
  svg.innerHTML = h;
  svg.addEventListener('click', e => {
    const pt = e.target.closest('.pt'); if (!pt) return;
    $('#chk_azim').checked = true;
    $('#body_azim').parentElement.classList.add('on');
    dialSet(pt.dataset.az); saveControls(); genUI();
  });
}
function dialSet(az) {
  dialSel = az;
  const k = AZ_ORDER.indexOf(az), a = k * Math.PI / 4, C = 66, R = 44;
  const n = document.querySelector('#dial .needle');
  if (n) { n.setAttribute('x2', C + (R - 10) * Math.sin(a)); n.setAttribute('y2', C - (R - 10) * Math.cos(a)); }
  document.querySelectorAll('#dial .pt').forEach(c2 => c2.classList.toggle('on', c2.dataset.az === az));
  $('#dialval').textContent = $('#chk_azim').checked ? az + ' view' : '';
}
dialBuild();

// ---- one button, four meanings ----
function genUI() {
  if (!S) return;
  const t = activeTab(), b = $('#gen');
  $('#controls').classList.toggle('busy', !!S.busy);
  if (S.train && S.train.running) {
    b.disabled = true;
    b.classList.remove('stop');
    b.textContent = 'Training…';
    b.title = 'a LoRA is training - one GPU';
    return;
  }
  if (S.busy) {   // the button MORPHS into the abort control
    b.disabled = false;
    b.classList.add('stop');
    b.textContent = `Stop ${S.busy.done}/${S.busy.total}`;
    b.title = 'abort this round: finished candidates stay, the in-flight render is interrupted';
    return;
  }
  b.classList.remove('stop');
  let on = true, label = 'Generate', tip = '';
  if (t === 'derive' && S.controls.ref0 == null) { on = false; tip = 'set ref0 — Derive breeds FROM an image'; }
  else if (t === 'camera') {
    label = 'Re-shoot';
    if (S.working == null) { on = false; tip = 'no working image to re-shoot'; }
    else if (!(camVal('azim') || camVal('elev') || camVal('dist'))) { on = false; tip = 'check at least one axis'; }
  } else if (t === 'tween') { on = false; tip = 'coming soon'; }
  b.disabled = !on;
  b.textContent = label;
  b.title = tip;
}

// ---------- Genealogy sheets: parents / siblings / children of the WI ----------
// v2 (2026-08-23): flat read-only carousels anchored to the Working Image.
// No walking, no decks - dbl-click an ancestor to make IT the WI instead.
async function renderGenealogy() {
  if (!$('#genea').open) return;
  if (!S || S.working == null) {
    famData = null; famKey = null;
    ['gpar', 'gsib', 'gkid'].forEach(n => fillStrip(n, [], 'no working image'));
    return;
  }
  const key = S.working + ':' + JSON.stringify(S.candidates);   // children grow as a round lands
  if (key !== famKey) {
    const r = await api('family', {id: S.working});
    if (r.error) return;
    famData = r; famKey = key;
  }
  const tile = t => {
    const im = thumb(t.id);
    im.addEventListener('dblclick', () => act('place', {id: t.id, target: 'working'}));
    return im;
  };
  fillStrip('gpar', famData.parents.map(tile), 'fiat — no reference images');
  fillStrip('gsib', famData.siblings.map(tile));
  fillStrip('gkid', famData.children.map(tile), 'no children yet');
}
{ // the Genealogy section itself: sticky open/close like any carousel
  const gn = $('#genea'), gk = 'open:genea';
  const saved = localStorage.getItem(gk);
  if (saved != null) gn.open = saved === '1';
  gn.addEventListener('toggle', () => { localStorage.setItem(gk, gn.open ? '1' : '0'); render(); });
}

// ---------- Output slots (the Candidates grid, inside the Generate panel) ----------
function renderSlots() {
  const cand = S.candidates[activeTab()] || [];
  const pending = (S.busy && S.busy.tab === activeTab()) ? S.busy.total - S.busy.done : 0;
  $('#output .n').textContent = S.slots;
  const sheet = $('#sheet');
  sheet.innerHTML = '';
  for (let k = 0; k < S.slots; k++) {
    const id = cand[k];
    const d = document.createElement('div');
    d.className = 'slot drop'; d.dataset.target = 'slot'; d.dataset.index = k; d.tabIndex = 0;
    if (id != null) {
      d.appendChild(thumb(id));
      d.addEventListener('dblclick', () => act('place', {id, target: 'working'}));
    } else if (k < cand.length + pending) {
      d.innerHTML = '<div class="hint busy">generating…</div>';
    } else {
      d.innerHTML = '<div class="hint">empty</div>';
    }
    sheet.appendChild(d);
  }
  layoutSlots();
}

// ---------- Status + Session bar ----------
function statusBar() {
  $('#comfy-dot').className = 'dot ' + (S.comfy_ok ? 'ok' : 'bad');
  const p = $('#prog'), t = $('#progtxt');
  if (S.busy) {
    p.style.visibility = 'visible';
    p.max = S.busy.total; p.value = S.busy.done;
    t.textContent = S.busy.done + '/' + S.busy.total;
  } else { p.style.visibility = 'hidden'; t.textContent = ''; }
}

{ // controls|output divider inside the Generate panel (height-adjustable controls)
  const gp = $('#genpanel'), bar = $('#hsplit'), KEY = 'split:gen';
  const apply = px => { gp.style.gridTemplateRows = px + 'px 6px minmax(80px,1fr)'; if (S) layoutSlots(); };
  const saved = +localStorage.getItem(KEY);
  if (saved) requestAnimationFrame(() => apply(saved));
  bar.addEventListener('pointerdown', e => {
    e.preventDefault(); bar.setPointerCapture(e.pointerId); bar.classList.add('drag');
    const move = ev => apply(Math.max(110, Math.min(gp.clientHeight - 90, Math.round(ev.clientY - gp.getBoundingClientRect().top))));
    const up = () => { bar.classList.remove('drag'); bar.removeEventListener('pointermove', move); bar.removeEventListener('pointerup', up); const h = parseInt(gp.style.gridTemplateRows, 10); if (h) localStorage.setItem(KEY, h); };
    bar.addEventListener('pointermove', move); bar.addEventListener('pointerup', up);
  });
  bar.addEventListener('dblclick', () => { localStorage.removeItem(KEY); gp.style.gridTemplateRows = ''; if (S) layoutSlots(); });
}

document.addEventListener('keydown', e => {
  if (typing(e)) return;
  if (e.key === 'Escape') {
    if (!$('#prunedlg').hidden) { closePrune(); return; }
    if (infoHide()) return;
    if (gridOn) { closeGrid(); return; }
    Sel.clear();
  }
});

let sheetCols = 1;
// the two top carousels behave identically: click selects (blue), dbl-click /
// Enter picks, Space peeks, arrows move, thumbs drag, the end buttons scroll while held
function renderCarousel(name, ids) {
  fillStrip(name, ids.map(id => {
    const im = thumb(id);
    im.addEventListener('dblclick', () => act('place', {id, target: 'working'}));
    return im;
  }));
  const strip = $('#car-' + name + ' .strip');
  if (name === 'hist' && S.working !== lastCurWorking) {
    lastCurWorking = S.working;
    const cur = strip.querySelector('[data-id="' + S.working + '"]');
    if (cur && !fullyVisible(cur, strip)) cur.scrollIntoView({inline: 'center', block: 'nearest'});
  }
}
let lastCurWorking = null;
function fullyVisible(el, box) {
  const a = el.getBoundingClientRect(), b = box.getBoundingClientRect();
  return a.left >= b.left && a.right <= b.right;
}
function layoutSlots() {
  // one square cell size chosen so all N slots fit the Output pane
  const grid = $('#sheet');
  const W = grid.clientWidth - 20, H = grid.clientHeight - 4;
  let best = {cols: 1, size: 0};
  for (let cols = 1; cols <= S.slots; cols++) {
    const rows = Math.ceil(S.slots / cols);
    const size = Math.min((W - 8 * (cols - 1)) / cols, (H - 8 * (rows - 1)) / rows);
    if (size > best.size) best = {cols, size};
  }
  sheetCols = best.cols;
  const size = Math.max(64, Math.floor(best.size));
  grid.style.gridTemplateColumns = 'repeat(' + best.cols + ', ' + size + 'px)';
  grid.style.gridAutoRows = size + 'px';
}
window.addEventListener('resize', () => S && render());

// ---------- controls ----------
function readControls() {
  const t = activeTab();
  const sEl = $('#seed_' + t);
  return {prompt: $('#prompt').value, negative: $('#negative').value, family: $('#family').value,
    steps: +$('#steps').value || 0, cfg: +$('#cfg').value || 0,
    ref0: S ? S.controls.ref0 : null,
    refs: S ? S.controls.refs : [null, null, null],
    lora: $('#lora').value, lora_strength: +$('#lstr').value, lock: $('#lock').value,
    vary: +$('#vary').value, whitebg: $('#whitebg').checked,
    width: +$('#width').value || 1024, height: +$('#height').value || 1024,
    fresh_model: $('#fresh').checked,
    tab: t,
    seed: sEl ? (+sEl.value || 0) : 0,           // the ACTIVE tab's seed
    seed_create: +$('#seed_create').value || 0,
    seed_derive: +$('#seed_derive').value || 0,
    seed_camera: +$('#seed_camera').value || 0,
    outputs_create: +$('#outputs_create').value || 6,
    outputs_derive: +$('#outputs_derive').value || 6,
    outputs_camera: +$('#outputs_camera').value || 4,
    pov_azim: camVal('azim'), pov_elev: camVal('elev'), pov_dist: camVal('dist')};
}

// ---- draggable divider between the preview image and the board/sheet ----
// The grid is `<preview> 6px <sheet>`; dragging rewrites the first track and
// localStorage keeps it. layoutSlots() sizes cells from clientWidth, so the
// sheet has to be re-laid out as the drag moves, not just at the end.
(function () {
  const work = $('#work'), bar = $('#split');
  const KEY = 'split:work';
  const apply = px => {
    work.style.gridTemplateColumns = `${px}px 6px minmax(220px,1fr)`;
    if (S) layoutSlots();
  };
  const saved = +localStorage.getItem(KEY);
  if (saved) requestAnimationFrame(() => apply(saved));
  bar.addEventListener('pointerdown', e => {
    e.preventDefault();
    bar.setPointerCapture(e.pointerId);
    bar.classList.add('drag');
    const move = ev => {
      const px = Math.round(ev.clientX - work.getBoundingClientRect().left);
      apply(Math.max(220, Math.min(work.clientWidth - 226, px)));
    };
    const up = () => {
      bar.classList.remove('drag');
      bar.removeEventListener('pointermove', move);
      bar.removeEventListener('pointerup', up);
      const w = parseInt(work.style.gridTemplateColumns, 10);
      if (w) localStorage.setItem(KEY, w);
    };
    bar.addEventListener('pointermove', move);
    bar.addEventListener('pointerup', up);
  });
  bar.addEventListener('dblclick', () => { localStorage.removeItem(KEY); work.style.gridTemplateColumns = ''; if (S) layoutSlots(); });
})();
let saveT = null;
function saveControls() { clearTimeout(saveT); saveT = setTimeout(() => api('controls', readControls()), 400); }
['#prompt', '#negative', '#family', '#steps', '#cfg', '#lora', '#lstr', '#lock', '#vary', '#width', '#height', '#whitebg', '#fresh',
 '#seed_create', '#seed_derive', '#seed_camera', '#outputs_create', '#outputs_derive', '#outputs_camera']
  .forEach(s => $(s).addEventListener('input', () => { $('#varyv').textContent = $('#vary').value; saveControls(); }));
// dbl-click the Working Image = copy it to ref0 (works from any tab;
// the same w/h sync as the Space-click gesture)
$('#stagebox').addEventListener('dblclick', async () => {
  if (!S || S.working == null) return;
  await api('place', {id: S.working, target: 'ref0'});
  const m = S.meta[S.working];
  if (m) { $('#width').value = m.w; $('#height').value = m.h; saveControls(); }
  flash(`ref0 ← #${S.working}`);
  refresh();
});

// ONE rule: click on ANY image while Space is held = copy it to ref0
// (shortcut for dragging it into the ref0 slot). Plain clicks only select.
document.addEventListener('click', async e => {
  if (!spaceHeld || !S || !e.target.closest('[data-target]')) return;
  const id = Sel.id();          // mousedown just focused the clicked box
  if (id == null) return;
  await api('place', {id, target: 'ref0'});
  const m = S.meta[id];
  if (m) { $('#width').value = m.w; $('#height').value = m.h; saveControls(); }
  flash(`ref0 ← #${id}`);
  refresh();
});

// hidden setting (settings page later): selecting an image anywhere scrolls
// History to it, if it is there. localStorage sync_history_to_selection=0 off.
let lastHistSync = null;
function syncHistory() {
  // selecting an image outside History mirrors the blue border onto its
  // History twin; the strip scrolls (centered) ONLY when the selection just
  // changed AND the twin is not fully visible - manual scrolls stay put
  if (!S || Sel.target === 'hist') { lastHistSync = Sel.id(); return; }
  const id = Sel.id();
  const k = id == null ? -1 : S.history.indexOf(id);
  if (k < 0) { lastHistSync = id; return; }
  const im = document.querySelector(`img[data-target="hist"][data-index="${k}"]`);
  if (!im) { lastHistSync = id; return; }
  const changed = id !== lastHistSync;
  lastHistSync = id;
  if (localStorage.getItem('sync_history_to_selection') === '0') return;
  if (changed && !fullyVisible(im, im.closest('.strip')))
    im.scrollIntoView({inline: 'center', block: 'nearest'});
}
// family is a fiat-only choice: live while ref0 + ref slots are empty
// (references exist only in the Klein graph; the stage may hold anything)
function familyUI() {
  // v3: the TAB is the fiat gate - the dropdown is always live on Create,
  // and Derive is Klein by construction (no dropdown there at all)
  if (!S) return;
  const fam = activeTab() === 'derive' ? 'klein' : ($('#family').value || 'klein');
  const f = S.families[fam] || S.families.klein;
  $('#steps').placeholder = f.steps; $('#cfg').placeholder = f.cfg;
  // Klein ignores its -ive prompt: only models that read it show the box
  $('#negative').style.display = (activeTab() === 'create' && fam !== 'klein') ? '' : 'none';
  // white bg is a Flux-only mechanism (prompt prefix): off-screen otherwise.
  // The eventual solution is the user-editable boilerplate-vars table.
  $('#whitebg').parentElement.style.display = fam === 'klein' ? '' : 'none';
  const noLora = activeTab() === 'create' && fam === 'illustrious';
  $('#lora').parentElement.classList.toggle('off', noLora);
  $('#lstr').parentElement.classList.toggle('off', noLora);
}
$('#family').addEventListener('change', () => { $('#steps').value = ''; $('#cfg').value = ''; familyUI(); saveControls(); });
['create', 'derive', 'camera'].forEach(t => $('#outputs_' + t).addEventListener('change', () => {
  if (activeTab() === t) act('slots', {slots: +$('#outputs_' + t).value || 1});
}));
$('#gen').addEventListener('click', async () => {
  if (S && S.busy) {          // morphed: Stop
    await api('abort', {});
    refresh();
    return;
  }
  const t = activeTab();
  let r;
  if (t === 'camera') {
    r = await api('pov', {azim: camVal('azim'), elev: camVal('elev'), dist: camVal('dist'),
      outputs: +$('#outputs_camera').value || 4, seed: +$('#seed_camera').value || 0});
  } else {
    r = await api('generate', {...readControls(), op: t,
      outputs: +$('#outputs_' + t).value || 6});
  }
  if (r.error) alert(r.error);
  refresh();
});
$('#projsel').addEventListener('change', () => act('project', {name: $('#projsel').value}));
$('#projnew').addEventListener('click', () => {
  const name = prompt('new project name (letters, digits, - _):');
  if (name) act('project', {name: name.trim()});
});
$('#gc').addEventListener('click', async () => {
  if (!confirm('Archive every image that is not pinned, not in the live working set (WI / refs / candidates), and not an ancestor of those? Files move to <store>/archive/.')) return;
  const r = await api('gc', {}); flash(`archived ${r.removed}, kept ${r.kept}`); refresh();
});
// ---------- the carousel component ----------
// Every .car gets: sticky open/close, single-image scroll steps that
// accelerate to fast scroll after [GS carousel_step] steps (0 = fast
// immediately), and a +/- toggling between exactly two thumb sizes.
// New carousels inherit ALL of it from the markup pattern alone.
const GS = {carousel_step: +(localStorage.getItem('GS:carousel_step') ?? 3)};
document.querySelectorAll('.car .arrow').forEach(btn => {
  const strip = btn.parentElement.querySelector('.strip'), dir = +btn.dataset.dir;
  let t = null, n = 0;
  const w = () => { const im = strip.querySelector('[data-id]'); return (im ? im.getBoundingClientRect().width : 64) + 6; };
  const step = () => strip.scrollBy({left: dir * w(), behavior: 'smooth'});
  const fast = () => { clearInterval(t); t = setInterval(() => { strip.scrollLeft += dir * 14; }, 16); };
  btn.addEventListener('mousedown', () => {
    clearInterval(t); n = 0;
    if (!GS.carousel_step) { fast(); return; }
    step(); n = 1;
    t = setInterval(() => { if (n++ < GS.carousel_step) step(); else fast(); }, 300);
  });
  ['mouseup', 'mouseleave'].forEach(ev => btn.addEventListener(ev, () => clearInterval(t)));
});
document.querySelectorAll('details.car').forEach(c => {
  const k = 'open:' + c.id;
  const saved = localStorage.getItem(k);
  if (saved != null) c.open = saved === '1';
  c.addEventListener('toggle', () => {
    if (c._forced !== undefined && c.open === c._forced) { c._forced = undefined; return; }
    localStorage.setItem(k, c.open ? '1' : '0');
    render();
  });
  const sm = c.querySelector('summary');
  const b = document.createElement('button');
  b.className = 'sz'; b.title = 'toggle thumbnail size';
  const kk = 'size:' + c.id;
  const setSz = () => { const big = localStorage.getItem(kk) === '1'; c.classList.toggle('big', big); b.textContent = big ? '-' : '+'; };
  b.addEventListener('click', e => {
    e.preventDefault(); e.stopPropagation();
    localStorage.setItem(kk, localStorage.getItem(kk) === '1' ? '0' : '1');
    setSz(); render();
  });
  const gb = document.createElement('button');
  gb.className = 'sz';
  gb.title = 'show as a full-workspace grid';
  gb.innerHTML = icon(I.grid, 13);
  gb.addEventListener('click', e => {
    e.preventDefault(); e.stopPropagation();
    openGrid(c.id.replace('car-', ''));
  });
  // PROXIMITY: the controls sit right after the title+count cluster, never
  // floated across an ultrawide row (association problem, user-reported)
  const nEl = sm.querySelector('.n');
  if (nEl) nEl.after(b, gb); else sm.append(b, gb);
  setSz();
});
$('#gridclose').innerHTML = icon(I.close, 20);
$('#gridclose').addEventListener('click', closeGrid);
$('#gridsz').addEventListener('click', () => {
  localStorage.setItem('size:grid', localStorage.getItem('size:grid') === '1' ? '0' : '1');
  renderGrid();
});
function stepHistory(d) {
  if (!S || !S.history.length) return;
  let i = S.history.indexOf(S.working); i = i < 0 ? S.history.length - 1 : Math.max(0, Math.min(S.history.length - 1, i + d));
  act('place', {id: S.history[i], target: 'working'});
}

// ---------- focus + keyboard ----------
let spaceHeld = false;
function slotPinned(t) {   // board cells never take a replacing drop/paste
  return S && t.target === 'pin' && t.index < S.pins.length;
}
document.addEventListener('mousedown', e => {
  const d = e.target.closest('[data-target]'); if (!d) return;
  Sel.set(d.dataset.target, d.dataset.index);
});
const typing = e => ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName);
document.addEventListener('keydown', async e => {
  if (typing(e)) return;
  const id = Sel.id();
  if (gridOn) {
    if (e.key === ' ') {
      e.preventDefault();
      if (!e.repeat) gridPeek(true);          // hold = full-size preview
      return;
    }
    if (!e.key.startsWith('Arrow')) return;   // modal: arrows browse, Esc closes
    queueMicrotask(() => gridPeek());         // preview follows the selection while held
  }
  if (e.key === ' ') {                       // hold = the stage shows the selection; on the
    e.preventDefault();                      // preview image itself: shows its PARENT (provenance)
    if (!e.repeat) { spaceHeld = true; Sel.apply(); }
  } else if (e.key === 'Delete' || e.key === 'Backspace') {
    if (e.shiftKey && id != null) {   // PRUNE: a collectable leaf just goes; anything with impact asks
      e.preventDefault();
      const plan = await api('prune', {id, force: false});
      if (plan.error) { notice(plan.error); return; }
      if (plan.branch === 1 && plan.archive.length === 1) {
        await api('prune', {id, force: false, apply: true});
        refresh();
      } else openPrune(id, plan);
      return;
    }
    if (Sel.target === 'asset') {   // remove the ENTRY; the image is untouched
      const a = curAssetObj();
      const en = a && a.dataset[Sel.index];
      if (en) { e.preventDefault(); act('asset', {op: 'remove', name: a.name, path: en.path}); }
      return;
    }
    if (id != null && RW.includes(Sel.target)) {   // r/o sheets (history, genealogy) can't be edited
      e.preventDefault();
      await api('clear', {target: Sel.target, index: Sel.index}); refresh();
    }
  } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'c') {
    if (id != null) { e.preventDefault(); copyImage(id); }
  } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'x') {
    if (id != null) { e.preventDefault(); await copyImage(id); if (RW.includes(Sel.target)) { await api('clear', {target: Sel.target, index: Sel.index}); refresh(); } }
  } else if (e.key === 'Enter') {
    if (id != null && Sel.target !== 'working') {
      act('place', {id, target: 'working'});
    }
  } else if (e.key.startsWith('Arrow')) {
    const d = {ArrowLeft: -1, ArrowRight: 1, ArrowUp: -sheetCols, ArrowDown: sheetCols}[e.key];
    if (Sel.target === 'slot') { e.preventDefault(); Sel.set('slot', Math.max(0, Math.min(S.slots - 1, Sel.index + d))); }
    else if (Sel.target === 'pin') { e.preventDefault(); Sel.set('pin', Math.max(0, Math.min(S.pins.length - 1, Sel.index + Math.sign(d)))); }
    else if (Sel.target === 'working' && Math.abs(d) === 1) stepHistory(d);
    else if (Sel.target === 'ref' && Math.abs(d) === 1) { Sel.set('ref', Math.max(0, Math.min(2, Sel.index + d))); }
    else if (Sel.target === 'grid' && listFor('grid')) {   // a GRID: up/down move by a row
      e.preventDefault();
      const cols = getComputedStyle($('#gridbody')).gridTemplateColumns.split(' ').length || 1;
      const step = {ArrowLeft: -1, ArrowRight: 1, ArrowUp: -cols, ArrowDown: cols}[e.key];
      Sel.set('grid', Math.max(0, Math.min(listFor('grid').length - 1, Sel.index + step)));
    }
    else if (listFor(Sel.target) && Math.abs(d) === 1) { e.preventDefault(); Sel.set(Sel.target, Math.max(0, Math.min(listFor(Sel.target).length - 1, Sel.index + d))); }
  } else if (e.key === 'p' && id != null) {
    act('pin', {id, on: true});   // P pins the selection from ANYWHERE; Del inside Pinned unpins
  }
});

document.addEventListener('keyup', e => { if (e.key === ' ') { spaceHeld = false; peek(null); gridPeekEnd(); } });
window.addEventListener('blur', () => { spaceHeld = false; peek(null); gridPeekEnd(); });
let peekId = null;
function peek(id) {
  peekId = id;
  let ov = $('#peek');
  if (!ov) { ov = document.createElement('img'); ov.id = 'peek'; $('#stagebox').appendChild(ov); }
  if (id == null) { ov.style.display = 'none'; ov.removeAttribute('src'); return; }
  ov.src = imgURL(id); ov.style.display = 'block';
}

// ---------- Info Window: right-click any Image ----------
const SVG_COPY = icon(I.copy, 16);
const SVG_TICK = icon(I.done, 16);
let infoEl = null;
function infoHide() { if (infoEl && !infoEl.hidden) { infoEl.hidden = true; return true; } return false; }
async function showInfo(id, x, y) {
  // id = number (current project) or {path} (any project's image)
  const q = typeof id === 'object' ? id : {id};
  const r0 = await api('meta', q);         // always: the gc verdict is computed live
  if (r0.error) { notice(r0.error); return; }
  const m = r0;
  if (!infoEl) {
    infoEl = document.createElement('div');
    infoEl.id = 'infowin';
    document.body.appendChild(infoEl);
    document.addEventListener('mousedown', e => { if (!infoEl.hidden && !infoEl.contains(e.target)) infoHide(); });
  }
  infoEl.innerHTML = '';
  const path = m.path || '';
  const cut = Math.max(path.lastIndexOf('/'), path.lastIndexOf(String.fromCharCode(92)));
  const dir = cut < 0 ? '' : path.slice(0, cut + 1);
  const rows = [['file', path],
    ['size', (m.w && m.h) ? m.w + ' × ' + m.h : null, true],   // true = no copy button
    ['created', m.ts],
    ['gc', m.gc, true],
    ['refs', (m.parents || []).map(q => dir + q + '.png').join(String.fromCharCode(10)) || null],
    ['prompt', m.recipe && m.recipe.prompt]];
  for (const [k, v, nocopy] of rows) {
    if (v == null || v === '') continue;
    const row = document.createElement('div'); row.className = 'iw-row';
    const kk = document.createElement('span'); kk.className = 'iw-k'; kk.textContent = k;
    const vv = document.createElement('span'); vv.className = 'iw-v'; vv.textContent = v;
    if (nocopy) { row.append(kk, vv); infoEl.appendChild(row); continue; }
    const c = document.createElement('button'); c.className = 'iw-c'; c.innerHTML = SVG_COPY; c.title = 'copy to clipboard';
    c.addEventListener('click', () => {
      navigator.clipboard.writeText(String(v));
      c.innerHTML = SVG_TICK; c.title = 'Copied!'; c.classList.add('done');
      setTimeout(() => { c.innerHTML = SVG_COPY; c.title = 'copy to clipboard'; c.classList.remove('done'); }, 1200);
    });
    row.append(kk, vv, c);
    infoEl.appendChild(row);
  }
  infoEl.hidden = false;
  const r = infoEl.getBoundingClientRect();
  infoEl.style.left = Math.min(x, innerWidth - r.width - 12) + 'px';
  infoEl.style.top = Math.min(y, innerHeight - r.height - 12) + 'px';
}
document.addEventListener('contextmenu', e => {
  const d = e.target.closest('[data-id]');
  if (d) { e.preventDefault(); showInfo(+d.dataset.id, e.clientX, e.clientY); return; }
  const pth = e.target.closest('[data-path]');
  if (pth) { e.preventDefault(); showInfo({path: pth.dataset.path}, e.clientX, e.clientY); }
});

// ---------- clipboard ----------
async function copyImage(id) {
  // PNG bitmap for Paint/Photoshop/browsers + the file path as text for editors
  try {
    const blob = await (await fetch(imgURL(id))).blob();
    const item = {'image/png': blob};
    try { item['text/plain'] = new Blob([S.meta[id].path], {type: 'text/plain'}); } catch (_) {}
    await navigator.clipboard.write([new ClipboardItem(item)]);
    flash(`copied #${id} to clipboard`);
  } catch (err) {
    // some browsers refuse multi-type items: fall back to image only
    try {
      const blob = await (await fetch(imgURL(id))).blob();
      await navigator.clipboard.write([new ClipboardItem({'image/png': blob})]);
      flash(`copied #${id} (image only)`);
    } catch (e2) { flash('copy failed: ' + e2); }
  }
}
document.addEventListener('paste', async e => {
  if (typing(e) && e.target.id !== 'prompt') return;
  const items = [...(e.clipboardData?.items || [])];
  const img = items.find(it => it.type.startsWith('image/'));
  if (img && (Sel.target === 'agrid' || Sel.target === 'asset')) {
    e.preventDefault();
    const a = curAssetObj();
    if (!a) { flash('create an asset first'); return; }
    const r = await fetch('/api/import', {method: 'POST', body: img.getAsFile()}).then(x => x.json());
    if (r.error) { flash(r.error); return; }
    await assetAddId(a, r.id);
    flash(`pasted into "${a.name}"`);
    refresh();
    return;
  }
  if (img) { e.preventDefault(); await importBlob(img.getAsFile(), slotPinned(Sel.get()) ? {target: 'pin', index: 999} : selTarget()); return; }
  const txt = e.clipboardData?.getData('text') || '';
  if (/^https?:\/\/\S+$/.test(txt.trim()) && !typing(e)) { e.preventDefault(); await importURL(txt.trim(), selTarget()); }
});
function flash(msg) {
  const el = $('#msg');
  el.classList.remove('notice');
  el.textContent = msg;
  setTimeout(() => { if (el.textContent === msg && !el.classList.contains('notice')) el.textContent = ''; }, 3000);
}
function toast(msg, kind) {
  // dismissable, stays ~8s: long enough to digest a multi-part message
  const t = document.createElement('div');
  t.className = 'toast' + (kind ? ' ' + kind : '');
  t.textContent = msg;
  t.addEventListener('click', () => t.remove());
  $('#toasts').appendChild(t);
  setTimeout(() => t.remove(), 8000);
}
function notice(msg) {
  // a REFUSAL deserves to be seen: toast + the bar flashes
  toast(msg, 'warn');
  const bar = $('#bottom');
  bar.classList.remove('alert'); void bar.offsetWidth;   // restart the animation
  bar.classList.add('alert');
}

// ---------- drag & drop ----------
function dragStart(e, id) {
  const url = imgURL(id);
  e.dataTransfer.effectAllowed = 'all';
  e.dataTransfer.setData('application/x-evolver', String(id));
  e.dataTransfer.setData('text/uri-list', url);
  e.dataTransfer.setData('text/plain', S.meta[id] ? S.meta[id].path : url);
  e.dataTransfer.setData('DownloadURL', `image/png:${id}.png:${url}`);   // Chromium: drag out as a file
}
// never show the no-entry cursor, never let a missed drop navigate the page
['dragenter', 'dragover'].forEach(t => document.addEventListener(t, e => {
  e.preventDefault();
  let d = e.target.closest('.drop');
  if (d && d.dataset.target === 'pin' && +d.dataset.index < 900) d = d.closest('#car-pin');
  e.dataTransfer.dropEffect = d ? 'copy' : 'none';
  document.querySelectorAll('.drop.over').forEach(x => x !== d && x.classList.remove('over'));
  if (d) d.classList.add('over');
}));
document.addEventListener('dragleave', e => { if (!e.relatedTarget) document.querySelectorAll('.drop.over').forEach(x => x.classList.remove('over')); });
document.addEventListener('drop', async e => {
  e.preventDefault();
  document.querySelectorAll('.drop.over').forEach(x => x.classList.remove('over'));
  if (e.target.closest('#assets')) { await assetDrop(e.dataTransfer); return; }
  let d = e.target.closest('.drop'); if (!d) return;
  if (d.dataset.target === 'pin' && +d.dataset.index < 900) d = d.closest('#car-pin');
  const target = {target: d.dataset.target, index: +(d.dataset.index || 0)};
  if (target.target !== 'pin') Sel.set(target.target, target.index);
  const dt = e.dataTransfer;
  const own = dt.getData('application/x-evolver');
  if (own) { const r = await api('place', {id: +own, target: target.target, index: target.index}); if (r.error) flash(r.error); refresh(); return; }
  const files = [...(dt.files || [])].filter(f => f.type.startsWith('image/') || /\.(png|jpe?g|webp|gif|bmp)$/i.test(f.name));
  if (files.length) {
    // first file to the drop target; extra files flow into following slots
    await importBlob(files[0], target);
    for (let k = 1; k < files.length && target.target === 'slot'; k++) await importBlob(files[k], {target: 'slot', index: target.index + k});
    return;
  }
  const uri = (dt.getData('text/uri-list') || dt.getData('text/plain') || '').split('\n').map(s => s.trim()).find(s => s && !s.startsWith('#'));
  if (uri) {
    if (uri.startsWith('data:')) { await importBlob(await (await fetch(uri)).blob(), target); return; }
    const html = dt.getData('text/html');
    const m = html && html.match(/<img[^>]+src="([^"]+)"/i);
    await importURL(m ? m[1].replace(/&amp;/g, '&') : uri, target);
  }
});
async function importBlob(blob, target) {
  const r = await fetch('/api/import', {method: 'POST', body: blob}).then(r => r.json());
  if (r.error) { flash(r.error); return; }
  await api('place', {id: r.id, target: target.target, index: target.index});
  refresh();
}
async function importURL(url, target) {
  if (url.startsWith('data:')) return importBlob(await (await fetch(url)).blob(), target);
  if (url.startsWith(location.origin + '/img/')) return api('place', {id: +url.split('/').pop(), ...target}).then(refresh);
  flash('fetching ' + url.slice(0, 60) + '…');
  const r = await api('import_url', {url});
  if (r.error) { flash(r.error); return; }
  await api('place', {id: r.id, target: target.target, index: target.index});
  refresh();
}

setMode(mode);
refresh();
</script>
</body></html>
"""


def main():
    global STORE, ROOT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".",
                    help="the GLOBAL root (v4): projects are subdirs, assets "
                         "and app config live beside them. Absolute, relative "
                         "to cwd, or omitted = cwd itself")
    ap.add_argument("--project", default=None,
                    help="project to open (default: config.json last_project, "
                         "else the first existing project, else 'default')")
    ap.add_argument("--port", type=int, default=8189)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--embed-workflow", action="store_true",
                    help="also embed UI geometry (a `workflow` chunk) in "
                         "output pngs so they drag into the ComfyUI "
                         "frontend editable. Off by default: it is the "
                         "bulk of the file. (Settings page later.)")
    args = ap.parse_args()
    global EMBED_WORKFLOW
    EMBED_WORKFLOW = args.embed_workflow

    ROOT = Path(args.root).expanduser().resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    name = args.project or load_config().get("last_project")
    if not name or not (ROOT / name).is_dir():
        existing = list_projects()
        name = existing[0] if existing else "default"
    STORE = open_project(name)
    print(f"root: {ROOT}  project: {name}")
    app = web.Application(client_max_size=256 * 1024 ** 2)
    app.router.add_get("/", index)
    app.router.add_get("/api/state", api_state)
    app.router.add_post("/api/generate", api_generate)
    app.router.add_post("/api/pov", api_pov)
    app.router.add_post("/api/family", api_family)
    app.router.add_post("/api/meta", api_meta)
    app.router.add_post("/api/discard", api_discard)
    app.router.add_post("/api/prune", api_prune)
    app.router.add_post("/api/abort", api_abort)
    app.router.add_post("/api/project", api_project)
    app.router.add_post("/api/asset", api_asset)
    app.router.add_get("/file/{path:.+}", serve_file)
    app.router.add_post("/api/import_folder", api_import_folder)
    app.router.add_post("/api/train", api_train)
    app.router.add_post("/api/train_abort", api_train_abort)
    app.router.add_post("/api/controls", api_controls)
    app.router.add_post("/api/place", api_place)
    app.router.add_post("/api/clear", api_clear)
    app.router.add_post("/api/pin", api_pin)
    app.router.add_post("/api/slots", api_slots)
    app.router.add_post("/api/import", api_import)
    app.router.add_post("/api/import_url", api_import_url)
    app.router.add_post("/api/gc", api_gc)
    app.router.add_get("/img/{id:\\d+}", serve_image)
    app.router.add_get("/img/{project}/{id:\\d+}", serve_image)

    print(f"evolver store: {STORE.root}  ({sum(1 for i in STORE.images if STORE.alive(i))} images)")
    print(f"open:          http://{args.host}:{args.port}/")
    print("(ComfyUI must be running at 127.0.0.1:8188 before you generate)")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
