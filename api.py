"""api.py - the HTTP surface the frontend talks to (aiohttp).

Handlers validate + translate; the work is done by the operators
(generate.do_generate, camera.do_camera, training.do_training) and the
data layer (store, trash, asset, lineage). One App context per process:
the open store, the job runner, the settings.

    GET  /                       frontend/index.html (live from disk, no-store)
    GET  /static/{path}          frontend assets
    GET  /img/{project}/{id}     a stored image
    GET  /file/{path}            any image under the root (asset browser)
    GET  /api/state              the full UI snapshot (polled)
    POST /api/<name>             everything else (see ROUTES)
"""
import asyncio
import mimetypes
import re
import urllib.request
from pathlib import Path

from aiohttp import web

import asset as asset_mod
import camera
import comfy_client
import lineage
import trash
from controls import persistable, restore_from_image, sanitize_controls
from generate import FAMILIES, GenerateConfig, do_generate
from image_file import IMAGE_EXTS, list_images, sha1_of
from jobs import Jobs
from lora import list_loras
from project import (RESERVED, is_valid_name, list_projects, root, root_rel)
from store import open_project
from training import (TrainConfig, abort_training, do_training, log_path,
                      sync_dataset)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


class App:
    """Process-wide context (singleton contract: one server, one root, one
    open project)."""

    def __init__(self, store, embed_workflow=False):
        self.store = store
        self.jobs = Jobs()
        self.embed_workflow = embed_workflow

    def snapshot(self):
        s = self.store
        snap = s.snapshot()
        snap.update({
            "projects": list_projects(),
            "assets": asset_mod.load_assets(),
            "train": self.jobs.train_status(),
            "comfy_ok": comfy_client.is_alive(busy=bool(self.jobs.busy)),
            "busy": self.jobs.busy,
            "families": {k: {"label": v["label"], "steps": v["steps"], "cfg": v["cfg"]}
                         for k, v in FAMILIES.items()},
            "pov_azim": camera.AXIS_AZIMUTH, "pov_elev": camera.AXIS_ELEVATION,
            "pov_dist": camera.AXIS_DISTANCE,
            "loras": list_loras()})
        return snap


def ctx(request):
    return request.app["ctx"]


def ok(**kw):
    return web.json_response(kw)


def error(msg, status=400):
    return web.json_response({"error": msg}, status=status)


# ---------- pages & files ----------

async def index(request):
    """The UI, fresh from disk on every request (edits appear on refresh;
    no-store kills Edge's HTML caching)."""
    return web.FileResponse(FRONTEND_DIR / "index.html",
                            headers={"Cache-Control": "no-store"})


async def serve_static(request):
    rel = request.match_info["path"]
    f = (FRONTEND_DIR / rel).resolve()
    try:
        f.relative_to(FRONTEND_DIR)
    except ValueError:
        raise web.HTTPNotFound()
    if not f.is_file():
        raise web.HTTPNotFound()
    ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
    return web.FileResponse(f, headers={"Cache-Control": "no-store",
                                        "Content-Type": ctype})


async def serve_image(request):
    """/img/<project>/<id>: project-qualified, because /img/<id> alone is
    ambiguous across projects (browser cache served a stale image after a
    switch, user-caught). File existence is the truth (archived = 404)."""
    i = int(request.match_info["id"])
    proj = request.match_info.get("project")
    if proj:
        if not is_valid_name(proj):
            raise web.HTTPNotFound()
        f = root() / proj / "images" / f"{i}.png"
    else:                       # legacy unqualified URL: current project
        f = ctx(request).store.path(i)
    if not f.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(f, headers={
        "Content-Disposition": f'inline; filename="{i}.png"',
        "Cache-Control": "no-cache"})


async def serve_file(request):
    """Any image under the root by root-relative path (the asset browser
    shows images from ANY project). Strictly contained."""
    try:
        f = (root() / request.match_info["path"]).resolve()
        f.relative_to(root())
    except Exception:
        raise web.HTTPNotFound()
    if not (f.is_file() and f.suffix.lower() in IMAGE_EXTS):
        raise web.HTTPNotFound()
    return web.FileResponse(f, headers={"Cache-Control": "no-cache"})


# ---------- state & controls ----------

async def handle_state(request):
    return web.json_response(ctx(request).snapshot())


async def handle_controls(request):
    """Persist control edits (so a reload keeps your prompt etc.)."""
    store = ctx(request).store
    edits = await request.json()
    with store.lock:
        store.state["controls"].update(persistable(edits))
        store.save_state()
    return ok()


async def handle_place(request):
    store = ctx(request).store
    b = await request.json()
    i, target, index = b.get("id"), b["target"], int(b.get("index", 0))
    if not store.alive(i):
        return error(f"no such image {i}", 404)
    with store.lock:
        if target == "working":
            store.set_working(i)
            restore_from_image(store, i)
            store.save_state()
        elif target == "ref":
            store.state["controls"]["refs"][index] = i
            store.save_state()
        elif target == "ref0":
            store.state["controls"]["ref0"] = i
            store.save_state()
        elif target == "slot":
            store.place_slot(i, index)
        elif target == "pin":
            store.pin(i, True, index)
        else:
            return error("bad target")
    return ok()


async def handle_clear(request):
    store = ctx(request).store
    b = await request.json()
    target, index = b["target"], int(b.get("index", 0))
    with store.lock:
        if target == "working":
            store.set_working(None)
        elif target == "ref":
            store.state["controls"]["refs"][index] = None
        elif target == "ref0":
            store.state["controls"]["ref0"] = None
        elif target == "slot":
            c = store.cands()
            if index < len(c):
                c.pop(index)
        elif target == "pin":
            if index < len(store.state["pins"]):
                store.state["pins"].pop(index)
        store.save_state()
    return ok()


async def handle_pin(request):
    b = await request.json()
    ctx(request).store.pin(b["id"], bool(b["on"]))
    return ok()


async def handle_slots(request):
    b = await request.json()
    ctx(request).store.set_slots(b["slots"])
    return ok()


# ---------- generators ----------

async def handle_generate(request):
    app = ctx(request)
    store = app.store
    why = app.jobs.idle_error()
    if why:
        return error(why, 409)
    given = await request.json()
    op = given.get("op") or "derive"
    if op not in ("create", "derive"):
        op = "derive"
    c = sanitize_controls(given, store.alive)
    if op == "create":               # fiat: the tab IS the no-refs gate
        c["ref0"] = None
        c["refs"] = [None, None, None]
    else:                            # derive: Klein-only, per spec
        c["family"] = "klein"
    cfg = GenerateConfig.from_controls(c, op)
    with store.lock:
        store.state["controls"] = c
        store.state["slots"] = max(1, min(64, int(given.get("outputs") or store.state["slots"])))
        store.save_state()
        total = store.state["slots"]
    tab = "create" if op == "create" else "derive"
    app.jobs.start_round(tab, total, lambda progress, should_abort: do_generate(
        store, cfg, progress, should_abort, app.embed_workflow), label="generate")
    return ok(started=True)


async def handle_pov(request):
    """Camera generator. Same busy lock as Generate."""
    app = ctx(request)
    store = app.store
    why = app.jobs.idle_error()
    if why:
        return error(why, 409)
    q = await request.json()
    az, el, di = q.get("azim") or None, q.get("elev") or None, q.get("dist") or None
    if not camera.valid_axes(az, el, di):
        return error("bad camera token")
    if not (az or el or di):
        return error("no camera axis selected")
    if not store.alive(store.state["working"]):
        return error("no working image to re-shoot")   # budding needs a parent
    cfg = camera.CameraConfig(store.state["working"], az, el, di, int(q.get("seed") or 0))
    with store.lock:
        store.state["slots"] = max(1, min(64, int(q.get("outputs") or store.state["slots"])))
        store.save_state()
        total = store.state["slots"]
    app.jobs.start_round("camera", total, lambda progress, should_abort: camera.do_camera(
        store, cfg, progress, should_abort, app.embed_workflow), label="camera")
    return ok(started=True)


async def handle_abort(request):
    """Stop the current round NOW: no further candidates are queued and the
    in-flight ComfyUI job is interrupted. Finished candidates stay."""
    if not ctx(request).jobs.request_abort():
        return ok(aborted=False)
    try:
        comfy_client.interrupt()
    except Exception as e:
        print(f"interrupt failed: {type(e).__name__}: {e}")
    return ok(aborted=True)


# ---------- lineage & trash ----------

async def handle_family(request):
    q = await request.json()
    fam = lineage.family(ctx(request).store, q.get("id"))
    if fam is None:
        return error(f"unknown image {q.get('id')}", 404)
    return web.json_response(fam)


async def handle_meta(request):
    """Full metadata for ONE image (Info Window): by id in the current
    project, or by root-relative PATH for any project's image."""
    store = ctx(request).store
    q = await request.json()
    i = q.get("id")
    if i is None and q.get("path"):
        try:
            rel = root_rel(q["path"])
        except Exception:
            return error("path outside the root")
        if rel.split("/")[0] == store.name:
            try:
                i = int(rel.split("/")[-1][:-4])
            except ValueError:
                return error("bad path")
        else:
            m = lineage.foreign_meta(rel)
            if m is None:
                return error(f"no record for {rel}", 404)
            return web.json_response(m)
    with store.lock:
        if i not in store.images:
            return error(f"unknown image {i}", 404)
        m = store.meta(i)
        m["gc"] = trash.verdict(store, i)
        return web.json_response(m)


async def handle_discard(request):
    b = await request.json()
    why = trash.discard(ctx(request).store, b.get("id"))
    if why:
        return error(f"kept: {why}", 409)
    return ok()


async def handle_prune(request):
    """{id, force, apply}: apply=false returns the impact plan for the
    dialog; apply=true executes it (recomputed server-side)."""
    store = ctx(request).store
    b = await request.json()
    i, force = b.get("id"), bool(b.get("force"))
    if not store.alive(i):
        return error(f"no such image {i}", 404)
    plan = trash.prune_apply(store, i, force) if b.get("apply") else trash.prune_plan(store, i, force)
    return web.json_response(plan)


async def handle_gc(request):
    return web.json_response(trash.gc(ctx(request).store))


# ---------- import ----------

async def handle_import(request):
    """Raw image bytes (drop/paste/file) -> stored image id."""
    data = await request.read()
    if not data:
        return error("empty upload")
    try:
        i, _ = ctx(request).store.import_bytes(data)
    except Exception as e:
        return error(f"not a readable image: {e}")
    return ok(id=i)


async def handle_import_url(request):
    """A URL dropped from another browser: fetched server-side (no CORS)."""
    b = await request.json()
    url = b["url"]
    if not re.match(r"^https?://", url):
        return error("only http(s) urls")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 evolve"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read(64 * 1024 ** 2)
        i, _ = ctx(request).store.import_bytes(data)
    except Exception as e:
        return error(f"fetch failed: {e}")
    return ok(id=i)


async def handle_import_folder(request):
    """Bulk-import every image in a DISK folder (recursive) - the server
    reads straight from disk. Optionally lands each image in an asset,
    caption seeded from its gleaned prompt."""
    store = ctx(request).store
    b = await request.json()
    folder = Path(str(b.get("path") or "").strip().strip('"')).expanduser()
    if not folder.is_dir():
        return error(f"not a folder: {folder}")
    asset_name = (b.get("asset") or "").strip() or None

    def work():
        added = dups = skipped = 0
        files = list_images(folder)
        for q in files:
            try:
                data = q.read_bytes()
            except OSError:
                skipped += 1
                continue
            i, new = store.import_bytes(data)
            added += new
            dups += not new
            if asset_name:
                pr = ((store.images[i].get("recipe") or {}).get("prompt") or "")
                with store.lock:
                    asset_mod.add_project_image(asset_name, store.name, i, pr)
        return {"added": added, "duplicates": dups, "skipped": skipped, "total": len(files)}

    r = await asyncio.get_event_loop().run_in_executor(None, work)
    return ok(**r)


# ---------- projects, assets, training ----------

async def handle_project(request):
    """Switch or create a project (singleton: refuses while generating)."""
    app = ctx(request)
    if app.jobs.busy:
        return error("busy - wait or Stop first", 409)
    b = await request.json()
    name = (b.get("name") or "").strip()
    try:
        app.store = open_project(name)
    except ValueError as e:
        return error(str(e))
    print(f"project: {name}")
    return ok(project=name)


async def handle_asset(request):
    """Asset CRUD (v4): create / delete / add / remove / describe / add_lora."""
    store = ctx(request).store
    b = await request.json()
    with store.lock:
        assets = asset_mod.load_assets()
        why = asset_mod.apply_op(assets, b.get("op"), (b.get("name") or "").strip(),
                                 b.get("path"), b.get("description"))
        if why:
            return error(why, 404 if why.startswith("no asset") else 400)
        asset_mod.save_assets(assets)
    return ok(assets=assets)


async def handle_train(request):
    """Start a training job. The button click IS the user's say-so."""
    app = ctx(request)
    b = await request.json()
    name = (b.get("name") or "").strip()
    family = b.get("family") or "zimage"
    if family not in ("zimage", "klein", "illustrious"):
        return error(f"bad family {family!r}")
    if app.jobs.busy:
        return error("generating - wait or Stop first", 409)
    if app.jobs.training_running():
        return error("a training job is already running", 409)
    a = asset_mod.find_asset(asset_mod.load_assets(), name)
    if a is None:
        return error(f"no asset {name!r}", 404)
    try:
        sync_dataset(a)
    except ValueError as e:
        return error(str(e))
    cfg = TrainConfig(name, family, b.get("steps"))
    app.jobs.start_training(name, family, log_path(name), lambda: do_training(cfg))
    return ok(started=True, log=str(log_path(name)))


async def handle_train_abort(request):
    app = ctx(request)
    if not app.jobs.training_running():
        return ok(aborted=False)
    killed = abort_training()
    app.jobs.train["error"] = "aborted by user"
    return ok(aborted=killed)


ROUTES = {
    "state": handle_state, "controls": handle_controls, "place": handle_place,
    "clear": handle_clear, "pin": handle_pin, "slots": handle_slots,
    "generate": handle_generate, "pov": handle_pov, "abort": handle_abort,
    "family": handle_family, "meta": handle_meta, "discard": handle_discard,
    "prune": handle_prune, "gc": handle_gc,
    "import": handle_import, "import_url": handle_import_url,
    "import_folder": handle_import_folder,
    "project": handle_project, "asset": handle_asset,
    "train": handle_train, "train_abort": handle_train_abort,
}


def create_app(store, embed_workflow=False):
    app = web.Application(client_max_size=256 * 1024 ** 2)
    app["ctx"] = App(store, embed_workflow)
    app.router.add_get("/", index)
    app.router.add_get("/static/{path:.+}", serve_static)
    app.router.add_get("/api/state", handle_state)
    for name, handler in ROUTES.items():
        app.router.add_post(f"/api/{name}", handler)
    app.router.add_get("/file/{path:.+}", serve_file)
    app.router.add_get("/img/{id:\\d+}", serve_image)
    app.router.add_get("/img/{project}/{id:\\d+}", serve_image)
    return app
