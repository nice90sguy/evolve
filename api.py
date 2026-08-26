"""api.py - the HTTP surface the frontend talks to (aiohttp).

Handlers validate + translate; the work is done by the operators
(generate.do_generate, camera.do_camera, training.do_training) and the
data layer (store, trash, lora, lineage). One App context per process:
the store, the job runner, the settings.

    GET  /                  frontend/index.html (live from disk, no-store)
    GET  /static/{path}     frontend assets
    GET  /img/{id}          a stored image
    GET  /api/state         the full UI snapshot (polled)
    POST /api/<name>        everything else (see ROUTES)
"""
import asyncio
import mimetypes
import re
import urllib.request
from pathlib import Path

from aiohttp import web
from pydantic import ValidationError

import camera
import comfy_client
import lineage
import trash
from controls import persistable, restore_from_image, sanitize_controls
from model_family import model_families_for_ui
from generate import GenerateConfig, do_generate
from image_file import list_images
from jobs import Jobs
import lora as lora_mod
from project import clean_tags, load_settings, save_settings
from training import (TrainConfig, abort_training, do_training, log_path,
                      sync_dataset)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


class App:
    """Process-wide context (singleton contract: one server, one root)."""

    def __init__(self, store, embed_workflow=False):
        self.store = store
        self.jobs = Jobs()
        self.embed_workflow = embed_workflow

    def snapshot(self):
        snap = self.store.snapshot()
        snap.update({
            "loras": [x.model_dump(mode="json") for x in lora_mod.load_loras()],
            "settings": load_settings(),
            "train": self.jobs.train_status(),
            "comfy_ok": comfy_client.is_alive(busy=bool(self.jobs.busy)),
            "busy": self.jobs.busy,
            "families": model_families_for_ui(),
            "pov_azim": camera.AXIS_AZIMUTH, "pov_elev": camera.AXIS_ELEVATION,
            "pov_dist": camera.AXIS_DISTANCE,
            "lora_menu": lora_mod.menu()})
        return snap


def ctx(request):
    return request.app["ctx"]


def ok(**kw):
    return web.json_response(kw)


def error(msg, status=400):
    return web.json_response({"error": msg}, status=status)


def invalid(e):
    """A pydantic ValidationError as one readable 400."""
    msgs = [f"{'.'.join(str(x) for x in err['loc']) or 'input'}: {err['msg']}" for err in e.errors()]
    return error("; ".join(msgs))


def id_list(b):
    """{'id': 3} or {'ids': [3, 4]} -> [3, 4]."""
    ids = b.get("ids")
    if ids is None:
        ids = [b.get("id")]
    return [int(i) for i in ids if i is not None]


# ---------- pages & files ----------

async def index(request):
    """?classic=1 serves the pre-shell layout (an escape hatch while the
    task-rail shell is on trial)."""
    page = "classic.html" if request.query.get("classic") else "index.html"
    return web.FileResponse(FRONTEND_DIR / page,
                            headers={"Cache-Control": "no-store"})


async def serve_static(request):
    f = (FRONTEND_DIR / request.match_info["path"]).resolve()
    try:
        f.relative_to(FRONTEND_DIR)
    except ValueError:
        raise web.HTTPNotFound()
    if not f.is_file():
        raise web.HTTPNotFound()
    ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
    return web.FileResponse(f, headers={"Cache-Control": "no-store", "Content-Type": ctype})


async def serve_image(request):
    """/img/<id>. File existence is the truth (purged = 404). Bytes may
    legitimately CHANGE under an id (external swap-in is a supported user
    power, 2026-08-26), so the browser must revalidate: no-cache +
    Last-Modified gives cheap localhost 304s (the old lag was the ping,
    not revalidation)."""
    i = int(request.match_info["id"])
    f = ctx(request).store.path(i)
    if not f.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(f, headers={
        "Content-Disposition": f'inline; filename="{i}.png"',
        "Cache-Control": "no-cache"})


# ---------- state & controls ----------

async def handle_state(request):
    return web.json_response(ctx(request).snapshot())


async def handle_settings(request):
    """App settings (default_tags, ...). Returns the settings as saved."""
    b = await request.json()
    return web.json_response(save_settings(**b))


async def handle_controls(request):
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
            store.touch([i])
            store.save_state()
        elif target == "ref0":
            store.state["controls"]["ref0"] = i
            store.touch([i])
            store.save_state()
        elif target == "slot":
            return error("Output is read-only: nothing gets into it but a round", 409)
        elif target == "pin":
            store.pin(i, True)
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
            return error("Output is read-only (Del trashes the image instead)", 409)
        elif target == "pin":
            pins = store.pins()
            if index < len(pins):
                store.pin(pins[index], False)
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


# ---------- tags & descriptions ----------

async def handle_tag(request):
    """{id|ids, add:[..], remove:[..], cascade:bool} -> {touched:[ids]}."""
    store = ctx(request).store
    b = await request.json()
    ids = id_list(b)
    add, remove = clean_tags(b.get("add") or []), clean_tags(b.get("remove") or [])
    if not ids or not (add or remove):
        return error("nothing to do")
    touched = store.tag(ids, add=add, remove=remove, cascade=bool(b.get("cascade")))
    return ok(touched=touched)


async def handle_archive(request):
    """{id|ids, on:bool} -> {touched:[ids]}. Archiving skips pinned images
    (force:true unpins them); restoring never touches words."""
    store = ctx(request).store
    b = await request.json()
    ids = id_list(b)
    if not ids:
        return error("nothing to do")
    if b.get("on", True):
        touched = store.archive(ids, force=bool(b.get("force")))
        with store.lock:
            store.forget(touched)
            store.save_state()
    else:
        touched = store.restore(ids)
    return ok(touched=touched)


async def handle_winav(request):
    """{dir: -1|1}: browser back/forward for the Working Image. Restores
    the landed image's recipe, exactly like picking it."""
    store = ctx(request).store
    b = await request.json()
    i = store.nav_step(1 if int(b.get("dir", -1)) > 0 else -1)
    if i is None:
        return ok(id=None)
    with store.lock:
        restore_from_image(store, i)
        store.save_state()
    return ok(id=i, around=store.nav_neighbors())


async def handle_cwd(request):
    """Set the working folder (created if missing). Fiat images land here.
    Also a stat pass, so Explorer deletions show up on the next render."""
    store = ctx(request).store
    b = await request.json()
    if not store.set_cwd(b.get("dir") or ""):
        return error(f"bad folder {b.get('dir')!r}")
    store.check_files()
    return ok()


async def handle_dir_info(request):
    """{dir}: the images in a folder (direct and with subfolders) and the
    words they carry - the Folder Info popup's data."""
    from store import valid_dir
    store = ctx(request).store
    b = await request.json()
    d = valid_dir(b.get("dir") or "")
    if d is None:
        return error(f"bad folder {b.get('dir')!r}")
    with store.lock:
        direct, below = [], []
        for i in store.alive_ids():
            di = store.image_dir(i)
            if di == d:
                direct.append(i)
            elif di.startswith(d + "/"):
                below.append(i)
        words = {}
        for i in direct + below:
            for w in store.tags(i):
                words[w] = words.get(w, 0) + 1
        return ok(dir=d, direct=direct, below=below, words=words)


async def handle_check(request):
    """Stat every known file (cheap; no walk): refresh the missing set."""
    return ok(missing=ctx(request).store.check_files())


async def handle_forget_missing(request):
    """{ids?}: tombstone missing records (all of them by default)."""
    b = await request.json()
    ids = b.get("ids")
    return ok(forgotten=ctx(request).store.forget_missing(ids))


async def handle_move(request):
    """{id|ids, to}: move images to another folder (drag in the tree)."""
    store = ctx(request).store
    b = await request.json()
    moved = store.move(id_list(b), b.get("to") or "")
    return ok(moved=moved)


async def handle_mkdir(request):
    from store import valid_dir
    b = await request.json()
    d = valid_dir(b.get("dir") or "")
    if d is None:
        return error(f"bad folder {b.get('dir')!r}")
    (ctx(request).store.dir / d).mkdir(parents=True, exist_ok=True)
    return ok(dir=d)


async def handle_rescan(request):
    """Start a background rescan (Explorer edits become history; alien files
    are imported in place). Progress = snapshot.scan_busy; the result lands
    in snapshot.rescan (the UI toasts it)."""
    store = ctx(request).store
    if store.scan_busy:
        return error("a rescan is already running", 409)
    import threading
    threading.Thread(target=store.rescan, daemon=True).start()
    return ok(started=True)


async def handle_describe(request):
    b = await request.json()
    if not ctx(request).store.describe(b.get("id"), b.get("description") or ""):
        return error(f"no such image {b.get('id')}", 404)
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
    if op == "create":
        c["ref0"] = None
        c["refs"] = [None, None, None]
    else:
        c["family"] = "klein"
    try:
        cfg = GenerateConfig.from_controls(c, op)
    except ValidationError as e:
        return invalid(e)
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
    app = ctx(request)
    store = app.store
    why = app.jobs.idle_error()
    if why:
        return error(why, 409)
    q = await request.json()
    if not store.alive(store.state["working"]):
        return error("no working image to re-shoot")
    try:
        cfg = camera.CameraConfig(source_id=store.state["working"],
                                  azim=q.get("azim") or None, elev=q.get("elev") or None,
                                  dist=q.get("dist") or None, seed=int(q.get("seed") or 0))
    except ValidationError as e:
        return invalid(e)
    with store.lock:
        store.state["slots"] = max(1, min(64, int(q.get("outputs") or store.state["slots"])))
        store.save_state()
        total = store.state["slots"]
    app.jobs.start_round("camera", total, lambda progress, should_abort: camera.do_camera(
        store, cfg, progress, should_abort, app.embed_workflow), label="camera")
    return ok(started=True)


async def handle_abort(request):
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
    """Full metadata for ONE image (Info Window)."""
    store = ctx(request).store
    q = await request.json()
    i = q.get("id")
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
        return error(why, 404)
    return ok()


async def handle_prune(request):
    store = ctx(request).store
    b = await request.json()
    i, force = b.get("id"), bool(b.get("force"))
    if not store.alive(i):
        return error(f"no such image {i}", 404)
    plan = trash.prune_apply(store, i, force) if b.get("apply") else trash.prune_plan(store, i, force)
    return web.json_response(plan)


async def handle_empty_trash(request):
    """{apply: bool}: the impact plan, or the deed (OS recycle bin)."""
    store = ctx(request).store
    b = await request.json()
    if not b.get("apply"):
        return web.json_response(trash.empty_trash(store, apply=False))
    r = await asyncio.get_event_loop().run_in_executor(
        None, lambda: trash.empty_trash(store, apply=True))
    return web.json_response(r)


# ---------- import ----------

async def handle_import(request):
    """Raw image bytes (drop/paste/file) -> stored image id. Optional
    ?tags=a,b puts words on it."""
    data = await request.read()
    if not data:
        return error("empty upload")
    tags = clean_tags((request.query.get("tags") or "").split(","))
    try:
        i, _ = ctx(request).store.import_bytes(data, tags=tags)
    except Exception as e:
        return error(f"not a readable image: {e}")
    return ok(id=i)


async def handle_import_url(request):
    b = await request.json()
    url = b["url"]
    if not re.match(r"^https?://", url):
        return error("only http(s) urls")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 evolve"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read(64 * 1024 ** 2)
        i, _ = ctx(request).store.import_bytes(data, tags=clean_tags(b.get("tags") or []))
    except Exception as e:
        return error(f"fetch failed: {e}")
    return ok(id=i)


async def handle_import_folder(request):
    """Bulk-import every image in a DISK folder (recursive), optionally
    putting words on each (e.g. a LoRA's dataset word)."""
    store = ctx(request).store
    b = await request.json()
    folder = Path(str(b.get("path") or "").strip().strip('"')).expanduser()
    if not folder.is_dir():
        return error(f"not a folder: {folder}")
    tags = clean_tags(b.get("tags") or [])

    def work():
        added = dups = skipped = 0
        files = list_images(folder)
        for q in files:
            try:
                data = q.read_bytes()
            except OSError:
                skipped += 1
                continue
            _, new = store.import_bytes(data, tags=tags)
            added += new
            dups += not new
        return {"added": added, "duplicates": dups, "skipped": skipped, "total": len(files)}

    r = await asyncio.get_event_loop().run_in_executor(None, work)
    return ok(**r)


# ---------- LoRAs & training ----------

async def handle_lora(request):
    """create / delete / add_file on loras.json. Dataset membership = the
    word lora_dataset_<name> (use /api/tag); captions = /api/describe."""
    store = ctx(request).store
    b = await request.json()
    with store.lock:
        loras = lora_mod.load_loras()
        why = lora_mod.apply_op(loras, b.get("op"), (b.get("name") or "").strip(),
                                b.get("path"), b.get("family"))
        if why:
            return error(why, 404 if why.startswith("no LoRA") else 400)
        lora_mod.save_loras(loras)
    return ok(loras=[x.model_dump(mode="json") for x in loras])


async def handle_train(request):
    app = ctx(request)
    b = await request.json()
    name = (b.get("name") or "").strip()
    try:
        cfg = TrainConfig(name=name, family=b.get("family") or "zimage", steps=b.get("steps"))
    except ValidationError as e:
        return invalid(e)
    if app.jobs.busy:
        return error("generating - wait or Stop first", 409)
    if app.jobs.training_running():
        return error("a training job is already running", 409)
    if lora_mod.find_lora(lora_mod.load_loras(), name) is None:
        return error(f"no LoRA {name!r}", 404)
    try:
        sync_dataset(app.store, name)
    except ValueError as e:
        return error(str(e))
    app.jobs.start_training(name, cfg.family.value, log_path(name), lambda: do_training(cfg))
    return ok(started=True, log=str(log_path(name)))


async def handle_train_abort(request):
    app = ctx(request)
    if not app.jobs.training_running():
        return ok(aborted=False)
    killed = abort_training()
    app.jobs.train["error"] = "aborted by user"
    return ok(aborted=killed)


ROUTES = {
    "state": handle_state, "settings": handle_settings, "controls": handle_controls,
    "place": handle_place, "clear": handle_clear, "pin": handle_pin, "slots": handle_slots,
    "tag": handle_tag, "describe": handle_describe, "archive": handle_archive,
    "cwd": handle_cwd, "move": handle_move, "mkdir": handle_mkdir, "rescan": handle_rescan,
    "check": handle_check, "forget_missing": handle_forget_missing, "dir_info": handle_dir_info,
    "winav": handle_winav,
    "generate": handle_generate, "pov": handle_pov, "abort": handle_abort,
    "family": handle_family, "meta": handle_meta, "discard": handle_discard,
    "prune": handle_prune, "empty_trash": handle_empty_trash,
    "import": handle_import, "import_url": handle_import_url,
    "import_folder": handle_import_folder,
    "lora": handle_lora, "train": handle_train, "train_abort": handle_train_abort,
}


def create_app(store, embed_workflow=False):
    app = web.Application(client_max_size=256 * 1024 ** 2)
    app["ctx"] = App(store, embed_workflow)
    app.router.add_get("/", index)
    app.router.add_get("/static/{path:.+}", serve_static)
    app.router.add_get("/api/state", handle_state)
    for name, handler in ROUTES.items():
        app.router.add_post(f"/api/{name}", handler)
    app.router.add_get("/img/{id:\\d+}", serve_image)
    return app
