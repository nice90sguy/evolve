"""evolve - a virtual production studio: images bred by selective breeding
with auditable heredity, driving ComfyUI headlessly.

    python evolve.py --root D:/evolve_root [--port 8189]

--root is the ONE store (mandatory - nothing depends on the shell's cwd):
images/, journal.jsonl, state.json, config.json, loras.json, loras/,
_train/, _debug/. Images are grouped, filtered and given meaning by TAGS;
there are no project subdirectories. ComfyUI must already be running at
127.0.0.1:8188; it is never launched or killed from here. Generation
happens only on your click.

Layout of this package: api.py (HTTP) -> generate/camera/training
(operators) -> store/trash/lora/lineage (data) -> build_payload +
templates/ (graphs) -> comfy_client / image_* / project (leaves). The UI is
frontend/ (served live from disk). tools/ holds the standalone drivers.
"""
import argparse

from aiohttp import web

import comfy_client
import lora
import project
from api import create_app
from store import Store


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the store root")
    ap.add_argument("--port", type=int, default=8189)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--embed-workflow", action="store_true",
                    help="also embed UI geometry (a `workflow` chunk) in output "
                         "pngs so they drag into the ComfyUI frontend editable "
                         "(off by default: it is the bulk of the file)")
    args = ap.parse_args()

    comfy_client.configure_console()
    root = project.set_root(args.root)
    comfy_client.DEBUG_DIR = root / "_debug"
    old = [d.name for d in root.iterdir()
           if d.is_dir() and (d / "images").is_dir() and d.name != "images"]
    if old and not (root / "journal.jsonl").exists():
        raise SystemExit(f"{root} holds project subdirs ({', '.join(old)}) but no root "
                         "journal: run  python tools/migrate_projects.py --root "
                         f"{root}  first (tags replaced projects, 2026-08-25)")
    store = Store(root)
    stranded = [i for i in store.alive_ids()
                if store.is_archived(i) and not store.path(i).is_file()
                and (root / "images" / store.images[i]["file"]).is_file()]
    if stranded:
        raise SystemExit(f"{len(stranded)} archived image(s) still live in images/ "
                         "(pre-Places layout): run  python tools/migrate_projects.py "
                         f"--root {root} --places  first")
    import threading
    from trash import sweep_orphans
    orphans = sweep_orphans(store)       # fresh images listed in no Output -> trash
    if orphans:
        print(f"threw away {len(orphans)} orphaned fresh image(s): {orphans}")

    def startup_rescan():
        def bar(done, total, phase):
            print(f"\r  rescan {phase} {done}/{total}\x1b[K", end="", flush=True)
        r = store.rescan(progress=bar)
        print(f"\rrescan: {r['moved']} moved, {r['imported']} imported, {r['missing']} missing, "
              f"{r['revived']} revived" + (f", {len(r['skipped'])} skipped" if r["skipped"] else ""))
        for line in r["skipped"][:10]:
            print("  " + line)
    threading.Thread(target=startup_rescan, daemon=True).start()   # never block the UI
    try:
        lora.load_loras()            # fail fast on a pre-family / pre-rename loras.json
    except lora.LorasFormatError as e:
        raise SystemExit(str(e))
    app = create_app(store, embed_workflow=args.embed_workflow)
    print(f"root:  {root}  ({len(store.alive_ids())} images, "
          f"{len(store.words())} words)")
    print(f"open:  http://{args.host}:{args.port}/")
    print("(ComfyUI must be running at 127.0.0.1:8188 before you generate)")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
