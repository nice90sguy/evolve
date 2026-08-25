"""evolve - a virtual production studio: images bred by selective breeding
with auditable heredity, driving ComfyUI headlessly.

    python evolve.py --root D:/evolve_root [--port 8189]

--root is the ONE store (mandatory - nothing depends on the shell's cwd):
images/, journal.jsonl, state.json, config.json, assets.json, loras/,
_train/, _debug/. Images are grouped, filtered and given meaning by TAGS;
there are no project subdirectories. ComfyUI must already be running at
127.0.0.1:8188; it is never launched or killed from here. Generation
happens only on your click.

Layout of this package: api.py (HTTP) -> generate/camera/training
(operators) -> store/trash/asset/lineage/lora (data) -> build_payload +
templates/ (graphs) -> comfy_client / image_* / project (leaves). The UI is
frontend/ (served live from disk). tools/ holds the standalone drivers.
"""
import argparse

from aiohttp import web

import comfy_client
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
    app = create_app(store, embed_workflow=args.embed_workflow)
    print(f"root:  {root}  ({len(store.alive_ids())} images, "
          f"{len(store.words())} words)")
    print(f"open:  http://{args.host}:{args.port}/")
    print("(ComfyUI must be running at 127.0.0.1:8188 before you generate)")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
