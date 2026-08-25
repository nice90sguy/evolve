"""asset.py - assets are DATA, not directories (architecture v4).

<root>/assets.json is a list of {name, loras: [paths], dataset: [{path,
description}]}. Paths are root-relative posix; images live wherever they
live (any project). `description` IS the training caption and must START
with the asset name (= the LoRA trigger; validated at edit time in the UI,
enforced at train time, never auto-repaired - the akasakas incident).
"""
from project import image_rel, is_valid_name, read_json, root, root_rel, write_json


def load_assets():
    return read_json(root() / "assets.json", [])


def save_assets(assets):
    write_json(root() / "assets.json", assets)


def find_asset(assets, name):
    return next((a for a in assets if a.get("name") == name), None)


def dataset_ids(project_name):
    """Image ids of THIS project that any asset's dataset lists ->
    {id: [(asset_name, path), ...]}. These are GC roots for the project."""
    pref = project_name + "/images/"
    out = {}
    for a in load_assets():
        for e in a.get("dataset", []):
            q = str(e.get("path", ""))
            if q.startswith(pref) and q.endswith(".png"):
                try:
                    out.setdefault(int(q[len(pref):-4]), []).append((a["name"], q))
                except ValueError:
                    pass
    return out


def add_entry(a, path, description):
    """Append a dataset entry once (no-op if the path is already listed)."""
    if not any(e["path"] == path for e in a["dataset"]):
        a["dataset"].append({"path": path, "description": description})


def remove_paths(assets, paths):
    """Drop dataset entries for the given root-relative paths, everywhere."""
    gone = set(paths)
    for a in assets:
        a["dataset"] = [e for e in a["dataset"] if e["path"] not in gone]


def append_lora(name, rel):
    """Record a freshly trained LoRA on its asset (persisted)."""
    assets = load_assets()
    a = find_asset(assets, name)
    if a is not None and rel not in a["loras"]:
        a["loras"].append(rel)
        save_assets(assets)
    return a is not None


def seeded_description(name, prompt):
    """Caption seed for a new dataset entry: `name + ", " + recipe prompt`."""
    prompt = (prompt or "").strip()
    return f"{name}, {prompt}" if prompt else name


def apply_op(assets, op, name, path=None, description=None):
    """One asset CRUD operation on the list, in place. ops: create / delete /
    add / remove / describe / add_lora. Returns an error message or None.
    Descriptions are stored EXACTLY as given."""
    a = find_asset(assets, name)
    if op == "create":
        if not is_valid_name(name):
            return (f"bad asset name {name!r} (it is also the LoRA trigger: "
                    "letters, digits, - _, no spaces)")
        if a:
            return f"asset {name!r} exists"
        assets.append({"name": name, "loras": [], "dataset": []})
        return None
    if a is None:
        return f"no asset {name!r}"
    if op == "delete":
        assets.remove(a)
        return None
    if op not in ("add", "remove", "describe", "add_lora"):
        return f"bad op {op!r}"
    try:
        rel = root_rel(path)
    except Exception:
        return "path outside the root"
    if op == "add":
        add_entry(a, rel, description or name)
    elif op == "remove":
        a["dataset"] = [e for e in a["dataset"] if e["path"] != rel]
    elif op == "describe":
        for e in a["dataset"]:
            if e["path"] == rel:
                e["description"] = description or ""
    elif op == "add_lora":
        if rel not in a["loras"]:
            a["loras"].append(rel)
    return None


def add_project_image(name, project_name, image_id, prompt):
    """Bulk-import helper: land a project image in an asset, caption seeded
    from its gleaned prompt (persisted). Unknown asset = no-op."""
    assets = load_assets()
    a = find_asset(assets, name)
    if a is None:
        return False
    add_entry(a, image_rel(project_name, image_id), seeded_description(name, prompt))
    save_assets(assets)
    return True
