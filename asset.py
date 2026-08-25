"""asset.py - assets are DATA (v4), now just a name and its LoRAs.

<root>/assets.json = [{name, loras: [root-relative paths]}]. The training
DATASET is the word `lora_dataset_<name>` on images (unarchived); the
caption is the image's own description with the trigger prefixed at sync
time (`<name>, <description>`; bare `<name>` if empty). Names double as
the LoRA trigger and are never auto-decorated.
"""
from project import is_valid_name, read_json, root, root_rel, write_json

DATASET_PREFIX = "lora_dataset_"


def load_assets():
    out = []
    for a in read_json(root() / "assets.json", []):
        if a.get("name"):
            out.append({"name": a["name"], "loras": list(a.get("loras") or [])})
    return out


def save_assets(assets):
    write_json(root() / "assets.json", [{"name": a["name"], "loras": a["loras"]} for a in assets])


def find_asset(assets, name):
    return next((a for a in assets if a.get("name") == name), None)


def dataset_tag(name):
    return DATASET_PREFIX + name


def dataset_ids(store, name):
    """Unarchived images carrying the asset's dataset word, by id."""
    w = dataset_tag(name)
    return [i for i in store.with_word(w) if not store.is_archived(i)]


def caption(name, description):
    """The training caption: trigger prefixed by the app. Returns
    (caption, warning-or-None) - a description that already starts with
    the trigger is a double-prefix smell, not an error."""
    d = (description or "").strip()
    warn = f"description already starts with the trigger {name!r}" if d.startswith(name) else None
    return (f"{name}, {d}" if d else name), warn


def append_lora(name, rel):
    """Record a freshly trained LoRA on its asset (persisted)."""
    assets = load_assets()
    a = find_asset(assets, name)
    if a is not None and rel not in a["loras"]:
        a["loras"].append(rel)
        save_assets(assets)
    return a is not None


def apply_op(assets, op, name, path=None):
    """create / delete / add_lora on the list, in place. Returns an error
    message or None. (Dataset membership and descriptions are tag /
    describe operations on images, not asset operations.)"""
    a = find_asset(assets, name)
    if op == "create":
        if not is_valid_name(name):
            return (f"bad asset name {name!r} (it is also the LoRA trigger: "
                    "letters, digits, - _, no spaces)")
        if a:
            return f"asset {name!r} exists"
        assets.append({"name": name, "loras": []})
        return None
    if a is None:
        return f"no asset {name!r}"
    if op == "delete":
        assets.remove(a)
        return None
    if op == "add_lora":
        try:
            rel = root_rel(path)
        except Exception:
            return "path outside the root"
        if rel not in a["loras"]:
            a["loras"].append(rel)
        return None
    return f"bad op {op!r}"
