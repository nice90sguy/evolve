"""controls.py - the generator controls, defined ONCE.

CONTROLS is the single table of every widget the UI persists: defaults,
the persist filter, the sanitiser applied on load/receive, and the
restore-from-recipe rule (picking an image restores the run state that
GENERATED it). The client renders/reads widgets by these same names.
"""
from model_family import DEFAULT_MODEL_FAMILY, parse_model_family
from project import load_settings

TABS = ("create", "derive", "camera")

CONTROLS = {"prompt": "", "negative": "", "family": "klein",
            "ref0": None, "refs": [None, None, None],
            "lora": "", "lora_strength": 1.0, "lock": "SOFT_LOCK",
            "vary": 0.0, "seed": 0, "whitebg": True,
            "width": 1024, "height": 1024, "fresh_model": False,
            "size_from_ref0": True,        # Derive: render at ref0's size (else w/h as typed)
            "steps": 0, "cfg": 0,          # 0 = the family's default
            "tab": "create",               # active action-panel tab (v3)
            "seed_create": 0, "seed_derive": 0, "seed_camera": 0,
            "outputs_create": 6, "outputs_derive": 6, "outputs_camera": 4,
            # camera axes: None = axis unchecked, its token is OMITTED
            "pov_azim": None, "pov_elev": None, "pov_dist": None}


def fresh_controls():
    c = dict(CONTROLS)
    c["refs"] = [None, None, None]
    s = load_settings()                    # w/h defaults live in config (Settings page later)
    c["width"], c["height"] = s["default_width"], s["default_height"]
    return c


def sanitize_controls(given, alive):
    """Defaults + the known keys of `given`, with dead image ids dropped and
    refs padded to exactly three."""
    c = fresh_controls()
    c.update({k: v for k, v in given.items() if k in CONTROLS})
    if not alive(c.get("ref0")):
        c["ref0"] = None
    c["refs"] = [i if alive(i) else None for i in (c.get("refs") or [])][:3]
    c["refs"] += [None] * (3 - len(c["refs"]))
    if c.get("tab") not in TABS:
        c["tab"] = "create"
    c["family"] = parse_model_family(c.get("family"), DEFAULT_MODEL_FAMILY).value
    return c


def persistable(edits):
    return {k: v for k, v in edits.items() if k in CONTROLS}


def restore_from_image(store, i):
    """Picking an image restores the run state that GENERATED it into the
    store's controls: every widget, ref0 = its PARENT (empty for fiat - so
    Generate re-rolls that round with new seeds), extras from its recorded
    co-parents, and the tab that made it (tab-switch-on-pick, v3). Imports
    have no recipe: ref0 = none."""
    rec = store.images[i]
    r = rec.get("recipe")
    c = store.state["controls"]
    if not r:
        c["ref0"] = None
        return
    c["prompt"] = r.get("prompt", c["prompt"])
    c["negative"] = r.get("negative", "")
    c["family"] = parse_model_family(r.get("family"), DEFAULT_MODEL_FAMILY).value
    c["steps"] = int(r.get("steps") or 0)
    c["cfg"] = float(r.get("cfg") or 0)
    c["seed"] = int(r.get("seed") or 0)
    c["whitebg"] = bool(r.get("whitebg", c["whitebg"]))
    c["width"], c["height"] = int(r.get("width", c["width"])), int(r.get("height", c["height"]))
    if r.get("lock"):
        c["lock"] = r["lock"]
    c["vary"] = float(r.get("vary") or 0.0)
    parents = [q for q in (rec.get("parents") or []) if store.alive(q)]
    if r.get("op") == "pov":
        # a Camera candidate restores its tab + axis positions (checked =
        # token was present); ref0 = its single parent
        c["tab"] = "camera"
        c["pov_azim"] = r.get("pov_azim")
        c["pov_elev"] = r.get("pov_elev")
        c["pov_dist"] = r.get("pov_dist")
        c["ref0"] = parents[0] if parents else None
        c["refs"] = [None, None, None]
        c["seed_camera"] = int(r.get("seed") or 0)
        return
    c["lora"] = r.get("lora") or ""
    if r.get("lora_strength") is not None:
        c["lora_strength"] = float(r["lora_strength"])
    had_ref0 = bool(r.get("ref0", r.get("use_working")))   # old recipes: use_working
    c["ref0"] = parents[0] if (had_ref0 and parents) else None
    extras = parents[1:] if had_ref0 else parents
    c["refs"] = (extras + [None, None, None])[:3]
    c["tab"] = "derive" if (c["ref0"] is not None
                            or any(x is not None for x in c["refs"])) else "create"
    c["seed_" + c["tab"]] = int(r.get("seed") or 0)
