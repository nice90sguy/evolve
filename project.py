"""project.py - the global ROOT and the projects under it (architecture v4).

    <root>/config.json        app-level config (last_project, colors, ...)
    <root>/assets.json        the asset list (see asset.py)
    <root>/loras/<name>/      trained LoRAs      (reserved)
    <root>/_train/<name>/     transient datasets (reserved)
    <root>/_debug/            last_payload.json  (reserved)
    <root>/<project>/         images/NNN.png + journal.jsonl + state.json + archive/

A PROJECT is just a subdir; opening one has a single effect: the path
context. `<project>/<id>` is the global image name. NO uids anywhere.
The root is set ONCE per process (singleton contract: one server, one
root, one open project) and nothing here depends on the shell's cwd.
"""
import json
import re
from pathlib import Path

RESERVED = {"assets", "loras", "_train", "_debug"}   # never project names
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
PALETTE_SIZE = 8
SHARED_ASSETS = "shared_assets"     # reserved project name, amber tint

_ROOT = None


def set_root(path):
    """Fix the global root for this process (created if missing)."""
    global _ROOT
    _ROOT = Path(path).expanduser().resolve()
    _ROOT.mkdir(parents=True, exist_ok=True)
    (_ROOT / "_debug").mkdir(exist_ok=True)
    return _ROOT


def root():
    if _ROOT is None:
        raise RuntimeError("project.set_root() has not been called")
    return _ROOT


# ---------- tiny json-file helpers (the read/write cadence) ----------

def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path, data, indent=1):
    Path(path).write_text(json.dumps(data, indent=indent), encoding="utf-8")


# ---------- names & paths ----------

def is_valid_name(name):
    """Project AND asset names share one rule: letters, digits, - _, no
    spaces, never a reserved word (an asset name is also a LoRA trigger)."""
    return bool(name) and bool(NAME_RE.fullmatch(name)) and name not in RESERVED


def root_rel(p):
    """Canonical root-relative posix path; ValueError if outside the root."""
    q = Path(p)
    q = (q if q.is_absolute() else root() / q).resolve()
    return q.relative_to(root()).as_posix()


def contained(p):
    """Resolve p (root-relative or absolute) and require it inside the root;
    returns the resolved Path or None."""
    try:
        q = (root() / p).resolve()
        q.relative_to(root())
        return q
    except Exception:
        return None


def project_dir(name):
    if not is_valid_name(name):
        raise ValueError(f"bad project name {name!r}")
    return root() / name


def image_path(project_name, i):
    return root() / project_name / "images" / f"{i}.png"


def image_rel(project_name, i):
    """The global image name: '<project>/images/<id>.png'."""
    return f"{project_name}/images/{i}.png"


def parse_image_ref(rel):
    """'<project>/images/<id>.png' -> (project, id) or None."""
    parts = str(rel).split("/")
    if len(parts) != 3 or parts[1] != "images" or not parts[2].endswith(".png"):
        return None
    try:
        return parts[0], int(parts[2][:-4])
    except ValueError:
        return None


def list_projects():
    """A project is any root subdir that looks like one (has images/ or a
    journal). Reserved names and dot-dirs are never projects."""
    out = []
    for d in sorted(root().iterdir()):
        if not d.is_dir() or d.name in RESERVED or d.name.startswith("."):
            continue
        if (d / "images").is_dir() or (d / "journal.jsonl").exists():
            out.append(d.name)
    return out


# ---------- config ----------

def load_config():
    return read_json(root() / "config.json", {})


def save_config(**kw):
    c = load_config()
    c.update(kw)
    write_json(root() / "config.json", c)


def project_color(name):
    """Stable tint per project: assigned on first sight (next free slot of
    the 8-hue palette), persisted in config.json so a project keeps its
    colour for life - never roster-order, which would reshuffle everyone's
    colour when a project is added. shared_assets is reserved: always the
    same amber, outside the cycle."""
    if name == SHARED_ASSETS:
        return "shared"
    colors = load_config().get("colors") or {}
    if name not in colors:
        used = set(colors.values())
        colors[name] = next((i for i in range(PALETTE_SIZE) if i not in used),
                            len(colors) % PALETTE_SIZE)
        save_config(colors=colors)
    return colors[name]
