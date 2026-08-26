"""project.py - the ROOT: one store, one journal, one state, app settings.

    <root>/images/NNN.png       every image (ids global, never reused)
    <root>/journal.jsonl        append-only: image / tag / describe / hist / purge
    <root>/state.json           the live UI state
    <root>/config.json          app settings (default_tags, ...)
    <root>/loras.json           [{name, files: [{path, family}]}]  (see lora.py)
    <root>/loras/<name>/        trained LoRAs
    <root>/_train/<name>/       transient datasets
    <root>/_debug/              last_payload.json

Tags replaced projects (2026-08-25): there are no subdirectories of images
and no archive dir - files never move. The root is set ONCE per process
and nothing here depends on the shell's cwd.
"""
import json
import re
from pathlib import Path

NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
TAG_RE = re.compile(r"[^\s,]{1,64}")          # a word: no whitespace, no commas
RESERVED_WORDS = {"archived"}                 # a state bit, never a word
DEFAULT_SETTINGS = {"default_tags": [], "default_width": 1024, "default_height": 1024}

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


# ---------- tiny json-file helpers ----------

def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path, data, indent=1):
    Path(path).write_text(json.dumps(data, indent=indent), encoding="utf-8")


# ---------- names ----------

def is_valid_name(name):
    """LoRA names (= trigger words): letters, digits, - _, no spaces."""
    return bool(name) and bool(NAME_RE.fullmatch(name))


def is_valid_tag(word):
    return bool(word) and bool(TAG_RE.fullmatch(word)) and word not in RESERVED_WORDS


def clean_tags(words):
    """Normalise a tag list: strings, stripped, valid, deduped, order kept."""
    out = []
    for w in words or []:
        w = str(w).strip()
        if is_valid_tag(w) and w not in out:
            out.append(w)
    return out


def root_rel(p):
    """Canonical root-relative posix path; ValueError if outside the root."""
    q = Path(p)
    q = (q if q.is_absolute() else root() / q).resolve()
    return q.relative_to(root()).as_posix()


# ---------- settings ----------

def load_config():
    return read_json(root() / "config.json", {})


def save_config(**kw):
    c = load_config()
    c.update(kw)
    write_json(root() / "config.json", c)


def load_settings():
    """App settings with defaults filled in (default_tags: applied to every
    image the app makes or imports)."""
    c = load_config()
    s = dict(DEFAULT_SETTINGS)
    s.update({k: c[k] for k in DEFAULT_SETTINGS if k in c})
    s["default_tags"] = clean_tags(s["default_tags"])
    for k in ("default_width", "default_height"):
        try:
            s[k] = max(16, min(4096, int(s[k])))
        except (TypeError, ValueError):
            s[k] = 1024
    return s


def save_settings(**kw):
    s = {k: v for k, v in kw.items() if k in DEFAULT_SETTINGS}
    if "default_tags" in s:
        s["default_tags"] = clean_tags(s["default_tags"])
    save_config(**s)
    return load_settings()
