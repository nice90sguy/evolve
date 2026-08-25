"""lora.py - the LoRA dropdown: names in, files out.

Canonical name = ASSET name, resolving to the NEWEST of that asset's
loras[]; recipes always record the RESOLVED path (lora_file), never the
alias, so a retrain changes the config (and the sibling key) instead of
silently changing old recipes' meaning. Loose files under <root>/loras are
listed by root-relative path. Nothing is CWD-relative (v4).
"""
from pathlib import Path

import asset
from project import root


def list_loras():
    """Dropdown entries: asset aliases first, then loose *_comfy / root
    *.safetensors files under <root>/loras (asset-owned files show as the
    alias only). .bak files are never listed."""
    assets = asset.load_assets()
    out = [a["name"] for a in assets if a.get("loras")]
    owned = {q for a in assets for q in a.get("loras", [])}
    found = set()
    rl = root() / "loras"
    if rl.is_dir():
        for q in rl.rglob("*_comfy.safetensors"):
            found.add(q.relative_to(root()).as_posix())
        for q in rl.glob("*.safetensors"):
            found.add("loras/" + q.name)
    return out + sorted(found - owned)


def resolve_lora(name):
    """Dropdown value -> absolute file, or None. Asset alias -> newest of
    its loras[]; else a root-relative path; else an absolute path."""
    if not name:
        return None
    a = asset.find_asset(asset.load_assets(), name)
    if a and a.get("loras"):
        q = root() / a["loras"][-1]
        return q.resolve() if q.is_file() else None
    for q in (root() / name, Path(name)):
        if q.is_file():
            return q.resolve()
    return None
