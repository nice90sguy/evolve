"""lora.py - the LoRA dropdown: names in, files out, PER MODEL FAMILY.

A LoRA only loads into the model it was trained for, so the dropdown is
filtered by the active family and every lookup takes the family. Canonical
name = ASSET name, resolving to the NEWEST of that asset's LoRAs for that
family; recipes always record the RESOLVED path (lora_file), never the
alias, so a retrain changes the config (and the sibling key) instead of
silently changing old recipes' meaning. Loose files are listed by
root-relative path and must sit at loras/<asset>/<family>/ - anything
elsewhere is ignored with a console note.
"""
from pathlib import Path

import asset
from model_family import ModelFamily, parse_model_family
from project import root


def _loose_files():
    """{ModelFamily: {rel path}} of *.safetensors under loras/*/<family>/
    (.bak never; a file outside a family dir is reported once)."""
    out = {f: set() for f in ModelFamily}
    rl = root() / "loras"
    if not rl.is_dir():
        return out
    for q in rl.rglob("*.safetensors"):
        rel = q.relative_to(root()).as_posix()
        fam = asset.family_of_lora_path(rel)
        if fam is None:
            print(f"lora ignored (not under loras/<asset>/<family>/): {rel}")
            continue
        out[fam].add(rel)
    return out


def list_loras():
    """{family: [dropdown entries]} - asset aliases (that have a LoRA for
    the family) first, then loose files not owned by an asset."""
    assets = asset.load_assets()
    loose = _loose_files()
    out = {}
    for fam in ModelFamily:
        names = [a.name for a in assets if a.loras_for(fam)]
        owned = {e.path for a in assets for e in a.loras_for(fam)}
        out[fam.value] = names + sorted(loose[fam] - owned)
    return out


def resolve_lora(name, family):
    """Dropdown value -> absolute file for THIS family, or None. Asset
    alias -> newest of its LoRAs for the family; else a root-relative path
    under the family's dir; else an absolute path (family unchecked)."""
    if not name:
        return None
    fam = parse_model_family(family)
    a = asset.find_asset(asset.load_assets(), name)
    if a is not None:
        mine = a.loras_for(fam)
        if not mine:
            return None
        q = root() / mine[-1].path
        return q.resolve() if q.is_file() else None
    rel = str(name).replace("\\", "/")
    if asset.family_of_lora_path(rel) == fam and (root() / rel).is_file():
        return (root() / rel).resolve()
    p = Path(name)
    if p.is_absolute() and p.is_file():
        return p.resolve()
    return None
