"""lora.py - the LoRA dropdown: names in, files out, PER MODEL FAMILY.

A LoRA only loads into the model it was trained for, so the dropdown is
filtered by the active family. The user is offered exactly ONE choice per
asset per family: the asset's NAME, resolving to the newest LoRA recorded
for that family in loras.json. Nothing else is listed - not older
versions, not the musubi-native files, not stray files on disk (put a
file in loras.json to make it usable). Recipes always record the RESOLVED
path (lora_file), never the alias, so a retrain changes the config (and
the sibling key) instead of silently changing old recipes' meaning.
"""
import asset
from model_family import ModelFamily, parse_model_family
from project import root


def list_loras():
    """{family: [asset names that have a LoRA for that family]}."""
    assets = asset.load_assets()
    return {fam.value: [a.name for a in assets if a.loras_for(fam)] for fam in ModelFamily}


def resolve_lora(name, family):
    """Asset name -> absolute file of its newest LoRA for THIS family, or
    None (unknown asset, no LoRA for the family, or the file is missing)."""
    if not name:
        return None
    a = asset.find_asset(asset.load_assets(), name)
    if a is None:
        return None
    mine = a.loras_for(parse_model_family(family))
    if not mine:
        return None
    q = root() / mine[-1].path
    return q.resolve() if q.is_file() else None
