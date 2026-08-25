"""asset.py - assets are DATA (v4): a name and its LoRAs, per model family.

<root>/loras.json = [{name, loras: [{path, family}]}]. `path` is root-
relative posix and lives at loras/<name>/<family>/<file>.safetensors -
the family is recorded explicitly AND must agree with the directory (a
LoRA is specific to its model; the dropdown only ever offers the active
family's). Legacy bare-string entries are rejected at load with a pointer
to the migration tool. The training DATASET is the word
`lora_dataset_<name>` on images (unarchived); the caption is the image's
description with the trigger prefixed at sync time.
"""
from typing import List

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from model_family import ModelFamily, parse_model_family
from project import NAME_RE, is_valid_name, read_json, root, root_rel, write_json

DATASET_PREFIX = "lora_dataset_"
LORAS_FILE = "loras.json"        # was assets.json (renamed 2026-08-25)


def lora_dir(name, family):
    return root() / "loras" / name / parse_model_family(family).value


def family_of_lora_path(rel):
    """loras/<asset>/<family>/<file> -> ModelFamily, else None."""
    parts = str(rel).replace("\\", "/").split("/")
    if len(parts) == 4 and parts[0] == "loras":
        try:
            return parse_model_family(parts[2])
        except ValueError:
            return None
    return None


class LoraEntry(BaseModel):
    path: str
    family: ModelFamily

    @model_validator(mode="after")
    def _dir_agrees(self):
        fam = family_of_lora_path(self.path)
        if fam is None:
            raise ValueError(f"{self.path}: LoRA files live at loras/<asset>/<family>/")
        if fam != self.family:
            raise ValueError(f"{self.path}: directory says {fam}, entry says {self.family}")
        return self


class Asset(BaseModel):
    name: str = Field(pattern=NAME_RE.pattern)
    loras: List[LoraEntry] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v):
        if not is_valid_name(v):
            raise ValueError(f"bad asset name {v!r}")
        return v

    def loras_for(self, family):
        fam = parse_model_family(family)
        return [e for e in self.loras if e.family == fam]


class AssetsFormatError(RuntimeError):
    pass


def load_assets():
    """[Asset]. Raises AssetsFormatError on a pre-family file."""
    raw = read_json(root() / LORAS_FILE, [])
    out = []
    for a in raw:
        try:
            out.append(Asset.model_validate(a))
        except ValidationError as e:
            raise AssetsFormatError(
                f"{LORAS_FILE} entry {a.get('name')!r} is not in the per-family format "
                f"({e.errors()[0].get('msg')}); run tools/migrate_projects.py --loras") from e
    return out


def save_assets(assets):
    write_json(root() / LORAS_FILE, [a.model_dump(mode="json") for a in assets])


def find_asset(assets, name):
    return next((a for a in assets if a.name == name), None)


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


def append_lora(name, rel, family):
    """Record a freshly trained LoRA on its asset (persisted)."""
    assets = load_assets()
    a = find_asset(assets, name)
    if a is None:
        return False
    entry = LoraEntry(path=rel, family=parse_model_family(family))
    if not any(e.path == rel for e in a.loras):
        a.loras.append(entry)
        save_assets(assets)
    return True


def apply_op(assets, op, name, path=None, family=None):
    """create / delete / add_lora on the list, in place. Returns an error
    message or None."""
    a = find_asset(assets, name)
    if op == "create":
        if not is_valid_name(name):
            return (f"bad asset name {name!r} (it is also the LoRA trigger: "
                    "letters, digits, - _, no spaces)")
        if a:
            return f"asset {name!r} exists"
        assets.append(Asset(name=name))
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
        fam = family or family_of_lora_path(rel)
        if fam is None:
            return f"{rel}: LoRA files live at loras/<asset>/<family>/"
        try:
            entry = LoraEntry(path=rel, family=parse_model_family(fam))
        except (ValueError, ValidationError) as e:
            return f"bad LoRA entry: {e}"
        if not any(e.path == rel for e in a.loras):
            a.loras.append(entry)
        return None
    return f"bad op {op!r}"
