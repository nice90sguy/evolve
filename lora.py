"""lora.py - LoRAs: a name (= the trigger word) with trained files per model
family, a dataset that is simply the word `lora_dataset_<name>` on images,
and the dropdown that offers them.

    <root>/loras.json                 [{name, files: [{path, family}]}]
    <root>/loras/<name>/<family>/     the .safetensors files (+ logs/)

A LoRA is specific to its model: every file records its family AND lives
in that family's directory (the two must agree). THE DROPDOWN OFFERS
EXACTLY ONE CHOICE PER LoRA PER FAMILY - the name, resolving to the newest
file recorded for that family; never raw files, older versions or
musubi-native files (a file becomes usable only by being in loras.json).
Recipes record the RESOLVED path (lora_file), never the name, so a retrain
changes the config instead of silently changing old recipes' meaning.
The caption for training = `<name>, <image description>` (trigger prefixed
by the app at sync time; bare `<name>` if the description is empty).
"""
from typing import List

from pydantic import BaseModel, ValidationError, field_validator, model_validator

from model_family import ModelFamily, parse_model_family
from project import is_valid_name, read_json, root, root_rel, write_json

LORAS_FILE = "loras.json"
DATASET_PREFIX = "lora_dataset_"


# ---------- the data ----------

def lora_dir(name, family):
    return root() / "loras" / name / parse_model_family(family).value


def family_of_path(rel):
    """loras/<name>/<family>/<file> -> ModelFamily, else None."""
    parts = str(rel).replace("\\", "/").split("/")
    if len(parts) == 4 and parts[0] == "loras":
        try:
            return parse_model_family(parts[2])
        except ValueError:
            return None
    return None


class LoraFile(BaseModel):
    path: str                  # root-relative posix
    family: ModelFamily

    @model_validator(mode="after")
    def _dir_agrees(self):
        fam = family_of_path(self.path)
        if fam is None:
            raise ValueError(f"{self.path}: LoRA files live at loras/<name>/<family>/")
        if fam != self.family:
            raise ValueError(f"{self.path}: directory says {fam}, entry says {self.family}")
        return self


class LoRA(BaseModel):
    name: str
    files: List[LoraFile] = []

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v):
        if not is_valid_name(v):
            raise ValueError(f"bad LoRA name {v!r} (it is the trigger word: letters, digits, - _)")
        return v

    def files_for(self, family):
        fam = parse_model_family(family)
        return [f for f in self.files if f.family == fam]


class LorasFormatError(RuntimeError):
    pass


def load_loras():
    """[LoRA]. Raises LorasFormatError on a pre-family / pre-rename file."""
    out = []
    for raw in read_json(root() / LORAS_FILE, []):
        try:
            if "files" not in raw and "loras" in raw:
                raise ValueError("entries are `files`, not `loras`")
            out.append(LoRA.model_validate(raw))
        except (ValidationError, ValueError) as e:
            msg = e.errors()[0].get("msg") if isinstance(e, ValidationError) else str(e)
            raise LorasFormatError(f"{LORAS_FILE} entry {raw.get('name')!r} is not in the current "
                                   f"format ({msg}); run tools/migrate_projects.py --loras") from e
    return out


def save_loras(loras):
    write_json(root() / LORAS_FILE, [x.model_dump(mode="json") for x in loras])


def find_lora(loras, name):
    return next((x for x in loras if x.name == name), None)


def record_file(name, rel, family):
    """Record a freshly trained file on its LoRA (persisted)."""
    loras = load_loras()
    x = find_lora(loras, name)
    if x is None:
        return False
    if not any(f.path == rel for f in x.files):
        x.files.append(LoraFile(path=rel, family=parse_model_family(family)))
        save_loras(loras)
    return True


def apply_op(loras, op, name, path=None, family=None):
    """create / delete / add_file on the list, in place. Returns an error
    message or None. (Dataset membership and descriptions are tag /
    describe operations on images.)"""
    x = find_lora(loras, name)
    if op == "create":
        if not is_valid_name(name):
            return (f"bad LoRA name {name!r} (it is the trigger word: letters, digits, - _, "
                    "no spaces)")
        if x:
            return f"LoRA {name!r} exists"
        loras.append(LoRA(name=name))
        return None
    if x is None:
        return f"no LoRA {name!r}"
    if op == "delete":
        loras.remove(x)
        return None
    if op == "add_file":
        try:
            rel = root_rel(path)
        except Exception:
            return "path outside the root"
        fam = family or family_of_path(rel)
        if fam is None:
            return f"{rel}: LoRA files live at loras/<name>/<family>/"
        try:
            entry = LoraFile(path=rel, family=parse_model_family(fam))
        except (ValueError, ValidationError) as e:
            return f"bad LoRA file: {e}"
        if not any(f.path == rel for f in x.files):
            x.files.append(entry)
        return None
    return f"bad op {op!r}"


# ---------- the dataset ----------

def dataset_tag(name):
    return DATASET_PREFIX + name


def dataset_ids(store, name):
    """Unarchived images carrying the LoRA's dataset word, by id."""
    return [i for i in store.with_word(dataset_tag(name)) if not store.is_archived(i)]


def caption(name, description):
    """(caption, warning-or-None): a description that already starts with
    the trigger is a double-prefix smell, not an error."""
    d = (description or "").strip()
    warn = f"description already starts with the trigger {name!r}" if d.startswith(name) else None
    return (f"{name}, {d}" if d else name), warn


# ---------- the dropdown ----------

def menu():
    """{family: [LoRA names that have a file for that family]}."""
    loras = load_loras()
    return {fam.value: [x.name for x in loras if x.files_for(fam)] for fam in ModelFamily}


def resolve(name, family):
    """LoRA name -> absolute path of its newest file for THIS family, or
    None (unknown name, no file for the family, or the file is missing)."""
    if not name:
        return None
    x = find_lora(load_loras(), name)
    if x is None:
        return None
    mine = x.files_for(parse_model_family(family))
    if not mine:
        return None
    q = root() / mine[-1].path
    return q.resolve() if q.is_file() else None
