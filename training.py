"""training.py - Make LoRA: sync the LoRA's dataset -> train -> record.

    sync:   images carrying `lora_dataset_<name>` (unarchived) ->
            <root>/_train/<name>/<id>.png + <id>.txt (caption = trigger
            prefixed to the image's description, AT SYNC TIME);
            copy-new / delete-dropped / skip-unchanged.
    train:  lora_train.get_trainer(family).train(...) ->
            <root>/loras/<name>/<family>/   (a LoRA is specific to its model)
    record: the _comfy file is recorded on the LoRA WITH its family.

TrainConfig is a validated pydantic model: an unknown or untrainable family
is refused before anything runs. The user's click IS the say-so (red line
4). One GPU: mutually exclusive with Generate in both directions (jobs.py).
"""
import shutil
import subprocess
from typing import Optional

from pydantic import BaseModel, Field, field_validator

import lora_train
from lora import caption, dataset_ids, lora_dir, record_file
from comfy_client import free_vram_quietly
from model_family import ModelFamily, family_info
from lora_train import common
from lora_train.common import unix_path
from project import is_valid_name, root


class TrainConfig(BaseModel):
    name: str                       # the LoRA (= trigger word)
    family: ModelFamily = ModelFamily.ZIMAGE
    steps: Optional[int] = Field(None, gt=0, le=100_000)   # None = the trainer's default

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v):
        if not is_valid_name(v):          # pydantic `pattern` is a search, not a full match
            raise ValueError(f"bad LoRA name {v!r}")
        return v

    @field_validator("family")
    @classmethod
    def _trainable(cls, v):
        if not family_info(v).trainable:
            raise ValueError(f"{v} is not trainable here")
        return v


def train_dir(name):
    return root() / "_train" / name


def log_path(name):
    return root() / "_train" / f"{name}.log"


def sync_dataset(store, name):
    """Dataset word -> staged dataset dir. ValueError if empty. Warnings
    (double-prefix captions) are printed, not fatal."""
    ids = dataset_ids(store, name)
    if not ids:
        raise ValueError(f"empty dataset: no unarchived image carries lora_dataset_{name}")
    ds = train_dir(name)
    ds.mkdir(parents=True, exist_ok=True)
    want = {str(i): i for i in ids}
    for f in ds.iterdir():                    # delete-dropped
        if f.stem not in want:
            f.unlink()
    n_copied = 0
    for stem, i in want.items():
        src, dst = store.path(i), ds / (stem + ".png")
        st = src.stat()
        if not (dst.exists() and dst.stat().st_size == st.st_size
                and int(dst.stat().st_mtime) == int(st.st_mtime)):
            shutil.copy2(src, dst)
            n_copied += 1
        text, warn = caption(name, store.images[i].get("description"))
        if warn:
            print(f"#{i}: {warn}")
        (ds / (stem + ".txt")).write_text(text, encoding="utf-8")
    print(f"sync {name}: {len(want)} images ({n_copied} copied)")
    return ds


def do_training(cfg):
    """Blocking: runs the whole pipeline (call from a worker thread).
    Returns the root-relative path of the new ComfyUI-format LoRA."""
    trainer = lora_train.get_trainer(cfg.family)
    ok, why = trainer.available()
    if not ok:
        raise common.TrainError(why)
    ds = train_dir(cfg.name)
    out = lora_dir(cfg.name, cfg.family)
    lp = log_path(cfg.name)
    print(f"raw log: tail -f {unix_path(lp)}")
    free_vram_quietly()
    with lp.open("w", encoding="utf-8", errors="replace") as log:
        native = trainer.train(ds, cfg.name, cfg.steps or trainer.defaults["steps"],
                               out, log)
        comfy = trainer.to_comfy(native, log)
    rel = comfy.relative_to(root()).as_posix()
    record_file(cfg.name, rel, cfg.family)
    print(f"training done: {comfy.name} -> LoRA {cfg.name} [{cfg.family}]")
    return rel


def abort_training():
    """Kill the current trainer subprocess TREE (taskkill /T)."""
    proc = common.CURRENT_PROC
    if proc is None or proc.poll() is not None:
        return False
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    return True
