"""training.py - Make LoRA: sync the asset's dataset -> train -> record.

    sync:   dataset list -> <root>/_train/<name>/ (copy-new / delete-dropped /
            skip-unchanged); descriptions become the .txt sidecars AT SYNC
            TIME; the trigger-prefix rule is enforced here, fail-fast.
    train:  lora_train.get_trainer(family).train(...) -> <root>/loras/<name>/
    record: the _comfy file is appended to the asset's loras[].

The user's click IS the say-so (red line 4). One GPU: mutually exclusive
with Generate in both directions (jobs.py).
"""
import shutil
import subprocess
from dataclasses import dataclass

import lora_train
from asset import append_lora
from comfy_client import free_vram_quietly
from lora_train import common
from lora_train.common import unix_path
from project import root


@dataclass
class TrainConfig:
    asset: str
    family: str = "zimage"
    steps: int = None        # None = the trainer's default (400)


def train_dir(name):
    return root() / "_train" / name


def log_path(name):
    return root() / "_train" / f"{name}.log"


def sync_dataset(a):
    """Asset dict -> staged dataset dir. File names are the source path
    mangled ('/' -> '__'): deterministic, so reordering the list never
    churns the dir. ValueError on a bad dataset."""
    name = a["name"]
    bad = [e["path"] for e in a["dataset"]
           if not (e.get("description") or "").startswith(name)]
    if bad:
        raise ValueError("captions must START with the trigger "
                         f"{name!r}; fix: " + ", ".join(bad[:8]))
    missing = [e["path"] for e in a["dataset"] if not (root() / e["path"]).is_file()]
    if missing:
        raise ValueError("missing files: " + ", ".join(missing[:8]))
    if not a["dataset"]:
        raise ValueError("empty dataset")
    ds = train_dir(name)
    ds.mkdir(parents=True, exist_ok=True)
    want = {}
    for e in a["dataset"]:
        stem = e["path"].replace("/", "__")
        stem = stem[:-4] if stem.endswith(".png") else stem
        want[stem] = e
    for f in ds.iterdir():                    # delete-dropped
        if f.stem not in want:
            f.unlink()
    n_copied = 0
    for stem, e in want.items():
        src, dst = root() / e["path"], ds / (stem + ".png")
        st = src.stat()
        if not (dst.exists() and dst.stat().st_size == st.st_size
                and int(dst.stat().st_mtime) == int(st.st_mtime)):
            shutil.copy2(src, dst)
            n_copied += 1
        (ds / (stem + ".txt")).write_text(e["description"], encoding="utf-8")
    print(f"sync {name}: {len(want)} images ({n_copied} copied)")
    return ds


def do_training(cfg):
    """Blocking: runs the whole pipeline (call from a worker thread).
    Returns the root-relative path of the new ComfyUI-format LoRA."""
    trainer = lora_train.get_trainer(cfg.family)
    ok, why = trainer.available()
    if not ok:
        raise common.TrainError(why)
    ds = train_dir(cfg.asset)
    out = root() / "loras" / cfg.asset
    lp = log_path(cfg.asset)
    print(f"raw log: tail -f {unix_path(lp)}")
    free_vram_quietly()
    with lp.open("w", encoding="utf-8", errors="replace") as log:
        native = trainer.train(ds, cfg.asset, cfg.steps or trainer.defaults["steps"],
                               out, log)
        comfy = trainer.to_comfy(native, log)
    rel = comfy.relative_to(root()).as_posix()
    append_lora(cfg.asset, rel)
    print(f"training done: {comfy.name} -> asset {cfg.asset}")
    return rel


def abort_training():
    """Kill the current trainer subprocess TREE (taskkill /T): killing only
    the parent would orphan the GPU process. Never kill by exe path - uv
    venvs all share one interpreter image."""
    proc = common.CURRENT_PROC
    if proc is None or proc.poll() is not None:
        return False
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    return True
