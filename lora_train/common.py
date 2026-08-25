"""lora_train.common - what every trainer shares: the musubi-tuner venv,
model lookup, dataset validation + staging, version naming, and the
byte-wise streamed subprocess runner (tqdm redraws mirrored to a log).

Trainers are musubi-tuner CLI wrappers; musubi lives in its own venv
(d:/projects/musubi-tuner/venv). uv-venv pythons all show the uv base
interpreter in the process list - never kill by exe path (CLAUDE.md).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from comfy_client import COMFY_DIR

MUSUBI = Path(r"d:\projects\musubi-tuner")
MUSUBI_PY = MUSUBI / "venv" / "Scripts" / "python.exe"
ACCELERATE = MUSUBI / "venv" / "Scripts" / "accelerate.exe"
MODELS = COMFY_DIR / "models"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
CURRENT_PROC = None      # the subprocess run_streamed is driving right now (for abort)


class TrainError(RuntimeError):
    """Validation or subprocess failure (message carries the log path)."""


def model_path(kind, name):
    p = MODELS / kind / name
    if not p.is_file():
        raise TrainError(f"model not found: {p}")
    return p


def unix_path(p):
    """C:\\foo\\bar -> /c/foo/bar, for tail -f from git bash."""
    s = str(p).replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        s = "/" + s[0].lower() + s[2:]
    return s


def validate_dataset(dataset_dir, name):
    """Images + same-stem .txt captions, every caption STARTING with `name`
    (the trigger). Hard-fails with a full list of offenders before any GPU
    work. Returns [(image_path, caption_path)]."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise TrainError(f"dataset folder not found: {dataset_dir}")
    pairs, missing, badstart = [], [], []
    for p in sorted(dataset_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        cap = p.with_suffix(".txt")
        if not cap.exists():
            missing.append(p.name)
            continue
        if not cap.read_text(encoding="utf-8").lstrip().startswith(name):
            badstart.append(cap.name)
            continue
        pairs.append((p, cap))
    problems = []
    if missing:
        problems.append("images without a .txt caption:\n  " + "\n  ".join(missing))
    if badstart:
        problems.append(f"captions that do not START with '{name}':\n  "
                        + "\n  ".join(badstart))
    if problems:
        raise TrainError("dataset validation failed -\n" + "\n".join(problems))
    if not pairs:
        raise TrainError(f"no images found in {dataset_dir}")
    return pairs


def next_version(out_dir, name):
    pat = re.compile(re.escape(name) + r"_v(\d+)_")
    taken = [int(m.group(1)) for p in Path(out_dir).glob(f"{name}_v*")
             if (m := pat.match(p.name))] if Path(out_dir).is_dir() else []
    return max(taken, default=0) + 1


def version_name(out_dir, name):
    return f"{name}_v{next_version(out_dir, name):03d}_{datetime.now():%Y%m%d-%H%M%S}"


def newest_native(out_dir, name):
    """The latest musubi-native weights for `name` (to continue from), or None."""
    pat = re.compile(re.escape(name) + r"_v(\d+)_")
    cands = [p for p in Path(out_dir).glob(f"{name}_v*.safetensors")
             if pat.match(p.name) and not p.name.endswith("_comfy.safetensors")] \
        if Path(out_dir).is_dir() else []
    return max(cands, key=lambda p: int(pat.match(p.name).group(1))) if cands else None


def stage_dataset(pairs, name, resolution, repeats):
    """Copy the pairs into a temp dir with a musubi dataset.toml. Returns
    (staging_dir, toml_path); caller removes the dir."""
    staging = Path(tempfile.mkdtemp(prefix=f"lora_{name}_"))
    for i, (img, cap) in enumerate(pairs):
        shutil.copy2(img, staging / f"img_{i:03d}{img.suffix.lower()}")
        shutil.copy2(cap, staging / f"img_{i:03d}.txt")
    toml = staging / "dataset.toml"
    toml.write_text(
        f'[general]\nresolution = [{resolution}, {resolution}]\n'
        f'batch_size = 1\nenable_bucket = true\ncaption_extension = ".txt"\n\n'
        f'[[datasets]]\nimage_directory = "{str(staging).replace(chr(92), "/")}"\n'
        f'num_repeats = {repeats}\n', encoding="utf-8")
    return staging, toml


def run_streamed(label, cmd, log, live=None, cwd=MUSUBI):
    """Run a subprocess, mirroring EVERYTHING (including tqdm \\r redraws,
    read byte-wise) to the run's log file. On a real console (isatty) the
    \\r redraws are echoed as a live-updating progress line, like native
    tqdm; when piped, only full \\n lines go to stdout. Raises TrainError."""
    global CURRENT_PROC
    header = f"[{label}] {' '.join(str(c) for c in cmd[:2])} ..."
    print(header)
    log.write(header + "\n")
    if live is None:
        live = sys.stdout.isatty()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, cwd=str(cwd), env=env)
    CURRENT_PROC = proc
    buf, redrawing = bytearray(), False
    pending = None   # text before a bare \r: line if \n follows (CRLF), else a tqdm redraw

    def emit(text, as_line):
        nonlocal redrawing
        if as_line:
            if redrawing:
                sys.stdout.write("\n")
                redrawing = False
            print(f"  {text}")
        elif live:
            sys.stdout.write(f"\r  {text}\x1b[K")
            sys.stdout.flush()
            redrawing = True

    for byte in iter(lambda: proc.stdout.read(1), b""):
        if pending is not None:
            if byte == b"\n":
                emit(pending, True)
                pending = None
                continue
            emit(pending, False)
            pending = None
        if byte in (b"\r", b"\n"):
            if buf:
                text = buf.decode("utf-8", "replace")
                log.write(text + "\n")
                log.flush()
                if byte == b"\r":
                    pending = text
                else:
                    emit(text, True)
                buf.clear()
        else:
            buf += byte
    if pending is not None:
        emit(pending, False)
    if redrawing:
        sys.stdout.write("\n")
        sys.stdout.flush()
    proc.wait()
    CURRENT_PROC = None
    if proc.returncode != 0:
        raise TrainError(f"{label} failed with exit code {proc.returncode} - full log: {log.name}")


def convert_lora(native, comfy, log):
    """musubi convert_lora --target other: native -> ComfyUI format."""
    run_streamed("convert", [
        MUSUBI_PY, MUSUBI / "src/musubi_tuner/convert_lora.py",
        "--input", native, "--output", comfy, "--target", "other"], log)
    if not comfy.exists():
        raise TrainError(f"conversion finished but {comfy} not found")
    return comfy


class Trainer:
    """The per-family interface. Subclasses set `family`, `label`,
    `defaults` and implement `_train` + `to_comfy`."""
    family = ""
    label = ""
    defaults = {"rank": 32, "lr": 1e-4, "resolution": 768, "repeats": 10, "steps": 400}

    def available(self):
        """(True, "") or (False, reason) - whether this trainer can run here."""
        if not MUSUBI_PY.is_file():
            return False, f"musubi-tuner venv not found at {MUSUBI_PY}"
        return True, ""

    def train(self, dataset_dir, name, steps, out_dir, log, **overrides):
        """Train `name` from the captioned images in dataset_dir for `steps`
        steps; weights land in out_dir/<name>_vNNN_<ts>.safetensors (musubi-
        native). `log` is an open text file every subprocess mirrors into.
        Returns the native weights path."""
        ok, why = self.available()
        if not ok:
            raise TrainError(why)
        pairs = validate_dataset(dataset_dir, name)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        opts = dict(self.defaults)
        opts.update({k: v for k, v in overrides.items() if v is not None})
        version = version_name(out_dir, name)
        print(f"training {version} [{self.label}]: {len(pairs)} images, {steps} steps, "
              f"{opts['resolution']}px, rank {opts['rank']}")
        staging, toml = stage_dataset(pairs, name, opts["resolution"], opts["repeats"])
        try:
            native = self._train(toml, name, version, steps, out_dir, log, opts)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        if not native.exists():
            raise TrainError(f"training finished but {native} not found - full log: {log.name}")
        return native

    def _train(self, toml, name, version, steps, out_dir, log, opts):
        raise NotImplementedError

    def to_comfy(self, native, log=None):
        """Native -> ComfyUI-loadable file beside it (cached: returns the
        existing _comfy file if present)."""
        raise NotImplementedError

    def comfy_path(self, native):
        native = Path(native)
        return native.with_name(native.stem + "_comfy.safetensors")


def detect_family(path):
    """Which model a .safetensors LoRA belongs to, from its OWN metadata
    (musubi writes ss_network_module / ss_base_model_version) with a key-
    name sniff as fallback. Returns a ModelFamily or None. Used only to migrate
    files into loras/<name>/<family>/ - at use time the directory and the
    loras.json entry are the authority."""
    from model_family import ModelFamily
    from safetensors import safe_open
    with safe_open(str(path), "np") as f:      # metadata + key names only, no torch
        md = f.metadata() or {}
        keys = list(f.keys())
    mod = (md.get("ss_network_module") or "").lower()
    base = (md.get("ss_base_model_version") or md.get("modelspec.architecture") or "").lower()
    if "flux_2" in mod or "flux_2" in base or "flux.2" in base:
        return ModelFamily.KLEIN
    if "zimage" in mod or "z_image" in base or "zimage" in base:
        return ModelFamily.ZIMAGE
    if "sdxl" in base or "stable-diffusion-xl" in base:
        return ModelFamily.ILLUSTRIOUS
    sample = " ".join(keys[:50])
    if "double_blocks" in sample and "img_attn" in sample:
        return ModelFamily.KLEIN
    if "lora_unet_layers" in sample or "diffusion_model.layers" in sample:
        return ModelFamily.ZIMAGE
    if "input_blocks" in sample or "lora_unet_input_blocks" in sample:
        return ModelFamily.ILLUSTRIOUS
    return None
