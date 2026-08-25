"""Create, refine, and/or render with per-character Z-Image LoRAs (musubi-tuner).

Usage:  python char_lora.py <char> [--prompt "..."] [options]

One script per character, two modes selected by --prompt:
- NON-EMPTY --prompt: inference only, never trains. Renders through ComfyUI
  with the character's current LoRA (errors if none trained yet).
- EMPTY/omitted --prompt: trains, but only if the dataset (images/captions)
  or training params changed since the last run. Change detection uses a
  manifest (file names + sizes + mtimes + params) stored with each version,
  so adds, deletes, renames, and stale-dated copies are all caught.

Training targets ostris's De-Turbo DiT (musubi's docs recommend Base, but
its Base training loses likeness — musubi-tuner#908, verified here) and by
default CONTINUES from the latest version's weights. Default 400 steps per run
(user-settled: training loss is flat/uninformative from step 1, so
convergence is judged by rendering, not by loss; --steps N overrides for
one run). --from-scratch retrains fresh; --force-train retrains even with
no changes.
Every version is kept: loras/<char>/<char>_vNNN_<timestamp>.safetensors
(musubi-native, needed to continue training) + _comfy.safetensors (what
ComfyUI loads) + .manifest.json. Nothing is ever overwritten.

State lives under ./loras/<char>/ relative to the CWD — run from your game's
project root. First run needs --images pointing at the dataset folder
(image files + same-name .txt captions; captions should contain the trigger
word). The trigger defaults to the character name itself — name characters
after the trigger in your captions (e.g. "akaslizabeth"), or set --trigger.
Settings persist in loras/<char>/config.json; CLI flags override and
update them.

Inference mirrors the proven Z-Image graph (KSampler euler/sgm_uniform,
ApplyTrainedLora from comfyUI-Realtime-Lora). Renders go into --out-dir
(default: current directory) named <prefix>NNNNN.png — canonical 5-digit
zero-padded, --out-prefix default "img" (e.g. img00032.png) — numbering
continues past existing files, NOTHING is ever overwritten. --num-images
renders a batch in one call (latent batch size), all images are saved.
Seed printed for reuse; exact graph written to last_payload.json.
Renders on plain Turbo, 20 steps / cfg 2 / strength 1.0 (the good-era
recipe recovered from Feb-10 PNG forensics): TRAIN on De-Turbo, RENDER on
Turbo — the de-distilled surrogate is for training only.

Runs in the ComfyUI venv; training subprocesses run in the musubi-tuner
venv. ComfyUI must be up for inference; training only asks it (best-effort)
to free VRAM first.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from pose_from_char import (API, COMFY, SCRATCH_PREFIX, next_index, queue,
                            save_render, wait)

# Windows pipes default to cp1252 and block-buffering; musubi/tqdm output is
# full of unicode, and `> file` / `| tee` must receive lines as they happen.
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

MUSUBI = Path(r"d:\projects\musubi-tuner")
MUSUBI_PY = MUSUBI / "venv" / "Scripts" / "python.exe"
ACCELERATE = MUSUBI / "venv" / "Scripts" / "accelerate.exe"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

TRAIN_DEFAULTS = {
    # De-Turbo, NOT Base: musubi's Z-Image Base training underperforms badly
    # (verified live — 400-step Base LoRA lost the likeness entirely, matching
    # kohya-ss/musubi-tuner#908); De-Turbo is the user's proven recipe.
    "dit": "z_image_de_turbo_v1_bf16.safetensors",
    "vae": "ae.safetensors",
    "text_encoder": "qwen_3_4b.safetensors",
    "rank": 32,
    "lr": 1e-4,
    "resolution": 768,     # user's intentional tune (matches their fast node-era bakes)
    "steps": 400,          # flat default, user decision: loss is uninformative, judge by renders
    "fp8": True,                                 # fp8 DiT+TE: safe alongside a loaded ComfyUI
}
# TRAIN on De-Turbo, RENDER on plain Turbo: De-Turbo is the de-distilled
# training surrogate; Turbo is the distillation-optimized inference model.
# Values recovered from the good-era renders (Feb 10 PNG-embedded graphs:
# turbo, 20 steps, cfg 2.0, strength 1.0). NB the installed turbo file has
# no "_v1" in its name.
INFER_DEFAULTS = {
    "dit": "z_image_turbo_bf16.safetensors",
    "steps": 20,
    "cfg": 2.0,
    "width": 1024,
    "height": 1024,
    "strength": 1.0,
}
# Params whose change alone triggers a retrain (matches the realtime node's hash).
STALENESS_PARAMS = ("dit", "rank", "lr", "resolution", "steps")


def model_path(kind, name):
    p = COMFY / "models" / kind / name
    if not p.exists():
        sys.exit(f"model not found: {p}")
    return p


def load_config(char_dir, char, args):
    """Load loras/<char>/config.json, create on first run, apply CLI overrides."""
    cfg_file = char_dir / "config.json"
    if cfg_file.exists():
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    else:
        if not args.images:
            sys.exit(f"no config for '{char}' yet - first run needs --images <dataset folder>")
        cfg = {"images": None, "trigger": char, "current_version": None,
               "train": dict(TRAIN_DEFAULTS), "infer": dict(INFER_DEFAULTS)}
    for key, val in TRAIN_DEFAULTS.items():      # fill keys added after a config was created
        cfg["train"].setdefault(key, val)
    for key, val in INFER_DEFAULTS.items():
        cfg["infer"].setdefault(key, val)
    if args.images:
        cfg["images"] = str(Path(args.images).expanduser().resolve())
    if args.trigger:
        cfg["trigger"] = args.trigger
    for key in ("rank", "lr", "resolution"):
        val = getattr(args, key)
        if val is not None:
            cfg["train"][key] = val
    for key in ("steps", "cfg", "width", "height", "strength"):
        val = getattr(args, f"infer_{key}", None) if key in ("steps", "cfg") else getattr(args, key)
        if val is not None:
            cfg["infer"][key] = val
    if args.infer_dit:
        cfg["infer"]["dit"] = args.infer_dit
    char_dir.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def scan_dataset(images_dir, trigger):
    """Return ({filename: [size, mtime_ns]}, [(img_path, caption_text)]).
    The file map covers images AND caption .txt files, so caption edits count."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        sys.exit(f"dataset folder not found: {images_dir}")
    files, pairs = {}, []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            st = p.stat()
            files[p.name] = [st.st_size, st.st_mtime_ns]
            cap_file = p.with_suffix(".txt")
            if cap_file.exists():
                cst = cap_file.stat()
                files[cap_file.name] = [cst.st_size, cst.st_mtime_ns]
                caption = cap_file.read_text(encoding="utf-8").strip()
                if trigger not in caption:
                    print(f"WARNING: caption {cap_file.name} lacks trigger '{trigger}'")
            else:
                caption = trigger
                print(f"WARNING: no caption for {p.name}, using bare trigger")
            pairs.append((p, caption))
    if not pairs:
        sys.exit(f"no images found in {images_dir}")
    return files, pairs


def load_manifest(char_dir, version_name):
    mf = char_dir / f"{version_name}.manifest.json"
    return json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else None


def unix_path(p):
    """C:\\foo\\bar -> /c/foo/bar, for tail -f from git bash."""
    s = str(p).replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        s = "/" + s[0].lower() + s[2:]
    return s


def run_streamed(label, cmd, log, live=None):
    """Run a subprocess, mirroring EVERYTHING (including tqdm \\r redraws,
    read byte-wise) to the run's log file. On a real console (isatty) the
    \\r redraws are echoed as a live-updating progress line, like native
    tqdm; when piped, only full \\n lines go to stdout."""
    header = f"[{label}] {' '.join(str(c) for c in cmd[:2])} ..."
    print(header)
    log.write(header + "\n")
    if live is None:
        live = sys.stdout.isatty()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, cwd=str(MUSUBI), env=env)
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
    if proc.returncode != 0:
        sys.exit(f"{label} failed with exit code {proc.returncode} - full log: {log.name}")


def free_comfy_vram():
    """Best-effort: ask a running ComfyUI to unload models before training."""
    import urllib.request
    try:
        req = urllib.request.Request(API + "/free",
                                     json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                     {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
        print("asked ComfyUI to free VRAM")
    except Exception:
        pass


def train(char, char_dir, cfg, args):
    t = cfg["train"]
    files, pairs = scan_dataset(cfg["images"], cfg["trigger"])
    params = {k: t[k] for k in STALENESS_PARAMS}

    latest = load_manifest(char_dir, cfg["current_version"]) if cfg["current_version"] else None
    if latest and not args.force_train:
        if latest["files"] == files and latest["params"] == params:
            print(f"up to date: nothing changed since {cfg['current_version']} - not training")
            print("(use --force-train to retrain anyway)")
            return

    continuing = latest is not None and not args.from_scratch
    steps = args.steps if args.steps else t["steps"]

    vnum = (int(latest["version"].split("_")[-2][1:]) + 1) if latest else 1
    version_name = f"{char}_v{vnum:03d}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    lora_native = char_dir / f"{version_name}.safetensors"
    lora_comfy = char_dir / f"{version_name}_comfy.safetensors"

    mode = "continue from " + cfg["current_version"] if continuing else "from scratch"
    print(f"training {version_name}: {len(pairs)} images, {steps} steps, {mode}")

    log_dir = char_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{version_name}.log"
    print(f"raw log: tail -f {unix_path(log_path)}")

    dit = model_path("diffusion_models", t["dit"])
    vae = model_path("vae", t["vae"])
    te = model_path("text_encoders", t["text_encoder"])

    staging = Path(tempfile.mkdtemp(prefix=f"lora_{char}_"))
    log = open(log_path, "w", encoding="utf-8")
    try:
        for i, (img, caption) in enumerate(pairs):
            shutil.copy2(img, staging / f"img_{i:03d}{img.suffix.lower()}")
            (staging / f"img_{i:03d}.txt").write_text(caption, encoding="utf-8")
        toml = staging / "dataset.toml"
        toml.write_text(
            f'[general]\nresolution = [{t["resolution"]}, {t["resolution"]}]\n'
            f'batch_size = 1\nenable_bucket = true\ncaption_extension = ".txt"\n\n'
            f'[[datasets]]\nimage_directory = "{str(staging).replace(chr(92), "/")}"\n'
            f'num_repeats = 10\n', encoding="utf-8")

        free_comfy_vram()
        run_streamed("cache-latents", [
            MUSUBI_PY, MUSUBI / "src/musubi_tuner/zimage_cache_latents.py",
            f"--dataset_config={toml}", f"--vae={vae}"], log)
        te_cmd = [MUSUBI_PY, MUSUBI / "src/musubi_tuner/zimage_cache_text_encoder_outputs.py",
                  f"--dataset_config={toml}", f"--text_encoder={te}", "--batch_size=1"]
        if t["fp8"]:
            te_cmd.append("--fp8_llm")
        run_streamed("cache-text-encoder", te_cmd, log)

        train_cmd = [
            ACCELERATE, "launch", "--num_cpu_threads_per_process=1", "--mixed_precision=bf16",
            MUSUBI / "src/musubi_tuner/zimage_train_network.py",
            f"--dit={dit}", f"--vae={vae}", f"--text_encoder={te}",
            f"--dataset_config={toml}", "--sdpa", "--mixed_precision=bf16",
            "--timestep_sampling=shift", "--weighting_scheme=none", "--discrete_flow_shift=2.0",
            "--optimizer_type=adamw8bit", f"--learning_rate={t['lr']}",
            "--network_module=networks.lora_zimage",
            f"--network_dim={t['rank']}", f"--network_alpha={t['rank']}",
            f"--max_train_steps={steps}", "--max_data_loader_n_workers=2",
            "--persistent_data_loader_workers", "--gradient_checkpointing",
            f"--output_dir={char_dir}", f"--output_name={version_name}", "--seed=42",
        ]
        if t["fp8"]:
            train_cmd += ["--fp8_base", "--fp8_scaled", "--fp8_llm"]
        if continuing:
            train_cmd.append(f"--network_weights={char_dir / (cfg['current_version'] + '.safetensors')}")
        if steps >= 1000:
            train_cmd.append("--save_every_n_steps=500")
        run_streamed("train", train_cmd, log)

        if not lora_native.exists():
            sys.exit(f"training finished but {lora_native} not found")
        run_streamed("convert", [
            MUSUBI_PY, MUSUBI / "src/musubi_tuner/convert_lora.py",
            "--input", lora_native, "--output", lora_comfy, "--target", "other"], log)
    finally:
        log.close()
        shutil.rmtree(staging, ignore_errors=True)

    manifest = {"version": version_name, "parent": cfg["current_version"] if continuing else None,
                "files": files, "params": params, "steps": steps, "images": len(pairs)}
    (char_dir / f"{version_name}.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    cfg["current_version"] = version_name
    (char_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"trained -> {lora_comfy}")


def infer(char, char_dir, cfg, args):
    version = args.lora_version or cfg["current_version"]
    if not version:
        sys.exit(f"no LoRA trained for '{char}' yet - run without --prompt to train")
    if args.lora_version:  # allow bare "v002" by matching the prefix
        matches = sorted(char_dir.glob(f"{char}_{version}*_comfy.safetensors")) \
            if not (char_dir / f"{version}_comfy.safetensors").exists() else \
            [char_dir / f"{version}_comfy.safetensors"]
        if not matches:
            sys.exit(f"no LoRA version matching '{version}' in {char_dir}")
        lora_comfy = matches[-1]
    else:
        lora_comfy = char_dir / f"{version}_comfy.safetensors"
    if not lora_comfy.exists():
        sys.exit(f"LoRA file missing: {lora_comfy}")
    if cfg["trigger"] not in args.prompt:
        print(f"WARNING: prompt lacks trigger word '{cfg['trigger']}'")

    inf = cfg["infer"]
    model_path("diffusion_models", inf["dit"])   # existence check
    seed = args.seed if args.seed is not None else random.randint(1, 999_999_999_999)
    out_dir = Path(args.out_dir).expanduser().resolve()

    graph = {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": inf["dit"], "weight_dtype": "default"}},
        "lora": {"class_type": "ApplyTrainedLora",
                 "inputs": {"strength": inf["strength"], "model": ["unet", 0],
                            "lora_path": str(lora_comfy)}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": cfg["train"]["text_encoder"], "type": "lumina2",
                            "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": cfg["train"]["vae"]}},
        "pos": {"class_type": "CLIPTextEncode",
                "inputs": {"text": args.prompt, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["clip", 0]}},
        "latent": {"class_type": "EmptySD3LatentImage",
                   "inputs": {"width": inf["width"], "height": inf["height"],
                              "batch_size": args.num_images}},
        "samp": {"class_type": "KSampler",
                 "inputs": {"seed": seed, "steps": inf["steps"], "cfg": inf["cfg"],
                            "sampler_name": "euler", "scheduler": "sgm_uniform", "denoise": 1,
                            "model": ["lora", 0], "positive": ["pos", 0], "negative": ["neg", 0],
                            "latent_image": ["latent", 0]}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["samp", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["dec", 0],
                            "filename_prefix": f"{SCRATCH_PREFIX}/{args.out_prefix}"}},
    }
    pid = queue({"client_id": "char_lora", "prompt": graph})
    print(f"queued {lora_comfy.name} -> {pid} (seed {seed}, {inf['steps']} steps, "
          f"cfg {inf['cfg']}, {args.num_images} image(s))")
    srcs = wait(pid)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = next_index(out_dir, args.out_prefix)
    for src in srcs:
        dest = save_render(src, out_dir, args.out_prefix, n)
        n += 1
        print(f"rendered -> {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("char", help="character name; state lives in ./loras/<char>/")
    ap.add_argument("--prompt", default="",
                    help="non-empty: render this prompt (never trains); empty: train if stale")
    ap.add_argument("--images", default=None, help="dataset folder (required on first run)")
    ap.add_argument("--trigger", default=None,
                    help="trigger word (default: the character name itself)")
    # training
    ap.add_argument("--force-train", action="store_true", help="train even if nothing changed")
    ap.add_argument("--from-scratch", action="store_true",
                    help="don't continue from the latest version")
    ap.add_argument("--steps", type=int, default=None,
                    help="training steps for THIS run (default 400, not persisted)")
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--resolution", type=int, default=None)
    # inference
    ap.add_argument("--out-dir", default=".",
                    help="directory renders are written into (default: current directory)")
    ap.add_argument("--out-prefix", default="img",
                    help='render filename prefix; files are <prefix>NNNNN.png (default "img")')
    ap.add_argument("--num-images", type=int, default=1,
                    help="images per render call (latent batch size)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--infer-steps", type=int, default=None, dest="infer_steps")
    ap.add_argument("--infer-cfg", type=float, default=None, dest="infer_cfg")
    ap.add_argument("--infer-dit", default=None,
                    help="DiT for rendering (e.g. z_image_de_turbo_v1_bf16.safetensors)")
    ap.add_argument("--strength", type=float, default=None, help="LoRA strength")
    ap.add_argument("--lora-version", default=None, help='render with an older bake, e.g. "v002"')
    args = ap.parse_args()

    char_dir = Path.cwd() / "loras" / args.char
    cfg = load_config(char_dir, args.char, args)
    if args.prompt.strip():
        infer(args.char, char_dir, cfg, args)
    else:
        train(args.char, char_dir, cfg, args)


if __name__ == "__main__":
    main()
