"""Train a FLUX.2 Klein character LoRA (a GENETIC IDENTITY) via musubi-tuner.

Usage:  python char_lora_flux.py --name <LoraName> --dataset <dir> [options]

v1 is TRAINING ONLY (rendering + evolve.py integration come next).
Invoking the script IS the say-so to train - there is no lazy mode yet.

- --dataset: a folder of images, each with a same-name .txt caption.
  Every caption must START with the LoRA name (the proven zimage/musubi
  trigger convention; kept for flux until proven unnecessary). Validation
  is hard: missing captions or wrong leading token abort before any GPU
  work, listing the offending files.
- TRAIN on flux2-klein-base-9b (BFL's official trainable variant - the
  distilled klein 9B is inference-only), RENDER later on the installed
  flux-2-klein-9b-fp8: train-on-base -> render-on-distilled is the
  BFL-recommended split (same doctrine as Z-Image De-Turbo -> Turbo).
- Recipe from musubi docs/flux_2.md: flux2_shift timestep sampling,
  adamw8bit, bf16, fp8 DiT+TE, gradient checkpointing, rank 32/alpha 32,
  seed 42; flat 400 steps default (loss is uninformative - judge by
  rendering versions; --steps N for longer runs).
- Versioned, never-overwritten outputs in .\\loras\\<name>\\ (CWD-relative:
  run from your project root), musubi-native format. ComfyUI-format
  conversion is deferred to the evolve integration step.
- Every run mirrors ALL subprocess output (incl. tqdm \\r redraws) to
  loras\\<name>\\logs\\<version>.log - tail -f friendly.
- Asks a running ComfyUI to free VRAM first (best-effort; never launches
  or kills ComfyUI).

Models expected in the ComfyUI tree (overridable):
  diffusion_models\\flux-2-klein-base-9b.safetensors   (gated BFL repo)
  vae\\flux2-vae.safetensors                          (installed)
  text_encoders\\qwen3-8b-flux2\\text_encoder\\model-0000*-of-00004.safetensors
      (BFL originals; the ComfyUI fp8mixed repack is CONFIRMED rejected -
      musubi reads its fp8 layers as half-width tensors)
"""
import argparse
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from char_lora_zimage import (ACCELERATE, IMAGE_EXTS, MUSUBI, MUSUBI_PY,
                              free_comfy_vram, model_path, run_streamed,
                              unix_path)

MODEL_VERSION = "klein-base-9b"


def validate_dataset(dataset, name):
    """Images + same-stem .txt captions, every caption starting with `name`.
    Hard-fails with a full list of offenders before any GPU work."""
    dataset = Path(dataset)
    if not dataset.is_dir():
        sys.exit(f"dataset folder not found: {dataset}")
    pairs, missing, badstart = [], [], []
    for p in sorted(dataset.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        cap = p.with_suffix(".txt")
        if not cap.exists():
            missing.append(p.name)
            continue
        text = cap.read_text(encoding="utf-8").lstrip()
        if not text.startswith(name):
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
        sys.exit("dataset validation failed -\n" + "\n".join(problems))
    if not pairs:
        sys.exit(f"no images found in {dataset}")
    return pairs


def next_version(out_dir, name):
    pat = re.compile(re.escape(name) + r"_v(\d+)_")
    taken = [int(m.group(1)) for p in out_dir.glob(f"{name}_v*")
             if (m := pat.match(p.name))] if out_dir.is_dir() else []
    return max(taken, default=0) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True,
                    help="LoRA name = trigger word; captions must start with it")
    ap.add_argument("--dataset", required=True,
                    help="folder of images + same-name .txt captions")
    ap.add_argument("--steps", type=int, default=400,
                    help="training steps (default 400; judge by renders, not loss)")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=768,
                    help="training resolution (768 = the proven fast tune)")
    ap.add_argument("--num-repeats", type=int, default=10)
    ap.add_argument("--blocks-to-swap", type=int, default=0,
                    help="CPU-offload blocks (max 16 for klein-9b) if VRAM is tight")
    ap.add_argument("--out-dir", default="loras",
                    help="LoRA library root (default .\\loras, CWD-relative)")
    ap.add_argument("--dit", default="flux-2-klein-base-9b.safetensors")
    ap.add_argument("--vae", default="flux2-vae.safetensors")
    ap.add_argument("--te",
                    default="qwen3-8b-flux2/text_encoder/model-00001-of-00004.safetensors",
                    help="BFL Qwen3-8B shards (the ComfyUI fp8mixed repack is "
                         "REJECTED by musubi: fp8 layers load as half-width)")
    args = ap.parse_args()

    pairs = validate_dataset(args.dataset, args.name)
    dit = model_path("diffusion_models", args.dit)
    vae = model_path("vae", args.vae)
    te = model_path("text_encoders", args.te)

    out_dir = Path(args.out_dir).expanduser().resolve() / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    version = f"{args.name}_v{next_version(out_dir, args.name):03d}_" \
              f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    lora_out = out_dir / f"{version}.safetensors"

    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{version}.log"
    print(f"training {version}: {len(pairs)} images, {args.steps} steps, "
          f"{args.resolution}px, rank {args.rank}")
    print(f"raw log: tail -f {unix_path(log_path)}")

    staging = Path(tempfile.mkdtemp(prefix=f"lora_{args.name}_"))
    log = open(log_path, "w", encoding="utf-8")
    try:
        for i, (img, cap) in enumerate(pairs):
            shutil.copy2(img, staging / f"img_{i:03d}{img.suffix.lower()}")
            shutil.copy2(cap, staging / f"img_{i:03d}.txt")
        toml = staging / "dataset.toml"
        toml.write_text(
            f'[general]\nresolution = [{args.resolution}, {args.resolution}]\n'
            f'batch_size = 1\nenable_bucket = true\ncaption_extension = ".txt"\n\n'
            f'[[datasets]]\nimage_directory = "{str(staging).replace(chr(92), "/")}"\n'
            f'num_repeats = {args.num_repeats}\n', encoding="utf-8")

        free_comfy_vram()
        run_streamed("cache-latents", [
            MUSUBI_PY, MUSUBI / "src/musubi_tuner/flux_2_cache_latents.py",
            f"--dataset_config={toml}", f"--vae={vae}",
            f"--model_version={MODEL_VERSION}", "--vae_dtype=bfloat16"], log)
        run_streamed("cache-text-encoder", [
            MUSUBI_PY, MUSUBI / "src/musubi_tuner/flux_2_cache_text_encoder_outputs.py",
            f"--dataset_config={toml}", f"--text_encoder={te}",
            "--batch_size=1", f"--model_version={MODEL_VERSION}",
            "--fp8_text_encoder"], log)

        train_cmd = [
            ACCELERATE, "launch", "--num_cpu_threads_per_process=1",
            "--mixed_precision=bf16",
            MUSUBI / "src/musubi_tuner/flux_2_train_network.py",
            f"--model_version={MODEL_VERSION}",
            f"--dit={dit}", f"--vae={vae}", f"--text_encoder={te}",
            f"--dataset_config={toml}", "--sdpa", "--mixed_precision=bf16",
            "--timestep_sampling=flux2_shift", "--weighting_scheme=none",
            f"--optimizer_type=adamw8bit", f"--learning_rate={args.lr}",
            "--network_module=networks.lora_flux_2",
            f"--network_dim={args.rank}", f"--network_alpha={args.rank}",
            f"--max_train_steps={args.steps}",
            "--max_data_loader_n_workers=2", "--persistent_data_loader_workers",
            "--gradient_checkpointing", "--fp8_base", "--fp8_scaled",
            "--fp8_text_encoder",
            f"--output_dir={out_dir}", f"--output_name={version}", "--seed=42",
        ]
        if args.blocks_to_swap > 0:
            train_cmd.append(f"--blocks_to_swap={args.blocks_to_swap}")
        run_streamed("train", train_cmd, log)

        if not lora_out.exists():
            sys.exit(f"training finished but {lora_out} not found - "
                     f"full log: {log_path}")
    finally:
        log.close()
        shutil.rmtree(staging, ignore_errors=True)

    print(f"trained -> {lora_out}")
    print("(musubi-native format; ComfyUI conversion lands with the evolve "
          "integration. Render on the DISTILLED klein - never on base.)")


if __name__ == "__main__":
    main()
