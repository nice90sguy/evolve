"""Z-Image character LoRA (the user's photoreal line).

TRAIN on ostris De-Turbo (`z_image_de_turbo_v1_bf16`), RENDER on plain
Turbo - never the other way round: musubi's Base training loses likeness
(kohya-ss/musubi-tuner#908, confirmed live 2026-08-18) and inferring on
De-Turbo gave bad quality. Continues from the newest native version in
out_dir when one exists. Recipe: shift/none/2.0, adamw8bit, fp8, grad-ckpt,
rank 32, 768px (the user's intentional fast tune), flat 400 steps -
training loss is uninformative for LoRA convergence; judge by rendering.
"""
import sys

from lora_train.common import (ACCELERATE, MUSUBI, MUSUBI_PY, Trainer,
                               convert_lora, model_path, newest_native,
                               run_streamed)

DIT = "z_image_de_turbo_v1_bf16.safetensors"
VAE = "ae.safetensors"
TEXT_ENCODER = "qwen_3_4b.safetensors"


class Trainer(Trainer):
    family = "zimage"
    label = "Z-Image (De-Turbo train / Turbo render)"
    defaults = {"rank": 32, "lr": 1e-4, "resolution": 768, "repeats": 10,
                "steps": 400, "fp8": True, "continue_from_latest": True}

    def available(self):
        ok, why = super().available()
        if not ok:
            return ok, why
        try:
            model_path("diffusion_models", DIT)
        except Exception as e:
            return False, str(e)
        return True, ""

    def _train(self, toml, name, version, steps, out_dir, log, opts):
        dit = model_path("diffusion_models", DIT)
        vae = model_path("vae", VAE)
        te = model_path("text_encoders", TEXT_ENCODER)
        prev = newest_native(out_dir, name) if opts["continue_from_latest"] else None
        print("continuing from " + prev.name if prev else "from scratch")
        run_streamed("cache-latents", [
            MUSUBI_PY, MUSUBI / "src/musubi_tuner/zimage_cache_latents.py",
            f"--dataset_config={toml}", f"--vae={vae}"], log)
        te_cmd = [MUSUBI_PY, MUSUBI / "src/musubi_tuner/zimage_cache_text_encoder_outputs.py",
                  f"--dataset_config={toml}", f"--text_encoder={te}", "--batch_size=1"]
        if opts["fp8"]:
            te_cmd.append("--fp8_llm")
        run_streamed("cache-text-encoder", te_cmd, log)
        train_cmd = [
            ACCELERATE, "launch", "--num_cpu_threads_per_process=1", "--mixed_precision=bf16",
            MUSUBI / "src/musubi_tuner/zimage_train_network.py",
            f"--dit={dit}", f"--vae={vae}", f"--text_encoder={te}",
            f"--dataset_config={toml}", "--sdpa", "--mixed_precision=bf16",
            "--timestep_sampling=shift", "--weighting_scheme=none", "--discrete_flow_shift=2.0",
            "--optimizer_type=adamw8bit", f"--learning_rate={opts['lr']}",
            "--network_module=networks.lora_zimage",
            f"--network_dim={opts['rank']}", f"--network_alpha={opts['rank']}",
            f"--max_train_steps={steps}", "--max_data_loader_n_workers=2",
            "--persistent_data_loader_workers", "--gradient_checkpointing",
            f"--output_dir={out_dir}", f"--output_name={version}", "--seed=42",
        ]
        if opts["fp8"]:
            train_cmd += ["--fp8_base", "--fp8_scaled", "--fp8_llm"]
        if prev:
            train_cmd.append(f"--network_weights={prev}")
        if steps >= 1000:
            train_cmd.append("--save_every_n_steps=500")
        run_streamed("train", train_cmd, log)
        return out_dir / f"{version}.safetensors"

    def to_comfy(self, native, log=None):
        comfy = self.comfy_path(native)
        if comfy.exists():
            return comfy
        return convert_lora(native, comfy, log or sys.stdout)
