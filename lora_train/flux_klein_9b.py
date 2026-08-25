"""FLUX.2 Klein character LoRA.

Trains on `flux-2-klein-base-9b` (BFL's official trainable variant - the
distilled Klein is inference-only; train-on-base -> render-on-distilled is
BFL-blessed, the De-Turbo -> Turbo doctrine again). The base repo is GATED
on HF. TE = BFL's Qwen3-8B shards (the ComfyUI fp8mixed repack is REJECTED
by musubi: fp8 layers load as half-width). Recipe from musubi docs/flux_2.md.

KEY-NAME BUG (2026-08-21): musubi's convert_lora writes double-block
attention keys Flux.1-style (img_attn_qkv); ComfyUI's Klein weights are
img_attn.qkv, so 64/224 keys silently failed to load. fix_flux2_keys()
renames in place after every conversion. Always grep comfyui.log for
`lora key not loaded` after a new LoRA.
"""
import sys

from lora_train.common import (ACCELERATE, MUSUBI, MUSUBI_PY, Trainer,
                               convert_lora, model_path, run_streamed)

MODEL_VERSION = "klein-base-9b"
DIT = "flux-2-klein-base-9b.safetensors"
VAE = "flux2-vae.safetensors"
TEXT_ENCODER = "qwen3-8b-flux2/text_encoder/model-00001-of-00004.safetensors"

FLUX2_KEY_FIX = {"img_attn_qkv": "img_attn.qkv", "img_attn_proj": "img_attn.proj",
                 "txt_attn_qkv": "txt_attn.qkv", "txt_attn_proj": "txt_attn.proj"}


def fix_flux2_keys(comfy):
    """Rewrite a converted LoRA so every key matches ComfyUI's Flux2 naming.
    Idempotent; keeps a .bak of the original the first time."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    with safe_open(str(comfy), "pt") as f:
        meta = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    fixed, n = {}, 0
    for k, v in tensors.items():
        nk = k
        for a, b in FLUX2_KEY_FIX.items():
            if "." + a + "." in nk:
                nk = nk.replace("." + a + ".", "." + b + ".")
                n += 1
        fixed[nk] = v
    if n:
        bak = comfy.with_suffix(comfy.suffix + ".bak")
        if not bak.exists():
            comfy.rename(bak)
        save_file(fixed, str(comfy), metadata=meta)
        print(f"fixed {n} Flux2 LoRA key names in {comfy.name}")
    return n


class Trainer(Trainer):
    family = "klein"
    label = "FLUX.2 Klein 9B (base train / distilled render)"
    defaults = {"rank": 32, "lr": 1e-4, "resolution": 768, "repeats": 10,
                "steps": 400, "blocks_to_swap": 0}

    def available(self):
        ok, why = super().available()
        if not ok:
            return ok, why
        try:
            model_path("diffusion_models", DIT)
            model_path("text_encoders", TEXT_ENCODER)
        except Exception as e:
            return False, f"{e} (the base-9B HF repo is gated - accept the license)"
        return True, ""

    def _train(self, toml, name, version, steps, out_dir, log, opts):
        dit = model_path("diffusion_models", DIT)
        vae = model_path("vae", VAE)
        te = model_path("text_encoders", TEXT_ENCODER)
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
            "--optimizer_type=adamw8bit", f"--learning_rate={opts['lr']}",
            "--network_module=networks.lora_flux_2",
            f"--network_dim={opts['rank']}", f"--network_alpha={opts['rank']}",
            f"--max_train_steps={steps}",
            "--max_data_loader_n_workers=2", "--persistent_data_loader_workers",
            "--gradient_checkpointing", "--fp8_base", "--fp8_scaled",
            "--fp8_text_encoder",
            f"--output_dir={out_dir}", f"--output_name={version}", "--seed=42",
        ]
        if opts["blocks_to_swap"] > 0:
            train_cmd.append(f"--blocks_to_swap={opts['blocks_to_swap']}")
        run_streamed("train", train_cmd, log)
        return out_dir / f"{version}.safetensors"

    def to_comfy(self, native, log=None):
        comfy = self.comfy_path(native)
        if not comfy.exists():
            convert_lora(native, comfy, log or sys.stdout)
        fix_flux2_keys(comfy)
        return comfy
