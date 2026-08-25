"""Qwen-Image-Edit-2511 edit graph with the multiple-angles LoRA on the
model (templates/qwen_edit_camera.json): the camera re-shoot engine.

The source image goes to BOTH text encoders (reference conditioning) and
to VAEEncode (the latent, denoise 1.0) - so output dims follow the staged
source and the viewpoint is the only variable across a sweep.

THE SETTLED RECIPE stacks the 4-step Lightning LoRA after the angle LoRA:
4 steps / cfg 1.0 is both 6x faster AND better than 20 / 2.5 - this is an
EDIT task, and fewer steps at cfg 1 stay closer to the source.
"""
from build_payload import load_template, payload, set_output

DEFAULT_STEPS, DEFAULT_CFG, DEFAULT_STRENGTH = 4, 1.0, 1.0
SLOW_STEPS, SLOW_CFG = 20, 2.5          # without Lightning


def build(image_name, prompt, negative="", seed=0, steps=DEFAULT_STEPS,
          cfg=DEFAULT_CFG, strength=DEFAULT_STRENGTH, lightning=True,
          out_prefix="evolve"):
    pl = load_template("qwen_edit_camera")
    g = pl["prompt"]
    g["img"]["inputs"]["image"] = image_name
    g["pos"]["inputs"]["prompt"] = prompt
    g["neg"]["inputs"]["prompt"] = negative
    g["gene"]["inputs"]["strength_model"] = strength
    g["samp"]["inputs"].update(seed=seed, steps=steps, cfg=cfg)
    if not lightning:
        g.pop("fast", None)
        g["samp"]["inputs"]["model"] = ["gene", 0]
    set_output(g, out_prefix)
    return payload(g)


# model file names, read from the template (single source of truth) - the
# depth_warp spike builds its own graphs from these
_T = load_template("qwen_edit_camera")["prompt"]
MODEL_GGUF = _T["unet"]["inputs"]["unet_name"]
LORA_ANGLES = _T["gene"]["inputs"]["lora_name"]
LORA_LIGHTNING = _T["fast"]["inputs"]["lora_name"]
CLIP_NAME = _T["clip"]["inputs"]["clip_name"]
VAE_NAME = _T["vae"]["inputs"]["vae_name"]
del _T
