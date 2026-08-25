"""Z-Image Turbo fiat graph - the settled inference recipe (euler /
sgm_uniform, 20 steps, cfg 2, strength 1.0) from templates/zimage_turbo.json.
Model files (z_image_turbo_bf16 + qwen_3_4b lumina2 + ae) live in the
template. RENDER on Turbo, never on De-Turbo (that is a training surrogate)."""
from build_payload import attach_lora, load_template, payload, set_output

DEFAULT_STEPS, DEFAULT_CFG = 20, 2.0


def build(prompt, negative, seed, width, height, steps=DEFAULT_STEPS, cfg=DEFAULT_CFG,
          lora_path=None, lora_strength=1.0, batch_size=1, out_prefix="evolve"):
    pl = load_template("zimage_turbo")
    g = pl["prompt"]
    g["pos"]["inputs"]["text"] = prompt
    g["neg"]["inputs"]["text"] = negative
    g["latent"]["inputs"].update(width=width, height=height, batch_size=batch_size)
    g["samp"]["inputs"].update(seed=seed, steps=steps, cfg=cfg)
    if lora_path:
        g["samp"]["inputs"]["model"] = [attach_lora(g, lora_path, lora_strength), 0]
    set_output(g, out_prefix)
    return payload(g)
