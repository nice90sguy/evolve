"""Illustrious (SDXL) fiat graph: checkpoint -> KSampler
(euler_ancestral / normal, 28 steps, cfg 6) from templates/illustrious_sdxl.json.
SDXL LoRAs wait for the unified trainer."""
from build_payload import load_template, payload, set_output

DEFAULT_STEPS, DEFAULT_CFG = 28, 6.0


def build(prompt, negative, seed, width, height, steps=DEFAULT_STEPS, cfg=DEFAULT_CFG,
          batch_size=1, out_prefix="evolve"):
    pl = load_template("illustrious_sdxl")
    g = pl["prompt"]
    g["pos"]["inputs"]["text"] = prompt
    g["neg"]["inputs"]["text"] = negative
    g["latent"]["inputs"].update(width=width, height=height, batch_size=batch_size)
    g["samp"]["inputs"].update(seed=seed, steps=steps, cfg=cfg)
    set_output(g, out_prefix)
    return payload(g)
