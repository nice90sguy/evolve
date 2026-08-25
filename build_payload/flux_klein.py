"""FLUX.2 Klein 9B graph: prompt-only fiat, or 1-4 reference images entering
as ReferenceLatent chains in BOTH positive and negative conditioning, with
IdentityFeatureTransferFinal pinning identity to reference 0.

Sampling is the settled recipe baked into templates/flux_klein_9b.json:
euler via KSamplerSelect + Flux2Scheduler (4 steps) + CFGGuider cfg 1.0 +
SamplerCustomAdvanced; negative = ConditioningZeroOut of the prompt.
"""
from build_payload import attach_lora, load_template, payload, set_output

# preset -> (similarity_floor, softmax_temperature), from
# IdentityFeatureTransferFinal.PRESETS in ComfyUI-Flux2Klein-Enhancer.
# Presets are pure similarity gates - block schedules are identical.
PRESETS = {
    "SOFT_LOCK": (0.5, 0.07),
    "MID_LOCK": (0.2, 0.07),
    "HARD_LOCK": (0.04, 0.025),
}
LOCKS = list(PRESETS)
WHITE_BG_PREFIX = "Solid plain white background. "
DEFAULT_STEPS, DEFAULT_CFG = 4, 1.0


def build(prompt, seed, width, height, refs=(), lock="SOFT_LOCK", vary=0.0,
          lora_path=None, lora_strength=1.0, steps=None, cfg=None,
          identity_refs="0", batch_size=1, out_prefix="evolve"):
    """refs: staged input-dir filenames; refs[0] = identity source (jittered
    by `vary` - additive randn on its latent via KJ InjectNoiseToLatent,
    useful band 0.1-0.3), later refs = extras steering composition.
    No refs = fiat (IFT node removed). identity_refs = the Final node's
    reference_indices string ("0", "0,2", "all")."""
    pl = load_template("flux_klein_9b")
    p = pl["prompt"]
    model_src = "unet"
    if lora_path:
        model_src = attach_lora(p, lora_path, lora_strength)
    pos_src, neg_src = "txt", "zero"
    for i, name in enumerate(refs):
        p[f"img_{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        p[f"enc_{i}"] = {"class_type": "VAEEncode",
                         "inputs": {"pixels": [f"img_{i}", 0], "vae": ["vae", 0]}}
        lat = f"enc_{i}"
        if i == 0 and vary > 0:
            p["jit"] = {"class_type": "InjectNoiseToLatent",
                        "inputs": {"latents": ["enc_0", 0], "strength": 0.0,
                                   "noise": ["enc_0", 0], "normalize": False,
                                   "average": False, "mix_randn_amount": vary,
                                   "seed": seed}}
            lat = "jit"
        p[f"pos_{i}"] = {"class_type": "ReferenceLatent",
                         "inputs": {"conditioning": [pos_src, 0], "latent": [lat, 0]}}
        p[f"neg_{i}"] = {"class_type": "ReferenceLatent",
                         "inputs": {"conditioning": [neg_src, 0], "latent": [lat, 0]}}
        pos_src, neg_src = f"pos_{i}", f"neg_{i}"
    p["guider"]["inputs"]["positive"] = [pos_src, 0]
    p["guider"]["inputs"]["negative"] = [neg_src, 0]
    if refs and lock in PRESETS:
        floor, temp = PRESETS[lock]
        p["ift"]["inputs"].update(model=[model_src, 0], preset=lock,
                                  similarity_floor=floor, softmax_temperature=temp,
                                  reference_index=0, reference_indices=identity_refs)
    else:
        p.pop("ift", None)
        p["guider"]["inputs"]["model"] = [model_src, 0]
    p["sched"]["inputs"].update(width=width, height=height)
    if steps:
        p["sched"]["inputs"]["steps"] = steps
    if cfg:
        p["guider"]["inputs"]["cfg"] = cfg
    p["latent"]["inputs"].update(width=width, height=height, batch_size=batch_size)
    p["txt"]["inputs"]["text"] = prompt
    p["noise"]["inputs"]["noise_seed"] = seed
    set_output(p, out_prefix)
    return payload(p)
