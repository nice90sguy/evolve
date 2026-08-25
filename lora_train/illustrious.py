"""Illustrious (SDXL) character LoRA - NOT IMPLEMENTED yet.

The interface is here so the LoRA editor can offer the family and get an
honest refusal. SDXL LoRA training needs kohya sd-scripts (or musubi's
SDXL path once it exists) - none installed. The user's existing
Illustrious-era LoRAs are design references / training-data sources for
Klein, not adapters that load into it."""
from lora_train.common import Trainer

REASON = "Illustrious/SDXL LoRA training is not wired up yet (needs kohya sd-scripts)"


class Trainer(Trainer):
    family = "illustrious"
    label = "Illustrious SDXL (not available)"

    def available(self):
        return False, REASON

    def _train(self, toml, name, version, steps, out_dir, log, opts):
        raise NotImplementedError(REASON)

    def to_comfy(self, native, log=None):
        raise NotImplementedError(REASON)
