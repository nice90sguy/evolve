"""model_family.py - the MODEL FAMILY, defined once: the closed set of models the
app generates with and trains LoRAs for, and what each one can do.

    ModelFamily.KLEIN       FLUX.2 Klein 9B   references + IFT, LoRAs, trainable
    ModelFamily.ZIMAGE      Z-Image Turbo     fiat only, LoRAs, trainable
    ModelFamily.ILLUSTRIOUS Illustrious SDXL  fiat only, no LoRAs yet, not trainable

A LoRA is specific to its family: files live at
<root>/loras/<name>/<family>/*.safetensors and every dropdown, recipe,
trainer and loras.json entry carries the family explicitly (never inferred from
a filename at use time). parse_model_family() is the one validator everything
(pydantic models, controls, the HTTP layer) goes through.
"""
from dataclasses import dataclass
from enum import Enum

from build_payload import flux_klein, illustrious, zimage


class ModelFamily(str, Enum):
    KLEIN = "klein"
    ZIMAGE = "zimage"
    ILLUSTRIOUS = "illustrious"

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class ModelFamilyInfo:
    label: str
    steps: int
    cfg: float
    references: bool     # ReferenceLatent + identity transfer (Derive)
    lora: bool           # ApplyTrainedLora in the graph
    trainable: bool      # a lora_train Trainer exists and works


MODEL_FAMILY_INFO = {
    ModelFamily.KLEIN: ModelFamilyInfo("Flux2 Klein", flux_klein.DEFAULT_STEPS, flux_klein.DEFAULT_CFG,
                             references=True, lora=True, trainable=True),
    ModelFamily.ZIMAGE: ModelFamilyInfo("Z-Image Turbo", zimage.DEFAULT_STEPS, zimage.DEFAULT_CFG,
                              references=False, lora=True, trainable=True),
    ModelFamily.ILLUSTRIOUS: ModelFamilyInfo("Illustrious (WAI v16)", illustrious.DEFAULT_STEPS,
                                   illustrious.DEFAULT_CFG,
                                   references=False, lora=False, trainable=False),
}
DEFAULT_MODEL_FAMILY = ModelFamily.KLEIN
MODEL_FAMILY_NAMES = [f.value for f in ModelFamily]


def parse_model_family(value, default=None):
    """str | ModelFamily -> ModelFamily. ValueError for anything else (or `default`
    if given)."""
    if isinstance(value, ModelFamily):
        return value
    try:
        return ModelFamily(str(value))
    except ValueError:
        if default is not None:
            return default
        raise ValueError(f"unknown model family {value!r} (one of: {', '.join(MODEL_FAMILY_NAMES)})")


def family_info(family):
    return MODEL_FAMILY_INFO[parse_model_family(family)]


def model_families_for_ui():
    return {f.value: {"label": i.label, "steps": i.steps, "cfg": i.cfg,
                      "references": i.references, "lora": i.lora, "trainable": i.trainable}
            for f, i in MODEL_FAMILY_INFO.items()}
