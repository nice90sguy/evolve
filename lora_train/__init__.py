"""lora_train - character LoRA training, one Trainer per model family,
all with the same interface (lora_train.common.Trainer):

    trainer = lora_train.get_trainer(ModelFamily.ZIMAGE)
    ok, why = trainer.available()
    native = trainer.train(dataset_dir, name, steps, out_dir, log)
    comfy  = trainer.to_comfy(native)

"from THESE images, for THIS MANY steps" - nothing else is a knob unless
explicitly overridden. Training is never run without the user's say-so
(red line 4): the call site is the say-so. lora_train.common.detect_family
reads a .safetensors file's own metadata to say which family it is.
"""
from model_family import ModelFamily, parse_model_family
from lora_train import flux_klein_9b, illustrious, zimage_turbo

TRAINERS = {
    ModelFamily.ZIMAGE: zimage_turbo.Trainer,
    ModelFamily.KLEIN: flux_klein_9b.Trainer,
    ModelFamily.ILLUSTRIOUS: illustrious.Trainer,
}


def get_trainer(family):
    return TRAINERS[parse_model_family(family)]()
