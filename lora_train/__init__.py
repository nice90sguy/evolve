"""lora_train - character LoRA training, one Trainer per model family,
all with the same interface (lora_train.common.Trainer):

    trainer = lora_train.get_trainer("zimage")
    ok, why = trainer.available()
    native = trainer.train(dataset_dir, name, steps, out_dir, log)
    comfy  = trainer.to_comfy(native)

"from THESE images, for THIS MANY steps" - nothing else is a knob unless
explicitly overridden. Training is never run without the user's say-so
(red line 4): the call site is the say-so.
"""
from lora_train import flux_klein_9b, illustrious, zimage_turbo

TRAINERS = {
    "zimage": zimage_turbo.Trainer,
    "klein": flux_klein_9b.Trainer,
    "illustrious": illustrious.Trainer,
}


def get_trainer(family):
    if family not in TRAINERS:
        raise ValueError(f"no trainer for family {family!r} (have: {', '.join(TRAINERS)})")
    return TRAINERS[family]()
