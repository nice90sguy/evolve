"""Train a character LoRA from a captioned folder - the standalone face of
lora_train (evolve's Make-LoRA button does the same through training.py).

    python tools/train_lora.py --family zimage --name akasjulie --dataset D:/ds/julie ^
        --out-dir D:/evolve_root/loras [--steps 400]

--dataset = images + same-stem .txt captions, every caption STARTING with
--name (the trigger; never auto-decorated). Output:
<out-dir>/<name>/<name>_vNNN_<ts>.safetensors (musubi-native) + the
_comfy.safetensors ComfyUI loads. Every subprocess streams to the console
and to <out-dir>/<name>/logs/<version>.log (tail -f path printed).
Running this IS the say-so to train.
"""
import argparse
import sys
from pathlib import Path

import _cli  # noqa: F401

import lora_train
from comfy_client import free_vram_quietly
from model_family import MODEL_FAMILY_NAMES
from lora_train.common import TrainError, unix_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="zimage", choices=MODEL_FAMILY_NAMES)
    ap.add_argument("--name", required=True, help="LoRA name = trigger word")
    ap.add_argument("--dataset", required=True, help="folder of images + .txt captions")
    ap.add_argument("--out-dir", required=True, help="LoRA library root (<root>/loras)")
    ap.add_argument("--steps", type=int, default=None, help="default: the trainer's (400)")
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--resolution", type=int, default=None, help="768 = the proven fast tune")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--no-convert", action="store_true", help="skip the ComfyUI conversion")
    a = ap.parse_args()

    trainer = lora_train.get_trainer(a.family)
    ok, why = trainer.available()
    if not ok:
        sys.exit(f"{a.family}: {why}")
    out = Path(a.out_dir).expanduser().resolve() / a.name / a.family   # loras/<name>/<family>/
    logs = out.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{a.name}_{__import__('time').strftime('%Y%m%d-%H%M%S')}.log"
    print(f"raw log: tail -f {unix_path(log_path)}")
    free_vram_quietly()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            native = trainer.train(a.dataset, a.name, a.steps or trainer.defaults["steps"],
                                   out, log, rank=a.rank, lr=a.lr,
                                   resolution=a.resolution, repeats=a.repeats)
            print(f"trained -> {native}")
            if not a.no_convert:
                print(f"comfy  -> {trainer.to_comfy(native, log)}")
    except TrainError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
