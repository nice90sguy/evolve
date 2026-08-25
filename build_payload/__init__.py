"""build_payload - ComfyUI API-format graphs, one module per model family,
every one of them TEMPLATE-DRIVEN: the graph shape lives in
../templates/<name>.json (runnable as-is), the builder only patches values.

    flux_klein   FLUX.2 Klein 9B: refs (ReferenceLatent) + identity transfer
    zimage       Z-Image Turbo fiat (optional LoRA)
    illustrious  Illustrious SDXL fiat
    qwen_edit    Qwen-Image-Edit-2511 + multiple-angles LoRA (camera re-shoot)

Node-id vocabulary shared by all templates: unet clip vae txt/pos neg
latent samp dec save (+ img_i enc_i pos_i neg_i ift guider sched noise in
the Klein graph, gene for a LoRA, fast for the Lightning LoRA).
"""
import copy
import json
from pathlib import Path

from comfy_client import CLIENT_ID, SCRATCH_PREFIX

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
_CACHE = {}


def load_template(name):
    """A deep copy of templates/<name>.json (parsed once per process)."""
    if name not in _CACHE:
        _CACHE[name] = json.loads((TEMPLATES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return copy.deepcopy(_CACHE[name])


def attach_lora(graph, lora_path, strength, model_node="unet", node="gene"):
    """Insert an ApplyTrainedLora 'gene' node after model_node; returns the
    node id downstream consumers should read the model from."""
    graph[node] = {"class_type": "ApplyTrainedLora",
                   "inputs": {"strength": strength, "model": [model_node, 0],
                              "lora_path": str(lora_path)}}
    return node


def set_output(graph, out_prefix):
    graph["save"]["inputs"]["filename_prefix"] = f"{SCRATCH_PREFIX}/{out_prefix}"


def payload(graph, client_id=CLIENT_ID):
    return {"client_id": client_id, "prompt": graph}
