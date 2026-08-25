"""Template-driven builders: every graph is internally consistent (all
node links resolve, one SaveImage under the scratch prefix, our client id)
and the knobs land where ComfyUI reads them. Run: python tests/test_payloads.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_payload import flux_klein, illustrious, qwen_edit, zimage
from comfy_client import CLIENT_ID, SCRATCH_PREFIX


def check(label, payload):
    assert payload["client_id"] == CLIENT_ID, label
    g = payload["prompt"]
    for nid, node in g.items():
        assert "class_type" in node and "inputs" in node, (label, nid)
        for k, v in node["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in g, f"{label}: {nid}.{k} -> missing node {v[0]}"
    saves = [n for n in g.values() if n["class_type"] == "SaveImage"]
    assert len(saves) == 1 and saves[0]["inputs"]["filename_prefix"].startswith(SCRATCH_PREFIX + "/"), label
    print("OK  ", label, "-", len(g), "nodes")
    return g


def test_flux_klein():
    g = check("klein fiat", flux_klein.build("P", 7, 1024, 768))
    assert "ift" not in g and g["guider"]["inputs"]["model"] == ["unet", 0]
    assert g["txt"]["inputs"]["text"] == "P" and g["noise"]["inputs"]["noise_seed"] == 7
    assert g["sched"]["inputs"]["width"] == 1024 and g["latent"]["inputs"]["height"] == 768
    g = check("klein refs", flux_klein.build("P", 7, 1024, 768, ["a.png", "b.png"], "MID_LOCK", 0.2,
                                             "X:/l.safetensors", 0.8, steps=8, cfg=1.5,
                                             identity_refs="all", batch_size=3))
    assert g["ift"]["inputs"]["preset"] == "MID_LOCK" and g["ift"]["inputs"]["model"] == ["gene", 0]
    assert g["ift"]["inputs"]["reference_indices"] == "all"
    assert g["guider"]["inputs"]["positive"] == ["pos_1", 0] and g["pos_0"]["inputs"]["latent"] == ["jit", 0]
    assert g["jit"]["inputs"]["mix_randn_amount"] == 0.2 and g["gene"]["inputs"]["strength"] == 0.8
    assert g["sched"]["inputs"]["steps"] == 8 and g["guider"]["inputs"]["cfg"] == 1.5
    assert g["latent"]["inputs"]["batch_size"] == 3


def test_zimage_and_sdxl():
    g = check("zimage", zimage.build("P", "N", 3, 832, 1216, 20, 2.0, "X:/z.safetensors", 0.9))
    assert g["samp"]["inputs"]["model"] == ["gene", 0] and g["neg"]["inputs"]["text"] == "N"
    assert g["samp"]["inputs"]["seed"] == 3 and g["latent"]["inputs"]["width"] == 832
    g = check("zimage no lora", zimage.build("P", "", 3, 832, 1216))
    assert "gene" not in g and g["samp"]["inputs"]["steps"] == zimage.DEFAULT_STEPS
    g = check("sdxl", illustrious.build("P", "N", 3, 832, 1216))
    assert g["samp"]["inputs"]["cfg"] == illustrious.DEFAULT_CFG and g["samp"]["inputs"]["model"] == ["ckpt", 0]


def test_qwen_edit():
    g = check("qwen lightning", qwen_edit.build("s.png", "<sks> x", "", 5))
    assert g["samp"]["inputs"]["model"] == ["fast", 0] and g["img"]["inputs"]["image"] == "s.png"
    assert g["samp"]["inputs"]["steps"] == 4 and g["pos"]["inputs"]["prompt"] == "<sks> x"
    g = check("qwen slow", qwen_edit.build("s.png", "<sks> x", "neg", 5, 20, 2.5, 0.7, lightning=False))
    assert "fast" not in g and g["samp"]["inputs"]["model"] == ["gene", 0]
    assert g["gene"]["inputs"]["strength_model"] == 0.7 and g["neg"]["inputs"]["prompt"] == "neg"
    assert qwen_edit.MODEL_GGUF.endswith(".gguf") and "Lightning" in qwen_edit.LORA_LIGHTNING


if __name__ == "__main__":
    test_flux_klein()
    test_zimage_and_sdxl()
    test_qwen_edit()
    print("ALL OK")
