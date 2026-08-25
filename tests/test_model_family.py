"""Model family as a shared, validated constraint: ModelFamily, the pydantic
configs (Generate/Camera/Train), per-family asset LoRA entries, dropdown
resolution and detect_family. Run: python tests/test_family.py"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pydantic import ValidationError
from safetensors.numpy import save_file

import asset
import lora
import project
from camera import CameraConfig
from model_family import ModelFamily, family_info, parse_model_family
from generate import GenerateConfig
from lora_train.common import detect_family
from training import TrainConfig


def raises(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except (ValidationError, ValueError) as e:
        return str(e)
    raise AssertionError(f"{fn} accepted {a} {kw}")


def test_family_enum():
    assert parse_model_family("klein") is ModelFamily.KLEIN and parse_model_family(ModelFamily.ZIMAGE) is ModelFamily.ZIMAGE
    assert parse_model_family("nope", ModelFamily.KLEIN) is ModelFamily.KLEIN
    assert "unknown model family" in raises(parse_model_family, "sdxl")
    assert family_info("illustrious").trainable is False and family_info("klein").references is True
    print("family enum ok")


def test_configs():
    c = GenerateConfig.from_controls({"family": "zimage", "prompt": "p", "width": 1000, "vary": 9}, "create")
    assert c.family is ModelFamily.ZIMAGE and c.width == 992 and c.vary == 0.5 and c.ref_ids == []
    c = GenerateConfig.from_controls({"family": "zimage", "ref0": 3, "refs": [None, 4, None]}, "derive")
    assert c.family is ModelFamily.KLEIN and c.ref_ids == [3, 4] and c.has_ref0
    c = GenerateConfig.from_controls({"family": "illustrious", "lora": "julie"}, "create")
    assert c.lora == ""                                   # SDXL takes no LoRA: dropped, not refused
    assert "Klein-only" in raises(GenerateConfig, op="derive", family="zimage")
    assert "reference" in raises(GenerateConfig, family="zimage", ref_ids=[1])
    assert "multiple" in raises(GenerateConfig, width=1000)
    assert "less than or equal" in raises(GenerateConfig, lora_strength=9)
    assert raises(GenerateConfig, family="sdxl")
    cam = CameraConfig(source_id=3, elev="low")
    assert cam.any_axis() and cam.azim is None
    assert "bad azimuth" in raises(CameraConfig, source_id=3, azim="sideways")
    assert "no camera axis" in raises(CameraConfig, source_id=3)
    assert raises(CameraConfig, source_id=0, elev="low")
    t = TrainConfig(asset="julie", family="klein")
    assert t.family is ModelFamily.KLEIN and t.steps is None
    assert "not trainable" in raises(TrainConfig, asset="julie", family="illustrious")
    assert raises(TrainConfig, asset="bad name", family="zimage")
    assert raises(TrainConfig, asset="julie", family="zimage", steps=0)
    print("configs ok")


def test_assets_and_dropdown():
    rt = Path(tempfile.mkdtemp(prefix="evolve_fam_"))
    try:
        project.set_root(rt)
        (rt / "loras" / "julie" / "klein").mkdir(parents=True)
        (rt / "loras" / "julie" / "zimage").mkdir(parents=True)
        (rt / "loras" / "loose" / "klein").mkdir(parents=True)
        for p in ("julie/klein/julie_v001_x_comfy.safetensors", "julie/klein/julie_v002_x_comfy.safetensors",
                  "julie/zimage/julie_v001_z_comfy.safetensors", "loose/klein/other_comfy.safetensors",
                  "julie/klein/julie_v002_x_comfy.safetensors.bak"):
            (rt / "loras" / p).write_bytes(b"x")
        (rt / "loras" / "stray.safetensors").write_bytes(b"x")     # not under a family dir: ignored
        a = asset.Asset(name="julie")
        assert asset.apply_op([a], "add_lora", "julie", "loras/julie/klein/julie_v001_x_comfy.safetensors") is None
        assert asset.apply_op([a], "add_lora", "julie", "loras/julie/klein/julie_v002_x_comfy.safetensors") is None
        assert asset.apply_op([a], "add_lora", "julie", "loras/julie/zimage/julie_v001_z_comfy.safetensors") is None
        bad = asset.apply_op([a], "add_lora", "julie", "loras/julie/zimage/julie_v001_z_comfy.safetensors", "klein")
        assert bad and "directory says" in bad
        asset.save_assets([a])
        assets = asset.load_assets()
        assert [e.family for e in assets[0].loras] == [ModelFamily.KLEIN, ModelFamily.KLEIN, ModelFamily.ZIMAGE]
        d = lora.list_loras()
        assert d["klein"] == ["julie", "loras/loose/klein/other_comfy.safetensors"], d
        assert d["zimage"] == ["julie"] and d["illustrious"] == []
        assert lora.resolve_lora("julie", "klein").name == "julie_v002_x_comfy.safetensors"   # newest for the family
        assert lora.resolve_lora("julie", "zimage").name == "julie_v001_z_comfy.safetensors"
        assert lora.resolve_lora("julie", "illustrious") is None
        assert lora.resolve_lora("loras/loose/klein/other_comfy.safetensors", "zimage") is None  # wrong family
        assert lora.resolve_lora("loras/loose/klein/other_comfy.safetensors", "klein") is not None
        # a pre-family assets.json is refused loudly
        project.write_json(rt / "assets.json", [{"name": "julie", "loras": ["loras/julie/x.safetensors"]}])
        try:
            asset.load_assets()
            raise AssertionError("legacy assets accepted")
        except asset.AssetsFormatError as e:
            assert "migrate_projects.py --loras" in str(e)
        # detect_family from file metadata / keys
        f = rt / "k.safetensors"
        save_file({"lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": np.zeros((2, 2), dtype=np.float32)},
                  str(f), metadata={"ss_network_module": "networks.lora_flux_2"})
        assert detect_family(f) is ModelFamily.KLEIN
        save_file({"lora_unet_layers_0.lora_down.weight": np.zeros((2, 2), dtype=np.float32)}, str(f),
                  metadata={"ss_network_module": "networks.lora_zimage"})
        assert detect_family(f) is ModelFamily.ZIMAGE
        save_file({"w": np.zeros((2, 2), dtype=np.float32)}, str(f), metadata={})
        assert detect_family(f) is None
        print("assets/dropdown ok")
    finally:
        shutil.rmtree(rt, ignore_errors=True)


if __name__ == "__main__":
    test_family_enum()
    test_configs()
    test_assets_and_dropdown()
    print("ALL OK")
