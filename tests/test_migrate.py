"""tools/migrate_projects.py on a synthetic pre-tags root: two projects
(one journaled, one hand-copied with chunk-only provenance), duplicates,
an archive, pins, an old-shape assets.json. Run: python tests/test_migrate.py"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))
from PIL import Image

import project
from image_file import write_png
from image_meta import evolve_chunk
from store import Store


def main():
    rt = Path(tempfile.mkdtemp(prefix="evolve_mig_"))
    try:
        project.set_root(rt)
        # --- project "soft": a real journal (old per-project store) ---
        soft = rt / "soft"
        (soft / "images").mkdir(parents=True)
        (soft / "archive").mkdir()
        recs = []
        for i, (color, parents, prompt) in enumerate([((1, 2, 3), [], "eve"), ((4, 5, 6), [1], "kid"),
                                                       ((7, 8, 9), [2], "grandkid")], start=1):
            im = Image.new("RGB", (8, 8), color)
            chunks = {"evolve": evolve_chunk("soft", i, "gen", {"prompt": prompt, "seed": i}, {})}
            write_png(im, soft / "images" / f"{i}.png", chunks)
            recs.append({"t": "image", "id": i, "file": f"{i}.png", "source": "gen", "w": 8, "h": 8,
                         "sha1": None, "recipe": {"prompt": prompt, "seed": i}, "parents": parents,
                         "ts": "2026-08-20 10:00:00"})
        # image 3 was archived (gc event + file in archive/)
        (soft / "images" / "3.png").rename(soft / "archive" / "3.png")
        with (soft / "journal.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps({"t": "hist", "id": 1}) + "\n")
            f.write(json.dumps({"t": "gc", "ids": [3]}) + "\n")
        (soft / "state.json").write_text(json.dumps({"working": 2, "slots": 6, "pins": [2],
                                                     "candidates": {"create": [], "derive": [], "camera": []},
                                                     "controls": {}}), encoding="utf-8")
        # --- project "shared": hand-copied files, no journal ---
        shared = rt / "shared"
        (shared / "images").mkdir(parents=True)
        shutil.copy2(soft / "images" / "1.png", shared / "images" / "1.png")      # duplicate bytes
        # a child rendered from soft/1, chunk-only provenance with a verifiable parent
        import hashlib
        sha256 = hashlib.sha256((soft / "images" / "1.png").read_bytes()).hexdigest()
        payload = {"img_0": {"class_type": "LoadImage", "inputs": {"image": "evolve/soft/1.png", "is_changed": sha256}},
                   "txt": {"class_type": "CLIPTextEncode", "inputs": {"text": "from one"}},
                   "samp": {"class_type": "KSampler", "inputs": {"seed": 77, "steps": 4, "cfg": 1.0,
                                                                 "positive": ["txt", 0]}}}
        chunks = {"prompt": json.dumps(payload),
                  "evolve": evolve_chunk("soft", 9, "gen", {"prompt": "from one", "seed": 77}, {"evolve/soft/1.png": 1})}
        write_png(Image.new("RGB", (8, 8), (9, 9, 9)), shared / "images" / "73.png", chunks)
        # a jpg alien
        Image.new("RGB", (8, 8), (20, 20, 20)).save(shared / "images" / "74.jpg")
        (rt / "assets.json").write_text(json.dumps([
            {"name": "julie", "loras": ["loras/julie/j_comfy.safetensors"],
             "dataset": [{"path": "shared/images/1.png", "description": "julie head shot"},
                         {"path": "shared/images/73.png", "description": "julie, standing"},
                         {"path": "soft/images/2.png", "description": "julie"}]}]), encoding="utf-8")

        import migrate_projects
        report = migrate_projects.migrate(rt)
        print("\n".join(report))
        st = Store(rt)
        ids = st.alive_ids()
        assert len(ids) == 4, ids                       # 1,2,3(archived) + 73 ; dup collapsed; jpg ignored by png glob
        by_prompt = {st.images[i]["recipe"]["prompt"]: i for i in ids}
        one, kid, gk, f1 = by_prompt["eve"], by_prompt["kid"], by_prompt["grandkid"], by_prompt["from one"]
        assert st.tags(one) == ["soft", "shared", "lora_dataset_julie"], st.tags(one)
        assert st.tags(kid) == ["soft", "pinned", "lora_dataset_julie"], st.tags(kid)
        assert st.tags(gk) == ["soft", "archived"]
        assert st.images[kid]["parents"] == [one] and st.images[gk]["parents"] == [kid]
        assert st.images[f1]["parents"] == [one], st.images[f1]      # verified via sha256
        assert st.images[one]["description"] == "head shot"          # trigger stripped
        assert st.images[f1]["description"] == "standing"
        assert st.images[kid]["description"] == "kid"                # asset said bare trigger -> seeded prompt stays
        assert st.history == [one]
        # the synthetic LoRA path never existed on disk: dropped, reported, family entries otherwise
        assert json.loads((rt / "assets.json").read_text()) == [{"name": "julie", "loras": []}]
        assert any("dropped" in line for line in report)
        assert (rt / "_migrated" / "soft" / "journal.jsonl").exists() and not (rt / "soft").exists()
        # scan the leftover jpg as an alien
        rep = migrate_projects.scan(rt, rt / "_migrated" / "shared" / "images", ["alien"])
        st = Store(rt)
        assert len(st.alive_ids()) == 5 and st.with_word("alien"), rep
        print("migrate ALL OK")
    finally:
        shutil.rmtree(rt, ignore_errors=True)


if __name__ == "__main__":
    main()
