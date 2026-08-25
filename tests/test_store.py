"""Store / trash / controls / lineage semantics on a throwaway root.
Run:  python tests/test_store.py   (or pytest tests/)"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image

import asset
import lineage
import project
import trash
from controls import restore_from_image
from store import open_project


def fresh_root():
    rt = Path(tempfile.mkdtemp(prefix="evolve_test_"))
    project.set_root(rt)
    return rt


IM = Image.new("RGB", (8, 8), (1, 2, 3))


def test_roots_and_prune():
    rt = fresh_root()
    try:
        st = open_project("p")
        A = st.add_image(IM, "gen", recipe={"prompt": "eve", "seed": 1})
        X = st.add_image(IM, "gen", recipe={"prompt": "x", "seed": 2}, parents=[A])
        k = st.add_image(IM, "gen", recipe={"prompt": "k", "seed": 3}, parents=[X])
        k2 = st.add_image(IM, "gen", recipe={"prompt": "k", "seed": 4}, parents=[X])
        st.pin(k, True)
        assert trash.root_reason(st, k) == "pinned"
        assert trash.root_reason(st, X).startswith("ancestor of")
        assert trash.root_reason(st, k2) is None
        assert trash.discard(st, X) is not None          # integrity refusal
        assert trash.discard(st, k2) is None
        assert not st.alive(k2) and (rt / "p" / "archive" / f"{k2}.png").exists()
        plan = trash.prune_plan(st, X)
        assert plan["archive"] == [] and {e["id"] for e in plan["keep"]} == {X, k}
        plan = trash.prune_apply(st, X, force=True)
        assert sorted(plan["archive"]) == [X, k] and plan["unpin"] == [k]
        assert st.alive(A) and not st.alive(X) and not st.alive(k)
        assert st.state["pins"] == []
        # asset entries are roots too
        B = st.add_image(IM, "gen", recipe={"prompt": "b", "seed": 5}, parents=[A])
        assets = [{"name": "julie", "loras": [], "dataset": [{"path": st.rel(B), "description": "julie"}]}]
        asset.save_assets(assets)
        assert trash.root_reason(st, B) == "in asset julie"
        assert trash.root_reason(st, A).startswith("ancestor of")
        assert trash.gc(st)["removed"] == 0
        print("roots/prune ok")
    finally:
        shutil.rmtree(rt, ignore_errors=True)


def test_siblings_and_restore():
    rt = fresh_root()
    try:
        st = open_project("p")
        A = st.add_image(IM, "gen", recipe={"prompt": "eve", "seed": 1, "op": "create"})
        r = {"op": "derive", "prompt": "x", "family": "klein", "seed": 10, "lock": "MID_LOCK",
             "vary": 0.2, "ref0": True, "width": 1024, "height": 768, "lora": "julie",
             "lora_strength": 0.7, "steps": 4, "cfg": 1.0, "whitebg": False}
        c1 = st.add_image(IM, "gen", recipe=r, parents=[A])
        c2 = st.add_image(IM, "gen", recipe=dict(r, seed=11), parents=[A])
        c3 = st.add_image(IM, "gen", recipe=dict(r, seed=12, vary=0.3), parents=[A])
        fam = lineage.family(st, c1)
        assert [t["id"] for t in fam["siblings"]] == [c1, c2]      # config modulo seed
        assert [t["id"] for t in fam["parents"]] == [A]
        assert [t["id"] for t in lineage.family(st, A)["children"]] == [c1, c2, c3]
        restore_from_image(st, c1)
        c = st.state["controls"]
        assert c["tab"] == "derive" and c["ref0"] == A and c["seed_derive"] == 10
        assert c["lock"] == "MID_LOCK" and c["lora"] == "julie" and c["width"] == 1024
        p = st.add_image(IM, "pov", recipe={"op": "pov", "pov_elev": "low", "seed": 3,
                                            "prompt": "<sks> low-angle shot"}, parents=[c1])
        restore_from_image(st, p)
        assert c["tab"] == "camera" and c["pov_elev"] == "low" and c["ref0"] == c1
        # history: bred-from only, consecutive dupes collapse
        st.hist_append(A); st.hist_append(A); st.hist_append(c1)
        st.save_state()
        assert st.history == [A, c1]
        # reload round-trips
        st2 = open_project("p")
        assert st2.history == [A, c1] and st2.state["controls"]["tab"] == "camera"
        print("siblings/restore ok")
    finally:
        shutil.rmtree(rt, ignore_errors=True)


def test_import_preserves_chunks():
    rt = fresh_root()
    try:
        import io
        import json
        from PIL.PngImagePlugin import PngInfo
        st = open_project("p")
        info = PngInfo()
        info.add_text("prompt", json.dumps({"1": {"class_type": "KSampler", "inputs": {"seed": 5, "steps": 9, "cfg": 2.0}},
                                            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}}}))
        info.add_text("evolve", json.dumps({"v": 1, "project": "other", "id": 99}))
        buf = io.BytesIO()
        Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(buf, "PNG", pnginfo=info)
        i, new = st.import_bytes(buf.getvalue())
        assert new
        i2, new2 = st.import_bytes(buf.getvalue())
        assert i2 == i and not new2                            # sha1 dedupe
        with Image.open(st.path(i)) as im:
            assert im.mode == "RGB" and im.getpixel((0, 0)) == (255, 255, 255)   # flattened on white
            assert "prompt" in im.info                                            # layer 0
            ev = json.loads(im.info["evolve"])
            assert ev["project"] == "p" and ev["id"] == i                         # ours wins
        rec = st.images[i]["recipe"]
        assert rec["prompt"] == "hello" and rec["seed"] == 5 and rec["op"] == "import"
        print("import ok")
    finally:
        shutil.rmtree(rt, ignore_errors=True)


if __name__ == "__main__":
    test_roots_and_prune()
    test_siblings_and_restore()
    test_import_preserves_chunks()
    print("ALL OK")
