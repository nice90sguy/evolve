"""tools/recover.py: each strategy recovers a deleted store file from the
copy evolve keeps, in priority order. Run: python tests/test_recover.py"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import project
from image_file import sha1_of
from store import Store


def main():
    rt = Path(tempfile.mkdtemp(prefix="evolve_rec_"))
    comfy = Path(tempfile.mkdtemp(prefix="evolve_reccomfy_"))
    try:
        project.set_root(rt)
        st = Store(rt)
        px = lambda c: Image.new("RGB", (8, 8), c)
        a = st.add_image(px((1, 1, 1)), "gen", recipe={"prompt": "a", "seed": 111})   # staged
        b = st.add_image(px((2, 2, 2)), "gen", recipe={"prompt": "b", "seed": 222})   # _train
        (rt / "_migrated" / "old" / "images").mkdir(parents=True)
        orig = rt / "_migrated" / "old" / "images" / "9.png"
        px((3, 3, 3)).save(orig)
        # migrated records carry the ORIGINAL bytes' sha1 (absorb passes it in)
        c = st.add_image(px((3, 3, 3)), "import", recipe={"prompt": "c"}, sha1=sha1_of(orig.read_bytes()))
        d = st.add_image(px((4, 4, 4)), "gen", recipe={"prompt": "d", "seed": 444})   # scratch seed
        e = st.add_image(px((5, 5, 5)), "gen", recipe={"prompt": "e", "seed": 555})   # unrecoverable
        # plant the copies where evolve leaves them
        staged = comfy / "input" / "evolve" / rt.name
        staged.mkdir(parents=True)
        shutil.copy2(st.path(a), staged / f"{a}.png")
        (rt / "_train" / "x").mkdir(parents=True)
        shutil.copy2(st.path(b), rt / "_train" / "x" / f"{b}.png")
        scratch = comfy / "output" / "evolve_scratch"
        scratch.mkdir(parents=True)
        info = PngInfo()
        info.add_text("prompt", json.dumps({"noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": 444}}}))
        px((4, 4, 4)).save(scratch / "evolve_00001_.png", pnginfo=info)
        # the disaster
        for i in (a, b, c, d, e):
            st.path(i).unlink()
        import recover
        out = []
        r = recover.recover(rt, comfy, apply=False, log=out.append)
        assert r == {"missing": 5, "matched": 4, "restored": 0}, r
        assert not st.path(a).exists()                                   # dry run wrote nothing
        r = recover.recover(rt, comfy, apply=True, log=out.append)
        assert r["restored"] == 4
        st2 = Store(rt)
        for i in (a, b, c, d):
            assert st2.path(i).is_file(), i
        assert Image.open(st2.path(d)).getpixel((0, 0)) == (4, 4, 4)
        assert Image.open(st2.path(a)).getpixel((0, 0)) == (1, 1, 1)
        assert not st2.path(e).exists() and st2.check_files() == 1
        text = "\n".join(out)
        assert "staged hardlink" in text and "_train copy" in text and "_migrated original" in text and "scratch render" in text
        assert "NOT FOUND: 1" in text
        print("recover ALL OK")
    finally:
        shutil.rmtree(rt, ignore_errors=True)
        shutil.rmtree(comfy, ignore_errors=True)


if __name__ == "__main__":
    main()
