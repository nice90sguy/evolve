"""Places: one image one place, birth dirs, moves, trash-as-place, rescan
(external move / alien file / missing). Run: python tests/test_places.py"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image

import project
import trash
from store import DEFAULT_DIR, TRASH, Store, valid_dir

IM = Image.new("RGB", (8, 8), (1, 2, 3))


def main():
    rt = Path(tempfile.mkdtemp(prefix="evolve_places_"))
    try:
        project.set_root(rt)
        st = Store(rt)
        assert valid_dir("a/b") == "a/b" and valid_dir("loras/x") is None
        assert valid_dir("..") is None and valid_dir(".trash") == TRASH and valid_dir("a/.trash") is None
        A = st.add_image(IM, "gen", recipe={"prompt": "eve", "seed": 1})
        assert st.image_dir(A) == DEFAULT_DIR and (rt / DEFAULT_DIR / "1.png").is_file()
        st.set_cwd("chars/julie")
        B = st.add_image(IM, "gen", recipe={"prompt": "b", "seed": 2})   # fiat -> cwd
        assert st.image_dir(B) == "chars/julie"
        kid = st.add_image(Image.new("RGB", (8, 8), (9, 9, 9)), "gen",
                           recipe={"prompt": "kid", "seed": 3}, parents=[B],
                           dir=st.birth_dir(B))
        assert st.image_dir(kid) == "chars/julie"                        # mother's folder
        # move (the app's own)
        assert st.move([A], "chars/julie") == [A] and (rt / "chars/julie" / "1.png").is_file()
        assert "chars" in st.dirs() and "chars/julie" in st.dirs() and st.dirs()[-1] == TRASH
        # trash is a place; pinned => not archived
        st.pin(kid, True)
        assert st.archive([kid]) == [] and not st.is_archived(kid)
        assert st.archive([B]) == [B] and st.is_archived(B)
        assert (rt / TRASH / f"{B}.png").is_file()
        assert trash.verdict(st, B).startswith("archived - kept")        # kid is live
        st.restore([B])
        assert st.image_dir(B) == "chars/julie"                          # home remembered
        st.archive([B])
        st.pin(kid, False)
        st.archive([kid])
        assert sorted(st.garbage()) == [B, kid]
        # rescan: external move, alien file, missing
        (rt / "scenes").mkdir()
        shutil.move(str(rt / "chars/julie" / "1.png"), str(rt / "scenes" / "1.png"))
        Image.new("RGB", (8, 8), (77, 3, 5)).save(rt / "scenes" / "alien.png")
        Image.new("RGB", (8, 8), (99, 1, 1)).save(rt / "scenes" / "alien2.jpg")
        r = st.rescan()
        assert st.image_dir(A) == "scenes", st.images[A]
        assert r["moved"] == 1 and r["imported"] == 2 and r["skipped"], r
        aliens = [i for i in st.alive_ids() if st.images[i]["source"] == "scan"]
        assert len(aliens) == 2 and all(st.image_dir(i) == "scenes" for i in aliens)
        # journal replays to the same shape
        st2 = Store(rt)
        assert st2.image_dir(A) == "scenes" and st2.is_archived(B) and st2.images[B]["home"] == "chars/julie"
        assert len(st2.alive_ids()) == len(st.alive_ids())
        # missing: delete an alien's file
        (rt / "scenes" / "alien.png").unlink()
        r = st2.rescan()
        assert r["missing"] == 1 and st2.missing
        print("places ALL OK")
    finally:
        shutil.rmtree(rt, ignore_errors=True)


if __name__ == "__main__":
    main()
