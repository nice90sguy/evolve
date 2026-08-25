"""Store / tags / trash / controls / lineage semantics on a throwaway root.
Run:  python tests/test_store.py   (or pytest tests/)"""
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import asset
import lineage
import project
import trash
from controls import restore_from_image
from store import ARCHIVED, PINNED, Store


def fresh_root():
    rt = Path(tempfile.mkdtemp(prefix="evolve_test_"))
    project.set_root(rt)
    return rt


IM = Image.new("RGB", (8, 8), (1, 2, 3))


def test_tags_and_cascade():
    rt = fresh_root()
    try:
        project.save_settings(default_tags=["softlock", "(c)2026"])
        st = Store(rt)
        A = st.add_image(IM, "gen", recipe={"prompt": "eve", "seed": 1}, tags=st.birth_tags())
        assert st.tags(A) == ["softlock", "(c)2026"]
        st.tag([A], add=["julie", PINNED])
        X = st.add_image(IM, "gen", recipe={"prompt": "x", "seed": 2}, parents=[A],
                         tags=st.birth_tags(A))
        assert st.tags(X) == ["softlock", "(c)2026", "julie"]      # pinned not inherited
        k = st.add_image(IM, "gen", recipe={"prompt": "k", "seed": 3}, parents=[X], tags=st.birth_tags(X))
        k2 = st.add_image(IM, "gen", recipe={"prompt": "k", "seed": 4}, parents=[X], tags=st.birth_tags(X))
        co = st.add_image(IM, "gen", recipe={"prompt": "co", "seed": 5}, parents=[A, X], tags=st.birth_tags(A))
        assert st.descendants(A) == [X, k, k2, co] and st.descendants(X) == [k, k2]
        st.tag([k2], add=[ARCHIVED])
        touched = st.tag([X], add=["freddy"], cascade=True)          # ADD skips archived
        assert touched == [X, k] and "freddy" not in st.tags(k2)
        st.tag([k], add=["mine"])                                     # independent child word survives
        touched = st.tag([X], remove=["julie"], cascade=True)         # REMOVE does not skip
        assert touched == [X, k, k2] and "mine" in st.tags(k) and "julie" not in st.tags(k2)
        assert st.words()["freddy"] == 2 and st.with_word(PINNED) == [A]
        # garbage = archived and not an ancestor of anything live
        st.tag([X], add=[ARCHIVED])
        assert st.garbage() == [k2]                                   # X has live child k
        st.tag([k, co], add=[ARCHIVED])
        assert st.garbage() == [X, k, k2, co]
        purged = st.purge()
        assert purged == [X, k, k2, co] and not st.alive(X) and st.alive(A)
        assert not st.path(X).exists()
        # journal replays to the same state
        st2 = Store(rt)
        assert st2.alive_ids() == [A] and st2.tags(A) == ["softlock", "(c)2026", "julie", PINNED]
        assert not st2.alive(X) and st2.words() == {"softlock": 1, "(c)2026": 1, "julie": 1, PINNED: 1}
        print("tags/cascade/purge ok")
    finally:
        shutil.rmtree(rt, ignore_errors=True)


def test_trash_ops():
    rt = fresh_root()
    try:
        st = Store(rt)
        A = st.add_image(IM, "gen", recipe={"prompt": "eve", "seed": 1})
        X = st.add_image(IM, "gen", recipe={"prompt": "x", "seed": 2}, parents=[A])
        k = st.add_image(IM, "gen", recipe={"prompt": "k", "seed": 3}, parents=[X])
        k2 = st.add_image(IM, "gen", recipe={"prompt": "k", "seed": 4}, parents=[X])
        st.pin(k, True)
        assert st.pins() == [k]
        st.state["candidates"]["derive"] = [k2, X]
        st.state["working"] = X
        assert trash.sweep(st, "derive") == 1 and st.is_archived(k2) and not st.is_archived(X)
        assert st.state["candidates"]["derive"] == [X]
        plan = trash.prune_plan(st, X)
        assert plan["archive"] == [X] and plan["keep"] == [{"id": k, "why": "pinned"}] and plan["already"] == 1
        plan = trash.prune_apply(st, X, force=True)
        assert sorted(plan["archive"]) == [X, k] and plan["unpin"] == [k]
        assert st.is_archived(k) and not st.has(k, PINNED) and st.state["working"] is None
        assert trash.discard(st, A) is None and st.is_archived(A)
        assert trash.verdict(st, A) == "archived - purgeable"
        assert trash.purge(st)["removed"] == 4
        assert trash.discard(st, A) == "not found"
        print("trash ops ok")
    finally:
        shutil.rmtree(rt, ignore_errors=True)


def test_siblings_and_restore():
    rt = fresh_root()
    try:
        st = Store(rt)
        A = st.add_image(IM, "gen", recipe={"prompt": "eve", "seed": 1, "op": "create"})
        r = {"op": "derive", "prompt": "x", "family": "klein", "seed": 10, "lock": "MID_LOCK",
             "vary": 0.2, "ref0": True, "width": 1024, "height": 768, "lora": "julie",
             "lora_strength": 0.7, "steps": 4, "cfg": 1.0, "whitebg": False}
        c1 = st.add_image(IM, "gen", recipe=r, parents=[A])
        c2 = st.add_image(IM, "gen", recipe=dict(r, seed=11), parents=[A])
        c3 = st.add_image(IM, "gen", recipe=dict(r, seed=12, vary=0.3), parents=[A])
        fam = lineage.family(st, c1)
        assert [t["id"] for t in fam["siblings"]] == [c1, c2]
        assert [t["id"] for t in lineage.family(st, A)["children"]] == [c1, c2, c3]
        restore_from_image(st, c1)
        c = st.state["controls"]
        assert c["tab"] == "derive" and c["ref0"] == A and c["seed_derive"] == 10
        st.hist_append(A); st.hist_append(A); st.hist_append(c1)
        st.describe(c1, "a description")
        st.save_state()
        st2 = Store(rt)
        assert st2.history == [A, c1] and st2.images[c1]["description"] == "a description"
        assert st2.images[c2]["description"] == "x"                  # seeded from the prompt
        print("siblings/restore ok")
    finally:
        shutil.rmtree(rt, ignore_errors=True)


def test_import_and_assets():
    rt = fresh_root()
    try:
        project.save_settings(default_tags=["imp"])
        st = Store(rt)
        info = PngInfo()
        info.add_text("prompt", json.dumps({"1": {"class_type": "KSampler", "inputs": {"seed": 5, "steps": 9, "cfg": 2.0}},
                                            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}}}))
        info.add_text("evolve", json.dumps({"v": 1, "project": "other", "id": 99}))
        buf = io.BytesIO()
        Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(buf, "PNG", pnginfo=info)
        i, new = st.import_bytes(buf.getvalue(), tags=["lora_dataset_julie"])
        assert new and st.tags(i) == ["imp", "lora_dataset_julie"]
        i2, new2 = st.import_bytes(buf.getvalue(), tags=["again"])
        assert i2 == i and not new2 and "again" in st.tags(i)          # sha1 re-attach + word
        with Image.open(st.path(i)) as im:
            assert im.getpixel((0, 0)) == (255, 255, 255) and "prompt" in im.info
            assert json.loads(im.info["evolve"])["id"] == i
        assert st.images[i]["recipe"]["prompt"] == "hello" and st.images[i]["description"] == "hello"
        asset.save_assets([asset.Asset(name="julie")])
        assert asset.dataset_ids(st, "julie") == [i]
        assert asset.caption("julie", "hello") == ("julie, hello", None)
        assert asset.caption("julie", "julie x")[1] is not None          # double prefix warning
        assert asset.caption("julie", "") == ("julie", None)
        st.tag([i], add=[ARCHIVED])
        assert asset.dataset_ids(st, "julie") == []                      # archived leaves datasets
        print("import/assets ok")
    finally:
        shutil.rmtree(rt, ignore_errors=True)


if __name__ == "__main__":
    test_tags_and_cascade()
    test_trash_ops()
    test_siblings_and_restore()
    test_import_and_assets()
    print("ALL OK")
