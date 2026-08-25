"""HTTP smoke: start evolve on a throwaway root (port 8199), exercise the
API without ComfyUI (no generation). Run: python tests/test_server.py"""
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from PIL import Image

import project
from store import Store

PORT = 8199
BASE = f"http://127.0.0.1:{PORT}"


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=5) as r:
        return r.status, r.read()


def state():
    return json.loads(get("/api/state")[1])


def main():
    rt = Path(tempfile.mkdtemp(prefix="evolve_srv_"))
    project.set_root(rt)
    project.save_settings(default_tags=["softlock"])
    im = Image.new("RGB", (8, 8), (1, 2, 3))
    st = Store(rt)
    A = st.add_image(im, "gen", recipe={"prompt": "eve", "seed": 1}, tags=st.birth_tags())
    X = st.add_image(im, "gen", recipe={"prompt": "x", "seed": 2}, parents=[A], tags=st.birth_tags(A))
    k = st.add_image(im, "gen", recipe={"prompt": "k", "seed": 3}, parents=[X], tags=st.birth_tags(X))
    st.save_state()
    srv = subprocess.Popen([sys.executable, str(HERE / "evolve.py"), "--root", str(rt),
                            "--port", str(PORT)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen(BASE + "/api/state", timeout=1).read()
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise SystemExit("server did not start")
        s, page = get("/")
        assert b'id="wordbar"' in page and b"/static/js/evolve.js" in page, "index"
        assert get("/static/js/evolve.js")[0] == 200 and get("/static/css/evolve.css")[0] == 200
        snap = state()
        assert snap["all_ids"] == [A, X, k] and snap["words"] == {"softlock": 3}
        assert snap["settings"]["default_tags"] == ["softlock"] and snap["tags"][str(k)] == ["softlock"]
        assert sorted(snap["loras"]) == ["illustrious", "klein", "zimage"] and snap["families"]["klein"]["lora"] is True
        s, r = post("/api/settings", {"default_tags": ["softlock", " julie ", "bad word", ""]})
        assert r["default_tags"] == ["softlock", "julie"], r
        s, r = post("/api/tag", {"id": X, "add": ["freddy"], "cascade": True})
        assert s == 200 and r["touched"] == [X, k]
        s, r = post("/api/tag", {"ids": [A], "add": ["pinned"]})
        assert state()["pins"] == [A]
        s, r = post("/api/describe", {"id": k, "description": "a kid"})
        assert state()["descriptions"][str(k)] == "a kid"
        s, m = post("/api/meta", {"id": k})
        assert m["tags"] == ["softlock", "freddy"] and m["description"] == "a kid" and m["gc"] == "live"
        s, _ = post("/api/place", {"id": k, "target": "working"})
        s, fam = post("/api/family", {"id": A})
        assert [t["id"] for t in fam["children"]] == [X]
        s, plan = post("/api/prune", {"id": X})
        assert s == 200 and sorted(plan["archive"]) == [X, k]
        s, plan = post("/api/prune", {"id": X, "apply": True})
        snap = state()
        assert "archived" in snap["tags"][str(X)] and snap["working"] is None
        s, m = post("/api/meta", {"id": X})
        assert m["gc"].startswith("archived - purgeable")
        s, r = post("/api/gc", {})
        assert r["removed"] == 2 and state()["all_ids"] == [A]
        try:
            get(f"/img/{X}")
            raise AssertionError("purged image still served")
        except urllib.error.HTTPError as e:
            assert e.code == 404
        s, r = post("/api/discard", {"id": A})
        assert s == 200 and "archived" in state()["tags"][str(A)]
        s, r = post("/api/asset", {"op": "create", "name": "julie"})
        assert s == 200 and r["assets"] == [{"name": "julie", "loras": []}]
        s, r = post("/api/asset", {"op": "create", "name": "bad name"})
        assert s == 400
        s, r = post("/api/train", {"name": "julie", "family": "zimage"})
        assert s == 400 and "empty dataset" in r["error"], r
        s, r = post("/api/tag", {"id": A, "add": ["lora_dataset_julie"], "remove": ["archived"]})
        s, r = post("/api/train", {"name": "julie", "family": "illustrious"})
        assert s == 400 and "not trainable" in r["error"], r       # validated before anything runs
        s, r = post("/api/train", {"name": "julie", "family": "sdxl"})
        assert s == 400 and "family" in r["error"], r
        s, r = post("/api/generate", {"op": "create", "family": "zimage", "lora_strength": 99})
        assert s == 400 and "lora_strength" in r["error"], r
        s, r = post("/api/asset", {"op": "add_lora", "name": "julie", "path": "loras/julie/x.safetensors"})
        assert s == 400 and "loras/<asset>/<family>/" in r["error"], r
        (rt / "loras" / "julie" / "klein").mkdir(parents=True)
        (rt / "loras" / "julie" / "klein" / "julie_v001_comfy.safetensors").write_bytes(b"x")
        s, r = post("/api/asset", {"op": "add_lora", "name": "julie", "path": "loras/julie/klein/julie_v001_comfy.safetensors"})
        assert s == 200 and r["assets"][0]["loras"] == [{"path": "loras/julie/klein/julie_v001_comfy.safetensors", "family": "klein"}], r
        assert state()["loras"]["klein"] == ["julie"] and state()["loras"]["zimage"] == []
        s, r = post("/api/controls", {"prompt": "saved", "bogus": 1})
        snap = state()
        assert snap["controls"]["prompt"] == "saved" and "bogus" not in snap["controls"]
        s, r = post("/api/pov", {"elev": "low"})
        assert s == 400 and "working" in r["error"]
        s, r = post("/api/abort", {})
        assert r["aborted"] is False
        print("server smoke ALL OK")
    finally:
        srv.kill()
        srv.wait()
        shutil.rmtree(rt, ignore_errors=True)


if __name__ == "__main__":
    main()
