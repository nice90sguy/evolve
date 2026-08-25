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
from store import open_project

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


def main():
    rt = Path(tempfile.mkdtemp(prefix="evolve_srv_"))
    project.set_root(rt)
    im = Image.new("RGB", (8, 8), (1, 2, 3))
    st = open_project("p")
    A = st.add_image(im, "gen", recipe={"prompt": "eve", "seed": 1})
    X = st.add_image(im, "gen", recipe={"prompt": "x", "seed": 2}, parents=[A])
    k = st.add_image(im, "gen", recipe={"prompt": "k", "seed": 3}, parents=[X])
    st.save_state()
    srv = subprocess.Popen([sys.executable, str(HERE / "evolve.py"), "--root", str(rt),
                            "--project", "p", "--port", str(PORT)],
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
        assert b'id="prunedlg"' in page and b"/static/js/evolve.js" in page, "index"
        assert get("/static/js/evolve.js")[1][:20] and get("/static/css/evolve.css")[0] == 200
        s, snap = post("/api/state", {}) if False else (200, json.loads(get("/api/state")[1]))
        assert snap["project"] == "p" and snap["all_ids"] == [A, X, k] and "families" in snap
        assert snap["pov_elev"][0][0] == "low" and snap["comfy_ok"] in (True, False)
        s, _ = post("/api/place", {"id": k, "target": "working"})
        assert s == 200
        s, _ = post("/api/pin", {"id": k, "on": True})
        s, fam = post("/api/family", {"id": A})
        assert [t["id"] for t in fam["children"]] == [X]
        s, m = post("/api/meta", {"id": X})
        assert m["gc"].startswith("ancestor of"), m
        s, m = post("/api/meta", {"path": f"{rt}/p/images/{k}.png"})
        assert m["gc"] == "pinned"
        s, d = post("/api/discard", {"id": X})
        assert s == 409 and d["error"].startswith("kept:")
        s, plan = post("/api/prune", {"id": X, "force": True})
        assert s == 200 and sorted(plan["archive"]) == [X, k] and plan["unpin"] == [k]
        s, plan = post("/api/prune", {"id": X, "force": True, "apply": True})
        assert sorted(plan["archive"]) == [X, k]
        assert get(f"/img/p/{A}")[0] == 200
        try:
            get(f"/img/p/{X}")
            raise AssertionError("archived image still served")
        except urllib.error.HTTPError as e:
            assert e.code == 404
        s, r = post("/api/asset", {"op": "create", "name": "julie"})
        assert s == 200 and r["assets"][0]["name"] == "julie"
        s, r = post("/api/asset", {"op": "add", "name": "julie", "path": f"p/images/{A}.png"})
        assert r["assets"][0]["dataset"][0]["description"] == "julie"
        s, r = post("/api/asset", {"op": "create", "name": "bad name"})
        assert s == 400
        s, r = post("/api/train", {"name": "nope", "family": "zimage"})
        assert s == 404
        s, r = post("/api/train", {"name": "julie", "family": "illustrious"})
        assert s == 200, r                       # starts, then fails honestly
        time.sleep(1.0)
        snap = json.loads(get("/api/state")[1])
        assert snap["train"] and not snap["train"]["running"] and "kohya" in snap["train"]["error"], snap["train"]
        s, r = post("/api/controls", {"prompt": "saved", "bogus": 1})
        snap = json.loads(get("/api/state")[1])
        assert snap["controls"]["prompt"] == "saved" and "bogus" not in snap["controls"]
        s, r = post("/api/project", {"name": "q"})
        assert s == 200 and json.loads(get("/api/state")[1])["project"] == "q"
        s, r = post("/api/project", {"name": "loras"})
        assert s == 400
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
