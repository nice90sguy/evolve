"""Frontend contract: the JS parses (node --check), every element id the JS
addresses with $('#id') exists in index.html, and every /api/<name> the JS
calls has a server route. Run: python tests/test_frontend.py"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
FRONT = HERE / "frontend"


def test_js_parses():
    r = subprocess.run(["node", "--check", str(FRONT / "js" / "evolve.js")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    print("OK   js parses")


def test_ids_exist():
    js = (FRONT / "js" / "evolve.js").read_text(encoding="utf-8")
    html = (FRONT / "index.html").read_text(encoding="utf-8")
    have = set(re.findall(r'id="([^"]+)"', html))
    used = set(re.findall(r"\$\('#([A-Za-z0-9_-]+)'\)", js))
    # ids the JS creates itself (the Space-peek overlay) or builds per tab
    dynamic = {u for u in used if u.endswith("_")} | {"peek"}
    missing = sorted(u for u in used - have - dynamic if not re.search(r"\W", u))
    assert not missing, f"ids used in JS but absent from index.html: {missing}"
    print(f"OK   {len(used)} element ids resolve")


def test_api_routes():
    import api
    js = (FRONT / "js" / "evolve.js").read_text(encoding="utf-8")
    called = set(re.findall(r"api\('([a-z_]+)'", js))
    missing = sorted(called - set(api.ROUTES))
    assert not missing, f"JS calls api() names with no route: {missing}"
    print(f"OK   {len(called)} api names routed")


if __name__ == "__main__":
    test_js_parses()
    test_ids_exist()
    test_api_routes()
    print("ALL OK")
