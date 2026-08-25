"""comfy_client.py - everything that talks to a running ComfyUI.

All scripts assume the service is UP at http://127.0.0.1:8188 and is NEVER
launched or killed from here. Payloads are API-format JSON
({"client_id", "prompt": {node_id: {class_type, inputs}}}); outputs land in
<ComfyUI>/output/<subfolder>; inputs must be staged under <ComfyUI>/input.

Every wait shows live progress (red line 5): per-step sampler bars from the
websocket plus an elapsed heartbeat whenever nothing has printed for 15s
(model loads emit no events).
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

COMFY_DIR = Path(__file__).resolve().parents[1]     # this repo lives inside ComfyUI
INPUT_DIR = COMFY_DIR / "input"
OUTPUT_DIR = COMFY_DIR / "output"
API_URL = "http://127.0.0.1:8188"
SCRATCH_PREFIX = "evolve_scratch"   # output subfolder renders go to; copied out after
CLIENT_ID = "evolve"                # progress events route to the queuing client id
DEBUG_DIR = Path(__file__).resolve().parent / "_debug"   # last_payload.json (evolve
                                                         # repoints this at <root>/_debug)


class ComfyError(RuntimeError):
    """ComfyUI rejected, failed or timed out a job (tools turn it into exit)."""


def queue(payload):
    """POST /prompt -> prompt_id. The exact submitted graph is always written
    to DEBUG_DIR/last_payload.json."""
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        (DEBUG_DIR / "last_payload.json").write_text(json.dumps(payload, indent=1),
                                                     encoding="utf-8")
    except OSError:
        pass
    req = urllib.request.Request(API_URL + "/prompt", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)["prompt_id"]
    except urllib.error.HTTPError as e:
        # surface ComfyUI's validation report instead of a bare 400
        body = e.read().decode("utf-8", "replace")
        try:
            body = json.dumps(json.loads(body), indent=1)
        except Exception:
            pass
        raise ComfyError(f"ComfyUI rejected the graph (HTTP {e.code}):\n{body[:3000]}")


def free_vram():
    """POST /free (unload_models + free_memory): the latter resets the node
    cache so loaders reload weights from DISK - the only cure for LoRA
    deltas compounding on the shared base weights (ComfyUI bug #11021).
    Klein-only work should not call this per render: the resident model
    is what makes re-renders instant."""
    req = urllib.request.Request(
        API_URL + "/free",
        json.dumps({"unload_models": True, "free_memory": True}).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        r.read()


def free_vram_quietly():
    """Best-effort free_vram() (before training: ComfyUI may not be up)."""
    try:
        free_vram()
        print("asked ComfyUI to free VRAM")
    except Exception:
        pass


def interrupt():
    """POST /interrupt: stop the in-flight job. Aborts by choice only -
    it cannot un-wedge a CUDA OOM (that needs a ComfyUI restart)."""
    urllib.request.urlopen(API_URL + "/interrupt", data=b"", timeout=2).read()


_PING = {"t": 0.0, "ok": False, "fails": 0}


def is_alive(busy=False):
    """Cached liveness for a status dot. A round in flight is live proof it
    is up (never ping a busy worker - the event loop stalls under GPU load
    and the dot flickered red, user-reported). One slow answer is not
    death: only 3 consecutive failures turn it red."""
    if busy:
        _PING["ok"], _PING["fails"] = True, 0
        return True
    now = time.time()
    if now - _PING["t"] > 5:
        _PING["t"] = now
        try:
            urllib.request.urlopen(API_URL + "/system_stats", timeout=2).read()
            _PING["ok"], _PING["fails"] = True, 0
        except Exception:
            _PING["fails"] += 1
            if _PING["fails"] >= 3:
                _PING["ok"] = False
    return _PING["ok"]


def stream_progress(prompt_id, client_id, state, start):
    """Daemon thread body: stream ComfyUI's websocket events for prompt_id
    (None = everything, see tools/watch.py) and paint an in-place per-step
    progress bar. Best-effort - any failure just means no bar; the poll
    heartbeat in wait_entry still shows life."""
    try:
        import asyncio
        import aiohttp             # ComfyUI's own server dep, present in its venv
    except ImportError:
        return

    def bar(v, m, node):
        fill = "#" * (24 * v // m)
        print(f"\r  {node or 'step'} {v}/{m} [{fill:<24}] "
              f"{int(time.time() - start)}s\x1b[K", end="", flush=True)
        state["last"], state["bar"] = time.time(), True

    async def run():
        async with aiohttp.ClientSession() as sess:
            url = API_URL.replace("http", "ws", 1) + f"/ws?clientId={client_id}"
            async with sess.ws_connect(url, heartbeat=30) as ws:
                while not state["done"]:
                    try:
                        msg = await ws.receive(timeout=5)
                    except asyncio.TimeoutError:
                        continue
                    if msg.type in (aiohttp.WSMsgType.CLOSED,
                                    aiohttp.WSMsgType.CLOSING,
                                    aiohttp.WSMsgType.ERROR):
                        return
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    ev = json.loads(msg.data)
                    d = ev.get("data", {})
                    if prompt_id is not None and \
                            d.get("prompt_id") not in (None, prompt_id):
                        continue
                    if ev.get("type") == "progress" and d.get("max"):
                        bar(d["value"], d["max"], d.get("node"))
                    elif ev.get("type") == "progress_state":   # newer protocol
                        for nid, nd in d.get("nodes", {}).items():
                            if nd.get("state") == "running" and nd.get("max", 0) > 1:
                                bar(nd.get("value", 0), nd["max"], nid)

    try:
        asyncio.run(run())
    except Exception:
        pass


def wait_entry(prompt_id, timeout=300, client_id=CLIENT_ID):
    """Poll /history until the prompt completes, with live progress the
    whole time. Returns the raw history entry; raises ComfyError."""
    start = time.time()
    state = {"last": start, "bar": False, "done": False}
    threading.Thread(target=stream_progress,
                     args=(prompt_id, client_id, state, start),
                     daemon=True).start()
    try:
        deadline = start + timeout
        while time.time() < deadline:
            with urllib.request.urlopen(f"{API_URL}/history/{prompt_id}") as r:
                h = json.load(r)
            if prompt_id in h:
                entry = h[prompt_id]
                if state["bar"]:
                    print(flush=True)
                if entry["status"]["status_str"] != "success":
                    raise ComfyError(
                        f"render failed: {json.dumps(entry['status'], indent=1)}")
                return entry
            if time.time() - state["last"] > 15:
                print(f"\r  ... {int(time.time() - start)}s elapsed "
                      f"(loading models / queued)\x1b[K", end="", flush=True)
                state["bar"] = True
            time.sleep(3)
        if state["bar"]:
            print(flush=True)
        raise ComfyError(f"timed out waiting for {prompt_id}")
    finally:
        state["done"] = True


def output_images(entry):
    """History entry -> output image paths (SaveImage nodes)."""
    imgs = [o for out in entry["outputs"].values() for o in out.get("images", [])]
    return [OUTPUT_DIR / o["subfolder"] / o["filename"] for o in imgs]


def wait(prompt_id, timeout=300, client_id=CLIENT_ID):
    """Wait for a render (with live progress); return output image paths."""
    return output_images(wait_entry(prompt_id, timeout, client_id))


def render(payload, timeout=600, client_id=CLIENT_ID):
    """queue + wait in one call -> output image paths."""
    return wait(queue(payload), timeout, client_id)


def configure_console():
    """utf-8 + line-buffered stdout so redirects and `| tee` get lines
    immediately (Windows pipes default to cp1252 and block-buffering)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass
