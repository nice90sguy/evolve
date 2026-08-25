"""Attach a live progress display to a ComfyUI run already in flight.

    python tools/watch.py [client_id]     (default "evolve")

Read-only: registers on ComfyUI's websocket under the given client_id and
paints the per-step sampler bar. Progress events are routed to the
client_id the run was QUEUED with, so it must match: evolve and the tools
use "evolve"; tween.py uses "tween". Ctrl-C to stop; never affects the run.
One watcher per client_id at a time - a second connection with the same id
takes over the event stream.
"""
import sys
import time

import _cli  # noqa: F401

from comfy_client import CLIENT_ID, stream_progress

if __name__ == "__main__":
    client = sys.argv[1] if len(sys.argv) > 1 else CLIENT_ID
    print(f"watching client_id '{client}' - Ctrl-C to stop")
    state = {"last": time.time(), "bar": False, "done": False}
    try:
        stream_progress(None, client, state, time.time())
    except KeyboardInterrupt:
        print()
