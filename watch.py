"""Attach a live progress display to a ComfyUI run already in flight.

Usage:  python watch.py [client_id]     (default "tween")

Read-only: registers on ComfyUI's websocket under the given client_id and
paints the per-step sampler bar. Progress events are routed to the
client_id the run was QUEUED with, so it must match: akasutils scripts use
"tween" (tween.py) and "akasutils" (pose_from_char.py / evolve.py).
Ctrl-C to stop; never affects the run.

Only needed for runs launched without their own bar (pre-upgrade scripts,
or watching from a second terminal). One watcher per client_id at a time -
a second connection with the same id takes over the event stream.
"""
import sys
import time

from pose_from_char import _ws_progress

if __name__ == "__main__":
    client = sys.argv[1] if len(sys.argv) > 1 else "tween"
    print(f"watching client_id '{client}' - Ctrl-C to stop")
    state = {"last": time.time(), "bar": False, "done": False}
    try:
        _ws_progress(None, client, state, time.time())
    except KeyboardInterrupt:
        print()
