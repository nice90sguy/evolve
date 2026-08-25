"""jobs.py - the one-GPU job runner: at most ONE generator round or ONE
training job at a time, with a busy flag the UI polls, an abort flag the
operators check between candidates, and error capture (an operator that
fails prints to the console and the busy flag clears - never a wedged UI).
"""
import asyncio
import threading
import time


class Jobs:
    def __init__(self):
        self.busy = None          # {"total": n, "done": k, "tab": t} while generating
        self.abort = False        # set by /api/abort; checked between candidates
        self.train = None         # {"asset","family","running","error","log","started"}

    # ---------- generator rounds ----------

    def idle_error(self):
        """Why a new job cannot start now, or None."""
        if self.busy:
            return "already generating"
        if self.train and self.train.get("running"):
            return "training in progress"
        return None

    def should_abort(self):
        return self.abort

    def report(self, done, total=None, tab=None):
        if self.busy:
            self.busy["done"] = done
            if total is not None:
                self.busy["total"] = total
            if tab:
                self.busy["tab"] = tab

    def start_round(self, tab, total, fn, label="generate"):
        """Mark busy NOW (the client refreshes right after the request
        returns and must see a busy state or it stops polling), then run
        fn() in a worker thread; fn gets (progress, should_abort)."""
        self.abort = False
        self.busy = {"total": total, "done": 0, "tab": tab}
        loop = asyncio.get_event_loop()

        def work():
            return fn(lambda done: self.report(done), self.should_abort)

        async def run():
            try:
                await loop.run_in_executor(None, work)
            except Exception as e:
                print(f"{label} failed: {type(e).__name__}: {e}")
            finally:
                self.busy = None
        asyncio.ensure_future(run())

    def request_abort(self):
        if not self.busy:
            return False
        self.abort = True
        return True

    # ---------- training ----------

    def train_status(self):
        t = self.train
        if not t:
            return None
        return {k: t[k] for k in ("asset", "family", "running", "error", "log")} | \
            {"elapsed": int(time.time() - t["started"])}

    def start_training(self, asset_name, family, log_path, fn):
        """fn() does the work in a daemon thread; its exception (if any)
        becomes the visible error."""
        self.train = {"asset": asset_name, "family": family, "running": True,
                      "error": None, "started": time.time(), "log": str(log_path)}

        def work():
            try:
                fn()
            except BaseException as e:          # SystemExit from a wrapper too
                self.train["error"] = f"{type(e).__name__}: {e}"
            finally:
                self.train["running"] = False
        threading.Thread(target=work, daemon=True).start()

    def training_running(self):
        return bool(self.train and self.train.get("running"))
