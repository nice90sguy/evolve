"""tools/_cli.py - import this first in every tool: puts the evolve package
dir on sys.path (tools/ is a sibling of the modules, not inside them) and
makes the console utf-8 + line-buffered so `| tee` and redirects stream.

    import _cli            # noqa: F401
    from comfy_client import queue, wait
"""
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from comfy_client import ComfyError, configure_console  # noqa: E402

configure_console()


def exit_on_comfy_error(fn):
    """Run fn(); a ComfyError becomes a clean exit with its message."""
    try:
        return fn()
    except ComfyError as e:
        sys.exit(str(e))
