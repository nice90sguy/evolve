"""imgmeta.py - dump an image's embedded metadata as JSON, and nothing else.

    python imgmeta.py IMAGE [IMAGE ...]           all chunks, one JSON object
    python imgmeta.py IMAGE -k prompt             just that chunk
    python imgmeta.py IMAGE --keys                list chunk names
    python imgmeta.py IMAGE | jq '.prompt."68"'   stdout is always valid JSON

Chunks whose text parses as JSON are NESTED, not left as escaped strings, so
jq can walk straight into them. ComfyUI PNGs carry two: `prompt` (the
API-format payload our drivers post to /prompt) and `workflow` (the UI graph
with node geometry, usually the bulk of the file).

Binary chunks (icc_profile, raw exif) cannot be JSON, so they appear as
{"_bytes": N} rather than being silently dropped. EXIF UserComment and
ImageDescription are also checked, since webp/jpeg keep generator metadata
there instead of in PNG text chunks.

NOTE for provenance work: images written by evolve.py carry NO embedded
metadata (Store.add_image does a bare img.save). Their lineage lives in the
store's journal.jsonl - recipe + parents per image id. Walking an image back
to its fiat/eve ancestor is a journal query, not an image query; this tool
will correctly report {} for them.
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXIF_TEXT_TAGS = {270: "ImageDescription", 37510: "UserComment"}


def _coerce(v, raw=False):
    """One metadata value -> something JSON can hold, preferring structure."""
    if isinstance(v, (bytes, bytearray)):
        return {"_bytes": len(v)}
    if isinstance(v, str):
        if not raw and v.lstrip()[:1] in "{[":
            try:
                return json.loads(v)
            except ValueError:
                pass
        return v
    if isinstance(v, tuple):
        return list(v)
    try:
        json.dumps(v)
        return v
    except TypeError:
        return repr(v)


def read_meta(path, raw=False):
    out = {}
    with Image.open(path) as im:
        for k, v in im.info.items():
            out[k] = _coerce(v, raw)
        # webp/jpeg keep generator metadata in EXIF, not in text chunks
        try:
            ex = im.getexif()
        except Exception:
            ex = None
        for tag, name in EXIF_TEXT_TAGS.items():
            if ex and tag in ex and name not in out:
                v = ex[tag]
                if isinstance(v, bytes):
                    v = v.split(b"\x00")[-1].decode("utf-8", "replace")
                out[name] = _coerce(v, raw)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("-k", "--key", help="emit only this chunk")
    ap.add_argument("--keys", action="store_true", help="emit the chunk names only")
    ap.add_argument("--raw", action="store_true",
                    help="leave embedded JSON as escaped strings")
    ap.add_argument("-c", "--compact", action="store_true")
    a = ap.parse_args()

    def one(p):
        m = read_meta(p, a.raw)
        if a.keys:
            return sorted(m)
        if a.key:
            return m.get(a.key)
        return m

    paths = [Path(p) for p in a.images]
    for p in paths:
        if not p.is_file():
            sys.exit(f"no such file: {p}")
    # one image -> the object itself; several -> keyed by path, so the shape
    # is predictable per invocation rather than per file count
    doc = one(paths[0]) if len(paths) == 1 else {str(p): one(p) for p in paths}
    json.dump(doc, sys.stdout, indent=None if a.compact else 1,
              ensure_ascii=False, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
