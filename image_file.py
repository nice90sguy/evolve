"""image_file.py - image FILES: hashing, text chunks, safe png writes,
hardlink staging and the <prefix>NNNNN.png output convention.

Images are IMMUTABLE once stored: nothing here ever rewrites a file in
place. Text chunks are PRESERVED, never dropped (layer 0: never destroy
image metadata on import).
"""
import hashlib
import io
import os
import re
import shutil
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from image_utils import flatten_alpha

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def sha1_of(data):
    return hashlib.sha1(data).hexdigest()


def text_chunks(img):
    """The string-valued metadata of an opened image (png tEXt chunks)."""
    return {k: v for k, v in img.info.items() if isinstance(v, str)}


def open_bytes(data):
    """Decode image bytes -> (loaded PIL image, its text chunks)."""
    img = Image.open(io.BytesIO(data))
    img.load()
    return img, text_chunks(img)


def flattened_rgb(img):
    """RGB copy, alpha flattened onto white (the import/staging cadence)."""
    return flatten_alpha(img)


def write_png(img, path, chunks=None):
    """Save with EVERY given text chunk re-embedded (a bare img.save() would
    destroy them)."""
    info = PngInfo()
    for k, v in (chunks or {}).items():
        if isinstance(v, str):
            info.add_text(k, v)
    img.save(path, "PNG", pnginfo=info)


def link_or_copy(src, dst):
    """Hardlink (free, passes ComfyUI's input-dir containment honestly -
    realpath is a no-op on hardlinks); copy2 is the cross-volume fallback."""
    dst = Path(dst)
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def list_images(folder, recursive=True):
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(q for q in it if q.is_file() and q.suffix.lower() in IMAGE_EXTS)


def next_index(out_dir, prefix):
    """First free canonical index: existing <prefix>NNNNN.png files are never
    overwritten, numbering continues after the highest one."""
    pat = re.compile(re.escape(prefix) + r"(\d{5})\.png$")
    taken = [int(m.group(1)) for p in Path(out_dir).glob(f"{prefix}*.png")
             if (m := pat.fullmatch(p.name))]
    return max(taken, default=0) + 1


def save_render(src, out_dir, prefix, index, size=None):
    """Copy a scratch render to out_dir/<prefix><index:05d>.png. COPY, not
    move: resubmitting an identical graph is a full ComfyUI cache hit whose
    history points at the PREVIOUS run's scratch file - it must still be
    there. If the render was snapped to /16, resize the copy back to the
    exact target size (metadata dropped in that case)."""
    dest = Path(out_dir) / f"{prefix}{index:05d}.png"
    img = Image.open(src)
    if size and img.size != size:
        img.resize(size, Image.LANCZOS).save(dest)
    else:
        shutil.copy2(str(src), str(dest))
    return dest


def save_renders(srcs, out_dir, prefix, size=None, log=print):
    """The tools' output cadence: mkdir, continue numbering, copy each."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = next_index(out_dir, prefix)
    dests = []
    for src in srcs:
        dests.append(save_render(src, out_dir, prefix, n, size))
        n += 1
        if log:
            log(f"rendered -> {dests[-1]}")
    return dests
