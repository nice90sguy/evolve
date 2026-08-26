"""camera.py - the Camera operator: re-shoot the working image at an
ABSOLUTE camera spec with the Qwen multiple-angles LoRA (budding: single
parent = the WI).

Trained grammar is a closed enum: `<sks> <azimuth> <elevation> <distance>`.
The tokens SET the camera in the subject's frame, they do not delta it
("close-up" on an already-close image is a no-op); each axis is optional -
an unchecked axis OMITS its token. Seed never moves the angle: N outputs
= N takes of the SAME camera. Findings 1-14 of the viewpoint campaign are
in CLAUDE.md; the short version: elevation and distance are the reliable
axes, azimuth needs background geometry for chirality, ~1/4 turn envelope.
"""
import re
from typing import Optional

from PIL import Image
from pydantic import BaseModel, Field, field_validator, model_validator

from build_payload import qwen_edit
from comfy_client import free_vram, queue, wait
from image_meta import harvest_chunks
from trash import sweep

# alias -> the exact trained token (do not paraphrase the values)
AZIMUTH = {
    "front":       "front view",                  # 0
    "front-right": "front-right quarter view",    # 45
    "right":       "right side view",             # 90
    "back-right":  "back-right quarter view",     # 135
    "back":        "back view",                   # 180
    "back-left":   "back-left quarter view",      # 225
    "left":        "left side view",              # 270
    "front-left":  "front-left quarter view",     # 315
}
ELEVATION = {
    "low":      "low-angle shot",    # -30
    "eye":      "eye-level shot",    #   0
    "elevated": "elevated shot",     #  30
    "high":     "high-angle shot",   #  60
}
DISTANCE = {
    "close":  "close-up",      # x0.6
    "medium": "medium shot",   # x1.0
    "wide":   "wide shot",     # x1.8
}
ALIASES = {"fr": "front-right", "br": "back-right", "bl": "back-left",
           "fl": "front-left", "closeup": "close", "close-up": "close",
           "med": "medium", "eye-level": "eye"}

# (value, label) lists for the UI - trained names ARE the labels
AXIS_AZIMUTH = [(k, k) for k in AZIMUTH]
AXIS_ELEVATION = [("low", "low-angle"), ("eye", "eye-level"),
                  ("elevated", "elevated"), ("high", "high-angle")]
AXIS_DISTANCE = [("close", "close-up"), ("medium", "medium"), ("wide", "wide")]

STEPS, CFG, STRENGTH = qwen_edit.DEFAULT_STEPS, qwen_edit.DEFAULT_CFG, qwen_edit.DEFAULT_STRENGTH
LORA_LABEL = "qwen multiple-angles + lightning"


def camera_prompt(azim=None, elev=None, dist=None):
    """The token string for the checked axes (None = omitted)."""
    toks = [AZIMUTH[azim] if azim else None, ELEVATION[elev] if elev else None,
            DISTANCE[dist] if dist else None]
    return "<sks> " + " ".join(t for t in toks if t)


def valid_axes(azim, elev, dist):
    return ((not azim or azim in AZIMUTH) and (not elev or elev in ELEVATION)
            and (not dist or dist in DISTANCE))


def parse_vp(spec):
    """'[VP: right, low, medium]' or 'right,low,medium' -> (prompt, tag).
    Missing fields default to eye-level / medium shot (the probe's full
    grammar). ValueError on unknown tokens."""
    body = spec.strip()
    m = re.fullmatch(r"\[\s*VP\s*:\s*(.*?)\s*\]", body, re.I)
    if m:
        body = m.group(1)
    parts = [p.strip().lower() for p in re.split(r"[,\s]+", body) if p.strip()]
    parts = [ALIASES.get(p, p) for p in parts]
    az, el, di = None, "eye", "medium"
    for p in parts:
        if p in AZIMUTH:
            az = p
        elif p in ELEVATION:
            el = p
        elif p in DISTANCE:
            di = p
        else:
            raise ValueError(f"unknown viewpoint token {p!r} in {spec!r}\n"
                             f"  azimuth:   {', '.join(AZIMUTH)}\n"
                             f"  elevation: {', '.join(ELEVATION)}\n"
                             f"  distance:  {', '.join(DISTANCE)}")
    if az is None:
        raise ValueError(f"no azimuth in {spec!r} (one of: {', '.join(AZIMUTH)})")
    return camera_prompt(az, el, di), f"{az}-{el}-{di}"


class CameraConfig(BaseModel):
    source_id: int = Field(ge=1)          # the WI
    azim: Optional[str] = None            # None = axis unchecked, token omitted
    elev: Optional[str] = None
    dist: Optional[str] = None
    seed: int = Field(0, ge=0)            # 0 = random

    @field_validator("azim")
    @classmethod
    def _azim(cls, v):
        if v is not None and v not in AZIMUTH:
            raise ValueError(f"bad azimuth {v!r} (one of: {', '.join(AZIMUTH)})")
        return v

    @field_validator("elev")
    @classmethod
    def _elev(cls, v):
        if v is not None and v not in ELEVATION:
            raise ValueError(f"bad elevation {v!r} (one of: {', '.join(ELEVATION)})")
        return v

    @field_validator("dist")
    @classmethod
    def _dist(cls, v):
        if v is not None and v not in DISTANCE:
            raise ValueError(f"bad distance {v!r} (one of: {', '.join(DISTANCE)})")
        return v

    @model_validator(mode="after")
    def _some_axis(self):
        if not self.any_axis():
            raise ValueError("no camera axis selected")
        return self

    def any_axis(self):
        return bool(self.azim or self.elev or self.dist)


def do_camera(store, cfg, progress=None, should_abort=None, embed_workflow=False):
    """N takes of the same camera on the WI; returns the new image ids."""
    from generate import random_seed
    tab = "camera"
    with store.lock:
        c = store.state["controls"]
        c["pov_azim"], c["pov_elev"], c["pov_dist"] = cfg.azim, cfg.elev, cfg.dist
    sweep(store, tab)
    n = store.begin_round(tab)
    if n <= 0 or not store.alive(cfg.source_id) or not cfg.any_axis():
        return []
    prompt = camera_prompt(cfg.azim, cfg.elev, cfg.dist)
    name = store.stage_ref(cfg.source_id)
    inputs = {name: cfg.source_id}
    base = cfg.seed or random_seed()
    if store.state.get("last_round_lora"):      # always a LoRA round: #11021 rule
        free_vram()
    store.note_round(base, True)
    store.hist_append(cfg.source_id)            # history: Camera appends its true source
    tags = store.birth_tags(cfg.source_id)
    dest = store.birth_dir(cfg.source_id)
    ids = []
    for k in range(n):
        if should_abort and should_abort():
            print(f"round aborted at {k}/{n}")
            break
        seed = base + k
        payload = qwen_edit.build(name, prompt, "", seed, STEPS, CFG, STRENGTH,
                                  lightning=True, out_prefix="pov")
        pid = queue(payload)
        print(f"camera {k + 1}/{n} [{cfg.azim}/{cfg.elev}/{cfg.dist}] -> {pid} (seed {seed})")
        src = wait(pid, timeout=600)[0]
        recipe = {"op": "pov", "pov_azim": cfg.azim, "pov_elev": cfg.elev, "pov_dist": cfg.dist,
                  "prompt": prompt, "negative": "", "family": "qwen_edit",
                  "steps": STEPS, "cfg": CFG, "seed": seed,
                  "lora": LORA_LABEL, "ref0": True}
        with Image.open(src) as im:
            i = store.add_image(im.convert("RGB"), "pov", recipe, [cfg.source_id],
                                chunks=harvest_chunks(im, payload, embed_workflow),
                                inputs=inputs, tags=tags, dir=dest)
        store.add_candidate(tab, i)
        ids.append(i)
        if progress:
            progress(k + 1)
    return ids
