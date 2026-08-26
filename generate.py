"""generate.py - the Generate operator: Create (fiat) and Derive (breeding).

do_generate(store, GenerateConfig) sweeps the tab's unkept candidates,
then fills every slot with one candidate each (seed = base + i), each its
own ComfyUI submission, live per-step progress on the console, the slot
filling in as each lands. Knows nothing about widgets: the HTTP layer
turns controls into a GenerateConfig (a validated pydantic model - bad
values are rejected there, never silently coerced here).
"""
import random
from typing import List, Literal

from PIL import Image
from pydantic import BaseModel, Field, model_validator

from build_payload import flux_klein, illustrious, zimage
from build_payload.flux_klein import LOCKS, WHITE_BG_PREFIX
from comfy_client import free_vram, queue, wait
from model_family import DEFAULT_MODEL_FAMILY, ModelFamily, family_info, parse_model_family
from image_meta import harvest_chunks
from image_utils import snap16
from lora import resolve
from trash import sweep

MAX_VARY = 0.5      # additive randn on the reference latent: useful band 0.1-0.3
MAX_SIDE = 4096


def random_seed():
    return random.randint(1, 999_999_999_999)


class GenerateConfig(BaseModel):
    op: Literal["create", "derive"] = "create"
    prompt: str = ""
    negative: str = ""
    family: ModelFamily = DEFAULT_MODEL_FAMILY
    ref_ids: List[int] = Field(default_factory=list)   # [ref0, extras...], alive ids only
    has_ref0: bool = False                            # was slot 0 the continuity carrier?
    lora: str = ""                                    # dropdown value (alias or path)
    lora_strength: float = Field(1.0, ge=0.0, le=4.0)
    lock: Literal["SOFT_LOCK", "MID_LOCK", "HARD_LOCK"] = "SOFT_LOCK"
    vary: float = Field(0.0, ge=0.0, le=MAX_VARY)
    seed: int = Field(0, ge=0)                        # 0 = random
    whitebg: bool = True
    width: int = Field(1024, ge=16, le=MAX_SIDE, multiple_of=16)
    height: int = Field(1024, ge=16, le=MAX_SIDE, multiple_of=16)
    steps: int = Field(0, ge=0, le=200)               # 0 = family default
    cfg: float = Field(0.0, ge=0.0, le=30.0)
    fresh_model: bool = False

    @model_validator(mode="after")
    def _family_can_do_this(self):
        if self.op == "derive" and self.family != ModelFamily.KLEIN:
            raise ValueError("Derive is Klein-only (references + identity transfer)")
        if self.ref_ids and not family_info(self.family).references:
            raise ValueError(f"{self.family} takes no reference images")
        if self.lora and not family_info(self.family).lora:
            raise ValueError(f"{self.family} cannot apply a LoRA")
        if self.has_ref0 and not self.ref_ids:
            raise ValueError("has_ref0 without a ref0")
        return self

    @classmethod
    def from_controls(cls, c, op):
        """Sanitised controls (dead ids already dropped) -> config. Create
        is the no-refs gate; Derive is Klein-only; references force klein.
        Raises pydantic.ValidationError on anything out of range."""
        ref0 = c.get("ref0")
        refs = [ref0] if (op != "create" and ref0 is not None) else []
        if op != "create":
            refs += [i for i in (c.get("refs") or []) if i is not None]
        family = parse_model_family(c.get("family"), DEFAULT_MODEL_FAMILY)
        if refs or op == "derive":
            family = ModelFamily.KLEIN
        lora = c.get("lora") or ""
        if not family_info(family).lora:
            lora = ""                                  # e.g. SDXL: LoRAs later
        return cls(op=op, prompt=c.get("prompt") or "", negative=c.get("negative") or "",
                   family=family, ref_ids=refs, has_ref0=bool(refs) and ref0 is not None,
                   lora=lora, lora_strength=float(c.get("lora_strength", 1.0)),
                   lock=c.get("lock") or "SOFT_LOCK",
                   vary=min(MAX_VARY, max(0.0, float(c.get("vary") or 0.0))),
                   seed=int(c.get("seed") or 0), whitebg=bool(c.get("whitebg", True)),
                   width=snap16(int(c.get("width") or 1024)),
                   height=snap16(int(c.get("height") or 1024)),
                   steps=int(c.get("steps") or 0), cfg=float(c.get("cfg") or 0),
                   fresh_model=bool(c.get("fresh_model")))


def do_generate(store, cfg, progress=None, should_abort=None, embed_workflow=False):
    """Rule (a): sweep + clear the tab's outputs, then fill every slot from
    the config. Returns the new image ids."""
    tab = "create" if cfg.op == "create" else "derive"
    sweep(store, tab)                    # unkept candidates -> archive, quietly
    n = store.begin_round(tab)
    if n <= 0:
        return []
    fam = family_info(cfg.family)
    steps = cfg.steps or fam.steps
    cfg_scale = cfg.cfg or fam.cfg
    refs, inputs = [], {}
    for i in cfg.ref_ids:
        name = store.stage_ref(i)
        refs.append(name)
        inputs[name] = i
    prompt = (WHITE_BG_PREFIX + cfg.prompt) if cfg.whitebg else cfg.prompt
    base = cfg.seed or random_seed()
    lora_abs = resolve(cfg.lora, cfg.family) if cfg.lora else None
    if cfg.lora and lora_abs is None:
        raise FileNotFoundError(f"no {cfg.family} LoRA for {cfg.lora!r}")
    lock = cfg.lock if refs else None
    vary = cfg.vary if refs else 0.0

    # ComfyUI bug #11021: LoRA deltas compound on the shared base weights
    # across runs. The only clean reset is /free with free_memory. Do it
    # automatically after any round that used a LoRA, or every round if asked.
    if cfg.fresh_model or store.state.get("last_round_lora"):
        print("reloading models (fresh weights: "
              + ("forced" if cfg.fresh_model else "previous round used a LoRA") + ")")
        free_vram()
    store.note_round(base, cfg.lora)
    mother = cfg.ref_ids[0] if cfg.has_ref0 else None
    store.touch(cfg.ref_ids)                 # being bred from is a touch
    if mother is not None:
        store.hist_append(mother)            # history: bred-from only, co-parents excluded
    tags = store.birth_tags(mother)          # mother's words (minus pinned) + defaults
    dest = store.birth_dir(mother)           # bred images land in the mother's folder

    ids = []
    for k in range(n):
        if should_abort and should_abort():
            print(f"round aborted at {k}/{n}")
            break
        seed = base + k
        if cfg.family == ModelFamily.ZIMAGE:
            payload = zimage.build(cfg.prompt, cfg.negative, seed, cfg.width, cfg.height,
                                   steps, cfg_scale, lora_abs, cfg.lora_strength)
        elif cfg.family == ModelFamily.ILLUSTRIOUS:
            payload = illustrious.build(cfg.prompt, cfg.negative, seed, cfg.width,
                                        cfg.height, steps, cfg_scale)
        else:
            payload = flux_klein.build(prompt, seed, cfg.width, cfg.height, refs, lock, vary,
                                       lora_abs, cfg.lora_strength,
                                       steps if steps != fam.steps else None,
                                       cfg_scale if cfg_scale != fam.cfg else None)
        pid = queue(payload)
        print(f"candidate {k + 1}/{n} [{fam.label}] -> {pid} (seed {seed})")
        src = wait(pid, timeout=600)[0]
        recipe = {"op": cfg.op, "prompt": cfg.prompt, "negative": cfg.negative,
                  "family": cfg.family.value, "steps": steps, "cfg": cfg_scale,
                  "whitebg": cfg.whitebg,
                  "seed": seed, "width": cfg.width, "height": cfg.height,
                  "lock": lock, "vary": vary, "lora": cfg.lora or None,
                  "lora_file": str(lora_abs) if lora_abs else None,
                  "lora_strength": cfg.lora_strength if cfg.lora else None,
                  "ref0": cfg.has_ref0}
        with Image.open(src) as im:
            i = store.add_image(im.convert("RGB"), "gen", recipe, list(cfg.ref_ids),
                                chunks=harvest_chunks(im, payload, embed_workflow),
                                inputs=inputs, tags=tags, dir=dest, fresh=True)
        store.add_candidate(tab, i)
        ids.append(i)
        if progress:
            progress(k + 1)
    return ids


__all__ = ["GenerateConfig", "do_generate", "random_seed", "LOCKS"]
