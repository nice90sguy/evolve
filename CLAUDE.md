# akasutils — context for Claude

A set of utilities to facilitate designing **game characters, scenes, LoRA
adaptors, and identity-preserving transforms** for Ren'Py games, by driving
ComfyUI entirely through its HTTP API — the ComfyUI UI is not part of the
loop. This file carries everything learned building the pipeline, so a fresh
session starts fully up to speed.

## The project this serves

The user modded the game *Charmed* (`d:\home\josh\Downloads\charmed\Charmed-0.4-pc`)
extensively and used those mods as the **testbed** for this pipeline — Charmed
characters (casino bunny girls etc.) validated the tooling. The goal is
**original game(s) built from scratch**: storyboarding and dialog by the user,
art produced through this pipeline. The `charmed2/` output prefix and the
1920x1080 sprite deliverable format are testbed-era conventions that will
generalize into per-game config.

The user has **already created many character LoRAs** (they will point to them
when relevant — location not yet known). Note: their character design work is
Illustrious/SDXL-era; those LoRAs will not load into Flux.2 Klein — treat them
as design references / training-data sources, not as loadable adapters, until
told otherwise.

## Working with the user

- The user is the art director: they look at images and give terse verdicts
  ("camera lower", "wrong hand", "100% FAIL", "almost 👌🏻"). Claude operates
  everything headlessly and iterates.
- **Claude pre-screens every batch by reading the output PNGs directly** (the
  Read tool renders images). Standard technique: crop the region under
  judgment (e.g. the hand) with PIL and read the crop — small regions judge
  far better than full frames. Only surface candidates worth the user's eye,
  and report failures honestly, per failure mode.
- **👟 protocol** (applies to every request): 1) push back if the request
  doesn't make sense; 2) nail down any ambiguity — meaning GET THE USER'S
  ANSWER, never build a "works either way" design around an open question;
  3) run with it only when genuinely all-clear. 👟 in a message invokes this
  and does not always mean "code away".
- Content boundary, settled explicitly: Claude does not author prompts for or
  drive renders of explicit sexual imagery. The pipeline/infrastructure is
  content-neutral and fully documented, so the user runs those themselves
  (also: Klein is expected to be poor at explicit content — a model-training
  gap, not a workflow bug). Suggestive/pin-up level content is fine.

## Environment

- ComfyUI at `D:\projects\ComfyUI` (this repo lives inside it), version ~0.32,
  venv at `..\.venv` (Python 3.12, torch cu130). Launch: activate the venv,
  `python main.py --enable-manager` (the old ComfyUI-Manager folder is
  `.DISABLED`; the manager is core-integrated).
- **All scripts here assume the service is UP** at `http://127.0.0.1:8188`
  (health check: `GET /system_stats`). Kill: find the PID listening on 8188.
- GPUs: RTX 5090 32GB (primary — a 4-step Klein render takes seconds, so be
  generous with seed sweeps) + RTX 2080 8GB.
- Models (all installed): **FLUX.2 Klein 9B fp8**
  (`flux-2-klein-9b-fp8.safetensors`), text encoder
  `qwen_3_8b_fp8mixed.safetensors` (CLIPLoader type `flux2`), VAE
  `flux2-vae.safetensors`.
- Key custom packs: `ComfyUI-Flux2Klein-Enhancer` (identity nodes),
  `comfyui-nkd-klein-tools` (NKDKleinPresampling/Postsampling — used by the
  user's downloaded template workflows, not by our pipeline), kjnodes,
  cg-use-everywhere, comfyui_essentials, impact pack, controlnet_aux.

## Identity transfer — node knowledge

- Use **`IdentityFeatureTransferFinal`** (in the Enhancer pack's
  `identity_feature_transfer.py`). The older V3 exists only for compatibility
  with the user's downloaded template workflows (myaiforce "F2K Identity
  Transfer" single/multi reference — kept as comparison baselines).
- Final's presets are pure similarity gates — block schedules are identical
  across presets; only `similarity_floor`/`softmax_temperature` differ:
  SOFT 0.5/0.07, MID 0.2/0.07, HARD 0.04/0.025. `custom` allows sweeping the
  floor continuously if a character holds identity poorly.
- **SOFT_LOCK is the proven default** (user-verified repeatedly): identity
  holds while pose/camera stay free. Hard locks anchor composition to the
  reference framing. When dialing in a new shot/character, render all three
  presets on ONE seed and let the user pick.
- Refs enter as `ReferenceLatent` chains (core nodes) in BOTH positive and
  negative conditioning. `reference_index`/`reference_indices` on the Final
  node select which refs drive IDENTITY; all refs steer composition
  regardless. Convention: first ref = primary identity (`reference_index` 0).
  Final also supports per-reference subject masks (1–8) — the future route to
  two characters in one CG without feature bleed — and optional sigma-decay.
- Identity transfer is style-agnostic feature matching on VAE latents (not a
  face model): references in any style work, but the OUTPUT is rendered in
  Klein's style. Plan for original characters: design in Illustrious → one
  Klein pass → approved **Klein-native master reference** per character → all
  subsequent renders reference the master. Never mix raw Illustrious renders
  with Klein output in one game (style drift).

## Driving the API — patterns

- Payloads are API-format JSON: `{"client_id": ..., "prompt": {node_id:
  {class_type, inputs}}}` to `POST /prompt`; poll `GET /history/<prompt_id>`
  until the id appears; output filenames are in `outputs.<node>.images`;
  files land in `D:\projects\ComfyUI\output\<subfolder>`. Input images must
  be staged into `D:\projects\ComfyUI\input\` first.
- Node-id vocabulary used across all templates/scripts: `unet clip vae img_i
  enc_i txt zero pos_i neg_i ift guider sampler sched noise latent samp dec
  save` (+ `msk lat` in inpaint graphs, `split` for partial denoise).
- Sampling: euler via `KSamplerSelect` + `Flux2Scheduler` (4 steps) + 
  `CFGGuider` cfg 1.0 + `SamplerCustomAdvanced`; negative = 
  `ConditioningZeroOut` of the prompt, ref-chained like the positive.
- API-format JSON opened in the ComfyUI UI shows an EMPTY canvas (no node
  geometry — a silent no-op). `..\charmed_pipeline\api_to_ui.py` converts any
  payload into a loadable graph: topological auto-layout, with exact
  socket/widget definitions fetched from the live server's
  `/object_info/<class>` (widget vs socket: combo lists and INT/FLOAT/STRING/
  BOOLEAN are widgets; `control_after_generate` inputs get an extra "fixed"
  widget value).
- Process lessons: write Python drivers as FILES and run them (long inline
  `python -c` in PowerShell trips quoting and command filters); guard scripts
  with `if __name__ == "__main__":` — importing a helper from a script whose
  queue/poll code is module-level re-runs the whole batch.

## Matting / finalize

- `finalize.py`: **rembg + isnet-anime** (user-verified better than
  InSPyReNet/`transparent_background`, which missed an enclosed white pocket
  in hair — both are installed). First isnet-anime use downloads ~176MB to
  `~/.u2net` (already cached). ~5s/image warm.
- Raw mattes from BOTH tools leave character interiors at alpha 240–254
  (~2–5% translucent → ghosting over dark scenes). Fix: harden the matte
  (≤40→0, ≥220→255, linear stretch between) — keeps anti-aliased edges,
  matches game-sprite near-binary alpha.
- Then trim to alpha bbox → scale to 1080 height → center on a 1920x1080
  transparent canvas → webp (quality 95). ALWAYS write and inspect the
  `_darkcheck.png` (composite over dark charcoal): white-halo/pocket/edge
  failures are invisible on light backgrounds.
- **Generate on a solid WHITE background, never black** — matting dark
  clothing off black fails; "black background" requests are delivered as
  transparent sprites instead.
- Trim+scale normalization makes separately-generated sprites of the same
  character converge to near-identical geometry (verified: a full re-render's
  face landed on the same image-point as the inpainted variants').

## Hard-won generation learnings (the rock-paper-scissors campaign)

Getting a specific hand gesture (or any precise pose detail) onto a character
— the full failure ladder, so it is never re-climbed:

1. **Winner: full re-render with an orientation-matched gesture crop as a
   secondary reference**, identity pinned to the primary (`--identity-refs 0`
   conceptually; "all" also worked since the crop was the same character's
   hand). Crop a clean instance of the gesture — best source is an earlier
   render of the same character in the same style, even a failed one —
   **mirror/rotate it into the exact target orientation first** (the model
   copies orientation with shape; it will not mentally rotate a reference),
   place it on white, clean stray non-white fragments from the crop (they
   teach the model to grow hair/objects). Then seed-sweep; hit rate ~2/8.
2. Prompt-only gesture requests: 0/9. The model satisfies "hand" and
   "gesture" separately (parks the gesture in genre-prior locations like
   raised-beside-the-shoulder), and confuses gesture vocabulary (scissors →
   devil horns). Naming the gesture first and describing both hands helps but
   does not fix it.
3. Masked inpainting (`SetLatentNoiseMask` on a feathered mask, full-strength
   sigmas) is EXCELLENT for small local variants — rock and paper gestures
   landed in one round when the hand only needed re-posing in place — but bad
   at inventing structure: generous masks grow phantom extra hands, tight
   masks make the hand vanish (the laziest valid completion). Locate mask
   coordinates by cropping the source image and looking at it.
4. Variant sets made by inpainting one base share the frozen latent outside
   the mask: outputs differ by max 4–6/255 there (VAE decode ripple at the
   feathered boundary) — visually identical, flip cleanly in-game. For
   bit-identical, composite the masked region back onto the base PNG before
   matting (**offered, not yet implemented**).
5. Paste-then-refine (composite gesture pixels, partial denoise via
   `SplitSigmas` low half on an 8-step schedule): denoise ≤0.5 barely touches
   the paste (halo survives); 0.6–0.75 blends edges and adds detail but
   struggles to bridge anatomy (floating wrist/cuff gaps); a follow-up
   junction-only full-denoise pass connected it but degraded the gesture.
   Erode+blur the paste alpha to kill white fringe before refining. The user
   rejected this route — and per-image manual touch-up doesn't scale.
6. Secondary refs pull their subject INTO the composition: a hand ref made
   extra hands appear elsewhere in ~5/8 renders; "her other hand is not
   visible" phrasing does not reliably suppress it. Pre-screen and reject.
7. **Seed dominates lock level** for detail quality — SOFT vs MID changed
   hands very little; the seed changed everything. Sweep seeds, not knobs.
8. Camera moves: LEAD the prompt with the camera instruction and describe its
   visual consequences ("her knees large in the foreground, the underside of
   her jaw visible"); a kept seed anchors the old composition — use a fresh
   one. Burying "low angle" mid-prompt does nothing.
9. Reproduction prompts: describe the character fully (hair, eyes, outfit
   piece by piece) even though refs are attached, then state "The only
   change: ...". Femininity/nails drift on hands unless stated ("slender
   feminine hand, pink nail polish").

## Repo contents & siblings

- `generate_from_refs.py` — the workhorse: multi-ref identity render
  (`--ref "primary.webp,extra.webp"`, `--locks SOFT_LOCK,MID_LOCK,...` on a
  shared seed with `_soft/_mid/_hard` filename tags, `--identity-refs`,
  `--finalize`); writes `last_payload.json` (gitignored) every run.
- `finalize.py` — matting step, importable (`from finalize import finalize`)
  and CLI.
- `template_single_ref.json` — the API-format graph `generate_from_refs.py`
  patches (single-ref nodes are replaced by a built chain at runtime).
- `requirements.txt` — documents deps; the ComfyUI venv already satisfies it.
- Sibling `..\charmed_pipeline\` — legacy PowerShell version (`generate.ps1`,
  same features), `api_to_ui.py`, README with the manual curl-level steps,
  and the original experiment payloads.

## Roadmap

- Character canonicalization util (Illustrious design → Klein master ref).
- Scene-state variants (bedroom-night / bedroom-mess): single-reference
  `ReferenceLatent` edit prompts — layout holds, prompt the delta.
- Multi-character scenes via Final's per-reference subject masks.
- LoRA tooling: the user's existing character LoRAs first (location TBD);
  training new ones (kohya-style CLI) only once characters have approved
  render sets and a need Klein references can't meet.
