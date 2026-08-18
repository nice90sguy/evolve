# akasutils — context for Claude

A set of utilities to facilitate designing **game characters, scenes, LoRA
adaptors, and identity-preserving transforms** for Ren'Py games, by driving
ComfyUI entirely through its HTTP API — the UI is not part of the loop.

## The project this serves

The user modded the game *Charmed* (`d:\home\josh\Downloads\charmed\Charmed-0.4-pc`)
extensively and is using those mods as a **testbed** for this art pipeline —
Charmed characters (the casino bunny girls, etc.) validate the tooling, but the
goal is **original game(s) built from scratch**, with the user's own characters
and scenes. Expect the Charmed-specific bits (output prefix `charmed2/`,
1920x1080 sprite format) to generalize into per-game config eventually.

Division of labor: the user is the art director — they look at images and give
feedback ("camera lower", "wrong hand", "100% fail"). Claude drives everything
headlessly: builds payloads, queues renders, **pre-screens outputs by reading
the image files** (crop the region under judgment for close inspection), and
only surfaces candidates worth the user's eye. The user's 👟 protocol applies:
push back / nail down (get their answer — never build a both-ways design around
an open question) / run with it.

## Environment

- ComfyUI at `D:\projects\ComfyUI` (this repo lives inside it); service assumed
  UP at `http://127.0.0.1:8188`; scripts run in the ComfyUI venv (`..\.venv`).
- GPU: RTX 5090 (32GB) — 4-step Klein renders take seconds; be generous with
  seed sweeps.
- Models: **FLUX.2 Klein 9B fp8** (`flux-2-klein-9b-fp8.safetensors`), text
  encoder `qwen_3_8b_fp8mixed.safetensors` (CLIPLoader type `flux2`), VAE
  `flux2-vae.safetensors`. Sampling: euler, `Flux2Scheduler` 4 steps, CFG 1.
- Identity: `IdentityFeatureTransferFinal` from `ComfyUI-Flux2Klein-Enhancer`
  (presets = similarity_floor/temperature: SOFT 0.5/0.07, MID 0.2/0.07,
  HARD 0.04/0.025; per-reference subject masks; `reference_indices` selects
  which refs drive identity vs merely steer composition).
- Matting: `rembg` + **isnet-anime** (user-verified better than InSPyReNet,
  which missed enclosed background pockets in hair); model cache `~/.u2net`.

## Pipeline conventions

- Payloads are API-format JSON; node ids in the templates/scripts:
  `unet clip vae img_i enc_i txt zero pos_i neg_i ift guider sampler sched
  noise latent samp dec save`. Reference images chain through `ReferenceLatent`
  in both positive and negative conditioning; first ref = primary identity
  (`reference_index` 0).
- **Generate on a solid WHITE background** — never black; dark clothing on
  black won't matte.
- Finalize step: matte → harden alpha (≤40→0, ≥220→255, stretch between; raw
  mattes leave interiors ~240–254 which ghosts over dark scenes) → trim →
  scale to 1080h → center on 1920x1080 transparent canvas → webp; always
  render the `_darkcheck.png` composite and inspect it.
- API-format JSON shows an EMPTY canvas if opened in the ComfyUI UI (no node
  geometry). `..\charmed_pipeline\api_to_ui.py` converts any payload to a
  loadable graph, synthesizing geometry via a topological layout and pulling
  exact socket/widget definitions from the live server's `/object_info`.
- Ren'Py side: sprites are registered EXPLICITLY (e.g. Charmed's
  `game/Pages/images/casino.rpy` maps `"tag attr attr"` names onto the
  underscore auto-image names) — new image files need an alias or shows fail
  with "does not accept attributes".

## Learnings (hard-won, from the rock-paper-scissors campaign)

Getting a specific HAND GESTURE (or any precise pose detail) onto a character:

1. **Winner: full re-render with an orientation-matched gesture crop as a
   secondary reference** (`--identity-refs 0` keeps identity on the primary).
   Crop a clean instance of the gesture (best source: an earlier render of the
   same character in the same style), **rotate/mirror it into the exact target
   orientation first** — the model copies orientation along with shape and
   will not mentally rotate a reference. Then it's a seed lottery with good
   odds; pre-screen crops of the target region.
2. Prompt-only gesture requests fail (~0/9): the model satisfies "hand" and
   "gesture" separately, parks gestures in genre-prior locations (raised
   beside the shoulder), and confuses gesture vocabulary (scissors → devil
   horns).
3. Masked inpainting (`SetLatentNoiseMask`) is for SMALL LOCAL variants, where
   it's excellent (expression/held-item swaps stay near-pixel-identical
   outside the mask because variants share the frozen latent). It's bad at
   inventing whole hands: generous masks grow phantom extra hands, tight
   masks make the hand vanish entirely (laziest valid completion).
4. Paste-then-refine (composite pixels + partial denoise) blends edges well at
   denoise ≥0.6 but struggles to bridge anatomy (floating wrists); the user
   rejected it — and manual touch-up doesn't scale to a hundred images.
5. Secondary refs pull their subject INTO the composition — a hand ref makes
   extra hands appear elsewhere (~5/8 renders); "her other hand is not
   visible" phrasing does not reliably suppress it. Pre-screen and reject.
6. Seed dominates lock level for detail quality; SOFT_LOCK is the proven
   default (user preference, verified repeatedly). When dialing in a shot,
   render all presets on one seed (`--locks SOFT_LOCK,MID_LOCK,HARD_LOCK`).
7. Camera moves: LEAD the prompt with the camera instruction and describe its
   visual consequences ("knees large in the foreground, underside of her jaw
   visible"); use a fresh seed — a kept seed anchors the old composition.
8. Always describe the character fully (hair, eyes, outfit piece by piece)
   even though refs are attached, then state "The only change: ...".

## Roadmap (the "LoRA adaptors" part is future work)

- Character canonicalization: user designs in Illustrious → one Klein
  identity-transfer pass → approved Klein-native master refs per character.
- Per-character LoRA training (kohya-style CLI, scriptable) once a character
  has ~20–40 approved renders; slots in as another util here.
- Scene-state variants (bedroom-night / bedroom-mess) via single-reference
  `ReferenceLatent` edit prompts — same identity machinery, scenes instead of
  characters.
