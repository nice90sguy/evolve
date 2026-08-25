# akasutils

Utilities for driving ComfyUI headlessly (character renders for the game art
pipeline). **All scripts assume the ComfyUI service is up** at
`http://127.0.0.1:8188` and run in the **ComfyUI venv**
(`D:\projects\ComfyUI\.venv` — see `requirements.txt`).

All render/tween waits stream **live progress** to the console: per-step
sampler bars (via ComfyUI's websocket) plus an elapsed heartbeat while
models load. Independent fallbacks:
`tail -f /d/projects/ComfyUI/user/comfyui.log` (ComfyUI's mirrored console,
including the sampler tqdm) and `nvidia-smi -l 2`.

## Scripts

### pose_from_char.py

Pose/expression variants of a character from one or more reference images
(FLUX.2 Klein + IdentityFeatureTransferFinal), matted to a transparent webp.
Successor of `generate_from_refs.py` / `charmed_pipeline\generate.ps1`.

```
python pose_from_char.py --ref char.webp --prompt "..." --out-dir chars/anita
python pose_from_char.py --ref "primary.webp,gesture.webp" --identity-refs 0 ^
    --locks SOFT_LOCK,MID_LOCK,HARD_LOCK --seed 960 --prompt "..." --num-images 4
```

- First `--ref` = primary identity reference; later refs steer pose/gesture
  (orientation-matched crops work best — the model copies orientation).
- Transparent refs are flattened onto white and the prompt gets a
  white-background prefix (`--no-flatten-alpha` disables both).
- Render dims default to the primary ref's; the matted webp keeps the same
  dims, so variants drop in beside the original sprite. Matting is on by
  default (`--no-finalize` for the raw render only).
- Renders land in `--out-dir` (default `.`) as `<prefix>NNNNN.png` — 5-digit
  zero-pad, `--out-prefix` default `img`, numbering continues past existing
  files, never overwrites. `--num-images N` batches N renders per call.
- Multiple `--locks` render once per preset on the same seed, tagging the
  prefix (`img_soft00001.png` / `img_mid00001.png`).
  The submitted graph is saved to `last_payload.json` (viewable in the UI via
  `charmed_pipeline\api_to_ui.py`).

### char_lora_zimage.py

Per-character Z-Image LoRA management: train, refine, render. One script,
two modes selected by `--prompt`:

```
python char_lora_zimage.py akaslizabeth --images D:\path\to\dataset    # first run: train
python char_lora_zimage.py akaslizabeth                                # retrain IF dataset changed
python char_lora_zimage.py akaslizabeth --prompt "akaslizabeth ..." --out-dir renders --num-images 4
```

- Non-empty `--prompt` = inference only (never trains); empty = train iff
  images/captions/params changed (manifest-based change detection).
- Trains on Z-Image **De-Turbo** via musubi-tuner (venv at
  `d:\projects\musubi-tuner`; musubi's Base training loses likeness — #908),
  continuing from the latest version by default (`--from-scratch`,
  `--force-train`); flat 400 steps at 768px per run (`--steps N` to override),
  raw log at `loras\<char>\logs\<version>.log`. Renders on plain **Turbo**
  (20 steps / cfg 2 / strength 1.0 — train on De-Turbo, render on Turbo)
  into `--out-dir` (default `.`) as `<prefix>NNNNN.png`
  (5-digit, `--out-prefix` default `img`, numbering continues, never
  overwrites); `--num-images N` batches N renders per call.
- Every version kept under `.\loras\<char>\`: musubi-native + `_comfy`
  converted + manifest. Settings persist in `loras\<char>\config.json`.
- Dataset: folder of images + same-name `.txt` captions containing the
  trigger word (default: the character name itself; `--trigger` to differ).

### char_lora_flux.py

FLUX.2 Klein character LoRA training (v1: training only; rendering +
evolve.py integration to come). Trains on **klein-base-9b** (BFL's
trainable variant — the distilled klein is inference-only), for later
rendering on the installed distilled `flux-2-klein-9b-fp8`.

```
python char_lora_flux.py --name akasnoir --dataset D:\path\to\dataset
```

- `--dataset`: images + same-name `.txt` captions; every caption must
  START with `--name` (hard-validated before any GPU work).
- Recipe per musubi `docs/flux_2.md`: flux2_shift, adamw8bit, bf16,
  fp8 DiT+TE, grad-ckpt, rank 32, flat 400 steps (`--steps` to override).
- Versioned output in `.\loras\<name>\` + raw log in `logs\`.
- Requires `flux-2-klein-base-9b.safetensors` in `models\diffusion_models`
  (gated BFL repo — accept the license on HF first).

### evolve.py — the Evolver

An always-on image editor for iterative generation (aiohttp, single file).
Design record: `docs/app_design.md`.

```
python evolve.py --root mystore --port 8189   # then open http://127.0.0.1:8189/
```

- One **working image**; N **candidate slots** (type the number). **Generate**
  clears every unpinned slot and refills them from the controls as they are
  now — prompt, **ref0** (the identity/continuity carrier — a real slot, set
  apart) plus up to 3 extra references, LoRA + strength, lock, vary, seed,
  white-bg, size. Ref0 empty = fiat candidates.
- Double-click / Enter / drag-to-stage picks a candidate as the new working
  image — and restores the run state that GENERATED it: every widget, and
  **ref0 = its parent** (empty for a fiat image), so Generate immediately
  re-rolls that image's round with new seeds. **Space previews any selected
  image on the stage** (ref0 included — after a pick that's the parent, i.e.
  provenance at a keystroke). **Space + click on any image copies it into
  ref0** — the drag-to-ref0 shortcut; plain clicks only ever select. Del on
  ref0 empties it (= fiat). Selecting an image anywhere also scrolls History
  to it if present (hidden setting `sync_history_to_selection`, localStorage,
  default on).
  **Pinning** (📍 / `p`) moves a candidate off the sheet onto the
  **Pinned board** — your working memory — freeing its slot; the board and
  the sheet share one cell size and each collapses to a strip. Drop any image
  on the board to pin it; drag board images into the stage or ref slots.
  Unpinning sends it to **Discarded** (never deleted) — drag back to a slot or
  pick to revive. Every Generate refills *all* slots.
- Two top carousels, identical behaviour (click selects, double-click/Enter
  picks, hold Space to show the selection on the stage — move the selection
  while holding to A/B, ‹ › scroll while held): **History** = every image
  that was ever the working image; **Discarded** = displaced candidates.
  Pick any to continue from there. Nothing forward is lost. Seed is a *base*
  seed (0 = random); candidate k renders with base+k, and the base used last
  round is shown beside the field.
- Borders: green = pinned slot; blue outline = keyboard focus (the target of
  Del / Ctrl-C / Enter / arrows).
- **Model family** (fiat only — while the stage and ref slots are empty):
  Flux2 Klein, Z-Image Turbo, or Illustrious (WAI v16). Steps/cfg default per
  family (0 = default); the dark-red box is the negative prompt. References,
  lock, vary and white-bg are Klein-only; the LoRA dropdown serves Klein and
  Z-Image. Fiat a look in Illustrious, pick it, click the ref0 badge to
  `working`, and continue in Klein.
- **Collect garbage** purges only images that are not history, pinned, in a
  slot, a reference, or an ancestor of those.
- Clipboard / drag-drop everywhere: drop files from Explorer, images from
  other browser tabs, paste from Paint/Photoshop/browsers into any box;
  Ctrl-C copies the focused image (PNG + path text), Ctrl-X copies and
  clears, Del clears, ←/→ move, `p` pins.
- Store: `<root>/images/<id>.png` + `journal.jsonl` (full recipe + parents
  per image) + `state.json`. Generation = the Klein identity graph
  (`template_klein.json`); per-step progress on the console.
- `evolve_v0.py` is the retired rows-as-generations prototype (still runs
  against the old tree roots).

### tween.py

Tween two keyframe images into a video (Wan 2.2 FLF2V, native ComfyUI
two-expert graph). The keyframes are ground truth at t=0/t=1; the prompt
describes the MOTION between them. Chain segments on shared keyframes
for pixel-perfect joins.

```
python tween.py --start beat1.png --end beat2.png --prompt "she stoops and picks up the banknote"
```

- Needs `wan2.2_i2v_high/low_noise_14B_fp8_scaled` + umt5 fp8 + wan 2.1
  VAE. Dims default to the start keyframe (/16-snapped); `--frames` 4n+1
  (81 ≈ 5s @ 16fps); 20 steps split across the experts, cfg 3.5, shift 8.
- The experts run as TWO submissions (latent handoff + `/free` VRAM flush
  between): both resident at once exceeds 32GB and ComfyUI won't evict
  expert 1 at an in-graph swap — the cause of the 2026-08-20 OOMs.
- Output `<prefix>NNNNN.mp4` in `--out-dir` (default `.`), never
  overwrites; seed printed. evolve integration (parent/child keyframe
  check) later.

### watch.py

Attach a live progress bar to a ComfyUI run already in flight (read-only,
Ctrl-C safe): `python watch.py [client_id]` — default `tween`;
pose_from_char/evolve queue as `akasutils`. For runs started without their
own bar or watching from a second terminal.

### finalize.py

The matting step, standalone: `python finalize.py <render.png> <out_name>`.
Mattes in place (output keeps the render's dims); `--normalize` is the legacy
trim → scale-to-1080 → center-on-1920x1080 deliverable path.
Generate on a white background — dark clothing on black won't matte.

## Prompting recipe (what works)

Describe the character fully even though a ref is attached, then state
"The only change: ...". Lead with camera instructions when moving the camera.
SOFT_LOCK is the proven default; harden only if identity drifts.
