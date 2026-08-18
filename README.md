# akasutils

Utilities for driving ComfyUI headlessly (character renders for the game art
pipeline). **All scripts assume the ComfyUI service is up** at
`http://127.0.0.1:8188` and run in the **ComfyUI venv**
(`D:\projects\ComfyUI\.venv` — see `requirements.txt`).

## Scripts

### generate_from_refs.py

Identity-transfer render from one or more reference images (FLUX.2 Klein +
IdentityFeatureTransferFinal), with optional matting to a game-ready
1920x1080 transparent webp. Port of `charmed_pipeline\generate.ps1`.

```
python generate_from_refs.py --ref base.webp --prompt "..." --out mychar_pose --finalize
python generate_from_refs.py --ref "primary.webp,gesture.webp" --identity-refs 0 ^
    --locks SOFT_LOCK,MID_LOCK,HARD_LOCK --seed 960 --prompt "..." --out pose
```

- First `--ref` = primary identity reference; later refs steer pose/gesture
  (orientation-matched crops work best — the model copies orientation).
- Multiple `--locks` render once per preset on the same seed (`_soft`/`_mid`/`_hard`).
- Renders land in `..\output\charmed2\`; the submitted graph is saved to
  `last_payload.json` (viewable in the UI via `charmed_pipeline\api_to_ui.py`).

### finalize.py

The matting step, standalone: `python finalize.py <render.png> <out_name>`.
Generate on a white background — dark clothing on black won't matte.

## Prompting recipe (what works)

Describe the character fully even though a ref is attached, then state
"The only change: ...". Lead with camera instructions when moving the camera.
SOFT_LOCK is the proven default; harden only if identity drifts.
