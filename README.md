# evolve

A virtual production studio for game art: images are bred by **selective
breeding with auditable heredity**, driving ComfyUI entirely through its
HTTP API. **ComfyUI must be up** at `http://127.0.0.1:8188`; everything
runs in the **ComfyUI venv** (`D:\projects\ComfyUI\.venv`, see
`requirements.txt`) and never launches or kills ComfyUI.

Every long wait streams **live progress** to the console (per-step sampler
bars from ComfyUI's websocket + an elapsed heartbeat while models load).
Fallbacks: `tail -f /d/projects/ComfyUI/user/comfyui.log`, `nvidia-smi -l 2`.

```
python evolve.py --root D:/evolve_root [--project NAME] [--port 8189] [--embed-workflow]
```

`--root` is mandatory and is the ONLY location the app depends on (nothing is
cwd-relative). Under it: one subdir per **project** (`<project>/images/NNN.png`
+ `journal.jsonl` + `state.json` + `archive/`), `assets.json` (assets are data:
`{name, loras[], dataset[{path, description}]}`), `loras/<name>/`, `_train/`
(transient datasets), `_debug/last_payload.json`, `config.json`.

## Layout

Runtime (what `evolve.py` needs), dependency order bottom-up:

| module | role |
|---|---|
| `project.py` | the global root, projects, config, name rules, json helpers |
| `comfy_client.py` | queue / wait (live progress) / free_vram / interrupt / liveness |
| `image_utils.py` | snap16, alpha flattening, megapixel fit, matting (`matte`, CLI) |
| `image_file.py` | sha1, text chunks, `write_png` (metadata preserved), hardlink staging, `<prefix>NNNNN.png` output convention |
| `image_meta.py` | the `evolve` png chunk, chunk harvesting, import gleaning; CLI dumps any image's metadata as JSON |
| `add_geometry.py` | API-format payload -> UI-loadable workflow (the optional `workflow` chunk) |
| `templates/*.json` | the four graphs, runnable as-is: `flux_klein_9b`, `zimage_turbo`, `illustrious_sdxl`, `qwen_edit_camera` |
| `build_payload/` | template-driven builders, one per family (`flux_klein`, `zimage`, `illustrious`, `qwen_edit`) |
| `controls.py` | the generator controls table (defaults, sanitise, restore-from-recipe) |
| `store.py` | `Store`: images + journal + live state; `open_project` |
| `trash.py` | roots, gc, discard, sweep, prune (archives, never deletes) |
| `asset.py`, `lora.py`, `lineage.py` | assets.json CRUD; LoRA dropdown resolution; siblings/family/foreign meta |
| `generate.py`, `camera.py`, `training.py` | the operators: `do_generate(GenerateConfig)`, `do_camera(CameraConfig)`, `do_training(TrainConfig)` |
| `lora_train/` | `Trainer` interface (`common.py`) + `zimage_turbo`, `flux_klein_9b`, `illustrious` (not available yet) |
| `jobs.py` | one-GPU job runner: busy flag, abort, training status |
| `api.py` | aiohttp handlers, snapshot composition, routes, `create_app` |
| `frontend/` | `index.html`, `css/evolve.css`, `js/evolve.js` - served live from disk |
| `evolve.py` | the entry point (argparse + wiring, nothing else) |

Development / standalone (`tools/`, each imports `_cli` first, which puts the
package on `sys.path`):

| tool | role |
|---|---|
| `pose_from_char.py` | multi-ref identity render + matting to a transparent webp (the original workhorse) |
| `camera_probe.py` | viewpoint sweeps with the Qwen multiple-angles LoRA (`--vp "[VP: right, low, medium]"`) |
| `train_lora.py` | train a character LoRA from a captioned folder (any family) |
| `lora_test.py` | render a prompt with a LoRA (native files converted on the fly) |
| `tween.py` | two keyframes -> video via Wan 2.2 FLF2V (two-expert handoff) |
| `watch.py` | attach a progress bar to a run already in flight |
| `depth_warp.py`, `quantize_te_fp8.py` | parked spikes |

`tests/` - `python tests/test_store.py` (store/trash/lineage/controls),
`test_payloads.py` (builders), `test_frontend.py` (JS parses, ids and api
names resolve), `test_server.py` (HTTP smoke on a throwaway root, no ComfyUI
needed). `evolve_v0.py` is the retired rows-as-generations prototype.

## Conventions

- Payloads are API-format JSON; renders go to `output/evolve_scratch/` and are
  COPIED out (a repeat-seed rerun is a ComfyUI cache hit pointing at the
  earlier scratch file). Progress events route to client id `evolve`.
- Images are immutable; metadata is never destroyed on import; the journal is
  the sole lineage authority (the png's `evolve` chunk carries no ancestry).
- Naming: modules are lowercase nouns; functions `verb_noun`; operators keep
  `do_`; HTTP handlers `handle_<name>`; dataclasses `XxxConfig`; constants
  UPPER; JS camelCase, DOM ids kebab-case.
- Model lore (identity locks, matting, the viewpoint campaign, LoRA
  doctrine) lives in `CLAUDE.md`.
