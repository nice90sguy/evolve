# Evolve application design — pairing record

**Author:** Codex (OpenAI), based on design pairing with Josh  
**Date:** 2026-08-21  
**Status:** Product/UX exploration record, not an implementation specification

## How to use this document

This document records the design discussion around the future of `evolve.py`: the user's observed workflows, settled decisions, accepted suggestions, conceptual boundaries, and questions that remain open.

Future agents should distinguish among:

- **Decided** — explicitly settled by the user.
- **Recommended** — a proposal accepted in spirit or currently preferred, but still open to refinement.
- **Open** — deliberately unresolved.

This session was for product-design pairing, criticism, and creative thinking. It did **not** authorize development work. The user granted a one-time exception to the directory's read-only rule solely to create this file. Do not infer permission for other writes.

## Working relationship and scope

The user's intended division of labour is:

- Claude remains focused on coding and technical execution.
- Codex is the partner for creative design, critical thinking, product/UX reasoning, and constructive criticism of Claude's output.
- The conversation should remain conversational, concise, and Socratic: probe assumptions, identify conflicts, distinguish workflows, and periodically return from details to the overall design.
- Web research should be used when current practice or external lore would materially improve decisions about image generation for games, but not reflexively.

The existing project rules in `claude.md` continue to apply, notably:

- Never generate an image without asking first.
- Never judge or pre-screen an image before the user does.
- Never write generated images to a session scratchpad.
- Never initiate training without explicit permission.
- Long work must expose live progress.
- The user is the art director.

## Product frame

### Decided: the game is the top-level context

The application is used while building visual assets for a particular Ren'Py game. The current example project/theme is **Softlock**, in which the player avoids Groundhog Day-like loops imposed by antagonists.

The user's creative process can begin with a character or a narrative situation. For example, "working on Anna" may mean designing Anna's appearance, expressions, and clothes, or it may mean rehearsing/shooting a picnic scene involving Anna.

### Decided: Evolve is a virtual production studio, not the story authoring environment

Ren'Py owns:

- Story text and dialogue
- Variables, RNG, choices, and branching logic
- The executable narrative sequence

Evolve owns:

- Casting and visual development
- Locations, props, and character looks
- Shooting stills and creating rushes
- Selecting keyframes
- Building and iterating visual scene plans
- Animating transitions between keyframes
- Preparing and tracking game assets

The user's concise formulation is: **the story is written in Ren'Py; Evolve casts, shoots, and creates rushes.**

Initially, Evolve will maintain its production structure independently. Later it may read Ren'Py scripts to discover or synchronize scene and asset requirements.

## Core vocabulary

The following movie/production vocabulary was accepted:

- **Game** — top-level project.
- **Cast** — characters available to the game.
- **Character look** — a character in specific wardrobe; wardrobe is not independent of the character.
- **Location** — a reusable setting, with a loose collection of approved views.
- **Prop** — a generally reusable item such as a gun, camera, bottle, or furniture.
- **Scene** — an encounter or larger narrative/production unit.
- **Beat** — a meaningful portion of a scene.
- **Shot** — a visual moment within a beat.
- **Rushes / takes / candidates** — generated alternatives for consideration.
- **Primary select** — the take currently representing a shot.
- **Approved alternate** — a useful alternative attached to a shot but not normally visible.
- **Keyframe** — a selected still used in a scene sequence.
- **Tween** — generated motion between two concrete keyframes.
- **Asset** — a prepared visual deliverable available to Ren'Py.
- **Lab** — the on-demand image-evolution workspace; critically, it is not itself a named project object or storage container.

The existing evolutionary vocabulary and provenance doctrine from `claude.md` remain relevant: parent 0 is the continuity carrier/mother; secondary references create a DAG overlay; images retain auditable heredity.

## What the current prototype has enabled

The current `evolve.py` UI presents rows as generations, siblings in horizontal carousels, and a selected image's parent-0 ancestry as highlighted rows. A large preview panel occupies the right side. A bottom workbench row receives a mother plus additional reference slots.

Observed operations include:

1. Generate candidates from text alone.
2. Generate from text plus a steering character LoRA at a chosen strength.
3. Generate from text plus one mother image for identity transfer.
4. Select a candidate as the focus of later operations.
5. Open a selected image in Explorer.
6. Copy its Windows or POSIX-style full path.
7. Delete it.
8. Use it as the mother/primary identity-transfer reference.
9. Use it in secondary reference slots.
10. View it enlarged.
11. Reapply the same prompt/references to extend the candidate list.
12. Change the prompt and replace the current candidates.
13. Observe and navigate parent-0 ancestors.
14. Delete all but selected images in a candidate list.
15. Purge a family tree.
16. Create a new root/work directory.
17. Browse a row or the entire generated project tree.
18. Copy keyframe paths into `tween.py` manually.

The screenshot discussed in the session showed the youngest generation selected, its ancestors highlighted, unrelated material hidden, and a broken secondary-reference thumbnail. That secondary reference was a room image pasted into a reference slot. It worked for generation but did not persist across sessions. Also, because relatedness was based only on parent 0, the room's row disappeared under "hide non-relatives" even though it materially contributed to the composite.

This is evidence that parent-0 genealogy remains valuable but is insufficient as the application's sole navigation and relationship model.

## Overall application architecture

### Decided/recommended layering

The application should be understood as four related layers:

1. **Lab** — produces every static image and records its full provenance.
2. **Game catalogue** — describes what the production has: cast, character looks, props, locations, and approved assets.
3. **Storyboard / Scene Studio** — holds scenes, beats, shots, optional branches, placeholders, keyframes, transitions, and tweens.
4. **Ren'Py integration** — eventually exports or synchronizes a sufficiently resolved production plan.

Images are not copied between these layers. A single managed image can be linked into many contexts while retaining one identity and one history.

### Recommended main navigation

A likely high-level navigation model is:

- Lab / Evolve Image actions
- Cast
- Locations
- Props
- Scenes
- Assets
- Project-wide Rushes/Search
- Project settings/art direction

The precise navigation chrome remains open. A stable desktop shell may use project navigation on the left, the active workspace centrally, and an optional Inspector/preview on the right.

## The Lab

### Decided: what the Lab is

The Lab is simply a fast way to iterate an image. It has no name, is not a persisted "board" that the user must manage, and need not have its own home screen.

It appears after an evolution command such as:

- **Evolve Image…** — begin with no image, using prompt and optionally a LoRA.
- **Evolve existing…** — choose a managed image or browse the filesystem.
- **Evolve clipboard** — import/deduplicate pixels from the clipboard and begin from them.
- Select an image anywhere and invoke a quick action such as `E` or double-click.

The exact shortcuts are recommended rather than finally specified.

### Decided: rolling contact-sheet interaction

The Lab's main surface is a rolling contact sheet. Nine was discussed as a useful example, not a required fixed count.

The settled interaction is:

- Candidates occupy the available contact-sheet cells.
- Pinning a candidate reserves its cell.
- The user is responsible for unpinning when more variation capacity is wanted.
- Changing the prompt, references, LoRA, or other generation inputs affects the unpinned cells.
- The next generation round replaces/refills only unpinned cells.
- Displaced images are not automatically deleted; they remain accessible in rushes/history.
- If every cell is pinned, no capacity remains for new takes until at least one is unpinned.

The explicit versus automatic generation trigger remains open. Existing project doctrine says generation only follows deliberate user action, so parameter editing should not silently launch expensive work unless the user later chooses that behaviour explicitly.

### Decided: recall and evolutionary backtracking are different

Two forms of history access are required:

1. **Recall a take** — find an old candidate and return it to an unpinned contact-sheet cell.
2. **Step back through iterations** — revisit an earlier point in the evolutionary process and continue from it.

An iteration path might read:

```text
Opening shot → gun corrected → floor-level → anime → eye adjustment
```

Selecting a candidate and evolving it pushes a new iteration. The user needs to scroll or step backward to the state from ten minutes earlier and try new variations from there.

### Recommended: browser-like path, graph-like storage

The visible history should feel like a browser stack or breadcrumb through the current evolution path, while the underlying provenance remains a graph.

- **Continue from here** creates a new continuation without deleting existing descendants.
- The history strip shows the active path, not the entire graph.
- A compact fork indicator such as `2 other continuations` reveals alternatives.
- No branch naming, checkout, merge, or other Git vocabulary should be required.
- Existing later work remains accessible when a new continuation is started from an older iteration.

### Decided/recommended: do not put destructive pruning in the normal Lab flow

"Branch from here" is useful. "Prune and branch from here" should not be a prominent Lab command.

Pruning belongs in lineage/cleanup tooling, where dependencies can be checked. The Lab may offer a non-destructive **Abandon later attempt** action that hides an unwanted continuation from the active trail while leaving actual deletion to garbage collection.

### Open: returning to an older iteration

It remains unresolved whether stepping back makes the old iteration immediately editable or initially read-only with a deliberate **Continue from here** action. The latter reduces accidental mutation; the former is faster.

## Preview and comparison

### Decided: support both transient and persistent preview

The user should be able to choose between:

- A dynamic large preview on demand.
- A permanent, resizable preview/Inspector panel.

Recommended desktop conventions:

- Single-click selects an image and updates the optional Inspector.
- Spacebar opens a temporary large Quick Look preview.
- Escape or Space closes it.
- Arrow keys navigate neighbouring candidates without closing.
- A toolbar control or shortcut toggles the docked preview panel and remembers the preference.
- Narrow windows adapt the panel into an overlay.

Ctrl-click should not be used merely to preview because it is valuable and conventional for multi-selection.

These recommendations follow familiar asset-browser patterns: Adobe Bridge supports full-screen preview with Space and adjacent-image navigation with arrow keys; Microsoft's list/details guidance recommends adaptive side-by-side views when space permits.

Sources:

- <https://helpx.adobe.com/bridge/using/preview-compare-images-bridge.html>
- <https://learn.microsoft.com/en-us/windows/apps/develop/ui/controls/list-details>

### Decided: A/B comparison only

The application should support enlarged comparison of no more than two images.

Recommended behaviour:

- Ctrl-click selects up to two candidates.
- A Compare command opens A/B side-by-side inspection.
- Zoom and pan are synchronized by default.
- A can remain anchored while the user steps through alternatives in B.
- Compare mode is neutral: it does not force a winner, pin, pass, or promote either image.

## Image ingestion, identity, and search

### Decided: pasted/dropped images are both global and contextual

Any externally pasted or dropped image must immediately become durable managed project data. A reference slot must never depend on clipboard lifetime, browser cache, or a temporary filename.

The same imported image should:

- Appear in a project-wide Imports/source collection.
- Be attached to the current character, location, prop, shot, or Lab operation.
- Avoid duplicate storage through content/pixel hashing.
- Create additional relationships when reused rather than duplicate files.

Optional source/licensing notes can be attached to the managed image record.

Copying an application image outward should place usable bitmap data on the clipboard, and where practical make its source path available too, so Photoshop and other tools work naturally.

### Decided: image labelling is for recall, not Ren'Py coding

User-entered image names/tags are:

- Free-form
- Searchable
- Intended for rapid recall
- Searchable using regular expressions; users are expected to understand regex

Search should always default to the **entire project**, not the current character/scene. Optional filters and context badges can narrow results.

### Recommended: separate human annotations from generated asset names

Ren'Py-facing filenames can be machine-generated and need not serve as the user's primary mnemonic. A useful generated taxonomy combines semantic context with a short immutable suffix, for example:

- `anna_picnic_stand_ffb3.png`
- `park_picnic_wide_day_a91c.png`
- `anna_picnic_stand_to_run_72de.mp4`

Internally, every image should retain an immutable identity. An exported filename should be frozen once Ren'Py may refer to it, preventing later relabelling from silently breaking code.

## Provenance as a first-class universal feature

### Decided

Every static image must carry a complete history of both:

- **Evolution** — how it was produced: prompts, LoRAs, parameters, mother, secondary references, composites, transformations, batches, and siblings.
- **Use** — where it participates: character references, looks, location galleries, props, datasets, Lab iterations, shots, keyframes, tweens, assets, and exports.

This may be reflected in directories and metadata, but the UI concept is a project-wide relationship graph. Directory ancestry alone cannot express secondary references, reuse, composites, or story usage.

History must be available everywhere an image appears.

### Recommended lineage interaction

Every image should expose a **Lineage** action opening a dedicated explorer centred on that image. It should provide rapid navigation among:

- Ancestors
- Siblings
- Descendants
- Secondary references
- Composite inputs and outputs
- Uses throughout the project

The current prototype's immediate ancestry visibility is valuable, but permanently showing all ancestry rows will not scale. In normal work, show the active iteration path and compact relationship indicators; open the full Lineage view on demand.

### Decided/recommended garbage collection

Garbage collection should find genuinely orphaned work, but "unpinned" alone is not enough. Reachability must protect any image that is:

- Pinned or deliberately retained
- Named/tagged if naming is treated as retention
- Used as a reference
- An ancestor of protected work
- Included in a LoRA dataset
- Attached to cast, looks, props, locations, scenes, shots, or assets
- An endpoint of a tween
- Exported or otherwise referenced externally

Only unreachable images become cleanup candidates. Deletion should be reviewed and dependency-aware, never an automatic side effect of ordinary Lab branching.

## Game catalogue

### Decided entity model

```text
Game
├─ Cast
│  └─ Character
│     ├─ Identity/canonical references
│     ├─ Identity LoRA and dataset
│     └─ Character looks (wardrobe)
├─ Locations
├─ Props
├─ Scenes
└─ Assets
```

### Characters and wardrobe

- Wardrobe is character-specific; characters do not share clothes.
- A wardrobe entry is a **character look**, created by generating the character wearing it from the outset.
- The garment is not designed as a separate reusable object.
- A scene casts both a character and one of that character's looks.
- Each character should have one identity LoRA trained across several outfits rather than a separate LoRA per look.
- The dataset should span wardrobe, pose, expression, and viewpoint so the LoRA learns the character rather than one costume.
- Character-look reference images remain useful during shooting even when an identity LoRA exists.

### Character LoRA workflow

Creating a character LoRA is a separate process within the character context:

- Dataset images may originate anywhere, including web sources, generative tools, imports, and Lab results.
- Dataset selection/curation should be easy.
- Dataset captions/tags should be semi-automated and reviewable.
- Training remains an explicit user-authorized operation with visible progress.

The detailed dataset UX and captioning workflow remain to be designed.

### Props

Props are general reusable production items: guns, cameras, furniture, bottles, and similar objects. They are not tied to one character.

### Locations

For now, locations should use a loose gallery of named approved views rather than a formal viewpoint × lighting matrix. Reliable camera control has not yet been established. If a dependable ComfyUI camera-control workflow is found, structured camera behaviour can be added under the hood later.

### Art style

The game's primary art style should be chosen at project level up front. Anime/realistic style transfers may still be explored in the Lab, but the catalogue should not automatically multiply every entity into parallel style trees.

A complete alternate visual edition could be introduced deliberately later, but ad hoc style experiments remain experiments until promoted.

## Storyboard and Scene Studio

### Decided: the storyboard is a layer over image production

Scenes can exist without images. The user may:

- Create a starting scene or a sequence of image placeholders.
- Add optional branches.
- Fill placeholders in any order.
- Begin generating before the scene is complete.
- Build an entire game's production plan with missing images.

Static images backing placeholders are always created in the Lab. A placeholder can invoke Evolve and receive the selected result, or an existing image can be attached later.

Keyframes and transitions belong to the story/scene plan even when concrete assets are still missing.

### Decided: scene creation is usually discovered incrementally

The user typically does not pre-author a formal shot list in Evolve. A shot list may exist mentally or on paper. In the app, the common flow is:

1. Establish the first still.
2. Select it.
3. Add/shoot what happens next.
4. Name shots and group them into beats later if useful.

Therefore Scene Studio should be **append-first**, not require forms or placeholders before visual experimentation.

### Decided/recommended: compact branching map plus linear active route

Optional scene branches should be represented by a compact Git-tree-like scene map, while the main editing surface remains linear.

Example:

```text
                    ┌─ Wasps ─ Runs away
Picnic ─ Listening ─┤
                    ├─ Argument ─ Storms off
                    └─ Wine ─ Pours glass
```

Recommended interaction:

- A shallow map shows small shot thumbnails connected by lines.
- Shared shots appear once before a fork.
- Branches occupy compact lanes.
- The active route is bright; inactive routes are muted.
- Clicking a node focuses that shot.
- Clicking a branch label makes it the main linear filmstrip.
- **Shoot next** extends the active route.
- **Fork here** adds another continuation.
- Connection badges expose tween status and continuity-review warnings.
- Large beats or inactive branches can collapse.

This map is navigation and context, not a freeform node editor.

### Decided: primary select plus hidden approved alternates

Each shot has:

- One primary select used by the visible storyboard.
- Any number of approved alternates, readily available but not always visible.

Small interchangeable differences such as expression or minor gesture belong as alternates. Different action/staging belongs in separate shots.

### Decided: continuity warnings live on transitions

If an earlier shot's primary changes, later shots should not flip automatically to alternatives and should not all be marked recursively.

Instead, the directly affected transition receives a quiet **continuity review** indicator. The user can:

- Accept the small mismatch and clear the indicator.
- Retake the next shot using the new previous image.
- Choose a compatible approved alternate.

If the next shot is retaken, its former primary can remain an approved alternate. Any newly affected following transition is reviewed in turn. This avoids a combinatorial alternate tree.

Tweens are stricter: changing either exact endpoint makes that tween stale and requires review/regeneration.

### Open: route reconvergence

The discussion settled visible forks but did not decide whether different scene routes may reconverge on one shared shot node in the production map, or whether later reuse should appear as linked copies in separate routes.

## Shooting, compositing, and pipeline routing

### Asset-first and exploration-first are both valid

One expected shooting workflow begins with established cast, character look, location, and props. These references bootstrap the first composition. Subsequent stills use the preceding selected image as continuity while identity/location assets continue to steer the result.

However, the user's real experiment demonstrated that character, style, location, shots, and animation intent may all emerge during free exploration. Therefore the app must also support retrospective promotion of Lab discoveries into cast, locations, scenes, and sequences.

### Recommended: semantic reference roles

Raw `ref0`/`ref1`/`ref2` is too technical as the primary UX. The first composition may contain references with roles such as:

- Character: Anna
- Look: picnic dress
- Location: park
- Prop: wine bottle

On later shots, the previous frame becomes the continuity carrier while those assets protect identity and context. The underlying workflow may still map these roles to specific ComfyUI reference slots.

### Under-the-hood pipeline routing

The long-term aspiration is to use movie language naturally:

- “Dolly in” may invoke a camera-control pipeline.
- “She stands up” may invoke image transfer for the next still.
- A transition between two fixed keyframes invokes tween/video generation.

The UI should expose intent rather than ComfyUI graphs. A recommended compromise is an automatic method with a small visible interpretation such as:

- `Method: camera move`
- `Method: identity transfer`
- `Method: composition`
- `Method: tween`

An override can remain available without presenting a forest of technical options.

Mixed directions such as “Dolly in as she stands” remain a hard ambiguity: they may require a combined workflow or separate target-frame and transition instructions. This was not finally resolved.

### Static-frame direction versus transition direction

The user's successful experiment confirms two distinct kinds of text:

- **Next-frame direction** — describes the desired new still/end state.
- **Transition direction** — describes motion connecting two already chosen keyframes.

Both may use movie language, but they should not be conflated.

## Animation and tweening

### Decided

- Animation/tweening belongs in scene building, not in the Lab.
- Tweens are attempted only when actual images back both keyframe placeholders.
- The user may need to iterate keyframe selection, transition prompts, and timing.
- If no tween is supplied, the default transition is a dissolve.
- If tweening is supplied, the scene contains a timed series of generated motion segments; timing is important.
- Ren'Py integration likely occurs only after the necessary keyframes and transitions are sufficiently defined.

Recommended scene interaction:

- Any Lab result can be attached to a storyboard keyframe without copying a path.
- Selected keyframes can be reordered into a sequence.
- Each connection/edge owns its transition direction, tween outputs, duration, and status.
- Tween regeneration is performed from the edge itself.

## Lessons from the office/femme-fatale experiment

The user supplied an important real workflow. It was exploratory rather than planned:

1. Create an opening office shot using a pre-existing character LoRA assembled earlier from web and generated images.
2. Correct the pose/weapon aim.
3. Reframe to a floor-level view while preserving the corrected aim.
4. Experiment with an anime style transfer and decide it works.
5. Adjust the character's appearance; the anime result implicitly becomes "the" character reference.
6. Generate a new standing keyframe.
7. Work around composition difficulty by separately producing a character plate on white and a clean room plate.
8. Recombine/reframe those references into the next close keyframe.
9. Generate a further wardrobe-change keyframe.
10. Only afterward, select four non-consecutive lineage images as animation keyframes and create three separately prompted tweens between them.

The user did not know at the outset that the experiment would become an animation. `tween.py` itself was developed during the process. This was exploratory learning about what the app could enable and what the user might want.

The example demonstrates that the current genealogy is simultaneously carrying three different structures:

1. **Evolution provenance** — how each still descended from previous stills.
2. **Production decomposition/recomposition** — character plate + location plate → composite.
3. **Scene sequence** — a curated subset of stills joined by transition-specific prompts.

These structures must remain linked but should not be forced into one UI hierarchy.

It also shows that inconvenient workarounds should not automatically become permanent first-class workflows. Operations such as **isolate character**, **create clean plate**, and **recompose** may be valuable, but should be generalized only after their intended semantics and repeatability are understood.

## Broad design principles established

1. **The game is the context; the image is the universal medium.**
2. **Every static image is born in the Lab.**
3. **The Lab is an action/workspace, not an object the user must organize.**
4. **The contact sheet is for rapid iterative comparison, with pins consuming generation capacity.**
5. **Nothing useful should disappear merely because the user backs up and tries another continuation.**
6. **Provenance and use are first-class and accessible everywhere.**
7. **Genealogy is a view of image history, not the application's global spatial layout.**
8. **Cast, locations, props, and scenes are semantic catalogue/story objects that link to managed images.**
9. **Wardrobe is character-specific; props are general.**
10. **A character has one identity LoRA across looks, not one LoRA per outfit.**
11. **Scene Studio supports empty placeholders and incremental discovery.**
12. **A shot has one primary select and hidden approved alternates.**
13. **Continuity is reviewed at transitions; it does not trigger automatic cascading alternate switches.**
14. **Tween dependencies are exact and become stale when endpoints change.**
15. **The scene map may branch, while the main working route stays linear.**
16. **Ren'Py remains the authority for story logic.**
17. **Technical ComfyUI pipelines should remain under the hood, with their interpreted method visible and overridable.**
18. **Destructive pruning is cleanup, not ordinary exploration.**

## Outstanding design questions

The next design conversation should continue Socratically from these unresolved areas:

1. What is the exact Lab screen layout: placement of ingredients, prompt, contact sheet, iteration strip, rush recall, and optional Inspector?
2. Does changing generation input automatically refill unpinned cells, or is there always an explicit Shoot/Generate action?
3. When stepping back to an old iteration, is it immediately live or initially read-only until **Continue from here**?
4. How are old takes recalled into a full contact sheet: drag/drop into a chosen unpinned cell, swap, or another gesture?
5. How should full Lineage be visualized without becoming a sprawling node graph?
6. What exact data/metadata should make a generated image safe from garbage collection?
7. How should character dataset curation and semi-automatic training captions work?
8. How much project-level art direction is inherited automatically by Lab operations?
9. How are semantic references displayed and edited without exposing excessive technical options?
10. How does automatic pipeline interpretation behave for mixed camera and subject directions?
11. Can storyboard routes reconverge, and if so, how should the compact scene map represent it?
12. What timing controls and preview facilities are needed for dissolves and tween chains?
13. What is the eventual contract between Evolve's generated asset catalogue and Ren'Py scripts?

