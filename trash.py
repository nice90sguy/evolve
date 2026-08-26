"""trash.py - what may go (2026-08-25).

archived is a BIT on the record (Store.archive / restore). discard = set
it on one image; sweep = on the unpinned candidates of a tab before a
round; prune = on a branch. Pinned images are never archived unless
forced (which unpins them). Archived images stay on disk, hidden; GARBAGE
= archived AND not an ancestor of any unarchived image (Store.garbage),
and PURGE deletes those files - the only surviving integrity rule.
"""
from store import PINNED


def discard(store, i):
    """Shift+Del on a leaf: archive it and clear it from every live slot.
    Returns an error string or None."""
    with store.lock:
        if not store.alive(i):
            return "not found"
        if store.has(i, PINNED):
            return "kept: pinned (unpin first)"
        store.archive([i])
        store.forget([i])
        store.save_state()
        return None


def sweep(store, tab):
    """Before a round: that tab's candidates still FRESH (spawned and never
    touched since) are thrown away. Touching anything - making it the WI,
    referencing, pinning, tagging, moving... - is the keep decision.
    Orphaned fresh images (see Store.orphan_fresh) go with them."""
    with store.lock:
        doomed = [q for q in store.state["candidates"].get(tab, [])
                  if store.is_fresh(q) and not store.is_archived(q)]
        doomed += store.orphan_fresh()
        if doomed:
            store.archive(doomed)
            store.forget(doomed)
            store.save_state()
            print(f"swept {len(doomed)} unkept candidate(s) from {tab}")
        return len(doomed)


def sweep_orphans(store):
    """Enforce the invariant at startup: fresh images listed in no tab's
    Output are thrown away quietly (journaled as a move to the trash)."""
    with store.lock:
        doomed = store.orphan_fresh()
        if doomed:
            store.archive(doomed)
            store.forget(doomed)
            store.save_state()
        return doomed


def prune_plan(store, root_id, force=False):
    """The branch = root_id + its mother-line descendants. SAFE (default):
    archive everything in it except pinned images. FORCE: pinned too
    (they are unpinned). Returns the plan the dialog shows; apply=False."""
    if not store.alive(root_id):
        return None
    branch = [root_id] + store.descendants(root_id)
    pinned = [i for i in branch if store.has(i, PINNED)]
    already = [i for i in branch if store.is_archived(i)]
    archive = [i for i in branch if i not in already and (force or i not in pinned)]
    keep = [{"id": i, "why": "pinned"} for i in pinned if not force]
    datasets = [{"lora": w[len("lora_dataset_"):], "id": i}
                for i in archive for w in store.tags(i) if w.startswith("lora_dataset_")]
    s = store.state
    c = s["controls"]
    return {"root": root_id, "branch": len(branch), "archive": archive,
            "already": len(already), "keep": keep,
            "unpin": [i for i in archive if i in pinned],
            "dataset_removals": datasets,
            "live": {"working": s["working"] in archive,
                     "ref0": c.get("ref0") in archive,
                     "refs": sum(1 for r in c["refs"] if r in archive)},
            "outside_refs": sum(1 for j, r in store.images.items()
                                if store.alive(j) and j not in branch
                                and any(q in branch for q in (r.get("parents") or []))),
            "force": force}


def prune_apply(store, root_id, force=False):
    with store.lock:
        plan = prune_plan(store, root_id, force)
        if not plan or not plan["archive"]:
            return plan
        store.archive(plan["archive"], force=force)
        store.forget(plan["archive"])
        store.save_state()
        print(f"pruned #{root_id}: {len(plan['archive'])} trashed"
              + (f", {len(plan['unpin'])} unpinned" if plan["unpin"] else ""))
        return plan


def empty_trash(store, apply=False):
    """apply=False -> the impact plan for the dialog; True -> recycle."""
    if not apply:
        return store.empty_trash_plan()
    ids = store.empty_trash()
    return {"removed": len(ids), "kept": len(store.alive_ids())}


def verdict(store, i):
    """The Info Window's gc line."""
    if store.images[i].get("purged"):
        return "emptied to the recycle bin"
    if not store.is_archived(i):
        return "live"
    bearing = "a live image descends from it - emptying will leave missing-parent placeholders"
    return f"in the trash ({bearing})" if i in store.load_bearing() else "in the trash"
