"""trash.py - what may go, and when (the trash doctrine, 2026-08-24).

ROOTED = GC-protected. Durable roots are pins and asset-dataset entries
(gids later); every root locks its reference images TRANSITIVELY -
provenance. Integrity for rooted images is never broken. Beyond that:
P is THE keep decision - unpinned candidates are swept before every round,
Shift+Del discards, prune clears a whole branch. Everything ARCHIVES
(<project>/archive/), nothing is ever deleted.
"""
import asset


def durable_roots(store):
    """{id: reason} for pins and this project's asset-dataset images."""
    why = {q: "pinned" for q in store.state["pins"]}
    for i, uses in asset.dataset_ids(store.name).items():
        why.setdefault(i, f"in asset {uses[0][0]}")
    return why


def closure(store, ids):
    """ids plus every alive ancestor (parents, transitively)."""
    keep, todo = set(), list(ids)
    while todo:
        i = todo.pop()
        if i in keep or not store.alive(i):
            continue
        keep.add(i)
        todo.extend(store.images[i].get("parents") or [])
    return keep


def protected(store):
    """Manual-GC roots: durable roots + the live working set + candidates,
    transitively. History is NOT a root."""
    roots = set(durable_roots(store)) | store.live_set() | \
        set().union(*store.state["candidates"].values())
    return closure(store, roots)


def gc(store):
    """Manual sweep of everything unreachable from the roots."""
    with store.lock:
        keep = protected(store)
        doomed = [i for i in store.images if store.alive(i) and i not in keep]
        n = store.archive(doomed)
        return {"removed": n, "kept": len(keep)}


def root_reason(store, i):
    """Why image i must NOT be archived - or None if it may. Only DURABLE
    roots count and their ancestor chains (referential integrity); the live
    working set is NOT a reason - discarding the WI just clears the stage."""
    why = durable_roots(store)
    if i in why:
        return why[i]
    seen = set()
    for r0, label in why.items():
        todo = [r0]
        while todo:
            x = todo.pop()
            if x in seen or not store.alive(x):
                continue
            seen.add(x)
            for par in (store.images[x].get("parents") or []):
                if par == i:
                    return f"ancestor of #{r0} ({label})"
                todo.append(par)
    return None


def verdict(store, i):
    """The Info Window's gc line."""
    if store.images[i].get("gone"):
        return "archived"
    return root_reason(store, i) or "collectable"


def discard(store, i):
    """Shift+Del: this image is trash, now. Quietly archives it and clears
    it from every live slot; refuses ONLY when referential integrity would
    break (returns the reason)."""
    with store.lock:
        if not store.alive(i):
            return "not found"
        why = root_reason(store, i)
        if why:
            return why
        store.forget([i])
        store.archive([i])
        store.save_state()
        return None


def sweep(store, tab):
    """Before a round: that tab's candidates the user did not pin are
    silently archived (usable-to-crap is ~1:100). Never those in use (WI,
    ref0, refs), never anything integrity depends on."""
    with store.lock:
        live = store.live_set()
        doomed = [q for q in store.state["candidates"].get(tab, [])
                  if q not in live and root_reason(store, q) is None]
        n = store.archive(doomed)
        if n:
            print(f"swept {n} unkept candidate(s) from {tab}")
        return n


def prune_plan(store, root_id, force=False):
    """PRUNE is per BRANCH: the mother-line subtree under root_id (parent 0
    - the tree of the lineage doctrine; co-parent edges are overlay and NOT
    followed). SAFE (default): archive every branch member not on a path to
    a durable root - "dead twigs". FORCE: durable roots INSIDE the branch
    are un-marked (unpinned, asset entries removed) and go too; only images
    that rooted things OUTSIDE the branch still reference are kept."""
    s = store.state
    if not store.alive(root_id):
        return None
    kids0 = {}
    for j, r in store.images.items():
        if not store.alive(j):
            continue
        ps = r.get("parents") or []
        if ps:
            kids0.setdefault(ps[0], []).append(j)
    branch, todo = set(), [root_id]
    while todo:
        x = todo.pop()
        if x in branch:
            continue
        branch.add(x)
        todo.extend(kids0.get(x, []))
    pins = set(s["pins"])
    aset = asset.dataset_ids(store.name)
    durable = pins | set(aset)
    roots = (durable - branch) if force else durable
    keep = closure(store, roots)
    archive = sorted(branch - keep)
    kept = sorted(branch & keep)
    why = {}
    for k in kept:
        if k in pins and not force:
            why[k] = "pinned"
        elif k in aset and not force:
            why[k] = "in asset " + aset[k][0][0]
        else:
            why[k] = ("referenced by a rooted image outside the branch"
                      if force else "spine to a rooted image")
    outside_refs = sum(1 for j, r in store.images.items()
                       if store.alive(j) and j not in branch
                       and any(q in branch for q in (r.get("parents") or [])))
    c = s["controls"]
    return {"root": root_id, "branch": len(branch), "archive": archive,
            "keep": [{"id": k, "why": why[k]} for k in kept],
            "unpin": [i for i in archive if i in pins],
            "asset_removals": [{"asset": a, "path": q}
                               for i in archive for a, q in aset.get(i, [])],
            "live": {"working": s["working"] in archive,
                     "ref0": c.get("ref0") in archive,
                     "refs": sum(1 for r in c["refs"] if r in archive)},
            "outside_refs": outside_refs, "force": force}


def prune_apply(store, root_id, force=False):
    with store.lock:
        plan = prune_plan(store, root_id, force)
        if not plan or not plan["archive"]:
            return plan
        ids = set(plan["archive"])
        s = store.state
        for i in plan["unpin"]:
            if i in s["pins"]:
                s["pins"].remove(i)
        if plan["asset_removals"]:
            assets = asset.load_assets()
            asset.remove_paths(assets, [r["path"] for r in plan["asset_removals"]])
            asset.save_assets(assets)
        store.forget(ids)
        store.archive(plan["archive"])
        store.journal_event({"t": "prune", "root": root_id, "ids": plan["archive"],
                             "unpinned": plan["unpin"],
                             "assets": plan["asset_removals"], "force": force})
        store.save_state()
        print(f"pruned #{root_id}: {len(ids)} archived"
              + (f", {len(plan['unpin'])} unpinned" if plan["unpin"] else "")
              + (f", {len(plan['asset_removals'])} asset entries removed"
                 if plan["asset_removals"] else ""))
        return plan
