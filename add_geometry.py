"""add_geometry.py - convert an API-format payload into a loadable UI graph.

Adopted 2026-08-22 from ..\\..\\charmed_pipeline\\api_to_ui.py (kept there for
the legacy pipeline) so evolve can embed an optional `workflow` chunk
(--embed-workflow): API-format JSON opened in the ComfyUI UI shows an EMPTY
canvas, so the geometry chunk is what makes a png draggable-and-editable.

Changes from the original: `convert_payload(prompt_dict) -> workflow_dict`
(in-memory, importable), and /object_info responses are cached per node
class for the life of the process - evolve calls this once per candidate,
and the graph SHAPE repeats within a round, so only the first candidate of
a session pays the HTTP round trips. Exact socket/widget rules unchanged:
combo lists and INT/FLOAT/STRING/BOOLEAN are widgets; control_after_generate
inputs get an extra "fixed" widget value.
"""

import json
import sys
import urllib.parse
import urllib.request

API = "http://127.0.0.1:8188"
_INFO_CACHE = {}


def object_info(cls):
    if cls not in _INFO_CACHE:
        with urllib.request.urlopen(
                f"{API}/object_info/{urllib.parse.quote(cls)}") as r:
            _INFO_CACHE[cls] = json.load(r)[cls]
    return _INFO_CACHE[cls]


def is_widget(spec):
    t = spec[0]
    if isinstance(t, list):                  # combo
        return True
    return t in ("INT", "FLOAT", "STRING", "BOOLEAN")


def convert_payload(prompt):
    """API-format node dict -> UI workflow dict (topological auto-layout)."""
    depth = {}

    def calc_depth(key, seen=()):
        if key in depth:
            return depth[key]
        if key in seen:
            return 0
        d = 0
        for v in prompt[key]["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and str(v[0]) in prompt:
                d = max(d, calc_depth(str(v[0]), seen + (key,)) + 1)
        depth[key] = d
        return d
    for key in prompt:
        calc_depth(key)

    ids = {key: i + 1 for i, key in enumerate(prompt)}
    nodes, links, link_id = [], [], 0
    col_rows = {}

    for key, node in prompt.items():
        cls = node["class_type"]
        info = object_info(cls)
        req = info["input"].get("required", {})
        opt = info["input"].get("optional", {})
        all_inputs = list(req.items()) + list(opt.items())

        widgets, inputs_def = [], []
        for name, spec in all_inputs:
            val = node["inputs"].get(name)
            linked = isinstance(val, list) and len(val) == 2 and str(val[0]) in prompt
            if is_widget(spec) and not linked:
                widgets.append(val if val is not None else
                               (spec[1].get("default")
                                if len(spec) > 1 and isinstance(spec[1], dict) else None))
                if len(spec) > 1 and isinstance(spec[1], dict) and \
                        spec[1].get("control_after_generate"):
                    widgets.append("fixed")
            else:
                t = spec[0] if not isinstance(spec[0], list) else "COMBO"
                entry = {"name": name, "type": t, "link": None}
                if linked:
                    link_id += 1
                    src_key, src_slot = str(val[0]), val[1]
                    links.append([link_id, ids[src_key], src_slot,
                                  ids[key], len(inputs_def), t])
                    entry["link"] = link_id
                inputs_def.append(entry)

        out_names = info.get("output_name") or info.get("output") or []
        out_types = info.get("output") or []
        outputs_def = [{"name": n,
                        "type": out_types[i] if i < len(out_types) else "*",
                        "links": []}
                       for i, n in enumerate(out_names)]

        d = depth[key]
        row = col_rows.get(d, 0)
        col_rows[d] = row + 1
        tall = cls in ("CLIPTextEncode", "IdentityFeatureTransferFinal", "LoadImage")
        nodes.append({
            "id": ids[key], "type": cls,
            "pos": [80 + d * 360, 80 + row * 340],
            "size": [330, 300] if tall else [270, 120],
            "flags": {}, "order": d, "mode": 0,
            "inputs": inputs_def, "outputs": outputs_def,
            "title": f"{cls} ({key})",
            "properties": {"Node name for S&R": cls},
            "widgets_values": widgets,
        })

    for l in links:
        _, src_id, src_slot, *_ = l
        for n in nodes:
            if n["id"] == src_id and src_slot < len(n["outputs"]):
                n["outputs"][src_slot]["links"].append(l[0])

    return {"id": "evolve", "revision": 0,
            "last_node_id": max(ids.values()), "last_link_id": link_id,
            "nodes": nodes, "links": links, "groups": [], "config": {},
            "extra": {}, "version": 0.4}


def convert(payload_path, out_path):
    with open(payload_path, encoding="utf-8") as f:
        data = json.load(f)
    wf = convert_payload(data.get("prompt", data))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else src.replace(".json", "_ui.json")
    convert(src, out)
