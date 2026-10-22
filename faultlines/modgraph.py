"""Step 1: turn Graphify's big code graph into a small, readable map of files,
and stitch the broken links between projects (repos).

Why: Graphify records `import urllib3` inside requests as a dangling placeholder
(a "stub") instead of connecting it to the real urllib3 code. We find those stubs
and connect them, and we say how sure we are about each connection."""
import json, re, math, collections

SKIP_DIRS = ("test", "tests", "docs", "doc", "dummyserver", "tools", "examples",
             "benchmarks", "bench", "scripts", "ext", "build", "dist")
# How strongly a shock passes along each kind of link. These are ASSUMPTIONS, shown to the user.
# Chance that ONE use-site passes a shock on. Importing a name is a weak link (it may never run);
# actually calling or inheriting from it is much stronger.
RELATION_PRIOR = {"inherits": 0.50, "calls": 0.35, "uses": 0.20,
                  "imports_from": 0.12, "imports": 0.10, "references": 0.10}
PRIOR_CAP = 0.90
DEPENDENCY_RELS = set(RELATION_PRIOR)

def _is_code_file(path):
    if not path or not path.endswith(".py"):
        return False
    parts = path.lower().split("/")
    return not any(p in SKIP_DIRS or p.startswith("test_") or p == "conftest.py" or p == "setup.py"
                   for p in parts[:-1] + [parts[-1]])

def _dotted(path):
    p = re.sub(r"^(src|lib)/", "", path)[:-3]
    p = re.sub(r"/__init__$", "", p)
    return p.replace("/", "_").lower()

def load(graph_path):
    g = json.load(open(graph_path))
    return g["nodes"], g.get("links") or g.get("edges")

def build_module_graph(graph_path, drop_isolated=True):
    nodes, links = load(graph_path)
    N = {n["id"]: n for n in nodes}
    # --- 1. one node per real source file ("module") -------------------------------
    mods, by_file = {}, {}
    for n in nodes:
        repo, sf = n.get("repo"), n.get("source_file")
        if repo and _is_code_file(sf):
            key = f"{repo}::{sf}"
            by_file[(repo, sf)] = key
            m = mods.setdefault(key, {"id": key, "repo": repo, "file": sf, "symbols": 0,
                                      "label": sf.split("/")[-1] if not sf.endswith("__init__.py")
                                      else sf.split("/")[-2] + "/", "dotted": _dotted(sf)})
            if n.get("file_type") == "code":
                m["symbols"] += 1
    def mod_of(nid):
        n = N.get(nid)
        if not n or not n.get("repo"):
            return None
        return by_file.get((n["repo"], n.get("source_file")))
    # --- 2. index modules by dotted name so placeholders can be matched -------------
    index = collections.defaultdict(list)
    for m in mods.values():
        index[m["dotted"]].append(m["id"])
        # python-certifi's repo name differs from its package name; also index by package-only path
    # --- 3. collapse code-level links into file-level links -----------------------
    agg = {}
    stitched_stub_hits = collections.Counter()
    for l in links:
        rel = l.get("relation")
        if rel not in DEPENDENCY_RELS:
            continue
        a = mod_of(l["source"])
        if not a:
            continue
        b = mod_of(l["target"])
        stitch = None
        if not b:
            t = N.get(l["target"])
            if t and not t.get("repo") and t.get("file_type") == "concept":
                cands = [c for c in index.get(re.sub(r"[^a-z0-9_]", "", t["id"].lower()), [])
                         if mods[c]["repo"] != mods[a]["repo"]]
                if cands:
                    # exactly one match -> confident; several -> uncertain (a model can judge)
                    b, stitch = cands[0], {"stub": t["id"], "candidates": cands,
                                           "confidence": round(0.95 if len(cands) == 1 else 1 / len(cands), 2)}
                    stitched_stub_hits[t["id"]] += 1
        if not b or a == b:
            continue
        e = agg.setdefault((a, b), {"rels": collections.Counter(), "lines": [], "stitch": None,
                                    "inferred": 0})
        e["rels"][rel] += 1
        if l.get("confidence") == "INFERRED":
            e["inferred"] += 1
        if len(e["lines"]) < 6 and l.get("source_location"):
            e["lines"].append([l.get("source_file"), l["source_location"], rel])
        if stitch and not e["stitch"]:
            e["stitch"] = stitch
    # --- 4. build final edges. Direction = "a depends on b", so a shock flows b -> a --
    edges = []
    for (a, b), e in agg.items():
        # each use-site is its own small chance; combine them: 1 - product of (1 - chance)
        keep = 1.0
        for r, n in e["rels"].items():
            keep *= (1 - RELATION_PRIOR[r]) ** n
        prior = round(min(PRIOR_CAP, 1 - keep), 3)
        if mods[a]["file"].endswith("__init__.py"):   # package front doors mostly just re-export
            prior = round(prior * 0.5, 3)
        cross = mods[a]["repo"] != mods[b]["repo"]
        edges.append({"dependent": a, "dependency": b, "cross_repo": cross,
                      "relations": dict(e["rels"]), "evidence_lines": e["lines"],
                      "provenance": "stitched" if e["stitch"] else ("inferred" if e["inferred"] and e["inferred"] == sum(e["rels"].values()) else "extracted"),
                      "stitch": e["stitch"], "prior": prior,
                      "stitch_confidence": e["stitch"]["confidence"] if e["stitch"] else 1.0})
    if drop_isolated:
        used = {x for e in edges for x in (e["dependent"], e["dependency"])}
        mods = {k: v for k, v in mods.items() if k in used}
    return {"modules": list(mods.values()), "edges": edges,
            "stubs_stitched": dict(stitched_stub_hits)}

if __name__ == "__main__":
    import sys
    mg = build_module_graph(sys.argv[1])
    json.dump(mg, open(sys.argv[2], "w"), indent=1)
    cr = [e for e in mg["edges"] if e["cross_repo"]]
    print(len(mg["modules"]), "files,", len(mg["edges"]), "links,", len(cr), "cross-project links")
    print(collections.Counter(m["repo"] for m in mg["modules"]))
