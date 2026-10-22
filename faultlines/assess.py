"""Step 2: Gemma judges each link between projects ("edge assessor").

Graphify marks everything it reads straight from code as 100% certain, so on its own the
simulation would have nothing uncertain to work with. Gemma reads the real code around each
link and says how likely a problem in the dependency is to reach the dependent file.
Each link is judged by a JURY (several votes, optionally two model sizes), see jury.py."""
import json, os, re
import numpy as np
from faultlines.jury import Jury, LEVEL_P, _norm as _jnorm   # LEVEL_P: ASSUMPTION, change and re-run

SYSTEM = """TASK: judge-link
You judge how likely a security problem inside file B (the dependency) would reach file A (the dependent).
Reply with JSON only: {"level": "low" | "medium" | "high", "why": "<one short sentence>", "evidence": "<text copied exactly from the CODE, max 120 characters>"}
Rules:
- high: A actively calls or builds on code from B for its main work.
- medium: A uses B in some features, or inherits from it.
- low: A only imports names for type hints, error classes, constants or compatibility, or the import is optional or guarded.
Never invent evidence. The evidence text must appear in the CODE."""

def edge_key(e):
    return f'{e["dependent"]}|{e["dependency"]}'

_norm = _jnorm

def _find_repo_dir(corpus, repo):
    for cand in (repo, repo.replace("python-", ""), repo.replace("_", "-")):
        p = os.path.join(corpus, cand)
        if os.path.isdir(p):
            return p
    return None

def snippet(e, corpus):
    """Real code around the link: the import line, a few lines around it, and a few places it is used."""
    repo, rel = e["dependent"].split("::", 1)
    root = _find_repo_dir(corpus, repo)
    if not root or not os.path.exists(os.path.join(root, rel)):
        return None
    lines = open(os.path.join(root, rel), errors="ignore").read().splitlines()
    locs = sorted({int(l[1][1:]) for l in e["evidence_lines"] if l[1].startswith("L")})
    if not locs:
        return None
    out, used = [], set()
    for ln in locs[:4]:
        for i in range(max(0, ln - 4), min(len(lines), ln + 3)):
            if i not in used:
                used.add(i); out.append(lines[i])
    names = set(re.findall(r"\b([A-Za-z_]\w{3,})\b", " ".join(lines[max(0, locs[0] - 1):locs[0] + 2])))
    names -= {"from", "import", "typing", "TYPE_CHECKING"}
    shown = 0
    for i, ln in enumerate(lines):
        if i in used or ln.lstrip().startswith(("import ", "from ", "#")):
            continue
        if any(re.search(rf"\b{re.escape(n)}\b", ln) for n in names) and shown < 4:
            out.append(ln); shown += 1
    return "\n".join(out)

def assess_edges(mg, model, corpus, cache_path, scope="cross", limit=None, progress=print):
    """`model` is a Gemma or a Jury. A single Gemma is wrapped as a jury of one vote (the old behaviour)."""
    jury = model if isinstance(model, Jury) else Jury([(model, 1)])
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    mods = {m["id"]: m for m in mg["modules"]}
    todo = [e for e in mg["edges"] if (e["cross_repo"] or scope == "all")]
    todo.sort(key=lambda e: (not e["cross_repo"], -sum(e["relations"].values())))
    todo = todo[:limit] if limit else todo
    misses = 0
    for n, e in enumerate(todo):
        k = edge_key(e)
        ck = f"{jury.name}|{k}"
        if ck in cache:
            continue
        code = snippet(e, corpus)
        if not code:
            cache[ck] = {"level": None, "why": "source code not found", "evidence": "", "verified": False, "source": "none"}
            continue
        dep = mods[e["dependency"]]
        total_sites = sum(e["relations"].values())
        user = (f"A = {e['dependent']}\nB = {e['dependency']} ({dep['symbols']} functions/classes)\n"
                f"Kinds of link: {e['relations']} ({total_sites} places in A use B; the CODE below shows only some of them)\nCODE:\n{code}")
        out = jury.judge(SYSTEM, user, code)
        if out:
            cache[ck] = out
        else:                                   # do NOT save failures, so a later run can try again
            misses += 1
            progress(f"  no usable answer for {k.split('|')[0].split('/')[-1]} <- {k.split('|')[1].split('/')[-1]}")
            if misses >= 3 and not any(str(v.get("source")) == jury.name for v in cache.values()):
                progress("  Model is not answering (is Ollama running? run: python selftest.py). Using default assumptions instead.")
                break
            continue
        misses = 0
        c = cache[ck]
        extra = f" (agreement {c['agreement']:.0%} of {c['n_verified']} verified votes)" if c.get("n_asked", 1) > 1 else ""
        progress(f"  judged {n + 1}/{len(todo)}: {k.split('|')[0].split('/')[-1]} <- {k.split('|')[1].split('/')[-1]}: {c['level']}{extra}")
        json.dump(cache, open(cache_path, "w"), indent=1)
    return {edge_key(e): cache.get(f"{jury.name}|{edge_key(e)}") for e in mg["edges"]}

def edge_probabilities(mg, assessments):
    """Final chance for each link, plus WHERE the number came from (used to label claims as hypotheses)."""
    probs, trust = [], []
    for e in mg["edges"]:
        a = assessments.get(edge_key(e))
        if a and a.get("level") and a.get("verified"):
            base, src = float(a.get("p", LEVEL_P[a["level"]])), "model-judged"
        else:
            base, src = e["prior"], "default-assumption"
        probs.append(round(base * e["stitch_confidence"], 3))
        trust.append(src if e["stitch_confidence"] >= 0.9 else "uncertain-link")
    return probs, trust

def edge_distributions(mg, assessments):
    """For links where the jury's votes disagree, the spread of votes: {edge index: (chances, weights)}.
    The simulator draws each run's chance for that link from this spread, so model doubt becomes risk."""
    out = {}
    for i, e in enumerate(mg["edges"]):
        a = assessments.get(edge_key(e))
        if a and a.get("level") and a.get("verified") and len(a.get("dist") or {}) > 1:
            vals = np.array([min(1.0, float(p) * e["stitch_confidence"]) for p in a["dist"]], float)
            w = np.array(list(a["dist"].values()), float)
            out[i] = (vals, w / w.sum())
    return out
