"""Rule-based stand-in used only for offline testing. It imitates the SHAPE of Gemma's answers."""
import json, re, zlib

def _h(*parts):
    return zlib.crc32(repr(parts).encode())   # stable across runs (the built-in hash is randomised per run)

def mock_answer(system, user, images, opts=None):
    opts = opts or {}
    if "TASK: judge-link" in system:
        code = user.split("CODE:", 1)[-1]
        lines = [l.strip() for l in code.splitlines() if l.strip()]
        imp = [l for l in lines if l.startswith(("from ", "import "))]
        uses = [l for l in lines if not l.startswith(("from ", "import ", "#"))]
        head = user.split("CODE:")[0]
        ev = (imp or lines or [""])[0][:100]
        first = ev.lower()
        if "exceptions" in first or "error" in first or "contrib" in first or "TYPE_CHECKING" in code:
            level = "low"
        elif "poolmanager" in first or "connectionpool" in first or "retry" in first:
            level = "high"
        else:
            level = "medium"
        if opts.get("temperature", 0) > 0 and (_h(user, opts.get("seed", 0)) % 5 == 0):
            order = ["low", "medium", "high"]
            level = order[max(0, min(2, order.index(level) + (1 if _h(opts.get("seed", 0), user, 1) % 2 else -1)))]
        return {"level": level, "why": f"offline rule: {len(uses)} use-lines next to the import.", "evidence": ev}
    if "TASK: find-weaknesses" in system:
        text = user.split("-----\n", 1)[-1]
        pats = [("except:", "Bare except hides real errors", "medium"), ("except Exception", "Very broad exception handler", "low"),
                ("TODO", "Unfinished work noted in code", "low"), ("FIXME", "Known problem noted in code", "medium"),
                ("eval(", "Dynamic code execution", "high"), ("verify=False", "TLS verification disabled", "high")]
        found = []
        for ln in text.splitlines():
            for p, title, sev in pats:
                if p in ln and len(found) < 3 and ln.strip()[:120] not in [f["quote"] for f in found]:
                    found.append({"title": title, "severity": sev, "quote": ln.strip()[:120],
                                  "why": "offline rule matched this line.", "fix_idea": "offline rule: review and narrow this."})
        return {"findings": found}
    if "TASK: write-brief" in system:
        facts = json.loads(user.split("FACTS:", 1)[1].split("\n\nYour last answer", 1)[0])
        name = facts["file"].split("::")[-1]
        f0 = (facts.get("findings") or [None])[0]
        title = (f0["title"] + " in " + name) if f0 else "Add a focused test for " + name
        return {"title": title, "why_it_matters": "Offline rule: other files depend on this one, so a small safe change here protects them.",
                "how_to_test": ["Run the existing tests.", "Add a test that fails before and passes after.", "Run the tests again."],
                "draft_pr_title": title, "draft_pr_body": "Offline rule: draft text only.", "difficulty": "easy"}
    if "TASK: translate-brief" in system:
        d = json.loads(user)
        return {"why_it_matters": "[offline stand-in translation] " + d["why_it_matters"], "how_to_test": ["[offline] " + x for x in d["how_to_test"]]}
    if "TASK: read-advisory" in system:
        names = re.findall(r"[A-Za-z_][A-Za-z0-9_\.]{3,}", user)
        return {"package": names[0] if names else "", "module_hint": "", "kind": "unknown"}
    if "TASK: controller" in system:
        st = json.loads(user.split("STATE:", 1)[1])
        done = [h["tool"] for h in st["history"]]
        q = st["question"].lower()
        if "find_module" not in done:
            words = [w for w in re.findall(r"[a-z0-9_/\.]+\.py|[a-z][a-z0-9_]{3,}", q)
                     if w not in ("what", "would", "which", "files", "most", "risk", "does", "this", "that", "with", "from", "happen", "compromised", "broken", "bug", "if")]
            return {"action": "tool", "name": "find_module", "args": {"query": words[0] if words else "urllib3"}}
        if "simulate" not in done:
            ids = st["history"][-1]["result"].get("matches", [])[:1]
            return {"action": "tool", "name": "simulate", "args": {"seeds": ids, "runs": 10000, "rng_seed": 42}}
        if "compare_weakest_link" not in done:
            return {"action": "tool", "name": "compare_weakest_link", "args": {}}
        if "harden" not in done and ("protect" in q or "fix" in q or "first" in q):
            return {"action": "tool", "name": "harden", "args": {}}
        if ("contribut" in q or "where" in q) and "fix_first" not in done:
            return {"action": "tool", "name": "fix_first", "args": {}}
        ev = {h["tool"]: h for h in st["history"] if "result" in h}
        sim = ev.get("simulate", {}).get("result", {}); cmp_ = ev.get("compare_weakest_link", {}).get("result", {"rows": []})
        claims = []
        top = (sim.get("top") or [None])[0]
        if top is None:
            return {"action": "answer", "summary": "Offline stand-in answer.", "claims": []}
        claims.append({"text": f"{top['file']} is the most exposed file, hit in {top['hit_prob']:.0%} of runs.",
                       "evidence": [ev["simulate"]["id"]], "numbers": {"hit_prob": top["hit_prob"]}})
        if cmp_.get("rows") and cmp_["rows"][0]["hit_prob"] > cmp_["rows"][0]["weakest_link"] + 0.05:
            r = cmp_["rows"][0]
            claims.append({"text": f"A weakest-link rule gives {r['file']} only {r['weakest_link']:.0%}, but the simulation gives {r['hit_prob']:.0%}.",
                           "evidence": [ev["compare_weakest_link"]["id"]],
                           "numbers": {"weakest_link": r["weakest_link"], "hit_prob": r["hit_prob"]}})
        if "fix_first" in ev and ev["fix_first"]["result"].get("fix_first"):
            r = ev["fix_first"]["result"]["fix_first"][0]
            claims.append({"text": f"Contribute to {r['file']} first: a failure there reaches {r['downstream_files_hit_on_average']} files on average.",
                           "evidence": [ev["fix_first"]["id"]], "numbers": {"downstream_files": r["downstream_files_hit_on_average"]}})
        return {"action": "answer", "summary": "Offline stand-in answer.", "claims": claims}
    return {}
