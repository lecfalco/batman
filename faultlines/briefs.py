"""Brief writer: turn "protect this file first" into a ready-to-use contribution brief.

Division of labour (so the numbers stay honest):
  * CODE computes the impact (the file's blast radius: how many files and projects a failure there reaches) and finds the line numbers.
  * GEMMA writes the human parts: why it matters, how to test, a draft pull-request title and body, and can translate
    the explanation into Hindi or Marathi. Gemma is told not to write numbers; a check rejects percentages and unknown files.
If Gemma's text fails the check twice, a plain template brief is used instead and labelled as such."""
import json, re

LANGS = {"en": "English", "hi": "Hindi", "mr": "Marathi"}

SYSTEM = """TASK: write-brief
You help a new contributor make a useful first contribution to an open-source project. You get FACTS about one file
(computed by code, trust them; the file has a large blast radius, meaning a failure there reaches many other files and projects) and some WEAKNESS findings (each quotes real code). Reply with JSON only:
{"title":"<short contribution title>","why_it_matters":"<2-3 sentences, plain words>","how_to_test":["<step>","<step>","<step>"],
 "draft_pr_title":"<one line>","draft_pr_body":"<short markdown: what changed, why, how it was tested>",
 "difficulty":"easy"|"medium"|"hard"}
Rules:
- Do NOT write any percentages or statistics; the impact numbers are added separately.
- Only mention files and findings that appear in FACTS. Do not invent file names, functions or commands you cannot see.
- If there are no findings, propose adding a focused test or documentation for the file and say that no specific bug was found.
- how_to_test: generic but concrete steps (run the project's existing tests, add a test that fails before and passes after)."""

TRANSLATE = """TASK: translate-brief
Translate the values of the JSON the user gives you into {lang}. Keep file names, code, commands and project names in English.
Reply with JSON only, same keys: {{"why_it_matters":"...","how_to_test":["...","..."]}}"""

def build_facts(mg, row, weak):
    """Facts for one high-blast-radius file, all computed by code."""
    return {"file": row["id"], "project": row["repo"],
            "files_that_depend_on_it_directly": row["direct_dependents"],
            "other_projects_reached_if_it_fails": row["other_projects"],
            "findings": [{k: f[k] for k in ("title", "severity", "line", "quote", "why", "fix_idea")}
                         for f in (weak or {}).get("findings", [])]}

_PCT = re.compile(r"\d+(?:\.\d+)?\s*%")

def check_brief(b, facts, known_files):
    issues = []
    if not isinstance(b, dict):
        return ["not a JSON object"]
    for k in ("title", "why_it_matters", "draft_pr_title", "draft_pr_body"):
        if not isinstance(b.get(k), str) or not b[k].strip():
            issues.append(f"missing {k}")
    if not isinstance(b.get("how_to_test"), list) or not all(isinstance(x, str) for x in b.get("how_to_test", [])) or not b.get("how_to_test"):
        issues.append("missing how_to_test list")
    text = json.dumps(b)
    if _PCT.search(text):
        issues.append("contains a percentage (impact numbers are added by code)")
    for f in re.findall(r"[\w/\.\-]+\.py", text):
        if f not in known_files and f.split("/")[-1] not in known_files:
            issues.append(f"mentions unknown file {f}")
    return issues

def template_brief(facts):
    f0 = facts["findings"][0] if facts["findings"] else None
    name = facts["file"].split("::")[-1]
    if f0:
        title = f"{f0['title']} in {name}"
        why = (f"A failure in {name} would reach other files and projects. {f0['why']} A small fix here helps everything that depends on it.")
    else:
        title = f"Add a focused test or documentation for {name}"
        why = f"A failure in {name} would reach other files and projects, and no specific bug was found, so a test or clearer docs is the safest first contribution."
    return {"title": title, "why_it_matters": why,
            "how_to_test": ["Run the project's existing test suite.", "Add a test that fails before your change and passes after.", "Run the suite again."],
            "draft_pr_title": title, "draft_pr_body": f"What: {title}.\n\nWhy: {why}\n\nTested: existing suite plus one new test.",
            "difficulty": "medium", "source": "template"}

def write_brief(model, facts, known_files):
    user = "FACTS:\n" + json.dumps(facts, indent=1)
    b, issues = model.json(SYSTEM, user), []
    for attempt in range(2):
        issues = check_brief(b, facts, known_files) if b else ["no usable answer"]
        if not issues:
            b["source"] = model.name
            return b
        if attempt == 0:
            b = model.json(SYSTEM, user + "\n\nYour last answer was rejected because: " + "; ".join(issues) + ". Fix it.")
    t = template_brief(facts)
    t["rejected_because"] = issues
    return t

def translate_brief(model, brief, lang):
    if lang == "en" or lang not in LANGS:
        return None
    src = {"why_it_matters": brief["why_it_matters"], "how_to_test": brief["how_to_test"]}
    out = model.json(TRANSLATE.format(lang=LANGS[lang]), json.dumps(src, ensure_ascii=False))
    if (out and isinstance(out.get("why_it_matters"), str) and isinstance(out.get("how_to_test"), list)
            and len(out["how_to_test"]) == len(brief["how_to_test"])):
        return {"lang": lang, "language": LANGS[lang], "why_it_matters": out["why_it_matters"],
                "how_to_test": [str(x) for x in out["how_to_test"]],
                "note": "machine-translated by the model; the pull request text stays in English"}
    return None

def make_briefs(mg, ranking, weak_by_file, model, lang="en", top_n=3):
    known = {m["file"].split("/")[-1] for m in mg["modules"]} | {m["id"] for m in mg["modules"]} | {m["file"] for m in mg["modules"]}
    out = []
    for r in ranking[:top_n]:
        facts = build_facts(mg, r, weak_by_file.get(r["id"]))
        br = write_brief(model, facts, known)
        br.update({"file": r["id"], "project": r["repo"], "downstream_loss": r["downstream_loss"], "downstream_files": r["downstream_files"],
                   "dependents": r["direct_dependents"], "other_projects": r["other_projects"], "findings": facts["findings"],
                   "local": translate_brief(model, br, lang)})
        out.append(br)
    return out

def briefs_markdown(briefs, model_name):
    L = [f"# Where to contribute first\n", f"_Generated with {model_name}. Impact numbers are computed by code under stated assumptions, not predicted. "
         "Verify everything before opening a pull request._\n"]
    for i, b in enumerate(briefs, 1):
        L += [f"## {i}. {b['title']}", f"**File:** `{b['file']}`  ", f"**Blast radius:** if it fails, about {b['downstream_files']} other files are hit on average ({b['downstream_loss']} exposure units, under stated assumptions); "
              f"{b['dependents']} files depend on it directly" + (f"; other projects reached: {', '.join(b['other_projects'])}" if b["other_projects"] else "") + ".  ",
              f"**Difficulty:** {b.get('difficulty', '?')}  ", f"**Brief written by:** {b.get('source')}\n", b["why_it_matters"], "\n**Where:**"]
        L += [f"- line {f['line']}: `{f['quote']}` ({f['severity']}): {f['why']}" for f in b["findings"]] or ["- no specific weakness found; add a test or docs"]
        L += ["\n**How to test:**"] + [f"{j}. {s}" for j, s in enumerate(b["how_to_test"], 1)]
        L += ["\n**Draft pull request**", f"Title: {b['draft_pr_title']}\n", b["draft_pr_body"]]
        if b.get("local"):
            L += [f"\n**{b['local']['language']}** ({b['local']['note']})", b["local"]["why_it_matters"]] + [f"{j}. {s}" for j, s in enumerate(b["local"]["how_to_test"], 1)]
        L.append("")
    return "\n".join(L)
