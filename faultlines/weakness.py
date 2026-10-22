"""Weakness judge: Gemma reads a WHOLE file (not a snippet) and points at fragile code.

This uses Gemma 4's long memory (the small E2B/E4B models are listed with a 128K-token window; check yours).
Every finding must quote a line that really exists in the file; findings with an invented quote are dropped,
and we compute the line number ourselves (never trusting the model's line numbers)."""
import re
from faultlines.assess import _find_repo_dir

SYSTEM = """TASK: find-weaknesses
You review ONE whole source file from an open-source project that many other projects depend on.
Find up to 5 concrete places where a small contribution would make the file safer or more robust.
Good findings: unchecked input, swallowed or too-broad exceptions, missing timeouts or size limits, missing validation,
unsafe defaults, TODO/FIXME notes, deprecated usage, missing tests or documentation for tricky code.
Do not report style or naming.
Reply with JSON only:
{"findings":[{"title":"<short>","severity":"low"|"medium"|"high","quote":"<ONE line copied exactly from the FILE, max 120 characters>","why":"<one sentence>","fix_idea":"<one sentence>"}]}
Never invent a quote. If you find nothing worth reporting, reply {"findings":[]}."""

def _norm(s):
    return re.sub(r"\s+", " ", s or "").strip()

def read_module(corpus, module):
    root = _find_repo_dir(corpus, module["repo"])
    if not root:
        return None
    path = f"{root}/{module['file']}"
    try:
        return open(path, errors="ignore").read()
    except OSError:
        return None

def check_findings(text, findings):
    """Keep only findings whose quote is really in the file; attach the line number we find ourselves."""
    nt, lines, kept, dropped = _norm(text), text.splitlines(), [], 0
    for f in findings or []:
        if not isinstance(f, dict):
            dropped += 1
            continue
        q = _norm(str(f.get("quote", "")))
        if not q or q not in nt:
            dropped += 1
            continue
        line = next((i + 1 for i, l in enumerate(lines) if q in _norm(l)), None)
        sev = str(f.get("severity", "low")).lower()
        kept.append({"title": str(f.get("title", ""))[:120], "severity": sev if sev in ("low", "medium", "high") else "low",
                     "quote": q[:160], "line": line, "why": str(f.get("why", ""))[:240],
                     "fix_idea": str(f.get("fix_idea", ""))[:240]})
    return kept, dropped

def find_weaknesses(model, corpus, module, cache=None, max_chars=120000):
    key = f"{model.name}|{module['id']}"
    if cache is not None and key in cache:
        return cache[key]
    text = read_module(corpus, module)
    if text is None:
        return {"file": module["id"], "findings": [], "dropped": 0, "truncated": False, "chars": 0, "error": "source not found"}
    truncated = len(text) > max_chars
    shown = text[:max_chars]
    out = model.json(SYSTEM, f"FILE: {module['id']}\n-----\n{shown}")
    if not out:
        return {"file": module["id"], "findings": [], "dropped": 0, "truncated": truncated, "chars": len(shown), "error": "model gave no usable answer"}
    kept, dropped = check_findings(shown, out.get("findings"))
    res = {"file": module["id"], "findings": kept, "dropped": dropped, "truncated": truncated, "chars": len(shown), "source": model.name}
    if cache is not None:
        cache[key] = res
    return res
