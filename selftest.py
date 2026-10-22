#!/usr/bin/env python3
"""Run this FIRST (a few minutes): does your Gemma answer in the strict formats we need?
   python selftest.py            (uses FL_BACKEND, default ollama)     python selftest.py --backend gemini
   Add --think to also test thinking mode.  Tests 4 and 5 only apply to Ollama."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from faultlines.gemma import Gemma
from faultlines.assess import SYSTEM as JUDGE
from faultlines.harness import CONTROLLER, TOOL_SPECS, NATIVE_SYSTEM
from faultlines.weakness import SYSTEM as WEAK, check_findings
from faultlines.briefs import SYSTEM as BRIEF, check_brief
ap = argparse.ArgumentParser(); ap.add_argument("--backend"); ap.add_argument("--think", action="store_true"); a = ap.parse_args()
m = Gemma(a.backend, think=True if a.think else None); print("Testing:", m.name); ok = total = 0
def report(name, good, t, detail=""):
    global ok, total
    total += 1; ok += bool(good)
    print(f"[{'PASS' if good else 'FAIL'}] {name} ({t:.1f}s): {detail[:200]}")

t = time.time()
r = m.json(JUDGE, "A = requests::src/requests/exceptions.py\nB = urllib3::src/urllib3/exceptions.py (30 functions/classes)\nKinds of link: {'imports_from': 1}\nCODE:\nfrom urllib3.exceptions import HTTPError as BaseHTTPError\nclass RequestException(IOError):\n    pass")
good = isinstance(r, dict) and str(r.get("level", "")).lower() in ("low", "medium", "high") and "from urllib3.exceptions" in str(r.get("evidence", ""))
report("1 link judging (what the jury votes with)", good, time.time() - t, json.dumps(r))

t = time.time()
r = m.json(CONTROLLER, "STATE:" + json.dumps({"question": "What breaks if urllib3 is compromised?", "history": []}))
good = isinstance(r, dict) and r.get("action") == "tool" and r.get("name") in ("find_module", "simulate")
report("2 choosing a tool as JSON", good, time.time() - t, json.dumps(r))

t = time.time()
src = "import os\n\ndef load(path):\n    try:\n        return open(path).read()\n    except:\n        return None\n"
r = m.json(WEAK, "FILE: demo::load.py\n-----\n" + src)
kept, dropped = check_findings(src, (r or {}).get("findings"))
report("3 whole-file weakness finding (quote must be real)", bool(kept), time.time() - t, f"{len(kept)} verified, {dropped} dropped: {json.dumps(r)[:120]}")

t = time.time()
facts = {"file": "demo::load.py", "project": "demo", "files_that_depend_on_it_directly": 4, "other_projects_reached_if_it_fails": ["app"],
         "findings": [{"title": "Bare except", "severity": "medium", "line": 6, "quote": "except:", "why": "hides errors", "fix_idea": "catch OSError"}]}
r = m.json(BRIEF, "FACTS:\n" + json.dumps(facts, indent=1))
issues = check_brief(r, facts, {"load.py"}) if r else ["no answer"]
report("4 contribution brief passes the checker", not issues, time.time() - t, "; ".join(issues) or json.dumps(r)[:120])

if m.backend == "ollama":
    t = time.time()
    msg = m.chat_tools([{"role": "system", "content": NATIVE_SYSTEM}, {"role": "user", "content": "What breaks if urllib3 is compromised?"}], TOOL_SPECS)
    calls = (msg or {}).get("tool_calls") or []
    names = [c.get("function", {}).get("name") for c in calls]
    report("5 native tool calling", bool(calls) and names[0] in ("find_module", "simulate"), time.time() - t, f"tools: {names}" if msg else "no reply (model may not support tools)")
else:
    print("[SKIP] 5 native tool calling (Ollama only)")
if a.think:
    print(f"[INFO] thinking mode {'accepted' if m.think_ok else 'REJECTED by the server, so it was ignored'}")
print(f"\n{ok}/{total} passed. Model calls: {m.calls}, broken answers: {m.failures}")
if ok < total:
    print("Not reliable enough for the failed parts: run without --native-tools, use --backend gemini, or try a bigger model with FL_MODEL=gemma4:26b")
