"""Step 4: the harness. Gemma never touches the graph or does maths. It picks TOOLS; our code runs
them and keeps a numbered evidence list (E1, E2, ...). Gemma then writes claims that must cite that
evidence, and a verifier (plain code, not an AI) checks every claim before anything is shown."""
import json, re
import numpy as np
from faultlines.simulate import Sim
from faultlines.assess import edge_key

CONTROLLER = """TASK: controller
You answer questions about how a problem (a "shock") spreads between connected code projects.
You CANNOT calculate or read files. You can only call tools, then write an answer from their results.
Reply with JSON only, one of:
 {"action":"tool","name":"<tool>","args":{...}}
 {"action":"answer","summary":"<one sentence>","claims":[{"text":"...","evidence":["E1"],"numbers":{"<name>":<value>}}]}
Tools:
 find_module {"query": "<part of a file name or a project name>"}  -> matching file ids
 simulate {"seeds": ["<file id or project name>"], "runs": 10000, "rng_seed": 42} -> hit chance per file (empty seeds = this run's epicenter)
 compare_weakest_link {} -> files where the simulation and the simple weakest-link rule disagree (needs simulate first)
 paths {"target": "<file id>"} -> most likely route from the epicenter to a file (needs simulate first)
 harden {} -> which single file to protect first (needs simulate first)
 explain_link {"dependent": "<file id>", "dependency": "<file id>"} -> the recorded judgement for one link
 fix_first {} -> the files where a failure would reach the most downstream files and projects, with weaknesses found and a draft title
 weaknesses {"file": "<file id>"} -> weaknesses Gemma found by reading the whole file (each quotes real code)
Rules: every number you write in a claim's text must also be listed in its "numbers" with the exact value from the cited evidence. Cite evidence ids. No evidence, no claim. Use at most 6 tool calls."""

class World:
    def __init__(self, mg, probs, trust, assessments, runs_default=10000, dists=None):
        self.mg, self.trust, self.assess = mg, trust, assessments
        self.sim = Sim(mg, probs, dists)
        self.last = None
        self.ev = {}
        self._n = 0                      # evidence counter (kept apart from self.ev)
        self.briefs = []                 # set by run_demo: contribution briefs
        self.default_seeds = []          # set by run_demo: the run's epicenter, used when a question names none
        self.weak = {}                   # set by run_demo: weaknesses per file
        self.runs_default = runs_default
        self.edge_ix = {edge_key(e): i for i, e in enumerate(mg["edges"])}

    # ---- helpers -------------------------------------------------------------------
    def _resolve(self, x):
        ids = [m["id"] for m in self.mg["modules"]]
        if x in self.sim.ix:
            return [x]
        repo = [i for i in ids if i.split("::")[0] == x]
        if repo:
            return repo
        q = str(x).lower().replace("::", "/")
        return [i for i in ids if q in i.lower().replace("::", "/")][:1]

    def _record(self, tool, args, result, hypothesis=False, why=""):
        self._n += 1
        eid = f"E{self._n}"
        nums = []
        def walk(o):
            if isinstance(o, dict):
                [walk(v) for v in o.values()]
            elif isinstance(o, list):
                [walk(v) for v in o]
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                nums.append(float(o))
        walk(result)
        self.ev[eid] = {"id": eid, "tool": tool, "args": args, "result": result, "numbers": nums,
                        "hypothesis": hypothesis, "why_hypothesis": why}
        return {"tool": tool, "args": args, "id": eid, "result": result}

    # ---- tools ---------------------------------------------------------------------
    def find_module(self, query):
        q = str(query).lower()
        hits = [m for m in self.mg["modules"] if q in m["id"].lower()]
        hits.sort(key=lambda m: -m["symbols"])
        repos = sorted({m["id"].split("::")[0] for m in self.mg["modules"]})
        return self._record("find_module", {"query": query},
                            {"matches": [m["id"] for m in hits[:6]], "projects": repos if not hits else []})

    def simulate(self, seeds, runs=None, rng_seed=42):
        runs = int(min(max(int(runs or self.runs_default), 500), 20000))
        ids = [i for s in (seeds if isinstance(seeds, list) else [seeds]) for i in self._resolve(s)]
        if not ids and self.default_seeds:
            ids = list(self.default_seeds)
        if not ids:
            return self._record("simulate", {"seeds": seeds}, {"error": "no matching file or project"})
        r = self.sim.run([self.sim.ix[i] for i in ids], runs, int(rng_seed))
        self.last = r
        rows = sorted((n for n in r["nodes"] if n["id"] not in r["seeds"]), key=lambda n: -n["hit_prob"])[:8]
        per_repo = {}
        for n in r["nodes"]:
            per_repo.setdefault(n["repo"], []).append(n["expected_loss"])
        cross = [i for i, e in enumerate(self.mg["edges"]) if e["cross_repo"]]
        weak = [i for i in cross if self.trust[i] != "model-judged"]
        res = {"epicenter": r["seeds"][:3], "epicenter_files": len(r["seeds"]), "runs": runs, "rng_seed": int(rng_seed),
               "expected_loss_total": r["expected_loss_total"], "tail_loss_5pct_worst_runs": r["tail_loss_cvar"],
               "top": [{"file": n["id"], "hit_prob": n["hit_prob"], "exposure": n["exposure"]} for n in rows],
               "expected_loss_by_project": {k: round(sum(v), 1) for k, v in per_repo.items()}}
        why = f"{len(weak)} of {len(cross)} cross-project links use default assumptions or uncertain stitching" if weak else ""
        return self._record("simulate", {"seeds": ids[:3], "runs": runs, "rng_seed": int(rng_seed)}, res, bool(weak), why)

    def compare_weakest_link(self):
        if not self.last:
            return self._record("compare_weakest_link", {}, {"error": "run simulate first"})
        rows = sorted((n for n in self.last["nodes"] if n["id"] not in self.last["seeds"]),
                      key=lambda n: -(n["hit_prob"] - n["weakest_link"]))[:5]
        return self._record("compare_weakest_link", {}, {"rows": [
            {"file": n["id"], "weakest_link": n["weakest_link"], "hit_prob": n["hit_prob"]} for n in rows]})

    def paths(self, target):
        if not self.last:
            return self._record("paths", {}, {"error": "run simulate first"})
        t = self._resolve(target)
        if not t:
            return self._record("paths", {"target": target}, {"error": "unknown file"})
        best = None
        for s in self.last["seeds"]:
            r = self.sim.best_path(self.sim.ix[s], self.sim.ix[t[0]])
            if r and (not best or r[1] > best[1]):
                best = r
        if not best:
            return self._record("paths", {"target": t[0]}, {"error": "no route from the epicenter"})
        steps, bad = [], []
        for e in best[0]:
            edge = self.mg["edges"][e]
            steps.append({"from": edge["dependency"], "to": edge["dependent"], "chance": round(float(self.sim.p[e]), 3),
                          "basis": self.trust[e]})
            if self.trust[e] != "model-judged" and edge["cross_repo"]:
                bad.append(edge_key(edge))
        res = {"target": t[0], "route_chance": round(float(best[1]), 4), "steps": steps}
        return self._record("paths", {"target": t[0]}, res, bool(bad), "route crosses a link that is not model-judged" if bad else "")

    def harden(self):
        if not self.last:
            return self._record("harden", {}, {"error": "run simulate first"})
        seeds = [self.sim.ix[s] for s in self.last["seeds"]]
        rows = self.sim.hardening(seeds, runs=3000)[:5]
        base = rows[0]["loss_before"] if rows else 0
        return self._record("harden", {}, {"protect_first": [
            {"file": r["id"], "loss_removed": r["loss_removed"], "percent_removed": round(100 * r["loss_removed"] / base, 1) if base else 0}
            for r in rows]})

    def explain_link(self, dependent, dependency):
        k = f"{dependent}|{dependency}"
        a = self.assess.get(k)
        if k not in self.edge_ix:
            return self._record("explain_link", {"dependent": dependent, "dependency": dependency}, {"error": "no such link"})
        e = self.mg["edges"][self.edge_ix[k]]
        res = {"chance": round(float(self.sim.p[self.edge_ix[k]]), 3), "basis": self.trust[self.edge_ix[k]],
               "judgement": (a or {}).get("level"), "reason": (a or {}).get("why"), "quote": (a or {}).get("evidence"),
               "link_kinds": e["relations"]}
        return self._record("explain_link", {"dependent": dependent, "dependency": dependency}, res)

    def fix_first(self):
        if not self.briefs:
            return self._record("fix_first", {}, {"error": "no contribution briefs were made in this run"})
        rows = [{"file": b["file"], "other_projects_reached": len(b["other_projects"]), "downstream_files_hit_on_average": b["downstream_files"],
                 "downstream_exposure": b["downstream_loss"], "files_that_depend_on_it": b["dependents"],
                 "weaknesses_found": len(b["findings"]), "draft_title": b["title"]} for b in self.briefs]
        return self._record("fix_first", {}, {"fix_first": rows})

    def weaknesses(self, file):
        t = self._resolve(file)
        if not t or t[0] not in self.weak:
            return self._record("weaknesses", {"file": file}, {"error": "no weakness scan for this file (only top files are scanned)"})
        w = self.weak[t[0]]
        return self._record("weaknesses", {"file": t[0]}, {"file": t[0], "findings": [
            {"line": f["line"], "severity": f["severity"], "title": f["title"], "quote": f["quote"]} for f in w["findings"]]})

# ---- verifier (plain code, no AI) --------------------------------------------------------
_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(%?)")

def verify(answer, world):
    report = []
    for c in (answer or {}).get("claims", []):
        text, issues = str(c.get("text", "")), []
        evs = [world.ev[e] for e in c.get("evidence", []) if e in world.ev]
        if not evs:
            issues.append("cites no real evidence")
        pool = [x for e in evs for x in e["numbers"]]
        claimed = {k: v for k, v in (c.get("numbers") or {}).items() if isinstance(v, (int, float))}
        for k, v in claimed.items():
            if not any(abs(v - p) <= 0.006 + 0.002 * abs(p) for p in pool):
                issues.append(f"number {k}={v} is not in the cited evidence")
        for m in _NUM.finditer(text):
            val = float(m.group(1)) / (100 if m.group(2) else 1)
            if not m.group(2) and val <= 10:
                continue                                    # tiny counts are not checked
            if not any(abs(val - p) <= 0.006 + 0.002 * abs(p) for p in pool + list(claimed.values())):
                issues.append(f"'{m.group(0)}' in the text is not backed by evidence")
        known = {m["file"].split("/")[-1] for m in world.mg["modules"]} | {m["id"] for m in world.mg["modules"]}
        for f in re.findall(r"[\w/\.\-:]+\.py", text):
            if f not in known and f.split("/")[-1] not in known:
                issues.append(f"file '{f}' does not exist in the map")
        if issues:
            status = "rejected"
        elif any(e["hypothesis"] for e in evs):
            status = "hypothesis"
        else:
            status = "checked"
        report.append({"text": text, "status": status, "evidence": c.get("evidence", []), "issues": issues,
                       "why_hypothesis": [e["why_hypothesis"] for e in evs if e["hypothesis"]]})
    if not report:
        report = [{"text": "Insufficient evidence to answer.", "status": "rejected", "evidence": [], "issues": ["no claims"], "why_hypothesis": []}]
    return report

# ---- tools the controller may call -------------------------------------------------------
TOOLS = ("find_module", "simulate", "compare_weakest_link", "paths", "harden", "explain_link", "fix_first", "weaknesses")

def _spec(name, desc, props=None, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props or {}, "required": list(required)}}}

S = {"type": "string"}
TOOL_SPECS = [  # the same tools, described for Ollama's NATIVE tool calling
    _spec("find_module", "Find file ids matching part of a file name or a project name.", {"query": S}, ["query"]),
    _spec("simulate", "Run the Monte Carlo shock simulation from one or more epicenters (file ids or project names; empty = this run's epicenter).",
          {"seeds": {"type": "array", "items": S}, "runs": {"type": "integer"}, "rng_seed": {"type": "integer"}}, ["seeds"]),
    _spec("compare_weakest_link", "Files where the simulation and the simple weakest-link rule disagree (after simulate)."),
    _spec("paths", "Most likely route from the epicenter to a file (after simulate).", {"target": S}, ["target"]),
    _spec("harden", "Which single file to protect first (after simulate)."),
    _spec("explain_link", "The recorded judgement for one link.", {"dependent": S, "dependency": S}, ["dependent", "dependency"]),
    _spec("fix_first", "Files where a failure would reach the most downstream files and projects, with weaknesses found and a draft title."),
    _spec("weaknesses", "Weaknesses Gemma found by reading a whole file.", {"file": S}, ["file"]),
]
NATIVE_SYSTEM = ("You answer questions about how a problem (a shock) spreads between connected code projects, and where a "
                 "contribution would help most. You cannot calculate or read files: call the tools to gather evidence. "
                 "When you have enough evidence, stop calling tools and reply with the single word: done.")

def _run_tool(world, name, args):
    fn = getattr(world, name, None) if name in TOOLS else None
    if not fn:
        return {"tool": str(name), "args": args, "id": "-", "error": "unknown tool"}
    try:
        return fn(**args)
    except TypeError as err:
        return {"tool": str(name), "args": args, "id": "-", "error": f"bad arguments: {err}"}
    except Exception as err:                       # a tool bug must not crash the whole answer
        return {"tool": str(name), "args": args, "id": "-", "error": f"tool failed: {err}"}

def _native_collect(question, world, model, max_steps, log):
    """Let the model pick tools through the native tool-calling API. Returns the evidence history ([] if unavailable)."""
    msgs = [{"role": "system", "content": NATIVE_SYSTEM}, {"role": "user", "content": question}]
    history = []
    for _ in range(max_steps):
        msg = model.chat_tools(msgs, TOOL_SPECS)
        calls = (msg or {}).get("tool_calls") or []
        if not calls:
            break
        msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            fn = c.get("function", {})
            name, args = fn.get("name"), fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            rec = _run_tool(world, name, args)
            history.append(rec)
            log(f"  native step: {rec['tool']} -> {rec.get('id')}")
            msgs.append({"role": "tool", "tool_name": str(name), "content": json.dumps(rec.get("result", rec), default=str)[:6000]})
    return history

# ---- the loop ----------------------------------------------------------------------------
def ask(question, world, model, fallback=None, max_steps=6, log=print, native=False):
    history, trace, fails, active, repaired = [], [], 0, model, False   # `repaired` is per question
    if native and getattr(model, "backend", "") == "ollama":
        history = _native_collect(question, world, model, max_steps, log)
        trace = [{"tool": h["tool"], "id": h.get("id")} for h in history]
        if not history:
            log("  (native tool calling gave no tool calls; using the JSON tool loop)")
    start_forced = bool(history)
    for step in range(max_steps + 2):
        state = json.dumps({"question": question, "history": [{"tool": h["tool"], "args": h["args"], "id": h["id"], "result": h["result"]} if "result" in h else h for h in history]}, default=str)
        force = step >= max_steps or start_forced
        out = active.json(CONTROLLER + ("\nYou have enough evidence. Answer now." if force else ""), "STATE:" + state)
        if not out:
            fails += 1
            if fallback and active is model and fails >= 2:
                log(f"  [switching to backup model: {fallback.name}]"); active = fallback
            if fails > 5:
                break
            continue
        if out.get("action") == "tool" and not force:
            rec = _run_tool(world, out.get("name"), out.get("args") or {})
            history.append(rec); trace.append({"tool": rec["tool"], "id": rec.get("id")})
            log(f"  step {step + 1}: {rec['tool']} -> {rec.get('id')}")
            continue
        if out.get("action") == "answer":
            report = verify(out, world)
            if any(r["status"] == "rejected" for r in report) and not repaired:
                repaired = True
                issues = [{"claim": r["text"], "problems": r["issues"]} for r in report if r["status"] == "rejected"]
                history.append({"tool": "verifier", "args": {}, "id": "-", "result": {"rejected_claims": issues, "instruction": "rewrite the answer using only numbers from the evidence"}})
                log("  verifier rejected some claims; giving the model one chance to fix them")
                continue
            return {"question": question, "summary": out.get("summary", ""), "claims": report, "trace": trace,
                    "evidence": dict(world.ev), "model": active.name}
    return {"question": question, "summary": "Insufficient evidence: the model did not finish.", "claims": verify({}, world),
            "trace": trace, "evidence": dict(world.ev), "model": active.name}
