#!/usr/bin/env python3
"""Fault Lines / Fix-First Map, end to end:
   python run_demo.py --graph corpus/merged.json --corpus ./corpus --question "Where should I contribute first?" --out out
   (no --epicenter needed: the file with the biggest blast radius is used)"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from faultlines.modgraph import build_module_graph, RELATION_PRIOR
from faultlines.gemma import Gemma
from faultlines.jury import Jury
from faultlines.assess import assess_edges, edge_probabilities, edge_distributions, edge_key, LEVEL_P
from faultlines.harness import World, ask
from faultlines.scenario import read_advisory
from faultlines.weakness import find_weaknesses
from faultlines.briefs import make_briefs, briefs_markdown, LANGS
from faultlines.render import build_data, write_html

ap = argparse.ArgumentParser()
ap.add_argument("--graph", required=True, help="merged Graphify graph.json (see build_corpus.sh)")
ap.add_argument("--corpus", required=True, help="folder holding the cloned repos (to show Gemma real code)")
ap.add_argument("--epicenter", help="file id like urllib3::src/urllib3/connectionpool.py, a path fragment, or a project name (default: the file with the biggest blast radius)")
ap.add_argument("--advisory-text", help="paste a security advisory; Gemma finds the epicenter")
ap.add_argument("--advisory-image", help="screenshot of an advisory; Gemma reads the image")
ap.add_argument("--question", action="append", default=[], help="question for the Gemma harness (repeatable)")
ap.add_argument("--backend", default=None, help="ollama | gemini | mock (default: FL_BACKEND or ollama)")
ap.add_argument("--backup-backend", default=None, help="used if the main model keeps failing, e.g. gemini")
ap.add_argument("--scope", default="cross", choices=["cross", "all"], help="which links Gemma judges")
ap.add_argument("--runs", type=int, default=10000)
# --- using more of Gemma -------------------------------------------------------------------
ap.add_argument("--jury-n", type=int, default=3, help="votes per link from the main model (1 = old single-call behaviour)")
ap.add_argument("--big-backend", default=None, help="add a second, bigger model to the jury, e.g. gemini (online Gemma)")
ap.add_argument("--big-model", default=None, help="model name for the big jury member (else FL_MODEL default for that backend)")
ap.add_argument("--big-n", type=int, default=3, help="votes per link from the big model")
ap.add_argument("--think", action="store_true", help="ask Ollama for thinking mode (unverified on your model: run selftest.py)")
ap.add_argument("--native-tools", action="store_true", help="use Ollama's native tool calling for the controller")
ap.add_argument("--candidates", type=int, default=60, help="how many shared files to test as epicenters when ranking blast radius")
ap.add_argument("--weakness-top", type=int, default=5, help="read this many top blast-radius files whole and look for weaknesses (0 = skip)")
ap.add_argument("--max-file-chars", type=int, default=120000, help="longest file text shown to Gemma")
ap.add_argument("--briefs", type=int, default=3, help="write this many contribution briefs (0 = skip)")
ap.add_argument("--lang", default="en", choices=sorted(LANGS), help="also explain the briefs in this language (en, hi, mr)")
ap.add_argument("--out", default="out")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

model = Gemma(a.backend, think=True if a.think else None)
backup = Gemma(a.backup_backend) if a.backup_backend else None
members = [(model, max(1, a.jury_n))]
if a.big_backend:
    members.append((Gemma(a.big_backend, a.big_model), a.big_n))
jury = Jury(members)
print(f"Model: {model.name}" + (f"   Jury: {jury.name}" if jury.name != model.name else ""))
print("1/8 Building the file-level map and stitching links between projects...")
mg = build_module_graph(a.graph)
cross = [e for e in mg["edges"] if e["cross_repo"]]
print(f"    {len(mg['modules'])} files, {len(mg['edges'])} links, {len(cross)} between projects")
print("2/8 The Gemma jury judges each link between projects (cached, so reruns are instant)...")
cache = os.path.join(a.out, "judgements.json")
assess = assess_edges(mg, jury, a.corpus, cache, scope=a.scope)
probs, trust = edge_probabilities(mg, assess)
dists = edge_distributions(mg, assess)
njudged = sum(1 for x in trust if x == "model-judged")
judged = [v for v in assess.values() if v and v.get("level") and v.get("verified")]
low_agree = sum(1 for v in judged if v.get("agreement", 1) < 0.7)
print(f"    {njudged} of {len(cross)} links between projects judged by the model" + ("" if njudged == len(cross) else "; the rest use default assumptions (labelled as hypotheses)"))
if jury.members and (a.jury_n > 1 or a.big_backend):
    print(f"    {low_agree} of {len(judged)} judged links had under 70% agreement; {len(dists)} feed their vote spread into the simulation")
jury_info = {"name": jury.name, "judged": len(judged), "low_agreement": low_agree, "spread_links": len(dists),
             "models_disagree": sum(1 for v in judged if v.get("models_disagree"))}
world = World(mg, probs, trust, assess, a.runs, dists)
print("3/8 Fix-first ranking: simulating a failure in every shared file (same dice for each)...")
ranking = world.sim.blast_ranking(world.sim.candidates(a.candidates))
for r in ranking[:5]:
    print(f"    {r['id'].split('/')[-1]} ({r['repo']}): hits {r['downstream_files']} files on average, reaches {len(r['other_projects'])} other project(s)")
print("4/8 Choosing the epicenter...")
seeds = None
if a.advisory_text or a.advisory_image:
    sc = read_advisory(model, world, a.advisory_text or "", a.advisory_image)
    print("    advisory ->", json.dumps(sc)[:300])
    seeds = sc.get("seeds") if sc.get("ok") else None
if not seeds and a.epicenter:
    seeds = world._resolve(a.epicenter)
if not seeds and not a.epicenter and ranking:
    seeds = [ranking[0]["id"]]
    print("    no epicenter given: using the file with the biggest blast radius")
if not seeds:
    sys.exit("No epicenter found. Use --epicenter with a project name or a file path fragment.")
world.default_seeds = list(seeds)
name = (seeds[0].split("::")[0] + " (the whole project)") if len(seeds) > 1 else (
    seeds[0].split("::")[0] + "'s " + seeds[0].split("/")[-1])
print("    epicenter:", name)
print(f"5/8 Simulating {a.runs:,} quakes...")
res = world.sim.run([world.sim.ix[s] for s in seeds], a.runs, 42)
world.last = res
print("6/8 Protect-first ranking and assumption check...")
hard = world.sim.hardening([world.sim.ix[s] for s in seeds])
sens = world.sim.sensitivity([world.sim.ix[s] for s in seeds])
weak_by_file, briefs = {}, []
if a.weakness_top > 0 and ranking:
    print(f"7/8 Gemma reads the top {min(a.weakness_top, len(ranking))} blast-radius files whole and looks for weaknesses...")
    wcache_path = os.path.join(a.out, "weaknesses.json")
    wcache = json.load(open(wcache_path)) if os.path.exists(wcache_path) else {}
    mods = {m["id"]: m for m in mg["modules"]}
    for r in ranking[:a.weakness_top]:
        w = find_weaknesses(model, a.corpus, mods[r["id"]], wcache, a.max_file_chars)
        weak_by_file[r["id"]] = w
        print(f"    {r['id'].split('/')[-1]}: {len(w['findings'])} verified finding(s), {w['dropped']} dropped" + (f" ({w['error']})" if w.get("error") else "") + (" [file truncated]" if w.get("truncated") else ""))
        json.dump(wcache, open(wcache_path, "w"), indent=1)
if a.briefs > 0 and ranking:
    print(f"    Writing {min(a.briefs, len(ranking))} contribution brief(s)" + (f" (+ {LANGS[a.lang]})" if a.lang != "en" else "") + "...")
    briefs = make_briefs(mg, ranking, weak_by_file, model, a.lang, a.briefs)
    open(os.path.join(a.out, "briefs.md"), "w", encoding="utf-8").write(briefs_markdown(briefs, model.name))
    json.dump(briefs, open(os.path.join(a.out, "briefs.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
world.briefs, world.weak = briefs, weak_by_file
print(f"8/8 Asking Gemma {len(a.question)} question(s)...")
answers = []
for q in a.question:
    r = ask(q, world, model, backup, native=a.native_tools)
    answers.append(r)
    for c in r["claims"]:
        print(f"    [{c['status']}] {c['text']}")
assumptions = {"relation_prior": RELATION_PRIOR, "level_p": LEVEL_P}
data = build_data(mg, world.sim, res, trust, assess, hard, sens, answers, model.name, assumptions, name, edge_key,
                  briefs=briefs, jury_info=jury_info)
html = os.path.join(a.out, "fault_lines.html")
write_html(data, html)
json.dump({"epicenter": seeds, "blast_ranking": ranking[:15], "answers": answers, "hardening": hard, "sensitivity": sens, "jury": jury_info,
           "top": sorted([n for n in res["nodes"] if n["id"] not in seeds], key=lambda n: -n["hit_prob"])[:15]},
          open(os.path.join(a.out, "results.json"), "w"), indent=1, default=str)
calls = jury.calls
print(f"\nDone. Open {html} in your browser." + (f"  Briefs: {os.path.join(a.out, 'briefs.md')}" if briefs else "")
      + f"  Model calls: {calls}, broken answers: {jury.failures}")
