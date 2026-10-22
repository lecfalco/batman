"""Tests for the Fix-First Map additions. Run:  python tests/test_new.py   (or: python -m pytest -q)
No network, no Graphify and no model needed: Gemma is replaced by small scripted fakes."""
import json, os, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from faultlines.jury import Jury, aggregate, LEVEL_P
from faultlines.simulate import Sim
from faultlines.weakness import check_findings
from faultlines.briefs import check_brief, write_brief
from faultlines.harness import World, ask
from faultlines.assess import edge_probabilities
import fixture

class Fake:
    """A scripted stand-in for Gemma. `fn(system, user, temperature, seed)` returns a dict or None."""
    backend = "mock"
    def __init__(self, fn, name="fake"):
        self.fn, self.name, self.calls, self.failures = fn, name, 0, 0
    def json(self, system, user, images=None, retries=3, temperature=0.0, seed=7):
        self.calls += 1
        out = self.fn(system, user, temperature, seed)
        if out is None:
            self.failures += 1
        return out

def tiny(n=3):
    mods = [{"id": f"r::{c}.py", "repo": "r", "file": f"{c}.py", "symbols": 2, "label": f"{c}.py"} for c in "abcd"[:n]]
    edges = [{"dependent": f"r::{'abcd'[i + 1]}.py", "dependency": f"r::{'abcd'[i]}.py", "cross_repo": False, "relations": {"calls": 1},
              "evidence_lines": [], "prior": 0.5, "stitch": None, "stitch_confidence": 1.0, "provenance": "extracted"} for i in range(n - 1)]
    return {"modules": mods, "edges": edges}

def vote(level, quote="x = 1", verified=True, model="m"):
    return {"level": level, "why": "", "evidence": quote, "verified": verified, "model": model}

# ---- jury ---------------------------------------------------------------------------------
def test_jury_ignores_invented_quotes():
    a = aggregate([vote("high", verified=False), vote("high", verified=False), vote("low"), vote("low"), vote("low")], 5, "j")
    assert a["level"] == "low" and a["n_verified"] == 3 and a["verified"] and a["p"] == LEVEL_P["low"]

def test_jury_not_trusted_when_most_votes_fail():
    a = aggregate([vote("low")], 5, "j")                       # only 1 of 5 asked survived
    assert a["verified"] is False

def test_jury_spread_and_agreement():
    a = aggregate([vote("low")] * 3 + [vote("medium")], 4, "j")
    assert a["agreement"] == 0.75 and len(a["dist"]) == 2 and abs(sum(a["dist"].values()) - 1) < 1e-9
    assert abs(a["p"] - (3 * 0.2 + 0.5) / 4) < 1e-9

def test_jury_two_model_sizes_can_disagree():
    code = "x = 1"
    small = Fake(lambda s, u, t, sd: {"level": "low", "why": "w", "evidence": "x = 1"}, "small")
    big = Fake(lambda s, u, t, sd: {"level": "high", "why": "w", "evidence": "x = 1"}, "big")
    a = Jury([(small, 3), (big, 2)]).judge("S", "U", code)
    assert a["models_disagree"] and a["n_asked"] == 5 and small.calls == 3 and big.calls == 2

def test_jury_of_one_keeps_old_name():
    m = Fake(lambda *a: None, "solo")
    assert Jury([(m, 1)]).name == "solo"

# ---- simulator: vote spread -> risk --------------------------------------------------------
def test_vote_spread_changes_each_run():
    mg = tiny(2)
    plain = Sim(mg, [0.5])
    spread = Sim(mg, [0.5], {0: (np.array([0.0, 1.0]), np.array([0.5, 0.5]))})
    a, b = plain.run([0], 6000, 3), spread.run([0], 6000, 3)
    assert abs(a["nodes"][1]["hit_prob"] - 0.5) < 0.03 and abs(b["nodes"][1]["hit_prob"] - 0.5) < 0.03
    assert spread.draw(10, 1).std() >= 0 and not np.array_equal(plain.draw(500, 1), spread.draw(500, 1))

def test_no_spread_means_old_results_unchanged():
    mg = tiny(3)
    assert Sim(mg, [0.4, 0.4]).run([0], 2000, 5)["nodes"] == Sim(mg, [0.4, 0.4], {}).run([0], 2000, 5)["nodes"]

def test_blast_ranking_prefers_the_root():
    rows = Sim(tiny(4), [0.8, 0.8, 0.8]).blast_ranking(cands=[0, 1, 2])
    assert rows[0]["id"] == "r::a.py" and rows[0]["downstream_loss"] > rows[-1]["downstream_loss"]

# ---- weakness judge -----------------------------------------------------------------------
def test_weakness_quote_check_and_line_numbers():
    text = "import os\ntry:\n    go()\nexcept:\n    pass\n"
    kept, dropped = check_findings(text, [{"title": "t", "quote": "except:", "severity": "medium"},
                                          {"title": "ghost", "quote": "eval(user_input)", "severity": "high"}, "junk"])
    assert len(kept) == 1 and dropped == 2 and kept[0]["line"] == 4

# ---- brief writer -------------------------------------------------------------------------
def test_brief_rejects_percentages_and_unknown_files():
    good = {"title": "t", "why_it_matters": "w", "how_to_test": ["a"], "draft_pr_title": "p", "draft_pr_body": "b"}
    known = {"core.py"}
    assert check_brief(good, {}, known) == []
    assert check_brief(dict(good, why_it_matters="removes 55% of risk"), {}, known)
    assert check_brief(dict(good, draft_pr_body="edit ghost.py"), {}, known)

def test_brief_falls_back_to_template():
    facts = {"file": "r::core.py", "project": "r", "files_that_depend_on_it_directly": 2, "other_projects_reached_if_it_fails": [], "findings": []}
    m = Fake(lambda *a: {"title": "x", "why_it_matters": "99% safer"}, "bad")
    b = write_brief(m, facts, {"core.py"})
    assert b["source"] == "template" and b["rejected_because"]

# ---- harness: the repair bug and the default epicenter ------------------------------------------
def scripted_controller(system, user, temperature, seed):
    st = json.loads(user.split("STATE:", 1)[1])
    h = st["history"]
    if not h:
        return {"action": "tool", "name": "simulate", "args": {"seeds": ["r::a.py"], "runs": 2000, "rng_seed": 5}}
    sim = h[0]
    p = sim["result"]["top"][0]["hit_prob"]
    if any(x["tool"] == "verifier" for x in h):
        return {"action": "answer", "summary": "ok", "claims": [{"text": f"b.py is hit {p:.0%}", "evidence": [sim["id"]], "numbers": {"p": p}}]}
    return {"action": "answer", "summary": "bad", "claims": [{"text": "b.py is hit 99%", "evidence": [sim["id"]], "numbers": {"p": 0.99}}]}

def test_every_question_gets_a_repair_attempt():
    mg = tiny(3)
    probs, trust = edge_probabilities(mg, {})
    w = World(mg, probs, trust, {})
    m = Fake(scripted_controller, "ctl")
    for _ in range(2):                                          # the old code only repaired the FIRST question
        r = ask("what breaks?", w, m, log=lambda *a: None)
        assert all(c["status"] != "rejected" for c in r["claims"]), r["claims"]

def test_simulate_without_seeds_uses_the_run_epicenter():
    mg = tiny(3)
    probs, trust = edge_probabilities(mg, {})
    w = World(mg, probs, trust, {})
    w.default_seeds = ["r::a.py"]
    assert "error" not in w.simulate([], 1000)["result"]

# ---- whole pipeline, offline ---------------------------------------------------------------
def test_pipeline_offline_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        graph, corpus = fixture.build(d)
        out = os.path.join(d, "out")
        root = os.path.join(os.path.dirname(__file__), "..")
        r = subprocess.run([sys.executable, os.path.join(root, "run_demo.py"), "--graph", graph, "--corpus", corpus, "--backend", "mock",
                            "--jury-n", "3", "--briefs", "2", "--lang", "hi", "--question", "Where should I contribute first?", "--out", out],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:] + r.stdout[-800:]
        for f in ("fault_lines.html", "briefs.md", "briefs.json", "results.json", "judgements.json", "weaknesses.json"):
            assert os.path.exists(os.path.join(out, f)), f
        briefs = json.load(open(os.path.join(out, "briefs.json")))
        assert len(briefs) == 2 and briefs[0]["other_projects"] and briefs[0]["local"]["lang"] == "hi"
        html = open(os.path.join(out, "fault_lines.html"), encoding="utf-8").read()
        assert "__DATA__" not in html and "Where to contribute first" in html

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
