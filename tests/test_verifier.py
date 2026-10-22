"""Run with:  python -m pytest -q   (or:  python tests/test_verifier.py)
These prove the verifier rejects invented numbers, missing evidence and made-up files."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from faultlines.harness import World, verify
from faultlines.assess import edge_probabilities, assess_edges, _norm

def tiny_world():
    mods = [{"id": f"r::{n}.py", "repo": "r", "file": f"{n}.py", "symbols": 2, "label": f"{n}.py"} for n in "abc"]
    edges = [{"dependent": "r::b.py", "dependency": "r::a.py", "cross_repo": False, "relations": {"calls": 1},
              "evidence_lines": [], "prior": 0.5, "stitch": None, "stitch_confidence": 1.0, "provenance": "extracted"},
             {"dependent": "r::c.py", "dependency": "r::b.py", "cross_repo": False, "relations": {"calls": 1},
              "evidence_lines": [], "prior": 0.5, "stitch": None, "stitch_confidence": 1.0, "provenance": "extracted"}]
    mg = {"modules": mods, "edges": edges}
    probs, trust = edge_probabilities(mg, {})
    return World(mg, probs, trust, {})

def test_good_claim_passes():
    w = tiny_world(); rec = w.simulate(["r::a.py"], 2000)
    p = rec["result"]["top"][0]["hit_prob"]
    out = verify({"claims": [{"text": f"b.py is hit {p:.0%} of the time", "evidence": [rec["id"]], "numbers": {"p": p}}]}, w)
    assert out[0]["status"] in ("checked", "hypothesis"), out

def test_invented_number_rejected():
    w = tiny_world(); rec = w.simulate(["r::a.py"], 2000)
    out = verify({"claims": [{"text": "b.py is hit 99% of the time", "evidence": [rec["id"]], "numbers": {"p": 0.99}}]}, w)
    assert out[0]["status"] == "rejected"

def test_no_evidence_rejected():
    w = tiny_world()
    assert verify({"claims": [{"text": "c.py is risky", "evidence": [], "numbers": {}}]}, w)[0]["status"] == "rejected"

def test_unknown_file_rejected():
    w = tiny_world(); rec = w.simulate(["r::a.py"], 2000)
    assert verify({"claims": [{"text": "ghost.py is at risk", "evidence": [rec["id"]], "numbers": {}}]}, w)[0]["status"] == "rejected"

def test_same_seed_same_result():
    a, b = tiny_world(), tiny_world()
    assert a.simulate(["r::a.py"], 3000, 5)["result"]["top"] == b.simulate(["r::a.py"], 3000, 5)["result"]["top"]

def test_quote_guard():
    assert _norm("from  x   import y") in _norm("a\nfrom x import y\nb")
    assert _norm("from z import q") not in _norm("a\nfrom x import y\nb")

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
