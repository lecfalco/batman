"""The Gemma jury: ask Gemma the SAME question several times (and with more than one model size),
and let the votes, not a guess, become the probability.

Why: the old code mapped Gemma's answer low/medium/high to fixed numbers 0.2/0.5/0.8 (an assumption).
A jury turns the spread of votes into a measured number. If seven votes say "low" and one says "medium",
the link is probably weak AND we know how sure the models are. The vote spread is also fed into the
Monte Carlo (each simulated run draws that link's chance from the votes), so model doubt shows up as risk.

How a vote counts: it must be valid JSON, have a level, AND quote text that really appears in the code.
Votes with an invented quote are ignored. If fewer than half of the asked votes survive, the link is not
trusted and falls back to the default assumption (and any claim using it is labelled a hypothesis)."""
import collections, math, re

LEVEL_P = {"low": 0.2, "medium": 0.5, "high": 0.8}   # ASSUMPTION: what one vote is worth. Change and re-run.

def _norm(s):
    return re.sub(r"\s+", " ", s or "").strip()

class Jury:
    def __init__(self, members, temperature=0.7):
        """members: list of (Gemma, n_votes). The first vote of each model is greedy (temperature 0),
        the rest are sampled at `temperature` with different seeds."""
        self.members = [(m, int(n)) for m, n in members if int(n) > 0]
        self.temperature = temperature
        if len(self.members) == 1 and self.members[0][1] == 1:
            self.name = self.members[0][0].name
        else:
            self.name = "jury[" + " + ".join(f"{m.name} x{n}" for m, n in self.members) + "]"

    @property
    def calls(self):
        return sum(m.calls for m, _ in self.members)

    @property
    def failures(self):
        return sum(m.failures for m, _ in self.members)

    def judge(self, system, user, code):
        votes, asked = [], 0
        for m, n in self.members:
            for i in range(n):
                asked += 1
                temp = 0.0 if (n == 1 or i == 0) else self.temperature
                out = m.json(system, user, temperature=temp, seed=7 + 13 * i)
                if not out or str(out.get("level", "")).lower() not in LEVEL_P:
                    continue
                ev = str(out.get("evidence", ""))
                votes.append({"level": str(out["level"]).lower(), "why": str(out.get("why", ""))[:240],
                              "evidence": ev[:160], "verified": bool(ev) and _norm(ev) in _norm(code),
                              "model": m.name})
        return aggregate(votes, asked, self.name)

def aggregate(votes, asked, source):
    """Turn a list of votes into one assessment. Pure function, so it can be tested without a model."""
    if not votes:
        return None
    ver = [v for v in votes if v["verified"]]
    need = max(1, math.ceil(asked / 2))
    pool = ver or votes
    counts = collections.Counter(v["level"] for v in pool)
    p = sum(LEVEL_P[l] * c for l, c in counts.items()) / len(pool)
    modal = max(counts, key=lambda l: (counts[l], -abs(LEVEL_P[l] - p)))
    per_model = {}
    for v in pool:
        per_model.setdefault(v["model"], collections.Counter())[v["level"]] += 1
    modal_by_model = {k: c.most_common(1)[0][0] for k, c in per_model.items()}
    best = next(v for v in pool if v["level"] == modal)
    dist = {}
    for l, c in counts.items():
        dist[str(LEVEL_P[l])] = dist.get(str(LEVEL_P[l]), 0) + c / len(pool)
    return {"level": modal, "p": round(p, 3), "dist": dist, "votes": dict(counts),
            "n_asked": asked, "n_valid": len(votes), "n_verified": len(ver),
            "agreement": round(counts[modal] / len(pool), 2),
            "models_disagree": len(set(modal_by_model.values())) > 1,
            "per_model": {k: dict(c) for k, c in per_model.items()},
            "why": best["why"], "evidence": best["evidence"],
            "verified": len(ver) >= need, "source": source}
