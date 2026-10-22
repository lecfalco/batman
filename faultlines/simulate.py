"""Step 3: the shock simulator ("Monte Carlo": repeat a random experiment thousands of times
and count how often each thing happens).

A shock starts at one or more files (the epicenter) and travels along dependency links:
if file A depends on file B, a shock at B can reach A. Each link only passes the shock
on with some probability. We replay the quake many times with different dice rolls."""
import heapq
import numpy as np

class Sim:
    def __init__(self, mg, probs, dists=None):
        # dists: optional {edge index: (chances, weights)} from the Gemma jury. Where present, every run draws
        # that link's chance from the jury's votes instead of using one fixed number ("model doubt becomes risk").
        self.dists = dists or {}
        self.mods, self.E = mg["modules"], mg["edges"]
        self.ix = {m["id"]: i for i, m in enumerate(self.mods)}
        self.p = np.clip(np.array(probs, float), 0.0, 1.0)
        self.src = np.array([self.ix[e["dependency"]] for e in self.E])   # shock leaves from here
        self.dst = np.array([self.ix[e["dependent"]] for e in self.E])    # and arrives here
        # "exposure units": how much stuff lives in each file (classes + functions). Not dollars.
        self.w = np.array([max(1, m["symbols"]) for m in self.mods], float)

    def draw(self, runs, seed):
        rng = np.random.default_rng(seed)
        u = rng.random((runs, len(self.E)))                   # same dice as before, so old results do not change
        if not self.dists:
            return u < self.p                                 # True = this link passed the shock
        pm = np.broadcast_to(self.p, u.shape).copy()
        for i, (vals, w) in self.dists.items():
            pm[:, i] = rng.choice(vals, size=runs, p=w)       # this run's chance for a link the jury disagreed on
        return u < pm

    def propagate(self, seeds, alive, protected=()):
        R, V = alive.shape[0], len(self.mods)
        prot = np.zeros(V, bool)
        prot[list(protected)] = True
        hit = np.zeros((R, V), bool)
        hit[:, list(seeds)] = True
        order = np.zeros(V, int) - 1
        order[list(seeds)] = 0
        depth = 0
        while True:
            depth += 1
            new = hit.copy()
            for e in range(len(self.E)):
                d = self.dst[e]
                if prot[d]:
                    continue
                new[:, d] |= hit[:, self.src[e]] & alive[:, e]
            fresh = new.any(0) & ~hit.any(0)
            order[fresh & (order < 0)] = depth
            if (new == hit).all():
                return hit, order
            hit = new

    def weakest_link(self, seeds):
        """What a simple 'weakest link on the best single path' rule says (the old way)."""
        best = {s: 1.0 for s in seeds}
        heap = [(-1.0, s) for s in seeds]
        out_edges = {}
        for e in range(len(self.E)):
            out_edges.setdefault(self.src[e], []).append(e)
        while heap:
            b, n = heapq.heappop(heap)
            if -b < best.get(n, 0):
                continue
            for e in out_edges.get(n, []):
                nb = min(-b, self.p[e])
                if nb > best.get(self.dst[e], 0):
                    best[self.dst[e]] = nb
                    heapq.heappush(heap, (-nb, self.dst[e]))
        return best

    def best_path(self, seed, target):
        """Most likely single route from seed to target (product of link chances)."""
        best, prev, heap = {seed: 1.0}, {}, [(-1.0, seed)]
        out_edges = {}
        for e in range(len(self.E)):
            out_edges.setdefault(self.src[e], []).append(e)
        while heap:
            b, n = heapq.heappop(heap)
            if -b < best.get(n, 0):
                continue
            for e in out_edges.get(n, []):
                nb = -b * self.p[e]
                if nb > best.get(self.dst[e], 0):
                    best[self.dst[e]], prev[self.dst[e]] = nb, (n, e)
                    heapq.heappush(heap, (-nb, self.dst[e]))
        if target not in best:
            return None
        path, n = [], target
        while n != seed:
            pn, e = prev[n]
            path.append(e)
            n = pn
        return list(reversed(path)), best[target]

    def run(self, seeds, runs=10000, seed=42, tail=0.05, protected=(), alive=None):
        alive = self.draw(runs, seed) if alive is None else alive
        hit, order = self.propagate(seeds, alive, protected)
        total = hit @ self.w
        k = max(1, int(len(total) * tail))
        hp = hit.mean(0)
        wl = self.weakest_link(seeds)
        nodes = []
        for i, m in enumerate(self.mods):
            nodes.append({"id": m["id"], "repo": m["repo"], "label": m["label"], "file": m["file"],
                          "hit_prob": round(float(hp[i]), 4), "weakest_link": round(float(wl.get(i, 0.0)), 4),
                          "exposure": float(self.w[i]), "expected_loss": round(float(hp[i] * self.w[i]), 3),
                          "hops": int(order[i])})
        return {"seeds": [self.mods[s]["id"] for s in seeds], "runs": runs, "rng_seed": seed,
                "expected_loss_total": round(float(total.mean()), 2),
                "tail_loss_cvar": round(float(np.sort(total)[-k:].mean()), 2),  # average of the worst 5% of runs
                "nodes": nodes, "hit_matrix": hit, "alive": alive}

    def candidates(self, limit=60):
        """Files other projects depend on (or, if there are none, files anything depends on), busiest first."""
        cross, anyd = {}, {}
        for i, e in enumerate(self.E):
            b = int(self.src[i])
            anyd[b] = anyd.get(b, 0) + 1
            if e["cross_repo"]:
                cross[b] = cross.get(b, 0) + 1
        pool = cross or anyd
        return [k for k, _ in sorted(pool.items(), key=lambda kv: (-kv[1], -anyd[kv[0]]))][:limit]

    def blast_ranking(self, cands=None, runs=1500, seed=11):
        """If THIS file fails, how much lands downstream? Every candidate is replayed on the SAME dice, so ranks are fair.
        This is the core of the Fix-First Map: contribute where a failure would hurt the most others."""
        alive = self.draw(runs, seed)
        direct = {}
        for e in self.E:
            direct[e["dependency"]] = direct.get(e["dependency"], 0) + 1
        rows = []
        for c in (self.candidates() if cands is None else cands):
            hit, _ = self.propagate([c], alive)
            hp = hit.mean(0)
            hp[c] = 0.0
            repo = self.mods[c]["repo"]
            others = sorted({self.mods[j]["repo"] for j in np.nonzero(hp > 0.2)[0] if self.mods[j]["repo"] != repo})
            rows.append({"id": self.mods[c]["id"], "repo": repo, "label": self.mods[c]["label"],
                         "downstream_loss": round(float(hp @ self.w), 2), "downstream_files": round(float(hp.sum()), 2),
                         "other_projects": others, "direct_dependents": direct.get(self.mods[c]["id"], 0),
                         "runs": runs, "rng_seed": seed})
        return sorted(rows, key=lambda r: -r["downstream_loss"])

    def hardening(self, seeds, runs=3000, seed=42, top=25):
        """Protect one file at a time, replay the SAME dice rolls, see how much risk disappears."""
        alive = self.draw(runs, seed)
        base = self.run(seeds, runs, seed, alive=alive)
        cand = sorted((n for n in base["nodes"] if n["hit_prob"] > 0.02 and n["id"] not in base["seeds"]),
                      key=lambda n: -n["expected_loss"])[:top]
        rows = []
        for n in cand:
            r = self.run(seeds, runs, seed, protected=[self.ix[n["id"]]], alive=alive)
            rows.append({"id": n["id"], "label": n["label"], "repo": n["repo"],
                         "loss_before": base["expected_loss_total"], "loss_after": r["expected_loss_total"],
                         "loss_removed": round(base["expected_loss_total"] - r["expected_loss_total"], 2)})
        return sorted(rows, key=lambda r: -r["loss_removed"])

    def sensitivity(self, seeds, runs=2000, seed=7, wobble=0.10, trials=5, top=5):
        """Do the top-risk files stay the same if our assumptions are a bit off?"""
        base = self.run(seeds, runs, seed)
        key = lambda r: [n["id"] for n in sorted((x for x in r["nodes"] if x["id"] not in r["seeds"]),
                                                  key=lambda x: -x["expected_loss"])[:top]]
        b, rng, same = set(key(base)), np.random.default_rng(seed), []
        orig, orig_d = self.p.copy(), self.dists
        self.dists = {}                                       # wobble the single-number chances only
        for _ in range(trials):
            self.p = np.clip(orig + rng.uniform(-wobble, wobble, len(orig)), 0.02, 0.98)
            same.append(len(b & set(key(self.run(seeds, runs, seed)))) / top)
        self.p, self.dists = orig, orig_d
        return {"wobble": wobble, "trials": trials, "top_n": top, "avg_overlap": round(float(np.mean(same)), 2)}
