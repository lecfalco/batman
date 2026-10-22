"""Step 5: build the picture. Output is ONE self-contained web page (open it by double-clicking)."""
import json, math, os, random
import networkx as nx

PLATES = ["#4f7c82", "#b07d3b", "#7a5c8f", "#6b8f4e", "#a45a52", "#4a6a9a"]

def layout(mg):
    repos = sorted({m["repo"] for m in mg["modules"]}, key=lambda r: -sum(m["repo"] == r for m in mg["modules"]))
    size = {r: sum(m["repo"] == r for m in mg["modules"]) for r in repos}
    rad = {r: 0.10 + 0.05 * math.sqrt(size[r]) for r in repos}
    G = nx.Graph(); G.add_nodes_from(repos)
    for e in mg["edges"]:
        a, b = e["dependent"].split("::")[0], e["dependency"].split("::")[0]
        if a != b:
            w = G.get_edge_data(a, b, {"weight": 0})["weight"]
            G.add_edge(a, b, weight=w + 1)
    c = nx.spring_layout(G, seed=11, k=0.9, weight="weight") if len(repos) > 1 else {repos[0]: [0, 0]}
    cen = {r: [float(c[r][0]), float(c[r][1])] for r in repos}
    for _ in range(300):                                    # push plates apart so they never overlap
        for a in repos:
            for b in repos:
                if a < b:
                    dx, dy = cen[a][0] - cen[b][0], cen[a][1] - cen[b][1]
                    d = math.hypot(dx, dy) or 1e-3
                    need = (rad[a] + rad[b]) * 1.45
                    if d < need:
                        k = (need - d) / 2 / d
                        cen[a][0] += dx * k; cen[a][1] += dy * k; cen[b][0] -= dx * k; cen[b][1] -= dy * k
    pos = {}
    for r in repos:
        S = nx.Graph()
        ids = [m["id"] for m in mg["modules"] if m["repo"] == r]
        S.add_nodes_from(ids)
        S.add_edges_from((e["dependent"], e["dependency"]) for e in mg["edges"] if e["dependent"] in S and e["dependency"] in S)
        if len(ids) > 1:
            for i in ids:                                   # a faint pull to the middle keeps stray files from flying off
                S.add_edge("__hub__", i, weight=0.15)
            p = nx.spring_layout(S, seed=5, k=1.5 / math.sqrt(len(ids)), iterations=200)
            p = {i: p[i] for i in ids}
        else:
            p = {ids[0]: [0.0, 0.0]}
        cx_ = sum(v[0] for v in p.values()) / len(p); cy_ = sum(v[1] for v in p.values()) / len(p)
        d = sorted(math.hypot(v[0] - cx_, v[1] - cy_) for v in p.values())
        r90 = d[int(0.9 * (len(d) - 1))] or 1.0
        for i in ids:                                       # tame outliers so nobody crushes the cluster
            vx, vy = p[i][0] - cx_, p[i][1] - cy_
            rr = math.hypot(vx, vy) / r90
            f = 1.0 if rr <= 1.15 else (1.15 + 0.12 * (rr - 1.15)) / rr
            pos[i] = [cen[r][0] + vx / r90 * f * rad[r], cen[r][1] + vy / r90 * f * rad[r]]
    xs, ys = [v[0] for v in pos.values()], [v[1] for v in pos.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sc = max(x1 - x0, y1 - y0) or 1
    norm = {i: [(v[0] - x0) / sc, (v[1] - y0) / sc] for i, v in pos.items()}
    plates = [{"name": r, "color": PLATES[k % len(PLATES)], "n": size[r],
               "x": (cen[r][0] - x0) / sc, "y": (cen[r][1] - y0) / sc} for k, r in enumerate(repos)]
    return norm, plates, (x1 - x0) / sc, (y1 - y0) / sc

def build_data(mg, sim, result, trust, assessments, harden, sens, answers, model_name, assumptions, epicenter_name, edge_key, briefs=None, jury_info=None):
    pos, plates, w, h = layout(mg)
    ix = {m["id"]: i for i, m in enumerate(mg["modules"])}
    nodes = []
    for n in result["nodes"]:
        nodes.append({"id": n["id"], "label": n["file"].replace("src/", "").replace(".py", ""), "repo": n["repo"],
                      "x": round(pos[n["id"]][0], 4), "y": round(pos[n["id"]][1], 4), "hit": n["hit_prob"],
                      "weak": n["weakest_link"], "exp": n["exposure"], "loss": n["expected_loss"], "hops": n["hops"]})
    edges = []
    for i, e in enumerate(mg["edges"]):
        a = assessments.get(edge_key(e)) or {}
        edges.append({"a": ix[e["dependency"]], "b": ix[e["dependent"]], "p": round(float(sim.p[i]), 3), "cross": e["cross_repo"],
                      "basis": trust[i], "level": a.get("level"), "why": a.get("why"), "quote": a.get("evidence"),
                      "votes": a.get("votes"), "agree": a.get("agreement"), "disagree": bool(a.get("models_disagree"))})
    hm = result["hit_matrix"]
    rnd = random.Random(3)
    picks = rnd.sample(range(hm.shape[0]), 40)
    samples = [[int(j) for j in hm[r].nonzero()[0]] for r in picks]
    gap = sorted((n for n in nodes if n["id"] not in result["seeds"]), key=lambda n: -(n["hit"] - n["weak"]))
    g = gap[0] if gap else None
    if briefs:                                          # the headline is the contribution, not a method comparison
        b0 = briefs[0]
        op = b0["other_projects"]
        caption = (f"Fix first: {b0['file'].split('::')[-1]} ({b0['project']}). A failure there reaches about {b0['downstream_files']} other files"
                   + (f" and {len(op)} other project{'s' if len(op) != 1 else ''}." if op else "."))
    elif g and g["hit"] - g["weak"] >= 0.10:             # only show the weakest-link comparison when the gap is real
        caption = f"{g['label']}: the simple weakest-link rule says {g['weak']:.0%}; simulating every route together says {g['hit']:.0%}."
    else:
        caption = ""
    return {"epicenter_name": epicenter_name, "epicenter": [ix[s] for s in result["seeds"]], "plates": plates,
            "aspect": (w / h) if h else 1.0, "nodes": nodes, "edges": edges, "samples": samples, "caption": caption,
            "runs": result["runs"], "rng_seed": result["rng_seed"], "expected_loss": result["expected_loss_total"],
            "tail_loss": result["tail_loss_cvar"], "harden": harden[:5], "sens": sens, "answers": answers,
            "model": model_name, "offline": model_name.startswith("offline"), "assumptions": assumptions,
            "briefs": briefs or [], "jury": jury_info or {}, "judged": sum(1 for t in trust if t == "model-judged"), "cross": sum(1 for e in mg["edges"] if e["cross_repo"])}

def write_html(data, path):
    tpl = open(os.path.join(os.path.dirname(__file__), "viewer.html"), encoding="utf-8").read()
    blob = json.dumps(data).replace("</", "<\\/")
    open(path, "w", encoding="utf-8").write(tpl.replace("__DATA__", blob))
    return path
