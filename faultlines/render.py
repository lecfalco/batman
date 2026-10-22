import json
import math
import os
import random
import re
import networkx as nx

PLATES = ["#4f7c82", "#b07d3b", "#7a5c8f", "#6b8f4e", "#a45a52", "#4a6a9a"]

def layout(mg):
    repos = sorted({m["repo"] for m in mg["modules"]}, key=lambda r: -sum(m["repo"] == r for m in mg["modules"]))
    size = {r: sum(m["repo"] == r for m in mg["modules"]) for r in repos}
    rad = {r: 0.10 + 0.05 * math.sqrt(size[r]) for r in repos}
    G = nx.Graph()
    G.add_nodes_from(repos)
    for e in mg["edges"]:
        a, b = e["dependent"].split("::")[0], e["dependency"].split("::")[0]
        if a != b:
            w = G.get_edge_data(a, b, {"weight": 0})["weight"]
            G.add_edge(a, b, weight=w + 1)
    c = nx.spring_layout(G, seed=11, k=0.9, weight="weight") if len(repos) > 1 else {repos[0]: [0, 0]}
    cen = {r: [float(c[r][0]), float(c[r][1])] for r in repos}
    for _ in range(300):
        for a in repos:
            for b in repos:
                if a < b:
                    dx, dy = cen[a][0] - cen[b][0], cen[a][1] - cen[b][1]
                    d = math.hypot(dx, dy) or 1e-3
                    need = (rad[a] + rad[b]) * 1.45
                    if d < need:
                        k = (need - d) / 2 / d
                        cen[a][0] += dx * k
                        cen[a][1] += dy * k
                        cen[b][0] -= dx * k
                        cen[b][1] -= dy * k
    pos = {}
    for r in repos:
        S = nx.Graph()
        ids = [m["id"] for m in mg["modules"] if m["repo"] == r]
        S.add_nodes_from(ids)
        S.add_edges_from((e["dependent"], e["dependency"]) for e in mg["edges"] if e["dependent"] in S and e["dependency"] in S)
        if len(ids) > 1:
            for i in ids:
                S.add_edge("__hub__", i, weight=0.15)
            p = nx.spring_layout(S, seed=5, k=1.5 / math.sqrt(len(ids)), iterations=200)
            p = {i: p[i] for i in ids}
        else:
            p = {ids[0]: [0.0, 0.0]}
        cx_ = sum(v[0] for v in p.values()) / len(p)
        cy_ = sum(v[1] for v in p.values()) / len(p)
        d = sorted(math.hypot(v[0] - cx_, v[1] - cy_) for v in p.values())
        r90 = d[int(0.9 * (len(d) - 1))] or 1.0
        for i in ids:
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
    ix = {m["id"]: i + 1 for i, m in enumerate(mg["modules"])}
    seeds = set(result["seeds"])
    nodes = []
    for n in result["nodes"]:
        is_seed = n["id"] in seeds
        idx = ix[n["id"]]
        lbl = n["file"].replace("src/", "").replace(".py", "")
        if is_seed:
            lbl += "\n(Epicenter)"
        val = 35 if is_seed else max(10, min(35, int(n["hit_prob"] * 30 + 10)))
        color = {'background': '#ff7b72', 'border': '#ff7b72'} if is_seed else (
                {'background': '#58a6ff', 'border': '#58a6ff'} if n["hit_prob"] > 0 else
                {'background': '#8b949e', 'border': '#8b949e'})
        node_obj = {
            "id": idx,
            "label": lbl,
            "group": n["repo"],
            "value": val,
            "color": color,
            "font": {"color": "#fff", "multi": True if is_seed else False},
            "x": round(pos[n["id"]][0], 4),
            "y": round(pos[n["id"]][1], 4),
            "hit": n["hit_prob"],
            "weak": n["weakest_link"],
            "exp": n["exposure"],
            "loss": n["expected_loss"],
            "hops": n["hops"]
        }
        if is_seed:
            node_obj["shadow"] = {"enabled": True, "color": "rgba(255,123,114,0.6)", "size": 20}
        nodes.append(node_obj)
        
    edges = []
    for i, e in enumerate(mg["edges"]):
        a = assessments.get(edge_key(e)) or {}
        p_val = round(float(sim.p[i]), 3)
        u = ix.get(e["dependency"])
        v = ix.get(e["dependent"])
        if u and v:
            is_judged = trust[i] == "model-judged"
            edges.append({
                "from": u,
                "to": v,
                "arrows": "to",
                "color": {"color": "#ff7b72" if p_val >= 0.7 else "#58a6ff", "opacity": 0.8 if is_judged else 0.4},
                "width": 3 if p_val >= 0.7 else 2,
                "dashes": not is_judged,
                "title": f"Probability: {p_val} ({'Gemma Jury' if is_judged else 'Hypothesis'})",
                "a": u - 1, "b": v - 1, "p": p_val, "cross": e["cross_repo"],
                "basis": trust[i], "level": a.get("level"), "why": a.get("why"), "quote": a.get("evidence"),
                "votes": a.get("votes"), "agree": a.get("agreement"), "disagree": bool(a.get("models_disagree"))
            })
            
    hm = result["hit_matrix"]
    rnd = random.Random(3)
    picks = rnd.sample(range(hm.shape[0]), min(40, hm.shape[0]))
    samples = [[int(j) for j in hm[r].nonzero()[0]] for r in picks]
    gap = sorted((n for n in nodes if (n["id"] - 1) not in [ix[s] - 1 for s in result["seeds"]]), key=lambda n: -(n["hit"] - n["weak"]))
    g = gap[0] if gap else None
    if briefs:
        b0 = briefs[0]
        op = b0.get("other_projects", [])
        caption = (f"Fix first: {b0['file'].split('::')[-1]} ({b0['project']}). A failure there reaches about {b0['downstream_files']} other files"
                   + (f" and {len(op)} other project{'s' if len(op) != 1 else ''}." if op else "."))
    elif g and g["hit"] - g["weak"] >= 0.10:
        caption = f"{g['label']}: the simple weakest-link rule says {g['weak']:.0%}; simulating every route together says {g['hit']:.0%}."
    else:
        caption = ""
        
    return {
        "epicenter_name": epicenter_name,
        "epicenter": [ix[s] for s in result["seeds"]],
        "plates": plates,
        "aspect": (w / h) if h else 1.0,
        "nodes": nodes,
        "edges": edges,
        "samples": samples,
        "caption": caption,
        "runs": result["runs"],
        "rng_seed": result["rng_seed"],
        "expected_loss": result["expected_loss_total"],
        "tail_loss": result["tail_loss_cvar"],
        "harden": harden[:5],
        "sens": sens,
        "answers": answers,
        "model": model_name,
        "offline": model_name.startswith("offline"),
        "assumptions": assumptions,
        "briefs": briefs or [],
        "jury": jury_info or {},
        "judged": sum(1 for t in trust if t == "model-judged"),
        "cross": sum(1 for e in mg["edges"] if e["cross_repo"])
    }

def render_map(graph_data, output_path="out/fault_lines.html"):
    '''
    Injects graph data into the modern UI viewer.html template.
    '''
    template_path = os.path.join(os.path.dirname(__file__), "viewer.html")
    
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            html = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find {template_path}. Ensure it is in the same directory.")
        return
        
    json_data = json.dumps(graph_data).replace("</", "<\\/")
    
    # Replace placeholder in template, handling both raw placeholder and placeholder with fallback mock
    if "/*DATA_PLACEHOLDER*/" in html:
        html = re.sub(r'/\*DATA_PLACEHOLDER\*/\s*\{.*?\};', f'{json_data};', html, flags=re.DOTALL)
        html = html.replace("/*DATA_PLACEHOLDER*/", json_data)
    if "__DATA__" in html:
        html = html.replace("__DATA__", json_data)
        
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the rendered HTML
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"Success: Rendered polished interactive map to {output_path}")
    return output_path

def write_html(data, path):
    return render_map(data, path)

if __name__ == "__main__":
    print("Run this script via the main pipeline (harness.py or run_demo.py).")
