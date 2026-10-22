---
name: fault-lines
description: Find where an open-source contribution would protect the most downstream projects, or simulate how a bug or attack in one project spreads into others. Use when the user asks where to contribute, what breaks, how likely, or what to fix first, across several repos mapped with Graphify.
---

# Fault Lines: Fix-First Map

Builds on a Graphify map (graph.json). Stitches broken links between repos, has a Gemma jury judge each cross-repo link from real code,
ranks every shared file by blast radius, has Gemma read the top files whole for weaknesses, writes contribution briefs, and answers
questions only from numbered evidence.

## Use
1. `bash build_corpus.sh` (or merge your own repos with `graphify merge-graphs ... --out merged.json`).
2. `python selftest.py` to check the model passes the format tests.
3. `python run_demo.py --graph merged.json --corpus ./corpus --question "Where should I contribute first?" [--lang hi|mr] [--jury-n 3] [--native-tools]`.
4. Open `out/fault_lines.html` and read `out/briefs.md`.

## Rules for the assistant
- Never invent numbers. Every number must come from a tool result (E1, E2, ...). Cite the evidence id.
- Say "under these assumptions". Chances for links are assumptions shown in the picture, not predictions.
- A claim that depends on a link the jury did not confirm is a hypothesis, not a finding.
- Weakness findings must quote real code. Never suggest a contribution for a file or line you cannot see in the evidence.
- Briefs are drafts for a human to verify before any pull request.
