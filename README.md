# Fault Lines: Fix-First Map

**Which open-source file should you fix first, so that the most other projects stay safe?**

Fault Lines maps several connected open-source projects (with [Graphify](https://github.com/safishamsi/graphify)), simulates a failure in every shared file, and ranks files by **blast radius**: how many other files and projects a failure there would reach. Gemma 4 then reads the top files whole, finds real weaknesses (each one must quote real code), and writes a **ready-to-use contribution brief**: where, why, how to test, and a draft pull request. Optionally the explanation comes in Hindi or Marathi.

It draws the result as a "shake map": projects are plates, colour is the chance a file is hit, and the cracks between plates are the links where damage crosses from one project into another.

## What Gemma 4 does here (and what plain code does)
| Piece | Who | What |
|---|---|---|
| Map of files and links | Graphify | Code, imports, calls, which links were read directly |
| Link judge: **the Gemma jury** | Gemma (several votes, optionally 2 model sizes) | Each cross-project link is judged several times from real code. Every vote must quote code that really exists. The **spread of votes becomes the probability**, and the spread is fed into the simulation, so model doubt shows up as risk. This replaces guessed numbers with a measurement |
| Weakness judge | Gemma, reads the **whole file** (long context) | Finds fragile code; quotes must exist, line numbers are computed by code |
| Brief writer | Gemma | Why it matters, how to test, draft PR title and body, optional Hindi/Marathi. A checker rejects percentages and invented files; if Gemma fails twice a plain template is used and labelled |
| Advisory reader | Gemma (image input) | Reads a security advisory or a screenshot of one and picks the epicenter |
| Controller | Gemma, with **native tool calling** (`--native-tools`) or JSON | Picks tools to answer questions; cannot do maths or read files itself |
| Blast radius, Monte Carlo, ranking | Plain code | Replays every shared file as the epicenter on the same dice rolls; 10,000-run simulation for the chosen one |
| Verifier | Plain code | Every number must match tool output, every file must exist, else the claim is rejected |

## Quick start
```bash
pip install graphifyy numpy networkx
bash build_corpus.sh                      # 5 small projects -> corpus/merged.json (no AI model needed)
ollama pull gemma4:e4b                    # check the exact name at ollama.com/library/gemma4
python selftest.py                        # a few minutes: does your Gemma pass all the format checks?
python run_demo.py --graph corpus/merged.json --corpus corpus \
  --question "Where should I contribute first?" --lang hi --out out
```
Open `out/fault_lines.html` and `out/briefs.md`. No `--epicenter` needed: the file with the biggest blast radius is used (or pass `--epicenter urllib3`, a file path, or `--advisory-image shot.png`).

Using more of Gemma (all optional):
```bash
# two model sizes in the jury: small local + big hosted (set GEMINI_API_KEY and check the model name in Google AI Studio)
python run_demo.py ... --jury-n 5 --big-backend gemini --big-model <gemma-model-name> --big-n 3
python run_demo.py ... --think            # thinking mode (Ollama; selftest.py --think checks the server accepts it)
python run_demo.py ... --native-tools      # Ollama native tool calling for the controller
python run_demo.py ... --jury-n 1         # fast mode: one vote per link (old behaviour)
```
Time: the jury makes (links x votes) model calls and results are cached in `out/judgements.json`, so reruns are instant. Start with `--jury-n 3`.

Test with no model and no network (uses made-up projects and a rule-based stand-in that is **not Gemma**):
```bash
python tests/test_verifier.py && python tests/test_new.py
```

## How to read the numbers (please read this)
- Chances are **assumptions, not predictions**. Link priors are in `modgraph.py` (`RELATION_PRIOR`). Each Gemma vote is worth 0.2 / 0.5 / 0.8 (`jury.py`, `LEVEL_P`). The jury's *spread* is measured, but what one vote is worth is still our choice. Change and rerun.
- A link is only "model-judged" if at least half of the asked votes were valid **and** quoted real code; otherwise the default assumption is used and any claim using it is a **hypothesis**.
- "Exposure units" count functions and classes in a file. They are a rough size, not money.
- Blast radius = average number of other files hit (and how many other projects reached) when that file fails, with the same dice rolls for every candidate. It ranks files; it does not predict incidents.
- The picture also shows how stable the ranking is when every chance is shifted by up to 0.10.
- Claim labels: **checked** = numbers match the simulation; **hypothesis** = relies on links the jury did not confirm; **rejected** = removed.
- Links are assumed to pass shocks independently of each other.

## What changed from the first version
- Added: blast-radius ranking, the Gemma jury (+ vote spread in the simulation), whole-file weakness judge, contribution briefs (+ Hindi/Marathi), native tool calling, thinking-mode switch, a new "Where to contribute first" panel.
- Fixed: a bug where only the first question in a run could get a repair attempt after the verifier rejected a claim; evidence IDs no longer skip a number.
- Changed: link judging now shows up to 4 code locations (was 2) and tells Gemma how many places use the link. The picture's headline is now the fix-first result; the weakest-link comparison is shown only when the gap is real (at least 10 points).

## Known limits
- Not yet run against a real Gemma by the author of this edit: the new Gemma features (jury, whole-file reading, native tool calling, thinking mode, translation quality) are **unverified until you run `selftest.py` on your model**. The Gemini default model name (`gemma-4-31b-it`) is a guess; set `FL_MODEL`.
- The large Gemma models may not fit in 8 GB of graphics memory; check before relying on them locally.
- Tested here on Python projects only. File-level, not function-level.
- Graphify's cross-repo handling changes between versions. Newer releases may already link some cross-project calls, so re-check how many "stitched" links you really need on your installed version, and check Graphify's open issue about same-named nodes being merged.
- Gemma's judgement is noisy. The quote check, the jury and the ranking-stability check reduce, not remove, that risk.
- Briefs are drafts. Read the code and the project's contribution guide before opening a pull request.
- `docs/sample_picture_offline_standin.png` (older UI) and `docs/sample_new_ui_synthetic_data_mock.png` (made-up projects) were both made with the offline stand-in, not Gemma.

MIT licensed.
