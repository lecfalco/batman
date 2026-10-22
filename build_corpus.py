import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
CORPUS.mkdir(exist_ok=True)

GRAPHIFY = ROOT / "venv" / "Scripts" / "graphify.exe"
if not GRAPHIFY.exists():
    GRAPHIFY = Path("graphify")

REPOS = [
    ("requests", "https://github.com/psf/requests"),
    ("urllib3", "https://github.com/urllib3/urllib3"),
    ("idna", "https://github.com/kjd/idna"),
    ("python-certifi", "https://github.com/certifi/python-certifi"),
    ("charset_normalizer", "https://github.com/jawah/charset_normalizer"),
]

def run(cmd, cwd=None):
    print(f"Running: {' '.join(str(c) for c in cmd)}")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    res = subprocess.run(cmd, cwd=cwd, check=True, env=env)
    return res

for name, url in REPOS:
    repo_dir = CORPUS / name
    if not repo_dir.exists():
        print(f"Cloning {name}...")
        run(["git", "clone", "--depth", "1", "-q", url, str(repo_dir)])
    else:
        print(f"{name} already cloned.")

    graph_file = repo_dir / "graphify-out" / "graph.json"
    if not graph_file.exists():
        print(f"Graphifying {name}...")
        run([str(GRAPHIFY), "update", "."], cwd=str(repo_dir))
    else:
        print(f"Graph for {name} already exists.")

merged_file = CORPUS / "merged.json"
graph_files = [
    str(CORPUS / "requests" / "graphify-out" / "graph.json"),
    str(CORPUS / "urllib3" / "graphify-out" / "graph.json"),
    str(CORPUS / "idna" / "graphify-out" / "graph.json"),
    str(CORPUS / "python-certifi" / "graphify-out" / "graph.json"),
    str(CORPUS / "charset_normalizer" / "graphify-out" / "graph.json"),
]

print("Merging graphs into corpus/merged.json...")
merge_cmd = [str(GRAPHIFY), "merge-graphs"] + graph_files + ["--out", str(merged_file)]
run(merge_cmd, cwd=str(CORPUS))

print("Done! Generated:", merged_file)
