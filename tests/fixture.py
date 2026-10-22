"""Builds a tiny SYNTHETIC corpus and a Graphify-shaped merged.json so the whole pipeline can be tested
with no network, no Graphify and no model. The projects are made up (app, libnet, libparse); it is test data only."""
import json, os

FILES = {
 "libnet": {
  "src/libnet/core.py": "import socket\n\nclass Connection:\n    def open(self, host):\n        try:\n            return socket.create_connection((host, 80))\n        except:\n            return None  # TODO: timeouts\n\ndef fetch(url):\n    return Connection().open(url)\n",
  "src/libnet/errors.py": "class NetError(Exception):\n    pass\n\nclass Timeout(NetError):\n    pass\n",
  "src/libnet/pool.py": "from libnet.core import Connection\n\nclass Pool:\n    def get(self, host):\n        return Connection().open(host)\n",
 },
 "libparse": {
  "src/libparse/reader.py": "def parse(text):\n    return eval(text)  # FIXME: unsafe\n\ndef read_all(lines):\n    return [parse(l) for l in lines]\n",
  "src/libparse/util.py": "def clean(s):\n    return s.strip()\n",
 },
 "app": {
  "src/app/client.py": "from libnet.pool import Pool\nfrom libnet.errors import NetError\n\nclass Client:\n    def __init__(self):\n        self.pool = Pool()\n\n    def get(self, host):\n        try:\n            return self.pool.get(host)\n        except NetError:\n            return None\n",
  "src/app/loader.py": "from libparse.reader import parse\nfrom libnet.core import fetch\n\ndef load(url):\n    return parse(fetch(url))\n",
  "src/app/cli.py": "from app.client import Client\n\ndef main():\n    print(Client().get('example.org'))\n",
 },
}
# (dependent file, line, stub id, relation)
IMPORTS = [("app/src/app/client.py", 1, "libnet_pool", "imports_from"), ("app/src/app/client.py", 2, "libnet_errors", "imports_from"),
           ("app/src/app/loader.py", 1, "libparse_reader", "imports_from"), ("app/src/app/loader.py", 2, "libnet_core", "imports_from"),
           ("app/src/app/loader.py", 4, "libparse_reader", "calls"), ("app/src/app/loader.py", 4, "libnet_core", "calls")]

def build(root):
    corpus = os.path.join(root, "corpus")
    nodes, links, ids = [], [], {}
    for repo, files in FILES.items():
        for rel, text in files.items():
            p = os.path.join(corpus, repo, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(text)
            for k in range(3):                                   # a few code symbols per file
                nid = f"{repo}_{rel.split('/')[-1][:-3]}_{k}"
                nodes.append({"id": nid, "repo": repo, "source_file": rel, "file_type": "code", "label": nid})
                ids.setdefault((repo, rel), nid)
    stubs = sorted({s for _, _, s, _ in IMPORTS} | {"libnet_core"})
    nodes += [{"id": s, "file_type": "concept", "label": s} for s in stubs]
    for fil, ln, stub, rel in IMPORTS:
        repo, path = fil.split("/", 1)
        links.append({"source": ids[(repo, path)], "target": stub, "relation": rel, "confidence": "EXTRACTED",
                      "source_file": path, "source_location": f"L{ln}"})
    # one in-repo link so the in-project map is not empty
    links.append({"source": ids[("libnet", "src/libnet/pool.py")], "target": ids[("libnet", "src/libnet/core.py")],
                  "relation": "imports_from", "confidence": "EXTRACTED", "source_file": "src/libnet/pool.py", "source_location": "L1"})
    links.append({"source": ids[("app", "src/app/cli.py")], "target": ids[("app", "src/app/client.py")],
                  "relation": "imports_from", "confidence": "EXTRACTED", "source_file": "src/app/cli.py", "source_location": "L1"})
    graph = os.path.join(root, "merged.json")
    json.dump({"nodes": nodes, "links": links}, open(graph, "w"))
    return graph, corpus
