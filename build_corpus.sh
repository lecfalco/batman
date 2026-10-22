#!/usr/bin/env bash
# Downloads 5 small, related open-source Python projects and builds one Graphify map of all of them.
# Needs: git, and `pip install graphifyy`.  No AI model is needed for this step.
set -e
mkdir -p corpus && cd corpus
for r in psf/requests urllib3/urllib3 kjd/idna certifi/python-certifi jawah/charset_normalizer; do
  d=$(basename "$r"); [ -d "$d" ] || git clone --depth 1 -q "https://github.com/$r" "$d"
  (cd "$d" && graphify update . >/dev/null)           # builds $d/graphify-out/graph.json from the code
done
graphify merge-graphs requests/graphify-out/graph.json urllib3/graphify-out/graph.json idna/graphify-out/graph.json \
  python-certifi/graphify-out/graph.json charset_normalizer/graphify-out/graph.json --out merged.json
echo "Done: corpus/merged.json"
