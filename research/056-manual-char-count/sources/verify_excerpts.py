"""Check that each quoted excerpt appears verbatim in the saved MediaWiki API extract. Run: python3 verify_excerpts.py"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
ex = json.load(open(os.path.join(HERE, "excerpts.json")))
ok = 0
for e in ex:
    d = json.load(open(os.path.join(HERE, e["file"])))
    t = list(d["query"]["pages"].values())[0]["extract"]
    hit = e["quote"] in t; ok += hit
    print(hit, e["url"], e["quote"][:70])
print(ok, "/", len(ex))
