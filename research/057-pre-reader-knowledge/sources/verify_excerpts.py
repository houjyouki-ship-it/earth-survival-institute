import json, os
H = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(H, "cases.json")))
def text(files):
    t = ""
    for f in files.split("+"):
        p = list(json.load(open(os.path.join(H, f)))["query"]["pages"].values())[0]; t += p.get("extract", "")
    return t
n = ok = 0; bad = []
for c in C["cases"]:
    t = text(c["file"])
    for e in c["excerpts"]:
        n += 1; ok += e in t
        if e not in t: bad.append((c["id"], e))
for c in C["context"]:
    n += 1; t = text(c["file"]); ok += c["excerpt"] in t
    if c["excerpt"] not in t: bad.append(("context", c["excerpt"]))
print(json.dumps({"n": n, "matched": ok, "missing": bad}, ensure_ascii=False))
