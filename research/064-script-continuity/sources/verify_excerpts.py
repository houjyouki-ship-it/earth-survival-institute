import json, os
H = os.path.dirname(os.path.abspath(__file__)); J = json.load(open(os.path.join(H, "cases.json"), encoding="utf8"))
n = bad = 0
for c in J["cases"] + J["other"]:
    for f, s in c["ex"]:
        n += 1; t = open(os.path.join(H, f), encoding="utf8").read()
        if s not in t: bad += 1; print("MISSING", c["id"], f, s[:60])
print(n, "excerpts", bad, "missing")
