import json, os
H = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(H, "cases.json")))
ok = bad = 0
for c in D["cases"] + D["model_refs"]:
    t = open(os.path.join(H, c["src"]), encoding="utf8").read()
    for e in c["ex"]:
        if e in t: ok += 1
        else: bad += 1; print("NOT FOUND", c["id"], e[:80])
print("ok", ok, "bad", bad)
