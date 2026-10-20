import json, os
H = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(H, "cases.json"), encoding="utf8"))
ok = bad = 0
for c in D["script_loss"] + D["search"] + D["stack"] + D["notes"]:
    for s, ex in ((c["src"], c["ex"]), (c.get("src2"), c.get("ex2", []))):
        if not s: continue
        t = open(os.path.join(H, s), encoding="utf8").read()
        for e in ex:
            if e in t: ok += 1
            else: bad += 1; print("NOT FOUND", c["id"], e[:90])
print("ok", ok, "bad", bad)
