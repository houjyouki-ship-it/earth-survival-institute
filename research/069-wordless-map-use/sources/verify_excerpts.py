"""cases.json の抜き出しが保存本文にあるかを照合する。抄録は改行で折り返されているので、両方の空白を1つに詰めてから比べる。"""
import json, os
H = os.path.dirname(os.path.abspath(__file__)); J = json.load(open(os.path.join(H, "cases.json"), encoding="utf8"))
norm = lambda s: " ".join(s.split())
n = bad = 0
for grp in ("map", "landmark", "picto", "hist", "other"):
    for c in J[grp]:
        for f, s in c["ex"]:
            n += 1; t = norm(open(os.path.join(H, f), encoding="utf8").read())
            if norm(s) not in t: bad += 1; print("MISSING", grp, c["id"], f, s[:70])
print(n, "excerpts", bad, "missing")
