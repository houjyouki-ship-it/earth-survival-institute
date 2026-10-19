"""Wikipedia「Lost city」の大陸別の節で、見出し Rediscovered / Undiscovered / Uncertain or disputed / Status Unknown の下の行を数える。
伝説・架空の節（Undiscovered and fictional lost cities）は数えない。"""
import re, json, os
H = os.path.dirname(os.path.abspath(__file__))
t = open(os.path.join(H, "wp_Lost_city.txt"), encoding="utf8").read().split("\n")
cat = None; C = {}; start = False
for l in t:
    if l.startswith("== Lost cities by continent"): start = True; continue
    if l.startswith("== Undiscovered and fictional"): break
    if not start: continue
    m = re.match(r"^(=+)\s*(.*?)\s*=+$", l)
    if m:
        h = m.group(2); cat = h if h in ("Rediscovered", "Undiscovered", "Uncertain or disputed", "Status Unknown") else None; continue
    if l.strip() and cat: C.setdefault(cat, []).append(re.split(r" – | — | - ", l)[0][:60])
out = {k: {"n": len(v), "names": v} for k, v in C.items()}
json.dump(out, open(os.path.join(H, "lost_city_counts.json"), "w"), ensure_ascii=False, indent=1)
print({k: v["n"] for k, v in out.items()})
