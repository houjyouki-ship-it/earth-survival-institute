"""cases.json の引用文が保存した本文（wiki_texts.json）にあるかを確かめる（空白を詰めて照合）。"""
import json, re
sq = lambda s: re.sub(r"\s+", " ", s)
C = json.load(open("cases.json", encoding="utf8")); T = json.load(open("wiki_texts.json", encoding="utf8"))
txt = {k: sq(v["query"]["pages"][0].get("extract", "")) for k, v in T.items()}
res = []
for c in C["cases"]:
    for q in c["quotes"]: res.append((c["id"], q[:50], sq(q) in txt[c["page"]]))
res.append(("kentoshi", C["kentoshi_loss_quote"][:50], sq(C["kentoshi_loss_quote"]) in txt["Japanese_missions_to_Tang_China"]))
bad = [r for r in res if not r[2]]
print(len(res), "excerpts checked; all found:", not bad); [print("MISSING", b) for b in bad]
