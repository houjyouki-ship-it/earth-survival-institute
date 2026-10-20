"""Xia et al. 2022 補足資料 Table S4（Livestock）・S5（Partial Livestock、本文 Table 1 の数）・S6（No Livestock）、いずれも貿易なしを国ごとに読む。"""
import re, json
L = open("xia_moesm1.txt", encoding="utf8").read().split("\n")
def table(tag, nxt):
    s = [k for k, l in enumerate(L) if tag in l and "(millions)" in l][0]; e = [k for k, l in enumerate(L) if k > s and l.strip().startswith("Total")][0] + 1
    rows = {}
    for l in L[s:e]:
        tok = l.split()
        if len(tok) < 10 or not all(re.fullmatch(r"\d+(\.\d+)?", t) for t in tok[-9:]): continue
        name = " ".join(tok[:-9])
        if name in ("Total",): continue
        if name.startswith("Taiwan") or name == "": name = "Taiwan"
        rows[name] = [float(t) for t in tok[-9:]]
    return rows
cols = ["pop2010", "5Tg", "16Tg", "27Tg", "37Tg", "47Tg", "150Tg", "150Tg_hw", "150Tg_tw"]
out = {"source": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs43016-022-00573-0/MediaObjects/43016_2022_573_MOESM1_ESM.pdf",
       "columns": cols, "unit": "millions; deaths from starvation in Year 2, no trade",
       "S4_livestock": table("Table S4.", None), "S5_partial_livestock": table("Table S5.", None), "S6_no_livestock": table("Table S6.", None)}
for k in ("S4_livestock", "S5_partial_livestock", "S6_no_livestock"):
    t = out[k]; print(k, len(t), round(sum(v[0] for v in t.values()), 1), round(sum(v[6] for v in t.values()), 1), round(sum(v[5] for v in t.values()), 1))
json.dump(out, open("xia_tables.json", "w"), indent=0, ensure_ascii=False)
