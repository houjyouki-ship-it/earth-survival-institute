"""小地域の人口重みの重心と、重心どうしの大圏距離（km）を作る（計算のみ）。
国の座標：Google の countries.csv（developers.google.com/public-data/docs/canonical/countries_csv、国の中心の緯度経度）。
国→小地域：070 の make_cases.py の region()（ISO/ALIAS）をそのまま使う。重み：Xia 2022 S5 の 2010年人口。重心は単位球の上の重み付き平均【推測：国の中の人口の偏りは無視】。"""
import json, re, math, os
import numpy as np
src = open(os.path.join("..", "..", "070-group-deposit-overlap", "sources", "make_cases.py"), encoding="utf8").read()
G = {}; exec(src[: src.index("# 1) Xia")], G)
region = G["region"]; ISO = G["ISO"]; ALIAS = G["ALIAS"]
s = open("google_countries.html", encoding="utf8").read()
rows = re.findall(r"<tr>\s*<td>([A-Z]{2})</td>\s*<td>([-\d.]+)</td>\s*<td>([-\d.]+)</td>\s*<td>(.*?)</td>", s, re.S)
XY = {a: (float(la), float(lo)) for a, la, lo, _ in rows}
X = json.load(open("xia_tables.json", encoding="utf8")); S5 = X["S5_partial_livestock"]
acc = {}; miss = []
for name, v in S5.items():
    n = ALIAS.get(name, name)
    if n is None or n not in ISO: miss.append(name); continue
    a2 = ISO[n]["alpha-2"]; r = region(name)
    if a2 not in XY or r is None: miss.append(name); continue
    la, lo = map(math.radians, XY[a2]); w = v[0]
    acc.setdefault(r, np.zeros(4)); acc[r] += [w * math.cos(la) * math.cos(lo), w * math.cos(la) * math.sin(lo), w * math.sin(la), w]
# 070 の小地域で Xia の国がないもの（Micronesia など）は ISO 表のその小地域の国の座標の等重みの平均【推測】
REG70 = json.load(open(os.path.join("..", "..", "070-group-deposit-overlap", "sources", "cases.json"), encoding="utf8"))["regions"]
fallback = []
for r in REG70:
    if r in acc: continue
    fallback.append(r); acc[r] = np.zeros(4)
    for n, row in ISO.items():
        if (row["intermediate-region"] or row["sub-region"]) == r and row["alpha-2"] in XY:
            la, lo = map(math.radians, XY[row["alpha-2"]]); acc[r] += [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la), 1]
acc.pop("", None)
CEN = {}
for r, (x, y, z, w) in acc.items():
    CEN[r] = (math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x)))
def hav(a, b):
    (la1, lo1), (la2, lo2) = a, b; p1, p2 = math.radians(la1), math.radians(la2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))
REG = sorted(CEN); D = [[round(hav(CEN[a], CEN[b]), 1) for b in REG] for a in REG]
json.dump({"centroids": CEN, "regions": REG, "dist_km": D, "unmatched": miss, "fallback_unweighted": fallback, "n_country_coords": len(XY)}, open("centroids.json", "w"), ensure_ascii=False, indent=1)
print(len(XY), "coords;", len(miss), "unmatched:", miss)
for r in REG: print(f"{r:28s} {CEN[r][0]:7.2f} {CEN[r][1]:8.2f}")
i = REG.index; print("EAs-SAs", D[i("Eastern Asia")][i("Southern Asia")], "SEu-WAs", D[i("Southern Europe")][i("Western Asia")], "EAs-SEAs", D[i("Eastern Asia")][i("South-eastern Asia")], "NEu-SEu", D[i("Northern Europe")][i("Southern Europe")])
