"""研究089 第1部：短い尺度の β。日本 1940→1947（今村 1948、1947 年の境域に換算した 1940 年の人口）と
ドイツ 1939→1950（de.wikipedia の大都市の表）。log P1 = c + β log P0、95% は 2,000 回の bootstrap（seed 20261089）。
TC/V0 ではなく都市人口の弾力性の記述。"""
import csv, json, os, re
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(20261089)
def fit(p0, p1, B=2000):
    x, y = np.log(p0), np.log(p1); b = np.polyfit(x, y, 1)[0]; bs = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0: bs.append(np.polyfit(x[i], y[i], 1)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5]); return {"beta": float(b), "ci95": [float(lo), float(hi)], "n": int(len(x))}
JP = list(csv.DictReader(open(os.path.join(HERE, "japan_1940_1947.csv"), encoding="utf8")))
for r in JP:
    r["P0"] = int(r["P1940"]); r["P1"] = int(r["P1947"]); r["ok"] = abs(round((r["P1"] / r["P0"] - 1) * 1000) - int(r["rate_permil"])) <= 1
def jp(sel): s = [r for r in JP if sel(r)]; return fit(np.array([r["P0"] for r in s], float), np.array([r["P1"] for r in s], float))
E = {}
E["JP_all"] = jp(lambda r: True)
E["JP_rate_consistent"] = jp(lambda r: r["ok"])
E["JP_P1940_ge_50k"] = jp(lambda r: r["P0"] >= 50000)
E["JP_P1940_ge_30k"] = jp(lambda r: r["P0"] >= 30000)
E["JP_excl_Tokyo_ku"] = jp(lambda r: r["name"] != "東京区部")
E["JP_excl_Kanto_Tokai"] = jp(lambda r: r["pref"] not in ("東京", "神奈川", "愛知", "千葉", "埼玉"))
# 都市と郡部（県の総数 − 掲載の市）
PF = {r["pref"]: (int(r["P1947"]), int(r["P1940"])) for r in csv.DictReader(open(os.path.join(HERE, "prefs.csv"), encoding="utf8"))}
c1 = sum(r["P1"] for r in JP); c0 = sum(r["P0"] for r in JP); t1 = sum(v[0] for v in PF.values()); t0 = sum(v[1] for v in PF.values())
big = [r for r in JP if r["P0"] >= 500000]
E["JP_aggregate"] = {"cities_ratio": c1 / c0, "rural_ratio": (t1 - c1) / (t0 - c0), "total_ratio": t1 / t0, "big500k_ratio": sum(r["P1"] for r in big) / sum(r["P0"] for r in big),
                     "big500k": [r["name"] for r in big], "n_rate_inconsistent": sum(not r["ok"] for r in JP), "inconsistent": [r["name"] for r in JP if not r["ok"]]}
# ドイツ
t = pd.read_html(os.path.join(HERE, "sources", "de_gross.html"))
def num(v):
    s = re.sub(r"\[.*?\]", "", str(v)); s = re.sub(r"[^\d]", "", s); return float(s) if s else np.nan
rows = []
a = t[0]; a.columns = [c[1] if isinstance(c, tuple) else c for c in a.columns]
for _, r in a.iterrows(): rows.append((str(r["Name"]), num(r["1939[3]"]), num(r["1950"])))
b = t[1]
for _, r in b.iterrows(): rows.append((str(r["Name"]), num(r["Ew.  1939"]), num(r["Ew.  1950"])))
DE = [(n, p0, p1) for n, p0, p1 in rows if p0 > 0 and p1 > 0 and n != "nan"]
E["DE_1939_1950_all"] = fit(np.array([d[1] for d in DE]), np.array([d[2] for d in DE]))
d2 = [d for d in DE if d[1] >= 50000]; E["DE_1939_1950_P1939_ge_50k"] = fit(np.array([d[1] for d in d2]), np.array([d[2] for d in d2]))
E["DE_n_rows"] = len(DE); E["DE_sample"] = DE[:5]
json.dump({"episodes": E, "japan_rows": [{k: r[k] for k in ("name", "pref", "P0", "P1", "ok")} for r in JP], "germany_rows": DE},
          open(os.path.join(HERE, "episodes.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(json.dumps(E, ensure_ascii=False, indent=1))
