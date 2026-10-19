# 研究053：避難所と金庫の板の費用の比（open_problems 122）。計算のみ、ネット接続なし。
import json, time, os
import numpy as np
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261053; N = 200_000
rng = np.random.default_rng(SEED)
V = json.load(open(os.path.join(HERE, "..", "040-cohesion-vs-knowledge", "results.json")))["weighted"]
CPI = {1962: 30.3, 1991: 136.2, 2008: 215.3, 2013: 233.0, 2018: 251.1, 2019: 255.7, 2025: 321.9}  # 米 CPI-U 年平均（Minneapolis Fed の表）
def to25(x, y): return x * CPI[2025] / CPI[y]
def logu(a, b, n): return np.exp(rng.uniform(np.log(a), np.log(b), n))

# --- 避難所の1人あたり建設費のアンカー（2025年ドル）
gbp = rng.uniform(1.2, 1.6, N)   # £→$【推測】
anchors = {
 "A1_Ullmann1962_shelter_6mo": (to25(2125, 1962) + to25(2525, 1962)) / 2 * np.ones(N),
 "A2_SurvivalCondo_2018": to25(50e6 / 120, 2018) * np.ones(N),
 "A3_SouthPole_2008": to25(150e6 / 150, 2008) * np.ones(N),
 "A4_HalleyVI_2013": to25(26e6 / 52, 2013) * gbp,
 "A5_Biosphere2_1991": to25(150e6 / 8, 1991) * np.ones(N),
}
AN = list(anchors)
pick = rng.integers(0, len(AN), N)
spread = logu(0.5, 2.0, N)
cap_place = np.choose(pick, [anchors[a] for a in AN]) * spread
om_rate = logu(0.01, 0.05, N); om_years = rng.uniform(20, 100, N)
cap_place_om = cap_place * (1 + om_rate * om_years)
# 備蓄だけの版 D（Ullmann：食料と保管 $2,835/20人/182日）
food_py = to25(2835 / 20 / 182 * 365, 1962)
stock_place = food_py * rng.uniform(1, 3, N)

# --- TAB の費用
chars = logu(1e7, 1e8, N); prod = logu(1e5, 1e7, N); wage = logu(1e5, 2e5, N)
author = chars / prod * wage
eur = rng.uniform(1.0, 1.2, N); per_tab = logu(5e4, 5e6, N)
media = np.ceil(chars / per_tab) * 195 * eur * 3
C_TAB = author + media
C_TAB_withC = C_TAB + logu(1e4, 1e6, N)
author_share = author / C_TAB
# GROW の内容費用（TAB と同じ分布で独立）
chars_g = logu(1e7, 1e8, N); C_GROW = chars_g / logu(1e5, 1e7, N) * logu(1e5, 2e5, N) + np.ceil(chars_g / logu(5e4, 5e6, N)) * 195 * rng.uniform(1.0, 1.2, N) * 3

def q(x, ps=(0.05, 0.5, 0.95)): return [float(np.quantile(x, p)) for p in ps]
R = {"study": "053-refuge-vs-tablet-cost", "seed": SEED, "N": N,
     "anchors_per_place_2025usd": {a: q(anchors[a]) for a in AN},
     "food_per_person_year_2025usd": food_py,
     "C_TAB_q": q(C_TAB), "C_TAB_withC_q": q(C_TAB_withC), "author_share_q": q(author_share),
     "media_cost_q": q(media), "author_cost_q": q(author)}

WTS = [(a, u) for a in ("A_Ord", "A_XPT_S", "A_XPT_E") for u in ("U1", "U2", "U3")]
NS = [("REF_1e4", 1e4), ("REF_1e5", 1e5), ("REF_1e6", 1e6)]
res = {}
for a, u in WTS:
    w = V[a][u]; tab = w["TAB"]["total"]; key = f"{a}/{u}"; res[key] = {}
    for iv, n in NS:
        rv = w[iv]["total"] / tab
        heavy = w[iv]["from_heavy"] / w[iv]["total"] if w[iv]["total"] > 0 else None
        out = {"value_ratio": rv, "heavy_share": heavy}
        if rv > 0:
            ce = rv * C_TAB / (cap_place_om * n)
            out["CE_q"] = q(ce); out["P_CE_gt_1"] = float((ce > 1).mean())
            out["CE_median_by_anchor_capital_only"] = {AN[i]: float(np.median(rv * C_TAB[pick == i] / (cap_place[pick == i] * n))) for i in range(len(AN))}
            out["CE_withC_q"] = q(rv * C_TAB_withC / (cap_place_om * n))
            out["CE_stock_only_q"] = q(rv * C_TAB / (stock_place * n))
            out["P_CE_stock_only_gt_1"] = float(((rv * C_TAB / (stock_place * n)) > 1).mean())
            out["c_star_per_place_q"] = q(rv * C_TAB / n)
            out["log10_gap_median"] = float(np.log10(np.median(cap_place_om * n) / np.median(rv * C_TAB)))
        else:
            out.update({"CE_q": [0, 0, 0], "P_CE_gt_1": 0.0})
        res[key][iv] = out
    g_hi = w["GROW"]["total"] / tab; g_lo = 0.13
    lo, hi = sorted([g_lo, g_hi]); gv = logu(lo, hi, N) if hi > lo else np.full(N, hi)
    ce_g = gv * C_TAB / C_GROW
    res[key]["GROW"] = {"value_ratio_040": g_hi, "value_ratio_041": g_lo, "CE_q": q(ce_g), "P_CE_gt_1": float((ce_g > 1).mean())}
R["results"] = res

# 規模の目安：避難所の総費用（建設＋維持）
R["refuge_total_cost_q"] = {iv: q(cap_place_om * n) for iv, n in NS}
R["refuge_capital_cost_q"] = {iv: q(cap_place * n) for iv, n in NS}

M1, M2 = "A_Ord/U3", "A_XPT_S/U2"
H = {}
H["H65a"] = {"value": {iv: res[M1][iv]["CE_median_by_anchor_capital_only"] for iv in ("REF_1e5", "REF_1e6")},
             "rule": "A_Ord/U3: CE median < 1e-3 for every anchor, REF_1e5 and REF_1e6",
             "supported": all(v < 1e-3 for iv in ("REF_1e5", "REF_1e6") for v in res[M1][iv]["CE_median_by_anchor_capital_only"].values())}
pv = {f"{m}/{iv}": res[m][iv]["P_CE_gt_1"] for m in (M1, M2) for iv, _ in NS}
H["H65b"] = {"value": pv, "rule": "P(CE>1) < 0.01 for N=1e4..1e6 and both main weights", "supported": all(v < 0.01 for v in pv.values())}
H["H65c"] = {"value": res[M1]["REF_1e6"]["c_star_per_place_q"][1], "rule": "A_Ord/U3 REF_1e6 median c* < $100", "supported": res[M1]["REF_1e6"]["c_star_per_place_q"][1] < 100}
dv = {iv: res[M1][iv]["CE_stock_only_q"][1] for iv in ("REF_1e5", "REF_1e6")}
H["H65d"] = {"value": dv, "rule": "stock-only D: A_Ord/U3 CE median < 0.01", "supported": all(v < 0.01 for v in dv.values())}
H["H65e"] = {"value": R["author_share_q"][1], "rule": "median authoring share of C_TAB > 0.5", "supported": R["author_share_q"][1] > 0.5}
hv = {m: res[m]["REF_1e6"]["heavy_share"] for m in (M1, M2)}
H["H65f"] = {"value": hv, "rule": "REF_1e6 heavy share >= 0.8 for both main weights", "supported": all(v >= 0.8 for v in hv.values())}
gv_ = {m: res[m]["GROW"]["P_CE_gt_1"] for m in (M1, M2)}
H["H65g"] = {"value": gv_, "rule": "P(CE_GROW>1) < 0.5 for both main weights", "supported": all(v < 0.5 for v in gv_.values())}
R["hypotheses"] = H

# --- 図
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4.8))
labs = []; data = []
for iv, n in NS:
    for m in (M1, M2):
        if res[m][iv]["value_ratio"] > 0:
            data.append(np.log10(res[m][iv]["value_ratio"] * C_TAB / (cap_place_om * n))); labs.append(f"{iv}\n{m}")
data.append(np.log10(np.maximum((logu(0.13, max(0.131, V['A_Ord']['U3']['GROW']['total'] / V['A_Ord']['U3']['TAB']['total']), N)) * C_TAB / C_GROW, 1e-12))); labs.append("GROW\nA_Ord/U3")
ax.boxplot(data, whis=(5, 95), showfliers=False); ax.set_xticks(range(1, len(labs) + 1)); ax.set_xticklabels(labs, fontsize=7)
ax.axhline(0, color="r", ls="--", lw=0.8); ax.set_ylabel("log10 (value per $ relative to TAB)")
ax.set_title("Cost-effectiveness vs tablet primer (box 25-75%, whiskers 5-95%)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_ce_vs_tab.png"), dpi=130)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for i, a in enumerate(AN):
    ax.scatter([i], [np.median(anchors[a])], color="k")
ax.set_xticks(range(len(AN))); ax.set_xticklabels([a.split("_", 1)[1] for a in AN], fontsize=7, rotation=15)
ax.set_yscale("log"); ax.set_ylabel("capital cost per place (2025 USD)")
cs = res[M1]["REF_1e6"]["c_star_per_place_q"]; cs5 = res[M1]["REF_1e5"]["c_star_per_place_q"]
ax.axhspan(cs[0], cs[2], color="tab:blue", alpha=0.25, label="break-even c* REF_1e6 (A_Ord/U3, 5-95%)")
ax.axhspan(cs5[0], cs5[2], color="tab:orange", alpha=0.25, label="break-even c* REF_1e5 (A_Ord/U3, 5-95%)")
ax.axhline(food_py, color="g", ls=":", label="food stock per person-year (Ullmann 1962, 2025$)")
ax.legend(fontsize=7, loc="center right"); ax.set_title("Refuge cost anchors vs break-even cost per place", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_anchor_vs_breakeven.png"), dpi=130)

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: R[k] for k in ("anchors_per_place_2025usd", "C_TAB_q", "author_share_q", "media_cost_q", "refuge_capital_cost_q", "refuge_total_cost_q")}, indent=1))
for m in (M1, M2):
    for k, v in res[m].items(): print(m, k, json.dumps(v)[:600])
for k, v in H.items(): print(k, v["supported"], json.dumps(v["value"])[:300])
print("runtime", round(R["runtime_seconds"], 1))
