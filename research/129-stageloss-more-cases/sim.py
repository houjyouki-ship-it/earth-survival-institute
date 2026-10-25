"""研究129：段を失った崩壊の事例を増やして p|lost を詰める（open_problems 303＋297）。
計算のみ・ネット接続なし（資料は sources/ と 125 の sources/ に保存済み）。事前登録：PREREGISTRATION.md（2026-10-25 23:44 JST）。
注：F_heavy・V は生存確率や復興確率ではない（模型の中の価値の比）。"""
import json, os, math
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
R125 = json.load(open(os.path.join(RES, "125-pop-recovery-history", "results.json"), encoding="utf8"))
R128 = json.load(open(os.path.join(RES, "128-stag-given-stageloss", "results.json"), encoding="utf8"))
GR = json.load(open(os.path.join(RES, "128-stag-given-stageloss", "grid.json")))
r = R125["inputs"]["r_median_038"]
# ---- 再現：128 の M の数えと Jeffreys 事後の F_heavy ----
PS = np.array(GR["PS"]); SL = "slow_Tasmania"
FC = np.array([GR["GRID"][f"{p:.6f}|{SL}|D2rz|none"]["F_heavy"] for p in PS])
NR = np.array([GR["GRID"][f"{p:.6f}|{SL}|D2rz|none"]["new_heavy"] for p in PS]) / GR["V0_new_heavy"] - 1
def ix(x, xs, ys):
    y = np.interp(x, xs, ys); lo = x < xs[0]
    y[lo] = ys[0] + (x[lo] - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0]); return y
rng = np.random.default_rng(20261025)
def post(k, n, a0=0.5, b0=0.5, N=200000):
    d = rng.beta(a0 + k, b0 + n - k, size=N); f = ix(d, PS, FC); nv = ix(d, PS, NR)
    return {"k": k, "n": n, "p_mean": float(d.mean()), "F_mean": float(f.mean()), "F_q10_50_90": [float(np.quantile(f, q)) for q in (.1, .5, .9)],
            "new_heavy_rel_V0_mean": float(nv.mean())}
cM = R128["counts"]["M"]; assert (cM["lost_stag"], cM["n_lost"], cM["kept_stag"], cM["n_kept"]) == (3, 3, 0, 2)
rep = post(3, 3); ref = R128["posterior"]["M|jeffreys"]
assert abs(rep["F_mean"] - ref["F|slow|D2rz"]["mean"]) < 2e-3 and abs(rep["p_mean"] - ref["p_mean"]) < 2e-3, (rep, ref["F|slow|D2rz"]["mean"])
CHECK = {"F_mean_rep": rep["F_mean"], "F_mean_128": ref["F|slow|D2rz"]["mean"], "note": "128 と同じ乱数の種・同じ引き数で差 0（assert は <2e-3）"}
# ---- 新しい事例（本回開いた資料の数値のみ） ----
S = "sources/"; S125 = "125-pop-recovery-history/sources/"
NEW = {
 "Copan": dict(cls_items=dict(writing="ceased", metal=None, masonry="ceased", coin=None), extinct=False,
   P0=20000, P0_range=[18000, 25000], Pmin_upper=5000, t_start=800, t_start_alt=900, t_still_low=1550, t_rec_upper=2023, P_modern=45312, immig="不明",
   ev=[S + "wiki_Copan：'estimated total population of 18,000 to 25,000 people in the valley during the Late Classic'；'population declined in the 8th and 9th centuries from perhaps over 20,000 in the city to less than 5,000'；'abandoned in the early 10th century'；'last known king, Ukit Took', came to the throne on 6 February 822'；'only a few farming hamlets at the time of the arrival of the Spanish in the 16th century'；'Postclassic ... villagers who stole the stone from the monumental architecture ... to build their simple house platforms'",
       S125 + "wiki_Copan_Ruinas：population_total 45312（2023 projection、citypopulation.de。自治体の面積と谷の範囲の一致は未確認）"],
   causes=dict(trade="'had long needed to import basic necessities from outside'（依存はあるが途絶の記述なし）", disease="'Shortage and disease afflicted the massively overpopulated valley'", land="'deforestation of the Copán valley'；'overpopulation and a lack of local resources'", immig="記述なし")),
 "Cahokia": dict(extinct=True, P0=(10200 + 15300) / 2, P0_range=[6000, 40000], t_start=1200, t_abandon=1400, immig="あり（後の住民は子孫とは限らない）",
   ev=[S + "wiki_Cahokia：'between AD 1050 and 1100, Cahokia's population increased ... to between 10,200 and 15,300'；'estimate ... between 6,000 and 40,000 at its peak'；'The city is abandoned by 1400 CE with brief Oneota reoccupation'；'the Cahokia tribe was not necessarily descended from the earlier Mississippian-era people'；'By the end of the 12th century ... decline'"],
   causes=dict(trade="記述なし（交易網の存在のみ）", disease="'Diseases ... are another possible cause'（可能性）", land="'major floods ... 1100–1260 and 1340–1460'；'no evidence of human-caused erosion'；'droughts from CE 1100–1250'", immig="'brief Oneota reoccupation'")),
 "Italy_postRoman": dict(cls_items=dict(writing="?", metal="?", masonry="?", coin="?"), extinct=False,
   P0=8.5e6, P0_range=[8e6, 13e6], Pmin=4e6, t_start=165, t_min=800, t_rec_upper=1300, P_rec=11.5e6, immig="不明（本回の頁で文を確認せず）",
   ev=[S125 + "wiki_Demography_of_the_Roman_Empire：'Population of Italy and the islands in 165 CE' 標準の解釈 8–9（百万）、改訂の解釈 12–13",
       S125 + "wiki_Medieval_demography：Charlemagne の頃 'estimates of 4 million for Italy'；'Italy's population around 1300 ... between 10 and 13 million'",
       "途絶の根拠の文は本回探していない（規則では不明＝途絶に数えない）"],
   causes=dict(trade="'limited trade and political, social and economic dislocation'（欧州全体の記述）", disease="記述は本回なし", land="記述なし", immig="記述は本回なし")),
}
NOT_OBTAINED = {
 "Indus": "wiki_Indus_Valley_Civilisation：成熟期 1–5 百万の推定のみ。後期の 'most of the post-urban Late Harappan settlements were abandoned' に人口の数値なし",
 "Hittite_Anatolia": "wiki_Hittites：王国 14–15 万（Simon）または 20 万超（Bryce）。崩壊後の地域の人口の数値なし。なお wiki_Neo-Hittite_states に 'Luwian monumental inscriptions ... continue almost uninterrupted'（地域を北シリアまで取ると文字は途絶えず、予想の lost は崩れうる）",
 "Angkor": "wiki_Angkor：'between 750,000 and 1,000,000'、1431 の後 'population to migrate south'、'sparsely populated' のみで後の数値なし",
}
def lost(c):
    if c.get("extinct"): return True
    return sum(1 for v in c["cls_items"].values() if v == "ceased") >= 2
def judge(key, c, Pmin=None, t_start=None, P0=None):
    """125 の規則：pred = ln(P0/Pmin)/r。区間は上限で判定。消滅・放棄は停滞。"""
    if c.get("extinct"): return {"stag": True, "rule": "放棄", "length_years": None}
    P0 = P0 or c["P0"]; ts = t_start or c["t_start"]
    Pm = Pmin or c.get("Pmin") or c["Pmin_upper"]
    pred = math.log(P0 / Pm) / r
    T_up = c["t_rec_upper"] - ts
    T_lo = (c.get("t_still_low") or c.get("t_min")) - ts
    return {"stag": (T_up - pred) > 0, "pred": pred, "T_upper": T_up, "T_lower": T_lo, "excess_upper": T_up - pred, "excess_lower": T_lo - pred, "rule": "区間（上限で判定）"}
OUT = {"study": "129-stageloss-more-cases", "note": "F_heavy・V は生存確率や復興確率ではない", "r_median_038": r, "check_128": CHECK, "cases": {}, "not_obtained": NOT_OBTAINED}
for k, c in NEW.items():
    j = judge(k, c); OUT["cases"][k] = {**{a: b for a, b in c.items()}, "lost": lost(c), **j}
# Copán の感度：Pmin と t_start・P0
cp = NEW["Copan"]; T_up = cp["t_rec_upper"] - cp["t_start"]
OUT["copan_sens"] = {
 "Pmin_breakeven_upper": cp["P0"] / math.exp(r * T_up),  # これより Pmin が小さいと上限でも超過<=0（判定は停滞なし側）
 "Pmin_breakeven_lower": cp["P0"] / math.exp(r * (cp["t_still_low"] - cp["t_start"])),
 "grid": {f"Pmin={pm}|ts={ts}|P0={p0}": judge("Copan", cp, Pmin=pm, t_start=ts, P0=p0)["excess_upper"] for pm in (5000, 2000, 1000, 500) for ts in (800, 900) for p0 in (18000, 20000, 25000)}}
# Italy の感度：改訂の解釈 12.5M
it = NEW["Italy_postRoman"]
OUT["italy_sens"] = {f"P0={p0}": judge("Italy_postRoman", it, P0=p0) for p0 in (8e6, 8.5e6, 9e6, 12.5e6)}
# ---- 数え ----
new_lost = [k for k in NEW if OUT["cases"][k]["lost"]]; new_kept = [k for k in NEW if not OUT["cases"][k]["lost"]]
SETS = {
 "M128": (3, 3, 0, 2),
 "M128+Copan": (3 + OUT["cases"]["Copan"]["stag"], 4, 0, 2),
 "all_new(M128+Copan+Cahokia, kept+Italy)": (3 + sum(OUT["cases"][k]["stag"] for k in new_lost), 3 + len(new_lost), sum(OUT["cases"][k]["stag"] for k in new_kept), 2 + len(new_kept)),
 "X128+new": (R128["counts"]["X"]["lost_stag"] + sum(OUT["cases"][k]["stag"] for k in new_lost), R128["counts"]["X"]["n_lost"] + len(new_lost),
              R128["counts"]["X"]["kept_stag"] + sum(OUT["cases"][k]["stag"] for k in new_kept), R128["counts"]["X"]["n_kept"] + len(new_kept)),
 "Copan_no(Pmin<=breakeven)": (3, 4, 0, 2),
}
OUT["counts"] = {}; OUT["posterior"] = {}
for s, (ls, nl, ks, nk) in SETS.items():
    OUT["counts"][s] = {"lost_stag": int(ls), "n_lost": nl, "kept_stag": int(ks), "n_kept": nk}
    OUT["posterior"][s + "|jeffreys"] = post(int(ls), nl); OUT["posterior"][s + "|uniform"] = post(int(ls), nl, 1, 1)
    tab = [[int(ls), nl - int(ls)], [int(ks), nk - int(ks)]]
    OUT["counts"][s]["fisher_one_sided"] = float(stats.fisher_exact(tab, alternative="greater")[1])
# ---- 仮説 ----
cA = OUT["counts"]["all_new(M128+Copan+Cahokia, kept+Italy)"]
cpj = OUT["cases"]["Copan"]
H = {
 "H141a": {"class": "confirmatory", "value": f"{cA['lost_stag']}/{cA['n_lost']}", "supported": cA["lost_stag"] / cA["n_lost"] >= 0.75},
 "H141b": {"class": "confirmatory", "value": {"Copan_excess_lower_upper": [cpj["excess_lower"], cpj["excess_upper"]], "Cahokia": "放棄の長さは数値なし"},
           "supported": None if (cpj["excess_lower"] < 300 <= cpj["excess_upper"]) else cpj["excess_lower"] >= 300, "note": "Copán だけが測れ、下限と上限が 300 年をまたぐので判定保留"},
 "H141c": {"class": "confirmatory", "value": f"kept の停滞 {cA['kept_stag'] - 0}/{cA['n_kept'] - 2}（新しい kept）", "supported": (cA["kept_stag"] == 0)},
 "H141d": {"class": "prediction", "value": OUT["posterior"]["M128+Copan|jeffreys"]["F_mean"], "supported": 2.08 <= OUT["posterior"]["M128+Copan|jeffreys"]["F_mean"] <= 2.29},
 "H141e": {"class": "prediction(descriptive)", "value": f"判定できた {len(NEW)}/6", "supported": len(NEW) <= 2},
}
OUT["hypotheses"] = H
pred = {"Copan": True, "Cahokia": True, "Italy_postRoman": False}
OUT["prereg_class_mismatch"] = [k for k in NEW if pred[k] != OUT["cases"][k]["lost"]]
OUT["prereg_stag_mismatch"] = [k for k, v in {"Copan": True, "Cahokia": True, "Italy_postRoman": False}.items() if v != OUT["cases"][k]["stag"]]
# ---- 図 ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
xx = np.linspace(0, 1, 400)
for s, ls_ in (("M128", ":"), ("M128+Copan", "-"), ("all_new(M128+Copan+Cahokia, kept+Italy)", "--")):
    c = OUT["counts"][s]; ax[0].plot(xx, stats.beta.pdf(xx, .5 + c["lost_stag"], .5 + c["n_lost"] - c["lost_stag"]), ls_, label=f"{s[:22]} lost {c['lost_stag']}/{c['n_lost']}  F={OUT['posterior'][s+'|jeffreys']['F_mean']:.3f}")
ax[0].set_ylim(0, 12); ax[0].set_xlabel("p(stagnation | stage lost)"); ax[0].legend(fontsize=7); ax[0].set_title("129: posterior of p|lost (Jeffreys); F_heavy not a survival prob.", fontsize=8)
pms = np.linspace(300, 6000, 200)
for ts, p0, st in ((800, 20000, "-"), (900, 20000, "--"), (800, 25000, ":")):
    ax[1].plot(pms, [(2023 - ts) - math.log(p0 / pm) / r for pm in pms], st, label=f"upper excess t_start={ts} P0={p0}")
    ax[1].plot(pms, [(1550 - ts) - math.log(p0 / pm) / r for pm in pms], st, alpha=.4, label=f"lower excess t_start={ts} P0={p0}")
ax[1].axhline(0, c="k", lw=.8); ax[1].axhline(300, c="gray", ls=":"); ax[1].set_xlabel("Copán trough population Pmin (source: < 5,000)"); ax[1].set_ylabel("excess vs 038 median (years)")
ax[1].legend(fontsize=6); ax[1].set_title("Copán stagnation judgment depends on Pmin", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_plost_copan.png"), dpi=120)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("check", CHECK)
for k, v in OUT["cases"].items(): print(k, "lost", v["lost"], "stag", v["stag"], {a: round(b, 1) for a, b in v.items() if a in ("pred", "T_upper", "T_lower", "excess_upper", "excess_lower")})
print("copan breakeven", {a: round(b) for a, b in OUT["copan_sens"].items() if a != "grid"}); print({a: round(b) for a, b in OUT["copan_sens"]["grid"].items()})
print("italy", {a: (round(b["excess_lower"]), round(b["excess_upper"])) for a, b in OUT["italy_sens"].items()})
for s, c in OUT["counts"].items(): print(s, c, {pr: (round(OUT["posterior"][s + "|" + pr]["p_mean"], 3), round(OUT["posterior"][s + "|" + pr]["F_mean"], 3), [round(x, 3) for x in OUT["posterior"][s + "|" + pr]["F_q10_50_90"]], round(OUT["posterior"][s + "|" + pr]["new_heavy_rel_V0_mean"], 4)) for pr in ("jeffreys", "uniform")})
for h, v in H.items(): print(h, v["class"], v["supported"], v["value"])
print("mismatch class", OUT["prereg_class_mismatch"], "stag", OUT["prereg_stag_mismatch"])
