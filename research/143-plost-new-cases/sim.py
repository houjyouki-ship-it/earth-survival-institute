"""研究143：段を失った崩壊の事例を増やし p|lost を詰め（open_problems 303）、142 の媒体の表に入れる（329）。
計算のみ・ネット接続なし（資料は sources/ に保存済み）。事前登録：PREREGISTRATION.md（2026-10-27 03:34 JST）。
注：F_heavy・V・P_read は生存確率や復興確率ではない（模型の中の量）。"""
import json, os, math, runpy, shutil, time
import numpy as np
from scipy import stats, integrate
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
R125 = json.load(open(f"{RES}/125-pop-recovery-history/results.json", encoding="utf8"))
R129 = json.load(open(f"{RES}/129-stageloss-more-cases/results.json", encoding="utf8"))
GR = json.load(open(f"{RES}/128-stag-given-stageloss/grid.json"))
r = R125["inputs"]["r_median_038"]
PS = np.array(GR["PS"]); SL = "slow_Tasmania"
FC = np.array([GR["GRID"][f"{p:.6f}|{SL}|D2rz|none"]["F_heavy"] for p in PS])
def Fx(x):
    x = np.atleast_1d(x); y = np.interp(x, PS, FC); lo = x < PS[0]
    y[lo] = FC[0] + (x[lo] - PS[0]) * (FC[1] - FC[0]) / (PS[1] - PS[0]); return y
def post(k, n, a0=0.5, b0=0.5):
    """Beta 事後の平均と、F_heavy の事後平均（数値積分、乱数なし）。"""
    a, b = a0 + k, b0 + n - k; d = stats.beta(a, b)
    Fm = integrate.quad(lambda x: Fx(x)[0] * d.pdf(x), 0, 1, limit=200)[0]
    return {"k": k, "n": n, "p_mean": a / (a + b), "p_q10_90": [float(d.ppf(.1)), float(d.ppf(.9))], "F_mean": Fm}
# ---- 再現：129 の事後（MC、20 万本）と数値積分の差 ----
CHECK = {}
for s in ("M128+Copan", "all_new(M128+Copan+Cahokia, kept+Italy)"):
    c = R129["counts"][s]; q = post(c["lost_stag"], c["n_lost"]); ref = R129["posterior"][s + "|jeffreys"]
    CHECK[s] = {"p_diff": abs(q["p_mean"] - ref["p_mean"]), "F_diff": abs(q["F_mean"] - ref["F_mean"])}
    assert CHECK[s]["p_diff"] < 2e-3 and CHECK[s]["F_diff"] < 2e-3, CHECK
# ---- 本回の事例（sources/ の頁の数値だけ） ----
S = "sources/"
COPAN = dict(P0=28000, Pmin_upper=1000, t_start=800, t_trough=1200, t_still_low=1550, t_rec_upper=2023,
  ev=[S + "wiki_Classic_Maya_collapse（Freter・Gonlin・Webster の黒曜石の水和年代）：'Between 400 and 450, the population was estimated at a peak of twenty-eight thousand, between 750 and 800 the population then began to steadily decline. By 900 the population had fallen to fifteen thousand, and by 1200 the population was again less than 1000'",
      "129 の sources/wiki_Copan：'only a few farming hamlets ... 16th century'。現在の自治体 45,312（2023、125 の sources/）"])
ABANDON = {
 "Tikal": dict(P0_range=[10000, 90000], t_abandon="950〜11 世紀", ev=S + "wiki_Tikal：'Population estimates for Tikal vary from 10,000 to as high as 90,000'；'After 950, Tikal was all but deserted'；'Even these final inhabitants abandoned the city in the 10th or 11th centuries and the rainforest claimed the ruins for the next thousand years'；'Some of Tikal's population may have migrated to the Peten Lakes region'", migr="一部は移住の可能性"),
 "Tiwanaku": dict(P0_range=[10000, None], t_abandon="c.1050", ev=S + "wiki_Tiwanaku：'had a population greater than 10,000 individuals'；infobox 'abandoned = circa 1050 AD'；'collapsed around AD 1000'（乾燥化の説）", migr="記述なし"),
 "Mesa_Verde": dict(P0_range=[20000, 20000], t_abandon="1260〜1300", ev=S + "wiki_Mesa_Verde_National_Park：'An estimated 20,000 people lived in the region during the 13th century, but by the start of the 14th century the area was nearly uninhabited'；'tens of thousands of people emigrated or died from starvation'；'Many surviving emigrants may have relocated to southern Arizona and New Mexico'", migr="あり（移住）"),
 "Great_Zimbabwe": dict(P0_range=[10000, 20000], t_abandon="16〜17 世紀", ev=S + "wiki_Great_Zimbabwe：'Earlier estimates suggest a peak population of around 20,000 ... did not exceed 10,000'；'Major construction ... continued until the 15th century. The city was abandoned in the 16th or 17th century'；'It is unknown what caused Great Zimbabwe's demise'", migr="不明"),
}
NOT_OBTAINED = {
 "Teotihuacan": S + "wiki_Teotihuacan：都市 10〜20 万、'continued to be inhabited, though it never reached its previous levels of population'。盆地の崩壊後の人口の数値なし、4 項目の途絶の根拠の文なし",
 "Egypt_FIP": S + "wiki_First_Intermediate_Period_of_Egypt：'drastic reduction in population in the Nile Valley'（数値なし）。小さなピラミッド（Ibi、Merikare）は造られ、分類も kept 寄り",
 "Chaco": S + "wiki_Chaco_Culture_National_Historical_Park：放棄の記述はあるが人口の数値なし",
}
# 登録外の質的な反例候補（人口の数値なしで規則では数えない）
TIKAL_HIATUS = S + "wiki_Tikal：'The \"Tikal hiatus\" refers to a period between the late 6th to late 7th century where there was a lapse in the writing of inscriptions and large-scale construction'；'In 682, Jasaw Chan K'awiil I erected the first dated monument at Tikal in 120 years'；その後 Late Classic に人口の頂点。文字は『日常の使用』でなく碑文の途絶で、規則の項目に厳密には当たらない"
def judge_copan(P0=28000, Pmin=1000, ts=800, t_up=2023):
    pred = math.log(P0 / Pmin) / r
    return {"pred": pred, "T_upper": t_up - ts, "excess_upper": t_up - ts - pred, "excess_lower": COPAN["t_still_low"] - ts - pred, "stag": (t_up - ts - pred) > 0}
cj = judge_copan()
brk = COPAN["P0"] / math.exp(r * (COPAN["t_rec_upper"] - COPAN["t_start"]))
copan_grid = {f"P0={p0}|Pmin={pm}|ts={ts}": round(judge_copan(p0, pm, ts)["excess_upper"], 1)
              for p0 in (20000, 28000) for pm in (1000, 800, 500, 200) for ts in (750, 800, 900)}
# ---- 数え ----
nA = len(ABANDON)
SETS = {
 "M129(5/5)": (5, 5),
 "M129+abandon": (5 + nA, 5 + nA),                    # 放棄＝停滞（規則）
 "insitu(Greece,Britain,Copan_stag)": (3, 3),
 "insitu_Copan_no": (2, 3),
 "all_Copan_no": (4 + nA, 5 + nA),
 "unregistered:+TikalHiatus_as_lost_nostag(all)": (5 + nA, 6 + nA),
 "unregistered:+TikalHiatus(insitu)": (3, 4),
}
OUT = {"study": "143-plost-new-cases", "note": "F_heavy・P_read は生存確率や復興確率ではない", "r_median_038": r, "check_129": CHECK,
       "copan": {**COPAN, **cj, "Pmin_breakeven_upper": brk, "grid_excess_upper": copan_grid}, "abandon": ABANDON,
       "not_obtained": NOT_OBTAINED, "tikal_hiatus_note": TIKAL_HIATUS, "counts": {}, "posterior": {}}
for s, (k, n) in SETS.items():
    OUT["counts"][s] = {"lost_stag": k, "n_lost": n}
    OUT["posterior"][s + "|jeffreys"] = post(k, n); OUT["posterior"][s + "|uniform"] = post(k, n, 1, 1)
# 選び方の偏り：記録に残りにくい「段を失ったがすぐ戻った」事例が m 件抜けていたら p の点推定はいつ 0.75 を割るか
OUT["missing_quick_cases_to_drop_below_0.75"] = {s: next(m for m in range(0, 100) if k / (n + m) < 0.75) for s, (k, n) in SETS.items() if not s.startswith("unreg")}
# ---- 142 の媒体の表に p を入れる（142 の sim を写しで実行、run() を再利用） ----
W = "/tmp/rep143/research"; os.makedirs(f"{W}/142-stag-media-read", exist_ok=True)
for d in ("141-medium-tier-readability", "038-restart-state-evidence", "132-trough-lag-definition", "125-pop-recovery-history"):
    if not os.path.exists(f"{W}/{d}"): os.symlink(f"{RES}/{d}", f"{W}/{d}")
shutil.copy(f"{RES}/142-stag-media-read/sim.py", f"{W}/142-stag-media-read/sim.py")
_cwd = os.getcwd(); os.chdir(f"{W}/142-stag-media-read")
G = runpy.run_path(f"{W}/142-stag-media-read/sim.py", run_name="m142"); os.chdir(_cwd)
R142 = json.load(open(f"{RES}/142-stag-media-read/results.json"))
for k, o in G["RES2"]["A"]["media"].items():
    d = abs(o["P_read_H"] - R142["variants"]["A"]["media"][k]["P_read_H"]); CHECK[f"142A|{k}"] = d; assert d < 1e-12
KEYS = ("mom_mf", "fiche24", "rosetta", "piql", "silica")
MED = {}
for lab, s in (("p=0.9(142)", None), ("all", "M129+abandon"), ("insitu", "insitu(Greece,Britain,Copan_stag)"), ("insitu_Copan_no", "insitu_Copan_no"), ("unreg_insitu+Tikal", "unregistered:+TikalHiatus(insitu)")):
    p = 0.9 if s is None else OUT["posterior"][s + "|jeffreys"]["p_mean"]
    o, dl = G["run"]({"kind": "A", "p": p}); MED[lab] = {"p": p, "mean_delay": dl["mean_delay"], **{k: o[k]["P_read_H"] for k in KEYS}}
assert abs(MED["p=0.9(142)"]["fiche24"] - R142["variants"]["A"]["media"]["fiche24"]["P_read_H"]) < 1e-12
OUT["media_P_read_1000y"] = MED
# ---- 仮説 ----
pa = OUT["posterior"]["M129+abandon|jeffreys"]; pi = OUT["posterior"]["insitu(Greece,Britain,Copan_stag)|jeffreys"]
H = {
 "H155a": {"class": "confirmatory", "value": f"{SETS['M129+abandon'][0]}/{SETS['M129+abandon'][1]}", "supported": SETS["M129+abandon"][0] / SETS["M129+abandon"][1] >= 0.75},
 "H155b": {"class": "confirmatory", "value": {"Copan_stag": "3/3", "Copan_no": "2/3=0.667"}, "supported": None,
           "note": "Copán の谷の人口の下限が『1,000 未満』で損益分岐 %.0f 人をまたぐため、3/3 か 2/3 かが決まらず判定保留" % brk},
 "H155c": {"class": "prediction", "value": "規則で数えられる反例 0（Tikal の空白期は人口の数値なしで数えない）", "supported": False},
 "H155d": {"class": "prediction", "value": pa["F_mean"], "supported": 2.10 <= pa["F_mean"] <= 2.29},
 "H155e": {"class": "prediction", "value": MED["all"]["fiche24"], "supported": MED["all"]["fiche24"] < 0.3},
}
OUT["hypotheses"] = H; OUT["runtime_s"] = round(time.time() - T0, 1)
# ---- 図 ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5)); xx = np.linspace(0.001, 0.999, 400)
for s, ls_ in (("M129(5/5)", ":"), ("M129+abandon", "-"), ("insitu(Greece,Britain,Copan_stag)", "--"), ("insitu_Copan_no", "-.")):
    c = OUT["counts"][s]; ax[0].plot(xx, stats.beta.pdf(xx, .5 + c["lost_stag"], .5 + c["n_lost"] - c["lost_stag"]), ls_, label=f"{s[:24]} {c['lost_stag']}/{c['n_lost']} F={OUT['posterior'][s+'|jeffreys']['F_mean']:.3f}")
ax[0].set_ylim(0, 15); ax[0].set_xlabel("p(stagnation | stage lost)"); ax[0].legend(fontsize=7); ax[0].set_title("143: Jeffreys posterior; F_heavy is not a survival prob.", fontsize=8)
pms = np.linspace(100, 3000, 300)
for ts, st in ((750, ":"), (800, "-"), (900, "--")):
    ax[1].plot(pms, [(2023 - ts) - math.log(28000 / pm) / r for pm in pms], st, label=f"upper excess, t_start={ts}, P0=28,000")
ax[1].axhline(0, c="k", lw=.8); ax[1].axvline(1000, c="r", ls=":", label="source: trough < 1,000 (1200 CE)")
ax[1].set_xlabel("Copán valley trough population Pmin"); ax[1].set_ylabel("excess vs 038 median (years)"); ax[1].legend(fontsize=7)
ax[1].set_title("Copán: stagnation flips below Pmin ≈ %.0f" % brk, fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_plost_insitu.png"), dpi=120)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("check", {k: v for k, v in CHECK.items() if not k.startswith("142A")}, "142A max", max(v for k, v in CHECK.items() if k.startswith("142A")))
print("copan", {k: round(v, 1) for k, v in cj.items() if k != "stag"}, "stag", cj["stag"], "breakeven", round(brk)); print(copan_grid)
for s in SETS: print(s, OUT["counts"][s], {pr: (round(OUT["posterior"][s + "|" + pr]["p_mean"], 3), round(OUT["posterior"][s + "|" + pr]["F_mean"], 3)) for pr in ("jeffreys", "uniform")})
print("missing", OUT["missing_quick_cases_to_drop_below_0.75"])
for k, v in MED.items(): print(k, {a: round(b, 3) for a, b in v.items()})
for h, v in H.items(): print(h, v["class"], v["supported"], v["value"])
print("runtime", OUT["runtime_s"])
