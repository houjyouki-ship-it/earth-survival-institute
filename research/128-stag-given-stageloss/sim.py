"""研究128：125 の停滞の確率は周辺か、段を失った社会の中の値か（open_problems 302）。
計算のみ、ネット接続なし（資料は sources/ と 125 の sources/ に保存済み）。事前登録：PREREGISTRATION.md（2026-10-25 21:41 JST）。
注：F_heavy・V は生存確率や復興確率ではない。
格子の F_heavy は grid.py（127 の sim.py を exec し 127 の Ccond・Cmass・C1 と差 0 を assert）の grid.json を使う。"""
import json, os
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
R125 = json.load(open(os.path.join(RES, "125-pop-recovery-history", "results.json"), encoding="utf8"))
GR = json.load(open(os.path.join(HERE, "grid.json")))
assert max(GR["check_127"].values()) < 1e-12
S125 = "125-pop-recovery-history/sources/"; S128 = "sources/"
# ---- 1. 分類（規則は登録どおり：前にあった 4 項目のうち 2 つ以上が 50 年以上途絶 or 消滅で lost。None=前になかった、"?"=根拠なし） ----
CLS = {
 "Greece_LBA": dict(writing="ceased", metal="?", masonry="ceased", coin=None,
   ev=[S125 + "wiki_Greek_Dark_Ages：'the Linear B script ... ceased to be used'；'total disappearance of Linear B, palatial architecture'；'between 950 BC and 750 BC, the Greeks relearned how to write'"]),
 "Britain_postRoman": dict(writing="kept", metal="kept", masonry="ceased", coin="ceased",
   ev=[S125 + "wiki_Sub-Roman_Britain：'building in stone gradually came to an end'；'no newly minted coins ... after the very early 5th century'；'Latin continued to be used for writing'；Dinas Powys 'evidence of metalworking'",
       S128 + "wiki_Coinage_in_Anglo-Saxon_England：'production of coinage effectively came to an end' 5世紀初め、貨幣としての使用は 7 世紀初めから",
       S128 + "wiki_Anglo-Saxon_architecture：世俗の建物は主に木造、発掘された数百の集落で石積みは 10 件のみ"]),
 "Norse_Greenland": dict(extinct=True, ev=[S125 + "wiki_Norse_settlements_in_Greenland：'abandoned'"]),
 "England_BlackDeath": dict(writing="?", metal="?", masonry="?", coin="?", ev=["途絶の記述なし（Black_Death・Demographics_of_England）"]),
 "Europe_BlackDeath": dict(writing="?", metal="?", masonry="?", coin="?", ev=["途絶の記述なし（Black_Death）"]),
 "Hawaii_contact": dict(writing=None, metal=None, masonry="?", coin=None, ev=["前に文字・金属・貨幣なし"]),
 "LatinAmerica_conquest": dict(writing="?", metal="?", masonry="?", coin="?", ev=["地域の社会（スペイン系を含む）で途絶の記述なし。先住の文字の喪失は地域の社会としては数えない（規則）"]),
 "Easter_Island": dict(writing="ceased", metal=None, masonry=None, coin=None,
   ev=[S128 + "wiki_Rongorongo：'writing ceased and most tablets lost or destroyed in the 1860s'",
       S128 + "wiki_Moai：'carved ... mostly between 1250 and 1500'（1722 の前に新造は終わっており項目に数えない）"]),
 "ThirtyYearsWar_Germany": dict(writing="?", metal="?", masonry="?", coin="?", ev=["途絶の記述なし"]),
 "Cambodia_KhmerRouge": dict(writing="?", metal="?", masonry="?", coin="?", ev=["125 の頁に貨幣の廃止の記述なし（廃止は 1975〜1980 で 50 年未満なので規則上どちらでも数えない）"]),
}
def lost_main(c):
    if c.get("extinct"): return True
    return sum(1 for k in ("writing", "metal", "masonry", "coin") if c.get(k) == "ceased") >= 2
def lost_alt(c):  # 感度（登録外・記述）：前にあった項目の過半が途絶
    if c.get("extinct"): return True
    pres = [k for k in ("writing", "metal", "masonry", "coin") if c.get(k) not in (None,)]
    return sum(1 for k in pres if c[k] == "ceased") * 2 > len(pres)
# ---- 2. 停滞の有無は 125 の値・規則のまま ----
ALL = {c["key"]: c for c in R125["cases"] + R125["cases_supplementary"]}
def stag(key):
    c = ALL[key]
    if key == "Norse_Greenland": return True
    if c["status"] == "interval":
        ex_hi = (c["t_rec_upper"] - c["t_start"]) - c["pred038_median_years"]; return True if ex_hi > 0 else False
    ex = c["excess_vs038_median"]
    if c["status"] == "cens" and ex < 0: return None
    return ex > 0
M = [k for k in ("England_BlackDeath", "Europe_BlackDeath", "Greece_LBA", "Britain_postRoman", "Norse_Greenland")]
X = M + ["Hawaii_contact", "LatinAmerica_conquest", "Easter_Island", "ThirtyYearsWar_Germany", "Cambodia_KhmerRouge"]
# 125 の V2b（p=0.6）を再現
assert abs(sum(stag(k) for k in M) / len(M) - R125["V2_inputs"]["p_nr_no_immig"]) < 1e-12
OUT = {"study": "128-stag-given-stageloss", "note": "F_heavy・V は生存確率や復興確率ではない", "classification": {}}
for k in X:
    OUT["classification"][k] = {**{a: b for a, b in CLS[k].items()}, "lost": lost_main(CLS[k]), "lost_alt": lost_alt(CLS[k]), "stag": stag(k)}
def counts(keys, f):
    kl = [k for k in keys if f(CLS[k])]; kk = [k for k in keys if not f(CLS[k])]
    return {"lost_stag": sum(stag(k) for k in kl), "n_lost": len(kl), "kept_stag": sum(stag(k) for k in kk), "n_kept": len(kk), "lost": kl, "kept": kk}
CNT = {"M": counts(M, lost_main), "X": counts(X, lost_main), "M_alt": counts(M, lost_alt), "X_alt": counts(X, lost_alt)}
OUT["counts"] = CNT
# ---- 3. 格子 → 事後の F_heavy ----
PS = np.array(GR["PS"]); SL = "slow_Tasmania"
def Fcurve(kn, vn): return np.array([GR["GRID"][f"{p:.6f}|{kn}|{vn}"]["F_heavy"] for p in PS])
def Ncurve(): return np.array([GR["GRID"][f"{p:.6f}|{SL}|D2rz|none"]["new_heavy"] for p in PS]) / GR["V0_new_heavy"] - 1
CUR = {"slow|D2rz": Fcurve(SL, "D2rz|none"), "slow|D1": Fcurve(SL, "D1|main"), "slow|D4": Fcurve(SL, "D4|main"),
       "trigger_mix|D2rz": Fcurve("trigger_mix", "D2rz|none"), "fast_state|D2rz": Fcurve("fast_state", "D2rz|none")}
NREL = Ncurve()
rng = np.random.default_rng(20261025)
def interp_extrap(x, xs, ys):  # p<0.5 は線形外挿（0.5〜0.6 の傾き）
    y = np.interp(x, xs, ys); lo = x < xs[0]
    y[lo] = ys[0] + (x[lo] - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0]); return y
POST = {}
for setn in ("M", "X", "M_alt", "X_alt"):
    k, n = CNT[setn]["lost_stag"], CNT[setn]["n_lost"]
    for prior, (a0, b0) in (("jeffreys", (0.5, 0.5)), ("uniform", (1, 1))):
        d = rng.beta(a0 + k, b0 + n - k, size=200000)
        o = {"k": k, "n": n, "p_mean": float(d.mean()), "p_q10_50_90": [float(np.quantile(d, q)) for q in (0.1, 0.5, 0.9)],
             "P(p>=0.864)": float((d >= GR["p_mass"]).mean())}
        for cn, cv in CUR.items():
            fv = interp_extrap(d, PS, cv); o[f"F|{cn}"] = {"mean": float(fv.mean()), "q10_50_90": [float(np.quantile(fv, q)) for q in (0.1, 0.5, 0.9)]}
        nv = interp_extrap(d, PS, NREL); o["new_heavy_rel_V0"] = {"mean": float(nv.mean()), "q10_50_90": [float(np.quantile(nv, q)) for q in (0.1, 0.5, 0.9)]}
        POST[f"{setn}|{prior}"] = o
OUT["posterior"] = POST
OUT["grid_F_slow_D2rz"] = dict(zip([f"{p:.3f}" for p in PS], CUR["slow|D2rz"].tolist()))
OUT["grid_new_heavy_rel_V0"] = dict(zip([f"{p:.3f}" for p in PS], NREL.tolist()))
OUT["linearity_maxdev"] = float(np.max(np.abs(CUR["slow|D2rz"] - np.interp(PS, [PS[0], PS[-1]], [CUR["slow|D2rz"][0], CUR["slow|D2rz"][-1]]))))
# ---- 4. 仮説 ----
cM, cX = CNT["M"], CNT["X"]; pj = POST["M|jeffreys"]
H = {
 "H140a": {"class": "confirmatory", "value": f"{cM['lost_stag']}/{cM['n_lost']}", "rule": "M で p|lost>=0.75", "supported": cM["lost_stag"] / cM["n_lost"] >= 0.75},
 "H140b": {"class": "confirmatory", "value": f"{cX['kept_stag']}/{cX['n_kept']}", "rule": "X で kept かつ停滞 0 件", "supported": cX["kept_stag"] == 0},
 "H140c": {"class": "confirmatory", "value": f"{cX['lost_stag']}/{cX['n_lost']}", "rule": "X で p|lost>=0.75", "supported": cX["lost_stag"] / cX["n_lost"] >= 0.75},
 "H140d": {"class": "prediction", "value": pj["F|slow|D2rz"]["mean"], "rule": "M Jeffreys 事後平均の F_heavy>=2.10", "supported": pj["F|slow|D2rz"]["mean"] >= 2.10},
 "H140e": {"class": "prediction", "value": pj["new_heavy_rel_V0"]["mean"], "rule": "新しい側の重い分 V0 比 <= -0.14", "supported": pj["new_heavy_rel_V0"]["mean"] <= -0.14},
}
# 予測の分類との一致（記述）
pred = {"Greece_LBA": True, "Britain_postRoman": True, "Norse_Greenland": True, "England_BlackDeath": False, "Europe_BlackDeath": False,
        "Hawaii_contact": False, "LatinAmerica_conquest": False, "Easter_Island": True, "ThirtyYearsWar_Germany": False, "Cambodia_KhmerRouge": False}
OUT["prereg_class_mismatch"] = [k for k in X if pred[k] != lost_main(CLS[k])]
OUT["hypotheses"] = H
# Fisher の正確検定（記述）：段の喪失と停滞の関連
for s in ("M", "X", "X_alt"):
    c = CNT[s]; tab = [[c["lost_stag"], c["n_lost"] - c["lost_stag"]], [c["kept_stag"], c["n_kept"] - c["kept_stag"]]]
    OUT[f"fisher_{s}"] = {"table": tab, "p_one_sided": float(stats.fisher_exact(tab, alternative="greater")[1])}
# ---- 5. 図 ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for cn, cv in CUR.items(): ax[0].plot(PS, cv, "o-", ms=3, label=cn)
ax[0].axvline(0.6, ls=":", c="gray"); ax[0].axvline(GR["p_mass"], ls="--", c="gray")
ax[0].set_xlabel("p(stagnation | stage lost)"); ax[0].set_ylabel("F_heavy (H=1000, W)"); ax[0].legend(fontsize=7)
ax[0].set_title("128: F_heavy vs p|lost (not a survival prob.)", fontsize=9)
xx = np.linspace(0, 1, 400)
for s, ls in (("M", "-"), ("X", "--"), ("X_alt", ":")):
    c = CNT[s]; ax[1].plot(xx, stats.beta.pdf(xx, 0.5 + c["lost_stag"], 0.5 + c["n_lost"] - c["lost_stag"]), ls, label=f"{s}: {c['lost_stag']}/{c['n_lost']} (Jeffreys)")
ax[1].axvline(0.6, ls=":", c="gray"); ax[1].set_ylim(0, 8); ax[1].set_xlabel("p|lost"); ax[1].legend(fontsize=8)
ax[1].set_title("posterior of p|lost from historical cases", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_p_lost.png"), dpi=120)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(OUT["counts"], ensure_ascii=False))
for k, v in POST.items(): print(k, round(v["p_mean"], 3), "F slow D2rz", round(v["F|slow|D2rz"]["mean"], 3), v["F|slow|D2rz"]["q10_50_90"], "new", round(v["new_heavy_rel_V0"]["mean"], 4))
print("grid", {k: round(v, 4) for k, v in OUT["grid_F_slow_D2rz"].items()}, "lin dev", round(OUT["linearity_maxdev"], 4))
print("mismatch", OUT["prereg_class_mismatch"])
for s in ("M", "X", "X_alt"): print("fisher", s, OUT[f"fisher_{s}"])
for h, v in H.items(): print(h, v["class"], v["supported"], v["value"])
