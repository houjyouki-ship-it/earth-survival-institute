"""研究070 の事後の追加（事前登録の外、探索として報告する）。sim.py を「# ---------- 版 ----------」の前まで exec し、
避難所の地理での置き方の比較と、4つの地理（xia150・xia47・pop・refuge）を等確率に混ぜた版で置き方を比べる。"""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(HERE, "sim.py")
src = open(P, encoding="utf8").read(); cut = src.index("\n# ---------- 版 ----------\n")
G = {"__file__": P, "__name__": "sim070"}; exec(compile(src[:cut], P, "exec"), G)
place = G["place"]; reach_units = G["reach_units"]; rho70 = G["rho70"]; ratio_all = G["ratio_all"]; ver = G["ver"]; DEP = G["DEP"]; W = G["W"]
N = G["N"]; G_main = G["G_main"]; U_reach = G["U_reach"]; NR = G["NR"]; GUM = G["GUM"]; GMAX = G["GMAX"]
g = np.random.default_rng(20261170); U4 = g.integers(0, 4, N)
Wst = np.stack([W["xia150"], W["xia47"], W["pop"], W["refuge"]])[U4]
with np.errstate(divide="ignore"): key = np.where(Wst > 0, np.log(Wst) + GUM, -np.inf)
order = np.argsort(-key, axis=1, kind="stable")[:, :GMAX]; ok = (np.take_along_axis(key, order, 1) > -np.inf) & (np.arange(GMAX)[None, :] < G_main[:, None])
POS4 = np.where(ok, order, -1); POSR = place("refuge", G_main)
OUT = {}
for dk in DEP:
    for nm, pos in (("refuge", POSR), ("mix4", POS4)):
        if nm == "refuge" and dk in ("D_nat", "D_surv"): continue
        rr, _ = ratio_all(ver(rho=rho70(reach_units(pos, DEP[dk], U_reach))))
        OUT[f"{nm}_{dk}"] = {k: rr[k]["heavy"] for k in ("A_Ord/U3", "A_XPT_S/U2")}
json.dump(OUT, open(os.path.join(HERE, "explore_posthoc.json"), "w"), indent=1)
for k, v in OUT.items(): print(k, {a: round(b, 4) for a, b in v.items()})
