"""研究070：群の数と冊子の預け先の重なり（open_problems 173）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md。
069 の sim.py を行頭の「# ---------- 版 ----------」の前まで exec し、069 の主を差0で再現してから、
重い崩壊の標本で「G 個の独立な群が、それぞれ届く範囲の写しだけを使う」形に替える。新しい乱数は seed 20261070。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P69 = os.path.abspath(os.path.join(HERE, "..", "069-wordless-map-use", "sim.py"))
src = open(P69, encoding="utf8").read(); cut = src.index("\n# ---------- 版 ----------\n")
G69 = {"__file__": P69, "__name__": "sim069"}
exec(compile(src[:cut], P69, "exec"), G69)
r69 = json.load(open(os.path.join(HERE, "..", "069-wordless-map-use", "results.json"), encoding="utf8"))
N = G69["N"]; ver = G69["ver"]; ratio_all = G69["ratio_all"]; L66 = G69["L66"]; pp68 = G69["pp68"]; PM = G69["PM"]; get = G69["get"]; qf = G69["qf"]
R69 = G69["R69"]; rho69 = G69["rho69"]; C_T = G69["C_T"]; C_P = G69["C_P"]; C_pr = G69["C_pr"]
lam = G69["lam"]; rho1 = G69["rho1"]; r_reach = G69["r_reach"]; S_ind = G69["S_ind"]; f_i = G69["f_i"]; taup_061 = G69["taup_061"]
heavy_mask = G69["heavy_mask"]; G63 = G69["G65"]["G64"]["G63"]; h_R = G63["h_R"]; q_h = G63["q_h"]
KEYS = ["A_Ord/U3", "A_XPT_S/U2"]; PARTS = ("total", "heavy", "light"); M1 = "A_Ord/U3"
R = {"study": "070-group-deposit-overlap", "seed": 20261070}
N_P, KU = 100, 20; MU = N_P / KU

# ---------- 検査1：069 の主を差0で再現 ----------
RT = {}; DV = {}
PMAIN = PM["main"]; PPF = pp68()
RT["chk069"], DV["chk069"] = ratio_all(ver(rho=rho69(L66, PPF, PMAIN)))
d69 = max(abs(RT["chk069"][k][p] - r69["ratio_to_V0"]["main"][k][p]) for k in KEYS for p in PARTS)
R["check_069_main"] = {"max_abs_diff": d69, "reproduced": d69 < 1e-12}

# ---------- 資料 ----------
CJ = json.load(open(os.path.join(HERE, "sources", "cases.json"), encoding="utf8"))
REG = CJ["regions"]; NR = len(REG); ix = {r: i for i, r in enumerate(REG)}
CONT = [CJ["continent_of_region"][r] for r in REG]; CONTS = sorted(set(CONT))
same_cont = np.array([[CONT[i] == CONT[j] for j in range(NR)] for i in range(NR)])
def vec(d):
    v = np.zeros(NR)
    for k, x in d.items():
        if k in ix: v[ix[k]] += x
    return v
XS5 = CJ["xia_survivors_by_region"]["S5_partial_livestock"]; XS6 = CJ["xia_survivors_by_region"]["S6_no_livestock"]
from collections import Counter
W = {"xia150": vec({k: v["surv150"] for k, v in XS5.items()}), "xia47": vec({k: v["surv47"] for k, v in XS5.items()}),
     "pop": vec({k: v["pop2010"] for k, v in XS5.items()}), "S6": vec({k: v["surv150"] for k, v in XS6.items()}),
     "refuge": vec(Counter(CJ["refuges"]["asrs_2023"]["regions"].values()))}
W = {k: v / v.sum() for k, v in W.items()}
def largest_remainder(c, k=KU):
    c = np.asarray(c, float); q = c / c.sum() * k; u = np.floor(q).astype(int)
    for i in np.argsort(-(q - u), kind="stable")[: k - u.sum()]: u[i] += 1
    return u
cnt_nat = vec(Counter(n["region"] for n in CJ["national_libraries"] if n["region"]))
cnt_big = vec(Counter(l["region"] for l in CJ["largest_libraries"] if l["region"]))
popv = W["pop"]; even = np.zeros(NR); even[np.argsort(-popv, kind="stable")[:KU]] = 1
DEP = {"D_nat": largest_remainder(cnt_nat), "D_big": largest_remainder(cnt_big), "D_pop": largest_remainder(popv),
       "D_surv": largest_remainder(W["xia150"]), "D_even": even.astype(int)}
assert all(v.sum() == KU for v in DEP.values())
R["regions"] = REG; R["continent"] = CONT
R["weights"] = {k: {REG[i]: float(v[i]) for i in range(NR) if v[i] > 0} for k, v in W.items()}
R["deposit_units"] = {k: {REG[i]: int(v[i]) for i in range(NR) if v[i] > 0} for k, v in DEP.items()}
R["overlap"] = {}
for wk, wv in W.items():
    R["overlap"][wk] = {}
    for dk, dv in DEP.items():
        dc = same_cont @ dv
        R["overlap"][wk][dk] = {"o_sub": float(wv @ (dv > 0)), "o_cont": float(wv @ (dc > 0)), "mean_units_sub": float(wv @ dv), "mean_units_cont": float(wv @ dc)}
R["southern_share_check"] = {"xia_pop_ANZ_SAm_SAfr": float(sum(W["pop"][ix[r]] for r in ("Australia and New Zealand", "South America", "Southern Africa")))}

# ---------- 新しい乱数 ----------
g = np.random.default_rng(20261070)
GMAX = 10
U_geo = g.integers(0, 3, N)                       # 主の地理：xia150 / xia47 / pop を等確率【推測】
G_main = np.clip(np.round(10 ** g.uniform(0, 1, N)).astype(int), 1, GMAX)   # 1〜10【推測】
U_reach = g.uniform(0, 1, N) < 0.5                # True → 小地域、False → 大陸【推測】
GUM = g.gumbel(size=(N, NR))
def place(geo, Gv):
    """標本ごとに群の小地域を重みに比例して重複なしに選ぶ（Gumbel-top-k）。geo は名前か標本ごとの添字。戻り値 N×GMAX（-1 は群なし）。"""
    if isinstance(geo, str): Wm = np.broadcast_to(W[geo], (N, NR))
    else: Wm = np.stack([W["xia150"], W["xia47"], W["pop"]])[geo]
    with np.errstate(divide="ignore"): key = np.where(Wm > 0, np.log(Wm) + GUM, -np.inf)
    order = np.argsort(-key, axis=1, kind="stable")[:, :GMAX]
    ok = np.take_along_axis(key, order, 1) > -np.inf
    Gm = np.broadcast_to(np.asarray(Gv), (N,))
    ok &= np.arange(GMAX)[None, :] < Gm[:, None]
    return np.where(ok, order, -1)
def reach_units(pos, dep, reach):
    """群ごとに届く単位の数 D（N×GMAX、群なしは -1）。reach は "sub"・"cont"・"global" か標本ごとの真偽（True=sub）。"""
    dc = same_cont @ dep
    sub = np.where(pos >= 0, dep[np.maximum(pos, 0)], -1); con = np.where(pos >= 0, dc[np.maximum(pos, 0)], -1)
    glo = np.where(pos >= 0, KU, -1)
    if isinstance(reach, str): return {"sub": sub, "cont": con, "global": glo}[reach]
    return np.where(reach[:, None], sub, con)

def rho70(D, spread=False, ppf=PPF, pm=PMAIN):
    Df = D.astype(float); valid = D >= 0
    def f(T):
        hm = heavy_mask(T)
        pc = (1 - L66) * np.exp(-lam * T)
        Rr = R69(T, pc, ppf, pm)                     # 重い崩壊では 069 の読める確率、軽い崩壊では 1
        a = np.exp(-T / taup_061) * f_i
        f1 = (1 - h_R) * (1 - a) ** MU + h_R * (1 - q_h * a) ** MU
        Sg = np.where(valid, 1 - f1[:, None] ** np.maximum(Df, 0), 0.0)
        if spread:
            Gn = valid.sum(1)
            P = (1 - (1 - Rr) ** Gn) * (1 - np.prod(1 - Sg, axis=1))
        else:
            P = 1 - np.prod(np.where(valid, 1 - Rr[:, None] * Sg, 1.0), axis=1)
        light = S_ind(T, N_P, KU, f_i, taup_061)
        return 1 - (1 - rho1) * (1 - r_reach * np.where(hm, P, Rr * light))
    return f

# ---------- 検査2：G=1・世界に届く形で 069 の主と一致 ----------
pos1 = place("xia150", 1); D_glob1 = reach_units(pos1, DEP["D_nat"], "global")
RT["chk_G1_global"], DV["chk_G1_global"] = ratio_all(ver(rho=rho70(D_glob1)))
d2 = max(abs(RT["chk_G1_global"][k][p] - r69["ratio_to_V0"]["main"][k][p]) for k in KEYS for p in PARTS)
R["check_G1_global_equals_069"] = {"max_abs_diff": d2, "reproduced": d2 < 1e-12}

# ---------- 版 ----------
POS = {"main": place(U_geo, G_main)}
for geo in ("xia150", "xia47", "pop", "refuge", "S6"): POS["geo_" + geo] = place(geo, G_main)
for Gf in (1, 3, 10): POS[f"G{Gf}"] = place(U_geo, Gf)
R["groups_realised"] = {k: {"mean_G": float((v >= 0).sum(1).mean())} for k, v in POS.items()}
VS = {}
for dk in DEP: VS["main" if dk == "D_nat" else dk] = (POS["main"], dk, U_reach, False)
for geo in ("xia150", "xia47", "pop", "refuge", "S6"): VS["geo_" + geo] = (POS["geo_" + geo], "D_nat", U_reach, False)
for Gf in (1, 3, 10): VS[f"G{Gf}"] = (POS[f"G{Gf}"], "D_nat", U_reach, False)
for rc in ("sub", "cont", "global"): VS["reach_" + rc] = (POS["main"], "D_nat", rc, False)
VS["spread"] = (POS["main"], "D_nat", U_reach, True)
VS["spread_D_surv"] = (POS["main"], "D_surv", U_reach, True)
VS["G3_spread_cont"] = (POS["G3"], "D_nat", "cont", True)
VS["G3_spread_cont_D_surv"] = (POS["G3"], "D_surv", "cont", True)
VS["geo_refuge_D_surv"] = (POS["geo_refuge"], "D_surv", U_reach, False)
for nm, (pos, dk, rc, sp) in VS.items():
    RT[nm], DV[nm] = ratio_all(ver(rho=rho70(reach_units(pos, DEP[dk], rc), spread=sp)))
R["scan_G"] = []
for Gf in (1, 2, 3, 5, 10):
    pos = POS[f"G{Gf}"] if f"G{Gf}" in POS else place(U_geo, Gf); row = {"G": Gf}
    for lab, rc, sp in (("nospread_mix", U_reach, False), ("spread_cont", "cont", True), ("nospread_global", "global", False)):
        rr, _ = ratio_all(ver(rho=rho70(reach_units(pos, DEP["D_nat"], rc), spread=sp)))
        row[lab] = rr[M1]["heavy"]; row[lab + "_XPT"] = rr["A_XPT_S/U2"]["heavy"]
    R["scan_G"].append(row)
RT["069_main"] = r69["ratio_to_V0"]["main"]
R["ratio_to_V0"] = RT

# ---------- 費用あたり（TAB=1、069 と同じ式。置き方を替えても費用は同じ） ----------
C_map = C_pr; R["CE"] = {}
for key in KEYS:
    a, u = key.split("/"); R["CE"][key] = {}
    for p in ("total", "heavy"):
        vt = get("main", key)["TAB"][p]; vtc = get("main", key)["TC"][p]; R["CE"][key][p] = {}
        for nm in ["chk069", "main", "D_big", "D_pop", "D_surv", "D_even", "spread", "reach_global"]:
            dall = DV[nm][a][u]["TC"][p] - vtc
            R["CE"][key][p][nm] = qf(dall / vt * C_T / (C_P(100) + C_pr + C_map)) if dall > 0 else [0.0, 0.0, 0.0]

# ---------- 仮説 ----------
H = {}; h = lambda nm, k=M1: RT[nm][k]["heavy"]; h69 = r69["ratio_to_V0"]["main"][M1]["heavy"]
v = [h("main"), h69, h("main") - h69]; H["H82a"] = {"value": v, "rule": "main heavy <= 069 main - 0.010", "supported": v[2] <= -0.010}
v = [h("reach_global"), h69, h("reach_global") - h69]; H["H82b"] = {"value": v, "rule": "reach_global heavy >= 069 main + 0.010", "supported": v[2] >= 0.010}
v = [h("D_surv"), h("main"), h("D_surv") - h("main")]; H["H82c"] = {"value": v, "rule": "D_surv heavy >= D_nat + 0.010", "supported": v[2] >= 0.010}
dd = {dk: h("main" if dk == "D_nat" else dk) for dk in DEP}; H["H82d"] = {"value": dd, "rule": "D_big lowest among placements", "supported": min(dd, key=dd.get) == "D_big"}
v = R["overlap"]["xia150"]["D_big"]["o_sub"]; H["H82e"] = {"value": v, "rule": "o_sub(xia150, D_big) < 0.5", "supported": v < 0.5}
v = h("G3_spread_cont"); H["H82f"] = {"value": v, "rule": "G3 spread cont heavy >= 0.85", "supported": v >= 0.85}
vs = [h("geo_" + x) for x in ("xia150", "xia47", "pop", "refuge", "S6")]; H["H82g"] = {"value": [min(vs), max(vs), max(vs) - min(vs)], "rule": "range of geo versions <= 0.05", "supported": max(vs) - min(vs) <= 0.05}
v = len(CJ["switch_rule_count"]["sources"]); H["H82h"] = {"value": v, "rule": "numeric public sources >= 5", "supported": v >= 5}
for xx in H.values(): xx["supported"] = bool(xx["supported"])
R["hypotheses"] = H

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
order = np.argsort(-W["xia150"], kind="stable"); yy = np.arange(NR)[::-1]
for j, (k, col) in enumerate((("xia150", "C3"), ("xia47", "C1"), ("pop", "C0"), ("refuge", "C2"))):
    axs[0].barh(yy + 0.2 - 0.13 * j, W[k][order], height=0.13, color=col, label=f"survivor share: {k}")
ax2 = axs[0].twiny()
for j, (dk, mk) in enumerate((("D_nat", "o"), ("D_big", "s"), ("D_surv", "^"), ("D_even", "x"))):
    ax2.plot(DEP[dk][order], yy, mk, ms=5, mfc="none", label=f"units: {dk}")
ax2.set_xlabel("deposit units (of 20)"); ax2.legend(fontsize=6, loc="lower right")
axs[0].set_yticks(yy); axs[0].set_yticklabels([REG[i] for i in order], fontsize=7); axs[0].set_xlabel("share of survivors / population"); axs[0].legend(fontsize=6, loc="center right")
sg = R["scan_G"]
for lab, mk in (("nospread_mix", "o-"), ("spread_cont", "s--"), ("nospread_global", "^:")):
    axs[1].plot([r["G"] for r in sg], [r[lab] for r in sg], mk, label=lab)
axs[1].axhline(h69, color="k", ls=":", lw=0.8, label="069 main (1 group, global reach)")
axs[1].set_xscale("log"); axs[1].set_xlabel("number of independent groups G"); axs[1].set_ylabel("heavy-collapse TC / V0 (A_Ord/U3, D_nat)"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_overlap_scanG.png"), dpi=130); plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 6))
names = ["069_main", "main", "D_big", "D_pop", "D_surv", "D_even", "geo_xia150", "geo_xia47", "geo_pop", "geo_refuge", "geo_S6", "geo_refuge_D_surv",
         "G1", "G3", "G10", "reach_sub", "reach_cont", "reach_global", "spread", "spread_D_surv", "G3_spread_cont", "G3_spread_cont_D_surv"]
yy = np.arange(len(names))
ax.barh(yy - 0.2, [RT[n][M1]["heavy"] for n in names], height=0.4, label="A_Ord/U3")
ax.barh(yy + 0.2, [RT[n]["A_XPT_S/U2"]["heavy"] for n in names], height=0.4, color="C1", alpha=0.7, label="A_XPT_S/U2")
ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=7); ax.axvline(h69, color="k", ls=":", lw=0.7)
lo = min(min(RT[n][M1]["heavy"] for n in names), min(RT[n]["A_XPT_S/U2"]["heavy"] for n in names))
ax.set_xlim(max(0.0, lo - 0.03), 1.0); ax.set_xlabel("heavy-collapse TC / V0"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_versions.png"), dpi=130); plt.close(fig)

R["runtime_s"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float, ensure_ascii=False)
print("check069", R["check_069_main"], "checkG1", R["check_G1_global_equals_069"])
print("deposits", json.dumps(R["deposit_units"]))
print("overlap", json.dumps({w: {d: [round(x["o_sub"], 3), round(x["o_cont"], 3)] for d, x in v.items()} for w, v in R["overlap"].items()}))
print("groups", json.dumps(R["groups_realised"]))
for nm in RT: print(nm, {k: {p: round(RT[nm][k][p], 4) for p in PARTS} for k in KEYS})
print("scanG", json.dumps(R["scan_G"]))
print("CE total", json.dumps(R["CE"][M1]["total"])); print("CE heavy", json.dumps(R["CE"][M1]["heavy"]))
for k, xx in H.items(): print(k, xx["supported"], xx["value"])
print("runtime", R["runtime_s"])
