"""研究133：崩壊後の人口の下り坂 D を停滞（谷からの遅れ）と別に T2 の時計へ明示的に入れる（open_problems 309）。
注：F_heavy・V は生存確率や復興確率ではない。計算のみ、ネット接続なし。事前登録 PREREGISTRATION.md（07:43 JST）。
再現：132 の MIX|phi0・B|phi0 と差 0、D を pick と対にした版が定義A と一致（assert）。"""
import json, time, os, math, itertools
import numpy as np
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
P127 = os.path.join(RES, "127-stag-corr-horizon", "sim.py")
src = open(P127, encoding="utf8").read(); body = src[:src.index("\nSL = \"slow_Tasmania\"")]
G = {"__file__": P127, "__name__": "sim127"}
exec(compile(body, P127, "exec"), G)
print("prefix", round(time.time() - t0), "s", flush=True)
R132 = json.load(open(os.path.join(RES, "132-trough-lag-definition", "results.json"), encoding="utf8"))
R125 = json.load(open(os.path.join(RES, "125-pop-recovery-history", "results.json"), encoding="utf8"))
lost, KAPS, LK, r_pop, U_pick, U_stag, N = (G[k] for k in ("lost", "KAPS", "LK", "r_pop", "U_pick", "U_stag", "N"))
T2o = G["T2_stag"]; FIXED = G["FIXED"]; COMPS = G["COMPS"]
VA = np.asarray(R132["VA"], float); VB = np.asarray(R132["VB"], float)
idx = np.minimum((U_pick * len(VA)).astype(int), len(VA) - 1)
assert np.array_equal(G["pick"], VA[idx])
# 下り坂の入力（125 の decline_years）
dec = {c["key"]: (c["cls"], c["decline_years"]) for c in R125["cases"] + R125["cases_supplementary"]}
DC_main = [dec["Greece_LBA"][1], dec["Britain_postRoman"][1], dec["Easter_Island"][1]]
DC_supp = DC_main + [dec["Copan_valley"][1]]
DS_main = [dec["England_BlackDeath"][1], dec["Hawaii_contact"][1], dec["LatinAmerica_conquest"][1]]
DS_supp = DS_main + [dec[k][1] for k in ("ThirtyYearsWar_Germany", "Ireland_famine", "Cambodia_KhmerRouge")]
assert DC_main == [200, 200, 155] and DS_main == [102, 94, 150] and DC_supp[-1] == 425 and DS_supp[3:] == [30, 116, 3]
g = np.random.default_rng(20261026); U_D = g.uniform(size=N); U_D2 = g.uniform(size=N)
draw = lambda vals, U: np.asarray(vals, float)[np.minimum((U * len(vals)).astype(int), len(vals) - 1)]
PK = 1/6; pf = lambda k: np.where(lost[k], 0.9, PK)
MASK = {k: U_stag < pf(k) for k in COMPS}
SA = {k: np.where(MASK[k], VA[idx], 0.0) for k in COMPS}
SB = {k: np.where(MASK[k], VB[idx], 0.0) for k in COMPS}
CUR = {"phi": 0.0, "X": None, "pd": 0.0, "D": None}
def T2x(k, st, stag, r):
    t = T2o(k, st, stag, r)
    if k in FIXED: return t
    ex = 0.0
    if CUR["phi"]: ex = ex + CUR["phi"] * CUR["X"][k]
    if CUR["pd"]: ex = ex + CUR["pd"] * CUR["D"][k]
    return t + np.where((st == 2) | (st == 3), ex, 0.0)
G["T2_stag"] = T2x
SL = "slow_Tasmania"; MAIN3 = ("D2rz|none", "D1|main", "D4|main")
def run(S, phi=0.0, X=None, pd=0.0, D=None):
    CUR.update(phi=0.0, X=None, pd=0.0, D=None); vb = G["base_val_k"](S, r_pop)
    CUR.update(phi=phi, X=X, pd=pd, D=D)
    o = {vn: G["run_k"](KAPS[SL], LK[vn], S, r_pop, False, vb) for vn in MAIN3}
    CUR.update(phi=0.0, X=None, pd=0.0, D=None); return o
def Darr(DC, DS=None, green=False):
    Dc = draw(DC, U_D); Ds = draw(DS, U_D2) if DS else 0.0
    out = {}
    for k in COMPS:
        stagD = np.where(MASK[k] & ((idx != 2) | green), Dc, 0.0)
        out[k] = stagD + np.where(~MASK[k], Ds, 0.0)
    return out
add = lambda a, b: {k: a[k] + b[k] for k in COMPS}
OUT = {}; chk = {}
# R0 再現
o = run(SA); OUT["R0_A"] = o
o2 = run(SB); OUT["R0_B"] = o2
for vn in MAIN3:
    chk[f"A|{vn}"] = abs(o[vn]["F_heavy"] - R132["OUT"]["MIX|0.000000"][vn]["F_heavy"])
    chk[f"B|{vn}"] = abs(o2[vn]["F_heavy"] - R132["OUT"]["B|0.000000"][vn]["F_heavy"])
assert max(chk.values()) < 1e-12, chk
# E4 検算：D を pick と対に
D4 = {k: np.where(MASK[k] & (idx != 2), 200.0, 0.0) for k in COMPS}
S4 = add(SB, D4); chk["E4_arr"] = max(float(np.abs(S4[k] - SA[k]).max()) for k in COMPS); assert chk["E4_arr"] < 1e-9
o = run(S4); OUT["E4"] = o
for vn in MAIN3: chk[f"E4|{vn}"] = abs(o[vn]["F_heavy"] - OUT["R0_A"][vn]["F_heavy"])
assert max(chk[f"E4|{vn}"] for vn in MAIN3) < 1e-9
print("checks ok", {k: float(v) for k, v in chk.items()}, flush=True)
D1_ = Darr(DC_main); D2_ = Darr(DC_main, DS_main); D3_ = Darr(DC_supp); D7_ = Darr(DC_main, DS_supp); D8_ = Darr(DC_main, green=True)
VERS = {"E1": dict(S=add(SB, D1_)), "E2": dict(S=add(SB, D2_)), "E3": dict(S=add(SB, D3_)),
        "E5": dict(S=add(SB, D2_), pd=1.0, D=D2_), "E6": dict(S=add(SB, D2_), pd=0.5, D=D2_),
        "E7": dict(S=add(SB, D7_)), "E8": dict(S=add(SB, D8_)), "E9": dict(S=add(SB, D2_), phi=0.807, X=SB)}
for name, v in VERS.items():
    OUT[name] = run(v["S"], v.get("phi", 0.0), v.get("X"), v.get("pd", 0.0), v.get("D"))
    print(name, {vn.split("|")[0]: round(OUT[name][vn]["F_heavy"], 4) for vn in MAIN3}, round(time.time() - t0), "s", flush=True)
FH = lambda n, vn="D2rz|none": OUT[n][vn]["F_heavy"]
mean_Dc = {k: float(D1_[k][MASK[k] & (idx != 2)].mean()) for k in ("U3",) if k in COMPS}
order_ok = {n: bool(FH(n) >= FH(n, "D1|main") >= FH(n, "D4|main")) for n in OUT}
from scipy.stats import mannwhitneyu
mw_main = mannwhitneyu(DC_main, DS_main, alternative="greater", method="exact").pvalue
mw_supp = mannwhitneyu(DC_supp, DS_supp, alternative="greater", method="exact").pvalue
a = FH("R0_A")
H = {"H145a": {"E1": FH("E1"), "rel": FH("E1") / a - 1, "supported": abs(FH("E1") / a - 1) <= 0.03},
     "H145b": {"E2": FH("E2"), "rel": FH("E2") / FH("E1") - 1, "supported": abs(FH("E2") / FH("E1") - 1) <= 0.03},
     "H145c": {"E5": FH("E5"), "rel": FH("E5") / FH("E2") - 1, "supported": FH("E5") / FH("E2") - 1 <= -0.05},
     "H145d": {"order": order_ok, "supported": all(order_ok.values())},
     "H145e": {"vals": {n: FH(n) for n in VERS if n != "E5"}, "supported": all(1.73 - 1e-9 <= FH(n) <= 2.174 + 0.005 for n in VERS if n != "E5")},
     "H145f_descriptive": {"C_main": DC_main, "S_main": DS_main, "C_supp": DC_supp, "S_supp": DS_supp,
                           "median_C": float(np.median(DC_supp)), "median_S": float(np.median(DS_supp)), "mw_p_main": mw_main, "mw_p_supp": mw_supp}}
summ = {n: {vn.split("|")[0]: OUT[n][vn]["F_heavy"] for vn in MAIN3} for n in OUT}
summT = {n: {vn.split("|")[0]: OUT[n][vn]["F_total"] for vn in MAIN3} for n in OUT}
json.dump({"note": "F_heavy・V は生存確率や復興確率ではない", "inputs": {"DC_main": DC_main, "DC_supp": DC_supp, "DS_main": DS_main, "DS_supp": DS_supp, "VB": VB.tolist(), "seed_D": 20261026},
           "summary_F_heavy": summ, "summary_F_total": summT, "hyp": H, "checks": chk, "mean_Dc_U3": mean_Dc, "OUT": OUT, "runtime_s": time.time() - t0},
          open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
names = ["R0_B", "E1", "E2", "E3", "E7", "E8", "E6", "E5", "E9", "R0_A"]
fig, ax = plt.subplots(figsize=(8, 4.2)); x = np.arange(len(names))
for j, (vn, c) in enumerate(zip(MAIN3, ("C0", "C1", "C2"))):
    ax.bar(x + (j - 1) * 0.27, [FH(n, vn) for n in names], 0.27, color=c, label=vn.split("|")[0])
ax.axhline(1, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("F_heavy (slow kappa, p|lost 0.9)"); ax.legend(fontsize=7)
ax.set_title("133: explicit post-collapse downhill D in the T2 clock (not a survival probability)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_downhill.png"), dpi=120)
print(json.dumps({"summary": summ, "hyp": H}, ensure_ascii=False, default=float, indent=1)); print("runtime", round(time.time() - t0))
