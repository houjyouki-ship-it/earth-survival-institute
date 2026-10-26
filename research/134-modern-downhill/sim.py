"""研究134（133 の sim を流用）：現代の重い崩壊の下り坂 D_mod を Xia 2022 の 150 Tg の格子収量（nw_years.json）と大疫病の年数で置く（open_problems 311）。
元の説明：研究133：崩壊後の人口の下り坂 D を停滞（谷からの遅れ）と別に T2 の時計へ明示的に入れる（open_problems 309）。
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

R133 = json.load(open(os.path.join(RES, "133-explicit-downhill", "results.json"), encoding="utf8"))
NW = json.load(open(os.path.join(HERE, "nw_years.json"), encoding="utf8"))
D_nw = NW["years_below"]["0.95"]; D_nw_min = NW["argmin_year"]; D_nw_09 = NW["years_below"]["0.9"]
D_pan = [7, 2]  # Black Death 1346-1353、1918-1920 の流行（Wikipedia、sources/）
assert D_nw == 11 and D_nw_min == 3
OUT = {}; chk = {}
o = run(SB); OUT["R0_B"] = o
o1 = run(add(SB, Darr(DC_main))); OUT["R_E1"] = o1
for vn in MAIN3:
    chk[f"B|{vn}"] = abs(o[vn]["F_heavy"] - R133["OUT"]["R0_B"][vn]["F_heavy"])
    chk[f"E1|{vn}"] = abs(o1[vn]["F_heavy"] - R133["OUT"]["E1"][vn]["F_heavy"])
assert max(chk.values()) < 1e-12, chk
print("checks ok", chk, flush=True)
g2 = np.random.default_rng(20261027); U_C = g2.uniform(size=N)
def Dmix(acute, cC, nonstag=False):
    Dc = draw(DC_main, U_D); out = {}
    for k in COMPS:
        st = MASK[k] & (idx != 2)
        out[k] = np.where(st, acute + np.where(U_C < cC, Dc, 0.0), 0.0) + (np.where(~MASK[k], acute, 0.0) if nonstag else 0.0)
    return out
Dp = lambda: {k: np.where(MASK[k] & (idx != 2), draw(D_pan, U_D2), 0.0) for k in COMPS}
VERS = {"M1": Dmix(D_nw, 0), "M1_min3": Dmix(D_nw_min, 0), "M1_15": Dmix(15, 0), "M2_pan": Dp(),
        "M3_c05": Dmix(D_nw, 0.5), "M3_c025": Dmix(D_nw, 0.25), "M4_c1": Dmix(D_nw, 1.0), "M5_nonstag": Dmix(D_nw, 0, True)}
for name, D in VERS.items():
    OUT[name] = run(add(SB, D))
    print(name, {vn.split("|")[0]: round(OUT[name][vn]["F_heavy"], 4) for vn in MAIN3}, round(time.time() - t0), "s", flush=True)
FH = lambda n, vn="D2rz|none": OUT[n][vn]["F_heavy"]
b, e1 = FH("R0_B"), FH("R_E1"); mid = (b + e1) / 2
order_ok = {n: bool(FH(n) >= FH(n, "D1|main") >= FH(n, "D4|main")) for n in OUT}
H = {"H146a": {"D_nw": D_nw, "D_pan": D_pan, "supported": D_nw <= 20 and max(D_pan) <= 20},
     "H146b": {"M1": FH("M1"), "rel_B": FH("M1") / b - 1, "supported": 0 <= FH("M1") / b - 1 <= 0.03},
     "H146c": {"M3": FH("M3_c05"), "mid": mid, "rel": FH("M3_c05") / mid - 1, "supported": abs(FH("M3_c05") / mid - 1) <= 0.03},
     "H146d": {"order": order_ok, "supported": all(order_ok.values())}}
# c_C の分かれ目：F_heavy が中点に来る c_C（線形補間）
cs = [0, 0.25, 0.5, 1.0]; fs = [FH("M1"), FH("M3_c025"), FH("M3_c05"), FH("M4_c1")]
summ = {n: {vn.split("|")[0]: OUT[n][vn]["F_heavy"] for vn in MAIN3} for n in OUT}
summT = {n: {vn.split("|")[0]: OUT[n][vn]["F_total"] for vn in MAIN3} for n in OUT}
json.dump({"note": "F_heavy・V は生存確率や復興確率ではない", "inputs": {"nw": NW, "D_nw": D_nw, "D_pan": D_pan, "DC_main": DC_main, "seed_C": 20261027},
           "summary_F_heavy": summ, "summary_F_total": summT, "cC_curve": {"c": cs, "F": fs}, "hyp": H, "checks": chk, "OUT": OUT, "runtime_s": time.time() - t0},
          open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
yr = np.arange(1, 16); a1.plot(yr, NW["equal_mean_ratio"], "o-"); a1.axhline(0.95, ls="--", c="gray"); a1.set_xlabel("year after war (150 Tg)"); a1.set_ylabel("4-crop production ratio (proxy)"); a1.set_title("Xia2022 CLM5 150 Tg", fontsize=9)
names = ["R0_B", "M1_min3", "M1", "M1_15", "M2_pan", "M5_nonstag", "M3_c025", "M3_c05", "M4_c1", "R_E1"]; x = np.arange(len(names))
for j, (vn, c) in enumerate(zip(MAIN3, ("C0", "C1", "C2"))):
    a2.bar(x + (j - 1) * 0.27, [FH(n, vn) for n in names], 0.27, color=c, label=vn.split("|")[0])
a2.set_xticks(x); a2.set_xticklabels(names, fontsize=7, rotation=45); a2.set_ylabel("F_heavy (not a survival prob.)"); a2.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_modern_downhill.png"), dpi=120)
print(json.dumps({"summary": summ, "hyp": H, "cC": fs}, ensure_ascii=False, default=float, indent=1)); print("runtime", round(time.time() - t0))
