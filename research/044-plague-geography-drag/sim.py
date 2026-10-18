"""研究044：病巣の地理 f_foci、交易のつながり、再来の減りの分解（open_problems 126・128・129）。
事前登録：PREREGISTRATION.md。042 の sim.py を 043 と同じ切り方で exec し、038〜042 の乱数・関数をそのまま使う。
新しい乱数は seed 20261044。numpy＋matplotlib のみ、ネット接続なし（人口は sources/pop_by_country.json）。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
R = {"study": "044-plague-geography-drag", "seed": 20261044}

# ================= A. データ =================
POP = json.load(open(os.path.join(HERE, "sources", "pop_by_country.json"), encoding="utf8"))
TOT = float(sum(POP.values()))
# 厳しい一覧：Mead 2011 Box 41.1 と PMC10438918 が「今ある病巣」として名前を挙げた国
STRICT = ["United States", "Brazil", "Bolivia", "Peru", "Ecuador",
          "Democratic Republic of the Congo", "Uganda", "Kenya", "Tanzania", "Zambia", "Zimbabwe", "Mozambique", "Botswana",
          "South Africa", "Namibia", "Angola", "Madagascar", "Libya", "Algeria",
          "Iran", "Russia", "Kazakhstan", "China", "Mongolia", "India", "Myanmar", "Vietnam", "Indonesia"]
# 広い一覧：隣接（Canada, Mexico）、過去の病巣（Paraguay, Argentina, Venezuela）、Transcaucasia、Kurdistan（Iraq）、中央アジア
BROAD_ADD = ["Canada", "Mexico", "Paraguay", "Argentina", "Venezuela", "Azerbaijan", "Armenia", "Georgia", "Iraq",
             "Kyrgyzstan", "Uzbekistan", "Turkmenistan", "Tajikistan"]
for c in STRICT + BROAD_ADD: assert c in POP, c
S_strict = sum(POP[c] for c in STRICT) / TOT
S_broad = sum(POP[c] for c in STRICT + BROAD_ADD) / TOT
top = sorted(((POP[c] / TOT, c) for c in STRICT), reverse=True)[:6]
R["A1_foci_share"] = {"S_strict": S_strict, "S_broad": S_broad, "world_listed": TOT, "n_strict": len(STRICT), "n_broad": len(STRICT) + len(BROAD_ADD),
                      "top6_strict": [[c, s] for s, c in top], "S_strict_without_China_India": S_strict - (POP["China"] + POP["India"]) / TOT,
                      "f_foci_042_mean": 0.725}

# A2：India の第3次パンデミックの国全体の減り
IND = {}
for D_ in (10e6, 12e6):
    for P_ in (255e6, 300e6):
        for T_ in (25, 63):
            IND[f"D{D_/1e6:.0f}M_P{P_/1e6:.0f}M_T{T_}"] = -np.log(1 - D_ / P_) / T_
IND_LO, IND_HI = min(IND.values()), max(IND.values())
SWE = 1 / (363 / 22) * -np.log(1 - 0.15)   # 043 の Sweden（22回/363年、m=0.15）
R["A2_india_drag"] = {"cases": IND, "lo": IND_LO, "hi": IND_HI, "sweden_043_m0.15": SWE, "hi_over_sweden": IND_HI / SWE}

# ================= B. 042 の模型 =================
P42 = os.path.abspath(os.path.join(HERE, "..", "042-scenario-qF-plague", "sim.py"))
src = open(P42, encoding="utf8").read()
G = {"__file__": P42, "__name__": "sim042"}
exec(compile(src[:src.index("D, HV, PH, INFO = evaluate()")], P42, "exec"), G)
T = G["T"]; N = G["N"]; lu = G["lu"]; U7 = G["U7"]; SRC = G["SRC"]; assign = G["assign"]; V_gen = G["V_gen"]
finish = G["finish"]; per_heavy = G["per_heavy"]; weighted = G["weighted"]; causes = G["causes"]; HEAVY = G["HEAVY"]
r_F = G["r_F"]; r_E = G["r_pop"]; AUS = G["AUS"]; NG = G["NG"]; POP_W = G["POP_W"]; REF_U1 = G["REF_U1"]
DRAWS = G["G"]["DRAWS"]
r43 = json.load(open(os.path.join(HERE, "..", "043-plague-reintroduction-Nurb", "results.json")))
N_LO = r43["A4_Nc_bound"]["N_lo_conservative"]
FLOOR = 0.0005

def stage_times(st, n, wait, lit):
    d01e = np.where(T["U_lit"] < lit, T["d01"] * T["lit_mult"], T["d01"])
    tFe = np.where(st == 6, np.maximum(T["d_Fe"], wait(T["N_Fe"])), 0.0)
    t1 = np.where(st >= 5, np.maximum(tFe + d01e, wait(T["N_lens"])), 0.0)
    t2 = np.where(st >= 4, np.maximum(t1 + T["d12"], wait(T["N_T2e"])), 0.0)
    return np.select([st == 2, st == 3, st >= 4], [T["W"], np.zeros(N), t2], default=np.zeros(N))

def make_wait(n, rc, b, a, c, sat):
    """sat=False：一定の r=rc。sat=True：r(n)=b − a·n/(n+c)（b−a ≥ FLOOR に止めてある）。"""
    a = np.where(sat, np.minimum(a, b - FLOOR), 0.0); bma = b - a
    cc = np.where(sat, c, 1.0)
    def W(x): return np.log(x) / b + (a / b) / bma * np.log(bma * x + b * cc)
    def wait(Nk):
        top = np.maximum(Nk, n)
        return np.where(sat, W(top) - W(n), np.log(top / n) / rc)
    return wait

def value(d, wait_args):
    sv = d["sv"]; st = assign(sv, 1.0)
    if d["ext"] > 0: st = np.where(d["uext"] < d["ext"], 0, st)
    n = np.maximum(sv["nmax"], 1.0)
    T2s = stage_times(st, n, make_wait(n, *wait_args), d["lit"])
    return {"V0": V_gen(st, T2s, d["sc"], 0.3), "V2": V_gen(st, T2s, d["sc"], 0.3, tablet=True), "st": st}

# ---- B0：043 の連続版（B2）の再現 ----
Nc43 = lu(U7[6], N_LO, 1e7)
def legacy_args(k):
    Xa, Xo, Xp = causes(k, {})
    Nb = np.where(Xo | Xa, 0.0, np.where(Xp, Nc43, np.inf))
    sat = np.isfinite(Nb) & (Nb > 0)
    rc = np.where(Nb == 0, r_E, r_F)
    return (rc, r_F + 0 * rc, r_F - r_E, np.where(sat, Nb, 1.0), sat)
A = {"BASE": {k: value(SRC[k], legacy_args(k)) for k in SRC}}
D0, H0 = finish(A); P0 = per_heavy(D0, H0, keys=("U3", "CASC", "BIO"))
ref43 = r43["B"]["B2_Nlo_soft"]["U3_TAB"]
R["check_043_B2"] = {"U3_TAB": P0["U3"]["TAB"], "ref": ref43, "rel_err": abs(P0["U3"]["TAB"] - ref43) / ref43,
                     "reproduced": bool(abs(P0["U3"]["TAB"] - ref43) / ref43 < 1e-9)}

# ---- B1：新しい模型 ----
g8 = np.random.default_rng(20261044)
U8 = g8.uniform(size=(8, N))
PRI = {"phi": (0.1, 0.6), "plink": (0.0, 1.0), "lam": (0.061, 0.118), "m": (0.10, 0.20), "rho": (0.16, 0.57)}
def un(u, lohi): return lohi[0] + (lohi[1] - lohi[0]) * u

def new_args(k, cfg, info=None):
    Xa, Xo, _ = causes(k, {})
    phi = np.full(N, cfg["phi"]) if "phi" in cfg else un(U8[0], PRI["phi"])
    Sc = cfg.get("Sc", S_strict)
    p_loc = phi * Sc
    if k == "NUC_G": p_loc = np.where(AUS, 0.0, p_loc)
    Xloc = U8[1] < p_loc
    pl = np.full(N, cfg["plink"]) if "plink" in cfg else un(U8[2], PRI["plink"])
    Xlink = (U8[3] < pl) & ~Xloc
    lam = np.full(N, cfg["lam"]) if "lam" in cfg else un(U8[4], PRI["lam"])
    m = np.full(N, cfg["m"]) if "m" in cfg else un(U8[5], PRI["m"])
    Dmax = lam * -np.log(1 - m)
    rho = np.full(N, cfg["rho"]) if "rho" in cfg else un(U8[6], PRI["rho"])
    Nc = np.full(N, cfg["Nc"]) if "Nc" in cfg else Nc43
    Dloc = lu(U8[7], IND_LO, IND_HI) if cfg.get("local") == "india" else Dmax
    if cfg.get("no_rho_fix"): rho = np.ones(N)
    b = r_F - Xo * rho * (r_F - r_E)
    rc = np.where(Xa, r_E, np.where(Xloc, b - Dloc, b))
    sat = ~Xa & ~Xloc & Xlink
    if info is not None:
        info["floor_share"] = float(((rc < FLOOR) & ~sat).mean() + (sat & (b - Dmax < FLOOR)).mean())
        info["P_local"] = float((Xloc & ~Xa).mean()); info["P_link_only"] = float(sat.mean()); info["P_agent"] = float(Xa.mean())
    return (np.maximum(rc, FLOOR), b, Dmax, Nc, sat), Xloc, Xlink, Xa

def evaluate(cfg=None, keys=("U3", "CASC", "BIO")):
    cfg = cfg or {}; A = {"BASE": {}}; INFO = {}
    for k in SRC:
        inf_ = {}
        args, Xloc, Xlink, Xa = new_args(k, cfg, inf_)
        A["BASE"][k] = value(SRC[k], args)
        hv = np.isin(A["BASE"][k]["st"], HEAVY)
        if hv.any():
            inf_["heavy_share"] = float(hv.mean())
            inf_["plague_any_heavy"] = float(((Xloc | Xlink) & ~Xa)[hv].mean())
            inf_["plague_local_heavy"] = float((Xloc & ~Xa)[hv].mean())
        INFO[k] = inf_
    D, hv = finish(A)
    return D, hv, per_heavy(D, hv, keys=keys), INFO

def summ(cfg=None):
    D, hv, ph, info = evaluate(cfg)
    W = weighted(D, hv, anchors=("A_Ord", "A_XPT_S"))
    return {"U3_TAB": ph["U3"]["TAB"], "named_TAB": ph["named_pooled"]["TAB"], "CASC_TAB": ph["CASC"]["TAB"], "BIO_TAB": ph["BIO"]["TAB"],
            "w_AOrd_U3_TAB": W["A_Ord"]["U3"]["TAB"]["total"], "w_AXPTS_U2_TAB": W["A_XPT_S"]["U2"]["TAB"]["total"],
            "info_U3": info["U3"], "info_BIO": info["BIO"], "info_CASC": info["CASC"]}

S = {"B1_main": summ(), "B1_broad": summ({"Sc": S_broad}), "B1_local_india": summ({"local": "india"}),
     "B1_no_rho_fix": summ({"no_rho_fix": True}), "B1_link1": summ({"plink": 1.0}), "B1_link0": summ({"plink": 0.0})}
R["B"] = S
b43 = r43["B"]["B1_Nlo_threshold"]
R["compare_043"] = {"B1_U3_TAB": b43["U3_TAB"], "B2_soft_U3_TAB": ref43, "w_AOrd_U3_TAB": b43["w_AOrd_U3_TAB"], "w_AXPTS_U2_TAB": b43["w_AXPTS_U2_TAB"],
                    "n_star_B1": r43["n_star"]["B1"], "qF_eff_U3_B1": b43["qF_eff_U3"]}

# トルネード（U3 の TAB）
def u3(cfg): return evaluate(cfg, keys=("U3",))[2]["U3"]["TAB"]
TOR = {
 "phi (0.1/0.6)": {"lo": u3({"phi": 0.1}), "hi": u3({"phi": 0.6})},
 "p_link (0/1)": {"lo": u3({"plink": 0.0}), "hi": u3({"plink": 1.0})},
 "rho (0.16/0.57)": {"lo": u3({"rho": 0.16}), "hi": u3({"rho": 0.57})},
 "D_max (lam,m lo/hi)": {"lo": u3({"lam": 0.061, "m": 0.10}), "hi": u3({"lam": 0.118, "m": 0.20})},
 "N_c (N_lo/1e7)": {"lo": u3({"Nc": N_LO}), "hi": u3({"Nc": 1e7})},
 "S_c (strict/broad)": {"lo": u3({"Sc": S_strict}), "hi": u3({"Sc": S_broad})},
 "local drag (D_max/India)": {"lo": u3({}), "hi": u3({"local": "india"})},
}
for v in TOR.values(): v["range"] = abs(v["hi"] - v["lo"])
R["tornado_U3_TAB"] = TOR

# n*（U3 の事前、単一集団）
def sweep(cfg=None):
    cfg = cfg or {}; sc = DRAWS["GEN"]; rows = []
    for n_ in NG:
        sv = {"s": np.full(N, n_ / POP_W), "ntot": np.full(N, n_), "nmax": np.full(N, n_), "mcap": np.ones(N)}
        st = assign(sv, 1.0); args = new_args("U3", cfg)[0]
        T2s = stage_times(st, np.full(N, n_), make_wait(np.full(N, n_), *args), 0.3)
        rows.append({"n": float(n_), "R_tab": float((V_gen(st, T2s, sc, 0.3, tablet=True) - V_gen(st, T2s, sc, 0.3)).mean())})
    v = np.array([r["R_tab"] for r in rows]); idx = np.where(v >= 0.5 * REF_U1)[0]; ns_ = None
    if idx.size:
        i = int(idx[0])
        if i == 0: ns_ = float(NG[0])
        else:
            x0, x1 = np.log10(NG[i - 1]), np.log10(NG[i]); ns_ = float(10 ** (x0 + (0.5 * REF_U1 - v[i - 1]) / (v[i] - v[i - 1]) * (x1 - x0)))
    return {"rows": rows, "n_star": ns_}
SW = sweep(); SWl0 = sweep({"plink": 0.0}); SWl1 = sweep({"plink": 1.0})
R["n_star"] = {"main": SW["n_star"], "plink0": SWl0["n_star"], "plink1": SWl1["n_star"], "043_B1": r43["n_star"]["B1"]}
R["n_star_rows_main"] = SW["rows"]

# ================= 仮説 =================
m1 = S["B1_main"]; tmax = max(TOR, key=lambda k: TOR[k]["range"])
R["hypotheses"] = {
 "H56a": {"value": S_strict, "rule": "S_strict < 0.725", "supported": bool(S_strict < 0.725)},
 "H56b": {"value": [IND_HI, SWE], "rule": "India drag hi < 0.5 x Sweden drag", "supported": bool(IND_HI < 0.5 * SWE)},
 "H56c": {"value": [m1["U3_TAB"], b43["U3_TAB"], m1["U3_TAB"] / b43["U3_TAB"]], "rule": "U3 TAB >= 1.15 x 043 B1", "supported": bool(m1["U3_TAB"] >= 1.15 * b43["U3_TAB"])},
 "H56d": {"value": [tmax, TOR[tmax]["range"]], "rule": "largest tornado range is p_link", "supported": bool(tmax.startswith("p_link"))},
 "H56e": {"value": [m1["w_AOrd_U3_TAB"], b43["w_AOrd_U3_TAB"], m1["w_AOrd_U3_TAB"] / b43["w_AOrd_U3_TAB"]], "rule": "weighted A_Ord U3 within +-30% of 043 B1", "supported": bool(0.7 <= m1["w_AOrd_U3_TAB"] / b43["w_AOrd_U3_TAB"] <= 1.3)},
 "H56f": {"value": [SW["n_star"], r43["n_star"]["B1"]], "rule": "n* <= 043 B1", "supported": bool(SW["n_star"] is not None and SW["n_star"] <= r43["n_star"]["B1"])},
}

# ================= 図 =================
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
labs = [c for _, c in top] + ["other strict", "broad add", "no listed foci"]
vals = [s for s, _ in top] + [S_strict - sum(s for s, _ in top), S_broad - S_strict, 1 - S_broad]
ax[0].barh(range(len(labs)), vals, color=["C3"] * 7 + ["C1", "C0"])
ax[0].set_yticks(range(len(labs))); ax[0].set_yticklabels(labs, fontsize=8); ax[0].invert_yaxis()
ax[0].set_xlabel("share of world population (listed countries)")
ax[0].set_title(f"A1: countries with natural plague foci\nstrict {S_strict:.2f}, broad {S_broad:.2f} (042 f_foci mean 0.725)", fontsize=9)
ks = ["India 3rd pandemic (lo)", "India (hi)", "Sweden 1350-1713", "England"]
vv = [IND_LO, IND_HI, SWE, (1 / 8.5) * -np.log(1 - 0.15)]
ax[1].bar(range(4), [100 * x for x in vv], color=["C2", "C2", "C1", "C1"])
ax[1].axhline(2.28, color="k", ls="--", label="041: r_F - r_E = 2.28 pt")
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(ks, fontsize=8); ax[1].set_ylabel("population growth lost to plague (%/yr)")
ax[1].set_yscale("log"); ax[1].legend(fontsize=8); ax[1].set_title("A2: national plague drag, 2nd vs 3rd pandemic", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_foci_drag.png"), dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
names = list(TOR)
for i, k in enumerate(names):
    ax[0].plot([TOR[k]["lo"], TOR[k]["hi"]], [i, i], "k-", lw=6, alpha=0.3); ax[0].plot(TOR[k]["lo"], i, "bo"); ax[0].plot(TOR[k]["hi"], i, "ro")
ax[0].axvline(m1["U3_TAB"], color="k", ls=":", label="044 main"); ax[0].axvline(b43["U3_TAB"], color="C0", ls="--", label="043 B1")
ax[0].set_yticks(range(len(names))); ax[0].set_yticklabels(names, fontsize=8)
ax[0].set_xlabel("tablet-guide value per U3 heavy event (blue=lo, red=hi)"); ax[0].legend(fontsize=8); ax[0].set_title("tornado (044 model)", fontsize=9)
rows43 = r43.get("n_star_rows_B1")
for rows, lab, sty in ((SW["rows"], "044 main", "k-"), (SWl0["rows"], "044 p_link=0", "C2--"), (SWl1["rows"], "044 p_link=1", "C3--")):
    ax[1].semilogx([r["n"] for r in rows], [r["R_tab"] for r in rows], sty, label=lab)
ax[1].axhline(0.5 * REF_U1, color="gray", ls=":", label="half of U1 value")
ax[1].set_xlabel("single connected surviving group n"); ax[1].set_ylabel("tablet-guide value per event"); ax[1].legend(fontsize=8)
ax[1].set_title("n* under geography + trade-link model", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_tornado_nstar.png"), dpi=120); plt.close()

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=1, default=float)
out = {k: R[k] for k in R if k != "n_star_rows_main"}
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
print("runtime", round(R["runtime_seconds"], 1))
