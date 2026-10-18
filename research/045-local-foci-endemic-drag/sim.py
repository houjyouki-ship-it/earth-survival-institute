"""研究045：地元に病巣がある小さな集団でのペストの減り（open_problems 130）と r の下限の止め（131）。
事前登録：PREREGISTRATION.md。044 の sim.py を「S = {"B1_main"」の手前まで exec し、038〜044 の乱数・関数をそのまま使う。
新しい乱数は seed 20261045。numpy＋matplotlib のみ、ネット接続なし。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
R = {"study": "045-local-foci-endemic-drag", "seed": 20261045}

# ================= A. 病巣の地域の減り（sources/excerpts.json の数値） =================
INC = {"median": 1.4e-5, "iqr_hi": 3.2e-5, "max": 63e-5}       # Madagascar 1998-2016、415 commune、/人/年
CFR_UNT = {"WHO_lo": 0.30, "WHO_hi": 0.60, "Uganda_untreated": 15 / 19}
A1 = {}
for ik, inc in INC.items():
    for ck, cfr in CFR_UNT.items():
        for k in (1, 3):
            A1[f"{ik}_{ck}_k{k}"] = inc * cfr * k
R["A1_endemic_drag"] = {"cases": A1, "min": min(A1.values()), "max": max(A1.values()),
                        "median_WHOlo_k1": A1["median_WHO_lo_k1"], "median_Uganda_k3": A1["median_Uganda_untreated_k3"]}
A2 = {}
for cfr in (0.3, 1.0):
    for h in (1.0, 0.5):
        A2[f"cfr{cfr}_h{h}"] = -np.log(1 - 4000 * cfr / (4.084e6 * h))
R["A2_madagascar_1932"] = {"cases": A2, "min": min(A2.values()), "max": max(A2.values())}

# ================= B. 044 の模型 =================
P44 = os.path.abspath(os.path.join(HERE, "..", "044-plague-geography-drag", "sim.py"))
src = open(P44, encoding="utf8").read()
G = {"__file__": P44, "__name__": "sim044"}
exec(compile(src[:src.index('S = {"B1_main"')], P44, "exec"), G)
for nm in ("T", "N", "lu", "SRC", "assign", "V_gen", "finish", "per_heavy", "weighted", "causes", "HEAVY", "r_F", "r_E", "AUS",
           "NG", "POP_W", "REF_U1", "DRAWS", "U8", "PRI", "un", "S_strict", "Nc43", "N_LO", "stage_times", "IND_LO", "IND_HI", "SWE"):
    globals()[nm] = G[nm]
assert G["R"]["check_043_B2"]["reproduced"]
r44 = json.load(open(os.path.join(HERE, "..", "044-plague-geography-drag", "results.json")))

g9 = np.random.default_rng(20261045)
U9 = g9.uniform(size=(2, N))
DEND = (1e-5, 2e-3)

def make_wait(n, rc, b, a, c, sat, floor):
    """floor=None：止めない（r≤0 なら先の段に届かない＝待ち無限）。sat：r(n)=b − a·n/(n+c)。"""
    if floor is not None:
        a = np.where(sat, np.minimum(a, b - floor), 0.0); rc = np.maximum(rc, floor)
    else:
        a = np.where(sat, a, 0.0)
    bma = b - a; cc = np.where(sat, c, 1.0)
    n_eq = np.where(bma < 0, b * cc / np.maximum(-bma, 1e-300), np.inf)   # r=0 になる大きさ
    def W(x):
        arg = np.maximum(bma * x + b * cc, 1e-300)
        safe = np.where(np.abs(bma) < 1e-12, 1e-12, bma)
        return np.log(x) / b + (a / b) / safe * np.log(arg)
    def wait(Nk):
        top = np.maximum(Nk, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            ws = W(top) - W(n)
            wc = np.where(rc > 0, np.log(top / n) / np.where(rc > 0, rc, 1.0), np.where(top > n, np.inf, 0.0))
        ws = np.where((top >= n_eq) & (top > n), np.inf, ws)
        return np.where(sat, ws, wc)
    return wait

def value(d, wait_args, floor):
    sv = d["sv"]; st = assign(sv, 1.0)
    if d["ext"] > 0: st = np.where(d["uext"] < d["ext"], 0, st)
    n = np.maximum(sv["nmax"], 1.0)
    T2s = stage_times(st, n, make_wait(n, *wait_args, floor), d["lit"])
    return {"V0": V_gen(st, T2s, d["sc"], 0.3), "V2": V_gen(st, T2s, d["sc"], 0.3, tablet=True), "st": st}

def args45(k, cfg, info=None):
    """cfg['model']='044'：地元の減り D_max 一定（044 主）。'045'：d_loc(n)=D_end + D_max·n/(n+N_loc)。"""
    Xa, Xo, _ = causes(k, {})
    phi = np.full(N, cfg["phi"]) if "phi" in cfg else un(U8[0], PRI["phi"])
    p_loc = phi * S_strict
    if k == "NUC_G": p_loc = np.where(AUS, 0.0, p_loc)
    Xloc = (U8[1] < p_loc) & ~Xa
    pl = np.full(N, cfg["plink"]) if "plink" in cfg else un(U8[2], PRI["plink"])
    Xlink = (U8[3] < pl) & ~Xloc & ~Xa
    lam = np.full(N, cfg["lam"]) if "lam" in cfg else un(U8[4], PRI["lam"])
    m = np.full(N, cfg["m"]) if "m" in cfg else un(U8[5], PRI["m"])
    Dmax = lam * -np.log(1 - m)
    rho = np.full(N, cfg["rho"]) if "rho" in cfg else un(U8[6], PRI["rho"])
    Nc = np.full(N, cfg["Nc"]) if "Nc" in cfg else Nc43
    b = r_F - Xo * rho * (r_F - r_E)
    if cfg.get("model", "045") == "044":
        rc = np.where(Xa, r_E, np.where(Xloc, b - Dmax, b)); sat = Xlink
        bb = b; aa = Dmax; cc = Nc
    else:
        Dend = np.full(N, cfg["Dend"]) if "Dend" in cfg else lu(U9[0], *DEND)
        Nloc = Nc * cfg.get("Nloc_mult", 1.0)
        rc = np.where(Xa, r_E, b); sat = Xloc | Xlink
        bb = np.where(Xloc, b - Dend, b); aa = Dmax; cc = np.where(Xloc, Nloc, Nc)
    if info is not None:
        info["P_local"] = float(Xloc.mean()); info["P_link_only"] = float(Xlink.mean())
        info["nonpositive_r_inf"] = float((~sat & (rc <= 0)).mean() + (sat & (bb - aa <= 0)).mean())
        info["below_floor_r_inf"] = float((~sat & (rc < 0.0005)).mean() + (sat & (bb - aa < 0.0005)).mean())
    return (rc, bb, aa, cc, sat)

def evaluate(cfg=None, keys=("U3", "CASC", "BIO")):
    cfg = cfg or {}; floor = cfg.get("floor", 0.0005); A = {"BASE": {}}; INFO = {}
    for k in SRC:
        inf_ = {}
        A["BASE"][k] = value(SRC[k], args45(k, cfg, inf_), floor)
        INFO[k] = inf_
    D, hv = finish(A)
    return D, hv, per_heavy(D, hv, keys=keys), INFO

def summ(cfg=None):
    D, hv, ph, info = evaluate(cfg)
    W = weighted(D, hv, anchors=("A_Ord", "A_XPT_S"))
    return {"U3_TAB": ph["U3"]["TAB"], "named_TAB": ph["named_pooled"]["TAB"], "CASC_TAB": ph["CASC"]["TAB"], "BIO_TAB": ph["BIO"]["TAB"],
            "w_AOrd_U3_TAB": W["A_Ord"]["U3"]["TAB"]["total"], "w_AXPTS_U2_TAB": W["A_XPT_S"]["U2"]["TAB"]["total"], "info_U3": info["U3"]}

# 確認：044 の模型を新しい関数で再現
chk = summ({"model": "044"})
ref = r44["B"]["B1_main"]["U3_TAB"]
R["check_044_main"] = {"U3_TAB": chk["U3_TAB"], "ref": ref, "rel_err": abs(chk["U3_TAB"] - ref) / ref, "reproduced": bool(abs(chk["U3_TAB"] - ref) / ref < 1e-9)}

S = {"B_044": chk, "B_main": summ(), "B_Nloc_div10": summ({"Nloc_mult": 0.1}), "B_Dend0": summ({"Dend": 0.0}),
     "B_Dend_hi": summ({"Dend": DEND[1]}), "B_Dend_hi_Nloc_div10": summ({"Dend": DEND[1], "Nloc_mult": 0.1})}
R["B"] = S

# ================= C. 止めの感度（131） =================
C = {}
for mdl in ("044", "045"):
    for fl in (1e-4, 5e-4, 2e-3, None):
        C[f"{mdl}_floor_{fl}"] = summ({"model": mdl, "floor": fl})["U3_TAB"]
R["C_floor"] = C

# ================= トルネード（新しい模型、U3 の TAB） =================
def u3(cfg): return evaluate(cfg, keys=("U3",))[2]["U3"]["TAB"]
TOR = {
 "phi (0.1/0.6)": {"lo": u3({"phi": 0.1}), "hi": u3({"phi": 0.6})},
 "p_link (0/1)": {"lo": u3({"plink": 0.0}), "hi": u3({"plink": 1.0})},
 "rho (0.16/0.57)": {"lo": u3({"rho": 0.16}), "hi": u3({"rho": 0.57})},
 "D_max (lam,m lo/hi)": {"lo": u3({"lam": 0.061, "m": 0.10}), "hi": u3({"lam": 0.118, "m": 0.20})},
 "N_c (N_lo/1e7)": {"lo": u3({"Nc": N_LO}), "hi": u3({"Nc": 1e7})},
 "D_end (1e-5/2e-3)": {"lo": u3({"Dend": DEND[0]}), "hi": u3({"Dend": DEND[1]})},
 "N_loc mult (0.1/1)": {"lo": u3({"Nloc_mult": 0.1}), "hi": u3({"Nloc_mult": 1.0})},
}
for v in TOR.values(): v["range"] = abs(v["hi"] - v["lo"])
R["tornado_U3_TAB"] = TOR

# ================= n* =================
def sweep(cfg=None):
    cfg = cfg or {}; sc = DRAWS["GEN"]; rows = []; floor = cfg.get("floor", 0.0005)
    for n_ in NG:
        sv = {"s": np.full(N, n_ / POP_W), "ntot": np.full(N, n_), "nmax": np.full(N, n_), "mcap": np.ones(N)}
        st = assign(sv, 1.0); a_ = args45("U3", cfg)
        T2s = stage_times(st, np.full(N, n_), make_wait(np.full(N, n_), *a_, floor), 0.3)
        rows.append({"n": float(n_), "R_tab": float((V_gen(st, T2s, sc, 0.3, tablet=True) - V_gen(st, T2s, sc, 0.3)).mean())})
    v = np.array([r["R_tab"] for r in rows]); idx = np.where(v >= 0.5 * REF_U1)[0]; ns_ = None
    if idx.size:
        i = int(idx[0])
        if i == 0: ns_ = float(NG[0])
        else:
            x0, x1 = np.log10(NG[i - 1]), np.log10(NG[i]); ns_ = float(10 ** (x0 + (0.5 * REF_U1 - v[i - 1]) / (v[i] - v[i - 1]) * (x1 - x0)))
    return {"rows": rows, "n_star": ns_}
SW44 = sweep({"model": "044"}); SW = sweep(); SWd = sweep({"Nloc_mult": 0.1})
R["n_star"] = {"044_via045code": SW44["n_star"], "044_ref": r44["n_star"]["main"], "main": SW["n_star"], "Nloc_div10": SWd["n_star"]}
R["n_star_rows"] = {"044": SW44["rows"], "main": SW["rows"]}

# ================= 仮説 =================
m1 = S["B_main"]; m44 = r44["B"]["B1_main"]; tmax = max(TOR, key=lambda k: TOR[k]["range"])
fl_ok = all(abs(C[f"{md}_floor_{fl}"] / C[f"{md}_floor_0.0005"] - 1) <= 0.10 for md in ("044", "045") for fl in (1e-4, 2e-3, None))
R["hypotheses"] = {
 "H57a": {"value": R["A2_madagascar_1932"]["max"], "rule": "Madagascar 1932 drag max < 0.0025/yr", "supported": bool(R["A2_madagascar_1932"]["max"] < 0.0025)},
 "H57b": {"value": A1["max_Uganda_untreated_k3"], "rule": "endemic max < 0.002/yr", "supported": bool(A1["max_Uganda_untreated_k3"] < 0.002)},
 "H57c": {"value": [m1["U3_TAB"], m44["U3_TAB"], m1["U3_TAB"] / m44["U3_TAB"]], "rule": "U3 TAB >= 1.10 x 044 main", "supported": bool(m1["U3_TAB"] >= 1.10 * m44["U3_TAB"])},
 "H57d": {"value": [tmax, TOR[tmax]["range"]], "rule": "largest tornado range is N_c", "supported": bool(tmax.startswith("N_c"))},
 "H57e": {"value": [m1["w_AOrd_U3_TAB"], m44["w_AOrd_U3_TAB"], m1["w_AOrd_U3_TAB"] / m44["w_AOrd_U3_TAB"]], "rule": "weighted A_Ord U3 within +-30% of 044", "supported": bool(0.7 <= m1["w_AOrd_U3_TAB"] / m44["w_AOrd_U3_TAB"] <= 1.3)},
 "H57f": {"value": C, "rule": "all floor variants within +-10% of floor=5e-4, both models", "supported": bool(fl_ok)},
 "H57g": {"value": [SW["n_star"], r44["n_star"]["main"]], "rule": "n* within +-10% of 044", "supported": bool(SW["n_star"] is not None and abs(SW["n_star"] / r44["n_star"]["main"] - 1) <= 0.10)},
}

# ================= 図 =================
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
labs = ["endemic median\n(WHO lo, k=1)", "endemic median\n(79%, k=3)", "endemic max\n(79%, k=3)", "Madagascar 1932\n(min)", "Madagascar 1932\n(max)",
        "India 3rd\n(lo)", "India 3rd\n(hi)", "D_max 044\n(lo)", "D_max 044\n(hi)"]
Dlo = 0.061 * -np.log(0.9); Dhi = 0.118 * -np.log(0.8)
vals = [A1["median_WHO_lo_k1"], A1["median_Uganda_untreated_k3"], A1["max_Uganda_untreated_k3"], R["A2_madagascar_1932"]["min"], R["A2_madagascar_1932"]["max"],
        IND_LO, IND_HI, Dlo, Dhi]
ax[0].bar(range(len(vals)), [100 * v for v in vals], color=["C2"] * 3 + ["C1"] * 4 + ["C3"] * 2)
ax[0].axhline(100 * (0.0228), color="k", ls="--", label="041: r_F - r_E = 2.28 pt")
ax[0].set_yscale("log"); ax[0].set_xticks(range(len(vals))); ax[0].set_xticklabels(labs, fontsize=7, rotation=35, ha="right")
ax[0].set_ylabel("population growth lost to plague (%/yr)"); ax[0].legend(fontsize=8)
ax[0].set_title("A: plague drag in places WITH local foci (green: modern endemic, orange: 3rd pandemic)\nvs 2nd-pandemic Europe (red, used by 044 for local foci)", fontsize=8)
names = list(TOR)
for i, k in enumerate(names):
    ax[1].plot([TOR[k]["lo"], TOR[k]["hi"]], [i, i], "k-", lw=6, alpha=0.3); ax[1].plot(TOR[k]["lo"], i, "bo"); ax[1].plot(TOR[k]["hi"], i, "ro")
ax[1].axvline(m1["U3_TAB"], color="k", ls=":", label="045 main"); ax[1].axvline(m44["U3_TAB"], color="C0", ls="--", label="044 main")
ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names, fontsize=8)
ax[1].set_xlabel("tablet-guide value per U3 heavy event (blue=lo, red=hi)"); ax[1].legend(fontsize=8); ax[1].set_title("tornado (045 model)", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_local_drag_tornado.png"), dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
xs = ["1e-4", "5e-4", "2e-3", "none"]
for mdl, sty in (("044", "C0o-"), ("045", "C3s-")):
    ax[0].plot(xs, [C[f"{mdl}_floor_{fl}"] for fl in (1e-4, 5e-4, 2e-3, None)], sty, label=f"{mdl} model")
ax[0].set_xlabel("floor on post-collapse growth rate r (none = r<=0 never reaches next stage)"); ax[0].set_ylabel("U3 tablet value per heavy event")
ax[0].legend(fontsize=8); ax[0].set_title("C: sensitivity to the r floor (open problem 131)", fontsize=9)
for rows, lab, sty in ((SW44["rows"], "044 model", "C0--"), (SW["rows"], "045 main", "k-"), (SWd["rows"], "045 N_loc=N_c/10", "C3--")):
    ax[1].semilogx([r["n"] for r in rows], [r["R_tab"] for r in rows], sty, label=lab)
ax[1].axhline(0.5 * REF_U1, color="gray", ls=":", label="half of U1 value")
ax[1].set_xlabel("single connected surviving group n"); ax[1].set_ylabel("tablet-guide value per event"); ax[1].legend(fontsize=8)
ax[1].set_title("n* under the 045 local-foci model", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_floor_nstar.png"), dpi=120); plt.close()

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=1, default=float)
out = {k: R[k] for k in R if k != "n_star_rows"}
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
print("runtime", round(R["runtime_seconds"], 1))
