"""研究042：崩壊後の集団は疫病の繰り返す環境（E）か開拓地の環境（F）か（q_F、open_problems 123）と、
疫病を止める知識の効き（p_conv、124 の一部）。
事前登録：PREREGISTRATION.md。040 の sim.py を 041 と同じ切り方で exec し、038〜040 の乱数・関数をそのまま使う。
新しい乱数は seed 20261042（041 の再現用に 20261041 も同じ順で引く）。numpy＋matplotlib のみ、ネット接続なし。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import lgamma, exp, log

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
R = {"study": "042-scenario-qF-plague", "seed": 20261042}

# ---------- 1. 040 の模型を読み込む（041 と同じ切り方） ----------
P40 = os.path.abspath(os.path.join(HERE, "..", "040-cohesion-vs-knowledge", "sim.py"))
src = open(P40, encoding="utf8").read()
G = {"__file__": P40, "__name__": "sim040"}
exec(compile(src[:src.index("# ---------- 感度")], P40, "exec"), G)
exec(compile(src[src.index("# ---------- n* の走査"):src.index("SW = {")], P40, "exec"), G)
N = G["N"]; SRC = G["SRC"]; r_pop = G["r_pop"]; p_grow = G["p_grow"]; U5 = G["U5"]; lu = G["lu"]
assign = G["assign"]; V_gen = G["V_gen"]; fixed_src = G["fixed_src"]; weighted = G["weighted"]
per_heavy = G["per_heavy"]; HEAVY = G["HEAVY"]; POP_W = G["POP_W"]; NG = G["NG"]; REF_U1 = G["REF_U1"]
T = G["times"].__globals__            # 038 の名前空間（U_lit, d01, lit_mult, d_Fe, d12, W, N_Fe, N_lens, N_T2e, U_sv）
r41 = json.load(open(os.path.join(HERE, "..", "041-postcollapse-growth-rate", "results.json")))

# ---------- 2. 二段の人口待ち ----------
def times2(st, sv, lit, rF, rE, Nurb):
    """N_urb 未満は rF、以上は rE で増える。Nurb=0 で常に E、inf で常に F（038 の times と同じ段の式）。"""
    n = np.maximum(sv["nmax"], 1.0)
    def wait(Nk):
        top = np.maximum(Nk, n); a = np.clip(Nurb, n, top)
        return np.log(a / n) / rF + np.log(top / a) / rE
    d01e = np.where(T["U_lit"] < lit, T["d01"] * T["lit_mult"], T["d01"])
    tFe = np.where(st == 6, np.maximum(T["d_Fe"], wait(T["N_Fe"])), 0.0)
    t1 = np.where(st >= 5, np.maximum(tFe + d01e, wait(T["N_lens"])), 0.0)
    t2 = np.where(st >= 4, np.maximum(t1 + T["d12"], wait(T["N_T2e"])), 0.0)
    return np.select([st == 2, st == 3, st >= 4], [T["W"], np.zeros(N), t2], default=np.zeros(N)), wait(T["N_T2e"]), np.log(np.clip(Nurb, n, np.maximum(T["N_T2e"], n)) / n) / rF

def run_src(d, rF, Nurb):
    sv = d["sv"]; st = assign(sv, 1.0)
    if d["ext"] > 0: st = np.where(d["uext"] < d["ext"], 0, st)
    T2s, wtot, wF = times2(st, sv, d["lit"], rF, r_pop, Nurb)
    return {"V0": V_gen(st, T2s, d["sc"], 0.3), "V2": V_gen(st, T2s, d["sc"], 0.3, tablet=True), "st": st, "wtot": wtot, "wF": wF}

def finish(A):
    for iv in A:
        for k in ("AI", "GEN"): A[iv][k] = fixed_src(k, "BASE")
    B = A["BASE"]; D = {"TAB": {k: B[k]["V2"] - B[k]["V0"] for k in B}}
    for iv in A:
        if iv != "BASE": D[iv] = {k: A[iv][k]["V2"] - B[k]["V2"] for k in B}
    hv = {k: np.isin(B[k]["st"], HEAVY).astype(float) for k in B}
    return D, hv

# ---------- 3. 041 の再現（同じ乱数の順で引く） ----------
g6 = np.random.default_rng(20261041)
U6 = g6.uniform(size=(5, N)); r_F = lu(U6[0], 0.017, 0.028); dK = g6.uniform(0.0, 0.010, N); qF41 = g6.uniform(0, 1, N)
isF41 = U6[1] < qF41
A41 = {"BASE": {k: run_src(SRC[k], r_F, np.where(isF41, np.inf, 0.0)) for k in SRC}}
D41, H41 = finish(A41); P41 = per_heavy(D41, H41, keys=("U3", "CASC", "BIO"))
ref = r41["main"]["per_heavy_event"]["U3"]["TAB"]
R["check_041"] = {"U3_TAB": P41["U3"]["TAB"], "U3_TAB_041": ref, "rel_err": abs(P41["U3"]["TAB"] - ref) / ref,
                  "reproduced": bool(abs(P41["U3"]["TAB"] - ref) / ref < 1e-9)}

# ---------- 4. シナリオ別の E の原因（すべて【推測】、向きは sources の事実から） ----------
g7 = np.random.default_rng(20261042)
U7 = g7.uniform(size=(10, N))
P_AGENT = {"BIO": (0, 0.5), "CASC": (0, 0.3), "U3": (0, 0.25), "NAT": (0, 0.2)}
P_OTHER = {"CASC": (0.1, 0.5), "NUC_G": (0.05, 0.3)}; P_OTHER_BASE = (0.0, 0.2)
FOCI = (0.5, 0.95)
N_urb_main = lu(U7[6], 1e4, 1e7)
# NUC_G：最大の集団がオーストラリアか（038 の survivors と同じ乱数で）
u = T["U_sv"]
au = 27.92e6 * (0.5 + 0.5 * u[1]); alt = np.exp(np.log(4.647e7) + u[2] * (np.log(2.883e8) - np.log(4.647e7))) * (0.02 + 0.23 * u[3])
AUS = au >= alt
R["NUC_G_australia_share"] = float(AUS.mean())

def unif(u_, lohi): return lohi[0] + (lohi[1] - lohi[0]) * u_

def causes(k, cfg):
    pa = unif(U7[1], cfg.get("agent", {}).get(k, P_AGENT.get(k, (0, 0))))
    po = unif(U7[3], cfg.get("other", {}).get(k, P_OTHER.get(k, P_OTHER_BASE)))
    ff = np.full(N, cfg["foci_fixed"]) if "foci_fixed" in cfg else unif(U7[5], FOCI)
    if k == "NUC_G" and "foci_fixed" not in cfg: ff = np.where(AUS, 0.0, ff)
    Xa = U7[0] < pa; Xo = U7[2] < po; Xp = U7[4] < ff
    return Xa, Xo, Xp

PCONV = [0.0, 0.1, 0.25, 0.5, 1.0]
def evaluate(cfg=None, keys=("U3", "CASC", "BIO")):
    cfg = cfg or {}
    Nu = np.full(N, cfg["Nurb_fixed"]) if "Nurb_fixed" in cfg else N_urb_main
    used = U5[1] < p_grow
    A = {"BASE": {}}; A.update({f"GROW_{pc}": {} for pc in PCONV}); info = {}
    for k in SRC:
        Xa, Xo, Xp = causes(k, cfg)
        Nb = np.where(Xo | Xa, 0.0, np.where(Xp, Nu, np.inf))
        A["BASE"][k] = run_src(SRC[k], r_F, Nb)
        for pc in PCONV:
            conv = U7[7] < pc
            Ng = np.where(used & ~Xo & conv, np.inf, Nb)
            A[f"GROW_{pc}"][k] = run_src(SRC[k], np.where(used, r_F + dK, r_F), Ng)
        B = A["BASE"][k]; hv = np.isin(B["st"], HEAVY)
        if hv.any():
            wt = B["wtot"][hv]; wf = B["wF"][hv]; allF = np.isinf(Nb[hv]) | (wf >= wt - 1e-9)
            info[k] = {"heavy_share": float(hv.mean()), "qF_eff_heavy": float(allF.mean()), "E_always_heavy": float((Nb[hv] == 0).mean()),
                       "mixed_heavy": float((~allF & (Nb[hv] > 0)).mean()),
                       "wait_F_share_median": float(np.median(np.where(wt > 0, wf / np.maximum(wt, 1e-12), 1.0)))}
    D, hv = finish(A)
    ph = per_heavy(D, hv, keys=keys)
    return D, hv, ph, info

def breakeven(ph, key="U3"):
    tab = ph[key]["TAB"]; g = [ph[key][f"GROW_{pc}"] for pc in PCONV]
    if g[0] >= tab: return 0.0
    for i in range(1, len(PCONV)):
        if g[i - 1] < tab <= g[i]:
            return float(PCONV[i - 1] + (tab - g[i - 1]) / (g[i] - g[i - 1]) * (PCONV[i] - PCONV[i - 1]))
    return None

D, HV, PH, INFO = evaluate()
W = weighted(D, HV, anchors=("A_Ord", "A_XPT_S"))
R["main"] = {"per_heavy_event": PH, "info": INFO, "breakeven_pconv_U3": breakeven(PH), "breakeven_pconv_named": breakeven(PH, "named_pooled"),
             "weighted": {a: {u_: {x: W[a][u_][x]["total"] for x in W[a][u_]} for u_ in ("U2", "U3")} for a in W}}
R["compare_041"] = {"U3_TAB": ref, "named_TAB": r41["main"]["per_heavy_event"]["named_pooled"]["TAB"],
                    "CASC_TAB": r41["main"]["per_heavy_event"]["CASC"]["TAB"], "BIO_TAB": r41["main"]["per_heavy_event"]["BIO"].get("TAB"),
                    "w_AOrd_U3": r41["main"]["weighted"]["A_Ord"]["U3"]["TAB"], "w_AXPTS_U2": r41["main"]["weighted"]["A_XPT_S"]["U2"]["TAB"],
                    "n_star": r41["n_star"]["empirical_qF_U01"]}

# ---------- 5. n*（U3 の事前で単一集団を走査） ----------
def sweep2(cfg=None):
    cfg = cfg or {}
    Xa, Xo, Xp = causes("U3", cfg)
    Nu = np.full(N, cfg["Nurb_fixed"]) if "Nurb_fixed" in cfg else N_urb_main
    Nb = np.where(Xo | Xa, 0.0, np.where(Xp, Nu, np.inf)); sc = G["DRAWS"]["GEN"]; rows = []
    for n in NG:
        sv = {"s": np.full(N, n / POP_W), "ntot": np.full(N, n), "nmax": np.full(N, n), "mcap": np.ones(N)}
        st = assign(sv, 1.0); T2s = times2(st, sv, 0.3, r_F, r_pop, Nb)[0]
        rows.append({"n": float(n), "R_tab": float((V_gen(st, T2s, sc, 0.3, tablet=True) - V_gen(st, T2s, sc, 0.3)).mean())})
    v = np.array([r["R_tab"] for r in rows]); idx = np.where(v >= 0.5 * REF_U1)[0]; ns_ = None
    if idx.size:
        i = int(idx[0])
        if i == 0: ns_ = float(NG[0])
        else:
            x0, x1 = np.log10(NG[i - 1]), np.log10(NG[i]); ns_ = float(10 ** (x0 + (0.5 * REF_U1 - v[i - 1]) / (v[i] - v[i - 1]) * (x1 - x0)))
    return {"rows": rows, "n_star": ns_}
SWm = sweep2()
R["n_star"] = {"main": SWm["n_star"], "041": r41["n_star"]["empirical_qF_U01"]}
R["n_star_rows_main"] = SWm["rows"]

# ---------- 6. トルネード（U3 の TAB、他は主の事前のまま） ----------
def u3tab(cfg): return evaluate(cfg, keys=("U3",))[2]["U3"]["TAB"]
TOR = {
 "p_other_U3": {"lo": u3tab({"other": {"U3": (0, 0)}}), "hi": u3tab({"other": {"U3": (0.2, 0.2)}})},
 "f_foci": {"lo": u3tab({"foci_fixed": 0.0}), "hi": u3tab({"foci_fixed": 1.0})},
 "p_agent_U3": {"lo": u3tab({"agent": {"U3": (0, 0)}}), "hi": u3tab({"agent": {"U3": (0.25, 0.25)}})},
 "N_urb": {"lo": u3tab({"Nurb_fixed": 1e4}), "hi": u3tab({"Nurb_fixed": 1e7})},
}
for k, v in TOR.items(): v["range"] = abs(v["hi"] - v["lo"])
R["tornado_U3_TAB"] = TOR

# ---------- 7. 感度（判定に使わない） ----------
R["sens_Nurb"] = {}
for nu in (1e4, 1e5, 1e6, 1e7, np.inf):
    _, _, p, inf_ = evaluate({"Nurb_fixed": nu}, keys=("U3", "CASC", "BIO"))
    R["sens_Nurb"][str(nu)] = {"U3_TAB": p["U3"]["TAB"], "named_TAB": p["named_pooled"]["TAB"], "be_U3": breakeven(p),
                               "qF_eff_U3": inf_["U3"]["qF_eff_heavy"]}
_, _, p, inf_ = evaluate({"agent": {k: (0, 0) for k in ("BIO", "CASC", "U3", "NAT")}})
R["sens_no_agent"] = {"U3_TAB": p["U3"]["TAB"], "BIO_qF": inf_.get("BIO", {}).get("qF_eff_heavy"), "be_U3": breakeven(p)}
Df, Hf, p, inf_ = evaluate({"foci_fixed": 1.0})
R["sens_foci1"] = {"U3_TAB": p["U3"]["TAB"], "named_TAB": p["named_pooled"]["TAB"], "be_U3": breakeven(p)}

# ---------- 8. p_conv の目安（124 の一部、4例の数え方の幅） ----------
def beta_tail(a, b, x, m=20000):
    xs = np.linspace(x, 1, m); lb = lgamma(a) + lgamma(b) - lgamma(a + b)
    f = np.exp((a - 1) * np.log(np.maximum(xs, 1e-300)) + (b - 1) * np.log(np.maximum(1 - xs, 1e-300)) - lb)
    return float(np.trapezoid(f, xs) if hasattr(np, "trapezoid") else np.trapz(f, xs))
R["pconv_anchor"] = {f"s={s}_of_4": {"mean": (1 + s) / 6, "P_gt_0.77": beta_tail(1 + s, 1 + 4 - s, 0.77)} for s in (1, 2, 3)}
R["pconv_anchor"]["cases"] = {"Habsburg_cordon_1770": "success", "Marseille_1720": "failure (quarantine breached)",
                              "San_Francisco_1900": "failure/partial (lifted after 2.5 days; may have seeded wildlife)", "Eyam_1665": "success for neighbours / ambiguous"}

# ---------- 9. 仮説 ----------
U3 = PH["U3"]; be = R["main"]["breakeven_pconv_U3"]; ns_ = SWm["n_star"]; wA = R["main"]["weighted"]["A_Ord"]["U3"]["TAB"]
R["hypotheses"] = {
 "H54a": {"value": [U3["TAB"], ref, U3["TAB"] / ref], "rule": "U3 heavy TAB >= 1.2 x 041", "supported": bool(U3["TAB"] >= 1.2 * ref)},
 "H54b": {"value": [TOR["p_other_U3"]["range"], TOR["f_foci"]["range"]], "rule": "range(p_other 0->0.2) > range(f_foci 0->1)", "supported": bool(TOR["p_other_U3"]["range"] > TOR["f_foci"]["range"])},
 "H54c": {"value": [ns_, R["n_star"]["041"]], "rule": "n* <= 041", "supported": bool(ns_ is not None and ns_ <= R["n_star"]["041"])},
 "H54d": {"value": be, "rule": "breakeven p_conv >= 0.77 (None = never)", "supported": bool(be is None or be >= 0.77)},
 "H54e": {"value": [wA, R["compare_041"]["w_AOrd_U3"], wA / R["compare_041"]["w_AOrd_U3"]], "rule": "weighted A_Ord U3 TAB within +-30% of 041", "supported": bool(0.7 <= wA / R["compare_041"]["w_AOrd_U3"] <= 1.3)},
 "H54f": {"value": [TOR["N_urb"]["lo"], TOR["N_urb"]["hi"]], "rule": "U3 TAB(N_urb=1e7)/TAB(1e4) >= 1.5 or <= 1/1.5", "supported": bool(max(TOR["N_urb"]["hi"], TOR["N_urb"]["lo"]) >= 1.5 * min(TOR["N_urb"]["hi"], TOR["N_urb"]["lo"]))},
}

# ---------- 10. 図 ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ks = [k for k in ("BIO", "CASC", "U3", "NAT", "AS", "SV", "SO", "NUC_G", "NUC_R") if k in INFO]
x = np.arange(len(ks))
ax[0].bar(x, [INFO[k]["qF_eff_heavy"] for k in ks], color="C2", label="all-F wait (q_F eff.)")
ax[0].bar(x, [INFO[k]["mixed_heavy"] for k in ks], bottom=[INFO[k]["qF_eff_heavy"] for k in ks], color="C1", label="F then E (plague above N_urb)")
ax[0].bar(x, [INFO[k]["E_always_heavy"] for k in ks], bottom=[INFO[k]["qF_eff_heavy"] + INFO[k]["mixed_heavy"] for k in ks], color="C3", label="E always (agent / other)")
ax[0].axhline(0.5, color="k", ls=":", lw=1, label="041 mean q_F")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"{k}\n(heavy {INFO[k]['heavy_share']:.2f})" for k in ks], fontsize=7)
ax[0].set_ylabel("share of heavy-collapse samples"); ax[0].set_title("123: regime of heavy-collapse groups by scenario"); ax[0].legend(fontsize=7)
rows41 = r41["n_star_rows_main"]
ax[1].semilogx([r["n"] for r in rows41], [r["R_tab"] for r in rows41], "C0", label="041: q_F~U(0,1)")
ax[1].semilogx([r["n"] for r in SWm["rows"]], [r["R_tab"] for r in SWm["rows"]], "k", lw=2, label="042: scenario causes (U3 prior)")
ax[1].axhline(0.5 * REF_U1, color="gray", ls="--", lw=0.8, label="half of U1 value")
if ns_: ax[1].axvline(ns_, color="k", ls=":")
ax[1].set_xlabel("single connected surviving group n"); ax[1].set_ylabel("tablet-guide value per event"); ax[1].legend(fontsize=8)
ax[1].set_title("n* under cause-based regime model")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_regime_and_nstar.png"), dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
names = list(TOR); lo = [TOR[k]["lo"] for k in names]; hi = [TOR[k]["hi"] for k in names]
for i, k in enumerate(names):
    ax[0].plot([lo[i], hi[i]], [i, i], "k-", lw=6, alpha=0.3); ax[0].plot(lo[i], i, "bo"); ax[0].plot(hi[i], i, "ro")
ax[0].axvline(U3["TAB"], color="k", ls=":", label="042 main"); ax[0].axvline(ref, color="C0", ls="--", label="041")
ax[0].set_yticks(range(len(names))); ax[0].set_yticklabels(["p_other(U3) 0 / 0.2", "f_foci 0 / 1", "p_agent(U3) 0 / 0.25", "N_urb 1e4 / 1e7"])
ax[0].set_xlabel("tablet-guide value per U3 heavy event (blue=lo, red=hi)"); ax[0].legend(fontsize=8); ax[0].set_title("tornado")
ax[1].plot(PCONV, [U3[f"GROW_{pc}"] for pc in PCONV], "C2-o", label="GROW(p_conv), U3 heavy")
ax[1].axhline(U3["TAB"], color="k", ls="--", label="TAB")
for s, cc in ((1, "C3"), (2, "C1"), (3, "C4")):
    ax[1].axvline((1 + s) / 6, color=cc, ls=":", label=f"Beta mean, {s}/4 successes")
ax[1].set_xlabel("p_conv"); ax[1].set_ylabel("gain per U3 heavy event"); ax[1].legend(fontsize=7); ax[1].set_title("124: knowledge vs tablet guide")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_tornado_pconv.png"), dpi=120); plt.close()

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=1, default=float)
out = {k: R[k] for k in ("check_041", "NUC_G_australia_share", "compare_041", "n_star", "tornado_U3_TAB", "sens_Nurb", "sens_no_agent", "sens_foci1", "pconv_anchor", "hypotheses")}
out["main_ph"] = PH; out["info"] = INFO; out["be"] = [R["main"]["breakeven_pconv_U3"], R["main"]["breakeven_pconv_named"]]; out["weighted"] = R["main"]["weighted"]
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
print("runtime", round(R["runtime_seconds"], 1))
