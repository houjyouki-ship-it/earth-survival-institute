"""研究022 探索的解析（初回実行後に追加。登録仮説の判定には使わない）。
sim.py の results.json（c_table, grating_P_res）だけを読み、
(1) 読む人が絞り（開口）と照明 σ を最適化する場合の c_opt、
(2) c_opt での 1% 予算の単層 D_max と段階書式、を計算する。seed 固定。
"""
import json, math, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t0 = time.time()
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
R0 = json.load(open(f"{HERE}/results.json"))
rng = np.random.default_rng(20261122)
SIGMAS = (0.1, 0.2, 0.3, 0.45, 0.6); RMULTS = (0.0, 1.0, 2.0, 3.0, 5.0)
SIGN = (0.02, 0.035, 0.05, 0.07, 0.1); CTHS = (0.03, 0.05, 0.1, 0.15, 0.2)
f = lambda v: float("inf") if v == "inf" else float(v)
def Pres(sg, Rm, cth): return f(R0["grating_P_res"][f"sigma{sg}|R{Rm}"][str(cth)])
def wstar(sg, Rm, sn, tg):
    # c = 2 w*/P_res（任意の C_th で同じ w* が戻る。P_res が有限な C_th を使う）
    for cth in CTHS:
        c = f(R0["c_table"][f"{sg}|{Rm}|2d|{sn}|{tg}|{cth}"]); p = Pres(sg, Rm, cth)
        if math.isfinite(p): return c * p / 2 if math.isfinite(c) else float("inf")
    return float("inf")
# (1) 絞り：開口を t 倍にすると球面収差は t^4 倍（W040∝ρ^4）、長さの単位 λ/NA は 1/t 倍
OUT = {}
def wopt(R_lens, sn, tg):
    best = (float("inf"), None)
    for Rk in RMULTS:
        if Rk > R_lens or (Rk == 0 and R_lens > 0): continue
        t = 1.0 if R_lens == 0 else (Rk / R_lens) ** 0.25
        for sg in SIGMAS:
            w = wstar(sg, Rk, sn, tg) / t
            if w < best[0]: best = (w, dict(R_read=Rk, t=t, sigma_read=sg))
    return best
COPT = {}
for tg in (1e-4, 1e-3):
    for Rl in RMULTS:
        for sn in SIGN:
            w, how = wopt(Rl, sn, tg)
            for sg in SIGMAS:
                for cth in CTHS:
                    COPT[(tg, Rl, sn, sg, cth)] = dict(c=2 * w / Pres(sg, Rl, cth), how=how)
def q(x):
    x = np.array(x)
    return dict(p10=float(np.percentile(x, 10)), p50=float(np.percentile(x, 50)), p90=float(np.percentile(x, 90)),
                frac_inf=float(np.mean(~np.isfinite(x))), frac_in_1_2=float(np.mean((x >= 1) & (x <= 2))))
OUT["c_opt_distribution"] = {str(tg): q([v["c"] for k, v in COPT.items() if k[0] == tg]) for tg in (1e-4, 1e-3)}
OUT["c_opt_by_R_lens"] = {str(Rl): q([v["c"] for k, v in COPT.items() if k[0] == 1e-4 and k[1] == Rl]) for Rl in RMULTS}
OUT["c_fixed_by_R"] = {str(Rl): q([f(R0["c_table"][f"{sg}|{Rl}|2d|{sn}|0.0001|{cth}"]) for sg in SIGMAS for sn in SIGN for cth in CTHS]) for Rl in RMULTS}
OUT["stopdown_choice_R5_sn005"] = wopt(5.0, 0.05, 1e-4)[1]
OUT["stopdown_choice_R3_sn005"] = wopt(3.0, 0.05, 1e-4)[1]

# (2) 配分（sim.py と同じ式）
PERIOD_UM = 1000.0 / 1180.0; M_SINGLE = math.sqrt(1e-4 * 3e-3); F_BURST, L_BURST = 1e-3, 100
PX = (4096 * 2160) / 2e6; REEL = 120e9
def _binom_pmf(n, p):
    k = np.arange(n + 1); c = np.array([math.comb(n, i) for i in k], float)
    p = np.asarray(p, float)[..., None]; return c * p ** k * (1 - p) ** (n - k)
def residual_analytic(n, p, m, fb, L, nbits=8):
    p = np.atleast_1d(np.asarray(p, float)); m = np.broadcast_to(m, p.shape).astype(float); L = np.broadcast_to(L, p.shape).astype(float)
    q0 = p + m - 2 * p * m; alpha = fb / L; beta = 1.0 / L; pi = alpha / (alpha + beta); S_ = p.size
    e = np.zeros((S_, n + 1)); thr = n // 2
    for k in range(n + 1):
        a = _binom_pmf(k, np.full(S_, 0.5)); b = _binom_pmf(n - k, q0); tot = np.zeros((S_, n + 1))
        for i in range(k + 1): tot[:, i:i + n - k + 1] += a[:, i:i + 1] * b
        e[:, k] = tot[:, thr + 1:].sum(axis=1)
    T = np.zeros((S_, n + 1, n + 1))
    for k in range(n + 1):
        stay = _binom_pmf(k, 1 - beta); enter = _binom_pmf(n - k, alpha)
        for i in range(k + 1): T[:, k, i:i + n - k + 1] += stay[:, i:i + 1] * enter
    v = _binom_pmf(n, pi) * (1 - e)
    for _ in range(nbits - 1): v = np.einsum("sk,skj->sj", v, T) * (1 - e)
    return np.clip(1.0 - v.sum(axis=1), 0, 1)
S = 4000
def logu(lo, hi, n=S): return np.exp(rng.uniform(np.log(lo), np.log(hi), n))
p_ref = logu(1e-4, 1e-2); beta_s = rng.uniform(0, 2, S); m_s = logu(1e-4, 3e-3)
d_res = logu(1.35, 4.0); d_guide = logu(1.1, 1.6); d_guide_pess = logu(1.35, 2.3)
allc = np.array([v["c"] for k, v in COPT.items() if k[0] == 1e-4])
guidec = np.array([v["c"] for k, v in COPT.items() if k[0] == 1e-4 and k[1] in (2.0, 3.0)])  # Zuylen の最適単レンズは 2.5×Rayleigh
c_all = allc[rng.integers(0, allc.size, S)]; c_g = guidec[rng.integers(0, guidec.size, S)]
S_LIST = (1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20)
def area(n, s): return 1e6 * 8 * n * s * s / PX / REEL
RES = {}
for n in (5, 7):
    for s in S_LIST:
        RES[(n, s)] = residual_analytic(n, np.minimum(p_ref * (s / 2.0) ** (-beta_s), 0.49), m_s, F_BURST, np.full(S, max(1.0, L_BURST / s)))
TG = {5: 1e-4, 7: 1e-6}
def joint(n, s, d, c): return (RES[(n, s)] <= TG[n]) & (s * PERIOD_UM >= c * d)
OUT["P_readable_by_s_copt"] = {str(s): float((s * PERIOD_UM >= c_all * d_res).mean()) for s in S_LIST}
OUT["s_min_P90_copt"] = min([s for s in S_LIST if OUT["P_readable_by_s_copt"][str(s)] >= 0.9], default=None)
B = 0.01
rows = [dict(s=s, P=float(joint(7, s, d_res, c_all).mean()), D=B / area(7, s)) for s in S_LIST]
for r in rows: r["EMB"] = r["P"] * r["D"]
feas = [r for r in rows if r["P"] >= 0.9]
OUT["single_copt"] = dict(rows=rows, best_P90=max(feas, key=lambda r: r["D"]) if feas else None, best_expected=max(rows, key=lambda r: r["EMB"]))
GUIDE = {}
for thr in (0.99, 0.95, 0.9):
    sg = min([s for s in S_LIST if joint(5, s, d_res, c_all).mean() >= thr], default=None)
    GUIDE[str(thr)] = dict(s_g=sg, P_read=float(joint(5, sg, d_res, c_all).mean()) if sg else None,
                           area_frac_0p1MB=0.1 * area(5, sg) if sg else None)
OUT["guide_layer_copt"] = GUIDE
PG = np.round(np.linspace(0, 1, 21), 3)
def staged(dg, G_MB, thr_g):
    gl = GUIDE[str(thr_g)]
    if gl["s_g"] is None: return None
    ag = G_MB * area(5, gl["s_g"]); out = []
    for pg in PG:
        pe = gl["P_read"] * pg; rr = []
        for s in S_LIST:
            P = pe * float(joint(7, s, dg, c_g).mean()) + (1 - pe) * float(joint(7, s, d_res, c_all).mean())
            D = max(B - ag, 0) / area(7, s); rr.append(dict(s=s, P=P, D=D, EMB=P * D))
        fe = [r for r in rr if r["P"] >= 0.9]
        out.append(dict(P_g=float(pg), best_P90=max(fe, key=lambda r: r["D"]) if fe else None, best_expected=max(rr, key=lambda r: r["EMB"])))
    return dict(area_guide=ag, rows=out)
def cross(st):
    sgl = OUT["single_copt"]; eS = sgl["best_expected"]["EMB"]; dS = sgl["best_P90"]["D"] if sgl["best_P90"] else 0.0
    if st is None: return None
    r = st["rows"]
    return dict(area_guide=st["area_guide"], single_EMB=eS, single_D_P90=dS,
                P_g_star_expected=next((x["P_g"] for x in r if x["best_expected"]["EMB"] >= eS), None),
                P_g_min_P90=next((x["P_g"] for x in r if x["best_P90"] and x["best_P90"]["D"] > dS), None),
                at_Pg_1=dict(EMB=r[-1]["best_expected"]["EMB"], s_exp=r[-1]["best_expected"]["s"],
                             D_P90=r[-1]["best_P90"]["D"] if r[-1]["best_P90"] else None, s_P90=r[-1]["best_P90"]["s"] if r[-1]["best_P90"] else None),
                at_Pg_0p7=dict(EMB=r[14]["best_expected"]["EMB"], D_P90=r[14]["best_P90"]["D"] if r[14]["best_P90"] else None))
ST = {}
for name, dg in (("base", d_guide), ("pessimistic", d_guide_pess)):
    for G in (0.03, 0.1, 0.3):
        for thr in (0.99, 0.95):
            st = staged(dg, G, thr); ST[f"{name}|G{G}|guide{thr}"] = dict(cross=cross(st), curve=st)
OUT["staged_copt"] = {k: v["cross"] for k, v in ST.items()}
OUT["runtime_seconds"] = time.time() - t0
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o): return "inf"
    if isinstance(o, np.floating): return clean(float(o))
    return o
json.dump(clean(OUT), open(f"{HERE}/explore_results.json", "w"), ensure_ascii=False, indent=1)
# 図
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
xs = list(RMULTS)
ax[0].plot(xs, [OUT["c_fixed_by_R"][str(r)]["p50"] for r in xs], "o-", label="fixed aperture/focus (sim.py), median")
ax[0].plot(xs, [OUT["c_opt_by_R_lens"][str(r)]["p50"] for r in xs], "s-", label="reader stops down + picks σ, median")
ax[0].plot(xs, [OUT["c_opt_by_R_lens"][str(r)]["p90"] for r in xs], "s--", label="… 90th pct")
ax[0].axhspan(1, 2, color="gray", alpha=0.2, label="021 prior"); ax[0].set_yscale("log")
ax[0].set_xlabel("spherical aberration of lens (× Rayleigh limit)"); ax[0].set_ylabel("margin factor c"); ax[0].legend(fontsize=7)
ax[0].set_title("c vs lens aberration (target 1e-4)")
for key, col in (("base|G0.1|guide0.95", "C1"), ("pessimistic|G0.1|guide0.95", "C3")):
    cv = ST[key]["curve"]
    if cv is None: continue
    ax[1].plot(PG, [r["best_expected"]["EMB"] for r in cv["rows"]], color=col, label=f"staged {key}: expected MB")
    ax[1].plot(PG, [r["best_P90"]["D"] if r["best_P90"] else 0 for r in cv["rows"]], "--", color=col, label="… D_max P≥0.9")
ax[1].axhline(OUT["single_copt"]["best_expected"]["EMB"], color="k", lw=0.8, label="single: expected MB")
ax[1].axhline(OUT["single_copt"]["best_P90"]["D"] if OUT["single_copt"]["best_P90"] else 0, color="k", ls="--", lw=0.8, label="single: D_max P≥0.9")
ax[1].set_xlabel("P_g (lens guide succeeds)"); ax[1].set_ylabel("MB (1% reel, ≤1e-6)"); ax[1].legend(fontsize=6)
ax[1].set_title("Staged format with optimizing reader (exploratory)")
plt.tight_layout(); plt.savefig(f"{HERE}/fig_explore_stopdown_staged.png", dpi=130); plt.close()
print(json.dumps({k: OUT[k] for k in ("c_opt_distribution", "c_opt_by_R_lens", "c_fixed_by_R", "stopdown_choice_R5_sn005", "stopdown_choice_R3_sn005", "s_min_P90_copt", "guide_layer_copt")}, indent=1))
print("P_read", OUT["P_readable_by_s_copt"])
print("single", json.dumps({k: OUT["single_copt"][k] for k in ("best_P90", "best_expected")}))
print("staged", json.dumps(clean(OUT["staged_copt"]), indent=1))
print(f"runtime {OUT['runtime_seconds']:.1f}s")
