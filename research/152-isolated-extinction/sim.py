"""研究152：外の人口がない小集団の消滅率（open_problems 348）。
計算のみ。入力は 151 の sim.py の定義部（写しで実行、146・038 を含む）と 151 の results.json。
記録は sources/ の Wikipedia 本文（2026-10-27 取得、sources.md）。事前登録：PREREGISTRATION.md（21:41 JST）。
注：TC・V は生存確率や復興確率ではない（模型の中の量）。λ_iso は模型の中の推定。"""
import json, os, time, contextlib, io
import numpy as np
from scipy import stats, optimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
R151 = json.load(open(f"{RES}/151-deep-trough-join/results.json", encoding="utf8"))
src = open(f"{RES}/151-deep-trough-join/sim.py", encoding="utf8").read(); src = src[:src.index("# ---- λ の資料")]
ns = {"__file__": f"{RES}/151-deep-trough-join/sim.py", "__name__": "m152"}
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, "151head", "exec"), ns)
N, EVID, HEAVY, ST, SV, weights, resample, geff, ANCH, VARS, TH, TCG, eval_lam, TAB151 = (ns[k] for k in
    ("N", "EVID", "HEAVY", "ST", "SV", "weights", "resample", "geff", "ANCH", "VARS", "TH", "TCG", "eval_lam", "TAB"))
CHECK = {}
# ---- 再現：151 の λ=0.5・death（12 版） ----
for a in ANCH:
    for vn in VARS:
        k = f"{a}|{vn}|death|0.5"; d = abs(TAB151[k]["V_rel_lam0"] - R151["table"][k]["V_rel_lam0"]); CHECK[f"151_death05|{a}|{vn}"] = d; assert d < 1e-12, (k, d)
# ---- 深い標本の群の大きさ n = ntot/G_eff（151 と同じ引き直し） ----
SAMP = {}
for a in ANCH:
    w = weights(a)[0]
    evw = {s: float(np.mean(w[s] * np.isin(ST[s], HEAVY))) for s in EVID}
    VV = {"P_BIO": {"BIO": 1.0}, "P_CASC": {"CASC": 1.0}, "P_MIX": {"BIO": 0.5, "CASC": 0.5}, "P_EVIDW": evw}
    for vn in VARS:
        rng = np.random.default_rng(20261146); svx = {s: resample(s, VV[vn], rng) for s in ("AI", "GEN")}
        ws_, dp_, g_, n_ = [], [], [], []
        for s in list(EVID) + ["AI", "GEN"]:
            sv = SV[s] if s in EVID else svx[s]; hv = np.isin(ST[s], HEAVY)
            ws = np.broadcast_to(np.asarray(w[s], float), (N,)) * hv; ge = geff(sv)
            ws_.append(ws[hv]); dp_.append(sv["s"][hv] < TH); g_.append(ge[hv]); n_.append((sv["ntot"] / ge)[hv])
        SAMP[(a, vn)] = tuple(np.concatenate(x) for x in (ws_, dp_, g_, n_)); SAMP[(a, vn)] = SAMP[(a, vn)][:2] + (SAMP[(a, vn)][2].astype(int), SAMP[(a, vn)][3])
        ws, dp, g, n = SAMP[(a, vn)]
        f = float(np.sum(ws * dp) / np.sum(ws)); CHECK[f"fdeep|{a}|{vn}"] = abs(f - R151["table"][f"{a}|{vn}|merge|0.0"]["f_deep_surv"]); assert CHECK[f"fdeep|{a}|{vn}"] < 1e-9
def V_death(a, vn, lam_vec):
    """151 の death 版を、深い標本ごとの λ で（G'〜Bin(G,1−λ_i)）"""
    ws, dp, g, n = SAMP[(a, vn)]; tc0 = np.array([TCG[int(x)] for x in g]); tc1 = tc0.copy(); pext = np.zeros(len(g))
    for i in np.where(dp)[0]:
        G = g[i]; pm = stats.binom.pmf(np.arange(G + 1), G, 1 - lam_vec[i])
        tc1[i] = float(np.sum(pm * np.array([TCG[x] for x in range(G + 1)]))); pext[i] = pm[0]
    return float(np.sum(ws * tc1) / np.sum(ws * tc0)), float(np.sum(ws * pext) / np.sum(ws))
for a in ANCH:
    for vn in VARS:
        v, _ = V_death(a, vn, np.full(len(SAMP[(a, vn)][0]), 0.5)); CHECK[f"own_death05|{a}|{vn}"] = abs(v - R151["table"][f"{a}|{vn}|death|0.5"]["V_rel_lam0"])
        assert CHECK[f"own_death05|{a}|{vn}"] < 1e-9
# ---- 記録（sources.md）：(名前, N の下限, 上限, 隔絶の年数 下限, 上限, 消滅=1, N の出典の種類) ----
REC = [("Pitcairn_Bounty_1790-1808", 27, 27, 18, 18, 0, "fact"),
       ("Moriori_Chatham", 1600, 2000, 291, 791, 0, "fact"),
       ("Rapa_Nui", 2000, 3000, 522, 922, 0, "fact"),
       ("Tasmania", 3000, 15000, 8000, 8000, 0, "fact"),
       ("Norse_Greenland_after_last_contact", 250, 2500, 20, 90, 1, "guess_N"),
       ("Pitcairn_Polynesian", 10, 100, 20, 150, 1, "guess_N"),
       ("Henderson_Polynesian", 10, 100, 20, 200, 1, "guess_N")]
def nll(p, Nv, Tv, ev):
    lh0, b = p; h = np.exp(lh0) * (Nv / 100.0) ** (-b); H = h * Tv / 100.0
    return -float(np.sum(ev * np.log(-np.expm1(-H) + 1e-300) - (1 - ev) * H))
ev = np.array([r[5] for r in REC], float)
rngd = np.random.default_rng(20261152); NDRAW = 2000; FITS = {"free": [], "beta0": []}
for _ in range(NDRAW):
    Nv = np.array([np.exp(rngd.uniform(np.log(r[1]), np.log(r[2]))) for r in REC]); Tv = np.array([np.exp(rngd.uniform(np.log(r[3]), np.log(r[4]))) for r in REC])
    best = None
    for x0 in ((0.0, 0.5), (1.0, 1.0), (-1.0, 0.2), (2.0, 1.8)):
        rr = optimize.minimize(nll, x0, args=(Nv, Tv, ev), bounds=[(-12, 6), (0.0, 2.0)], method="L-BFGS-B")
        if best is None or rr.fun < best.fun: best = rr
    FITS["free"].append(best.x)
    r0 = optimize.minimize_scalar(lambda l: nll((l, 0.0), Nv, Tv, ev), bounds=(-12, 6), method="bounded"); FITS["beta0"].append((r0.x, 0.0))
FITS = {k: np.array(v) for k, v in FITS.items()}
# 上限の版：N≥1,000 の存続だけで（消滅 0 件、ただしグリーンランドの北欧人を N≥1,000 に数える版も）Jeffreys の片側 95 % 上限、大きさに依らない
expo_big = (291 + 522 + 8000) / 100.0  # 下限の年数の和（世紀）
JUP0 = stats.gamma.ppf(0.95, 0.5, scale=1 / expo_big); JUP1 = stats.gamma.ppf(0.95, 1.5, scale=1 / (expo_big + 0.2))
def lam_of(n, W, h0, b): return -np.expm1(-np.exp(h0) * (np.maximum(n, 1.0) / 100.0) ** (-b) * W / 100.0)
WS = (300, 1000, 3000); OUT_T = {}
for (a, vn) in SAMP:
    ws, dp, g, n = SAMP[(a, vn)]; wd = ws * dp
    for W in WS:
        for ver in ("free", "beta0", "harsh3_free"):
            fit = FITS["beta0" if ver == "beta0" else "free"]; mult = 3.0 if ver.startswith("harsh") else 1.0
            sel = fit[np.linspace(0, NDRAW - 1, 200).astype(int)]
            lams = np.array([np.sum(wd * np.minimum(1, lam_of(n, W, h0 + np.log(mult), b))) / np.sum(wd) for h0, b in sel])
            mid = sel[np.argsort(lams)[len(lams) // 2]]; lv = np.where(dp, lam_of(n, W, mid[0] + np.log(mult), mid[1]), 0.0)
            V, Pe = V_death(a, vn, lv)
            OUT_T[f"{a}|{vn}|W{W}|{ver}"] = {"lam_deep_mean_med": float(np.median(lams)), "lam_deep_mean_p05": float(np.quantile(lams, .05)),
                                            "lam_deep_mean_p95": float(np.quantile(lams, .95)), "V_rel_lam0_median_fit": V, "P_ext_heavy": Pe}
        for nm, hu in (("J95_0ev", JUP0), ("J95_1ev", JUP1)):
            lv = np.where(dp, -np.expm1(-hu * W / 100.0), 0.0); V, Pe = V_death(a, vn, lv)
            OUT_T[f"{a}|{vn}|W{W}|{nm}"] = {"lam_deep_mean_med": float(-np.expm1(-hu * W / 100.0)), "V_rel_lam0_median_fit": V, "P_ext_heavy": Pe}
ws, dp, g, n = SAMP[("A_XPT_S", "P_MIX")]; wd = ws * dp
def wq(x, w, q):
    o = np.argsort(x); c = np.cumsum(w[o]) / np.sum(w); return float(x[o][np.searchsorted(c, q)])
NSTAT = {"deep_group_size_wq05": wq(n[dp], wd[dp], .05), "deep_group_size_wq50": wq(n[dp], wd[dp], .5), "deep_group_size_wq95": wq(n[dp], wd[dp], .95),
         "deep_G_wmed": wq(g[dp].astype(float), wd[dp], .5)}
fr = FITS["free"]; FSTAT = {"h0_per_century_at_N100_median": float(np.exp(np.median(fr[:, 0]))), "beta_median": float(np.median(fr[:, 1])),
     "beta_p05": float(np.quantile(fr[:, 1], .05)), "beta_p95": float(np.quantile(fr[:, 1], .95)), "share_beta_at_bound2": float(np.mean(fr[:, 1] > 1.99)),
     "h_beta0_median": float(np.exp(np.median(FITS["beta0"][:, 0]))), "J95_0ev": float(JUP0), "J95_1ev": float(JUP1)}
# ---- 仮説 ----
k0 = "A_XPT_S|P_MIX"; H = {}
big_ext = [r[0] for r in REC if r[5] == 1 and r[2] >= 1000]; big_ext_lo = [r[0] for r in REC if r[5] == 1 and r[1] >= 1000]
H["H164a"] = {"class": "prediction", "value": {"extinct_with_N_upper_ge1000": big_ext, "extinct_with_N_lower_ge1000": big_ext_lo}, "supported": len(big_ext) <= 1}
v = OUT_T[f"{k0}|W1000|free"]["lam_deep_mean_med"]; H["H164b"] = {"class": "prediction", "value": v, "supported": v < 0.143}
H["H164c"] = {"class": "confirmatory", "value": max(v_ for k_, v_ in CHECK.items() if "death05" in k_), "supported": True}
worst = max(1 - OUT_T[f"{k0}|W1000|{x}"]["V_rel_lam0_median_fit"] for x in ("beta0", "J95_0ev", "J95_1ev"))
H["H164d"] = {"class": "prediction", "value": worst, "by": {x: 1 - OUT_T[f"{k0}|W1000|{x}"]["V_rel_lam0_median_fit"] for x in ("beta0", "J95_0ev", "J95_1ev")}, "supported": worst < 0.05}
mids = sorted(np.sqrt(r[1] * r[2]) for r in REC if r[5] == 1); H["H164e"] = {"class": "prediction", "value": float(np.median(mids)), "note": "N は 3 件とも推測の幅の幾何平均", "supported": float(np.median(mids)) < 100}
OUT = {"study": "152-isolated-extinction", "note": "TC・V は生存確率や復興確率ではない", "check_max": max(CHECK.values()), "check": CHECK, "records": REC,
       "fit": FSTAT, "deep_sizes_A_XPT_S_P_MIX": NSTAT, "table": OUT_T, "hypotheses": H}
# ---- 図 ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4)); nn = np.logspace(1, 6, 100)
for q in (0.05, 0.5, 0.95):
    h0, b = np.quantile(fr[:, 0], q), np.median(fr[:, 1]); ax[0].loglog(nn, lam_of(nn, 1000, h0, b), label=f"free fit h0 q{q}")
ax[0].loglog(nn, np.full_like(nn, -np.expm1(-JUP0 * 10)), "k--", label="Jeffreys 95% (0 ev, N>=1000)")
for r in REC: ax[0].plot([np.sqrt(r[1] * r[2])], [0.9 if r[5] else 0.002], "rv" if r[5] else "g^")
ax[0].axvspan(NSTAT["deep_group_size_wq05"], NSTAT["deep_group_size_wq95"], color="grey", alpha=.2); ax[0].set_xlabel("group size N"); ax[0].set_ylabel("loss prob. over W=1000 y"); ax[0].legend(fontsize=7)
for ver in ("free", "beta0", "harsh3_free", "J95_0ev", "J95_1ev"):
    ax[1].plot(WS, [OUT_T[f"{k0}|W{W}|{ver}"]["V_rel_lam0_median_fit"] for W in WS], "o-", label=ver)
ax[1].axhline(R151["table"][f"{k0}|death|0.5"]["V_rel_lam0"], c="k", ls=":", label="151 death λ=0.5"); ax[1].set_xscale("log"); ax[1].set_xlabel("W (years without outside population)"); ax[1].set_ylabel("heavy value rel. λ=0 (A_XPT_S P_MIX)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_isolated.png", dpi=120)
OUT["runtime_s"] = time.time() - T0
json.dump(OUT, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
print("check max", OUT["check_max"]); print("fit", FSTAT); print("sizes", NSTAT)
for k, v in OUT_T.items():
    if k.startswith(k0) or "W1000" in k: print(k, {x: round(y, 4) for x, y in v.items()})
print(json.dumps(H, ensure_ascii=False, default=float)); print("runtime", round(OUT["runtime_s"], 1))
