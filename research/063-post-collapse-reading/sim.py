"""研究063：崩壊後の社会が古い蔵書を探して読む確率と時期（open_problems 167）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md。
062 の sim.py を行頭の「# ---------- 版 ----------」の前まで exec し、062 の主の P_new を差0で再現してから、
1部が見つかる確率 fp を「全写しに共通の読む社会 R(T)（重い崩壊だけ）× 写しごとの f」に分ける。新しい乱数は seed 20261063。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P62 = os.path.abspath(os.path.join(HERE, "..", "062-copy-survival-correlation", "sim.py"))
src = open(P62, encoding="utf8").read(); cut = src.index("\n# ---------- 版 ----------\n")
G62 = {"__file__": P62, "__name__": "sim062"}
exec(compile(src[:cut], P62, "exec"), G62)
r62 = json.load(open(os.path.join(HERE, "..", "062-copy-survival-correlation", "results.json"), encoding="utf8"))
x62 = json.load(open(os.path.join(HERE, "..", "062-copy-survival-correlation", "explore_results.json"), encoding="utf8"))
r51 = json.load(open(os.path.join(HERE, "..", "051-seshat-stage-loss", "results.json"), encoding="utf8"))
R = {"study": "063-post-collapse-reading", "seed": 20261063}

G61 = G62["G61"]; G60 = G61["G60"]
N = G62["N"]; deltas_t = G62["deltas_t"]; ver = G62["ver"]; get = G62["get"]; rho1 = G62["rho1"]
taup_061 = G62["taup_061"]; tau_used = G62["tau_used"]; tau_census = G62["tau_census"]; fp = G62["fp"]; r_reach = G62["r_reach"]
h_R = G62["h_R"]; q_h = G62["q_h"]; C_T = G62["C_T"]; C_P = G62["C_P"]; qf = G62["qf"]
KEYS = ["A_Ord/U3", "A_XPT_S/U2"]; PARTS = ("total", "heavy", "light"); M1 = "A_Ord/U3"

# 検査：062 の主（p62）の重い崩壊などが差0で出る
chk = deltas_t(ver(rho=G62["rho_new"]()))
d = max(abs(chk[k.split("/")[0]][k.split("/")[1]]["TC"][p] / get("V0", k)["TC"][p] - r62["ratio_to_V0"]["main"]["p62"][k]["TC"][p]) for k in KEYS for p in PARTS)
R["check_062_p62"] = {"max_abs_diff": d, "reproduced": d < 1e-12}

# 重い崩壊の標本の印：V_t は T=max(T2s,0) だけを渡すので、成分ごとの T2s と照合して状態を引く
B = G60["B"]; COMPS = G60["COMPS"]
TK = [(k, np.maximum(B[k]["T2s"], 0.0), np.isin(B[k]["st"], (4, 5))) for k in COMPS]
amb = sum(1 for i in range(len(TK)) for j in range(i + 1, len(TK)) if np.array_equal(TK[i][1], TK[j][1]))
R["state_lookup"] = {"n_comps": len(TK), "identical_T_pairs": amb}
def heavy_mask(T):
    for k, t, hm in TK:
        if t.shape == T.shape and np.array_equal(t, T): return hm
    raise RuntimeError("T not matched")

# ---------- 資料 ----------
CJ = json.load(open(os.path.join(HERE, "sources", "cases.json"), encoding="utf8"))
SL = []
for c in CJ["script_loss"]:
    last = np.mean(c["last"]); end = c["read"] if c["read"] is not None else 2026
    SL.append({"id": c["id"], "last_mid": float(last), "years": float(end - last), "event": c["read"] is not None, "successor_read": c["successor_read"]})
n_ev = sum(s["event"] for s in SL); expo = sum(s["years"] for s in SL)
R["script_loss"] = {"cases": SL, "n_events": n_ev, "exposure_years": expo, "tau_mle": expo / n_ev,
                    "median_years_if_exponential_mle": expo / n_ev * np.log(2), "deciphered_lags": sorted(s["years"] for s in SL if s["event"]),
                    "successor_read_count": sum(s["successor_read"] for s in SL)}
SR = []
for c in CJ["search"]:
    lo = c["search"][0] - c["collapse"][0]; hi = c["search"][1] - c["collapse"][0]
    SR.append({"id": c["id"], "lag_range": [lo, hi], "lag_mid": (lo + hi) / 2})
ls = np.log([s["lag_mid"] for s in SR]); mu_s, sd_s = float(ls.mean()), float(ls.std(ddof=1))
R["search"] = {"cases": SR, "median_mid": float(np.median([s["lag_mid"] for s in SR])), "logmean": mu_s, "logsd": sd_s}
ST = [{"id": c["id"], "years": c["found"] - np.mean(c["copy"])} for c in CJ["stack"] if c["copy"]]
R["stack"] = {"cases": ST, "median": float(np.median([s["years"] for s in ST]))}
# 051：深い喪失（S2 か S4）のうち文字も失った割合
EP = r51["loss_episodes_main"]
deep = [e for e in EP if (e.get("S2") == 1 or e.get("S4") == 1) and e.get("S1") is not None]
k_loss = sum(e["S1"] == 1 for e in deep); n_deep = len(deep)
alt = [e for e in EP if (e.get("S2") == 1 or e.get("S4") == 1 or e.get("S4b") == 1) and e.get("S1") is not None]
k_alt = sum(e["S1"] == 1 for e in alt)
R["seshat_conditional"] = {"deep_S2orS4": {"n": n_deep, "lost_writing": k_loss, "ids": [e["PolID"] for e in deep]},
                           "deep_S2orS4orS4b": {"n": len(alt), "lost_writing": k_alt}, "all_S1": {"n": 57, "lost_writing": 2},
                           "n_loss_episodes": len(EP)}
T2s_heavy_U3_median = float(r62.get("T2s_heavy_U3_median", 843.0))

# ---------- 新しい乱数 ----------
g = np.random.default_rng(20261063)
f_i = g.uniform(0.5, 1.0, N); f_pop = g.uniform(0.8, 1.0, N)
pc_main = 1 - g.beta(k_loss + 0.5, n_deep - k_loss + 0.5, N)
pc_alt = 1 - g.beta(k_alt + 0.5, len(alt) - k_alt + 0.5, N)
pc_all = 1 - g.beta(2 + 0.5, 55 + 0.5, N)
tau_dec = 1 / g.gamma(n_ev, 1 / expo, N)      # 指数の率の事後 Gamma(n_ev, expo)（平坦な事前）
srch_lag = np.exp(mu_s + sd_s * np.sqrt(1 + 1 / len(SR)) * g.standard_normal(N))
R["draws"] = {"p_cont_main_mean": float(pc_main.mean()), "p_cont_alt_mean": float(pc_alt.mean()), "p_cont_all_mean": float(pc_all.mean()),
              "tau_dec_q": [float(x) for x in np.quantile(tau_dec, [0.05, 0.5, 0.95])],
              "decipher_median_years_q": [float(x) for x in np.quantile(tau_dec * np.log(2), [0.05, 0.5, 0.95])],
              "srch_lag_q": [float(x) for x in np.quantile(srch_lag, [0.05, 0.5, 0.95])]}

def S_ind(T, n_p, K, f, tp):
    Ke = min(n_p, K); m = n_p / Ke; a = np.exp(-T / tp) * f
    return 1 - ((1 - h_R) * (1 - a) ** m + h_R * (1 - q_h * a) ** m) ** Ke
def R_common(T, pc, dec="exp", primer=None, srch="tied"):
    hm = heavy_mask(T)
    if primer is not None: pd = np.full(N, float(primer))
    elif dec == "exp": pd = 1 - np.exp(-T / tau_dec)
    else: pd = np.zeros(N)
    read = pc + (1 - pc) * pd
    s = np.ones(N) if srch == "tied" else (T >= srch_lag).astype(float)
    return np.where(hm, s * read, 1.0)
def rho63(n_p=100, K=20, pc=None, f=None, tp=None, **kw):
    pcv = pc_main if pc is None else pc; fv = f_i if f is None else f; tpv = taup_061 if tp is None else tp
    return lambda T: 1 - (1 - rho1) * (1 - r_reach * R_common(T, pcv, **kw) * S_ind(T, n_p, K, fv, tpv))

# ---------- 版 ----------
NV = {"main": dict(), "pc_alt": dict(pc=pc_alt), "pc_all": dict(pc=pc_all), "pc_half": dict(pc=np.full(N, 0.5)), "pc_one": dict(pc=np.ones(N)),
      "strict": dict(dec="none"), "indep": dict(srch="indep"), "f_061": dict(f=fp), "census_tau": dict(tp=tau_census),
      "popular": dict(f=f_pop, tp=tau_used)}
VALS = {nm: deltas_t(ver(rho=rho63(**kw))) for nm, kw in NV.items()}
def g2(nm, key): a, u = key.split("/"); return VALS[nm][a][u]
R["ratio_to_V0"] = {nm: {key: {x: {p: g2(nm, key)[x][p] / get("V0", key)[x][p] for p in PARTS} for x in ("TAB", "TC")} for key in KEYS} for nm in NV}
R["ratio_062"] = {"main_nopointer": {key: r62["ratio_to_V0"]["main"]["p62"][key]["TC"] for key in KEYS},
                  "E1_fp_common": x62["versions"]["E1_fp_common"], "main_heavy_nopointer": x62["main_heavy"]}
def ratio(o, key, p):
    a, u = key.split("/"); return deltas_t(o)[a][u]["TC"][p] / get("V0", key)["TC"][p]
R["scan_np"] = [{"n_p": n, M1: {p: ratio(ver(rho=rho63(n_p=n)), M1, p) for p in ("heavy", "light")}} for n in [1, 3, 10, 30, 100, 1000]]
R["scan_primer"] = [{"p_primer": pp, M1: {p: ratio(ver(rho=rho63(primer=pp)), M1, p) for p in ("heavy", "light")}} for pp in [0.0, 0.25, 0.5, 0.75, 1.0]]
R["scan_pc"] = [{"p_cont": pv, M1: {p: ratio(ver(rho=rho63(pc=np.full(N, pv))), M1, p) for p in ("heavy",)}} for pv in [0.0, 0.25, 0.5, 0.75, 1.0]]
m_heavy = x62["main_heavy"]
rows = {r["n_p"]: r[M1]["heavy"] for r in R["scan_np"]}
R["saturation_10_of_1000"] = (rows[10] - m_heavy) / (rows[1000] - m_heavy)

# ---------- 費用あたり ----------
R["CE"] = {}
for key in KEYS:
    R["CE"][key] = {}
    for p in PARTS:
        vt = get("main", key)["TAB"][p]; vtc = get("main", key)["TC"][p]; dd = {}
        for nm in NV:
            dv = g2(nm, key)["TC"][p] - vtc
            dd[nm] = {"dV_over_TAB": dv / vt, "CE_q": qf(dv / vt * C_T / C_P(100)) if dv > 0 else [0.0, 0.0, 0.0]}
        dd["p62"] = {"CE_q": r62["CE"]["main"][key][p]["p62"]["CE_q"]}
        R["CE"][key][p] = dd
s_ce = r62["CE_STC_060"]; R["CE_STC_060"] = s_ce

# ---------- 仮説 ----------
rt = R["ratio_to_V0"]; H = {}
v = float(np.median(tau_dec * np.log(2))); H["H75a"] = {"value": v, "rule": "posterior median decipher lag > 843", "supported": v > 843}
v = R["search"]["median_mid"]; H["H75b"] = {"value": v, "rule": "median search lag < 843", "supported": v < 843}
v = R["draws"]["p_cont_main_mean"]; H["H75c"] = {"value": v, "rule": "mean p_cont (main) <= 0.8", "supported": v <= 0.8}
v = rt["main"][M1]["TC"]["heavy"]; lo, hi = x62["versions"]["E1_fp_common"]["heavy"], r62["ratio_to_V0"]["main"]["p62"][M1]["TC"]["heavy"]
H["H75d"] = {"value": [v, lo, hi], "rule": "E1(062) <= heavy main <= p62", "supported": lo <= v <= hi}
v = R["saturation_10_of_1000"]; H["H75e"] = {"value": v, "rule": "gain n_p=10 >= 60% of n_p=1000", "supported": v >= 0.6}
pr = {r["p_primer"]: r[M1]["heavy"] for r in R["scan_primer"]}; v = pr[1.0] - pr[0.0]
H["H75f"] = {"value": v, "rule": "primer 0->1 raises heavy >= 0.03", "supported": v >= 0.03}
v = [rt["popular"][M1]["TC"]["heavy"], rt["main"][M1]["TC"]["heavy"]]; H["H75g"] = {"value": v, "rule": "popular >= main (heavy)", "supported": v[0] >= v[1]}
v = [R["CE"][M1]["total"]["main"]["CE_q"][1], s_ce]; H["H75h"] = {"value": v, "rule": "CE main > S_TC|TC", "supported": v[0] > v[1]}
v = rt["main"][M1]["TC"]["heavy"] - rt["indep"][M1]["TC"]["heavy"]; H["H75i"] = {"value": v, "rule": "indep lowers heavy >= 0.03", "supported": v >= 0.03}
for h in H.values(): h["supported"] = bool(h["supported"])
R["hypotheses"] = H

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
cs = sorted(SL, key=lambda s: s["years"]); ys = np.arange(len(cs))
axs[0].barh(ys, [s["years"] for s in cs], color=["C0" if s["event"] else "0.7" for s in cs]); axs[0].set_yticks(ys); axs[0].set_yticklabels([s["id"] for s in cs], fontsize=7)
for s in SR: axs[0].plot(s["lag_range"], [len(cs) + 0.2 + SR.index(s) * 0.5] * 2, "C3-", lw=3)
axs[0].axvline(843, color="k", ls="--", lw=0.8, label="U3 heavy T2s median (843 y)")
axs[0].set_xlabel("years: script loss -> modern reading (blue; grey = still unread)\nred: collapse -> search of old books (continuity cases)"); axs[0].legend(fontsize=7)
Ts = np.linspace(0, 3000, 301)
for pv, col in ((float(pc_main.mean()), "C0"), (0.5, "C1"), (1.0, "C2")):
    axs[1].plot(Ts, [pv + (1 - pv) * (1 - np.exp(-t / tau_dec)).mean() for t in Ts], color=col, label=f"p_cont={pv:.2f}, decipher-exp")
    axs[1].plot(Ts, [pv * np.ones(1)[0] for t in Ts], color=col, ls=":", lw=0.8)
axs[1].plot(Ts, [(srch_lag <= t).mean() for t in Ts], "C3--", label="P(search started) indep")
axs[1].set_xlabel("years from collapse T"); axs[1].set_ylabel("common read factor R(T) (heavy)"); axs[1].set_ylim(0, 1.02); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_cases_R.png"), dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].semilogx([r["n_p"] for r in R["scan_np"]], [r[M1]["heavy"] for r in R["scan_np"]], "-o", label="063 main")
axs[0].semilogx([r["n_p"] for r in r62["scan_np"]], [r[M1]["heavy"] for r in r62["scan_np"]], "--s", color="k", label="062 main")
axs[0].axhline(m_heavy, color="0.5", lw=0.7); axs[0].set_xlabel("copies n_p"); axs[0].set_ylabel("heavy TC / V0 (A_Ord/U3)"); axs[0].legend(fontsize=7)
axs[1].plot([r["p_primer"] for r in R["scan_primer"]], [r[M1]["heavy"] for r in R["scan_primer"]], "-o", color="C2", label="primer readability if script lost")
axs[1].plot([r["p_cont"] for r in R["scan_pc"]], [r[M1]["heavy"] for r in R["scan_pc"]], "-^", color="C3", label="p_cont (script continuity)")
axs[1].axhline(m_heavy, color="0.5", lw=0.7); axs[1].set_xlabel("probability"); axs[1].set_ylabel("heavy TC / V0"); axs[1].legend(fontsize=7)
names = list(NV); vals = [rt[nm][M1]["TC"]["heavy"] for nm in names]
axs[2].barh(range(len(names)), vals); axs[2].set_yticks(range(len(names))); axs[2].set_yticklabels(names, fontsize=7)
axs[2].axvline(r62["ratio_to_V0"]["main"]["p62"][M1]["TC"]["heavy"], color="k", ls="--", lw=0.8, label="062 main")
axs[2].axvline(x62["versions"]["E1_fp_common"]["heavy"], color="C1", ls=":", label="062 E1"); axs[2].axvline(m_heavy, color="0.5", lw=0.7, label="no pointer")
axs[2].set_xlim(0.55, 0.95); axs[2].set_xlabel("heavy TC / V0"); axs[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_versions_scans.png"), dpi=130); plt.close(fig)

R["runtime_s"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float, ensure_ascii=False)
print("check", R["check_062_p62"], R["state_lookup"])
print("script_loss", json.dumps({k: v for k, v in R["script_loss"].items() if k != "cases"}))
print("search", json.dumps(R["search"])); print("stack", json.dumps(R["stack"])); print("seshat", json.dumps(R["seshat_conditional"]))
print("draws", json.dumps(R["draws"]))
for nm in NV: print(nm, {key: {p: round(rt[nm][key]["TC"][p], 3) for p in PARTS} for key in KEYS})
print("scan_np", json.dumps(R["scan_np"])); print("scan_primer", json.dumps(R["scan_primer"])); print("scan_pc", json.dumps(R["scan_pc"]))
print("sat10/1000", R["saturation_10_of_1000"])
print("CE total", json.dumps({nm: R["CE"][M1]["total"][nm]["CE_q"] for nm in R["CE"][M1]["total"]})); print("CE heavy", json.dumps({nm: R["CE"][M1]["heavy"][nm]["CE_q"] for nm in R["CE"][M1]["heavy"]}))
for h, dd in H.items(): print(h, dd["supported"], dd["value"])
print("runtime", R["runtime_s"])
