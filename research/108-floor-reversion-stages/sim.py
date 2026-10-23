"""研究108：損失が床への戻りなら段は消えずに低い釣り合いに落ちる。段を連続の技能で置き、重い崩壊の段の喪失・T2 到達・F_heavy を出し直す（open_problems 160）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md（2026-10-24 05:44 JST）。
104 の sim.py を「# ---------- 仮説」の前まで exec（057→…→038 の乱数をそのまま再現）。新しい乱数は seed 20261108。
模型：dz/dt = κ(ln N(t) − z)、N(t)=n e^{rt}、z(0)=ln N_pre。段 k を保つ ⇔ min_t z(t) ≥ ln N_k。
  z_min = ln n + (r/κ) ln(κA/r)（κA/r>1）、そうでなければ z(0)。A = ln N_pre − ln n + r/κ。κ→∞ で 038 の二値の割り当てに戻る。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P104 = os.path.abspath(os.path.join(HERE, "..", "104-ruin-glass-thresholds", "sim.py"))
src = open(P104, encoding="utf8").read(); cut = src.index("# ---------- 仮説")
G = {"__file__": P104, "__name__": "sim104"}
exec(compile(src[:cut], P104, "exec"), G)
R104 = json.load(open(os.path.join(os.path.dirname(P104), "results.json"), encoding="utf8"))
K = "A_Ord/U3"
d104 = max(abs(G["R"]["runs"][t][K]["F_heavy"] - R104["runs"][t][K]["F_heavy"]) for t in R104["runs"])
assert d104 < 1e-12
N, B, SRC, COMPS, FIXED, agg, V_m, q_TC, v0 = (G[k] for k in ("N", "B", "SRC", "COMPS", "FIXED", "agg", "V_m", "q_TC", "v0"))
U_lit, d01, d12, lit_mult, d_Fe, W = (G[k] for k in ("U_lit", "d01", "d12", "lit_mult", "d_Fe", "W"))
N_Fe, N_lens, N_T2e, r_pop = (G[k] for k in ("N_Fe", "N_lens", "N_T2e", "r_pop"))
R = {"study": "108-floor-reversion-stages", "seed": 20261108, "check_104_runs_reproduced_maxdiff": d104}
g = np.random.default_rng(20261108)
q3 = lambda v: [float(np.quantile(v, p)) for p in (0.05, 0.5, 0.95)]
POP_W = 8.232e9

# ---------- 0. 038 の二値の割り当てと st が一致する標本だけを動かす ----------
def level(x):   # x = ln n か z_min。段の番号（3=T2 を保つ、4=T1、5=T0 鉄、6=鉄の前）
    return np.select([x >= np.log(N_T2e), x >= np.log(N_lens), x >= np.log(N_Fe)], [3, 4, 5], default=6)
CONS = {}
for k in COMPS:
    if k in FIXED: continue
    st = B[k]["st"]; n = np.maximum(SRC[k]["sv"]["nmax"], 1.0); h = np.isin(st, (4, 5, 6))
    CONS[k] = h & (level(np.log(n)) == st)
R["consistency"] = {k: {"n_heavy": int(np.isin(B[k]["st"], (4, 5, 6)).sum()), "share_consistent": float(CONS[k][np.isin(B[k]["st"], (4, 5, 6))].mean()) if np.isin(B[k]["st"], (4, 5, 6)).any() else None} for k in CONS}

# ---------- 1. 床への戻りの最小値 ----------
def zmin(n, r, kap, Npre):
    lnn = np.log(n); A = np.log(Npre) - lnn + r / kap
    x = kap * A / r
    return np.where(x > 1, lnn + (r / kap) * np.log(np.maximum(x, 1)), np.log(Npre)), np.where(x > 1, np.exp(-(r / kap) * np.log(np.maximum(x, 1))), 0.0)

def new_st(k, kap, Npre):
    st = B[k]["st"]; n = np.maximum(SRC[k]["sv"]["nmax"], 1.0)
    zm, mult = zmin(n, r_pop, kap, Npre)
    s2 = np.where(CONS[k], np.minimum(st, level(zm)), st)
    return s2, mult

def T2_from(k, st, lag=None):
    if k in FIXED: return B[k]["T2s"]
    sv = SRC[k]["sv"]; lit = SRC[k]["lit"]; n = np.maximum(sv["nmax"], 1.0)
    add = 0.0 if lag is None else lag
    wait = lambda Nk: np.maximum(0.0, np.log(Nk / n) / r_pop + add)
    d01e = np.where(U_lit < lit, d01 * lit_mult, d01)
    tFe = np.where(st == 6, np.maximum(d_Fe, wait(N_Fe)), 0.0)
    t1 = np.where(st >= 5, np.maximum(tFe + d01e, wait(N_lens)), 0.0)
    t2 = np.where(st >= 4, np.maximum(t1 + d12, wait(N_T2e)), 0.0)
    return np.select([st == 2, st == 3, st >= 4], [W, np.zeros(N), t2], default=np.zeros(N))
R["check_T2_from_eq_B"] = max(float(np.abs(T2_from(k, B[k]["st"]) - B[k]["T2s"]).max()) for k in COMPS)
assert R["check_T2_from_eq_B"] == 0.0

def run(tag, kap, Npre=POP_W, lag=False):
    D = {}; out = {"shares": {}}; bT = []; nT = []; mults = []
    for k in COMPS:
        if k in FIXED:
            T2n = B[k]["T2s"]
        else:
            s2, mult = new_st(k, kap, Npre)
            T2n = T2_from(k, s2, lag=(1.0 / kap) if lag else None)
            h0 = np.isin(B[k]["st"], (4, 5, 6))
            if h0.any():
                bT.append(B[k]["T2s"][h0]); nT.append(T2n[h0]); mults.append(mult[h0])
            out["shares"][k] = {"heavy_base": float(h0.mean()), "heavy_new": float(np.isin(s2, (4, 5, 6)).mean()),
                                "T0_base": float(np.isin(B[k]["st"], (5, 6)).mean()), "T0_new": float(np.isin(s2, (5, 6)).mean())}
        D[k] = V_m(k, q_TC, True, T2n) - V_m(k, 0.3, False, T2n)
    v1 = agg({"x": D}, anchors=("A_Ord", "A_XPT_S"))
    for a, u in (("A_Ord", "U3"), ("A_XPT_S", "U2")):
        b_ = v0[a][u]["x"]; n_ = v1[a][u]["x"]
        out[f"{a}/{u}"] = {"F_total": n_["total"] / b_["total"], "F_heavy": n_["heavy"] / b_["heavy"]}
    b6 = np.concatenate(bT); n6 = np.concatenate(nT); m6 = np.concatenate(mults)
    out["heavy_T2_median_base"] = float(np.median(b6)); out["heavy_T2_median_new"] = float(np.median(n6))
    out["heavy_T2_median_rel"] = float(np.median(n6) / np.median(b6) - 1)
    out["thr_mult_heavy_q"] = q3(m6)
    sh = out["shares"]["U3"]
    out["U3_heavy_rel_drop"] = 1 - sh["heavy_new"] / sh["heavy_base"]
    out["U3_T0_rel_drop"] = 1 - sh["T0_new"] / sh["T0_base"]
    R.setdefault("runs", {})[tag] = out
    return out

# ---------- 2. κ の版 ----------
# 【事実（059 evidence、Muthukrishna 2014 実験2）】床への戻り a = 0.39（1手本）、0.37（5手本）／世代
# 【事実（Wikipedia Generation）】世代は "generally considered to be about 20–30 years"
a_lab = g.uniform(0.37, 0.39, N); Tg = g.uniform(20, 30, N)
k_lab = a_lab / Tg
# 【事実（Wikipedia Aboriginal Tasmanians）】孤立は約 8,000 年（Cosgrove 1995：約 6000 BCE の水没）、別の記述では 14,000〜12,000 BCE の水没（Hamacher 2023）。
#   骨角器と魚の喪失は約 4,000 年前（Ryan 1996、Manne 2003）。原因は経済の変化とする説もあり争いがある。人口は 3,000〜15,000（Madley 2008）。
# 人口一定の群：z(t) = ln n + (z0 − ln n)e^{−κt}、喪失は z = ln N_k のとき → κ = ln R / T_loss、R = (z0−ln n)/(ln N_k−ln n) 【推測：R ∈ [1.1, 10]】
T_main = g.uniform(3500, 4500, N)                 # 8,000 − 4,000（±500 は丸めの幅【推測】）
T_alt = g.uniform(10000, 12000, N)                # 14,000〜16,000 BP の水没 − 4,000
lnR = g.uniform(np.log(1.1), np.log(10), N)
k_hist = lnR / T_main; k_hist_alt = lnR / T_alt
a_hist = k_hist * Tg
R["kappa"] = {"k_lab_per_yr_q": q3(k_lab), "a_lab": [0.37, 0.39], "Tg": [20, 30],
              "k_hist_per_yr_q": q3(k_hist), "a_hist_per_gen_q": q3(a_hist), "a_hist_alt_per_gen_q": q3(k_hist_alt * Tg),
              "a_hist_upper_R10_T3500_Tg30": float(np.log(10) / 3500 * 30),
              "R_needed_for_a_0.05_T4000_Tg25": float(np.exp(0.05 / 25 * 4000)),
              "ratio_lab_over_hist_median": float(np.median(k_lab) / np.median(k_hist))}

BIG = 1e9
base = run("binary_kappa_inf", np.full(N, BIG))
R["check_binary_F_heavy"] = abs(base[K]["F_heavy"] - 1.0)
assert R["check_binary_F_heavy"] < 1e-9
run("lab", k_lab); run("lab_Npre1e7", k_lab, Npre=1e7); run("lab_lag", k_lab, lag=True)
run("hist", k_hist); run("hist_Npre1e7", k_hist, Npre=1e7); run("hist_lag", k_hist, lag=True); run("hist_alt", k_hist_alt)
run("lab_x0.1", k_lab * 0.1)
# 走査：κ を一定値で
KS = np.logspace(-5, 0, 26)
sweep = []
for kv in KS:
    o = run(f"sweep_{kv:.2e}", np.full(N, kv))
    sweep.append({"kappa": float(kv), "F_heavy": o[K]["F_heavy"], "U3_heavy_rel_drop": o["U3_heavy_rel_drop"], "U3_T0_rel_drop": o["U3_T0_rel_drop"],
                  "T2_rel": o["heavy_T2_median_rel"]})
R["sweep"] = sweep
# F_heavy が 1.1 を越える κ（走査で線形補間、log 軸）
fh = np.array([s["F_heavy"] for s in sweep]); lk = np.log10(KS)
idx = np.where(fh >= 1.1)[0]
R["kappa_crit_Fheavy_1.1"] = float(10 ** lk[idx.max()]) if idx.size else None
idx = np.where(np.array([s["U3_heavy_rel_drop"] for s in sweep]) >= 0.3)[0]
R["kappa_crit_heavy_drop_0.3"] = float(10 ** lk[idx.max()]) if idx.size else None

RU = R["runs"]
# ---------- 仮説 ----------
Hh = {}
v = R["kappa"]["a_hist_per_gen_q"][1]; Hh["H120a"] = {"class": "confirmatory", "value": R["kappa"]["a_hist_per_gen_q"], "rule": "median a_hist < 0.05 /gen", "supported": v < 0.05}
v = RU["hist"]["U3_heavy_rel_drop"]; Hh["H120b"] = {"class": "confirmatory", "value": v, "rule": "U3 heavy share relative drop >= 0.30 (kappa_hist)", "supported": v >= 0.30}
v = RU["lab"]["thr_mult_heavy_q"][1]; Hh["H120c"] = {"class": "prediction", "value": v, "rule": "median effective threshold multiplier in [0.2, 0.7] (lab)", "supported": 0.2 <= v <= 0.7}
v = RU["lab"]["U3_T0_rel_drop"]; Hh["H120d"] = {"class": "prediction", "value": v, "rule": "U3 T0 share relative drop < 0.20 (lab)", "supported": v < 0.20}
v = RU["lab"][K]["F_heavy"]; Hh["H120e"] = {"class": "prediction", "value": v, "rule": "F_heavy in [0.9, 1.1] (lab)", "supported": 0.9 <= v <= 1.1}
R["hypotheses"] = Hh
R["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
axs[0].semilogx(KS, fh, "o-", label="F_heavy (A_Ord/U3)")
axs[0].semilogx(KS, [1 + s["U3_heavy_rel_drop"] for s in sweep], "s--", label="1 + U3 heavy-share relative drop")
for nm, kk, c in (("lab (Muthukrishna a, Tg 20-30y)", k_lab, "C2"), ("Tasmania 4,000y", k_hist, "C3"), ("Tasmania alt 10-12ky", k_hist_alt, "C4")):
    lo, hi = np.quantile(kk, [0.05, 0.95]); axs[0].axvspan(lo, hi, color=c, alpha=0.15, label=nm)
axs[0].axhline(1, color="gray", lw=0.6); axs[0].set_xlabel("kappa = floor-reversion rate per year"); axs[0].legend(fontsize=7)
axs[0].set_title("108: stage retention under floor reversion (binary model = kappa to infinity)", fontsize=9)
names = [t for t in RU if not t.startswith("sweep")]
y = np.arange(len(names))
axs[1].barh(y - 0.2, [RU[t][K]["F_heavy"] for t in names], 0.4, label="F_heavy")
axs[1].barh(y + 0.2, [RU[t]["shares"]["U3"]["heavy_new"] / RU[t]["shares"]["U3"]["heavy_base"] for t in names], 0.4, label="U3 heavy share / binary")
axs[1].set_yticks(y); axs[1].set_yticklabels(names, fontsize=7); axs[1].axvline(1, color="gray", lw=0.6); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_floor_reversion.png"), dpi=120); plt.close(fig)

json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: R[k] for k in ("check_104_runs_reproduced_maxdiff", "check_T2_from_eq_B", "check_binary_F_heavy", "consistency", "kappa", "kappa_crit_Fheavy_1.1", "kappa_crit_heavy_drop_0.3")}, indent=1, ensure_ascii=False, default=float))
for t in names:
    o = RU[t]; print(t, "F_heavy", round(o[K]["F_heavy"], 4), "F_total", round(o[K]["F_total"], 4), "U3 heavy", round(o["shares"]["U3"]["heavy_new"], 4), "drop", round(o["U3_heavy_rel_drop"], 3),
                    "T0 drop", round(o["U3_T0_rel_drop"], 3), "T2rel", round(o["heavy_T2_median_rel"], 3), "mult", [round(x, 3) for x in o["thr_mult_heavy_q"]])
for s in sweep: print("sweep", f"{s['kappa']:.2e}", round(s["F_heavy"], 4), round(s["U3_heavy_rel_drop"], 3), round(s["U3_T0_rel_drop"], 3))
for k, h in Hh.items(): print(k, h["class"], h["supported"], json.dumps(h["value"], default=float)[:120])
print("runtime", round(R["runtime_s"], 1))
