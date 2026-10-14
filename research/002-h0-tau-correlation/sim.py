"""002: h0–τ 相関と増加型（パルス）ハザード
モデル: h(t) = h0*exp(-t/tau) + h_nat + h_pulse(t)
再現: python3 sim.py  (seed=20261014) -> results.json, fig_*.png   要 numpy, matplotlib
"""
import json, math, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

SEED = 20261014
N = 200_000           # 解析式で積分できる部分
N_DAMAGE = 100_000    # 時間刻みが必要な脆弱化モデル
SIGMA = 1.0
RHOS = [-0.8, -0.4, 0.0, 0.4, 0.8]
TS = [100, 1_000, 10_000, 100_000]

# --- 文献アンカー（001と同一） ---
h0_ord = 1 - (5/6)**(1/100)   # Ord(2020) 今世紀1/6 を年率化
TAU_MED = 200.0               # 001の仮定
h_nat = 1/87_000              # Snyder-Beattie et al.(2019) の上限
# XPT (Karger et al. 2023, Table 3) 2100年までの総絶滅リスク中央値: 超予測者1%, 専門家6%
# 期間は 2023-2100 の約78年と近似（推測）
XPT_YEARS = 78
h0_xpt_sf  = 1 - (1-0.01)**(1/XPT_YEARS)
h0_xpt_exp = 1 - (1-0.06)**(1/XPT_YEARS)

rng = np.random.default_rng(SEED)
Z1 = rng.normal(size=N); W = rng.normal(size=N)   # 共通乱数（全ρで同じZ1,Wを使う）

def draw(rho, sigma=SIGMA, k=1.0, z1=Z1, w=W, h0med=None):
    z2 = rho*z1 + math.sqrt(1-rho**2)*w
    m = h0_ord*k if h0med is None else h0med
    return m*np.exp(sigma*z1), TAU_MED*np.exp(sigma*z2)

def H_peril(T, h0, tau):
    return h0*tau*(-np.expm1(-T/tau))

# --- パルス: 累積ハザード P(T)=∫_0^T h_pulse。面積 = A*w に正規化 ---
def _Phi(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
def pulse_cum(T, A, w, t0, shape):
    area = A*w
    if shape == "rect":
        return A*min(max(T-t0, 0.0), w)
    if shape == "tri":   # [t0,t0+w] の三角形, ピーク 2A
        x = min(max((T-t0)/w, 0.0), 1.0)
        frac = 2*x*x if x <= 0.5 else 1-2*(1-x)**2
        return area*frac
    if shape == "gauss": # 中心 t0+w/2, sd=w/sqrt(12)（矩形と同分散）, t<0 を切って面積を再正規化
        c, s = t0+w/2, w/math.sqrt(12)
        lo = _Phi((0-c)/s)
        return area*(_Phi((T-c)/s)-lo)/(1-lo)
    raise ValueError(shape)

def S_mc(T, h0, tau, P=0.0, hn=h_nat):
    x = np.exp(-(H_peril(T, h0, tau) + P + hn*T))
    return float(x.mean()), float(x.std(ddof=1)/math.sqrt(x.size))

res = {"seed": SEED, "N": N, "N_damage": N_DAMAGE, "sigma": SIGMA,
       "h0_ord_annual": h0_ord, "tau_median": TAU_MED, "h_nat_upper": h_nat,
       "h0_xpt_superforecasters_annual": h0_xpt_sf, "h0_xpt_experts_annual": h0_xpt_exp}

# ============ (a) ρ 依存 ============
corr = {}
for rho in RHOS:
    h0, tau = draw(rho)
    d = {"empirical_corr_log": float(np.corrcoef(np.log(h0), np.log(tau))[0,1])}
    for T in TS:
        m, se = S_mc(T, h0, tau); d[f"S_{T}"] = m; d[f"SE_{T}"] = se
    # H5検証: 同じ Var(log h0τ)=σ²(2+2ρ) をもつ独立モデル（各σ'=σ√(1+ρ)）
    sp = SIGMA*math.sqrt(1+rho)
    h0e, taue = draw(0.0, sigma=sp)
    for T in TS: d[f"S_{T}_eqvar_indep"] = S_mc(T, h0e, taue)[0]
    # 「強いが短い」象限（h0>中央値 かつ τ<中央値）
    q = (h0 > h0_ord) & (tau < TAU_MED)
    s4 = np.exp(-(H_peril(1e4, h0, tau)+h_nat*1e4))
    d["frac_strong_short_worlds"] = float(q.mean())
    d["survival_share_strong_short_1e4"] = float(s4[q].sum()/s4.sum())
    d["sd_log_h0tau"] = float(np.std(np.log(h0*tau)))
    corr[rho] = d
res["rho_scan"] = corr
res["point_estimate"] = {f"S_{T}": float(math.exp(-(h0_ord*TAU_MED*(-math.expm1(-T/TAU_MED))+h_nat*T))) for T in TS}

# σ 感度（S(1e4)）
res["sigma_x_rho_S1e4"] = {s: {rho: S_mc(1e4, *draw(rho, sigma=s))[0] for rho in RHOS} for s in [0.5, 1.0, 1.5, 2.0]}

# H4 反例探索: h0 中央値を k 倍。中央値の累積危険 m=k*h0_ord*τ が大きいと符号が反転するか
ks = [0.1, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 10]
kscan = {}
for k in ks:
    kscan[k] = {"median_peril_S": math.exp(-k*h0_ord*TAU_MED),
                **{f"S1e4_rho{rho}": S_mc(1e4, *draw(rho, k=k))[0] for rho in [-0.8, 0.0, 0.8]}}
res["h0_scale_scan_S1e4"] = kscan
def find_flip(sigma, lo_k=0.3, hi_k=30.0):
    gap = lambda k: S_mc(1e4, *draw(0.8, sigma=sigma, k=k))[0] - S_mc(1e4, *draw(-0.8, sigma=sigma, k=k))[0]
    if gap(lo_k)*gap(hi_k) >= 0: return None
    for _ in range(40):
        mid = math.sqrt(lo_k*hi_k)
        if gap(mid)*gap(lo_k) > 0: lo_k = mid
        else: hi_k = mid
    return math.sqrt(lo_k*hi_k)
kstar = find_flip(SIGMA)
res["H4_sign_flip_k"] = kstar
res["H4_sign_flip_median_cum_hazard"] = kstar*h0_ord*TAU_MED if kstar else None
res["H4_sign_flip_by_sigma"] = {}
for s_ in [0.5, 1.0, 2.0]:
    kk = find_flip(s_)
    res["H4_sign_flip_by_sigma"][s_] = {"k": kk, "median_cum_hazard": kk*h0_ord*TAU_MED if kk else None,
                                        "median_peril_S": math.exp(-kk*h0_ord*TAU_MED) if kk else None}
# 文献アンカー別（Ord / XPT超予測者 / XPT専門家）
anch = {}
for name, hm in [("Ord_1/6", h0_ord), ("XPT_superforecasters_1pct", h0_xpt_sf), ("XPT_experts_6pct", h0_xpt_exp)]:
    anch[name] = {"h0_median": hm, **{f"S1e4_rho{rho}": S_mc(1e4, *draw(rho, h0med=hm))[0] for rho in RHOS},
                  **{f"S100_rho{rho}": S_mc(100, *draw(rho, h0med=hm))[0] for rho in RHOS}}
res["anchors_S"] = anch

# ============ (b) パルス ============
h0i, taui = draw(0.0)
base = {T: S_mc(T, h0i, taui)[0] for T in TS}
pulse = []
for Am in [0.1, 0.5, 1.0, 2.0]:
    for w in [30, 100, 300]:
        for t0 in [0, 50, 200]:
            for sh in ["rect", "tri", "gauss"]:
                A = Am*h0_ord
                r = {"A_mult_h0": Am, "A": A, "w": w, "t0": t0, "shape": sh, "area": A*w}
                for T in TS:
                    r[f"S_{T}"] = S_mc(T, h0i, taui, P=pulse_cum(T, A, w, t0, sh))[0]
                    r[f"dS_{T}"] = r[f"S_{T}"] - base[T]
                pulse.append(r)
res["pulse_baseline_rho0"] = {f"S_{T}": base[T] for T in TS}
res["pulse_grid_rho0"] = pulse
# 同面積 (A*w = h0_ord*100) で幅を変える
eqa = []
for Am, w in [(2, 50), (1, 100), (0.5, 200), (0.1, 1000), (0.01, 10000), (0.005, 20000)]:
    A = Am*h0_ord; r = {"A_mult_h0": Am, "w": w, "t0": 0, "shape": "rect", "area": A*w}
    for T in TS: r[f"S_{T}"] = S_mc(T, h0i, taui, P=pulse_cum(T, A, w, 0, "rect"))[0]
    eqa.append(r)
res["pulse_equal_area"] = eqa
# 文献規模のパルス（「現行推定に含まれない追加リスク」が100年(XPTは78年)で発現すると仮定）
lit = {}
for name, p, yrs in [("Ord_unknown_1/30", 1/30, 100), ("Ord_AI_1/10", 1/10, 100),
                     ("XPT_AI_experts_3pct", 0.03, XPT_YEARS), ("XPT_AI_superforecasters_0.38pct", 0.0038, XPT_YEARS)]:
    A = 1-(1-p)**(1/yrs)
    lit[name] = {"A": A, "w": yrs, **{f"S_{T}": S_mc(T, h0i, taui, P=pulse_cum(T, A, yrs, 0, "rect"))[0] for T in TS}}
res["pulse_literature_scale"] = lit
# パルス × ρ: 固定振幅 と 世界ごとに h0 に比例する振幅（A_i = h0_i）
pxr = {}
for rho in RHOS:
    h0, tau = draw(rho)
    s0 = S_mc(1e4, h0, tau)[0]
    s_fix = S_mc(1e4, h0, tau, P=h0_ord*100)[0]
    s_cpl = S_mc(1e4, h0, tau, P=h0*100)[0]
    pxr[rho] = {"S1e4_nopulse": s0, "S1e4_fixed": s_fix, "ratio_fixed": s_fix/s0,
                "S1e4_coupled": s_cpl, "ratio_coupled": s_cpl/s0}
res["pulse_x_rho_S1e4"] = pxr

# ============ (c) 反例探索: 脆弱化(damage)モデル ============
# 人為ハザード（危険の時代項+パルス）は、絶滅率の c 倍の率で非致死的大災害を起こし、
# 大災害1回ごとに以後の「危険の時代」項が (1+δ) 倍になる（自己励起）。
# c は XPT の大災害/絶滅 中央値比（専門家 20/6≈3.33, 超予測者 9.05/1）をアンカー。δ は仮定。
grid = np.unique(np.concatenate([np.arange(0, 1000, 1.0), np.arange(1000, 20000, 10.0),
                                 np.arange(20000, 100001, 100.0)]))
def damage_run(c, delta, pulse_spec, seed=SEED+1):
    r2 = np.random.default_rng(seed)
    z1 = r2.normal(size=N_DAMAGE); w_ = r2.normal(size=N_DAMAGE)
    h0, tau = draw(0.0, z1=z1, w=w_)
    mult = np.ones(N_DAMAGE); logS = np.zeros(N_DAMAGE); ncat = np.zeros(N_DAMAGE)
    out = {}; targets = set(TS)
    Pprev = 0.0
    for a, b in zip(grid[:-1], grid[1:]):
        hp = mult*h0*tau*(np.exp(-a/tau)-np.exp(-b/tau))
        if pulse_spec is not None:
            Pb = pulse_cum(b, *pulse_spec); dP = Pb-Pprev; Pprev = Pb
        else: dP = 0.0
        Hstep = hp + dP
        logS -= Hstep
        if delta > 0:
            u = r2.random(N_DAMAGE)              # 共通乱数（シナリオ間で同じ系列）
            alive = logS > -50.0                 # 生存重み<2e-22 の世界は凍結（数値オーバーフロー防止。結果への寄与は無視できる）
            ev = (u < -np.expm1(-c*Hstep)) & alive
            mult = np.where(ev, mult*(1+delta), mult); ncat += ev
        if int(b) in targets:
            x = np.exp(logS - h_nat*b)
            out[f"S_{int(b)}"] = float(x.mean()); out[f"SE_{int(b)}"] = float(x.std(ddof=1)/math.sqrt(x.size))
            out[f"mean_catastrophes_survivor_weighted_{int(b)}"] = float((ncat*x).sum()/x.sum())
    return out
dmg = {"grid_steps": int(grid.size-1)}
# 検証: δ=0 は解析式と一致するはず
dmg["validation_delta0_nopulse"] = damage_run(3.33, 0.0, None)
r2 = np.random.default_rng(SEED+1); z1 = r2.normal(size=N_DAMAGE); w_ = r2.normal(size=N_DAMAGE)
h0v, tauv = draw(0.0, z1=z1, w=w_)
dmg["validation_analytic"] = {f"S_{T}": S_mc(T, h0v, tauv)[0] for T in TS}
A1 = h0_ord
specs = {"none": None,
         "rect_t0_0_w100": (A1, 100, 0, "rect"), "tri_t0_0_w100": (A1, 100, 0, "tri"),
         "gauss_t0_0_w100": (A1, 100, 0, "gauss"),
         "rect_t0_200_w100": (A1, 100, 200, "rect"), "rect_t0_1000_w100": (A1, 100, 1000, "rect"),
         "rect_t0_0_w1000_A0.1": (0.1*A1, 1000, 0, "rect"), "rect_t0_0_w30_A3.33": (A1*100/30, 30, 0, "rect")}
for c, delta in [(3.33, 1.0), (9.05, 1.0), (3.33, 0.5)]:
    key = f"c{c}_delta{delta}"; dmg[key] = {}
    for nm, sp in specs.items():
        dmg[key][nm] = damage_run(c, delta, sp)
    # 純ハザード（δ=0）でのパルス効果（比較用）
dmg["delta0"] = {nm: damage_run(3.33, 0.0, sp) for nm, sp in specs.items() if nm in ("none", "rect_t0_0_w100", "rect_t0_1000_w100")}
res["damage_model"] = dmg

json.dump(res, open("results.json", "w"), indent=2, ensure_ascii=False)

# ============ 図 ============
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for T in TS:
    ax[0].plot(RHOS, [corr[r][f"S_{T}"] for r in RHOS], "o-", label=f"T={T:,} y")
ax[0].set_xlabel("rho = corr(log h0, log tau)"); ax[0].set_ylabel("P(survival to T)")
ax[0].set_title("S(T) vs rho (sigma=1, Ord/200y medians)"); ax[0].grid(alpha=.3); ax[0].legend()
for rho, ls in [(-0.8, "-"), (0.0, "--"), (0.8, ":")]:
    ax[1].semilogx(ks, [kscan[k][f"S1e4_rho{rho}"] for k in ks], ls, marker="o", label=f"rho={rho}")
if res.get("H4_sign_flip_k"): ax[1].axvline(res["H4_sign_flip_k"], color="gray", lw=.8)
ax[1].set_xlabel("k (h0 median = k x Ord)"); ax[1].set_ylabel("S(10,000 y)")
ax[1].set_title("Sign of rho effect flips at large k"); ax[1].grid(alpha=.3); ax[1].legend()
plt.tight_layout(); plt.savefig("fig_S_vs_rho.png", dpi=130)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for Am, col in zip([0.1, 0.5, 1.0, 2.0], ["C0", "C1", "C2", "C3"]):
    for sh, mk in [("rect", "o"), ("tri", "^"), ("gauss", "s")]:
        sel = [r for r in pulse if r["A_mult_h0"] == Am and r["shape"] == sh and r["t0"] == 0]
        ax[0].plot([r["w"] for r in sel], [-r["dS_10000"] for r in sel], mk+"-", color=col, alpha=.7,
                   label=f"A={Am}h0" if sh == "rect" else None)
ax[0].set_xscale("log"); ax[0].set_xlabel("pulse width w (y)"); ax[0].set_ylabel("loss in S(10,000 y)")
ax[0].set_title("Pulse loss (o rect, ^ tri, s gauss; t0=0)"); ax[0].legend(); ax[0].grid(alpha=.3)
tt = np.unique(np.concatenate([np.linspace(1, 600, 300), np.logspace(np.log10(600), 5, 80)]))
ax[1].semilogx(tt, [S_mc(T, h0i, taui)[0] for T in tt], "k-", label="no pulse")
for sh, t0, ls in [("rect", 0, "-"), ("gauss", 0, "--"), ("tri", 0, "-."), ("rect", 200, ":")]:
    ax[1].semilogx(tt, [S_mc(T, h0i, taui, P=pulse_cum(T, h0_ord, 100, t0, sh))[0] for T in tt], ls,
                   label=f"{sh}, t0={t0}, A=h0, w=100")
ax[1].set_xlabel("years"); ax[1].set_ylabel("P(survival)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
ax[1].set_title("Survival curves with pulse (rho=0)")
plt.tight_layout(); plt.savefig("fig_pulse_sensitivity.png", dpi=130)

fig, ax = plt.subplots(figsize=(9, 4.3))
names = [n for n in specs if n != "none"]
x = np.arange(len(names))
for j, key in enumerate(["c3.33_delta1.0", "c9.05_delta1.0", "c3.33_delta0.5"]):
    ax.bar(x+(j-1)*0.27, [1-dmg[key][n]["S_10000"]/dmg[key]["none"]["S_10000"] for n in names], 0.27, label=key)
ax.axhline(1-math.exp(-h0_ord*100), color="k", lw=1, ls="--", label="pure hazard: 1-exp(-A*w)")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("relative loss 1 - S_pulse/S_none at 10,000 y"); ax.set_title("Damage (self-exciting) model: same-area pulses")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("fig_damage_model.png", dpi=130)

print(json.dumps(res, indent=1, ensure_ascii=False))
