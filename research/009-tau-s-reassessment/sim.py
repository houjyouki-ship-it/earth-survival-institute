"""009: ショック率の減衰時間 τ_s を不確実パラメータとして、研究004–007の「速く回復する非致死ショックは長期生存を上げる」を再評価する。
再現: python3 sim.py > stdout.txt  (seed=20261014; 乱数は使わない決定論的計算) -> results.json, fig_*.png

モデル（研究004/008の加法回復モデル、008のラプラス汎関数で厳密計算。MCなし）:
  h(t) = B e^{-t/τ}[1+δ Σ_j e^{-(t-s_j)/r}] + h_nat,  ショック強度 λ(s)。τ=200, W=100, E[S(W)]=5/6 で B を較正、T=1e4 で評価。
  λ(s) の形: (i) 指数減衰 λ0 e^{-g s}（g=1/τ_s ≥ 0, g=0 は一定率）, (ii) 上昇→頭打ち λ0(3-2e^{-s/50})（Marani et al. 2021 の
  「極端な疫病の年確率は今後数十年で最大3倍」を形だけ借りたシナリオ。疫病以外への一般化は【推測】）。
外部データ（実際に開いた出典の公表集計値のみ）:
  Clauset 2018 Sci. Adv. eaao3580: 大規模国家間戦争（上位四分位, ≥26,625戦死者）1823–1939 に 19 件、うち 1914–1939 に 10 件、
  1940–2003 に 5 件。→ 発生率の指数トレンド g をポアソン尤度で推定（戦争発生率を「ショック率」の代理とするのは【推測】）。
  Marani et al. 2021 PNAS: スペイン風邪級の平均再来時間 1918年当時 181年(95%CI 151–223) / 現在(2000–2019) 877年(730–1081)。

事前に書いた仮説（本回の実行前に本ファイルへ記載。第三者のタイムスタンプはない）:
 H18a Clauset の2区間集計（1823–1939: 19件/117年, 1940–2003: 5件/64年）での指数トレンド g の 95% 尤度比区間は
      g=0（一定率）と g=1/200（τ_s=τ）の両方を含む（データは 004–007 の仮定と一定率を区別できない）。
 H18b g の事前分布を (P1) τ_s 対数一様 [25, 1e5]＋一定率に質量1/2、(P2) g 一様 [0, 1/25] とし、戦争データ尤度で更新した事後平均で、
      代表セル λ0=1/世紀, δ=2, r=20 の非致死ショックは S(1e4) を下げる（事後平均 drop>0）。P1・P2 の両方で成り立つ。
 H18c 大きさの非対称: λ0∈{0.3,1,3}, δ∈{0.5,1,2,3.93,5}, r∈{1..50} で、τ_s∈{25..∞} 全体でのショックによる S 上昇の最大値は、
      一定率（τ_s=∞）での S 低下の最大値の 1/3 未満。
 H18d 上昇→頭打ちシナリオでは全セル（r>0）で drop>0 で、r=20 の drop は一定率より大きい。
 H18e 3区間（1823–1913: 9件/91年, 1914–1939: 10件/26年, 1940–2003: 5件/64年）では、指数トレンド＋ポアソンの当てはまりが
      逸脱度検定で棄却される（p<0.05; 決定論的な τ_s モデル自体が誤指定）。
"""
from __future__ import annotations
import json, math, time
import numpy as np
class stats:  # scipy 不使用（1自由度のみ）
    class chi2:
        @staticmethod
        def ppf(q, df):
            assert q == 0.95 and df == 1; return 3.841458820694124
        @staticmethod
        def sf(x, df):
            assert df == 1; return math.erfc(math.sqrt(max(x, 0) / 2))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261014
H_NAT = 1 / 87_000; P_W = 1 / 6; T_EVAL = 10_000.0; TAU = 200.0; W = 100.0
t0 = time.time()
res = {"seed": SEED, "note": "deterministic; no RNG used", "tau": TAU, "W": W, "T_eval": T_EVAL, "h_nat": H_NAT}

GLx, GLw = np.polynomial.legendre.leggauss(16)
_NODES = {}
def nodes(T):
    if T not in _NODES:
        e1 = np.linspace(0, min(T, 20 * TAU), 801)
        e = e1 if T <= 20 * TAU else np.concatenate([e1, np.linspace(20 * TAU, T, 201)[1:]])
        a, b = e[:-1, None], e[1:, None]
        _NODES[T] = (((a + b) / 2 + (b - a) / 2 * GLx[None, :]).ravel(), ((b - a) / 2 * GLw[None, :]).ravel())
    return _NODES[T]
def I0(T): return TAU * (-math.expm1(-T / TAU))
def C(s, T, r):
    k = 1 / TAU + (0 if math.isinf(r) else 1 / r)
    return np.exp(-s / TAU) * (-np.expm1(-k * (T - s))) / k
def lam_shape(s, kind):
    """λ(s)/λ0. kind: float g (=1/τ_s, 0=一定率) または 'rise'."""
    if kind == "rise": return 3 - 2 * np.exp(-s / 50)
    return np.exp(-kind * s)
def neglogS(B, T, r, lam0c, delta, kind):
    s, w = nodes(T)
    return B * I0(T) + np.sum(w * lam0c / 100 * lam_shape(s, kind) * (-np.expm1(-B * delta * C(s, T, r)))) + H_NAT * T
def calibrate(r, lam0c, delta, kind):
    L = -math.log(1 - P_W); lo, hi = 0.0, 1e-2
    while neglogS(hi, W, r, lam0c, delta, kind) < L: hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if neglogS(mid, W, r, lam0c, delta, kind) < L: lo = mid
        else: hi = mid
    return (lo + hi) / 2
def S_exact(r, lam0c, delta, kind):
    B = calibrate(r, lam0c, delta, kind)
    return math.exp(-neglogS(B, T_EVAL, r, lam0c, delta, kind))
B0 = (-math.log(1 - P_W) - H_NAT * W) / I0(W)
S0 = math.exp(-B0 * I0(T_EVAL) - H_NAT * T_EVAL)
res["S_noshock"] = S0

# ---- 照合: 008 の結果を再現（τ_s=τ, λ0=1, δ=2）----
r008 = json.load(open("../008-rstar-analytic/results.json"))["drop_vs_r_compare"]
res["check_vs_008"] = {}
for ts, key in [(1 / 200, "200.0"), (0.0, "inf"), (1 / 100, "100.0")]:
    for r in [10, 62, 200]:
        d = S0 - S_exact(float(r), 1.0, 2.0, ts)
        res["check_vs_008"][f"ts{key}_r{r}"] = {"drop_009": d, "drop_008": r008[key][str(r)], "absdiff": abs(d - r008[key][str(r)])}

# ---- 1. 戦争データからの g（Clauset 2018 の公表集計値）----
def fit(bins):
    """bins: [(start_offset, length, count)]。λ(t)=λ0 e^{-g t}。λ0 は解析的にプロファイル。"""
    n = np.array([b[2] for b in bins], float); N = n.sum()
    def expo(g):
        a = np.array([b[0] for b in bins], float); L = np.array([b[1] for b in bins], float)
        return L.copy() if abs(g) < 1e-12 else (np.exp(-g * a) - np.exp(-g * (a + L))) / g
    def ll(g):
        e = expo(g); lam0 = N / e.sum(); mu = lam0 * e
        return float(np.sum(n * np.log(mu) - mu)), mu
    gg = np.linspace(-0.03, 0.06, 9001)
    lls = np.array([ll(g)[0] for g in gg]); i = int(lls.argmax()); ghat = gg[i]; lmax = lls[i]
    inside = gg[lls >= lmax - stats.chi2.ppf(0.95, 1) / 2]
    _, mu = ll(ghat)
    dev = float(2 * np.sum(np.where(n > 0, n * np.log(n / mu), 0) - (n - mu)))
    dof = len(bins) - 2
    return {"g_hat": float(ghat), "tau_s_hat": (1 / ghat if ghat > 0 else None), "CI95_g": [float(inside.min()), float(inside.max())],
            "LR_vs_const": float(2 * (lmax - ll(0.0)[0])), "p_vs_const": float(stats.chi2.sf(2 * (lmax - ll(0.0)[0]), 1)),
            "LR_vs_g1_200": float(2 * (lmax - ll(1 / 200)[0])), "p_vs_g1_200": float(stats.chi2.sf(2 * (lmax - ll(1 / 200)[0]), 1)),
            "deviance": dev, "dof": dof, "gof_p": (float(stats.chi2.sf(dev, dof)) if dof > 0 else None),
            "_ll": lambda g: ll(g)[0]}
two = fit([(0, 117, 19), (117, 64, 5)])
three = fit([(0, 91, 9), (91, 26, 10), (117, 64, 5)])
ll2 = two.pop("_ll"); three.pop("_ll")
res["war_fit_2bin"] = two; res["war_fit_3bin"] = three
res["war_data_source"] = "Clauset 2018 Sci. Adv. 4:eaao3580, large wars (>=26,625 battle deaths): 19 in 1823-1939 (10 in 1914-1939), 5 in 1940-2003"

# ---- 2. 疫病データからの含意（Marani et al. 2021 の公表値）----
# 20年の非重複窓が 2019 で終わるので 1918 は 1900–1919 窓（中央 1909.5）、現在は 2000–2019 窓（中央 2009.5）と【推測】→ Δ=100年
def ts_from(T_then, T_now, dt=100.0): return dt / math.log(T_now / T_then)
res["epidemic_implied_tau_s"] = {"point": ts_from(181, 877), "range_from_CIs": [ts_from(151, 1081), ts_from(223, 730)],
    "assumption": "Δt=100 yr between window midpoints (inferred, not stated by authors); forward projection by authors: up to 3x increase"}

# ---- 3. τ_s × r × (λ0, δ) の格子 ----
TS_LIST = [25.0, 50.0, 100.0, 200.0, 400.0, 1000.0, math.inf]
R_LIST = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, math.inf]
L_LIST = [0.3, 1.0, 3.0]; D_LIST = [0.5, 1.0, 2.0, 3.93, 5.0]
grid = {}
for ts in TS_LIST + ["rise"]:
    kind = "rise" if ts == "rise" else (0.0 if math.isinf(ts) else 1 / ts)
    for l in L_LIST:
        for d in D_LIST:
            for r in R_LIST:
                grid[f"ts{ts}_lam{l}_d{d}_r{r}"] = S0 - S_exact(r, l, d, kind)
res["grid_drop"] = grid  # drop = S_noshock - S_shock (>0: ショックが S を下げる)
def frac_raise(ts, rmax=50.0):
    ks = [k for k in grid if k.startswith(f"ts{ts}_") and float(k.split("_r")[1]) <= rmax]
    return sum(grid[k] < 0 for k in ks) / len(ks)
res["frac_raise_r_le_50"] = {str(ts): frac_raise(ts) for ts in TS_LIST + ["rise"]}
res["frac_raise_all_r"] = {str(ts): frac_raise(ts, math.inf) for ts in TS_LIST + ["rise"]}

# ---- 4. 事後平均（代表セル）----
G = np.concatenate([[0.0], 1 / np.geomspace(1e5, 25, 120)])          # g 格子（0=一定率）
# P1: τ_s 対数一様 [25,1e5] に 1/2、一定率に 1/2
lw = np.log(G[1:]); dlog = np.gradient(lw)
p1 = np.concatenate([[0.5], 0.5 * dlog / dlog.sum()])
# P2: g 一様 [0, 1/25]（台形重み）
dg = np.gradient(G); p2 = dg / dg.sum()
lik = np.array([math.exp(ll2(g) - ll2(two["g_hat"])) for g in G])
post = {"P1": p1 * lik / np.sum(p1 * lik), "P2": p2 * lik / np.sum(p2 * lik), "P1_prior": p1, "P2_prior": p2}
res["posterior_summary"] = {}
for nm in ["P1", "P2"]:
    w = post[nm]
    res["posterior_summary"][nm] = {"P(g<1/200 | data) i.e. tau_s>tau": float(w[G < 1 / 200].sum()),
                                    "P(g=0 | data)": float(w[0]), "post_mean_g": float(np.sum(w * G)),
                                    "prior_P(tau_s>tau)": float(post[nm + "_prior"][G < 1 / 200].sum())}
cells = [(1.0, 2.0, 20.0), (1.0, 2.0, 10.0), (1.0, 2.0, 50.0), (3.0, 2.0, 20.0), (0.3, 3.93, 20.0), (1.0, 5.0, 5.0)]
pm = {}
drop_curve = {}
for (l, d, r) in cells:
    dr = np.array([S0 - S_exact(r, l, d, g) for g in G])
    drop_curve[f"lam{l}_d{d}_r{r}"] = dr.tolist()
    row = {}
    for nm in ["P1", "P2", "P1_prior", "P2_prior"]:
        row[nm] = float(np.sum(post[nm] * dr))
    # 混合較正: 単一の B で 事後平均 S(W)=5/6（「1/6 は τ_s 不確実性込みの予測」と読む場合）
    for nm in ["P1", "P2"]:
        wgt = post[nm]; L = 1 - P_W
        def SW(B): return float(np.sum(wgt * np.array([math.exp(-neglogS(B, W, r, l, d, g)) for g in G])))
        lo, hi = 0.0, 1e-2
        for _ in range(40):
            mid = (lo + hi) / 2
            if SW(mid) > L: lo = mid
            else: hi = mid
        B = (lo + hi) / 2
        ST = float(np.sum(wgt * np.array([math.exp(-neglogS(B, T_EVAL, r, l, d, g)) for g in G])))
        row[nm + "_mixture_calibrated"] = S0 - ST
    pm[f"lam{l}_d{d}_r{r}"] = row
res["posterior_mean_drop"] = pm
res["G_grid"] = G.tolist()

# ---- 判定 ----
lo2, hi2 = two["CI95_g"]
raise_max = max(-v for k, v in grid.items() if not k.startswith("tsrise") and float(k.split("_r")[1]) <= 50)
drop_const_max = max(v for k, v in grid.items() if k.startswith("tsinf_") and float(k.split("_r")[1]) <= 50)
rise_cells = {k: v for k, v in grid.items() if k.startswith("tsrise")}
res["verdicts"] = {
    "H18a": bool(lo2 <= 0 <= hi2 and lo2 <= 1 / 200 <= hi2),
    "H18b": bool(pm["lam1.0_d2.0_r20.0"]["P1"] > 0 and pm["lam1.0_d2.0_r20.0"]["P2"] > 0),
    "H18b_values": {k: pm["lam1.0_d2.0_r20.0"][k] for k in pm["lam1.0_d2.0_r20.0"]},
    "H18c": bool(raise_max < drop_const_max / 3), "H18c_raise_max": raise_max, "H18c_drop_const_max": drop_const_max,
    "H18d": bool(all(v > 0 for v in rise_cells.values()) and
                 all(grid[f"tsrise_lam{l}_d{d}_r20.0"] > grid[f"tsinf_lam{l}_d{d}_r20.0"] for l in L_LIST for d in D_LIST)),
    "H18d_min_drop_rise": min(rise_cells.values()),
    "H18e": bool(three["gof_p"] is not None and three["gof_p"] < 0.05), "H18e_p": three["gof_p"],
}
res["runtime_seconds"] = time.time() - t0
json.dump(res, open("results.json", "w"), indent=1, ensure_ascii=False, default=float)

# ---- 図1: 戦争データの尤度と τ_s 別の drop ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
gg = np.linspace(-0.01, 0.04, 501)
ax[0].plot(gg * 1000, [math.exp(ll2(g) - ll2(two["g_hat"])) for g in gg], "k-", label="Clauset 2018, 2 bins")
ax[0].axvline(0, color="r", ls="--", label="constant rate (g=0)"); ax[0].axvline(5, color="b", ls="--", label="g=1/200 (tau_s=tau)")
ax[0].axhline(math.exp(-stats.chi2.ppf(0.95, 1) / 2), color="gray", ls=":", label="95% LR cut")
ax[0].set_xlabel("decay rate g of large-war onset rate (per 1000 yr)"); ax[0].set_ylabel("relative likelihood")
ax[0].set_title("Large interstate wars 1823-2003"); ax[0].legend(fontsize=7)
for key, st in [("lam1.0_d2.0_r20.0", "-"), ("lam1.0_d2.0_r50.0", "--"), ("lam3.0_d2.0_r20.0", "-."), ("lam1.0_d2.0_r10.0", ":")]:
    ax[1].semilogx(np.where(G > 0, 1 / np.maximum(G, 1e-12), 3e5), drop_curve[key], st, label=key)
ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xlabel("tau_s (years; rightmost point = constant rate)")
ax[1].set_ylabel("S_noshock - S(10,000)  (>0: shocks hurt)"); ax[1].set_title("Sign flips with tau_s (exact, tau=200, W=100)")
ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig_war_likelihood_and_drop.png", dpi=120); plt.close()

# ---- 図2: τ_s × r の符号マップ（λ0=1, δ=2）と posterior ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
tsl = TS_LIST + ["rise"]
M = np.array([[grid[f"ts{ts}_lam1.0_d2.0_r{r}"] for r in R_LIST] for ts in tsl])
vm = np.abs(M).max()
im = ax[0].imshow(M, cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
ax[0].set_xticks(range(len(R_LIST))); ax[0].set_xticklabels([("inf" if math.isinf(r) else f"{r:g}") for r in R_LIST], fontsize=7)
ax[0].set_yticks(range(len(tsl))); ax[0].set_yticklabels([str(t) for t in tsl], fontsize=7)
ax[0].set_xlabel("recovery time r"); ax[0].set_ylabel("tau_s"); ax[0].set_title("drop (red: shocks lower S), lambda0=1, delta=2")
plt.colorbar(im, ax=ax[0])
xs = np.where(G > 0, 1 / np.maximum(G, 1e-12), 3e5)
for nm, st in [("P1_prior", "c:"), ("P1", "c-"), ("P2_prior", "m:"), ("P2", "m-")]:
    ax[1].semilogx(xs, post[nm], st, label=nm)
ax[1].set_xlabel("tau_s (rightmost = constant rate)"); ax[1].set_ylabel("weight"); ax[1].set_title("prior vs posterior (war data)")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig_sign_map_posterior.png", dpi=120); plt.close()

out = {k: v for k, v in res.items() if k not in ("grid_drop", "G_grid")}
print(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print(f"runtime_sec={time.time() - t0:.1f}")
