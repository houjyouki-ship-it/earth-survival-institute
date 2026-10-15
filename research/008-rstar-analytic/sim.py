"""008: 符号反転点 r* の解析（研究004/005/007で r* が λ0・δ によらず (50,75)年に入る理由）
再現: python3 sim.py > stdout.txt  (seed=20261014) -> results.json, fig_*.png

モデル（研究004の加法回復モデルと同一）:
  h(t) = B e^{-t/τ} [1 + δ Σ_j e^{-(t-s_j)/r}] + h_nat,  ショックは強度 λ(s)=λ0 e^{-s/τ_s} のポアソン過程（004は τ_s=τ）。
  較正: E[S(W)] = 5/6（W=較正窓, 004は W=100）。評価: S(T), T=1e4。
  加法モデルはショックについて線形なので、ポアソン過程のラプラス汎関数から MC なしで厳密に
    -log S(T) = B I0(T) + ∫_0^T λ(s) [1 - exp(-B δ C(s,T))] ds + h_nat T,
    I0(T)=τ(1-e^{-T/τ}),  C(s,T)=e^{-s/τ}(1-e^{-κ(T-s)})/κ,  κ=1/τ+1/r
  が成り立つ（【計算上の恒等式】。以下 H17e で 004 の MC と照合）。
  1次近似（BδC≪1）では -log S(T) ≈ B[I0(T)+δJ(T)], J(T)=∫λ(s)C(s,T)ds。較正で B を消すと
    非致死ショックが S(T) を下げる ⇔ J(T)/J(W) > I0(T)/I0(W)
  となり、λ0 と δ が条件から消える（r* が λ0・δ によらない理由の候補）。

事前に書いた仮説（本回の実行前に本ファイルへ記載。第三者のタイムスタンプはない）:
 H17a τ=200, W=100, T=1e4 で 1次近似の r*_lin は (50,75) に入り、λ0∈{0.1,0.3,1,3}/世紀・δ∈{0.5,1,2,3.93,5} の
      厳密 r* はすべて r*_lin±10年に入る。
 H17b τ=200 で W∈{50,100,200} と振ると r*_lin/W は [0.4,0.8] に入る（r* は較正窓にほぼ比例）。
 H17c W=100 で τ∈{100,200,400} と振っても r*_lin の変化は2倍未満（τ依存は弱い）。
 H17d ショック率が一定（τ_s=∞）なら r*_lin=0（どの r>0 でも非致死ショックは S(1e4) を下げる）。
      τ_s を τ より短くすると r*_lin は大きくなる。
 H17e 解析式（ラプラス汎関数）は研究004の MC 値 S(1e4)（加法16セル）を全セル 3SE 以内で再現する。
"""
from __future__ import annotations
import json, math, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261014
H_NAT = 1 / 87_000
P_W = 1 / 6
T_EVAL = 10_000.0
t0 = time.time()
res = {"seed": SEED, "h_nat": H_NAT, "target_S_W": 1 - P_W, "T_eval": T_EVAL}

GLx, GLw = np.polynomial.legendre.leggauss(16)
def nodes(T, tau):
    """[0,T] の合成ガウス・ルジャンドル節点（前半を細かく）。"""
    e1 = np.linspace(0, min(T, 20 * tau), 801)
    e = e1 if T <= 20 * tau else np.concatenate([e1, np.linspace(20 * tau, T, 201)[1:]])
    a, b = e[:-1, None], e[1:, None]
    x = (a + b) / 2 + (b - a) / 2 * GLx[None, :]
    w = (b - a) / 2 * GLw[None, :]
    return x.ravel(), w.ravel()

def I0(T, tau): return tau * (-math.expm1(-T / tau))
def C(s, T, tau, r):
    k = 1 / tau + (0 if math.isinf(r) else 1 / r)
    return np.exp(-s / tau) * (-np.expm1(-k * (T - s))) / k
def lam(s, lam0c, tau_s): return lam0c / 100 * (np.ones_like(s) if math.isinf(tau_s) else np.exp(-s / tau_s))

def neglogS(B, T, tau, r, lam0c, delta, tau_s):
    s, w = nodes(T, tau)
    phi = np.sum(w * lam(s, lam0c, tau_s) * (-np.expm1(-B * delta * C(s, T, tau, r))))
    return B * I0(T, tau) + phi + H_NAT * T

def calibrate(W, tau, r, lam0c, delta, tau_s):
    L = -math.log(1 - P_W)
    lo, hi = 0.0, 1.0
    while neglogS(hi, W, tau, r, lam0c, delta, tau_s) < L: hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if neglogS(mid, W, tau, r, lam0c, delta, tau_s) < L: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def S_exact(W, tau, r, lam0c, delta, tau_s, T=T_EVAL):
    B = calibrate(W, tau, r, lam0c, delta, tau_s)
    return math.exp(-neglogS(B, T, tau, r, lam0c, delta, tau_s)), B

def S_noshock(W, tau, T=T_EVAL):
    B = (-math.log(1 - P_W) - H_NAT * W) / I0(W, tau)
    return math.exp(-B * I0(T, tau) - H_NAT * T)

def J(T, tau, r, tau_s):
    s, w = nodes(T, tau)
    return float(np.sum(w * lam(s, 1.0, tau_s) * C(s, T, tau, r)))

def lin_gap(r, W, tau, tau_s, T=T_EVAL):
    """>0 ならショックが S(T) を下げる（1次）。"""
    return J(T, tau, r, tau_s) / J(W, tau, r, tau_s) - I0(T, tau) / I0(W, tau)

def root(f, lo, hi, it=60):
    flo = f(lo)
    if flo * f(hi) > 0: return None
    for _ in range(it):
        mid = math.sqrt(lo * hi)
        if f(mid) * flo > 0: lo = mid
        else: hi = mid
    return math.sqrt(lo * hi)

R_LO, R_HI = 1e-3, 1e5
def rstar_lin(W, tau, tau_s, T=T_EVAL):
    g = lambda r: lin_gap(r, W, tau, tau_s, T)
    if g(R_LO) >= -1e-12: return 0.0   # 最小 r でも既に「下げる」または中立
    return root(g, R_LO, R_HI)

def rstar_exact(W, tau, lam0c, delta, tau_s, T=T_EVAL):
    S0 = S_noshock(W, tau, T)
    g = lambda r: S0 - S_exact(W, tau, r, lam0c, delta, tau_s, T)[0]
    return root(g, 1.0, 1e4, it=40)

# ---- 0. 閉形式でのJの検算（τ_s=τ）----
def J_closed(T, tau, r):
    k = 1 / tau + 1 / r; a = 1 / tau - 1 / r
    first = tau / 2 * (-math.expm1(-2 * T / tau))
    second = (T * math.exp(-k * T)) if abs(a) < 1e-15 else (math.exp(-k * T) - math.exp(-2 * T / tau)) / a
    return 0.01 / k * (first - second)
res["J_check"] = {f"r{r}_T{T}": [J(T, 200.0, r, 200.0), J_closed(T, 200.0, r)] for r in [10.0, 62.0, 199.0, 1000.0] for T in [100.0, 1e4]}

# ---- H17e: 004 の MC との照合 ----
mc004 = json.load(open("../004-common-shock-copula/results.json"))["longterm_fixed_S100"]["results"]
val = {}
for k, v in mc004.items():
    if not k.startswith("add"): continue
    _, rr, ll, dd = k.split("_")
    r = float(rr[1:]); l = float(ll[3:]); d = float(dd[1:])
    Sa, B = S_exact(100.0, 200.0, r, l, d, 200.0)
    val[k] = {"S_analytic": Sa, "S_mc004": v["S_10000"], "SE_mc004": v["SE_10000"], "z": (Sa - v["S_10000"]) / v["SE_10000"],
              "B_analytic": B, "B_mc004": v["B"]}
res["H17e_validation_vs_004"] = val

# 追加 MC（一定ショック率 τ_s=∞ の解析式検証, N=50,000）
def mc_const(lam0c, delta, r, W=100.0, tau=200.0, n=50_000):
    rng = np.random.default_rng([SEED, 8, int(lam0c * 100), int(delta * 100), int(r)])
    out = {}
    lamy = lam0c / 100
    k = rng.poisson(lamy * T_EVAL, size=n); m = int(k.max())
    s = np.where(np.arange(m)[None, :] < k[:, None], rng.uniform(0, T_EVAL, size=(n, m)), np.nan)
    def I(T):
        sj = np.where(s < T, s, np.nan)
        kk = 1 / tau + 1 / r
        c = np.exp(-sj / tau) * (-np.expm1(-kk * (T - sj))) / kk
        return I0(T, tau) + delta * np.nansum(c, axis=1)
    IW = I(W); L = -math.log(1 - P_W)
    lo, hi = 0.0, 1.0
    while -math.log(np.mean(np.exp(-hi * IW - H_NAT * W))) < L: hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if -math.log(np.mean(np.exp(-mid * IW - H_NAT * W))) < L: lo = mid
        else: hi = mid
    x = np.exp(-(lo + hi) / 2 * I(T_EVAL) - H_NAT * T_EVAL)
    Sa, _ = S_exact(W, tau, r, lam0c, delta, math.inf)
    return {"S_mc": float(x.mean()), "SE": float(x.std(ddof=1) / math.sqrt(n)), "S_analytic": Sa,
            "z": float((Sa - x.mean()) / (x.std(ddof=1) / math.sqrt(n)))}
res["H17e_const_rate_mc"] = {f"lam{l}_d{d}_r{r}": mc_const(l, d, r) for l in [1.0, 3.0] for d in [2.0] for r in [10.0, 50.0]}

# ---- H17a: 基準 τ=200, W=100 ----
r_lin = rstar_lin(100.0, 200.0, 200.0)
ex = {}
for l in [0.1, 0.3, 1.0, 3.0]:
    for d in [0.5, 1.0, 2.0, 3.93, 5.0]:
        ex[f"lam{l}_d{d}"] = rstar_exact(100.0, 200.0, l, d, 200.0)
res["H17a"] = {"rstar_lin": r_lin, "rstar_exact": ex,
               "max_abs_dev": max(abs(v - r_lin) for v in ex.values())}
# 極限：r→0 と r→∞ の比
res["limits_tau200_W100"] = {"I0_ratio": I0(T_EVAL, 200) / I0(100, 200),
    "J_ratio_r_to_0": (-math.expm1(-2 * T_EVAL / 200)) / (-math.expm1(-2 * 100 / 200)),
    "J_ratio_r_inf": J(T_EVAL, 200, math.inf, 200) / J(100, 200, math.inf, 200)}

# ---- H17b/c: W・τ スキャン（τ_s=τ）----
scan = {}
for tau in [100.0, 200.0, 400.0]:
    for W in [25.0, 50.0, 100.0, 200.0, 400.0]:
        rs = rstar_lin(W, tau, tau)
        scan[f"tau{tau}_W{W}"] = {"rstar_lin": rs, "rstar_over_W": rs / W if rs else None, "rstar_over_tau": rs / tau if rs else None}
res["scan_W_tau"] = scan
# 無次元曲線 r*/τ vs W/τ（T→∞ 近似として T=50τ）
wgrid = np.geomspace(0.02, 5, 40)
res["dimensionless_curve"] = {"W_over_tau": wgrid.tolist(),
    "rstar_over_tau": [rstar_lin(w * 200.0, 200.0, 200.0, T=50 * 200.0) / 200.0 for w in wgrid]}
# 線形回帰的な近似：小 W/τ での r*/W
res["dimensionless_small_w"] = {f"w{w}": rstar_lin(w * 200, 200.0, 200.0, T=1e4) / (w * 200) for w in [0.01, 0.02, 0.05]}

# ---- H17d: ショック率の時間尺度 τ_s ----
ts_scan = {}
for ts in [25.0, 50.0, 100.0, 200.0, 400.0, 1000.0, math.inf]:
    ts_scan[str(ts)] = rstar_lin(100.0, 200.0, ts)
res["H17d_tau_s_scan"] = ts_scan
res["H17d_const_rate_gap_small_r"] = {f"r{r}": lin_gap(r, 100.0, 200.0, math.inf) for r in [0.01, 1.0, 10.0, 50.0, 200.0]}
# 厳密版で一定率・小さい r の S 差
res["H17d_const_rate_exact_drop"] = {f"lam{l}_d{d}_r{r}": S_noshock(100.0, 200.0) - S_exact(100.0, 200.0, r, l, d, math.inf)[0]
                                     for l in [1.0, 3.0] for d in [2.0] for r in [1.0, 10.0, 50.0]}
# 004の減衰率（τ_s=τ）と一定率の比較
res["drop_vs_r_compare"] = {}
RG = [1, 2, 5, 10, 20, 50, 62, 75, 100, 200, 500, 2000]
S0 = S_noshock(100.0, 200.0)
for ts in [100.0, 200.0, math.inf]:
    res["drop_vs_r_compare"][str(ts)] = {str(r): S0 - S_exact(100.0, 200.0, float(r), 1.0, 2.0, ts)[0] for r in RG}
res["S_noshock"] = S0


# ---- 小窓近似の閉形式（W≪τ, r≪τ）: (1-e^{-x})/x = τ/(τ+τ_s), x=W/r* ----
def x_root(c):
    if c <= 0: return math.inf
    lo, hi = 1e-9, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        if (-math.expm1(-mid)) / mid > c: lo = mid
        else: hi = mid
    return (lo + hi) / 2
res["small_window_closed_form"] = {
    "derivation": "J(∞)≈r·ττ_s/(τ+τ_s), J(W)≈r[W-r(1-e^{-W/r})], I0(∞)/I0(W)≈τ/W より (1-e^{-x})/x=τ/(τ+τ_s)",
    "tau_s_eq_tau_rstar_over_W": 1 / x_root(0.5),
    "pred_vs_tau_s_W100_tau200": {str(ts): (100.0 / x_root(200.0 / (200.0 + ts)) if not math.isinf(ts) else 0.0)
                                  for ts in [25.0, 50.0, 100.0, 200.0, 400.0, 1000.0, math.inf]},
}

# ---- 判定 ----
bW = [scan[f"tau200.0_W{W}"]["rstar_over_W"] for W in [50.0, 100.0, 200.0]]
cT = [scan[f"tau{t}_W100.0"]["rstar_lin"] for t in [100.0, 200.0, 400.0]]
ts_vals = [ts_scan[str(t)] for t in [25.0, 50.0, 100.0, 200.0]]
res["verdicts"] = {
    "H17a": bool(r_lin is not None and 50 < r_lin < 75 and res["H17a"]["max_abs_dev"] <= 10),
    "H17b": bool(all(x is not None and 0.4 <= x <= 0.8 for x in bW)), "H17b_values": bW,
    "H17c": bool(max(cT) / min(cT) < 2), "H17c_values": cT,
    "H17d_const_zero": bool(ts_scan["inf"] == 0.0),
    "H17d_monotone_shorter_ts_larger": bool(all(ts_vals[i] > ts_vals[i + 1] for i in range(len(ts_vals) - 1))),
    "H17e": bool(all(abs(v["z"]) <= 3 for v in val.values()) and all(abs(v["z"]) <= 3 for v in res["H17e_const_rate_mc"].values())),
}
res["runtime_seconds"] = time.time() - t0
json.dump(res, open("results.json", "w"), indent=1, ensure_ascii=False, default=float)

# ---- 図1: 時間プロファイル（累積割合）----
tau = 200.0; tt = np.linspace(0, 1000, 1001)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
base = (1 - np.exp(-tt / tau)) / 1.0
ax[0].plot(tt, base, "k-", lw=2, label="baseline hazard e^{-t/tau}")
for r in [10.0, r_lin, 200.0]:
    Jt = np.array([J(t, tau, r, tau) if t > 0 else 0.0 for t in tt[::10]])
    ax[0].plot(tt[::10], Jt / J(T_EVAL, tau, r, tau), "--", label=f"shock-induced, r={r:.0f}")
for r in [10.0]:
    Jt = np.array([J(t, tau, r, math.inf) if t > 0 else 0.0 for t in tt[::10]])
    ax[0].plot(tt[::10], Jt / J(T_EVAL, tau, r, math.inf), ":", label=f"shock-induced, const rate, r={r:.0f}")
ax[0].axvline(100, color="gray", ls=":"); ax[0].set_xlabel("t (years)")
ax[0].set_ylabel("cumulative fraction of 10,000-yr integral"); ax[0].set_title("Why r*: share of hazard inside the 100-yr window")
ax[0].legend(fontsize=7)
for ts, st in [("100.0", "-o"), ("200.0", "-s"), ("inf", "-^")]:
    d = res["drop_vs_r_compare"][ts]
    ax[1].semilogx([float(k) for k in d], list(d.values()), st, ms=3, label=f"tau_s={ts}")
ax[1].axhline(0, color="k", lw=0.8); ax[1].axvline(r_lin, color="gray", ls=":")
ax[1].set_xlabel("recovery time r (years)"); ax[1].set_ylabel("S_noshock - S(10,000)  (>0: shocks hurt)")
ax[1].set_title("lambda0=1/century, delta=2, exact (Laplace functional)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig_profile_and_drop.png", dpi=120); plt.close()

# ---- 図2: r*/τ vs W/τ と τ_s ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
dc = res["dimensionless_curve"]
ax[0].loglog(dc["W_over_tau"], dc["rstar_over_tau"], "k-")
for tau_ in [100.0, 200.0, 400.0]:
    xs = [W / tau_ for W in [25.0, 50.0, 100.0, 200.0, 400.0]]
    ys = [scan[f"tau{tau_}_W{W}"]["rstar_lin"] / tau_ for W in [25.0, 50.0, 100.0, 200.0, 400.0]]
    ax[0].loglog(xs, ys, "o", ms=5, label=f"tau={tau_:.0f} (T=1e4)")
ax[0].set_xlabel("W / tau (calibration window / hazard decay time)"); ax[0].set_ylabel("r* / tau")
ax[0].set_title("Linear-order r*: depends only on W/tau"); ax[0].legend(fontsize=8)
tsk = [k for k in ts_scan if k != "inf"]
ax[1].semilogx([float(k) for k in tsk], [ts_scan[k] for k in tsk], "k-o", ms=4)
ax[1].axhline(0, color="r", ls="--", label="const rate (tau_s=inf): r*=%s" % ts_scan["inf"])
ax[1].set_xlabel("shock-rate decay time tau_s (years)"); ax[1].set_ylabel("r*_lin (years)")
ax[1].set_title("tau=200, W=100"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig_rstar_scaling.png", dpi=120); plt.close()

print(json.dumps(res, indent=1, ensure_ascii=False, default=float))
print(f"runtime_sec={time.time() - t0:.1f}")
