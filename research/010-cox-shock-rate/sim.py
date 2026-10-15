"""010: 確率的に揺らぐショック率（対数正規 OU の Cox 過程）で、較正（100年で1/6）と長期生存 S(1e4) の関係、
非致死ショックの符号がどう変わるか。Clauset 2018 の3区間集計で揺らぎの大きさを制約する。
再現: python3 sim.py > stdout.txt  (seed=20261014) -> results.json, fig_*.png

モデル（研究004/008/009の加法回復モデル。τ=200, W=100, T=1e4, h_nat=1/87,000, E[S(W)]=5/6 較正）:
  h(t) = B e^{-t/τ}[1+δ Σ_j e^{-(t-s_j)/r}] + h_nat
  ショックは Cox 過程: 強度 λ(s) = (λ0/100) m(s) exp(X(s) - σ²/2)、X は定常 OU（分散 σ², 相関 e^{-|Δ|/θ}）。
  m(s)=1（一定平均, τ_s=∞）または e^{-s/200}（004–007 の τ_s=τ）。E[λ(s)] = (λ0/100) m(s) は揺らぎによらない。
  λ の経路を条件とすれば加法モデルはポアソンのラプラス汎関数で厳密:
    S(T|λ) = exp(-B I0(T) - ∫ λ(s)[1-exp(-BδC(s,T))] ds - h_nat T)
  E[S] は λ 経路の MC 平均（N=20,000 経路、共通乱数）。B は E[S(W)]=5/6 になるよう経路平均で較正（MC較正）。
  X は区分定数（0–100年は dt=0.5、100–4000年は dt=1 を5年ビンに集約）。s>4000 の寄与は exp(-20) 程度で無視。
外部データ（Clauset 2018 Sci. Adv. eaao3580 の公表集計値。研究009で本文を確認）:
  大規模国家間戦争 1823–1913: 9件, 1914–1939: 10件, 1940–2003: 5件。
  Cox 過程の集計尤度 P(n1,n2,n3) = E[Π Pois(n_i; λ_w ∫_i e^{X-σ²/2})] を MC（20万経路）で評価、λ_w はプロファイル。
  戦争の揺らぎ構造（σ, θ）を一般の「ショック率」に移すのは【推測】。

事前に書いた仮説（本回の実行前に本ファイルへ記載。第三者のタイムスタンプはない）:
 H19a 3区間集計で一定ポアソン（σ=0）は逸脱度検定（自由度2）で p<0.01 で棄却され、OU Cox 過程の格子
      （σ∈{0.25..2}, θ∈{3..1000}）の最良セルは一定率より対数尤度が3以上高い。95%整合集合（最良から Δℓ≤3.0）は σ=0 を含まない。
 H19b 一定平均（τ_s=∞）, λ0=1/世紀, δ=2, r=20 で、整合集合内の全 (σ,θ) で drop>0（一定率 +0.0113 と同じ符号）。
 H19c 一定平均では、θ を固定して σ を上げると、全 r で |drop| が単調に減る（Jensen による緩和）。
 H19d 平均が τ_s=200 で減衰する場合、整合集合内の全 (σ,θ) で r=20 の drop<0（009 の −0.0148 と同じ符号）。
 H19e 揺らぎの現在状態を Clauset 集計（1940–2003 の少なさ）で条件づけると、平均回帰でショック率が今後上がるため、
      一定平均・r=20 の drop は無条件（定常）より大きくなり、その差は整合集合の全セルで ≥ +0.005。
"""
from __future__ import annotations
import json, math, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261014
H_NAT = 1 / 87_000; P_W = 1 / 6; T_EVAL = 10_000.0; TAU = 200.0; W = 100.0
S_MAX = 4000.0; N_FWD = 20_000; N_LIK = 200_000
t0 = time.time()
res = {"seed": SEED, "tau": TAU, "W": W, "T_eval": T_EVAL, "h_nat": H_NAT, "N_fwd": N_FWD, "N_lik": N_LIK, "S_max": S_MAX}

def I0(T): return TAU * (-math.expm1(-T / TAU))
def C(s, T, r):
    k = 1 / TAU + (0 if math.isinf(r) else 1 / r)
    return np.exp(-s / TAU) * (-np.expm1(-k * (T - s))) / k
B0 = (-math.log(1 - P_W) - H_NAT * W) / I0(W)
S0 = math.exp(-B0 * I0(T_EVAL) - H_NAT * T_EVAL)
res["S_noshock"] = S0

# ---- 時間ビン ----
DT1 = 0.5; NB1 = int(W / DT1)                      # 較正窓: 0.5年ビン ×200
DT2 = 1.0; BIN2 = 5; NB2 = int((S_MAX - W) / (DT2 * BIN2))  # 100–4000: 5年ビン ×780
edges = np.concatenate([np.arange(NB1 + 1) * DT1, W + np.arange(1, NB2 + 1) * BIN2])
SUB = 16
def bin_avg(fun, lo, hi):
    """各ビンでの fun(s) の平均（ビン内16点の中点則）。"""
    u = (np.arange(SUB) + 0.5) / SUB
    s = lo[:, None] + (hi - lo)[:, None] * u[None, :]
    return fun(s).mean(axis=1)
blo, bhi = edges[:-1], edges[1:]
bw = bhi - blo

def weight_vec(B, T, r, delta, lam0c, mkind):
    """ビンごとの ∫_bin λ̄ m(s) f(s) ds / ∫_bin 1 ds 相当: 経路の『ビン平均乗数』に掛ける重み。"""
    def f(s):
        m = 1.0 if mkind == "const" else np.exp(-s / TAU)
        val = np.where(s < T, lam0c / 100 * m * (-np.expm1(-B * delta * C(np.minimum(s, T), T, r))), 0.0)
        return val
    return bin_avg(f, blo, bhi) * bw

# ---- OU 経路: 各ビンの平均乗数 exp(X-σ²/2) ----
def ou_paths(sigma, theta, n, rng, x0=None):
    """返り値: (n, NB1+NB2) の各ビンにおける exp(X-σ²/2) のビン平均。"""
    out = np.empty((n, NB1 + NB2))
    x = rng.standard_normal(n) * sigma if x0 is None else x0.copy()
    a1 = math.exp(-DT1 / theta); c1 = sigma * math.sqrt(1 - a1 * a1)
    for k in range(NB1):
        out[:, k] = np.exp(x - sigma ** 2 / 2)
        x = a1 * x + c1 * rng.standard_normal(n)
    a2 = math.exp(-DT2 / theta); c2 = sigma * math.sqrt(1 - a2 * a2)
    for k in range(NB2):
        acc = np.zeros(n)
        for _ in range(BIN2):
            acc += np.exp(x - sigma ** 2 / 2)
            x = a2 * x + c2 * rng.standard_normal(n)
        out[:, NB1 + k] = acc / BIN2
    return out

def ES(B, T, r, delta, lam0c, mkind, M):
    w = weight_vec(B, T, r, delta, lam0c, mkind)
    x = np.exp(-B * I0(T) - H_NAT * T - M @ w)
    return x.mean(), x.std(ddof=1) / math.sqrt(len(x))

def calibrate(r, delta, lam0c, mkind, M):
    L = 1 - P_W
    # W 内のビンだけ使う（weight_vec は s<T でゼロ化するので M 全体でも同値だが高速化）
    def g(B):
        w = weight_vec(B, W, r, delta, lam0c, mkind)[:NB1]
        return np.exp(-B * I0(W) - H_NAT * W - M[:, :NB1] @ w).mean() - L
    lo, hi = 0.0, 1.2 * B0
    flo, fhi = g(lo), g(hi)
    while fhi > 0: hi *= 2; fhi = g(hi)
    assert flo > 0 > fhi, (flo, fhi)
    side = 0
    for _ in range(100):  # Illinois
        mid = (lo * fhi - hi * flo) / (fhi - flo); fm = g(mid)
        if abs(fm) < 1e-14 or (hi - lo) < 1e-15 * B0: break
        if fm * fhi < 0:
            lo, flo = hi, fhi; hi, fhi = mid, fm
            if side == -1: flo /= 2
            side = -1
        else:
            hi, fhi = mid, fm
            if side == 1: flo /= 2
            side = 1
    return mid

def S_shock(r, delta, lam0c, mkind, M):
    B = calibrate(r, delta, lam0c, mkind, M)
    m, se = ES(B, T_EVAL, r, delta, lam0c, mkind, M)
    return m, se, B

R_LIST = [1.0, 5.0, 20.0, 50.0, 200.0, math.inf]
PAIRS = [(1.0, 2.0), (3.0, 2.0)]
def rkey(r): return "inf" if math.isinf(r) else f"{r:g}"

# ---- 0. 検証: σ=0 を研究009の厳密値と照合 ----
r009 = json.load(open("../009-tau-s-reassessment/results.json"))["grid_drop"]
M0 = np.ones((4, NB1 + NB2))
chk = {}
for mk, gk in [("const", "0"), ("decay", str(1 / 200))]:
    for r in R_LIST:
        m, _, _ = S_shock(r, 2.0, 1.0, mk, M0)
        k9 = f"ts{'inf' if mk=='const' else '200.0'}_lam1.0_d2.0_r{'inf' if math.isinf(r) else f'{r:.1f}'}"
        chk[f"{mk}_r{rkey(r)}"] = {"drop_010": S0 - m, "drop_009": r009.get(k9), "abs_diff": abs(S0 - m - r009[k9]) if k9 in r009 else None}
res["check_sigma0"] = chk
print("sigma=0 check:", json.dumps(chk)[:400])

# ---- 1. Clauset 3区間の集計尤度 ----
counts = np.array([9, 10, 5]); spans = [(1823, 1914), (1914, 1940), (1940, 2004)]
YEARS = 2004 - 1823
def lgam(n): return math.lgamma(n + 1)
LAMW = np.exp(np.linspace(math.log(0.03), math.log(0.6), 241))
def lik_cell(sigma, theta, rng, dt=1.0, keep=False):
    n = N_LIK; nsteps = int(round(YEARS / dt))
    x = rng.standard_normal(n) * sigma
    a = math.exp(-dt / theta) if sigma > 0 else 1.0; c = sigma * math.sqrt(1 - a * a)
    A = np.zeros((n, 3)); t = 1823.0
    for k in range(nsteps):
        i = 0 if t < 1914 else (1 if t < 1940 else 2)
        A[:, i] += np.exp(x - sigma ** 2 / 2) * dt
        x = a * x + c * rng.standard_normal(n); t += dt
    # 最後の x は 2004 年初の状態
    # log L(λw) = log mean_paths Π Pois(n_i; λw A_i)
    const = -sum(lgam(k) for k in counts)
    logA = np.log(A)
    best = (-np.inf, None, None)
    for lw in LAMW:
        ll = (counts[None, :] * (math.log(lw) + logA) - lw * A).sum(axis=1) + const
        mx = ll.max(); v = mx + math.log(np.mean(np.exp(ll - mx)))
        if v > best[0]: best = (v, lw, ll)
    out = {"loglik": best[0], "lam_w_hat": best[1]}
    if keep:
        ll = best[2]; wts = np.exp(ll - ll.max()); wts /= wts.sum()
        out["x2004"] = x; out["w"] = wts; out["ESS"] = float(1 / np.sum(wts ** 2))
    return out

rngL = np.random.default_rng([SEED, 10, 1])
lam_c = counts.sum() / YEARS
ll_const = float((counts * np.log(lam_c * np.array([b - a for a, b in spans])) - lam_c * np.array([b - a for a, b in spans])).sum()
                 - sum(lgam(k) for k in counts))
mu_i = lam_c * np.array([b - a for a, b in spans])
dev_const = float(2 * np.sum(counts * np.log(counts / mu_i) - (counts - mu_i)))
p_const = math.exp(-dev_const / 2)  # χ²(2) の生存関数
res["clauset_const"] = {"lambda_hat_per_yr": lam_c, "loglik": ll_const, "deviance": dev_const, "p_chi2_df2": p_const}
print(f"const Poisson: dev={dev_const:.3f} p={p_const:.4g}")

SIGS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]; THS = [3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
lik = {}
for sg in SIGS:
    for th in THS:
        rng = np.random.default_rng([SEED, 10, 2, int(sg * 100), int(th)])
        o = lik_cell(sg, th, rng)
        lik[f"s{sg}_t{th:g}"] = o
        print(f"lik s={sg} th={th}: ll={o['loglik']:.3f} lw={o['lam_w_hat']:.4f}")
ll_best = max(v["loglik"] for v in lik.values())
consistent = [k for k, v in lik.items() if ll_best - v["loglik"] <= 3.0]
res["clauset_cox_grid"] = lik
res["clauset_best_loglik"] = ll_best
res["clauset_const_minus_best"] = ll_const - ll_best
res["consistent_set"] = consistent
# MC 誤差と dt 依存の確認（2セル）
dtchk = {}
for sg, th in [(1.0, 10.0), (1.0, 3.0)]:
    a = lik_cell(sg, th, np.random.default_rng([SEED, 10, 3]), dt=1.0)["loglik"]
    b = lik_cell(sg, th, np.random.default_rng([SEED, 10, 4]), dt=1.0)["loglik"]
    c = lik_cell(sg, th, np.random.default_rng([SEED, 10, 5]), dt=0.25)["loglik"]
    dtchk[f"s{sg}_t{th:g}"] = {"dt1_seedA": a, "dt1_seedB": b, "dt0.25": c}
res["clauset_mc_dt_check"] = dtchk
print("dt check", dtchk)

# ---- 2. 前向き: 定常 OU と Clauset 条件付き ----
FS = [0.25, 0.5, 1.0, 1.5, 2.0]; FT = [10.0, 30.0, 100.0, 300.0]
fwd = {}
def run_set(M, tag):
    d = {}
    for (l0, de) in PAIRS:
        for mk in ["const", "decay"]:
            for r in R_LIST:
                m, se, B = S_shock(r, de, l0, mk, M)
                d[f"lam{l0:g}_d{de:g}_{mk}_r{rkey(r)}"] = {"S": m, "SE": se, "drop": S0 - m, "B": B}
    return d
for sg in FS:
    for th in FT:
        key = f"s{sg}_t{th:g}"
        rng = np.random.default_rng([SEED, 10, 6, int(sg * 100), int(th)])
        M = ou_paths(sg, th, N_FWD, rng)
        fwd[key] = {"stationary": run_set(M, "stat")}
        # 条件付き: 2004年の状態を尤度重みでリサンプルし、2026年まで22年進めて s=0 とする
        o = lik_cell(sg, th, np.random.default_rng([SEED, 10, 7, int(sg * 100), int(th)]), keep=True)
        rngc = np.random.default_rng([SEED, 10, 8, int(sg * 100), int(th)])
        idx = rngc.choice(len(o["w"]), size=N_FWD, p=o["w"])
        x = o["x2004"][idx]; a = math.exp(-22 / th)
        x = a * x + sg * math.sqrt(1 - a * a) * rngc.standard_normal(N_FWD)
        Mc = ou_paths(sg, th, N_FWD, rngc, x0=x)
        fwd[key]["conditioned"] = run_set(Mc, "cond")
        fwd[key]["cond_info"] = {"ESS": o["ESS"], "mean_mult_s0": float(np.mean(np.exp(x - sg ** 2 / 2))),
                                 "mean_mult_first100": float(Mc[:, :NB1].mean()), "lam_w_hat": o["lam_w_hat"]}
        del M, Mc
        k20 = "lam1_d2_const_r20"
        print(f"fwd {key}: stat drop r20={fwd[key]['stationary'][k20]['drop']:+.4f}±{fwd[key]['stationary'][k20]['SE']:.4f} "
              f"cond={fwd[key]['conditioned'][k20]['drop']:+.4f} mult0={fwd[key]['cond_info']['mean_mult_s0']:.3f} "
              f"decay r20={fwd[key]['stationary']['lam1_d2_decay_r20']['drop']:+.4f}  t={time.time()-t0:.0f}s", flush=True)
res["forward"] = fwd

# σ=0 基準（決定論）
base = run_set(np.ones((2, NB1 + NB2)), "sigma0")
res["forward_sigma0"] = base

# ---- 3. 判定 ----
V = {}
fwd_cons = [k for k in consistent if k in fwd]
cons_not_in_fwd = [k for k in consistent if k not in fwd]
V["H19a"] = {"p_const": p_const, "best_minus_const": ll_best - ll_const,
             "sigma0_in_consistent": False,  # σ=0 は Δℓ = ll_best - ll_const で判定
             "const_within_3": (ll_best - ll_const) <= 3.0,
             "supported": bool(p_const < 0.01 and (ll_best - ll_const) >= 3.0)}
V["consistent_cells_in_forward_grid"] = fwd_cons
V["consistent_cells_not_in_forward_grid"] = cons_not_in_fwd
V["H19b"] = {k: fwd[k]["stationary"]["lam1_d2_const_r20"]["drop"] for k in fwd_cons}
V["H19b"]["supported"] = bool(all(v > 0 for v in V["H19b"].values()) and len(fwd_cons) > 0)
viol = []
for th in FT:
    for (l0, de) in PAIRS:
        for r in R_LIST:
            k = f"lam{l0:g}_d{de:g}_const_r{rkey(r)}"
            seq = [abs(base[k]["drop"])] + [abs(fwd[f"s{sg}_t{th:g}"]["stationary"][k]["drop"]) for sg in FS]
            if any(seq[i + 1] > seq[i] for i in range(len(seq) - 1)): viol.append(f"t{th:g}_{k}")
V["H19c"] = {"violations": viol, "n_checked": len(FT) * len(PAIRS) * len(R_LIST), "supported": len(viol) == 0}
V["H19d"] = {k: fwd[k]["stationary"]["lam1_d2_decay_r20"]["drop"] for k in fwd_cons}
V["H19d"]["supported"] = bool(all(v < 0 for v in V["H19d"].values()) and len(fwd_cons) > 0)
V["H19e"] = {k: fwd[k]["conditioned"]["lam1_d2_const_r20"]["drop"] - fwd[k]["stationary"]["lam1_d2_const_r20"]["drop"] for k in fwd_cons}
V["H19e"]["supported"] = bool(all(v >= 0.005 for v in V["H19e"].values()) and len(fwd_cons) > 0)
# r* 探索: 一定平均で S を上げる (drop<0) セル
raise_cells = []
for key in fwd:
    for cond in ["stationary", "conditioned"]:
        for k, v in fwd[key][cond].items():
            if "_const_" in k and v["drop"] < -2 * v["SE"]: raise_cells.append(f"{key}_{cond}_{k}:{v['drop']:+.4f}")
V["const_mean_cells_where_shock_raises_S"] = raise_cells
res["verdicts"] = V
for h in ["H19a", "H19b", "H19c", "H19d", "H19e"]: print(h, V[h]["supported"] if "supported" in V[h] else V[h])

# ---- 図 ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
Z = np.array([[lik[f"s{sg}_t{th:g}"]["loglik"] - ll_const for th in THS] for sg in SIGS])
im = ax[0].imshow(Z, origin="lower", cmap="viridis", aspect="auto")
ax[0].set_xticks(range(len(THS))); ax[0].set_xticklabels([f"{t:g}" for t in THS])
ax[0].set_yticks(range(len(SIGS))); ax[0].set_yticklabels([f"{s:g}" for s in SIGS])
for i in range(len(SIGS)):
    for j in range(len(THS)):
        k = f"s{SIGS[i]}_t{THS[j]:g}"
        ax[0].text(j, i, f"{Z[i,j]:.1f}" + ("*" if k in consistent else ""), ha="center", va="center", color="w", fontsize=8)
ax[0].set_xlabel("theta (OU correlation time, yr)"); ax[0].set_ylabel("sigma (log-rate SD)")
ax[0].set_title("Clauset 3-bin war counts: loglik(Cox) - loglik(const)\n* = within 3.0 of best")
plt.colorbar(im, ax=ax[0])
rr = [1, 5, 20, 50, 200, 2000]
for sg, ls in zip([0.0] + FS, ["k-", "C0-", "C1-", "C2-", "C3-", "C4-"]):
    if sg == 0: y = [base[f"lam1_d2_const_r{rkey(r)}"]["drop"] for r in R_LIST]
    else: y = [fwd[f"s{sg}_t30"]["stationary"][f"lam1_d2_const_r{rkey(r)}"]["drop"] for r in R_LIST]
    ax[1].plot(rr, y, ls, marker="o", label=f"sigma={sg:g}")
y = [fwd["s1.0_t30"]["conditioned"][f"lam1_d2_const_r{rkey(r)}"]["drop"] for r in R_LIST]
ax[1].plot(rr, y, "C2--", marker="s", label="sigma=1, conditioned on Clauset")
ax[1].axhline(0, color="gray", lw=0.8); ax[1].set_xscale("log")
ax[1].set_xticks(rr); ax[1].set_xticklabels(["1", "5", "20", "50", "200", "inf"])
ax[1].set_xlabel("recovery time r (yr)"); ax[1].set_ylabel("drop = S_noshock - S_shock at T=1e4")
ax[1].set_title("constant mean rate, lam0=1/century, delta=2, theta=30"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig_clauset_lik_and_drop.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for th, c in zip(FT, ["C0", "C1", "C2", "C3"]):
    ax[0].plot([0] + FS, [base["lam1_d2_const_r20"]["drop"]] + [fwd[f"s{sg}_t{th:g}"]["stationary"]["lam1_d2_const_r20"]["drop"] for sg in FS], c + "-o", label=f"const mean, theta={th:g}")
    ax[0].plot([0] + FS, [base["lam1_d2_decay_r20"]["drop"]] + [fwd[f"s{sg}_t{th:g}"]["stationary"]["lam1_d2_decay_r20"]["drop"] for sg in FS], c + ":^", label=f"tau_s=200, theta={th:g}")
ax[0].axhline(0, color="gray", lw=0.8); ax[0].set_xlabel("sigma"); ax[0].set_ylabel("drop at r=20")
ax[0].set_title("stationary OU: drop vs sigma (lam0=1, delta=2, r=20)"); ax[0].legend(fontsize=7)
for th, c in zip(FT, ["C0", "C1", "C2", "C3"]):
    ax[1].plot(FS, [fwd[f"s{sg}_t{th:g}"]["conditioned"]["lam1_d2_const_r20"]["drop"] - fwd[f"s{sg}_t{th:g}"]["stationary"]["lam1_d2_const_r20"]["drop"] for sg in FS], c + "-o", label=f"theta={th:g}")
ax[1].axhline(0.005, color="r", ls="--", lw=0.8, label="H19e threshold +0.005"); ax[1].axhline(0, color="gray", lw=0.8)
ax[1].set_xlabel("sigma"); ax[1].set_ylabel("drop(conditioned) - drop(stationary)")
ax[1].set_title("effect of conditioning on 'long peace' (const mean, r=20)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig_sigma_and_conditioning.png", dpi=120); plt.close()

res["runtime_seconds"] = time.time() - t0
json.dump(res, open("results.json", "w"), indent=1, ensure_ascii=False, default=float)
print("runtime", res["runtime_seconds"])
