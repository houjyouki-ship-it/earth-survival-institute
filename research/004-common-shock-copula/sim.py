"""004: 共通ショック（非致死的大災害）によるリスク間相関と脆弱化
再現: python3 sim.py > stdout.txt  (seed=20261014) -> results.json, fig_*.png

事前に書いた仮説（本回の実行前に本ファイルへ記載。第三者のタイムスタンプはない）:
 H7a Ordの表6.1の要因別値を独立に合算すると、合計は「~1 in 6」の丸め幅 [1/6.5, 1/5.5] に入り、
     表からはリスク間相関を識別できない。
 H7b 周辺確率をOrd値に固定したまま、減衰する共通ショック（λ0 ≤ 3回/世紀、δ ≤ 2、恒久・乗法）を入れても、
     100年合計リスクの低下は 0.01 未満で、独立合算(≈0.179)と1/6の差を単独では説明できない。
 H7c 100年合計を5/6生存に固定すると、S(1万年) は δ について単調に下がる（減衰ショック、乗法恒久）。
 H8' XPTの「絶滅/大災害」中央値比（エスカレーション確率qの粗い代理）は、超予測者と専門家で一桁(10倍)以上違う。
 H10 ショック率が一定で恒久・乗法の脆弱化がある場合、長期生存 S(∞)>0 となる境界は
     平均ハザードの条件 λδ<1/τ ではなく、経路の条件 λ·ln(1+δ)<1/τ で決まる。
"""
from __future__ import annotations
import json, math, time
from statistics import NormalDist
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261014
TAU = 200.0
H_NAT = 1 / 87_000
ND = NormalDist()
t0 = time.time()
res = {"seed": SEED, "tau": TAU, "h_nat_upper": H_NAT}

# ---------- 0. 入力（Ord 2020 表6.1, 今世紀の存続破局、個人の主観推定・丸め値）----------
ORD = {  # 自然3項目は「自然合計 1/10,000」で代表させ二重計上を避ける
    "unaligned_AI": 1 / 10, "engineered_pandemics": 1 / 30, "unforeseen_anthropogenic": 1 / 30,
    "other_anthropogenic": 1 / 50, "nuclear_war": 1 / 1000, "climate_change": 1 / 1000,
    "other_environmental": 1 / 1000, "natural_pandemics": 1 / 10_000, "total_natural": 1 / 10_000,
}
ORD_TOTAL = 1 / 6
p = np.array(list(ORD.values()))
amplified = np.array([k != "total_natural" for k in ORD])  # 自然リスクはショックで増幅しない
res["ord_inputs"] = {"values": ORD, "total": ORD_TOTAL,
    "meaning": "Ord個人の主観推定（今世紀・存続破局）。表注で『各値は容易に3倍上下しうる』『桁を示す目的』。"}

# ---------- 1. Ord表の内部整合性（H7a）----------
s_sum = float(p.sum())
u_ind = float(1 - np.prod(1 - p))
band = (1 / 6.5, 1 / 5.5)
res["ord_consistency"] = {
    "sum_of_rows": s_sum, "independent_union": u_ind, "one_in": 1 / u_ind,
    "comonotone_lower_bound": float(p.max()), "rounding_band_1in6": band,
    "independent_in_band": band[0] <= u_ind <= band[1],
    "competing_risk_reading_excess_over_1_6": s_sum / ORD_TOTAL - 1,
}

# 1a. 一因子ガウスコピュラ：周辺固定で合計リスク(和集合)をρの関数として（Gauss-Hermite求積、決定的）
xg, wg = np.polynomial.hermite_e.hermegauss(120)
wg = wg / wg.sum()
Phi = np.vectorize(ND.cdf)
qk = np.array([ND.inv_cdf(v) for v in p])

def union_copula(rho):
    if rho <= 0:
        return float(1 - np.prod(1 - p)) if rho == 0 else None
    a = (qk[None, :] - math.sqrt(rho) * xg[:, None]) / math.sqrt(1 - rho)
    cond = Phi(a)
    return float(1 - np.sum(wg * np.prod(1 - cond, axis=1)))

rhos = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
uc = {str(r): union_copula(r) for r in rhos}
lo, hi = 1e-6, 0.999
for _ in range(60):
    mid = (lo + hi) / 2
    if union_copula(mid) > ORD_TOTAL: lo = mid
    else: hi = mid
rho_star = (lo + hi) / 2
res["gaussian_copula"] = {"union_vs_rho": uc, "rho_for_exact_1_6": rho_star,
    "note": "一因子・等負荷の潜在相関ρ。ρはイベント相関ではなく潜在変数の相関。"}

# 1b. 「3倍上下」不確実性：各値を独立に 3^U (U~一様[-1,1]) 倍したときの独立合算の分布（仮定）
rng = np.random.default_rng(SEED)
U = rng.uniform(-1, 1, size=(200_000, p.size))
pu = np.clip(p * 3.0 ** U, 0, 0.999)
uu = 1 - np.prod(1 - pu, axis=1)
res["factor3_uncertainty_independent_union"] = {
    "assumption": "各行を独立に log一様 [1/3,3] 倍（Ord表注の『3倍』を幅として読む仮定）",
    "median": float(np.median(uu)), "p05": float(np.quantile(uu, 0.05)), "p95": float(np.quantile(uu, 0.95)),
    "P_in_band": float(np.mean((uu >= band[0]) & (uu <= band[1]))), "P_below_1_6": float(np.mean(uu < ORD_TOTAL)),
}

# ---------- 2. 共通ショック（減衰率 λ(t)=λ0 e^{-t/τ}）----------
def draw_shocks_decay(n, lam0_per_century, T, rng):
    lam0 = lam0_per_century / 100
    Lam = lam0 * TAU * (-math.expm1(-T / TAU))
    k = rng.poisson(Lam, size=n)
    m = int(k.max()) if k.max() > 0 else 0
    s = np.full((n, max(m, 1)), np.inf)
    if m > 0:
        u = rng.uniform(size=(n, m))
        t = -TAU * np.log1p(-u * (-math.expm1(-T / TAU)))
        mask = np.arange(m)[None, :] < k[:, None]
        t = np.where(mask, t, np.inf)
        s = np.sort(t, axis=1)
    return s

def I_mult(s, delta, T):
    """∫0^T e^{-t/τ} (1+δ)^{N(t)} dt（厳密）"""
    n, m = s.shape
    bounds = np.concatenate([np.zeros((n, 1)), np.minimum(s, T), np.full((n, 1), T)], axis=1)
    j = np.arange(m + 1)
    seg = np.exp(-bounds[:, :-1] / TAU) - np.exp(-bounds[:, 1:] / TAU)
    return TAU * np.sum(seg * (1 + delta) ** j[None, :], axis=1)

def I_add(s, delta, r, T):
    """∫0^T e^{-t/τ} [1+δ Σ_j e^{-(t-s_j)/r}] dt（回復つき加法、r=inf で恒久加法）"""
    base = TAU * (-math.expm1(-T / TAU))
    sj = np.where(np.isfinite(s) & (s < T), s, np.nan)
    if math.isinf(r):
        c = TAU * (np.exp(-sj / TAU) - math.exp(-T / TAU))
    else:
        k = 1 / TAU + 1 / r
        c = np.exp(sj / r) * (np.exp(-k * sj) - math.exp(-k * T)) / k
    return base + delta * np.nansum(c, axis=1)

def solve_b(I, target_p, extra=0.0):
    """1 - E[exp(-b I - extra)] = target_p を b について解く（I は共通乱数）"""
    lo, hi = 0.0, 1.0
    while 1 - np.mean(np.exp(-hi * I - extra)) < target_p: hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if 1 - np.mean(np.exp(-mid * I - extra)) < target_p: lo = mid
        else: hi = mid
    return (lo + hi) / 2

N2 = 100_000
LAM0S = [0.1, 0.3, 1.0, 3.0]          # 回/世紀（t=0での率）
DELTAS = [0.0, 0.5, 1.0, 2.0, 5.0]
union_grid, pair_corr = {}, {}
nat_extra = -math.log(1 - ORD["total_natural"])
for lam in LAM0S:
    rng = np.random.default_rng([SEED, int(lam * 1000), 1])
    s = draw_shocks_decay(N2, lam, 100.0, rng)
    for d in DELTAS:
        I = I_mult(s, d, 100.0)
        bs = np.array([solve_b(I, pk) for pk in p[amplified]])
        B = bs.sum()
        S_union = np.mean(np.exp(-B * I)) * (1 - ORD["total_natural"])
        union_grid[f"lam{lam}_d{d}"] = float(1 - S_union)
        # AI と工学的パンデミックのイベント相関（周辺は固定）
        pa, pb = p[0], p[1]
        pab = 1 - np.mean(np.exp(-bs[0] * I)) - np.mean(np.exp(-bs[1] * I)) + np.mean(np.exp(-(bs[0] + bs[1]) * I))
        pair_corr[f"lam{lam}_d{d}"] = float((pab - pa * pb) / math.sqrt(pa * (1 - pa) * pb * (1 - pb)))
res["common_shock_union_100y"] = {"grid": union_grid, "AI_bio_event_corr": pair_corr, "N": N2,
    "model": "λ(t)=λ0 e^{-t/τ}, 各ショックで全人為ハザードを(1+δ)倍（恒久）。周辺確率はOrd値に個別較正。ショック自体は非致死。"}

# ---------- 3. 長期：100年合計を 5/6 に固定したときの S(1万年)（H7c）----------
TS = [100, 1000, 10_000]
def longterm(lam, d, kind="mult", r=math.inf, n=N2):
    rng = np.random.default_rng([SEED, int(lam * 1000), 2])
    s = draw_shocks_decay(n, lam, 10_000.0, rng)
    f = (lambda T: I_mult(s, d, T)) if kind == "mult" else (lambda T: I_add(s, d, r, T))
    B = solve_b(f(100.0), ORD_TOTAL, extra=100 * H_NAT)
    out = {"B": B}
    for T in TS:
        x = np.exp(-B * f(float(T)) - H_NAT * T)
        out[f"S_{T}"] = float(x.mean()); out[f"SE_{T}"] = float(x.std(ddof=1) / math.sqrt(n))
    return out

lt = {}
for lam in LAM0S:
    for d in DELTAS:
        lt[f"mult_lam{lam}_d{d}"] = longterm(lam, d)
for lam in [0.3, 1.0]:
    for r in [math.inf, 200.0, 50.0, 10.0]:
        for d in [1.0, 5.0]:
            lt[f"add_r{r}_lam{lam}_d{d}"] = longterm(lam, d, "add", r)
S_indep_point = math.exp(-(math.log(6 / 5) - 100 * H_NAT) / (1 - math.exp(-0.5)) * (1 - math.exp(-10_000 / TAU)) - H_NAT * 10_000)
res["longterm_fixed_S100"] = {"results": lt, "S10000_no_shock_analytic": S_indep_point,
    "note": "各(λ0,δ)ごとに人為ハザード係数Bを E[S(100)]=5/6 に再較正。add=加法 1+δΣe^{-(t-s)/r}。"}

# ---------- 4. XPT：絶滅/大災害 中央値比（H8'）とショック率アンカー ----------
XPT = {"super": {"ext_total": 1.0, "cat_total": 9.05, "ext_AI": 0.38, "cat_AI": 2.13, "ext_nuc": 0.074, "cat_nuc": 4.0},
       "expert": {"ext_total": 6.0, "cat_total": 20.0, "ext_AI": 3.0, "cat_AI": 12.0, "ext_nuc": 0.55, "cat_nuc": 8.0}}
q = {g: {k: XPT[g][f"ext_{k}"] / XPT[g][f"cat_{k}"] for k in ["total", "AI", "nuc"]} for g in XPT}
q_ratio = {k: q["expert"][k] / q["super"][k] for k in ["total", "AI", "nuc"]}
lam_anchor = {g: -math.log(1 - XPT[g]["cat_total"] / 100) / 78 * 100 for g in XPT}  # 回/世紀
res["xpt"] = {"inputs_percent_medians": XPT, "q_ext_over_cat": q, "expert_over_super": q_ratio,
    "shock_rate_anchor_per_century": lam_anchor,
    "note": "中央値の比であり個人の比ではない。2022→2100 を78年とし一様率を仮定。"}

# ---------- 5. 一定ショック率・恒久乗法：長期生存の境界（H10）----------
def const_rate_survival(lam_c, d, n, Ts, rng):
    lam = lam_c / 100; L = math.log1p(d)
    Tmax = max(Ts)
    s_prev = np.zeros(n); j = 0
    acc = {T: np.zeros(n) for Ts_ in [Ts] for T in Ts_}
    alive = np.ones(n, bool)
    while alive.any():
        gap = rng.exponential(1 / lam, size=n)
        s_next = s_prev + gap
        for T in Ts:
            a = np.minimum(s_prev, T); b = np.minimum(s_next, T)
            e1 = np.minimum(j * L - a / TAU, 700.0); e2 = np.minimum(j * L - b / TAU, 700.0)
            acc[T] += TAU * (np.exp(e1) - np.exp(e2)) * alive
        s_prev = s_next; j += 1
        alive = s_prev < Tmax
        if j > 200_000: break
    return acc

N5 = 20_000
TS5 = [100, 1000, 10_000, 30_000, 100_000]
h10 = {}
for d in [1.0, 3.0]:
    lam_path = 100 / (TAU * math.log1p(d)); lam_mean = 100 / (TAU * d)
    grid = sorted(set([round(x, 3) for x in [0.4 * lam_mean, 0.8 * lam_mean, lam_mean,
                    (lam_mean + lam_path) / 2, 0.95 * lam_path, 1.1 * lam_path, 1.5 * lam_path]]))
    for lc in grid:
        rng = np.random.default_rng([SEED, int(lc * 1e4), int(d * 10), 5])
        acc = const_rate_survival(lc, d, N5, TS5, rng)
        B = solve_b(acc[100], ORD_TOTAL, extra=100 * H_NAT)
        row = {"lam_per_century": lc, "delta": d, "B": B,
               "drift_path_per_kyr": 1000 * (lc / 100 * math.log1p(d) - 1 / TAU),
               "drift_mean_per_kyr": 1000 * (lc / 100 * d - 1 / TAU)}
        for T in TS5:
            x = np.exp(-B * acc[T])  # 人為のみ（自然床は除外して境界を見る）
            row[f"Santh_{T}"] = float(x.mean()); row[f"SE_{T}"] = float(x.std(ddof=1) / math.sqrt(N5))
        h10[f"d{d}_lam{lc}"] = row
    h10[f"thresholds_d{d}"] = {"lam_mean_per_century": lam_mean, "lam_path_per_century": lam_path}
res["const_rate_threshold"] = h10
res["xpt_threshold_delta"] = {g: {"delta_path": math.exp(100 / (TAU * lam_anchor[g])) - 1,
                                  "delta_mean": 100 / (TAU * lam_anchor[g])} for g in lam_anchor}

# ---------- 判定 ----------
drops = [u_ind - union_grid[f"lam{l}_d{d}"] for l in LAM0S for d in [0.0, 0.5, 1.0, 2.0]]
mono = all(lt[f"mult_lam{l}_d{DELTAS[i]}"]["S_10000"] >= lt[f"mult_lam{l}_d{DELTAS[i+1]}"]["S_10000"]
           for l in LAM0S for i in range(len(DELTAS) - 1))
res["verdicts"] = {
    "H7a": bool(res["ord_consistency"]["independent_in_band"]),
    "H7b_max_drop_delta_le2": float(max(drops)), "H7b": bool(max(drops) < 0.01),
    "H7c_monotone": bool(mono),
    "H8prime_max_ratio": float(max(q_ratio.values())), "H8prime": bool(max(q_ratio.values()) >= 10),
}
json.dump(res, open("results.json", "w"), indent=1, ensure_ascii=False, default=float)

# ---------- 図 ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot([float(r) for r in uc], list(uc.values()), "k-o", ms=3)
ax[0].axhline(ORD_TOTAL, color="r", ls="--", label="Ord total 1/6")
ax[0].axhspan(*band, color="r", alpha=0.1, label="'~1 in 6' band")
ax[0].axvline(rho_star, color="gray", ls=":")
ax[0].set_xlabel("latent correlation rho (1-factor Gaussian copula)"); ax[0].set_ylabel("P(any existential catastrophe, 100y)")
ax[0].set_title("Ord marginals fixed: total vs correlation"); ax[0].legend(fontsize=8)
for lam in LAM0S:
    ax[1].plot(DELTAS, [union_grid[f"lam{lam}_d{d}"] for d in DELTAS], "-o", ms=3, label=f"lambda0={lam}/century")
ax[1].axhline(ORD_TOTAL, color="r", ls="--"); ax[1].axhspan(*band, color="r", alpha=0.1)
ax[1].set_xlabel("delta (hazard x(1+delta) per shock)"); ax[1].set_title("Common-shock model: total 100y risk")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig_union_correlation.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for lam in LAM0S:
    ax[0].plot(DELTAS, [lt[f"mult_lam{lam}_d{d}"]["S_10000"] for d in DELTAS], "-o", ms=3, label=f"lambda0={lam}")
ax[0].axhline(S_indep_point, color="k", ls=":", label="no shock")
ax[0].set_xlabel("delta"); ax[0].set_ylabel("S(10,000y) | S(100)=5/6"); ax[0].set_title("Decaying shocks, permanent multiplicative")
ax[0].legend(fontsize=8)
for d in [1.0, 3.0]:
    keys = [k for k in h10 if k.startswith(f"d{d}_lam")]
    xs = [h10[k]["lam_per_century"] for k in keys]
    for T, st in [(10_000, "--"), (100_000, "-")]:
        ax[1].plot(xs, [h10[k][f"Santh_{T}"] for k in keys], st, marker="o", ms=3, label=f"delta={d}, T={T}")
    th = h10[f"thresholds_d{d}"]
    ax[1].axvline(th["lam_path_per_century"], color="C3" if d == 1 else "C4", ls=":")
    ax[1].axvline(th["lam_mean_per_century"], color="C3" if d == 1 else "C4", ls="-.", alpha=0.5)
ax[1].set_yscale("log"); ax[1].set_xlabel("constant shock rate (per century)"); ax[1].set_ylabel("anthropogenic-only survival")
ax[1].set_title("H10: dotted=pathwise threshold, dash-dot=mean threshold"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig_longterm_thresholds.png", dpi=120); plt.close()

print(json.dumps(res, indent=1, ensure_ascii=False, default=float))
print(f"runtime_sec={time.time() - t0:.1f}")
