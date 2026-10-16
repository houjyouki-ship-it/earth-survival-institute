"""研究021：手読み層の配分最適化（open_problems 68）と常時多数決の残留の確定（69）。
事前登録：PREREGISTRATION.md（初回実行前）。seed 固定。numpy＋標準ライブラリのみ。
"""
import json, time, sys, platform, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
SEED = 20261021
rng = np.random.default_rng(SEED)
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
R = {"seed": SEED, "python": sys.version.split()[0], "platform": platform.platform(),
     "preregistration": open(f"{HERE}/PREREGISTRATION.timestamp.txt").read().strip()}

# ---- 020 から継承（同じチャネル） ----
M_SINGLE = math.sqrt(1e-4 * 3e-3)
F_BURST, L_BURST = 1e-3, 100
PX_PER_DATA_BYTE_PIQL = (4096 * 2160) / 2e6
PIQL_REEL = 120e9
PERIOD_UM = 1000.0 / 1180.0          # Piql 1180 lp/mm の1周期（µm）【事実：piql.com、018 sources】
FACT = dict(
    zuylen1981_utrecht=dict(magnification=266, NA=0.37, resolving_power_um_measured=1.35),
    zuylen1981_measured_range_um=[1.35, 4.0],
    zuylen1981_url="https://docs.backyardbrains.com/assets/files/1981_Zuylen_The_microscopes_Leeuwenhoek-8223514bf5c4d49541b838032778ce8b.pdf",
)
R["facts"] = FACT
r020 = json.load(open(f"{HERE}/../020-hand-residual-1e6/results.json"))
R["inherited_020"] = dict(A5f_residual_2MB=r020["hypotheses"]["H32d"]["candidates"][4]["residual"],
                          A5f_cost_p50=r020["cost_py_per_MB"]["A5f"]["p50"],
                          A7f_cost_p50=r020["cost_py_per_MB"]["A7f"]["p50"],
                          seed_reps_A5f=[r.get("A5f") for r in r020["seed_replicates"]])

# ---------------- 解析：k（バースト中の複製数）マルコフ連鎖 × 8ビット DP ----------------
def _binom_pmf(n, p):
    k = np.arange(n + 1)
    c = np.array([math.comb(n, i) for i in k], dtype=float)
    p = np.asarray(p, dtype=float)[..., None]
    return c * p ** k * (1 - p) ** (n - k)

def residual_analytic(n, p, m, f, L, nbits=8):
    """常時 n 本多数決（n 奇数）のバイト誤り率。p, m, L は同形の配列可。"""
    p = np.atleast_1d(np.asarray(p, float)); m = np.broadcast_to(m, p.shape).astype(float)
    L = np.broadcast_to(L, p.shape).astype(float)
    q0 = p + m - 2 * p * m
    alpha = f / L; beta = 1.0 / L
    pi = alpha / (alpha + beta)
    S = p.size
    # e[s,k] = P(多数決誤り | k 本がバースト中)
    e = np.zeros((S, n + 1))
    thr = n // 2
    for k in range(n + 1):
        a = _binom_pmf(k, np.full(S, 0.5))          # バースト中の誤り数
        b = _binom_pmf(n - k, q0)                    # それ以外の誤り数
        tot = np.zeros((S, n + 1))
        for i in range(k + 1):
            tot[:, i:i + n - k + 1] += a[:, i:i + 1] * b
        e[:, k] = tot[:, thr + 1:].sum(axis=1)
    # 遷移 T[s,k,k']
    T = np.zeros((S, n + 1, n + 1))
    for k in range(n + 1):
        stay = _binom_pmf(k, 1 - beta)               # 残る本数
        enter = _binom_pmf(n - k, alpha)             # 入る本数
        for i in range(k + 1):
            T[:, k, i:i + n - k + 1] += stay[:, i:i + 1] * enter
    v = _binom_pmf(n, pi) * (1 - e)
    for _ in range(nbits - 1):
        v = np.einsum("sk,skj->sj", v, T) * (1 - e)
    # 1-P_ok を桁落ちなく：誤り確率の和で直接は難しいので log1p で評価
    ok = v.sum(axis=1)
    return np.clip(1.0 - ok, 0, 1)

def residual_analytic_split(n, p, m, f, L):
    tot = residual_analytic(n, p, m, f, L)[0]
    nob = residual_analytic(n, p, m, 1e-300, L)[0]
    return dict(total=float(tot), no_burst=float(nob), burst_share=float(1 - nob / tot) if tot > 0 else None)

# ---------------- MC 検証（020 と同じチャネル実装） ----------------
def channel(nbits, p, burst, rr):
    flip = rr.random(nbits) < p
    if burst:
        f, L = burst
        starts = np.nonzero(rr.random(nbits) < f / L)[0]
        lens = rr.geometric(1.0 / L, len(starts))
        for s, l in zip(starts, lens):
            e = min(nbits, s + int(l))
            flip[s:e] ^= (rr.random(e - s) < 0.5)
    return flip

def mc_majority(n, p, m, f, L, n_bytes_total, chunk_bytes, rr):
    errs, done = 0, 0
    while done < n_bytes_total:
        nb = min(chunk_bytes, n_bytes_total - done)
        nbits = nb * 8
        votes = np.zeros(nbits, dtype=np.int8)
        for _ in range(n):
            wrong = channel(nbits, p, (f, L), rr) ^ (rr.random(nbits) < m)
            votes += wrong.astype(np.int8)
        bad = (votes > n // 2).reshape(-1, 8).any(axis=1)
        errs += int(bad.sum()); done += nb
    return errs, done

def poisson_cdf(k, lam):
    s, term = 0.0, math.exp(-lam)
    for i in range(k + 1):
        if i > 0: term *= lam / i
        s += term
    return min(1.0, s)

def poisson_two_sided_p(k, lam):
    lo = poisson_cdf(k, lam)
    hi = 1.0 - (poisson_cdf(k - 1, lam) if k > 0 else 0.0)
    return min(1.0, 2 * min(lo, hi))

def poisson_interval(lam, q=0.025):
    lo = 0
    while poisson_cdf(lo, lam) < q: lo += 1
    hi = lo
    while poisson_cdf(hi, lam) < 1 - q: hi += 1
    return lo, hi

print("MC validation...", flush=True)
rr = np.random.default_rng(SEED + 1)
VAL = []
for name, n, p, m, L, MB in (("A5f_base_40MB", 5, 1e-3, M_SINGLE, 100, 40),
                             ("A3f_base_4MB", 3, 1e-3, M_SINGLE, 100, 4),
                             ("A5f_p4e-3_4MB", 5, 4e-3, M_SINGLE, 100, 4),
                             ("A5f_p3e-3_L20_4MB", 5, 3e-3, M_SINGLE, 20, 4),
                             ("A7f_p8e-3_4MB", 7, 8e-3, M_SINGLE, 100, 4)):
    ts = time.time()
    k, nbytes = mc_majority(n, p, m, F_BURST, L, int(MB * 1e6), 2_000_000, rr)
    ra = float(residual_analytic(n, p, m, F_BURST, L)[0])
    lam = ra * nbytes
    lo, hi = poisson_interval(lam)
    row = dict(name=name, n=n, p=p, m=m, L=L, bytes=nbytes, mc_errors=k, mc_rate=k / nbytes,
               analytic=ra, expected_errors=lam, poisson95=[lo, hi], inside=bool(lo <= k <= hi),
               two_sided_p=poisson_two_sided_p(k, lam), seconds=time.time() - ts)
    VAL.append(row); print(row, flush=True)
R["validation"] = VAL

# ---------------- H33a, b, e：基準条件の解析値 ----------------
base = {f"n{n}": residual_analytic_split(n, 1e-3, M_SINGLE, F_BURST, L_BURST) for n in (3, 5, 7, 9)}
base_double = {f"n{n}": residual_analytic_split(n, 1e-3, M_SINGLE * 0.1, F_BURST, L_BURST) for n in (3, 5, 7)}
R["analytic_base"] = dict(single=base, double_read=base_double)

def solve_p(n, target, m=M_SINGLE, L=L_BURST, f=F_BURST):
    lo, hi = math.log(1e-7), math.log(0.2)
    if residual_analytic(n, 1e-7, m, f, L)[0] > target: return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if residual_analytic(n, math.exp(mid), m, f, L)[0] > target: hi = mid
        else: lo = mid
    return math.exp(0.5 * (lo + hi))
pstar = {f"n{n}": solve_p(n, 1e-6) for n in (5, 7)}
pstar_exact = {f"n{n}": solve_p(n, -math.log(0.9) / 1e6) for n in (5, 7, 9)}
R["p_star"] = dict(target_1e6=pstar, target_exact1MB=pstar_exact, exact_1MB_threshold=-math.log(0.9) / 1e6)
# m だけで（p→0）1e-6 に達するか
R["m_floor"] = {f"n{n}": float(residual_analytic(n, 1e-9, M_SINGLE, F_BURST, L_BURST)[0]) for n in (3, 5, 7)}

# ---------------- 配分最適化（A） ----------------
print("allocation...", flush=True)
S = 4000
def logu(lo, hi, n=S): return np.exp(rng.uniform(np.log(lo), np.log(hi), n))
p_ref = logu(1e-4, 1e-2); beta_s = rng.uniform(0, 2, S); m_s = logu(1e-4, 3e-3)
d_res = logu(1.35, 4.0); c_fac = logu(1.0, 2.0)
d_res_wide = logu(0.5, 4.0)  # 探索的：手引きで光学が改善した場合
t_px2 = logu(0.05, 1.0); t_bit = logu(0.2, 2.0)
SEC_PER_PY = 2000 * 3600
N_LIST = (3, 5, 7, 9, 11)
S_LIST = (1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10)
TARGETS = {"prose_1e-4": 1e-4, "1e-6": 1e-6, "exact_1MB": -math.log(0.9) / 1e6}
BUDGETS = (0.003, 0.01, 0.03)

def area_per_MB(n, s): return 1e6 * 8 * n * s * s / PX_PER_DATA_BYTE_PIQL / PIQL_REEL

def resolve_prob(s, d=d_res, c=c_fac): return (s * PERIOD_UM >= c * d)

configs = []
for n in N_LIST:
    for s in S_LIST:
        p_s = p_ref * (s / 2.0) ** (-beta_s)
        L_s = np.full(S, max(1.0, L_BURST / s))
        res = {mode: residual_analytic(n, np.minimum(p_s, 0.49), m_s * (0.1 if mode == "double" else 1.0), F_BURST, L_s)
               for mode in ("single", "double")}
        ok_res = resolve_prob(s)
        ok_res_wide = resolve_prob(s, d_res_wide, 1.0)
        ok_res_c1 = resolve_prob(s, d_res, 1.0)  # 探索的：Leeuwenhoek 級・余裕係数なし
        for mode in ("single", "double"):
            cost = (n * 8 * t_px2 * (2 if mode == "double" else 1) + 8 * t_bit * 0.5) * 1e6 / SEC_PER_PY
            row = dict(n=n, s=s, mode=mode, area_frac_per_MB=area_per_MB(n, s),
                       P_resolve=float(ok_res.mean()), P_resolve_wide_c1=float(ok_res_wide.mean()),
                       residual_p50=float(np.median(res[mode])), residual_p90=float(np.percentile(res[mode], 90)),
                       cost_py_per_MB_p50=float(np.median(cost)), cost_py_per_MB_p95=float(np.percentile(cost, 95)))
            for tn, tv in TARGETS.items():
                row[f"P_res_ok_{tn}"] = float((res[mode] <= tv).mean())
                row[f"P_joint_{tn}"] = float(((res[mode] <= tv) & ok_res).mean())
                row[f"P_joint_wide_{tn}"] = float(((res[mode] <= tv) & ok_res_wide).mean())
                row[f"P_joint_c1_{tn}"] = float(((res[mode] <= tv) & ok_res_c1).mean())
            configs.append(row)
R["configs"] = configs

# 018 の梯子：T2 到達と寿命
Z = rng.standard_normal((2, S))
T2 = 20 * np.exp(0.7 * Z[0]) + 30 * np.exp(0.7 * Z[1])
life = logu(500, 2000) * logu(0.3, 1.0) / np.log(2) ** 0.5 * (-np.log(rng.uniform(size=S))) ** 0.5
L_alloc = logu(10, 1000)

def best(target, budget, key="P_joint", thr=0.9):
    feas = [c for c in configs if c[f"{key}_{target}"] >= thr]
    if not feas: return None
    b = min(feas, key=lambda c: (c["area_frac_per_MB"], c["cost_py_per_MB_p50"]))
    D = budget / b["area_frac_per_MB"]
    t_read = b["cost_py_per_MB_p50"] * D / L_alloc
    return dict(n=b["n"], s=b["s"], mode=b["mode"], area_frac_per_MB=b["area_frac_per_MB"],
                P_joint=b[f"{key}_{target}"], D_max_MB=D, cost_py_per_MB_p50=b["cost_py_per_MB_p50"],
                total_py_p50=b["cost_py_per_MB_p50"] * D,
                P_read_before_life=float(np.mean(T2 + t_read < life)),
                P_T2_before_life=float(np.mean(T2 < life)))
ALLOC = {}
for key in ("P_joint", "P_joint_c1", "P_joint_wide"):
    for tn in TARGETS:
        for B in BUDGETS:
            ALLOC[f"{key}|{tn}|{B}"] = best(tn, B, key)
R["allocation"] = ALLOC
# 020 推奨（n=5, s=2, single）の実力
rec020 = [c for c in configs if c["n"] == 5 and c["s"] == 2 and c["mode"] == "single"][0]
R["rec020_under_priors"] = rec020

# 機械読み層の機会費用（018：T3 m23=943 で寿命内 0.26）
P_T3 = json.load(open(f"{HERE}/../018-physical-min-reader/results.json"))["ladder"]["P_read_before_life"]["T3_m23_943"]
R["machine_layer_opportunity"] = {str(B): dict(MB_lost=B * PIQL_REEL / 1e6, expected_MB_lost_T3=B * PIQL_REEL / 1e6 * P_T3)
                                  for B in BUDGETS}

# ---------------- 判定 ----------------
H = {}
v0 = VAL[0]
H["H33a"] = dict(analytic=base["n5"]["total"], mc_errors=v0["mc_errors"], expected=v0["expected_errors"],
                 poisson95=v0["poisson95"], in_range=bool(3e-7 <= base["n5"]["total"] <= 1e-6),
                 supported=bool(3e-7 <= base["n5"]["total"] <= 1e-6 and v0["inside"]))
H["H33b"] = dict(p_star=pstar["n5"], margin=pstar["n5"] / 1e-3 if pstar["n5"] else None,
                 supported=bool(pstar["n5"] is not None and pstar["n5"] < 2e-3))
P_s2 = float(resolve_prob(2).mean())
s_cont = np.quantile(c_fac * d_res / PERIOD_UM, 0.9)
s_min_grid = min([s for s in S_LIST if resolve_prob(s).mean() >= 0.9], default=None)
H["H33c"] = dict(P_resolve_s2=P_s2, s_min_grid_P90=s_min_grid, s_min_continuous_P90=float(s_cont),
                 P_resolve_by_s={str(s): float(resolve_prob(s).mean()) for s in S_LIST},
                 supported=bool(P_s2 < 0.5 and s_min_grid is not None and s_min_grid >= 4))
b1 = ALLOC["P_joint|1e-6|0.01"]
H["H33d"] = dict(best=b1, supported=bool(b1 is None or b1["D_max_MB"] < 10))
thr = -math.log(0.9) / 1e6
H["H33e"] = dict(A5f=base["n5"]["total"], A7f=base["n7"]["total"], threshold=thr,
                 supported=bool(base["n5"]["total"] > thr and base["n7"]["total"] <= thr))
R["hypotheses"] = H

# ---------------- 図 ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ps = np.logspace(-4, -1.7, 60)
for n, col in ((3, "C0"), (5, "C1"), (7, "C2"), (9, "C3")):
    ax[0].loglog(ps, residual_analytic(n, ps, M_SINGLE, F_BURST, L_BURST), color=col, label=f"n={n} analytic (L=100)")
    ax[0].loglog(ps, residual_analytic(n, ps, M_SINGLE, F_BURST, 20), color=col, ls=":", lw=1)
for v in VAL:
    col = {3: "C0", 5: "C1", 7: "C2"}[v["n"]]
    y = max(v["mc_rate"], 1e-9); lo, hi = v["poisson95"]
    ax[0].errorbar([v["p"]], [y], yerr=[[max(y - lo / v["bytes"], 0)], [max(hi / v["bytes"] - y, 0)]],
                   fmt="s" if v["L"] == 100 else "^", color=col, ms=6, mec="k")
ax[0].axhline(1e-6, color="r", ls="--", lw=1); ax[0].axhline(thr, color="m", ls="--", lw=1)
ax[0].text(1.1e-4, 1.3e-6, "1e-6", color="r", fontsize=8); ax[0].text(1.1e-4, 1.4e-7, "exact 1 MB (P0.9)", color="m", fontsize=8)
ax[0].set_ylim(1e-12, 1e-1); ax[0].set_xlabel("independent pixel error p"); ax[0].set_ylabel("byte error (majority of n)")
ax[0].set_title("Analytic DP vs MC (markers; dotted L=20)"); ax[0].legend(fontsize=7)
for n, col in ((5, "C1"), (7, "C2"), (9, "C3")):
    yy = [[c for c in configs if c["n"] == n and c["s"] == s and c["mode"] == "single"][0]["P_joint_1e-6"] for s in S_LIST]
    ax[1].plot(S_LIST, yy, "o-", color=col, label=f"n={n} joint P (1e-6)")
ax[1].plot(S_LIST, [float(resolve_prob(s).mean()) for s in S_LIST], "k--", label="P(resolve), Leeuwenhoek-grade")
ax[1].plot(S_LIST, [float(resolve_prob(s, d_res_wide, 1.0).mean()) for s in S_LIST], "k:", label="P(resolve), improved optics, c=1")
ax[1].axhline(0.9, color="gray", lw=0.8)
ax[1].set_xlabel("line-width scale s (x Piql 0.42 µm)"); ax[1].set_ylabel("probability"); ax[1].legend(fontsize=7)
ax[1].set_title("Feasibility vs line width")
fig.tight_layout(); fig.savefig(f"{HERE}/fig_residual_and_feasibility.png", dpi=120)

fig, ax = plt.subplots(figsize=(7.5, 4.6))
for key, ls in (("P_joint", "-"), ("P_joint_wide", "--")):
    for tn, col in (("prose_1e-4", "C0"), ("1e-6", "C1"), ("exact_1MB", "C2")):
        ys = [ALLOC[f"{key}|{tn}|{B}"]["D_max_MB"] if ALLOC[f"{key}|{tn}|{B}"] else np.nan for B in BUDGETS]
        ax.loglog([100 * b for b in BUDGETS], ys, ls, marker="o", color=col,
                  label=f"{tn} ({'Leeuwenhoek' if key=='P_joint' else 'improved, c=1'})")
ax.axhline(10, color="gray", lw=0.8); ax.set_xlabel("area budget (% of one Piql reel)"); ax.set_ylabel("D_max (MB), P≥0.9")
ax.set_title("Priority data readable by hand"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{HERE}/fig_dmax_vs_budget.png", dpi=120)

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(dict(hypotheses=H, analytic_base=R["analytic_base"], p_star=R["p_star"], m_floor=R["m_floor"],
                      allocation=ALLOC, rec020=rec020), ensure_ascii=False, indent=1, default=float))
print("DONE", R["runtime_seconds"])
