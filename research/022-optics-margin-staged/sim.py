"""研究022：余裕係数 c の部分コヒーレント結像計算（open_problems 71）と段階書式（72）。
事前登録：PREREGISTRATION.md（初回実行前）。seed 固定。numpy＋標準ライブラリのみ。
長さの単位は λ/NA（瞳の遮断周波数 NA/λ = 1）。
"""
import json, time, sys, platform, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
SEED = 20261022
rng = np.random.default_rng(SEED)
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
R = {"seed": SEED, "python": sys.version.split()[0], "platform": platform.platform(),
     "preregistration": open(f"{HERE}/PREREGISTRATION.timestamp.txt").read().strip()}
R["facts"] = dict(
    zuylen1981_realistic_formula="d = lambda / (1.3 NA) under uncontrolled lighting; 1.7 with good modern lens and central illumination; old biconvex lenses rarely exceed 1.5, as low as 0.5",
    zuylen1981_rayleigh_limit="spherical aberration limit = centre intensity reduced to 80% (Strehl 0.8); free aperture chosen at 2-5x Rayleigh limit; with parallel light useful images up to 20x",
    zuylen1981_optimal_simple_lens_um=[1.11, 1.17],
    zuylen1981_utrecht_measured_um=1.35, zuylen1981_range_um=[1.35, 4.0],
    rayleigh_dip="26% dip between maxima (Wikipedia Optical resolution)",
    abbe_condenser="r = 1.22 lambda/(NA_obj + NA_cond) (Wikipedia Optical resolution)",
)
PERIOD_UM = 1000.0 / 1180.0      # Piql 1 周期（µm）。s=1 の画素幅は PERIOD_UM/2

# ---------------- 正規分布の裾（erfc、Numerical Recipes erfcc、相対誤差<1.2e-7） ----------------
def erfc(x):
    x = np.asarray(x, float); z = np.abs(x); t = 1.0 / (1.0 + 0.5 * z)
    r = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
        t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return np.where(x >= 0, r, 2.0 - r)
def Phi(x): return 0.5 * erfc(-np.asarray(x) / math.sqrt(2))

# ---------------- 光学系 ----------------
RAYLEIGH_RMS = math.sqrt(-math.log(0.8)) / (2 * math.pi)   # Strehl 0.8 ≒ RMS 0.0754 波長（Maréchal 近似）
R["rayleigh_rms_waves"] = RAYLEIGH_RMS
def pupil(gx, gy, Rmult):
    rho2 = gx * gx + gy * gy
    inside = rho2 <= 1.0
    z40 = math.sqrt(5) * (6 * rho2 * rho2 - 6 * rho2 + 1)
    ph = 2 * math.pi * Rmult * RAYLEIGH_RMS * z40
    return np.where(inside, np.exp(1j * ph), 0.0)

def source_points(sigma, n=64):
    k = np.arange(n) + 0.5
    r = sigma * np.sqrt(k / n); th = k * math.pi * (3 - math.sqrt(5))
    return np.stack([r * np.cos(th), r * np.sin(th)], 1)

# 格子のコントラスト（50%デューティの振幅格子、周期 P）
MH = 12
def grating_contrast(P, sigma, Rm, nx=256):
    src = source_points(sigma)
    m = np.arange(-MH, MH + 1)
    cm = np.where(m == 0, 0.5, np.sin(np.pi * m / 2) / (np.pi * np.where(m == 0, 1, m)))
    x = np.linspace(0, P, nx, endpoint=False)
    ph = np.exp(2j * np.pi * np.outer(m, x) / P)               # (M, nx)
    I = np.zeros(nx)
    for fx, fy in src:
        Pm = pupil(m / P + fx, np.full(m.shape, fy), Rm)       # (M,)
        E = (cm * Pm) @ ph
        I += np.abs(E) ** 2
    I /= len(src)
    return float((I.max() - I.min()) / (I.max() + I.min()))

P_SCAN = np.exp(np.linspace(np.log(0.3), np.log(4.0), 260))
def p_res(contrasts, cth):
    ok = contrasts >= cth
    # 最大の P から下へ、初めて閾値を下回る点まで（以上の P はすべて閾値以上）
    if not ok[-1]: return float("inf")
    i = len(ok) - 1
    while i > 0 and ok[i - 1]: i -= 1
    if i == 0: return float(P_SCAN[0])
    # 線形補間（対数 P）
    c0, c1 = contrasts[i - 1], contrasts[i]
    t = (cth - c0) / (c1 - c0)
    return float(np.exp(np.log(P_SCAN[i - 1]) + t * (np.log(P_SCAN[i]) - np.log(P_SCAN[i - 1]))))

# 画素中心での像強度（離散畳み込み核、分離可能な積分）
DG = 0.05
G1 = np.arange(-1.7, 1.7 + 1e-9, DG)
GX, GY = np.meshgrid(G1, G1, indexing="ij")
NPIX = 128
def kernels(w, sigma, Rm, r):
    src = source_points(sigma)
    off = np.arange(-r, r + 1)
    ex = (w * np.sinc(w * G1))[:, None] * np.exp(2j * np.pi * np.outer(G1, off) * w) * DG   # (G, O)
    A = np.stack([pupil(GX + fx, GY + fy, Rm) for fx, fy in src])                            # (S, G, G)
    return np.einsum("gm,sgh,hn->smn", ex, A, ex, optimize=True)                            # (S, O, O)

def image_2d(patterns, K, r):
    S = K.shape[0]
    Kp = np.zeros((S, NPIX, NPIX), complex)
    idx = np.arange(-r, r + 1) % NPIX
    Kp[:, idx[:, None], idx[None, :]] = K
    FK = np.fft.fft2(Kp)
    out = []
    for a in patterns:
        Fa = np.fft.fft2(a)
        E = np.fft.ifft2(FK * Fa[None])
        out.append((np.abs(E) ** 2).mean(0))
    ones = np.abs(K.sum(axis=(1, 2))) ** 2
    return [o / ones.mean() for o in out]

N1D = 1 << 16
def image_1d(b, K):
    k1 = K.sum(axis=2)                       # y 方向に一様な縦縞
    r = (k1.shape[1] - 1) // 2
    kp = np.zeros((k1.shape[0], N1D), complex)
    kp[:, np.arange(-r, r + 1) % N1D] = k1
    E = np.fft.ifft(np.fft.fft(kp, axis=1) * np.fft.fft(b)[None], axis=1)
    I = (np.abs(E) ** 2).mean(0)
    return I / (np.abs(k1.sum(1)) ** 2).mean()

SIGN = (0.02, 0.035, 0.05, 0.07, 0.10)
TH = np.linspace(0.0, 1.3, 261)
BINS = np.linspace(-0.2, 2.0, 2201); BC = 0.5 * (BINS[1:] + BINS[:-1])
def p_opt(I, a):
    hc, _ = np.histogram(I[a == 1], BINS); hd, _ = np.histogram(I[a == 0], BINS)
    N = a.size
    res = {}
    for sn in SIGN:
        pc = (hc[None, :] * Phi((TH[:, None] - BC[None, :]) / sn)).sum(1)
        pd = (hd[None, :] * Phi((BC[None, :] - TH[:, None]) / sn)).sum(1)
        res[sn] = float(((pc + pd) / N).min())
    # 雑音なし（ISI のみ）
    cc = np.cumsum(hc); cd = np.cumsum(hd)
    res["isi_only"] = float(((cc + (cd[-1] - cd)) / N).min())  # 閾値以下の明部＋閾値超の暗部
    return res

SIGMAS = (0.1, 0.2, 0.3, 0.45, 0.6)
RMULTS = (0.0, 1.0, 2.0, 3.0, 5.0)
CTHS = (0.03, 0.05, 0.10, 0.15, 0.20)
W_GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 26))

# ---- 格子分解能 ----
print("grating...", flush=True)
GRAT = {}
for sg in SIGMAS:
    for Rm in RMULTS:
        con = np.array([grating_contrast(P, sg, Rm) for P in P_SCAN])
        GRAT[(sg, Rm)] = dict(contrast=con, P_res={c: p_res(con, c) for c in CTHS})
R["grating_P_res"] = {f"sigma{sg}|R{Rm}": {str(c): v for c, v in GRAT[(sg, Rm)]["P_res"].items()}
                      for (sg, Rm) in GRAT}
VALID = {str(sg): dict(P_res=GRAT[(sg, 0.0)]["P_res"][0.05], abbe=1 / (1 + sg),
                       ratio=GRAT[(sg, 0.0)]["P_res"][0.05] * (1 + sg)) for sg in SIGMAS}
R["validation_abbe"] = VALID

# ---- 画素誤り ----
print("pixels...", flush=True)
pats = [rng.integers(0, 2, (NPIX, NPIX)).astype(float) for _ in range(8)]
bar = rng.integers(0, 2, N1D).astype(float)
PERR = {}   # (sg, Rm, dim) -> list over w of dict
for wi, w in enumerate(W_GRID):
    r = int(min(20, math.ceil(6.0 / w)))
    for sg in SIGMAS:
        for Rm in RMULTS:
            K = kernels(w, sg, Rm, r)
            I2 = np.concatenate([x.ravel() for x in image_2d(pats, K, r)])
            A2 = np.concatenate([x.ravel() for x in pats])
            PERR.setdefault((sg, Rm, "2d"), []).append(p_opt(I2, A2))
            if Rm in (0.0, 3.0) or sg == 0.3:
                I1 = image_1d(bar, K)
                PERR.setdefault((sg, Rm, "1d"), []).append(p_opt(I1, bar))
    print(f"  w={w:.3f} done {time.time()-t0:.0f}s", flush=True)

def w_star(lst, sn, target):
    p = np.array([d[sn] for d in lst])
    ok = p <= target
    if not ok[-1]: return float("inf")
    i = len(ok) - 1
    while i > 0 and ok[i - 1]: i -= 1
    if i == 0: return float(W_GRID[0])
    lp0, lp1 = math.log(max(p[i - 1], 1e-300)), math.log(max(p[i], 1e-300))
    t = (math.log(target) - lp0) / (lp1 - lp0) if lp1 != lp0 else 1.0
    return float(np.exp(np.log(W_GRID[i - 1]) + t * (np.log(W_GRID[i]) - np.log(W_GRID[i - 1]))))

TARGETS = (1e-4, 1e-3)
CTAB = {}
for (sg, Rm, dim), lst in PERR.items():
    for sn in SIGN:
        for tg in TARGETS:
            ws = w_star(lst, sn, tg)
            for cth in CTHS:
                pr = GRAT[(sg, Rm)]["P_res"][cth]
                CTAB[(sg, Rm, dim, sn, tg, cth)] = 2 * ws / pr if math.isfinite(ws) and math.isfinite(pr) else float("inf")
R["c_table"] = {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}|{k[4]}|{k[5]}": v for k, v in CTAB.items()}
R["p_opt_curves"] = {f"sigma{k[0]}|R{k[1]}|{k[2]}": [dict(w=float(W_GRID[i]), **{str(kk): vv for kk, vv in d.items()})
                     for i, d in enumerate(v)] for k, v in PERR.items()}

# ---- 仮説 a, b, c, e ----
H = {}
H["H34e"] = dict(points=VALID, supported=bool(all(abs(v["ratio"] - 1) <= 0.15 for v in VALID.values())))
cA = CTAB[(0.3, 0.0, "2d", 0.05, 1e-4, 0.15)]
cB1 = CTAB[(0.3, 0.0, "1d", 0.05, 1e-4, 0.15)]
cC = CTAB[(0.3, 3.0, "2d", 0.05, 1e-4, 0.15)]
H["H34a"] = dict(c=cA, supported=bool(1.0 <= cA <= 2.0))
H["H34b"] = dict(c_2d=cA, c_1d=cB1, ratio=cA / cB1, supported=bool(cA / cB1 >= 1.2))
H["H34c"] = dict(c_R0=cA, c_R3=cC, ratio=cC / cA, supported=bool(cC / cA >= 1.2))

# c の分布（格子上一様）
def c_samples(tg, dim="2d", sig=SIGMAS, rms=RMULTS):
    return np.array([CTAB[(sg, Rm, dim, sn, tg, cth)] for sg in sig for Rm in rms for sn in SIGN for cth in CTHS])
CD = {str(tg): c_samples(tg) for tg in TARGETS}
def qs(x):
    f = x[np.isfinite(x)]
    return dict(p10=float(np.percentile(x, 10)), p50=float(np.percentile(x, 50)), p90=float(np.percentile(x, 90)),
                frac_inf=float(np.mean(~np.isfinite(x))), frac_in_1_2=float(np.mean((x >= 1) & (x <= 2))),
                min=float(f.min()), max_finite=float(f.max()))
R["c_distribution"] = {k: qs(v) for k, v in CD.items()}
R["c_distribution_R_le2"] = {str(tg): qs(c_samples(tg, rms=(0.0, 1.0, 2.0))) for tg in TARGETS}
R["c_distribution_1d"] = {str(tg): qs(np.array([CTAB[k] for k in CTAB if k[2] == "1d" and k[4] == tg])) for tg in TARGETS}

# ---------------- 配分（021 の事前分布を再利用、c だけ置換） ----------------
print("allocation...", flush=True)
M_SINGLE = math.sqrt(1e-4 * 3e-3)
F_BURST, L_BURST = 1e-3, 100
PX_PER_DATA_BYTE_PIQL = (4096 * 2160) / 2e6
PIQL_REEL = 120e9
def _binom_pmf(n, p):
    k = np.arange(n + 1)
    c = np.array([math.comb(n, i) for i in k], dtype=float)
    p = np.asarray(p, dtype=float)[..., None]
    return c * p ** k * (1 - p) ** (n - k)
def residual_analytic(n, p, m, f, L, nbits=8):   # 021 の解析式（同一）
    p = np.atleast_1d(np.asarray(p, float)); m = np.broadcast_to(m, p.shape).astype(float)
    L = np.broadcast_to(L, p.shape).astype(float)
    q0 = p + m - 2 * p * m
    alpha = f / L; beta = 1.0 / L
    pi = alpha / (alpha + beta)
    S_ = p.size
    e = np.zeros((S_, n + 1)); thr = n // 2
    for k in range(n + 1):
        a = _binom_pmf(k, np.full(S_, 0.5)); b = _binom_pmf(n - k, q0)
        tot = np.zeros((S_, n + 1))
        for i in range(k + 1):
            tot[:, i:i + n - k + 1] += a[:, i:i + 1] * b
        e[:, k] = tot[:, thr + 1:].sum(axis=1)
    T = np.zeros((S_, n + 1, n + 1))
    for k in range(n + 1):
        stay = _binom_pmf(k, 1 - beta); enter = _binom_pmf(n - k, alpha)
        for i in range(k + 1):
            T[:, k, i:i + n - k + 1] += stay[:, i:i + 1] * enter
    v = _binom_pmf(n, pi) * (1 - e)
    for _ in range(nbits - 1):
        v = np.einsum("sk,skj->sj", v, T) * (1 - e)
    return np.clip(1.0 - v.sum(axis=1), 0, 1)

S = 4000
def logu(lo, hi, n=S): return np.exp(rng.uniform(np.log(lo), np.log(hi), n))
p_ref = logu(1e-4, 1e-2); beta_s = rng.uniform(0, 2, S); m_s = logu(1e-4, 3e-3)
d_res = logu(1.35, 4.0)
d_guide = logu(1.1, 1.6)            # 手引き成功時【推測：Zuylen の最適単レンズ計算値と Utrecht 実測から】
d_guide_pess = logu(1.35, 2.3)      # 探索的：悲観的な手引き効果
c021 = logu(1.0, 2.0)
c022 = CD["0.0001"][rng.integers(0, CD["0.0001"].size, S)]
c022_1e3 = CD["0.001"][rng.integers(0, CD["0.001"].size, S)]
S_LIST = (1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20)
def area_per_MB(n, s): return 1e6 * 8 * n * s * s / PX_PER_DATA_BYTE_PIQL / PIQL_REEL
RES = {}
for n in (5, 7):
    for s in S_LIST:
        p_s = p_ref * (s / 2.0) ** (-beta_s)
        RES[(n, s)] = residual_analytic(n, np.minimum(p_s, 0.49), m_s, F_BURST, np.full(S, max(1.0, L_BURST / s)))
TG = {5: 1e-4, 7: 1e-6}
def readable(s, d, c): return s * PERIOD_UM >= c * d
def joint(n, s, d, c): return (RES[(n, s)] <= TG[n]) & readable(s, d, c)

PRES = {}
for name, c in (("c021", c021), ("c022", c022), ("c022_1e-3", c022_1e3)):
    PRES[name] = {str(s): float(readable(s, d_res, c).mean()) for s in S_LIST}
R["P_readable_by_s"] = PRES
smin = {k: min([s for s in S_LIST if v[str(s)] >= 0.9], default=None) for k, v in PRES.items()}
R["s_min_P90"] = smin
H["H34d"] = dict(s_min=smin["c022"], s_min_021=smin["c021"], supported=bool(smin["c022"] is not None and smin["c022"] >= 7))

BUDGET = 0.01
def single_best(c, thr=0.9):
    rows = []
    for s in S_LIST:
        P = float(joint(7, s, d_res, c).mean()); D = BUDGET / area_per_MB(7, s)
        rows.append(dict(s=s, P=P, D=D, EMB=P * D))
    feas = [r for r in rows if r["P"] >= thr]
    b = max(feas, key=lambda r: r["D"]) if feas else None
    e = max(rows, key=lambda r: r["EMB"])
    return dict(rows=rows, best_P90=b, best_expected=e)
SB = {"c022": single_best(c022), "c021": single_best(c021)}
R["single_layer"] = {k: dict(best_P90=v["best_P90"], best_expected=v["best_expected"]) for k, v in SB.items()}

G_MB = 0.1
sg_min = min([s for s in S_LIST if joint(5, s, d_res, c022).mean() >= 0.99], default=None)
P_guide_read = float(joint(5, sg_min, d_res, c022).mean()) if sg_min else 0.0
area_g = G_MB * area_per_MB(5, sg_min) if sg_min else float("inf")
R["guide_layer"] = dict(G_MB=G_MB, s_g=sg_min, P_read=P_guide_read, area_frac=area_g, share_of_budget=area_g / BUDGET)
PG = np.round(np.linspace(0, 1, 21), 3)
def staged(dg, c=c022):
    out = []
    for pg in PG:
        pe = P_guide_read * pg
        rows = []
        for s in S_LIST:
            P = pe * float(joint(7, s, dg, c).mean()) + (1 - pe) * float(joint(7, s, d_res, c).mean())
            D = (BUDGET - area_g) / area_per_MB(7, s)
            rows.append(dict(s=s, P=P, D=D, EMB=P * D))
        feas = [r for r in rows if r["P"] >= 0.9]
        b = max(feas, key=lambda r: r["D"]) if feas else None
        e = max(rows, key=lambda r: r["EMB"])
        out.append(dict(P_g=float(pg), best_P90=b, best_expected=e))
    return out
ST = {"base": staged(d_guide), "pessimistic_guide": staged(d_guide_pess)}
R["staged"] = ST
def crossings(st, single):
    eS = single["best_expected"]["EMB"]; dS = single["best_P90"]["D"] if single["best_P90"] else 0.0
    pe = next((r["P_g"] for r in st if r["best_expected"]["EMB"] >= eS), None)
    p9 = next((r["P_g"] for r in st if r["best_P90"] and r["best_P90"]["D"] > dS), None)
    return dict(P_g_star_expected=pe, P_g_min_P90=p9, single_EMB=eS, single_D_P90=dS,
                ratio_expected_at_Pg1=st[-1]["best_expected"]["EMB"] / eS,
                ratio_P90_at_Pg1=(st[-1]["best_P90"]["D"] / dS) if st[-1]["best_P90"] and dS else None)
CR = {k: crossings(v, SB["c022"]) for k, v in ST.items()}
R["staged_crossings"] = CR
H["H34f"] = dict(**CR["base"], supported=bool(CR["base"]["P_g_star_expected"] is not None and CR["base"]["P_g_star_expected"] < 0.5))
H["H34g"] = dict(P_g_min_P90=CR["base"]["P_g_min_P90"],
                 supported=bool(CR["base"]["P_g_min_P90"] is None or CR["base"]["P_g_min_P90"] >= 0.8))
R["hypotheses"] = H

# 探索的：Zuylen の k=λ/(d·NA) との照合（格子 C_th 別）
R["exploratory_k_equivalent"] = {f"sigma{sg}|R{Rm}": {str(c): 1 / GRAT[(sg, Rm)]["P_res"][c] for c in CTHS}
                                 for sg in SIGMAS for Rm in (0.0, 3.0)}

# ---------------- 図 ----------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for Rm, ls in ((0.0, "-"), (3.0, "--")):
    for sg, col in ((0.1, "C0"), (0.3, "C1"), (0.6, "C2")):
        ax[0].plot(P_SCAN, GRAT[(sg, Rm)]["contrast"], ls, color=col, label=f"σ={sg}, R={Rm:g}")
ax[0].axhline(0.15, color="gray", lw=0.7); ax[0].set_xscale("log"); ax[0].set_xlabel("grating period (λ/NA)")
ax[0].set_ylabel("image contrast"); ax[0].set_title("Binary grating contrast"); ax[0].legend(fontsize=7)
for (sg, Rm, dim), col, ls in (((0.3, 0.0, "2d"), "C1", "-"), ((0.3, 0.0, "1d"), "C1", ":"), ((0.3, 3.0, "2d"), "C3", "-"),
                               ((0.1, 0.0, "2d"), "C0", "-"), ((0.6, 0.0, "2d"), "C2", "-")):
    p = [max(d[0.05], 1e-12) for d in PERR[(sg, Rm, dim)]]
    ax[1].loglog(W_GRID, p, ls, color=col, label=f"σ={sg}, R={Rm:g}, {dim}")
ax[1].axhline(1e-4, color="gray", lw=0.7); ax[1].set_ylim(1e-8, 0.6)
ax[1].set_xlabel("pixel width w (λ/NA)"); ax[1].set_ylabel("pixel error (σ_n=0.05, best threshold)")
ax[1].set_title("Random-pixel reading error"); ax[1].legend(fontsize=7)
f = CD["0.0001"]; f = np.where(np.isfinite(f), f, 12)
ax[2].hist(f, bins=np.linspace(0.5, 12.5, 49), color="C1", alpha=0.8)
ax[2].axvspan(1, 2, color="gray", alpha=0.2, label="021 prior [1,2]")
ax[2].set_xlabel("margin factor c (12 = unreadable)"); ax[2].set_ylabel("grid cells")
ax[2].set_title("c over (σ, R, C_th, σ_n), target 1e-4"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_optics_c.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for k, col in (("c021", "gray"), ("c022", "C1"), ("c022_1e-3", "C2")):
    ax[0].plot(S_LIST, [PRES[k][str(s)] for s in S_LIST], "o-", color=col, label=k)
ax[0].axhline(0.9, color="k", lw=0.7); ax[0].set_xlabel("line-width multiple s"); ax[0].set_ylabel("P(readable)")
ax[0].set_title("Readability with Leeuwenhoek-class optics"); ax[0].legend(fontsize=8)
for k, col in (("base", "C1"), ("pessimistic_guide", "C3")):
    ax[1].plot(PG, [r["best_expected"]["EMB"] for r in ST[k]], "-", color=col, label=f"staged {k}: expected MB")
    ax[1].plot(PG, [r["best_P90"]["D"] if r["best_P90"] else 0 for r in ST[k]], "--", color=col, label=f"staged {k}: D_max (P≥0.9)")
ax[1].axhline(SB["c022"]["best_expected"]["EMB"], color="k", lw=0.8, label="single: expected MB")
ax[1].axhline(SB["c022"]["best_P90"]["D"] if SB["c022"]["best_P90"] else 0, color="k", ls="--", lw=0.8, label="single: D_max (P≥0.9)")
ax[1].set_xlabel("P_g (lens guide succeeds)"); ax[1].set_ylabel("MB (1% reel, residual ≤1e-6)")
ax[1].set_title("Staged format vs single layer"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_readability_staged.png", dpi=130); plt.close()

R["runtime_seconds"] = time.time() - t0
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o): return "inf"
    if isinstance(o, (np.floating,)): return clean(float(o))
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o
json.dump(clean(R), open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(clean(H), ensure_ascii=False, indent=1))
print("c_distribution", json.dumps(clean(R["c_distribution"])))
print("s_min", smin, "guide", R["guide_layer"])
print("single", json.dumps(clean(R["single_layer"])))
print("crossings", json.dumps(clean(CR)))
print(f"runtime {R['runtime_seconds']:.1f}s")
