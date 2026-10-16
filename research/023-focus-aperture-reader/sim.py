"""研究023：焦点・開口・σ を選ぶ読者の c と試験パターン段数（open_problems 74）。
事前登録：PREREGISTRATION.md（初回実行前）。seed 固定。numpy＋標準ライブラリのみ。
022 の部分コヒーレント結像にデフォーカス Z(2,0) を追加。
"""
import json, time, sys, platform, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
SEED = 20261023
rng = np.random.default_rng(SEED)
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
R022 = json.load(open(f"{HERE}/../022-optics-margin-staged/results.json"))
R = {"seed": SEED, "python": sys.version.split()[0], "platform": platform.platform(),
     "preregistration": open(f"{HERE}/PREREGISTRATION.timestamp.txt").read().strip(),
     "note_resolution": "NPIX=96, 4 patterns, 48 source points (022 used 128/8/64)"}

def erfc(x):
    x = np.asarray(x, float); z = np.abs(x); t = 1.0 / (1.0 + 0.5 * z)
    r = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
        t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return np.where(x >= 0, r, 2.0 - r)
def Phi(x): return 0.5 * erfc(-np.asarray(x) / math.sqrt(2))

RAYLEIGH_RMS = math.sqrt(-math.log(0.8)) / (2 * math.pi)
R["rayleigh_rms_waves"] = RAYLEIGH_RMS

def pupil(gx, gy, Rmult, D=0.0):
    rho2 = gx * gx + gy * gy
    inside = rho2 <= 1.0
    z40 = math.sqrt(5) * (6 * rho2 * rho2 - 6 * rho2 + 1)
    z20 = math.sqrt(3) * (2 * rho2 - 1)
    ph = 2 * math.pi * RAYLEIGH_RMS * (Rmult * z40 + D * z20)
    return np.where(inside, np.exp(1j * ph), 0.0)

def source_points(sigma, n=48):
    k = np.arange(n) + 0.5
    r = sigma * np.sqrt(k / n); th = k * math.pi * (3 - math.sqrt(5))
    return np.stack([r * np.cos(th), r * np.sin(th)], 1)

MH = 12
_GCACHE = {}
def grating_contrast(P, sigma, Rm, D=0.0, nx=256):
    if not (math.isfinite(P) and P > 1e-6):
        return 0.0
    key = (round(P, 5), sigma, Rm, D, nx)
    if key in _GCACHE: return _GCACHE[key]
    src = source_points(sigma)
    m = np.arange(-MH, MH + 1)
    cm = np.where(m == 0, 0.5, np.sin(np.pi * m / 2) / (np.pi * np.where(m == 0, 1, m)))
    x = np.linspace(0, P, nx, endpoint=False)
    ph = np.exp(2j * np.pi * np.outer(m, x) / P)
    I = np.zeros(nx)
    for fx, fy in src:
        Pm = pupil(m / P + fx, np.full(m.shape, fy), Rm, D)
        E = (cm * Pm) @ ph
        I += np.abs(E) ** 2
    I /= len(src)
    val = float((I.max() - I.min()) / (I.max() + I.min() + 1e-30))
    _GCACHE[key] = val
    return val

P_SCAN = np.exp(np.linspace(np.log(0.3), np.log(4.0), 200))
def p_res(contrasts, cth):
    ok = contrasts >= cth
    if not ok[-1]: return float("inf")
    i = len(ok) - 1
    while i > 0 and ok[i - 1]: i -= 1
    if i == 0: return float(P_SCAN[0])
    c0, c1 = contrasts[i - 1], contrasts[i]
    t = (cth - c0) / (c1 - c0) if c1 != c0 else 1.0
    return float(np.exp(np.log(P_SCAN[i - 1]) + t * (np.log(P_SCAN[i]) - np.log(P_SCAN[i - 1]))))

DG = 0.05
G1 = np.arange(-1.7, 1.7 + 1e-9, DG)
GX, GY = np.meshgrid(G1, G1, indexing="ij")
NPIX = 96
def kernels(w, sigma, Rm, D, r):
    src = source_points(sigma)
    off = np.arange(-r, r + 1)
    ex = (w * np.sinc(w * G1))[:, None] * np.exp(2j * np.pi * np.outer(G1, off) * w) * DG
    A = np.stack([pupil(GX + fx, GY + fy, Rm, D) for fx, fy in src])
    return np.einsum("gm,sgh,hn->smn", ex, A, ex, optimize=True)

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

SIGN = (0.02, 0.035, 0.05, 0.07, 0.10)
TH = np.linspace(0.0, 1.3, 261)
BINS = np.linspace(-0.2, 2.0, 1801); BC = 0.5 * (BINS[1:] + BINS[:-1])
def p_opt(I, a):
    hc, _ = np.histogram(I[a == 1], BINS); hd, _ = np.histogram(I[a == 0], BINS)
    N = a.size
    res = {}
    for sn in SIGN:
        pc = (hc[None, :] * Phi((TH[:, None] - BC[None, :]) / sn)).sum(1)
        pd = (hd[None, :] * Phi((BC[None, :] - TH[:, None]) / sn)).sum(1)
        res[sn] = float(((pc + pd) / N).min())
    return res

SIGMAS = (0.1, 0.2, 0.3, 0.45, 0.6)
RMULTS = (0.0, 1.0, 2.0, 3.0, 5.0)
DEFOCUS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
CTHS = (0.03, 0.05, 0.10, 0.15, 0.20)
W_GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 20))
TARGETS = (1e-4, 1e-3)

# ---- grating with defocus ----
print("grating...", flush=True)
GRAT = {}
for sg in SIGMAS:
    for Rm in RMULTS:
        for D in DEFOCUS:
            con = np.array([grating_contrast(P, sg, Rm, D) for P in P_SCAN])
            GRAT[(sg, Rm, D)] = dict(contrast=con, P_res={c: p_res(con, c) for c in CTHS})
R["grating_P_res"] = {f"sigma{sg}|R{Rm}|D{D}": {str(c): v for c, v in GRAT[(sg, Rm, D)]["P_res"].items()}
                      for (sg, Rm, D) in GRAT}

# ---- pixel errors (or load optics_cache.npz) ----
CACHE_PATH = f"{HERE}/optics_cache.npz"
if os.path.exists(CACHE_PATH):
    print("loading optics_cache...", flush=True)
    Z = np.load(CACHE_PATH, allow_pickle=True)
    WSTAR = {tuple(k): float(v) for k, v in Z["WSTAR"].item().items()}
    CTAB = {}
    for (sg, Rm, D, sn, tg), ws in WSTAR.items():
        for cth in CTHS:
            pr = GRAT[(sg, Rm, D)]["P_res"][cth]
            CTAB[(sg, Rm, D, sn, tg, cth)] = (2 * ws / pr) if math.isfinite(ws) and math.isfinite(pr) else float("inf")
else:
    print("pixels...", flush=True)
    pats = [rng.integers(0, 2, (NPIX, NPIX)).astype(float) for _ in range(4)]
    PERR = {}
    ncombo = len(W_GRID) * len(SIGMAS) * len(RMULTS) * len(DEFOCUS)
    done = 0
    for wi, w in enumerate(W_GRID):
        r = int(min(16, math.ceil(5.5 / w)))
        for sg in SIGMAS:
            for Rm in RMULTS:
                for D in DEFOCUS:
                    K = kernels(w, sg, Rm, D, r)
                    I2 = np.concatenate([x.ravel() for x in image_2d(pats, K, r)])
                    A2 = np.concatenate([x.ravel() for x in pats])
                    PERR.setdefault((sg, Rm, D), []).append(p_opt(I2, A2))
                    done += 1
        print(f"  w={w:.3f} ({done}/{ncombo}) {time.time()-t0:.0f}s", flush=True)

    def w_star(lst, sn, target):
        p = np.array([d[sn] for d in lst])
        ok = p <= target
        if not ok[-1]: return float("inf")
        i = len(ok) - 1
        while i > 0 and ok[i - 1]: i -= 1
        if i == 0: return float(W_GRID[0])
        lp0, lp1 = math.log(max(p[i - 1], 1e-300)), math.log(max(p[i], 1e-300))
        tt = (math.log(target) - lp0) / (lp1 - lp0) if lp1 != lp0 else 1.0
        return float(np.exp(np.log(W_GRID[i - 1]) + tt * (np.log(W_GRID[i]) - np.log(W_GRID[i - 1]))))

    WSTAR = {}
    CTAB = {}
    for (sg, Rm, D), lst in PERR.items():
        for sn in SIGN:
            for tg in TARGETS:
                ws = w_star(lst, sn, tg)
                WSTAR[(sg, Rm, D, sn, tg)] = ws
                for cth in CTHS:
                    pr = GRAT[(sg, Rm, D)]["P_res"][cth]
                    CTAB[(sg, Rm, D, sn, tg, cth)] = (2 * ws / pr) if math.isfinite(ws) and math.isfinite(pr) else float("inf")
    np.savez(CACHE_PATH, WSTAR=np.array(WSTAR, dtype=object))
    print(f"saved {CACHE_PATH}", flush=True)

# method check vs 022
c022_ref = float(R022["hypotheses"]["H34a"]["c"])
c023_ref = CTAB[(0.3, 0.0, 0.0, 0.05, 1e-4, 0.15)]
R["validation_vs_022"] = dict(c022=c022_ref, c023_D0=c023_ref, ratio=c023_ref / c022_ref,
                              within_15pct=bool(abs(c023_ref / c022_ref - 1) <= 0.15))

# ---- joint optimization ----
print("optimize...", flush=True)
def wopt_aperture_only(R_lens, sn, tg):
    best = (float("inf"), None)
    for Rk in RMULTS:
        if Rk > R_lens or (Rk == 0 and R_lens > 0): continue
        t = 1.0 if R_lens == 0 else (Rk / R_lens) ** 0.25
        for sg in SIGMAS:
            w = WSTAR[(sg, Rk, 0.0, sn, tg)] / t
            if w < best[0]:
                best = (w, dict(R_read=Rk, t=t, sigma_read=sg, D=0.0))
    return best

def wopt_joint(R_lens, sn, tg):
    best = (float("inf"), None)
    for Rk in RMULTS:
        if Rk > R_lens or (Rk == 0 and R_lens > 0): continue
        t = 1.0 if R_lens == 0 else (Rk / R_lens) ** 0.25
        for sg in SIGMAS:
            for D in DEFOCUS:
                w = WSTAR[(sg, Rk, D, sn, tg)] / t
                if w < best[0]:
                    best = (w, dict(R_read=Rk, t=t, sigma_read=sg, D=D))
    return best

def Pres_meas(sg, Rl, cth):
    return GRAT[(sg, Rl, 0.0)]["P_res"][cth]

COPT_AP = {}; COPT_JT = {}
for tg in TARGETS:
    for Rl in RMULTS:
        for sn in SIGN:
            wa, howa = wopt_aperture_only(Rl, sn, tg)
            wj, howj = wopt_joint(Rl, sn, tg)
            for sg in SIGMAS:
                for cth in CTHS:
                    pr = Pres_meas(sg, Rl, cth)
                    ca = 2 * wa / pr if math.isfinite(pr) else float("inf")
                    cj = 2 * wj / pr if math.isfinite(pr) else float("inf")
                    COPT_AP[(tg, Rl, sn, sg, cth)] = dict(c=ca, how=howa)
                    COPT_JT[(tg, Rl, sn, sg, cth)] = dict(c=cj, how=howj)

def q(xs):
    x = np.array(xs, float)
    return dict(p10=float(np.percentile(x, 10)), p50=float(np.percentile(x, 50)),
                p90=float(np.percentile(x, 90)), frac_inf=float(np.mean(~np.isfinite(x))),
                frac_in_1_2=float(np.mean((x >= 1) & (x <= 2))), n=int(x.size))

def cvals(table, tg):
    return [v["c"] for k, v in table.items() if k[0] == tg]

R["c_aperture_only"] = {str(tg): q(cvals(COPT_AP, tg)) for tg in TARGETS}
R["c_joint_focus"] = {str(tg): q(cvals(COPT_JT, tg)) for tg in TARGETS}
R["c_joint_by_R_lens"] = {str(Rl): q([v["c"] for k, v in COPT_JT.items() if k[0] == 1e-4 and k[1] == Rl]) for Rl in RMULTS}
R["c_aperture_by_R_lens"] = {str(Rl): q([v["c"] for k, v in COPT_AP.items() if k[0] == 1e-4 and k[1] == Rl]) for Rl in RMULTS}

# focus helps? fraction of cells where joint picks D!=0 and improves w
improve = []
focus_chosen = []
for Rl in RMULTS:
    for sn in SIGN:
        for tg in (1e-4,):
            wa, ha = wopt_aperture_only(Rl, sn, tg)
            wj, hj = wopt_joint(Rl, sn, tg)
            improve.append(wj / wa if wa > 0 else 1.0)
            focus_chosen.append(hj["D"] != 0.0)
R["focus_improvement"] = dict(median_w_ratio=float(np.median(improve)),
                               mean_w_ratio=float(np.mean(improve)),
                               frac_D_nonzero=float(np.mean(focus_chosen)),
                               p10=float(np.percentile(improve, 10)),
                               p90=float(np.percentile(improve, 90)))
R["example_choices"] = dict(
    R5_sn005=wopt_joint(5.0, 0.05, 1e-4)[1],
    R3_sn005=wopt_joint(3.0, 0.05, 1e-4)[1],
    R2_sn005=wopt_joint(2.0, 0.05, 1e-4)[1],
    R0_sn005=wopt_joint(0.0, 0.05, 1e-4)[1],
)

# ---- test-pattern aperture selection ----
print("test patterns...", flush=True)
# Candidate stop-down factors (relative aperture t), map to nearest R_read = R_lens * t^4
T_CAND = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.82, 0.90, 1.00])
C_VIS = 0.10
SIG_VIS = 0.03
N_MC = 200  # noise realizations per cell

def nearest_R(R_lens, t):
    if R_lens == 0: return 0.0, 1.0
    R_target = R_lens * (t ** 4)
    # exclude Rk=0 when lens has aberration (same rule as explore.py / wopt)
    cands = [Rk for Rk in RMULTS if 0 < Rk <= R_lens]
    if not cands: return R_lens, 1.0
    Rk = min(cands, key=lambda x: abs(x - R_target))
    t_eff = (Rk / R_lens) ** 0.25
    return Rk, t_eff

def stages_periods(N, Pmin=0.5, Pmax=3.0):
    if N == 1: return np.array([Pmax])
    return np.exp(np.linspace(np.log(Pmax), np.log(Pmin), N))  # coarse → fine

def t_star(R_lens, sg_illum, sn, tg=1e-4):
    """Optimal physical aperture t for reading (minimize w*), focus free, σ free among grid? 
    Register: σ chosen freely; for fairness fix illumination σ = sg_illum for pattern match,
    but allow σ_read among SIGMAS for t* (same as joint)."""
    w, how = wopt_joint(R_lens, sn, tg)
    return how["t"], how

def choose_aperture(R_lens, sg, D_focus, periods, t_cands, rng_local):
    """Registered rule: maximize #stages with C>=C_VIS; tie → finest resolved; tie → max C of finest."""
    best = None  # (n_ok, finest_idx, C_finest, t)
    for t in t_cands:
        Rk, t_eff = nearest_R(R_lens, t)
        # physical period in λ/NA_full units → in current units: P_phys * t_eff? 
        # Period P in object space (λ/NA_full). At stopped aperture, pupil units change:
        # spatial freq in pupil units scales with NA, so period in λ/NA_new = P_full * t_eff
        # Contrast computed in system with NA_new uses P' = P_phys / (λ/NA_new) = P_phys * NA_new/λ
        # If P_phys is given in λ/NA_full, then P' = P_phys * t_eff
        n_ok = 0; finest = -1; C_fine = -1.0
        for i, P_full in enumerate(periods):
            P_sys = P_full * t_eff
            C = grating_contrast(P_sys, sg, Rk, D_focus)
            C_obs = C + rng_local.normal(0, SIG_VIS)
            if C_obs >= C_VIS:
                n_ok += 1
                finest = i
                C_fine = C_obs
        key = (n_ok, finest, C_fine, -abs(t - 1.0))  # prefer more open on final tie? use -|t-1| weak
        # Actually prefer the stop that resolves finest: already have finest. For same finest, higher C.
        # Store t_eff as the chosen physical aperture ratio
        cand = (n_ok, finest, C_fine, t_eff)
        if best is None or cand[:3] > best[:3]:
            best = cand
    return best[3] if best else 1.0

# Evaluate H35c/d
pattern_results = {}
for N in (3, 5, 7, 9):
    periods = stages_periods(N)
    hits20 = []; hits10 = []; ratios = []; cells = 0
    for Rl in (1.0, 2.0, 3.0, 5.0):
        for sg in (0.2, 0.3, 0.45):
            sn = 0.05
            t_opt, how = t_star(Rl, sg, sn)
            # Use D from joint opt for reading; for patterns use D=0 or the chosen D?
            # Register: reader first focuses on a coarse pattern (D chosen to max contrast of coarsest),
            # then picks aperture. Exploratory detail: use D from joint for both (optimistic).
            D_use = how["D"]
            for trial in range(N_MC):
                rng_t = np.random.default_rng(SEED + 1000 * N + cells * N_MC + trial)
                t_ch = choose_aperture(Rl, sg, D_use, periods, T_CAND, rng_t)
                ratio = t_ch / t_opt if t_opt > 0 else float("inf")
                ratios.append(ratio)
                hits20.append(abs(ratio - 1) <= 0.20)
                hits10.append(abs(ratio - 1) <= 0.10)
            cells += 1
    pattern_results[str(N)] = dict(
        n_cells=cells, n_trials=cells * N_MC,
        P_within_20pct=float(np.mean(hits20)),
        P_within_10pct=float(np.mean(hits10)),
        ratio_p50=float(np.median(ratios)),
        ratio_p10=float(np.percentile(ratios, 10)),
        ratio_p90=float(np.percentile(ratios, 90)),
        periods=[float(p) for p in periods],
    )
R["test_pattern"] = pattern_results

# Also: procedure that first picks focus on coarsest stage, then aperture (more realistic)
def choose_focus_then_aperture(R_lens, sg, periods, t_cands, rng_local):
    # focus at full aperture on coarsest period
    P0 = periods[0]
    best_D, best_C = 0.0, -1e9
    for D in DEFOCUS:
        C = grating_contrast(P0, sg, R_lens if R_lens > 0 else 0.0, D) + rng_local.normal(0, SIG_VIS)
        if C > best_C:
            best_C, best_D = C, D
    t_ch = choose_aperture(R_lens, sg, best_D, periods, t_cands, rng_local)
    return t_ch, best_D

pattern_focus_first = {}
for N in (3, 5, 7):
    periods = stages_periods(N)
    hits20 = []; ratios = []
    for Rl in (1.0, 2.0, 3.0, 5.0):
        for sg in (0.2, 0.3, 0.45):
            t_opt, how = t_star(Rl, sg, 0.05)
            for trial in range(N_MC):
                rng_t = np.random.default_rng(SEED + 50000 + 1000 * N + int(Rl*10+sg*100) * N_MC + trial)
                t_ch, _ = choose_focus_then_aperture(Rl, sg, periods, T_CAND, rng_t)
                ratio = t_ch / t_opt if t_opt > 0 else float("inf")
                ratios.append(ratio); hits20.append(abs(ratio - 1) <= 0.20)
    pattern_focus_first[str(N)] = dict(P_within_20pct=float(np.mean(hits20)),
                                       ratio_p50=float(np.median(ratios)))
R["test_pattern_focus_first_exploratory"] = pattern_focus_first

# ---- hypotheses ----
c_ap = R["c_aperture_only"]["0.0001"]["p50"]
c_jt = R["c_joint_focus"]["0.0001"]["p50"]
H = {}
H["H35e"] = dict(**R["validation_vs_022"], supported=bool(R["validation_vs_022"]["within_15pct"]))
H["H35a"] = dict(c_joint=c_jt, c_aperture_only=c_ap, ratio=c_jt / c_ap,
                 supported=bool(c_jt / c_ap <= 0.95))
H["H35b"] = dict(c_joint=c_jt, threshold=1.45, supported=bool(c_jt <= 1.45))
H["H35c"] = dict(N=5, P_within_20pct=pattern_results["5"]["P_within_20pct"],
                 supported=bool(pattern_results["5"]["P_within_20pct"] >= 0.80))
H["H35d"] = dict(N=3, P_within_20pct=pattern_results["3"]["P_within_20pct"],
                 supported=bool(pattern_results["3"]["P_within_20pct"] < 0.50))
R["hypotheses"] = H

# exploratory: readability with joint c vs 022 explore
PERIOD_UM = 1000.0 / 1180.0
S_LIST = (1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20)
Salloc = 4000
rng_alloc = np.random.default_rng(SEED + 7)
def logu(lo, hi, n=Salloc): return np.exp(rng_alloc.uniform(np.log(lo), np.log(hi), n))
d_res = logu(1.35, 4.0)
allc = np.array(cvals(COPT_JT, 1e-4))
c_all = allc[rng_alloc.integers(0, allc.size, Salloc)]
R["P_readable_by_s_joint"] = {str(s): float((s * PERIOD_UM >= c_all * d_res).mean()) for s in S_LIST}
R["s_min_P90_joint"] = min([s for s in S_LIST if R["P_readable_by_s_joint"][str(s)] >= 0.9], default=None)

# 022 explore median for reference
try:
    E022 = json.load(open(f"{HERE}/../022-optics-margin-staged/explore_results.json"))
    R["ref_022_explore_c_p50"] = E022["c_opt_distribution"]["0.0001"]["p50"]
except Exception as e:
    R["ref_022_explore_c_p50"] = None

# ---- figures ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
xs = list(RMULTS)
ax[0].plot(xs, [R["c_aperture_by_R_lens"][str(r)]["p50"] for r in xs], "o-", label="aperture+σ only (D=0), median")
ax[0].plot(xs, [R["c_joint_by_R_lens"][str(r)]["p50"] for r in xs], "s-", label="focus+aperture+σ, median")
ax[0].plot(xs, [R["c_joint_by_R_lens"][str(r)]["p90"] for r in xs], "s--", label="… 90th pct")
ax[0].axhline(1.45, color="gray", ls=":", label="022 explore median 1.45")
ax[0].axhspan(1, 2, color="gray", alpha=0.15, label="021 prior")
ax[0].set_xlabel("spherical aberration of lens (× Rayleigh)"); ax[0].set_ylabel("margin factor c")
ax[0].set_title("c with vs without focus optimization"); ax[0].legend(fontsize=7)
Ns = [3, 5, 7, 9]
ax[1].plot(Ns, [pattern_results[str(n)]["P_within_20pct"] for n in Ns], "o-", label="|t/t*−1|≤20%")
ax[1].plot(Ns, [pattern_results[str(n)]["P_within_10pct"] for n in Ns], "s-", label="|t/t*−1|≤10%")
ax[1].axhline(0.80, color="k", lw=0.7, label="H35c threshold 0.8")
ax[1].axhline(0.50, color="k", ls="--", lw=0.7, label="H35d threshold 0.5")
ax[1].set_xlabel("number of test-pattern stages N"); ax[1].set_ylabel("P(within X% of t*)")
ax[1].set_title("Test-pattern aperture selection"); ax[1].legend(fontsize=7); ax[1].set_ylim(0, 1.05)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_focus_c_and_stages.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
# focus benefit heatmap-ish: w ratio by R
for Rl in RMULTS:
    ratios_R = []
    for sn in SIGN:
        wa, _ = wopt_aperture_only(Rl, sn, 1e-4)
        wj, _ = wopt_joint(Rl, sn, 1e-4)
        ratios_R.append(wj / wa)
    ax[0].plot([Rl]*len(ratios_R), ratios_R, "o", alpha=0.7)
ax[0].axhline(1.0, color="k", lw=0.7)
ax[0].axhline(0.95, color="gray", ls="--", label="H35a 5% line")
ax[0].set_xlabel("R_lens"); ax[0].set_ylabel("w*_joint / w*_aperture"); ax[0].legend(fontsize=7)
ax[0].set_title("Does focus shrink required pixel width?")
# readability
ax[1].plot(S_LIST, [R["P_readable_by_s_joint"][str(s)] for s in S_LIST], "o-", label="joint focus+aperture c")
ax[1].axhline(0.9, color="k", lw=0.7)
ax[1].set_xlabel("line-width multiple s"); ax[1].set_ylabel("P(readable)")
ax[1].set_title("Readability (Leeuwenhoek d, joint c)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_focus_benefit_readability.png", dpi=130); plt.close()

R["runtime_seconds"] = time.time() - t0

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o): return "inf"
    if isinstance(o, (np.floating,)): return clean(float(o))
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, (np.bool_,)): return bool(o)
    return o

json.dump(clean(R), open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(clean(H), ensure_ascii=False, indent=1))
print("c_ap", R["c_aperture_only"])
print("c_jt", R["c_joint_focus"])
print("focus", R["focus_improvement"])
print("patterns", json.dumps(clean(pattern_results), indent=1))
print("s_min", R["s_min_P90_joint"], "P7", R["P_readable_by_s_joint"].get("7"))
print(f"runtime {R['runtime_seconds']:.1f}s")
