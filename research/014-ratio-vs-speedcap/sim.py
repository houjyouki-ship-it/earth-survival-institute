"""研究014：追いつきは「比」か「速度上限」か — 事前登録検定（PREREGISTRATION.md）
(1) Maddison 2023 の所得倍増帯を FL>=1250 に広げ、先頭国4定義で log(追随の KM 中央年数) ~ log(先頭年数) の傾き β を推定。
(2) 記録レベルの右打ち切り対数正規 AFT（時代 γ を統制）＋国単位クラスター・ブートストラップ。
(3) 研究012/013 の読取梯子を同じ乱数で再実装（013 の値を厳密再現して検証）し、混合型 primer（比と速度上限の中間）で再評価。
seed 固定、numpy + matplotlib のみ、決定論的。
"""
import csv, json, time, math, collections, statistics
import numpy as np
t0 = time.time()
SEED = 20261015; SEED_BOOT = 20261017
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
SRC = HERE + "/../013-empirical-catchup-a/sources/owid_gdppc_maddison.csv"
R013 = json.load(open(HERE + "/../013-empirical-catchup-a/results.json"))

raw = collections.defaultdict(dict)
for r in csv.DictReader(open(SRC, encoding="utf-8")):
    if r["GDP per capita"] and r["Code"]:
        raw[r["Code"]][int(r["Year"])] = float(r["GDP per capita"])
AGG = {c for c in raw if c.startswith("OWID_")}
OIL = {"SAU","KWT","ARE","QAT","BHR","OMN","IRQ","IRN","LBY","DZA","AGO","NGA","GAB","GNQ","COG","VEN","ECU","TTO"}
YEARS = np.arange(1000, 2023); END = 2022.0

def annual(code):
    ys = np.array(sorted(raw[code])); vs = np.log(np.array([raw[code][y] for y in ys]))
    yy = np.arange(ys[0], ys[-1] + 1); lv = np.interp(yy, ys, vs)
    idx = np.searchsorted(ys, yy); gap = np.zeros(len(yy))
    for i, y in enumerate(yy):
        j = idx[i]; gap[i] = 0 if (j < len(ys) and ys[j] == y) else (ys[j] - ys[j-1])
    return yy, np.exp(lv), gap
SER = {c: annual(c) for c in raw}

def frontier(pioneers, max_gap=None):
    env = np.full(len(YEARS), -np.inf)
    for c in pioneers:
        yy, v, gap = SER[c]
        m = (YEARS >= yy[0]) & (YEARS <= yy[-1]); vv = np.interp(YEARS[m], yy, v)
        if max_gap is not None:
            gg = np.interp(YEARS[m], yy, gap); vv = np.where(gg <= max_gap, vv, -np.inf)
        env[m] = np.maximum(env[m], vv)
    return env
def smooth(env, w=31):
    lv = np.log(np.where(np.isfinite(env) & (env > 0), env, np.nan)); out = np.full(len(env), -np.inf); h = w // 2
    for i in range(len(env)):
        seg = lv[max(0, i - h): i + h + 1]
        if np.isfinite(lv[i]) and np.isfinite(seg).sum() > 0: out[i] = math.exp(np.nanmean(seg))
    return out
def cross_first(yy, v, L):
    if v[0] >= L: return None
    k = np.where(v >= L)[0]
    if len(k) == 0: return math.inf
    i = k[0]; f = (math.log(L) - math.log(v[i-1])) / (math.log(v[i]) - math.log(v[i-1]))
    return float(yy[i-1] + f)
def frontier_cross(env, L):
    k = np.where(env >= L)[0]; return float(YEARS[k[0]]) if len(k) else math.inf

def km(times, events):
    o = np.argsort(times, kind="stable"); t = np.asarray(times)[o]; e = np.asarray(events)[o]
    n = len(t); S = 1.0; ts, ss = [0.0], [1.0]; i = 0
    while i < n:
        j = i; d = 0
        while j < n and t[j] == t[i]: d += e[j]; j += 1
        if d > 0: S *= (1 - d / (n - i)); ts.append(float(t[i])); ss.append(S)
        i = j
    return np.array(ts), np.array(ss)
def km_quantile(ts, ss, q):
    F = 1 - ss; k = np.where(F >= q - 1e-12)[0]; return float(ts[k[0]]) if len(k) else None

def records(env, exclude, L_grid, FL_min, min_cens=0.5):
    recs, bands = [], []
    for L in L_grid:
        FL, F2L = frontier_cross(env, L), frontier_cross(env, 2 * L)
        if not (FL >= FL_min and F2L <= END): continue
        P = F2L - FL; bands.append({"L": L, "FL": FL, "F2L": F2L, "P": P})
        for c, (yy, v, gap) in SER.items():
            if c in AGG or c in exclude or c in OIL: continue
            cL = cross_first(yy, v, L)
            if cL is None or cL == math.inf or cL < F2L: continue
            c2 = cross_first(yy, v, 2 * L)
            if c2 is None: continue
            if c2 == math.inf: dur, ev = yy[-1] - cL, 0
            else: dur, ev = c2 - cL, 1
            if ev == 0 and dur < min_cens: continue
            recs.append({"code": c, "L": L, "P": P, "t_L": cL, "dur": float(dur), "event": ev})
    return recs, bands

def band_slope(recs, bands):
    rows = []
    for b in bands:
        rb = [r for r in recs if r["L"] == b["L"]]
        if len(rb) < 5: continue
        t, s = km([r["dur"] for r in rb], [r["event"] for r in rb]); md = km_quantile(t, s, 0.5)
        rows.append({**b, "n": len(rb), "n_events": int(sum(r["event"] for r in rb)), "km_median_dur": md})
    dd = [r for r in rows if r["km_median_dur"] is not None]
    out = {"bands": rows, "n_bands_used": len(dd)}
    if len(dd) > 2:
        X = np.log([r["P"] for r in dd]); Y = np.log([r["km_median_dur"] for r in dd]); Xc = X - X.mean()
        sl = float(Xc @ (Y - Y.mean()) / (Xc @ Xc)); rs = Y - Y.mean() - sl * Xc
        out.update(slope=sl, se=float(math.sqrt(rs @ rs / (len(dd) - 2) / (Xc @ Xc))), P_range=[float(min(r["P"] for r in dd)), float(max(r["P"] for r in dd))],
                   follower_median_range=[float(min(r["km_median_dur"] for r in dd)), float(max(r["km_median_dur"] for r in dd))])
    return out

# ---------- 右打ち切り対数正規 AFT（Nelder–Mead） ----------
SQ2 = math.sqrt(2.0)
_erfc = np.frompyfunc(math.erfc, 1, 1)
def log_sf(z):  # log(1-Φ(z))
    v = _erfc(z / SQ2).astype(float) / 2.0
    return np.where(v > 1e-300, np.log(np.maximum(v, 1e-300)), -0.5 * z * z - np.log(np.maximum(z, 1e-9) * math.sqrt(2 * math.pi)))
def make_nll(X, y, ev):
    def nll(th):
        b = th[:-1]; s = math.exp(th[-1]); z = (y - X @ b) / s
        return -float(np.sum(np.where(ev == 1, -0.5 * z * z - math.log(s) - 0.5 * math.log(2 * math.pi), log_sf(z))))
    return nll
def nelder_mead(f, x0, step=0.3, it=4000, tol=1e-10):
    n = len(x0); pts = [np.array(x0, float)] + [np.array(x0, float) + step * np.eye(n)[i] for i in range(n)]; fs = [f(p) for p in pts]
    for _ in range(it):
        o = np.argsort(fs); pts = [pts[i] for i in o]; fs = [fs[i] for i in o]
        if abs(fs[-1] - fs[0]) < tol: break
        c = np.mean(pts[:-1], axis=0); xr = c + (c - pts[-1]); fr = f(xr)
        if fr < fs[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fs[-2]: pts[-1], fs[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc_ = f(xc)
            if fc_ < fs[-1]: pts[-1], fs[-1] = xc, fc_
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]; fs = [fs[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(fs)); return pts[i], fs[i]
def design(recs, with_L=False):
    cols = [np.ones(len(recs)), np.log([r["P"] for r in recs]), (np.array([r["t_L"] for r in recs]) - 1950) / 50]
    if with_L: cols.append(np.log([r["L"] for r in recs]) - math.log(8000))
    return np.vstack(cols).T, np.log([r["dur"] for r in recs]), np.array([r["event"] for r in recs])
def fit_aft(recs, with_L=False, x0=None):
    X, y, ev = design(recs, with_L); f = make_nll(X, y, ev)
    if x0 is None:
        b0 = np.linalg.lstsq(X, y, rcond=None)[0]; x0 = list(b0) + [math.log(max(np.std(y), 0.1))]
    th, v = nelder_mead(f, x0); th, v = nelder_mead(f, th, step=0.05)   # 再起動で収束を確かめる
    names = ["alpha", "beta", "gamma"] + (["delta_logL"] if with_L else [])
    return {**{k: float(th[i]) for i, k in enumerate(names)}, "s": float(math.exp(th[-1])), "nll": v, "n": len(recs), "n_events": int(ev.sum())}, th
def boot_aft(recs, th0, B=200, with_L=False):
    rng = np.random.default_rng(SEED_BOOT); byc = collections.defaultdict(list)
    for r in recs: byc[r["code"]].append(r)
    codes = sorted(byc); out = collections.defaultdict(list)
    for _ in range(B):
        pick = rng.choice(len(codes), len(codes), replace=True); rr = [r for k in pick for r in byc[codes[k]]]
        res, _ = fit_aft(rr, with_L, x0=list(th0))
        for k in ("alpha", "beta", "gamma", "s"): out[k].append(res[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}, {k: [float(x) for x in v] for k, v in out.items()}

# ---------- (0) 013 の再現 ----------
P_PRIMARY = ["GBR", "NLD", "USA"]
L13 = [round(2000 * 2 ** (k / 4)) for k in range(0, 21)]
env1 = frontier(P_PRIMARY)
rec13, bd13 = records(env1, set(P_PRIMARY), L13, 1820, min_cens=-1)
t13, s13 = km([r["dur"] / r["P"] for r in rec13], [r["event"] for r in rec13])
bs13 = band_slope(rec13, bd13)
check013 = {"n_pairs": len(rec13), "n_pairs_013": R013["gdp_catchup"]["primary_first_UKNLUS_exoil"]["n_pairs"],
            "km_median_a": km_quantile(t13, s13, .5), "km_median_a_013": R013["gdp_catchup"]["primary_first_UKNLUS_exoil"]["km_median"],
            "slope": bs13.get("slope"), "slope_013": R013["gdp_catchup"]["primary_first_UKNLUS_exoil"]["EXPLORATORY_slope_logdur_vs_logpioneer"]["slope"]}
check013["reproduced"] = bool(check013["n_pairs"] == check013["n_pairs_013"] and abs(check013["km_median_a"] - check013["km_median_a_013"]) < 1e-9 and abs(check013["slope"] - check013["slope_013"]) < 1e-9)

# ---------- (1) 拡張した帯で4定義 ----------
L_GRID = [round(1000 * 2 ** (k / 4)) for k in range(0, 25)]
ALLC = [c for c in raw if c not in AGG and c not in OIL]
DEFS = {"D1_UK_NL_US": (env1, set(P_PRIMARY)), "D2_UK_only": (frontier(["GBR"]), {"GBR"}),
        "D3_world_max": (frontier(ALLC, max_gap=60), set()), "D4_D1_smoothed31": (smooth(env1), set(P_PRIMARY))}
BAND = {}; RECS = {}
for nm, (env, excl) in DEFS.items():
    rc, bd = records(env, excl, L_GRID, 1250); RECS[nm] = rc; BAND[nm] = band_slope(rc, bd)
    BAND[nm]["n_records"] = len(rc); BAND[nm]["n_countries"] = len({r["code"] for r in rc})
# 013 の帯（FL>=1820）だけ・新規の帯（FL<1820）だけ（D1）の内訳
b1 = BAND["D1_UK_NL_US"]["bands"]

# ---------- (2) AFT ----------
AFT = {}; AFT_TH = {}
for nm in DEFS:
    res, th = fit_aft(RECS[nm]); AFT[nm] = res; AFT_TH[nm] = th
boot1, bootdraws1 = boot_aft(RECS["D1_UK_NL_US"], AFT_TH["D1_UK_NL_US"])
AFT["D1_UK_NL_US"]["boot95"] = boot1
AFT_L, thL = fit_aft(RECS["D1_UK_NL_US"], with_L=True)
# β=1・β=0 を固定したときの尤度比（D1）
def fixed_beta_nll(recs, bfix):
    X, y, ev = design(recs); f = make_nll(X, y, ev)
    g = lambda th: f(np.array([th[0], bfix, th[1], th[2]]))
    th, v = nelder_mead(g, [AFT_TH["D1_UK_NL_US"][0], AFT_TH["D1_UK_NL_US"][2], AFT_TH["D1_UK_NL_US"][3]]); th, v = nelder_mead(g, th, step=0.05); return v
LR = {str(b): 2 * (fixed_beta_nll(RECS["D1_UK_NL_US"], b) - AFT["D1_UK_NL_US"]["nll"]) for b in (0.0, 0.5, 1.0)}
e_ref = float(np.median([r["t_L"] for r in RECS["D1_UK_NL_US"]]))

# ---------- (3) 読取梯子（012/013 と同じ乱数・同じ順序） ----------
rng = np.random.default_rng(SEED); N = 200_000; H = 1000.0
MEDIA = {"MDISC_BD": dict(tier=3, claim=(100, 500), site="room"), "Piql_film_QR": dict(tier=3, claim=(500, 2000), site="vault"),
         "Silica_glass": dict(tier=3, claim=(1e4, 1e5), site="room"), "DNA_silica": dict(tier=4, claim=(2000, 1e4), site="room"),
         "Microfilm_AgX": dict(tier=1, claim=(300, 500), site="room"), "Rosetta_Ni": dict(tier=2, claim=(1e3, 1e4), site="room"),
         "MOM_ceramic": dict(tier=1, claim=(1e4, 1e6), site="vault")}
F_RANGE = {"room": (0.1, 1.0), "vault": (0.3, 1.0)}; WK = 2.0
Z = {k: rng.standard_normal(N) for k in ["d01", "d12", "d23", "d34"]}
U_a = rng.uniform(np.log(0.2), np.log(0.7), N); U_find_p = rng.uniform(0.3, 0.9, N)
U_find = {k: rng.uniform(size=N) for k in ["primer", "corpus"]}
U_claim = {m: rng.uniform(size=N) for m in MEDIA}; U_f = {m: rng.uniform(size=N) for m in MEDIA}; U_surv = {m: rng.uniform(size=N) for m in MEDIA}
U01 = (U_a - np.log(0.2)) / (np.log(0.7) - np.log(0.2)); ND = statistics.NormalDist()
ZA = np.array([ND.inv_cdf(min(max(u, 1e-12), 1 - 1e-12)) for u in U01])
def lifetime(m):
    lo, hi = MEDIA[m]["claim"]; flo, fhi = F_RANGE[MEDIA[m]["site"]]
    claim = np.exp(np.log(lo) + U_claim[m] * (np.log(hi) - np.log(lo))); f = np.exp(np.log(flo) + U_f[m] * (np.log(fhi) - np.log(flo)))
    return claim * f / np.log(2) ** (1 / WK) * (-np.log(U_surv[m])) ** (1 / WK)
LIFE = {m: lifetime(m) for m in MEDIA}
N_K = {1: math.log2(SER["GBR"][1][SER["GBR"][0] == 1620][0] / SER["GBR"][1][SER["GBR"][0] == 1290][0]),
       2: math.log2(SER["GBR"][1][SER["GBR"][0] == 1948][0] / SER["GBR"][1][SER["GBR"][0] == 1620][0]),
       3: math.log2(SER["GBR"][1][SER["GBR"][0] == 2012][0] / SER["GBR"][1][SER["GBR"][0] == 1948][0])}
def tier_times(m23, primer_found=None, step=None, d12_med=30.0):
    steps = [20 * np.exp(0.7 * Z["d01"]), d12_med * np.exp(0.7 * Z["d12"]), m23 * np.exp(0.8 * Z["d23"]), 50 * np.exp(0.8 * Z["d34"])]
    T = np.zeros((5, N))
    for k, d in enumerate(steps):
        dd = np.where(primer_found, step(k, d), d) if (primer_found is not None and k >= 1) else d
        T[k + 1] = T[k] + dd
    return T
def run_cell(m23, step=None, d12_med=30.0):
    if step is not None:
        pf = (U_find["primer"] < U_find_p) & (LIFE["MOM_ceramic"] > tier_times(m23, d12_med=d12_med)[1]); T = tier_times(m23, pf, step, d12_med)
    else: T = tier_times(m23, d12_med=d12_med)
    cf = U_find["corpus"] < U_find_p; out = {"median_T3": float(np.median(T[3])), "P_survive": {}, "P_read": {}}
    for m, sp in MEDIA.items():
        tk = T[sp["tier"]]; alive = LIFE[m] > tk
        out["P_survive"][m] = float(np.mean(alive)); out["P_read"][m] = float(np.mean(alive & (tk <= H) & cf))
    return out
D23_GRID = [50, 100, 200, 300, 400, 600, 800, 1200, 1600]
def crit(vals):
    xs = np.array(D23_GRID, float); ys = np.array(vals)
    if ys[0] < 0.5: return "<%d" % D23_GRID[0]
    if ys[-1] >= 0.5: return ">%d" % D23_GRID[-1]
    i = int(np.where(ys < 0.5)[0][0])
    return float(np.exp(np.log(xs[i-1]) + (0.5 - ys[i-1]) * (np.log(xs[i]) - np.log(xs[i-1])) / (ys[i] - ys[i-1])))
fitG = R013["gdp_lognormal_fit"]
def ratio_step(a): return lambda k, d: a * d
def hybrid_step(alpha, beta, gamma, s, e):
    def f(k, d):
        n = N_K[k]; dur = n * np.exp(alpha + gamma * (e - 1950) / 50 + s * ZA) * (d / n) ** beta
        return np.minimum(dur, d)
    return f
A1 = AFT["D1_UK_NL_US"]
STRUCT = {"ratio_012_logU": ratio_step(np.exp(U_a)),
          "ratio_013_GDP_lognormal": ratio_step(np.minimum(np.exp(fitG["mu"] + fitG["sigma"] * ZA), 1.0)),
          "hybrid_AFT_D1_eref": hybrid_step(A1["alpha"], A1["beta"], A1["gamma"], A1["s"], e_ref),
          "hybrid_AFT_D1_e1850": hybrid_step(A1["alpha"], A1["beta"], A1["gamma"], A1["s"], 1850.0),
          "hybrid_AFT_D1_e1980": hybrid_step(A1["alpha"], A1["beta"], A1["gamma"], A1["s"], 1980.0),
          "hybrid_AFT_D1_beta_boot_hi": hybrid_step(A1["alpha"], boot1["beta"][1], A1["gamma"], A1["s"], e_ref),
          "hybrid_AFT_D1_beta1": hybrid_step(A1["alpha"], 1.0, A1["gamma"], A1["s"], e_ref)}
REEVAL = {}; base = {x: run_cell(x) for x in D23_GRID}
for nm, st in STRUCT.items():
    g = {x: run_cell(x, st) for x in D23_GRID}
    REEVAL[nm] = {"D23_star_primer": {m: crit([g[x]["P_survive"][m] for x in D23_GRID]) for m in MEDIA},
                  "synergy_P_read_m23_300": {m: g[300]["P_read"][m] - base[300]["P_read"][m] for m in MEDIA},
                  "synergy_Piql_by_m23": {str(x): g[x]["P_read"]["Piql_film_QR"] - base[x]["P_read"]["Piql_film_QR"] for x in D23_GRID},
                  "median_T3_m23_300": g[300]["median_T3"]}
D23_noprimer = {m: crit([base[x]["P_survive"][m] for x in D23_GRID]) for m in MEDIA}
same = lambda a, b: (a == b) if isinstance(a, str) or isinstance(b, str) else abs(a - b) < 1e-9
r13 = R013["reeval_012"]
check_ladder = {"no_primer": all(same(D23_noprimer[m], R013["D23_star_no_primer"][m]) for m in MEDIA),
                "ratio_012": all(same(REEVAL["ratio_012_logU"]["D23_star_primer"][m], r13["012_logU_0.2_0.7"]["D23_star_primer"][m]) and same(REEVAL["ratio_012_logU"]["synergy_P_read_m23_300"][m], r13["012_logU_0.2_0.7"]["synergy_P_read_m23_300"][m]) for m in MEDIA),
                "ratio_013_GDP": all(same(REEVAL["ratio_013_GDP_lognormal"]["D23_star_primer"][m], r13["GDP_lognormal"]["D23_star_primer"][m]) and same(REEVAL["ratio_013_GDP_lognormal"]["synergy_P_read_m23_300"][m], r13["GDP_lognormal"]["synergy_P_read_m23_300"][m]) for m in MEDIA)}
check_ladder["reproduced_exactly"] = bool(all(check_ladder.values()))
# d12=330 感度
D12_330 = {"no_primer_Piql": crit([run_cell(x, None, 330.0)["P_survive"]["Piql_film_QR"] for x in D23_GRID])}
for nm in ["ratio_013_GDP_lognormal", "hybrid_AFT_D1_eref"]:
    D12_330[nm] = crit([run_cell(x, STRUCT[nm], 330.0)["P_survive"]["Piql_film_QR"] for x in D23_GRID])

# 【探索的・事前登録外】n_k（技術段を何回の所得倍増とみなすか）を c 倍したときの Piql D23*・相乗効果（混合型、e_ref と 1850）
NK0 = dict(N_K); NK_SCAN = {}
for e_ in (e_ref, 1850.0):
    for cmul in (1, 2, 4, 8, 16):
        for k in N_K: N_K[k] = NK0[k] * cmul
        st = STRUCT["hybrid_AFT_D1_eref"] if e_ == e_ref else STRUCT["hybrid_AFT_D1_e1850"]
        g = {x: run_cell(x, st) for x in D23_GRID}
        NK_SCAN[f"e{int(e_)}_x{cmul}"] = {"Piql_D23_star_primer": crit([g[x]["P_survive"]["Piql_film_QR"] for x in D23_GRID]),
                                          "syn300_Piql": g[300]["P_read"]["Piql_film_QR"] - base[300]["P_read"]["Piql_film_QR"],
                                          "syn300_MDISC": g[300]["P_read"]["MDISC_BD"] - base[300]["P_read"]["MDISC_BD"]}
for k in N_K: N_K[k] = NK0[k]

# 混合型での段ごとの実効 a（参考）
eff_a = {}
for k, nm_ in [(1, "d12"), (2, "d23"), (3, "d34")]:
    d = [None, 30 * np.exp(0.7 * Z["d12"]), 300 * np.exp(0.8 * Z["d23"]), 50 * np.exp(0.8 * Z["d34"])][k]
    a_ = STRUCT["hybrid_AFT_D1_eref"](k, d) / d; eff_a[nm_] = {q: float(np.quantile(a_, q)) for q in (0.25, 0.5, 0.75)}

# ---------- 判定 ----------
def num(x): return x if isinstance(x, float) else (1e9 if str(x).startswith(">") else 0.0)
Hy = {}
Hy["H26a"] = {"value": BAND["D1_UK_NL_US"].get("slope"), "se": BAND["D1_UK_NL_US"].get("se")}; Hy["H26a"]["supported"] = bool(Hy["H26a"]["value"] < 0.5)
sl = {k: v.get("slope") for k, v in BAND.items()}; Hy["H26b"] = {"value": sl, "n_below_0.5": int(sum(1 for v in sl.values() if v is not None and v < 0.5))}
Hy["H26b"]["supported"] = bool(Hy["H26b"]["n_below_0.5"] >= 3)
Hy["H26c"] = {"beta": A1["beta"], "boot95": boot1["beta"]}; Hy["H26c"]["supported"] = bool(boot1["beta"][1] < 1.0)
Hy["H26d"] = {"gamma": A1["gamma"], "boot95": boot1["gamma"]}; Hy["H26d"]["supported"] = bool(A1["gamma"] < 0 and boot1["gamma"][1] < 0)
hP = REEVAL["hybrid_AFT_D1_eref"]
Hy["H26e"] = {"value": hP["D23_star_primer"]["Piql_film_QR"]}; Hy["H26e"]["supported"] = bool(num(hP["D23_star_primer"]["Piql_film_QR"]) > 809)
Hy["H26f"] = {"value": hP["synergy_P_read_m23_300"]["Piql_film_QR"]}; Hy["H26f"]["supported"] = bool(hP["synergy_P_read_m23_300"]["Piql_film_QR"] > 0.096)

results = {"seed": SEED, "seed_boot": SEED_BOOT, "N": N, "L_grid": L_GRID, "FL_min": 1250, "check_vs_013_gdp": check013, "check_vs_013_ladder": check_ladder,
           "band_slopes": BAND, "aft": AFT, "aft_with_logL_D1": AFT_L, "LR_fixed_beta_D1": LR, "e_ref_median_entry_year": e_ref,
           "n_doublings_GBR": N_K, "reeval": REEVAL, "D23_star_no_primer": D23_noprimer, "d12_330": D12_330, "hybrid_effective_a": eff_a, "EXPLORATORY_nk_multiplier_scan": NK_SCAN, "hypotheses": Hy,
           "labels": "GDP 倍増の追いつきを primer の回復短縮の代理とする対応は【推測】。n_k（英国の倍増回数）で技術段を所得倍増に換算するのも【推測】。"}

# ---------- 図 ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8)); cols = {"D1_UK_NL_US": "k", "D2_UK_only": "tab:blue", "D3_world_max": "tab:green", "D4_D1_smoothed31": "tab:orange"}
for nm, b in BAND.items():
    dd = [r for r in b["bands"] if r["km_median_dur"] is not None]
    x = np.array([r["P"] for r in dd]); y = np.array([r["km_median_dur"] for r in dd])
    ax[0].scatter(x, y, color=cols[nm], s=18, label=f"{nm}: slope={b['slope']:.2f}±{b['se']:.2f} ({len(dd)} bands)")
    xx = np.linspace(np.log(x.min()), np.log(x.max()), 20); ax[0].plot(np.exp(xx), np.exp(np.log(y).mean() + b["slope"] * (xx - np.log(x).mean())), color=cols[nm], lw=0.8)
xr = np.array([25, 600]); ax[0].plot(xr, xr * 0.584, "r--", lw=0.8, label="ratio model a=0.584 (slope 1)")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("pioneer frontier years to double (band L->2L)"); ax[0].set_ylabel("follower KM median years to double")
ax[0].set_title("ratio or speed cap? bands with pioneer entry >= 1250"); ax[0].legend(fontsize=6.3)
bd = np.array(bootdraws1["beta"]); ax[1].hist(bd, bins=30, color="gray"); ax[1].axvline(A1["beta"], color="k"); ax[1].axvline(1, color="r", ls="--"); ax[1].axvline(0.5, color="b", ls=":")
ax[1].set_xlabel("beta (AFT, D1, era-controlled), cluster bootstrap"); ax[1].set_title(f"beta={A1['beta']:.2f} [{boot1['beta'][0]:.2f},{boot1['beta'][1]:.2f}], gamma/50y={A1['gamma']:.2f}")
plt.tight_layout(); plt.savefig(HERE + "/fig_ratio_vs_speedcap.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4)); names = list(STRUCT); vals = [num(REEVAL[n]["D23_star_primer"]["Piql_film_QR"]) for n in names]
ax[0].barh(range(len(names)), [min(v, 1700) for v in vals], color="slategray"); ax[0].axvline(num(D23_noprimer["Piql_film_QR"]), color="r", ls="--", label="no primer")
for i, v in enumerate(vals): ax[0].annotate(f"{v:.0f}" if v < 1e8 else ">1600", (min(v, 1700), i), fontsize=7, xytext=(3, -3), textcoords="offset points")
ax[0].set_yticks(range(len(names))); ax[0].set_yticklabels(names, fontsize=7); ax[0].set_xlabel("Piql D23* with primer (years)"); ax[0].legend(fontsize=7); ax[0].set_title("Piql critical m23 by primer structure")
for n in names: ax[1].plot(D23_GRID, [REEVAL[n]["synergy_Piql_by_m23"][str(x)] for x in D23_GRID], "-o", ms=3, label=n)
ax[1].set_xscale("log"); ax[1].set_xlabel("m23 (years, no primer)"); ax[1].set_ylabel("Piql primer synergy in P_read"); ax[1].legend(fontsize=6); ax[1].set_title("synergy vs m23")
plt.tight_layout(); plt.savefig(HERE + "/fig_ladder_structures.png", dpi=130); plt.close()

results["runtime_seconds"] = time.time() - t0
json.dump(results, open(HERE + "/results.json", "w"), ensure_ascii=False, indent=1, default=str)
print("check013:", check013); print("ladder check:", check_ladder)
for nm, b in BAND.items(): print(f"[band {nm}] recs={b['n_records']} ctry={b['n_countries']} bands={b['n_bands_used']} slope={b.get('slope'):.3f}±{b.get('se'):.3f} P={b.get('P_range')} fol={b.get('follower_median_range')}")
print("D1 bands:", [(r["L"], round(r["FL"]), round(r["P"]), r["n"], r["n_events"], r["km_median_dur"] and round(r["km_median_dur"], 1)) for r in b1])
for nm, a in AFT.items(): print(f"[AFT {nm}]", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in a.items() if k != "boot95"})
print("boot95 D1:", boot1); print("AFT+logL:", AFT_L); print("LR fixed beta:", LR, "e_ref:", e_ref, "N_K:", N_K)
print("D23 no primer:", D23_noprimer)
for n, r in REEVAL.items(): print(f"[{n}] Piql D23*={r['D23_star_primer']['Piql_film_QR']} DNA={r['D23_star_primer']['DNA_silica']} syn300 Piql={r['synergy_P_read_m23_300']['Piql_film_QR']:.4f} MDISC={r['synergy_P_read_m23_300']['MDISC_BD']:.4f} medT3={r['median_T3_m23_300']:.1f}")
print("d12=330:", D12_330); print("NK scan:", json.dumps(NK_SCAN, default=str)); print("eff a:", eff_a)
print("hypotheses:", json.dumps({k: (v["supported"], {kk: vv for kk, vv in v.items() if kk != "supported"}) for k, v in Hy.items()}, default=str))
print("runtime_sec=%.1f" % results["runtime_seconds"])
