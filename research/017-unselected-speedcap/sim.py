"""研究017：選択なしの速度上限と離陸失敗（PREREGISTRATION.md）
(1) 暦の開始年グリッドにいる全国から「次の倍増までの年数」を数える選択なし記録で、014 と同じ AFT を推定し直す。
(2) 014 型（初回の上向き交差から数える）記録と比べ、選択の倍率 R(e) を出す。
(3) 倍増より先に 0.75 倍以下へ後退する確率（Aalen–Johansen）を両設計で比べる。
(4) 016 と同じ乱数の混合型梯子（015 の人口×所得 n_k）で Piql の D23* を再評価（015 を厳密再現して検証）。
seed 固定、numpy + matplotlib のみ。
"""
import csv, json, time, math, statistics, collections
import numpy as np
t0_ = time.time()
SEED = 20261015; SEED_BOOT = 20261017
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
SRC = HERE + "/../013-empirical-catchup-a/sources/owid_gdppc_maddison.csv"
R014 = json.load(open(HERE + "/../014-ratio-vs-speedcap/results.json"))
R015 = json.load(open(HERE + "/../015-nk-calibration/results.json"))
R016 = json.load(open(HERE + "/../016-isolated-speedcap/results.json"))

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
def frontier(pioneers):
    env = np.full(len(YEARS), -np.inf)
    for c in pioneers:
        yy, v, gap = SER[c]; m = (YEARS >= yy[0]) & (YEARS <= yy[-1]); env[m] = np.maximum(env[m], np.interp(YEARS[m], yy, v))
    return env
def frontier_cross(env, L):
    k = np.where(env >= L)[0]; return float(YEARS[k[0]]) if len(k) else math.inf
def first_cross_after(yy, v, i, L, up=True):
    """index i から後で初めて v>=L（up）/ v<=L（down）となる端数年。なければ inf。"""
    seg = v[i + 1:]; k = np.where(seg >= L)[0] if up else np.where(seg <= L)[0]
    if len(k) == 0: return math.inf
    j = i + 1 + k[0]; f = (math.log(L) - math.log(v[j-1])) / (math.log(v[j]) - math.log(v[j-1]))
    return float(yy[j-1] + f)

P_PRIMARY = ["GBR", "NLD", "USA"]; env1 = frontier(P_PRIMARY)
REV = 0.75

# ---------- (1) 選択なし記録 ----------
def unselected(grid, max_gap=10, obs_only=False, t0_min=None, first_only=False, exclude_oil=True):
    recs = []
    for c in sorted(SER):
        if c in AGG or c in P_PRIMARY or (exclude_oil and c in OIL): continue
        yy, v, gap = SER[c]; got = False
        for t0 in grid:
            if t0_min is not None and t0 < t0_min: continue
            if t0 < yy[0] or t0 > yy[-1]: continue
            i = int(t0 - yy[0])
            if gap[i] > max_gap or (obs_only and gap[i] != 0): continue
            y0 = float(v[i]); FL, F2 = frontier_cross(env1, y0), frontier_cross(env1, 2 * y0)
            if not (F2 <= t0 and FL >= 1250): continue
            c2 = first_cross_after(yy, v, i, 2 * y0); cr = first_cross_after(yy, v, i, REV * y0, up=False)
            if c2 == math.inf: dur, ev = float(yy[-1] - t0), 0
            else: dur, ev = c2 - t0, 1
            if ev == 0 and dur < 0.5: continue
            # 競合リスク：1=倍増、2=後退（倍増より先に 0.75y0 以下）、0=打ち切り
            if cr < (c2 if ev else yy[-1] + 1): cr_t, cr_ev = cr - t0, 2
            else: cr_t, cr_ev = dur, ev
            if first_only and got: continue
            recs.append({"code": c, "t0": float(t0), "y0": y0, "P": F2 - FL, "dur": float(dur), "event": ev, "cr_t": float(cr_t), "cr_ev": cr_ev})
            got = True
    return recs

# ---------- (2) 014 型の記録（014 sim.py と同じ定義）＋ 後退の競合リスク ----------
L_GRID = [round(1000 * 2 ** (k / 4)) for k in range(0, 25)]
def records014():
    recs = []
    for L in L_GRID:
        FL, F2L = frontier_cross(env1, L), frontier_cross(env1, 2 * L)
        if not (FL >= 1250 and F2L <= END): continue
        P = F2L - FL
        for c, (yy, v, gap) in SER.items():
            if c in AGG or c in P_PRIMARY or c in OIL: continue
            if v[0] >= L: continue
            k = np.where(v >= L)[0]
            if len(k) == 0: continue
            i = k[0]; f = (math.log(L) - math.log(v[i-1])) / (math.log(v[i]) - math.log(v[i-1])); cL = float(yy[i-1] + f)
            if cL < F2L: continue
            c2 = first_cross_after(yy, v, i - 1, 2 * L); cr = first_cross_after(yy, v, i, REV * L, up=False)
            if c2 == math.inf: dur, ev = float(yy[-1] - cL), 0
            else: dur, ev = c2 - cL, 1
            if ev == 0 and dur < 0.5: continue
            if cr < (c2 if ev else yy[-1] + 1): cr_t, cr_ev = cr - cL, 2
            else: cr_t, cr_ev = dur, ev
            recs.append({"code": c, "t0": cL, "y0": float(L), "P": P, "dur": float(dur), "event": ev, "cr_t": float(cr_t), "cr_ev": cr_ev})
    return recs

# ---------- AFT（014 と同じ実装） ----------
SQ2 = math.sqrt(2.0); _erfc = np.frompyfunc(math.erfc, 1, 1)
def log_sf(z):
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
            xe = c + 2 * (c - pts[-1]); fe = f(xe); pts[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fs[-2]: pts[-1], fs[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc_ = f(xc)
            if fc_ < fs[-1]: pts[-1], fs[-1] = xc, fc_
            else: pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]; fs = [fs[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(fs)); return pts[i], fs[i]
def design(recs):
    return (np.vstack([np.ones(len(recs)), np.log([r["P"] for r in recs]), (np.array([r["t0"] for r in recs]) - 1950) / 50]).T,
            np.log([r["dur"] for r in recs]), np.array([r["event"] for r in recs]))
def fit_aft(recs, x0=None):
    X, y, ev = design(recs); f = make_nll(X, y, ev)
    if x0 is None: b0 = np.linalg.lstsq(X, y, rcond=None)[0]; x0 = list(b0) + [math.log(max(np.std(y), 0.1))]
    th, v = nelder_mead(f, x0); th, v = nelder_mead(f, th, step=0.05)
    return {"alpha": float(th[0]), "beta": float(th[1]), "gamma": float(th[2]), "s": float(math.exp(th[3])), "nll": v, "n": len(recs),
            "n_events": int(ev.sum()), "n_countries": len({r["code"] for r in recs})}, th
A14 = R014["aft"]["D1_UK_NL_US"]; e_ref = R014["e_ref_median_entry_year"]
med = lambda a, e, P=118.0: math.exp(a["alpha"] + a["beta"] * math.log(P) + a["gamma"] * (e - 1950) / 50)
Rfun = lambda a, e: med(a, e) / med(A14, e)

GRID10 = list(range(1820, 2001, 10)); GRID25 = list(range(1825, 2001, 25))
U = unselected(GRID10); AU, thU = fit_aft(U)
R14 = records014(); A14re, th14 = fit_aft(R14)
check014 = {"n": len(R14), "n_014": A14["n"], "alpha_diff": abs(A14re["alpha"] - A14["alpha"]), "gamma_diff": abs(A14re["gamma"] - A14["gamma"])}
check014["reproduced"] = bool(check014["n"] == check014["n_014"] and check014["alpha_diff"] < 1e-3 and check014["gamma_diff"] < 1e-3)

rng_b = np.random.default_rng(SEED_BOOT); byc = collections.defaultdict(list)
for r in U: byc[r["code"]].append(r)
codes = sorted(byc); BD = collections.defaultdict(list)
for _ in range(200):
    pick = rng_b.choice(len(codes), len(codes), replace=True); rr = [r for k in pick for r in byc[codes[k]]]
    a, _ = fit_aft(rr, x0=list(thU))
    for k in ("alpha", "beta", "gamma", "s"): BD[k].append(a[k])
    BD["R_eref"].append(Rfun(a, e_ref)); BD["R_1850"].append(Rfun(a, 1850.0))
BOOT = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in BD.items()}

VAR = {"i_grid25": unselected(GRID25), "ii_obs_only": unselected(GRID10, obs_only=True),
       "iii_t0_ge_1950": unselected(GRID10, t0_min=1950), "iv_first_only": unselected(GRID10, first_only=True)}
VARFIT = {}
for k, rr in VAR.items():
    a, _ = fit_aft(rr); VARFIT[k] = {**a, "R_eref": Rfun(a, e_ref), "R_1850": Rfun(a, 1850.0)}
# 探索的（事前登録外）：産油国を含める・max_gap=30・後退閾値 0.5
EXPL = {}
a, _ = fit_aft(unselected(GRID10, exclude_oil=False)); EXPL["incl_oil"] = {**a, "R_eref": Rfun(a, e_ref)}
a, _ = fit_aft(unselected(GRID10, max_gap=30)); EXPL["max_gap30"] = {**a, "R_eref": Rfun(a, e_ref)}

# ---------- (3) 競合リスク（Aalen–Johansen） ----------
def aj_cif(recs, horizon=100.0):
    t = np.array([r["cr_t"] for r in recs]); e = np.array([r["cr_ev"] for r in recs]); o = np.argsort(t, kind="stable"); t, e = t[o], e[o]
    n = len(t); S = 1.0; C = {1: 0.0, 2: 0.0}; i = 0; curve = [(0.0, 0.0, 0.0)]
    while i < n and t[i] <= horizon:
        j = i; d = {1: 0, 2: 0}
        while j < n and t[j] == t[i]:
            if e[j] in d: d[e[j]] += 1
            j += 1
        at = n - i
        if d[1] + d[2] > 0:
            for k in (1, 2): C[k] += S * d[k] / at
            S *= 1 - (d[1] + d[2]) / at; curve.append((float(t[i]), C[1], C[2]))
        i = j
    return {"CIF_double_100": C[1], "CIF_reversal_100": C[2], "n": n, "n_rev_events": int(np.sum(e == 2))}, curve
CIF_U, curveU = aj_cif(U); CIF_14, curve14 = aj_cif(R14)
rb2 = np.random.default_rng(SEED_BOOT + 1); bU, b14 = [], []
by14 = collections.defaultdict(list)
for r in R14: by14[r["code"]].append(r)
c14 = sorted(by14)
for _ in range(500):
    bU.append(aj_cif([r for k in rb2.choice(len(codes), len(codes)) for r in byc[codes[k]]])[0]["CIF_reversal_100"])
    b14.append(aj_cif([r for k in rb2.choice(len(c14), len(c14)) for r in by14[c14[k]]])[0]["CIF_reversal_100"])
CIF_U["boot95"] = [float(np.percentile(bU, 2.5)), float(np.percentile(bU, 97.5))]; CIF_14["boot95"] = [float(np.percentile(b14, 2.5)), float(np.percentile(b14, 97.5))]
# 時代別（探索的）
CIF_ERA = {}
for lo, hi in [(1820, 1900), (1900, 1950), (1950, 2001)]:
    CIF_ERA[f"{lo}-{hi-1}"] = aj_cif([r for r in U if lo <= r["t0"] < hi])[0]

# ---------- (4) 梯子（015/016 と同じ乱数・同じ順序） ----------
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
def tier_times(m23, pf=None, step=None):
    steps = [20 * np.exp(0.7 * Z["d01"]), 30 * np.exp(0.7 * Z["d12"]), m23 * np.exp(0.8 * Z["d23"]), 50 * np.exp(0.8 * Z["d34"])]
    T = np.zeros((5, N))
    for k, d in enumerate(steps):
        dd = np.where(pf, step(k, d), d) if (pf is not None and k >= 1) else d
        T[k + 1] = T[k] + dd
    return T
def run_cell(m23, step=None):
    if step is not None:
        pf = (U_find["primer"] < U_find_p) & (LIFE["MOM_ceramic"] > tier_times(m23)[1]); T = tier_times(m23, pf, step)
    else: T = tier_times(m23)
    cf = U_find["corpus"] < U_find_p; out = {"P_survive": {}, "P_read": {}}
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
def num(x): return x if isinstance(x, float) else (1e9 if str(x).startswith(">") else 0.0)
NK_Y = {int(k): v for k, v in R015["n_k_pop_income"].items()}
def hybrid(a, e, nk=NK_Y):
    def f(k, d):
        n = nk[k]; return np.minimum(n * np.exp(a["alpha"] + a["gamma"] * (e - 1950) / 50 + a["s"] * ZA) * (d / n) ** a["beta"], d)
    return f
base = {x: run_cell(x) for x in D23_GRID}
def cell(a, e, p_takeoff=None):
    g = {x: run_cell(x, hybrid(a, e)) for x in D23_GRID}
    out = {"e": e, "Piql_D23": crit([g[x]["P_survive"]["Piql_film_QR"] for x in D23_GRID]), "MDISC_D23": crit([g[x]["P_survive"]["MDISC_BD"] for x in D23_GRID]),
           "syn300_Piql": g[300]["P_read"]["Piql_film_QR"] - base[300]["P_read"]["Piql_film_QR"]}
    return out
c15 = cell(A14, 1850.0); ref = R015["cells"]["pop_income_e1850"]
check015 = {"Piql_D23": abs(num(c15["Piql_D23"]) - num(ref["D23_star_primer"]["Piql_film_QR"])) < 1e-9,
            "syn300": abs(c15["syn300_Piql"] - ref["synergy_P_read_m23_300"]["Piql_film_QR"]) < 1e-9}
check015["reproduced_exactly"] = bool(all(check015.values()))
CELLS = {"A014_e1850": c15, "A014_eref": cell(A14, e_ref), "U_eref": cell(AU, e_ref), "U_1850": cell(AU, 1850.0)}
# 探索的：選択なし AFT で時代を走査
SCAN = {str(e): cell(AU, float(e))["Piql_D23"] for e in (1800, 1850, 1900, 1950, 2000)}
# 探索的：014 の上限で R(e_ref) に等しい等価時代
e_eq = e_ref + 50 * math.log(Rfun(AU, e_ref)) / A14["gamma"]
CELLS["A014_e_eq"] = cell(A14, e_eq)
# 探索的：s（ばらつき）だけ 014 に戻した選択なし AFT
AU_s14 = {**AU, "s": A14["s"]}; CELLS["U_eref_s014"] = cell(AU_s14, e_ref)
D23_nop = crit([base[x]["P_survive"]["Piql_film_QR"] for x in D23_GRID])

# ---------- 判定 ----------
R_main = Rfun(AU, e_ref)
Hy = {"H29a": {"R_eref": R_main, "boot95": BOOT["R_eref"], "supported": bool(R_main >= 1.5 and BOOT["R_eref"][0] > 1.2)},
      "H29b": {"gamma": AU["gamma"], "boot95": BOOT["gamma"], "supported": bool(AU["gamma"] < 0 and BOOT["gamma"][1] < 0)},
      "H29c": {"Piql_D23_U_eref": CELLS["U_eref"]["Piql_D23"], "supported": bool(num(CELLS["U_eref"]["Piql_D23"]) < 1600)},
      "H29d": {"variants": {k: v["R_eref"] for k, v in VARFIT.items()}, "n_same": int(sum((v["R_eref"] >= 1.5) == (R_main >= 1.5) for v in VARFIT.values()))},
      "H29e": {"CIF_reversal_100": CIF_U["CIF_reversal_100"], "boot95": CIF_U["boot95"], "supported": bool(CIF_U["CIF_reversal_100"] >= 0.10)},
      "H29f": {"CIF_014type": CIF_14["CIF_reversal_100"], "CIF_unselected": CIF_U["CIF_reversal_100"], "supported": bool(CIF_14["CIF_reversal_100"] < CIF_U["CIF_reversal_100"])}}
Hy["H29d"]["supported"] = bool(Hy["H29d"]["n_same"] >= 3)

# 探索的：時代別の KM 中央値（両設計）
def km(times, events):
    o = np.argsort(times, kind="stable"); t = np.asarray(times)[o]; e = np.asarray(events)[o]
    n = len(t); S = 1.0; ts, ss = [0.0], [1.0]; i = 0
    while i < n:
        j = i; d = 0
        while j < n and t[j] == t[i]: d += e[j]; j += 1
        if d > 0: S *= (1 - d / (n - i)); ts.append(float(t[i])); ss.append(S)
        i = j
    return np.array(ts), np.array(ss)
def km_med(rr):
    t, s = km([r["dur"] for r in rr], [r["event"] for r in rr]); k = np.where(1 - s >= 0.5)[0]; return float(t[k[0]]) if len(k) else None
ERA_KM = {}
for lo, hi in [(1820, 1900), (1900, 1950), (1950, 1980), (1980, 2001)]:
    ERA_KM[f"{lo}-{hi-1}"] = {"unselected": km_med([r for r in U if lo <= r["t0"] < hi]), "n_u": len([r for r in U if lo <= r["t0"] < hi]),
                              "type014": km_med([r for r in R14 if lo <= r["t0"] < hi]), "n_14": len([r for r in R14 if lo <= r["t0"] < hi])}

results = {"seed": SEED, "seed_boot": SEED_BOOT, "preregistration": "PREREGISTRATION.md", "reversal_threshold": REV,
           "check_014_records_refit": check014, "check_015": check015, "aft_014": A14, "aft_014_refit": A14re, "aft_unselected": AU, "aft_unselected_boot95": BOOT,
           "R_eref": R_main, "R_1850": Rfun(AU, 1850.0), "median_years_P118": {"014_eref": med(A14, e_ref), "U_eref": med(AU, e_ref), "014_1850": med(A14, 1850), "U_1850": med(AU, 1850)},
           "variants": VARFIT, "EXPLORATORY_variants": EXPL, "CIF_unselected": CIF_U, "CIF_014type": CIF_14, "EXPLORATORY_CIF_by_era": CIF_ERA,
           "EXPLORATORY_KM_median_by_era": ERA_KM, "ladder": CELLS, "D23_no_primer_Piql": D23_nop, "EXPLORATORY_era_scan_U_Piql_D23": SCAN, "e_eq_014": e_eq,
           "hypotheses": Hy, "labels": "GDP の倍増を崩壊後の技術段の回復の代理とするのは【推測】（014 以来）。選択なしの記録は「回復が始まったかどうか」を問わない平均的な速さ。"}

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
for lab, rr, col in [("unselected (calendar start)", U, "firebrick"), ("014-type (first upward crossing)", R14, "k")]:
    for (lo, hi), ls in zip([(1820, 1900), (1950, 2001)], ["--", "-"]):
        sub = [r for r in rr if lo <= r["t0"] < hi]
        if len(sub) < 10: continue
        t, s = km([r["dur"] for r in sub], [r["event"] for r in sub]); ax[0].step(t, 1 - s, where="post", color=col, ls=ls, label=f"{lab} {lo}-{hi-1} (n={len(sub)})")
ax[0].set_xlim(0, 150); ax[0].set_xlabel("years from start to income x2"); ax[0].set_ylabel("KM cumulative P(doubled)"); ax[0].legend(fontsize=6.5); ax[0].set_title("time to next doubling: selection matters")
ax[1].hist(BD["R_eref"], bins=30, color="gray"); ax[1].axvline(R_main, color="k"); ax[1].axvline(1.5, color="r", ls="--", label="H29a threshold 1.5")
ax[1].axvline(R016["EXPLORATORY_absolute"]["modern_open_over_cap_median"], color="b", ls=":", label="016 exploratory 1.70")
ax[1].set_xlabel("R(e_ref) = unselected median / 014 median (P=118)"); ax[1].legend(fontsize=7); ax[1].set_title(f"R={R_main:.2f} [{BOOT['R_eref'][0]:.2f},{BOOT['R_eref'][1]:.2f}], gamma_u={AU['gamma']:.2f}")
plt.tight_layout(); plt.savefig(HERE + "/fig_unselected_speedcap.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
for lab, cv, col in [("unselected", curveU, "firebrick"), ("014-type", curve14, "k")]:
    cv = np.array(cv); ax[0].step(cv[:, 0], cv[:, 2], where="post", color=col, label=f"{lab}: reversal to <=0.75 y0 first")
    ax[0].step(cv[:, 0], cv[:, 1], where="post", color=col, ls=":", label=f"{lab}: doubling first")
ax[0].set_xlim(0, 100); ax[0].set_xlabel("years"); ax[0].set_ylabel("Aalen-Johansen cumulative incidence"); ax[0].legend(fontsize=7); ax[0].set_title("takeoff failure proxy (competing risks)")
names = ["A014_eref", "A014_e1850", "A014_e_eq", "U_eref", "U_eref_s014", "U_1850"]; vals = [num(CELLS[n]["Piql_D23"]) for n in names]
ax[1].barh(range(len(names)), [min(v, 1700) for v in vals], color="slategray"); ax[1].axvline(num(D23_nop), color="r", ls="--", label="no primer")
for i, v in enumerate(vals): ax[1].annotate(f"{v:.0f}" if v < 1e8 else ">1600", (min(v, 1700), i), fontsize=7, xytext=(3, -3), textcoords="offset points")
ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names, fontsize=7); ax[1].set_xlabel("Piql D23* with primer, pop x income n_k (years)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(HERE + "/fig_takeoff_and_ladder.png", dpi=130); plt.close()


# ---------- 探索的（事前登録外）：時代依存の選択倍率、非離陸の裾の当てはまり ----------
R14_post = [r for r in R14 if r["t0"] >= 1950]; U_post = [r for r in U if r["t0"] >= 1950]
A14_post, _ = fit_aft(R14_post); AU_post = VARFIT["iii_t0_ge_1950"]
POST = {"aft_014type_t0ge1950": A14_post, "R_post_at": {str(e): med(AU_post, e) / med(A14_post, e) for e in (1960, 1975, 1990)}}
R14_pre = [r for r in R14 if r["t0"] < 1950]; U_pre = [r for r in U if r["t0"] < 1950]
A14_pre, _ = fit_aft(R14_pre); AU_pre, _ = fit_aft(U_pre)
POST["aft_014type_t0lt1950"] = A14_pre; POST["aft_U_t0lt1950"] = AU_pre
POST["R_pre_at"] = {str(e): med(AU_pre, e) / med(A14_pre, e) for e in (1850, 1900, 1930)}
ND_ = statistics.NormalDist()
def tail_check(rr, a, h=100.0):
    t, s = km([r["dur"] for r in rr], [r["event"] for r in rr]); k = np.where(t <= h)[0]; S_km = float(s[k[-1]])
    X, _, _ = design(rr); mu = X @ np.array([a["alpha"], a["beta"], a["gamma"]])
    S_mod = float(np.mean([1 - ND_.cdf((math.log(h) - m) / a["s"]) for m in mu]))
    return {"KM_S100": S_km, "model_mean_S100": S_mod, "n": len(rr)}
TAIL = {"unselected_all": tail_check(U, AU), "unselected_t0_le_1922": tail_check([r for r in U if r["t0"] <= 1922], AU),
        "type014_all": tail_check(R14, A14), "type014_t0_le_1922": tail_check([r for r in R14 if r["t0"] <= 1922], A14)}
results["EXPLORATORY_era_split"] = POST; results["EXPLORATORY_tail_S100"] = TAIL
print("POST", json.dumps(POST, default=str)); print("TAIL", TAIL)

results["runtime_seconds"] = time.time() - t0_
json.dump(results, open(HERE + "/results.json", "w"), ensure_ascii=False, indent=1, default=str)
print("check014", check014); print("check015", check015)
print("AFT014", A14); print("AFT014 refit", A14re); print("AFT U", AU); print("BOOT", BOOT)
print("R_eref", R_main, "R_1850", results["R_1850"], "med", results["median_years_P118"])
for k, v in VARFIT.items(): print("var", k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
for k, v in EXPL.items(): print("expl", k, v["R_eref"], v["n"])
print("CIF U", CIF_U); print("CIF 014", CIF_14); print("CIF era", CIF_ERA); print("KM era", ERA_KM)
print("ladder", json.dumps(CELLS, default=str)); print("scan", SCAN, "nop", D23_nop, "e_eq", e_eq)
print("H", json.dumps(Hy, default=str)); print("runtime", results["runtime_seconds"])
