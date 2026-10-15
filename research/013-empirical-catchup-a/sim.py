"""研究013：手引き（primer）による技術回復の短縮係数 a と m23 の経験的較正
(1) Maddison Project Database 2023（OWID CSV）で、追随国が「先頭国がすでに通過した所得倍増帯」を通過する年数の比 a を
    Kaplan–Meier（右打ち切りつき）で推定する。
(2) 技術個別の追いつき事例（Wikipedia で開いた日付）と Comin & Hobijn 2010 Table 2 の採用ラグ。
(3) 研究012の読取梯子モデルを同じ乱数で再実装し（012 の値を厳密再現して検証）、a の事前分布を経験分布に替えて再評価。
seed 固定。numpy + matplotlib のみ。決定論的に再現する。
"""
import csv, json, time, math, collections, statistics
import numpy as np
t0 = time.time()
SEED = 20261015            # 研究012 と同じ（012 の再現検証のため）
SEED_BOOT = 20261016       # ブートストラップ用
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
SRC = HERE + "/sources/owid_gdppc_maddison.csv"

# ======================= (1) Maddison：所得倍増帯の通過年数比 =======================
raw = collections.defaultdict(dict)
for r in csv.DictReader(open(SRC, encoding="utf-8")):
    if r["GDP per capita"] and r["Code"]:
        raw[r["Code"]][int(r["Year"])] = float(r["GDP per capita"])
AGG = {c for c in raw if c.startswith("OWID_")}          # World・旧国家（USSR 等）を除外
OIL = {"SAU","KWT","ARE","QAT","BHR","OMN","IRQ","IRN","LBY","DZA","AGO","NGA","GAB","GNQ","COG","VEN","ECU","TTO"}  # OPEC（現旧）＋湾岸・産油国
PIONEERS_PRIMARY = ["GBR", "NLD", "USA"]                 # 事前登録の主定義
PIONEERS_ALT = ["GBR", "USA"]                            # 感度：工業化の先頭のみ
YEARS = np.arange(1000, 2023)

def annual(code):
    """観測年の間を対数線形補間（外挿なし）。返り値：年配列、値配列、各年に最も近い観測までの距離（年）"""
    ys = np.array(sorted(raw[code])); vs = np.log(np.array([raw[code][y] for y in ys]))
    yy = np.arange(ys[0], ys[-1] + 1)
    lv = np.interp(yy, ys, vs)
    idx = np.searchsorted(ys, yy)
    gap = np.zeros(len(yy))
    for i, y in enumerate(yy):   # 観測区間の長さ（補間の粗さ）
        j = idx[i]
        gap[i] = 0 if (j < len(ys) and ys[j] == y) else (ys[j] - ys[j-1])
    return yy, np.exp(lv), gap

SER = {c: annual(c) for c in raw}

def frontier(pioneers):
    env = np.full(len(YEARS), -np.inf)
    for c in pioneers:
        yy, v, _ = SER[c]
        m = (YEARS >= yy[0]) & (YEARS <= yy[-1])
        env[m] = np.maximum(env[m], np.interp(YEARS[m], yy, v))
    return env

def cross_first(yy, v, L):
    """初めて L 以上になる年（前年との対数線形で小数年）。最初の観測がすでに L 以上なら None（入帯を観測できない）"""
    if v[0] >= L: return None
    k = np.where(v >= L)[0]
    if len(k) == 0: return math.inf
    i = k[0]; f = (math.log(L) - math.log(v[i-1])) / (math.log(v[i]) - math.log(v[i-1]))
    return float(yy[i-1] + f)

def cross_last(yy, v, L):
    """最後に下から L を越え、以後 L 未満に戻らない年"""
    if v[0] >= L: return None
    if v[-1] < L: return math.inf
    below = np.where(v < L)[0]; i = below[-1] + 1
    f = (math.log(L) - math.log(v[i-1])) / (math.log(v[i]) - math.log(v[i-1]))
    return float(yy[i-1] + f)

def frontier_cross(env, L):
    k = np.where(env >= L)[0]
    return float(YEARS[k[0]]) if len(k) else math.inf

L_GRID = [round(2000 * 2 ** (k / 4)) for k in range(0, 21)]   # 2000〜64000 の 1/4 オクターブ格子
END = 2022.0

def km(times, events):
    """Kaplan–Meier：生存関数 S(t)＝P(a > t)。返り値：段差の時刻と S"""
    o = np.argsort(times, kind="stable"); t = np.asarray(times)[o]; e = np.asarray(events)[o]
    n = len(t); S = 1.0; ts, ss = [0.0], [1.0]
    i = 0
    while i < n:
        j = i; d = 0
        while j < n and t[j] == t[i]:
            d += e[j]; j += 1
        at_risk = n - i
        if d > 0:
            S *= (1 - d / at_risk); ts.append(float(t[i])); ss.append(S)
        i = j
    return np.array(ts), np.array(ss)

def km_quantile(ts, ss, q):
    """F(t)=1−S(t) が q 以上になる最小の t。届かなければ None（打ち切りで未定）"""
    F = 1 - ss; k = np.where(F >= q - 1e-12)[0]
    return float(ts[k[0]]) if len(k) else None

def catchup_records(pioneers, crossing="first", exclude_oil=True, max_gap=None):
    env = frontier(pioneers); fc = cross_first if crossing == "first" else cross_last
    recs, bands = [], []
    for L in L_GRID:
        FL, F2L = frontier_cross(env, L), frontier_cross(env, 2 * L)
        if not (FL >= 1820 and F2L <= END): continue
        P = F2L - FL
        bands.append({"L": L, "frontier_year_L": FL, "frontier_year_2L": F2L, "pioneer_years": P})
        for c, (yy, v, gap) in SER.items():
            if c in AGG or c in pioneers or (exclude_oil and c in OIL): continue
            cL = fc(yy, v, L)
            if cL is None or cL == math.inf or cL < F2L: continue   # 入帯を観測できない／未入帯／先頭の完了前に入帯
            if max_gap is not None and gap[int(round(cL)) - yy[0]] > max_gap: continue
            c2 = fc(yy, v, 2 * L)
            if c2 is None: continue
            if c2 == math.inf:
                dur, ev = yy[-1] - cL, 0                        # 右打ち切り
            else:
                if max_gap is not None and gap[int(round(c2)) - yy[0]] > max_gap: continue
                dur, ev = c2 - cL, 1
            recs.append({"code": c, "L": L, "t_L": cL, "dur": float(dur), "event": ev, "a": float(dur / P),
                         "gap_at_L": float(gap[int(round(cL)) - yy[0]])})
    return recs, bands

def summarize(recs, bands, boot=True):
    a = np.array([r["a"] for r in recs]); e = np.array([r["event"] for r in recs])
    ts, ss = km(a, e)
    out = {"n_pairs": int(len(a)), "n_events": int(e.sum()), "n_countries": len({r["code"] for r in recs}),
           "km_q25": km_quantile(ts, ss, 0.25), "km_median": km_quantile(ts, ss, 0.5), "km_q75": km_quantile(ts, ss, 0.75),
           "km_q90": km_quantile(ts, ss, 0.9), "P_a_le_0.2": float(1 - np.interp(0.2, ts, ss, right=ss[-1]) if False else 1 - ss[np.searchsorted(ts, 0.2, side="right") - 1]),
           "P_a_le_0.7": float(1 - ss[np.searchsorted(ts, 0.7, side="right") - 1]),
           "P_a_le_1.0": float(1 - ss[np.searchsorted(ts, 1.0, side="right") - 1])}
    out["P_a_in_012_prior"] = out["P_a_le_0.7"] - out["P_a_le_0.2"]
    per_band = []
    for b in bands:
        rb = [r for r in recs if r["L"] == b["L"]]
        if len(rb) < 5: continue
        t2, s2 = km([r["a"] for r in rb], [r["event"] for r in rb])
        tdb, sdb = km([r["dur"] for r in rb], [r["event"] for r in rb])
        per_band.append({**b, "n": len(rb), "n_events": int(sum(r["event"] for r in rb)), "km_median_a": km_quantile(t2, s2, 0.5),
                         "km_median_dur": km_quantile(tdb, sdb, 0.5),
                         "median_dur_events": float(np.median([r["dur"] for r in rb if r["event"]])) if any(r["event"] for r in rb) else None})
    out["per_band"] = per_band
    if boot:   # 国単位のクラスター・ブートストラップ
        rngb = np.random.default_rng(SEED_BOOT); codes = sorted({r["code"] for r in recs})
        byc = collections.defaultdict(list)
        for r in recs: byc[r["code"]].append(r)
        qs = {"q25": [], "median": [], "q75": []}
        for _ in range(400):
            pick = rngb.choice(len(codes), len(codes), replace=True)
            rr = [r for k in pick for r in byc[codes[k]]]
            t3, s3 = km([r["a"] for r in rr], [r["event"] for r in rr])
            for nm, q in [("q25", .25), ("median", .5), ("q75", .75)]:
                v = km_quantile(t3, s3, q); qs[nm].append(np.nan if v is None else v)
        out["boot95"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in qs.items()}
    return out, (ts, ss)

def spearman(x, y):
    rk = lambda z: np.argsort(np.argsort(z)).astype(float)
    return float(np.corrcoef(rk(np.asarray(x)), rk(np.asarray(y)))[0, 1])

GDP = {}
variants = {"primary_first_UKNLUS_exoil": dict(pioneers=PIONEERS_PRIMARY, crossing="first", exclude_oil=True),
            "last_crossing": dict(pioneers=PIONEERS_PRIMARY, crossing="last", exclude_oil=True),
            "pioneer_UK_US": dict(pioneers=PIONEERS_ALT, crossing="first", exclude_oil=True),
            "include_oil": dict(pioneers=PIONEERS_PRIMARY, crossing="first", exclude_oil=False),
            "gap_le_10y": dict(pioneers=PIONEERS_PRIMARY, crossing="first", exclude_oil=True, max_gap=10)}
KMCURVE = {}
for nm, kw in variants.items():
    recs, bands = catchup_records(**kw)
    s, curve = summarize(recs, bands, boot=(nm in ("primary_first_UKNLUS_exoil", "pioneer_UK_US")))
    pb = s["per_band"]
    s["spearman_band_median_a_vs_frontier_year"] = spearman([b["frontier_year_L"] for b in pb], [b["km_median_a"] if b["km_median_a"] is not None else 9.9 for b in pb]) if len(pb) > 2 else None
    s["bands"] = bands
    dfn = [b for b in pb if b["km_median_a"] is not None]
    s["spearman_defined_bands_only"] = spearman([b["frontier_year_L"] for b in dfn], [b["km_median_a"] for b in dfn]) if len(dfn) > 2 else None
    dd = [b for b in pb if b["km_median_dur"] is not None]
    if len(dd) > 2:   # 【探索的】比モデル（傾き1）か速度上限モデル（傾き0）か：log(追随の中央年数) を log(先頭の年数) に回帰
        X = np.log([b["pioneer_years"] for b in dd]); Y = np.log([b["km_median_dur"] for b in dd]); Xc = X - X.mean()
        sl = float(Xc @ (Y - Y.mean()) / (Xc @ Xc)); rsd = Y - Y.mean() - sl * Xc
        s["EXPLORATORY_slope_logdur_vs_logpioneer"] = {"slope": sl, "se": float(math.sqrt(rsd @ rsd / (len(dd) - 2) / (Xc @ Xc))), "n_bands": len(dd)}
    s["max_interp_gap_at_entry"] = max(r["gap_at_L"] for r in recs)
    tD, sD = km([r["dur"] for r in recs], [r["event"] for r in recs])
    s["km_follower_doubling_years"] = {"q10": km_quantile(tD, sD, .1), "q25": km_quantile(tD, sD, .25), "median": km_quantile(tD, sD, .5), "q75": km_quantile(tD, sD, .75)}
    s["pioneer_years_by_band"] = [b["pioneer_years"] for b in bands]
    GDP[nm] = s; KMCURVE[nm] = curve
    if nm == "primary_first_UKNLUS_exoil": PRIMARY_RECS = recs
prim = GDP["primary_first_UKNLUS_exoil"]
# 例示：国×帯（主定義）の上位・下位
ex = sorted([r for r in PRIMARY_RECS if r["event"]], key=lambda r: r["a"])
prim["examples_fastest"] = ex[:8]; prim["examples_slowest_events"] = ex[-5:]
prim["selected"] = [r for r in PRIMARY_RECS if r["code"] in ("JPN", "KOR", "CHN", "TWN", "DEU", "ITA", "BRA", "IND", "SGP", "ESP")]

# 主定義の a の対数正規近似（KM 四分位から）
mu_gdp = math.log(prim["km_median"]); sig_gdp = (math.log(prim["km_q75"]) - math.log(prim["km_q25"])) / (2 * 0.6744897501960817)

# ======================= (2) 技術個別の事例（日付は sources/wiki/*.txt から） =======================
# 比 a = 追随者の所要年数 / 先駆者の所要年数。日付の不確かさは [lo, hi] で持つ（【推測】の部分は note に明記）。
def yr(y, m=7, d=1): return y + (m - 1) / 12 + (d - 1) / 365
CASES = [
 dict(name="Tu-4 (copy of B-29): program start -> first flight",
      pioneer=[yr(1938,1,1), yr(1942,9,21)], pioneer_alt_start=yr(1940,8,24),
      follower=[yr(1945,1,1), yr(1947,5,19)], follower_alt_start=yr(1945,7,1),
      note="B-29: Boeing began work 1938; XB-29 order 24 Aug 1940; first flight 21 Sep 1942. Tu-4: Stalin's copy order after B-29s interned in 1944 (crews left Jan 1945; exact order date not on page -> start 1945-01..1945-07 [推測]); first flight 19 May 1947."),
 dict(name="R-1 (copy of V-2/A-4): start -> first successful/test flight",
      pioneer=[yr(1936,7,1), yr(1942,10,3)], pioneer_alt_start=yr(1938,7,1),
      follower=[yr(1946,5,13), yr(1948,9,13)], follower_alt_start=yr(1947,4,1),
      note="A-4: larger rocket conceived summer 1936; design/construction ordered c.1938-39; first successful flight 3 Oct 1942. R-1: decree 13 May 1946; production authorized Apr 1947; first tests 13 Sep 1948."),
 dict(name="R-1 vs V-2: start -> entry into service",
      pioneer=[yr(1936,7,1), yr(1944,9,8)], pioneer_alt_start=yr(1938,7,1),
      follower=[yr(1946,5,13), yr(1950,11,28)], follower_alt_start=yr(1947,4,1),
      note="V-2 first operational launch 8 Sep 1944; R-1 entered service 28 Nov 1950."),
 dict(name="USSR bomb vs Manhattan: crash program -> first test",
      pioneer=[yr(1942,6,17), yr(1945,7,16)], pioneer_alt_start=yr(1939,10,1),
      follower=[yr(1945,8,15), yr(1949,8,29)], follower_alt_start=yr(1942,10,1),
      note="US: Roosevelt approved 17 Jun 1942 (alt: Briggs committee Oct 1939); Trinity 16 Jul 1945. USSR: accelerated after Aug 1945 (alt: authorized late 1942); RDS-1 29 Aug 1949."),
]
case_out = []
for c in CASES:
    P = c["pioneer"][1] - c["pioneer"][0]; F = c["follower"][1] - c["follower"][0]
    P2 = c["pioneer"][1] - c["pioneer_alt_start"]; F2 = c["follower"][1] - c["follower_alt_start"]
    rs = [F / P, F2 / P, F / P2, F2 / P2]
    case_out.append({"name": c["name"], "pioneer_years": P, "follower_years": F, "a_main": F / P, "a_range": [min(rs), max(rs)], "note": c["note"]})
# 知識だけの効果（同じ追随者で、諜報あり／なし）：Rhodes「a full two years」短縮（Wikipedia Soviet atomic bomb project）
F_with = yr(1949,8,29) - yr(1945,8,15); F_with_alt = yr(1949,8,29) - yr(1942,10,1)
knowledge_only = {"name": "USSR bomb: with espionage vs counterfactual without (Rhodes: accelerated by 'a full two years')",
                  "a_main": F_with / (F_with + 2.0), "a_range": sorted([F_with / (F_with + 2.0), F_with_alt / (F_with_alt + 2.0)])}
la = np.log([c["a_main"] for c in case_out] + [knowledge_only["a_main"]])
mu_case, sig_case = float(la.mean()), float(la.std(ddof=1))

# Comin & Hobijn 2010 Table 2（発明年、中央採用ラグ、平均ラグ）— sources/comin_hobijn_2010.txt 1139–1157 行
CH = [("Steam and motor ships",1788,123,120),("Railways-passengers",1825,101,97),("Railways-freight",1825,79,80),("Cars",1885,44,44),
      ("Trucks",1885,35,39),("Aviation-passengers",1903,29,34),("Aviation-freight",1903,42,43),("Telegraph",1835,33,46),("Telephone",1876,57,52),
      ("Cell phones",1973,16,15),("PCs",1973,14,14),("Internet users",1983,8,8),("MRIs",1977,5,5),("Blast oxygen steel",1950,16,16),("Electricity",1882,66,56)]
xs = np.array([c[1] for c in CH], float); ylog = np.log([c[2] for c in CH])
A = np.vstack([np.ones_like(xs), xs - 1900]).T; beta, res, *_ = np.linalg.lstsq(A, ylog, rcond=None)
resid = ylog - A @ beta; s2 = resid @ resid / (len(xs) - 2); se = math.sqrt(s2 * np.linalg.inv(A.T @ A)[1, 1])
CH_out = {"slope_log_lag_per_year": float(beta[1]), "se": se, "t": float(beta[1] / se), "halving_years": float(math.log(2) / -beta[1]) if beta[1] < 0 else None,
          "fitted_median_lag_1800": float(math.exp(beta[0] + beta[1] * (-100))), "fitted_median_lag_1950": float(math.exp(beta[0] + beta[1] * 50)),
          "spearman": spearman(xs, ylog), "overall_mean_lag": 45, "overall_median_lag": 32}

# 先駆者の階層所要年数（Wikipedia の年）
PIONEER_TIERS = {"T1_spectacles": 1290, "T2_compound_microscope": 1620, "T3_stored_program_computer": 1948, "T3_alt_point_contact_transistor": 1947}
pion = {"d12_pioneer": 1620 - 1290, "d23_pioneer": 1948 - 1620, "d12_prior_012_median": 30, "ratio_012_d12_to_pioneer": 30 / (1620 - 1290)}

# ======================= (3) 研究012 の再評価（同じ乱数・同じ手順） =======================
rng = np.random.default_rng(SEED); N = 200_000; H = 1000.0
MEDIA = {"MDISC_BD": dict(tier=3, claim=(100, 500), site="room"), "Piql_film_QR": dict(tier=3, claim=(500, 2000), site="vault"),
         "Silica_glass": dict(tier=3, claim=(1e4, 1e5), site="room"), "DNA_silica": dict(tier=4, claim=(2000, 1e4), site="room"),
         "Microfilm_AgX": dict(tier=1, claim=(300, 500), site="room"), "Rosetta_Ni": dict(tier=2, claim=(1e3, 1e4), site="room"),
         "MOM_ceramic": dict(tier=1, claim=(1e4, 1e6), site="vault")}
F_RANGE = {"room": (0.1, 1.0), "vault": (0.3, 1.0)}; WK = 2.0
Z = {k: rng.standard_normal(N) for k in ["d01", "d12", "d23", "d34"]}
U_a = rng.uniform(np.log(0.2), np.log(0.7), N)
U_find_p = rng.uniform(0.3, 0.9, N)
U_find = {k: rng.uniform(size=N) for k in ["primer", "corpus"]}
U_claim = {m: rng.uniform(size=N) for m in MEDIA}; U_f = {m: rng.uniform(size=N) for m in MEDIA}; U_surv = {m: rng.uniform(size=N) for m in MEDIA}
U01 = (U_a - np.log(0.2)) / (np.log(0.7) - np.log(0.2))      # a の共通一様乱数（分位点変換に使う）
NDIST = statistics.NormalDist()
ZA = np.array([NDIST.inv_cdf(min(max(u, 1e-12), 1 - 1e-12)) for u in U01])

def lifetime(m):
    lo, hi = MEDIA[m]["claim"]; flo, fhi = F_RANGE[MEDIA[m]["site"]]
    claim = np.exp(np.log(lo) + U_claim[m] * (np.log(hi) - np.log(lo))); f = np.exp(np.log(flo) + U_f[m] * (np.log(fhi) - np.log(flo)))
    med = claim * f; scale = med / np.log(2) ** (1 / WK)
    return scale * (-np.log(U_surv[m])) ** (1 / WK)
LIFE = {m: lifetime(m) for m in MEDIA}

def tier_times(m23, primer_found=None, a=None, d12_med=30.0, primer_tier=1):
    steps = [20 * np.exp(0.7 * Z["d01"]), d12_med * np.exp(0.7 * Z["d12"]), m23 * np.exp(0.8 * Z["d23"]), 50 * np.exp(0.8 * Z["d34"])]
    T = np.zeros((5, N))
    for k, d in enumerate(steps):
        mult = np.where(primer_found & (k >= primer_tier), a, 1.0) if primer_found is not None else 1.0
        T[k + 1] = T[k] + d * mult
    return T

def run_cell(m23, a=None, d12_med=30.0):
    p = U_find_p
    if a is not None:
        pf = U_find["primer"] < p
        pf &= LIFE["MOM_ceramic"] > tier_times(m23, d12_med=d12_med)[1]
        T = tier_times(m23, pf, a, d12_med)
    else:
        T = tier_times(m23, d12_med=d12_med)
    cf = U_find["corpus"] < p
    out = {"median_T3": float(np.median(T[3])), "E_T3_capped": float(np.mean(np.minimum(T[3], H))), "P_survive": {}, "P_read": {}}
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

def q_lognorm(mu, sig, power=1.0):
    return np.minimum(np.exp(power * (mu + sig * ZA)), 1.0)
def q_km(ts, ss):
    F = 1 - ss; idx = np.searchsorted(F, U01, side="left")       # F は非減少
    a = np.where(idx < len(ts), ts[np.minimum(idx, len(ts) - 1)], 1.0)
    return np.minimum(a, 1.0)

# 【探索的・事後追加（事前登録外）】速度上限型：先駆者の T2→T3（1620→1948）を英国の所得倍増回数に換算し、
# 追随国の1倍増あたり年数（KM、絶対年）を掛ける。a_rate = n_doub × D / 328（1 で上限）。
yyG, vG, _ = SER["GBR"]; g1620 = float(vG[yyG == 1620][0]); g1948 = float(vG[yyG == 1948][0]); n_doub = math.log2(g1948 / g1620)
recsP = PRIMARY_RECS; tD, sD = km([r["dur"] for r in recsP], [r["event"] for r in recsP])
FD = 1 - sD; idxD = np.searchsorted(FD, U01, side="left"); Dq = np.where(idxD < len(tD), tD[np.minimum(idxD, len(tD) - 1)], 1e9)
a_rate = np.minimum(n_doub * Dq / (1948 - 1620), 1.0)
RATE = {"UK_gdppc_1620": g1620, "UK_gdppc_1948": g1948, "n_doublings": n_doub, "follower_doubling_years_median": km_quantile(tD, sD, .5),
        "a_rate_median": float(np.median(a_rate)), "label": "exploratory, not pre-registered"}
PRIORS = {"012_logU_0.2_0.7": np.exp(U_a),
          "GDP_lognormal": q_lognorm(mu_gdp, sig_gdp),
          "GDP_KM_nonparam": q_km(*KMCURVE["primary_first_UKNLUS_exoil"]),
          "GDP_lognormal_knowledge_half": q_lognorm(mu_gdp, sig_gdp, 0.5),
          "cases_lognormal": q_lognorm(mu_case, sig_case),
          "EXPLORATORY_rate_limited": a_rate}
REEVAL = {}
base_grid = {m23: run_cell(m23) for m23 in D23_GRID}
for pn, a in PRIORS.items():
    g = {m23: run_cell(m23, a) for m23 in D23_GRID}
    REEVAL[pn] = {"a_quantiles": {q: float(np.quantile(a, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}, "a_mean": float(a.mean()),
                  "D23_star_primer": {m: crit([g[x]["P_survive"][m] for x in D23_GRID]) for m in MEDIA},
                  "synergy_P_read_m23_300": {m: g[300]["P_read"][m] - base_grid[300]["P_read"][m] for m in MEDIA},
                  "median_T3_m23_300": g[300]["median_T3"], "E_T3_reduction": {str(x): base_grid[x]["E_T3_capped"] - g[x]["E_T3_capped"] for x in D23_GRID},
                  "synergy_Piql_by_m23": {str(x): g[x]["P_read"]["Piql_film_QR"] - base_grid[x]["P_read"]["Piql_film_QR"] for x in D23_GRID},
                  "synergy_MDISC_by_m23": {str(x): g[x]["P_read"]["MDISC_BD"] - base_grid[x]["P_read"]["MDISC_BD"] for x in D23_GRID}}
D23_noprimer = {m: crit([base_grid[x]["P_survive"][m] for x in D23_GRID]) for m in MEDIA}
# 012 の値との照合（同じ乱数で再現できているか）
r012 = json.load(open(HERE + "/../012-archive-reading-ladder/results.json"))
check = {"D23_star_no_primer_012": {m: r012["D23_star"][m]["no_primer"] for m in MEDIA}, "D23_star_primer_012": {m: r012["D23_star"][m]["MOM_primer"] for m in MEDIA},
         "synergy_300_012": r012["grid"]["300"]["synergy_P_read"]}
def same(a_, b_): return (a_ == b_) if isinstance(a_, str) or isinstance(b_, str) else abs(a_ - b_) < 1e-9
check["reproduced_exactly"] = bool(all(same(D23_noprimer[m], check["D23_star_no_primer_012"][m]) and same(REEVAL["012_logU_0.2_0.7"]["D23_star_primer"][m], check["D23_star_primer_012"][m])
                                     and same(REEVAL["012_logU_0.2_0.7"]["synergy_P_read_m23_300"][m], check["synergy_300_012"][m]) for m in MEDIA))

# d12（T1→T2）の基線の感度：012 の 30年／100年／先駆者 330年
D12 = {}
for d12 in [30.0, 100.0, 330.0]:
    bg = {x: run_cell(x, None, d12) for x in D23_GRID}
    row = {"D23_star_no_primer": {m: crit([bg[x]["P_survive"][m] for x in ["x"] and D23_GRID]) for m in ["MDISC_BD", "Piql_film_QR", "DNA_silica"]}}
    for pn in ["012_logU_0.2_0.7", "GDP_lognormal"]:
        gp = {x: run_cell(x, PRIORS[pn], d12) for x in D23_GRID}
        row[pn] = {"D23_star_primer": {m: crit([gp[x]["P_survive"][m] for x in D23_GRID]) for m in ["MDISC_BD", "Piql_film_QR", "DNA_silica"]},
                   "synergy_Piql_300": gp[300]["P_read"]["Piql_film_QR"] - bg[300]["P_read"]["Piql_film_QR"]}
    D12[str(int(d12))] = row

# ======================= 判定 =======================
def num(x): return x if isinstance(x, float) else (1e9 if str(x).startswith(">") else 0.0)
Hy = {}
Hy["H25a"] = {"claim": "pooled KM median a (primary) < 0.5", "value": prim["km_median"], "boot95": prim["boot95"]["median"]}
Hy["H25a"]["supported"] = bool(prim["km_median"] < 0.5)
Hy["H25b"] = {"claim": "KM q25 >= 0.2 and q75 <= 0.7", "value": [prim["km_q25"], prim["km_q75"]]}
Hy["H25b"]["supported"] = bool(prim["km_q25"] >= 0.2 and prim["km_q75"] is not None and prim["km_q75"] <= 0.7)
Hy["H25c"] = {"claim": "Spearman(band median a, frontier year of L) < 0; aux: Comin-Hobijn slope < 0",
              "value": prim["spearman_band_median_a_vs_frontier_year"], "aux_CH_slope": CH_out["slope_log_lag_per_year"]}
Hy["H25c"]["supported"] = bool(prim["spearman_band_median_a_vs_frontier_year"] < 0)
Hy["H25c"]["aux_supported"] = bool(CH_out["slope_log_lag_per_year"] < 0)
p012 = REEVAL["012_logU_0.2_0.7"]; pg = REEVAL["GDP_lognormal"]
dPiql = num(pg["D23_star_primer"]["Piql_film_QR"]) / num(p012["D23_star_primer"]["Piql_film_QR"]) - 1
dSyn = pg["synergy_P_read_m23_300"]["Piql_film_QR"] - p012["synergy_P_read_m23_300"]["Piql_film_QR"]
Hy["H25d"] = {"claim": "|ΔD23*(Piql, primer)| <= 20% and |Δsynergy(Piql, m23=300)| <= 0.03 when prior -> GDP_lognormal",
              "value": {"D23_primer_012": p012["D23_star_primer"]["Piql_film_QR"], "D23_primer_GDP": pg["D23_star_primer"]["Piql_film_QR"], "rel_change": dPiql,
                        "synergy_012": p012["synergy_P_read_m23_300"]["Piql_film_QR"], "synergy_GDP": pg["synergy_P_read_m23_300"]["Piql_film_QR"], "diff": dSyn}}
Hy["H25d"]["supported"] = bool(abs(dPiql) <= 0.20 and abs(dSyn) <= 0.03)
Hy["H25e"] = {"claim": "012 d12 prior (30y) < 1/5 of pioneer T1->T2 (~330y) and pioneer T2->T3 (328y) in [200,600]", "value": pion}
Hy["H25e"]["supported"] = bool(pion["ratio_012_d12_to_pioneer"] < 0.2 and 200 <= pion["d23_pioneer"] <= 600)

results = {"seed": SEED, "seed_boot": SEED_BOOT, "N": N, "L_grid": L_GRID, "oil_excluded": sorted(OIL),
           "gdp_catchup": GDP, "gdp_lognormal_fit": {"mu": mu_gdp, "sigma": sig_gdp, "median": math.exp(mu_gdp)},
           "cases": case_out, "knowledge_only_case": knowledge_only, "case_lognormal_fit": {"mu": mu_case, "sigma": sig_case, "median": math.exp(mu_case)},
           "comin_hobijn": CH_out, "rate_limited_exploratory": RATE, "pioneer_tiers": PIONEER_TIERS, "pioneer_durations": pion,
           "reeval_012": REEVAL, "D23_star_no_primer": D23_noprimer, "check_vs_012": check, "d12_sensitivity": D12, "hypotheses": Hy,
           "labels": "a は【推測】による対応づけ（所得倍増帯の通過年数比を primer 短縮係数の代理とする）。a>1 は 1 に切る（primer が回復を遅らせない）【推測】。"}

# ======================= 図 =======================
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
for nm, sty in [("primary_first_UKNLUS_exoil", "k-"), ("last_crossing", "b--"), ("pioneer_UK_US", "g-."), ("include_oil", "r:"), ("gap_le_10y", "m-")]:
    ts, ss = KMCURVE[nm]; ax[0].step(ts, 1 - ss, sty, where="post", label=f"{nm} (n={GDP[nm]['n_pairs']}, med={GDP[nm]['km_median']:.2f})", lw=1.3 if nm.startswith("prim") else 0.9)
ax[0].axvspan(0.2, 0.7, color="orange", alpha=0.15, label="research 012 prior a in [0.2,0.7]")
ax[0].axvline(1, color="gray", lw=0.6); ax[0].set_xscale("log"); ax[0].set_xlim(0.03, 4)
ax[0].set_xlabel("a = follower years / pioneer years to double GDP per capita (same band)"); ax[0].set_ylabel("KM CDF  P(a' <= a)")
ax[0].set_title("Maddison 2023: catch-up ratio (right-censored at 2022)"); ax[0].legend(fontsize=6.5, loc="upper left")
pb = prim["per_band"]
ax[1].plot([b["frontier_year_L"] for b in pb], [b["km_median_a"] if b["km_median_a"] else np.nan for b in pb], "ko-", label="GDP band: KM median a (left axis)")
for b in pb: ax[1].annotate(f"{b['L']//1000}k", (b["frontier_year_L"], b["km_median_a"] or np.nan), fontsize=6)
ax[1].set_xlabel("year the pioneer frontier reached band floor L  /  invention year"); ax[1].set_ylabel("median catch-up ratio a"); ax[1].set_ylim(0, 1)
ax2 = ax[1].twinx(); ax2.scatter(xs, [c[2] for c in CH], c="tab:red", s=14, label="Comin-Hobijn median adoption lag (y, right)")
xx = np.linspace(1780, 1990, 50); ax2.plot(xx, np.exp(beta[0] + beta[1] * (xx - 1900)), "r--", lw=0.8); ax2.set_ylabel("adoption lag (years)", color="tab:red")
h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[1].legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
ax[1].set_title("later capability bands / technologies are caught up faster?")
plt.tight_layout(); plt.savefig(HERE + "/fig_catchup_ratio.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
cols = {"EXPLORATORY_rate_limited": "m", "012_logU_0.2_0.7": "orange", "GDP_lognormal": "k", "GDP_KM_nonparam": "gray", "GDP_lognormal_knowledge_half": "b", "cases_lognormal": "g"}
for pn, a in PRIORS.items():
    s_ = np.sort(a); ax[0].plot(s_, np.arange(1, N + 1) / N, color=cols[pn], label=pn)
for i, c in enumerate(case_out + [knowledge_only]):
    ax[0].errorbar(c["a_main"], 0.05 + 0.06 * i, xerr=[[c["a_main"] - c["a_range"][0]], [c["a_range"][1] - c["a_main"]]], fmt="g^", ms=4, lw=0.8)
    ax[0].annotate(c["name"].split(":")[0][:28], (c["a_main"], 0.05 + 0.06 * i), fontsize=5.5, xytext=(3, 2), textcoords="offset points")
ax[0].set_xscale("log"); ax[0].set_xlim(0.03, 1.6); ax[0].set_xlabel("a (step-duration multiplier when primer is found; capped at 1)"); ax[0].set_ylabel("CDF")
ax[0].set_title("a priors: research 012 vs empirical (triangles: tech cases)"); ax[0].legend(fontsize=6.5, loc="upper left")
for pn in PRIORS:
    ax[1].plot(D23_GRID, [REEVAL[pn]["synergy_Piql_by_m23"][str(x)] for x in D23_GRID], "-o", ms=3, color=cols[pn], label=f"Piql  {pn}")
    ax[1].plot(D23_GRID, [REEVAL[pn]["synergy_MDISC_by_m23"][str(x)] for x in D23_GRID], ":", color=cols[pn], lw=1)
ax[1].axvline(pion["d23_pioneer"], color="purple", lw=0.8, ls="--"); ax[1].annotate("pioneer T2->T3\n1620->1948", (pion["d23_pioneer"], 0.01), fontsize=7, color="purple")
ax[1].set_xscale("log"); ax[1].set_xlabel("m23 (median years T2->T3, no primer)"); ax[1].set_ylabel("primer synergy in P_read (solid Piql, dotted M-DISC)")
ax[1].set_title("research 012 re-evaluated under empirical a priors"); ax[1].legend(fontsize=6.5)
plt.tight_layout(); plt.savefig(HERE + "/fig_priors_and_reeval.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
pns = list(PRIORS); vals = [num(REEVAL[p]["D23_star_primer"]["Piql_film_QR"]) for p in pns]
ax[0].barh(range(len(pns)), vals, color=[cols[p] for p in pns]); ax[0].axvline(num(D23_noprimer["Piql_film_QR"]), color="r", ls="--", lw=1, label="no primer (435 y)")
ax[0].set_yticks(range(len(pns))); ax[0].set_yticklabels(pns, fontsize=7); ax[0].set_xlabel("Piql D23* with ceramic primer (years)")
for i, v in enumerate(vals): ax[0].annotate(f"{v:.0f}", (v, i), fontsize=7, xytext=(3, -3), textcoords="offset points")
ax[0].legend(fontsize=7); ax[0].set_title("critical m23 for Piql film by a prior")
d12s = list(D12); xx_ = np.arange(len(d12s))
ax[1].bar(xx_ - 0.27, [num(D12[k]["D23_star_no_primer"]["Piql_film_QR"]) for k in d12s], 0.27, color="r", label="no primer")
ax[1].bar(xx_, [num(D12[k]["012_logU_0.2_0.7"]["D23_star_primer"]["Piql_film_QR"]) for k in d12s], 0.27, color="orange", label="primer, 012 prior")
ax[1].bar(xx_ + 0.27, [num(D12[k]["GDP_lognormal"]["D23_star_primer"]["Piql_film_QR"]) for k in d12s], 0.27, color="k", label="primer, GDP prior")
ax[1].set_xticks(xx_); ax[1].set_xticklabels([f"d12 median {k} y" + (" (012)" if k == "30" else " (pioneer 1290->1620)" if k == "330" else "") for k in d12s], fontsize=7)
ax[1].set_ylabel("Piql D23* (years)"); ax[1].legend(fontsize=7); ax[1].set_title("T1->T2 baseline (d12) matters as much as the a prior")
plt.tight_layout(); plt.savefig(HERE + "/fig_D23_by_prior.png", dpi=130); plt.close()

results["runtime_seconds"] = time.time() - t0
json.dump(results, open(HERE + "/results.json", "w"), ensure_ascii=False, indent=1, default=str)
print("check_vs_012 reproduced_exactly:", check["reproduced_exactly"])
for nm, s in GDP.items():
    print(f"[GDP {nm}] pairs={s['n_pairs']} events={s['n_events']} countries={s['n_countries']} q25={s['km_q25']} med={s['km_median']} q75={s['km_q75']} q90={s['km_q90']} P(0.2<a<=0.7)={s['P_a_in_012_prior']:.3f} P(a<=1)={s['P_a_le_1.0']:.3f} rho={s['spearman_band_median_a_vs_frontier_year']}")
print("primary boot95:", prim.get("boot95")); print("per_band:", json.dumps([{k: b[k] for k in ("L", "frontier_year_L", "pioneer_years", "n", "n_events", "km_median_a")} for b in prim["per_band"]]))
print("spearman defined-only:", {k: v["spearman_defined_bands_only"] for k, v in GDP.items()}, "max gap:", {k: v["max_interp_gap_at_entry"] for k, v in GDP.items()})
print("follower doubling years KM:", {k: v["km_follower_doubling_years"] for k, v in GDP.items()}); print("pioneer years by band:", prim["pioneer_years_by_band"]); print("RATE:", RATE); print("slope ratio-vs-rate:", {k: v.get("EXPLORATORY_slope_logdur_vs_logpioneer") for k, v in GDP.items()})
print("band median durations:", [(b["L"], b["pioneer_years"], b["km_median_dur"]) for b in prim["per_band"]])
print("GDP LN fit mu=%.3f sigma=%.3f median=%.3f" % (mu_gdp, sig_gdp, math.exp(mu_gdp)))
print("cases:", json.dumps(case_out, indent=0)); print("knowledge_only:", knowledge_only); print("case LN: median=%.3f sigma=%.3f" % (math.exp(mu_case), sig_case))
print("Comin-Hobijn:", CH_out); print("pioneer:", pion)
print("D23* no primer:", D23_noprimer)
for pn, r in REEVAL.items():
    print(f"[{pn}] a q10/50/90={r['a_quantiles'][0.1]:.3f}/{r['a_quantiles'][0.5]:.3f}/{r['a_quantiles'][0.9]:.3f} D23*primer={ {m: (round(v) if isinstance(v, float) else v) for m, v in r['D23_star_primer'].items()} } syn300 Piql={r['synergy_P_read_m23_300']['Piql_film_QR']:.4f} MDISC={r['synergy_P_read_m23_300']['MDISC_BD']:.4f} Silica={r['synergy_P_read_m23_300']['Silica_glass']:.4f} medT3={r['median_T3_m23_300']:.1f}")
print("d12 sensitivity:", json.dumps(D12, default=str))
print("hypotheses:", json.dumps({k: (v["supported"], v.get("value")) for k, v in Hy.items()}, default=str, ensure_ascii=False))
print("runtime_sec=%.1f" % results["runtime_seconds"])
