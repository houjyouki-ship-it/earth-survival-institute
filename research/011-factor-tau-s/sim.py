"""011: 要因別 τ_s（open_problems 33）。戦争ショックと疫病ショックに別々のショック率の時間形 m_k(s) を与え、
研究008のラプラス汎関数（加法回復モデル）を多要因に拡張して S(1e4) と drop を厳密計算する。
再現: python3 sim.py > stdout.txt  (seed=20261014; 乱数は使わない決定論的計算) -> results.json, fig_*.png

データ（公開・取得済み、data/ 以下）:
  Marani et al. 2021 Zenodo V1.0 (doi:10.5281/zenodo.4626111): Epidemic16001945March2021.csv（論文の解析集合 1600–1945, 395件）,
    Epidemics_dataset_21_March_2021.xlsx（1500–2020; 1946–2019 の件数はここから）。
  Correlates of War Inter-State War v4.0 (Inter-StateWarData_v4.0.csv; 参加国単位 → WarNum ごとに StartYear1 最小値・BatDeath 合計)。
m_bio(s) の根拠: Marani の MEVD では疫病強度の分布は時間不変で、極端疫病の年確率 H1 は年間発生数 n に比例して動く
  （論文 Fig.2: H1(5.7‰/年) の変動は発生数の変動による）。よって「疫病ショック率の時間形」∝ 年間発生数とみなす（この写像は【推測】）。

モデル（研究008の加法回復モデルを要因 k ごとの独立ポアソン過程に拡張）:
  h(t) = B e^{-t/τ}[1 + Σ_k δ_k Σ_{j∈k} e^{-(t-s_kj)/r}] + h_nat,  λ_k(s) = (λ0_k/100) m_k(s),  m_k(0)=1
  ラプラス汎関数の積より  -log S(T) = B I0(T) + Σ_k ∫_0^T λ_k(s)[1-exp(-B δ_k C(s,T))] ds + h_nat T。
  τ=200, W=100, E[S(W)]=5/6 で B を較正（ショックごとに再較正）、T=1e4 で評価。drop = S_noshock(1e4) - S(1e4)（>0: ショックが S を下げる）。
  配分は総量を固定: Σ_k λ0_k = λ0_tot, Σ_k λ0_k δ_k = λ0_tot·δ̄。要因 k の「λ0δ 比」 w_k = λ0_k δ_k / Σ λ0 δ。

形 m_k(s):
  war_const: 1（τ_s=∞; Clauset 2018 の「定常と区別できない」）, war_200: e^{-s/200}（対照、004–007 の仮定）
  bio_decl: e^{-s/τ_bio}（過去減少の外挿; τ_bio は下の主推定）, bio_rise: 3-2e^{-s/50}（Marani「今後数十年で最大3倍」を形だけ。
    50年は【推測】）, bio_const: 1

事前に書いた仮説（本回の実行前に本ファイルへ記載。第三者のタイムスタンプはない）:
 主推定の定義（実行前に固定）: τ_bio = 1/(-ĝ)、ĝ は 1900–2019 の年次発生数への対数線形ポアソン回帰の最尤値（Marani の比較窓
   1900–1919 と 2000–2019 を含む期間）。ĝ≥0 なら H21 は Marani の2点（181→877年, 窓中心間100年）から τ を作って判定する。
 H20 戦争定常＋疫病上昇（war_const+bio_rise）では、r=20 の drop は調べた全配分（下記8配分）×λ0_tot∈{0.3,1,3}×δ̄∈{0.5,1,2,3.93,5}
     で正。
 H21 疫病のみ・過去減少形（bio_decl, τ_bio=主推定）では、r=20 の drop は負になりうる（≥1セルで負）。
     強い版 H21s: 上の λ0_tot×δ̄ 全15セルで負、かつ τ_bio を Marani 2点の95%帯（約51–84年）に変えても全セル負。
 H22 総 λ0·δ と λ0_tot を揃えた単一形の対照と比べて、混合の drop は r=20 で大きく、r=∞ で同程度以上（差 ≥ −1e-9）。
     (a) war_const+bio_rise 対 単一一定率（両要因の τ_s の「平均」: 戦争 ∞・疫病は上昇なので単一形は一定率とする）
     (b) war_const+bio_decl 対 単一指数 e^{-ḡ s}, ḡ = Σ_k w_k g_k（g_war=0, g_bio=1/τ_bio; λ0δ 比で平均した減衰率）
     全配分（all_war/all_bio を除く）× λ0_tot × δ̄ で判定。
 H23（批判的検証用）戦争定常＋疫病過去減少では、r=20 の drop の符号を反転させる戦争の λ0δ 比 w* が (0.4, 0.8) にあり、
     研究006の型配分（war 0.4, pandemic 0.3 を2要因に正規化、δ は 006 の倍率表と Ord 表6.1 から作る有効 δ）での戦争 λ0δ 比は
     w*±0.3 の帯に入る（＝この組み合わせでは符号が配分に依存し、政策上の結論にならない）。
     注: 実行前の手計算（1次近似, τ_bio=58 を仮定）で w*≈0.61 と見積もった。H23 はこの手計算の検証を兼ねる。
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261014
OUT = Path(__file__).resolve().parent
DATA = OUT / "data"
H_NAT = 1 / 87_000; P_W = 1 / 6; T_EVAL = 10_000.0; TAU = 200.0; W = 100.0
CHI2_95 = 3.841458820694124
t0 = time.time()
res = {"seed": SEED, "note": "deterministic; no RNG used", "tau": TAU, "W": W, "T_eval": T_EVAL, "h_nat": H_NAT}

# ============================ A. 疫病データ ============================
csv = pd.read_csv(DATA / "Epidemic16001945March2021.csv", header=None)
xl = pd.read_excel(DATA / "Epidemics_dataset_21_March_2021.xlsx")
xl["sy"] = pd.to_numeric(xl["Start Year"], errors="coerce")
xl = xl[xl["sy"].notna()].copy(); xl["sy"] = xl["sy"].astype(int)
post = xl[(xl.sy >= 1946) & (xl.sy <= 2019)]
YEARS = np.arange(1600, 2020)
n_year = np.zeros(len(YEARS), dtype=int)
for y in csv[0].astype(int): n_year[y - 1600] += 1
for y in post.sy: n_year[y - 1600] += 1
res["epi_data"] = {
    "csv_rows_1600_1945": int(len(csv)), "csv_deaths_missing_-999": int((csv[2] == -999).sum()),
    "xlsx_rows_1600_1945": int(((xl.sy >= 1600) & (xl.sy <= 1945)).sum()),
    "xlsx_rows_1946_2019": int(len(post)), "total_1600_2019": int(n_year.sum()),
    "max_per_year": int(n_year.max()),
    "note": "1600–1945 は論文の解析集合(CSV, 395件)、1946–2019 は xlsx の Start Year。deaths=-999 も年があれば件数に含む。",
}

def pois_ci(k, a=0.05):
    """正確（Garwood）95% CI。Poisson CDF の二分法。"""
    def cdf(kk, mu):
        if kk < 0: return 0.0
        return float(sum(math.exp(i * math.log(mu) - mu - math.lgamma(i + 1)) for i in range(kk + 1))) if mu > 0 else 1.0
    def solve(f, lo, hi):
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(mid) > 0: lo = mid
            else: hi = mid
        return (lo + hi) / 2
    lo = 0.0 if k == 0 else solve(lambda m: (1 - cdf(k - 1, m)) - a / 2 < 0 and 1 or -1, 1e-12, 10 * k + 10)
    hi = solve(lambda m: cdf(k, m) - a / 2 > 0 and 1 or -1, 1e-12, 10 * k + 50)
    return lo, hi

bins = {}
for b0 in range(1600, 2020, 20):
    k = int(n_year[b0 - 1600:b0 - 1600 + 20].sum()); lo, hi = pois_ci(k)
    bins[f"{b0}-{b0 + 19}"] = {"count": k, "rate_per_yr": k / 20, "ci95_per_yr": [lo / 20, hi / 20]}
res["epi_20yr_bins"] = bins
b1900 = bins["1900-1919"]["count"]; b2000 = bins["2000-2019"]["count"]

def ratio_ci(k1, k2):
    """2群ポアソン比 k1/k2（同じ露出）の尤度比 95% CI。"""
    def lr(rho):  # 条件付き二項 p=rho/(1+rho)
        p = rho / (1 + rho); n = k1 + k2; ph = k1 / n
        l = lambda q: k1 * math.log(q) + k2 * math.log(1 - q)
        return 2 * (l(ph) - l(p))
    est = k1 / k2
    def root(lo, hi):
        flo = lr(lo) - CHI2_95
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if (lr(mid) - CHI2_95) * flo > 0: lo = mid
            else: hi = mid
        return math.sqrt(lo * hi)
    return est, root(est / 1000, est), root(est, est * 1000)
rr = ratio_ci(b1900, b2000)
res["epi_ratio_1900s_over_2000s"] = {"est": rr[0], "ci95": [rr[1], rr[2]],
    "marani_T_ratio_877_over_181": 877 / 181, "marani_T_ratio_ci_extremes": [730 / 223, 1081 / 151]}

# MEVD 再構成（論文の GPD パラメータ: μ=0.001‰/yr, σ=0.0113, ξ=1.40, p0=0.62）
MU, SIG, XI, P0 = 1e-3, 0.0113, 1.40, 0.62
def H_gpd(i): return (1 - P0) * (1 + XI * (i - MU) / SIG) ** (-1 / XI)
Hs = H_gpd(5.7)
mevd = {}
for b0 in range(1600, 2020, 20):
    ny = n_year[b0 - 1600:b0 - 1600 + 20]
    H1 = float(np.mean(1 - (1 - Hs) ** ny))
    mevd[f"{b0}-{b0 + 19}"] = {"H1": H1, "T_years": (1 / H1) if H1 > 0 else None}
res["mevd_reconstruction"] = {"H_single_epidemic_5.7": Hs, "windows": mevd,
    "paper": {"T_1918_window": 181, "T_2000_2019": 877, "T_2000_2019_abstract_corrected": 879,
              "Py_range_1600_present_pct": [0.11, 0.89]}}

# 対数線形ポアソン回帰（プロファイル尤度）
def trend_fit(years, counts, center=None, g_test=(-1 / 200, -1 / 100, -1 / 50)):
    x = years - (center if center is not None else years.mean()); n = counts.astype(float); N = n.sum()
    def prof(g):
        z = g * x; m = z.max()
        return float(np.sum(n * z) - N * (m + math.log(np.sum(np.exp(z - m)))))
    lo, hi = -0.5, 0.5
    for _ in range(200):  # 凹関数の最大化（三分探索）
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if prof(a) < prof(b): lo = a
        else: hi = b
    gh = (lo + hi) / 2; lmax = prof(gh)
    def ci(scale):
        f = lambda g: 2 * (lmax - prof(g)) / scale - CHI2_95
        def root(a, b):
            fa = f(a)
            for _ in range(200):
                mid = (a + b) / 2
                if f(mid) * fa > 0: a = mid
                else: b = mid
            return (a + b) / 2
        return [root(gh - 0.5, gh), root(gh, gh + 0.5)]
    # 20年ビンでの過分散（Pearson χ²/df）
    ah = math.log(N / np.sum(np.exp(gh * x)))
    E = np.exp(ah + gh * x)
    nb = len(years) // 20
    O20 = n[:nb * 20].reshape(nb, 20).sum(1); E20 = E[:nb * 20].reshape(nb, 20).sum(1)
    phi = float(max(1.0, np.sum((O20 - E20) ** 2 / E20) / max(nb - 2, 1)))
    lr0 = 2 * (lmax - prof(0.0))
    tests = {f"g{g:.4f}": {"LR": 2 * (lmax - prof(g)), "p": math.erfc(math.sqrt(max(2 * (lmax - prof(g)), 0) / 2)),
                           "p_quasi": math.erfc(math.sqrt(max(2 * (lmax - prof(g)) / phi, 0) / 2))} for g in g_test}
    return {"tests_decline_tau_s_200_100_50": tests, "g_hat": gh, "ci95": ci(1.0), "phi_20yr_pearson": phi, "ci95_quasi": ci(phi),
            "tau_s_hat": (-1 / gh) if gh < 0 else None,
            "LR_vs_const": lr0, "p_vs_const": math.erfc(math.sqrt(max(lr0, 0) / 2)),
            "p_vs_const_quasi": math.erfc(math.sqrt(max(lr0 / phi, 0) / 2)), "n_events": int(N)}
fits = {}
for (a, b) in [(1600, 1945), (1600, 2019), (1900, 2019), (1920, 2019), (1946, 2019)]:
    sel = (YEARS >= a) & (YEARS <= b)
    fits[f"{a}-{b}"] = trend_fit(YEARS[sel].astype(float), n_year[sel])
res["epi_trend_fits"] = fits
for f in fits.values():
    f["tau_s_ci95"] = [(-1 / g) if g < 0 else None for g in f["ci95"][::-1]]
    f["tau_s_ci95_quasi"] = [(-1 / g) if g < 0 else None for g in f["ci95_quasi"][::-1]]
main = fits["1900-2019"]
if main["g_hat"] < 0:
    TAU_BIO = -1 / main["g_hat"]; tau_bio_src = "1900-2019 Poisson trend MLE"
else:
    TAU_BIO = 100 / math.log(877 / 181); tau_bio_src = "Marani two-point (fallback)"
res["tau_bio_main"] = {"value": TAU_BIO, "source": tau_bio_src}
res["tau_bio_marani_two_point"] = {"window_centres_1909.5_2009.5": 100 / math.log(877 / 181),
    "ci_band": [100 / math.log(1081 / 151), 100 / math.log(730 / 223)],
    "using_1918_as_time": 91.5 / math.log(877 / 181)}

# 大強度疫病（既知強度 ≥0.03‰/年: 論文の冪則域）の比率が時間で一定か（MEVD の前提の粗い確認）
xl["inten"] = pd.to_numeric(xl["Intensity (deaths per mil/year)"], errors="coerce")
xk = xl[(xl.sy >= 1600) & (xl.sy <= 2019)]
share = {}
for a, b in [(1600, 1699), (1700, 1799), (1800, 1899), (1900, 1945), (1946, 2019)]:
    s = xk[(xk.sy >= a) & (xk.sy <= b)]
    known = s[s["inten"] >= 0]  # -999 は除く（定義上 <10k 死者と推定、ここでは除外）
    big = int((known["inten"] >= 0.03).sum())
    share[f"{a}-{b}"] = {"all": int(len(s)), "known_intensity": int(len(known)), "ge_0.03": big,
                         "share_of_all": big / len(s) if len(s) else None}
res["epi_large_share_check"] = share

# ============================ B. 戦争データ（CoW 年次） ============================
war = pd.read_csv(DATA / "Inter-StateWarData_v4.0.csv")
war["bd"] = war.BatDeath.where(war.BatDeath >= 0, np.nan)
wg = war.groupby("WarNum").agg(start=("StartYear1", "min"), bd=("bd", "sum"), name=("WarName", "first"))
large = wg[wg.bd >= 26625]
WY = np.arange(1823, 2004)
nw_large = np.array([(large.start == y).sum() for y in WY]); nw_all = np.array([(wg.start == y).sum() for y in WY])
res["war_cow"] = {
    "n_wars": int(len(wg)), "n_large_ge_26625": int(len(large)),
    "large_1823_1939": int(((large.start >= 1823) & (large.start <= 1939)).sum()),
    "large_1914_1939": int(((large.start >= 1914) & (large.start <= 1939)).sum()),
    "large_1940_2003": int((large.start >= 1940).sum()),
    "clauset_2018_reported": {"n_wars": 95, "large_1823_1939": 19, "large_1914_1939": 10, "large_1940_2003": 5},
    "trend_large_annual": trend_fit(WY.astype(float), nw_large),
    "trend_all_annual": trend_fit(WY.astype(float), nw_all),
    "note": "BatDeath<0（欠測）は合計から除外。大規模の閾値は Clauset 2018 の 26,625。",
}
for k in ["trend_large_annual", "trend_all_annual"]:
    f = res["war_cow"][k]; f["tau_s_ci95"] = [(-1 / g) if g < 0 else None for g in f["ci95"][::-1]]

# ============================ C. 多要因ラプラス汎関数 ============================
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
def Cfun(s, T, r):
    k = 1 / TAU + (0 if math.isinf(r) else 1 / r)
    return np.exp(-s / TAU) * (-np.expm1(-k * (T - s))) / k
def shape(s, kind):
    """kind: ('exp', g) で e^{-g s}（g=0 一定）, ('rise', θ, M) で M-(M-1)e^{-s/θ}。"""
    if kind[0] == "exp": return np.exp(-kind[1] * s)
    if kind[0] == "rise": return kind[2] - (kind[2] - 1) * np.exp(-s / kind[1])
    raise ValueError(kind)
_PRE = {}
def pre(T, r, kind):
    key = (T, r, kind)
    if key not in _PRE:
        s, w = nodes(T); _PRE[key] = (w * shape(s, kind) / 100.0, Cfun(s, T, r))
    return _PRE[key]
def neglogS(B, T, r, factors):
    out = B * I0(T) + H_NAT * T
    for lam0, dl, kind in factors:
        if lam0 == 0 or dl == 0: continue
        wl, c = pre(T, r, kind)
        out += lam0 * float(np.sum(wl * (-np.expm1(-B * dl * c))))
    return out
def dneglogS(B, T, r, factors):
    out = I0(T)
    for lam0, dl, kind in factors:
        if lam0 == 0 or dl == 0: continue
        wl, c = pre(T, r, kind)
        out += lam0 * dl * float(np.sum(wl * c * np.exp(-B * dl * c)))
    return out
L_W = -math.log(1 - P_W)
def calibrate(r, factors):
    """f(B)=neglogS(B,W)-L は B に関して単調増加・凹。B=0 から Newton で左から単調収束。"""
    B = 0.0
    for _ in range(200):
        f = neglogS(B, W, r, factors) - L_W
        Bn = B - f / dneglogS(B, W, r, factors)
        if abs(Bn - B) <= 1e-15 * max(Bn, 1e-300): B = Bn; break
        B = Bn
    return B
def S_eval(r, factors):
    B = calibrate(r, factors); return math.exp(-neglogS(B, T_EVAL, r, factors)), B
S0 = math.exp(-((L_W - H_NAT * W) / I0(W)) * I0(T_EVAL) - H_NAT * T_EVAL)
res["S_noshock"] = S0
def drop(r, factors): return S0 - S_eval(r, factors)[0]

# 検算1: 008 の単一要因 drop（λ0=1, δ=2, r=20）
r008 = json.load(open(OUT.parent / "008-rstar-analytic" / "results.json"))
chk = {}
for ts, g in [("inf", 0.0), ("200.0", 1 / 200), ("100.0", 1 / 100)]:
    for rr_ in ["5", "20", "200", "2000"]:
        mine = drop(float(rr_), [(1.0, 2.0, ("exp", g))])
        chk[f"ts{ts}_r{rr_}"] = {"this": mine, "008": r008["drop_vs_r_compare"][ts][rr_], "diff": mine - r008["drop_vs_r_compare"][ts][rr_]}
res["check_vs_008"] = chk
res["check_vs_008_maxabs"] = max(abs(v["diff"]) for v in chk.values())
# 検算2: 同じ形・同じ δ の2要因 = 和の率の1要因（ポアソンの重ね合わせ）
a2 = drop(20.0, [(0.4, 2.0, ("exp", 0.0)), (0.6, 2.0, ("exp", 0.0))]); a1 = drop(20.0, [(1.0, 2.0, ("exp", 0.0))])
res["check_superposition"] = {"two_factor": a2, "one_factor": a1, "diff": a2 - a1}

# 006 の有効 δ（Ord 表6.1 の要因ハザードに型別倍率を重み付け）
ORD = {"unaligned_AI": 1 / 10, "engineered_pandemics": 1 / 30, "unforeseen_anthropogenic": 1 / 30,
       "other_anthropogenic": 1 / 50, "nuclear_war": 1 / 1000, "climate_change": 1 / 1000, "other_environmental": 1 / 1000}
MULT = {"war": {"nuclear_war": 1.0, "climate_change": 0.4, "other_environmental": 0.3, "unaligned_AI": 0.15,
                "engineered_pandemics": 0.15, "unforeseen_anthropogenic": 0.2, "other_anthropogenic": 0.2},
        "pandemic": {"engineered_pandemics": 1.0, "unaligned_AI": 0.25, "unforeseen_anthropogenic": 0.3, "nuclear_war": 0.1,
                     "climate_change": 0.1, "other_environmental": 0.1, "other_anthropogenic": 0.2},
        "tech_cascade": {"unaligned_AI": 1.0, "unforeseen_anthropogenic": 0.5, "other_anthropogenic": 0.4,
                         "engineered_pandemics": 0.2, "nuclear_war": 0.15, "climate_change": 0.1, "other_environmental": 0.1}}
PS = sum(ORD.values())
DEFF = {t: sum(m * ORD[k] for k, m in MULT[t].items()) / PS for t in MULT}
res["delta_eff_006"] = {"per_type_relative": DEFF, "ratio_bio_over_war": DEFF["pandemic"] / DEFF["war"],
    "note": "δ_type ∝ Σ_f M[type,f]·p_f/Σ p_f（Ord 表6.1, 研究006 の倍率表）。B を共通にした本モデルへの写像は【推測】。"}

# 配分: name -> list of (factor, rate share, relative δ)
ALLOC = {
    "all_war": [("war", 1.0, 1.0)],
    "all_bio": [("bio", 1.0, 1.0)],
    "bio_heavy_eqd": [("war", 0.25, 1.0), ("bio", 0.75, 1.0)],
    "equal_eqd": [("war", 0.5, 1.0), ("bio", 0.5, 1.0)],
    "r006_eqd": [("war", 0.4 / 0.7, 1.0), ("bio", 0.3 / 0.7, 1.0)],
    "r006_d006": [("war", 0.4 / 0.7, DEFF["war"]), ("bio", 0.3 / 0.7, DEFF["pandemic"])],
    "war_heavy_eqd": [("war", 0.75, 1.0), ("bio", 0.25, 1.0)],
    "r006_d006_tech200": [("war", 0.4, DEFF["war"]), ("bio", 0.3, DEFF["pandemic"]), ("tech", 0.3, DEFF["tech_cascade"])],
}
def war_share(alloc):
    tot = sum(f * d for _, f, d in ALLOC[alloc]); return sum(f * d for k, f, d in ALLOC[alloc] if k == "war") / tot
res["alloc_war_lambda_delta_share"] = {a: war_share(a) for a in ALLOC}
G_BIO = 1 / TAU_BIO
SHAPES = {"war_const": ("exp", 0.0), "war_200": ("exp", 1 / 200), "bio_decl": ("exp", G_BIO),
          "bio_rise": ("rise", 50.0, 3.0), "bio_const": ("exp", 0.0), "tech_200": ("exp", 1 / 200)}
SCEN = {"main_wconst_brise": ("war_const", "bio_rise"), "wconst_bdecl": ("war_const", "bio_decl"),
        "wconst_bconst": ("war_const", "bio_const"), "w200_brise": ("war_200", "bio_rise"),
        "w200_bdecl": ("war_200", "bio_decl")}
def factors_for(scen, alloc, lam_tot, dbar):
    ws, bs = SCEN[scen]; parts = ALLOC[alloc]
    norm = sum(f * d for _, f, d in parts)
    out = []
    for k, f, d in parts:
        kind = SHAPES[ws] if k == "war" else SHAPES[bs] if k == "bio" else SHAPES["tech_200"]
        out.append((lam_tot * f, dbar * d / norm, kind))
    return out
LAMS = [0.3, 1.0, 3.0]; DBS = [0.5, 1.0, 2.0, 3.93, 5.0]; RS_GRID = [20.0, 200.0, math.inf]
grid = {}
for sc in SCEN:
    for al in ALLOC:
        for l in LAMS:
            for d in DBS:
                for r in RS_GRID:
                    grid[f"{sc}|{al}|lam{l}|d{d}|r{r}"] = drop(r, factors_for(sc, al, l, d))
res["grid_drop"] = grid

# 対照（H22）
def comp_factors(kind, lam_tot, dbar): return [(lam_tot, dbar, kind)]
H22 = {"a": {}, "b": {}}
mixed_allocs = [a for a in ALLOC if a not in ("all_war", "all_bio")]
for al in mixed_allocs:
    wsh = war_share(al); gbar = (1 - wsh) * G_BIO if "tech" not in al else None
    for l in LAMS:
        for d in DBS:
            for r in [20.0, math.inf]:
                mix = grid[f"main_wconst_brise|{al}|lam{l}|d{d}|r{r}"]
                c = drop(r, comp_factors(("exp", 0.0), l, d)); c200 = drop(r, comp_factors(("exp", 1 / 200), l, d))
                H22["a"][f"{al}|lam{l}|d{d}|r{r}"] = {"mix": mix, "comp_const": c, "comp_tau200": c200, "diff": mix - c}
                if gbar is not None:
                    mixb = grid[f"wconst_bdecl|{al}|lam{l}|d{d}|r{r}"]
                    cb = drop(r, comp_factors(("exp", gbar), l, d))
                    H22["b"][f"{al}|lam{l}|d{d}|r{r}"] = {"mix": mixb, "comp_gbar": cb, "gbar": gbar, "diff": mixb - cb}
res["H22_detail"] = H22

# 代表曲線（λ0_tot=1, δ̄=2）
RCURVE = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 2000.0, math.inf]
curves = {}
for sc in SCEN:
    for al in ["r006_eqd", "r006_d006", "all_war", "all_bio"]:
        curves[f"{sc}|{al}"] = [drop(r, factors_for(sc, al, 1.0, 2.0)) for r in RCURVE]
for name, kind in [("single_const", ("exp", 0.0)), ("single_tau200", ("exp", 1 / 200)), ("single_bio_decl", ("exp", G_BIO)),
                   ("single_bio_rise", ("rise", 50.0, 3.0))]:
    curves[name] = [drop(r, [(1.0, 2.0, kind)]) for r in RCURVE]
res["curves_lam1_d2"] = {"r": [str(r) for r in RCURVE], "drop": curves}

# H21 感度: τ_bio を振った bio のみ
TB_SENS = {"main": TAU_BIO, "marani_2pt": res["tau_bio_marani_two_point"]["window_centres_1909.5_2009.5"],
           "marani_lo": res["tau_bio_marani_two_point"]["ci_band"][0], "marani_hi": res["tau_bio_marani_two_point"]["ci_band"][1]}
for k in ["1920-2019", "1946-2019"]:
    if fits[k]["tau_s_hat"]: TB_SENS[f"fit_{k}"] = fits[k]["tau_s_hat"]
for k, v in [("ci_lo", main["tau_s_ci95"][0]), ("ci_hi", main["tau_s_ci95"][1]),
             ("qci_lo", main["tau_s_ci95_quasi"][0]), ("qci_hi", main["tau_s_ci95_quasi"][1])]:
    if v: TB_SENS[f"main_{k}"] = v
h21 = {}
for nm, tb in TB_SENS.items():
    h21[nm] = {"tau_bio": tb, "drops_r20": {f"lam{l}_d{d}": drop(20.0, [(l, d, ("exp", 1 / tb))]) for l in LAMS for d in DBS}}
res["H21_detail"] = h21

# 上昇の時間尺度の感度（H20 の頑健性）
rise_sens = {}
for th in [10.0, 20.0, 50.0, 100.0, 300.0]:
    for M in [1.5, 3.0]:
        rise_sens[f"theta{th}_M{M}"] = min(drop(20.0, [(l * 4 / 7, d, ("exp", 0.0)), (l * 3 / 7, d, ("rise", th, M))]) for l in LAMS for d in DBS)
res["H20_rise_sensitivity_min_drop_r20_r006eqd"] = rise_sens

# H23: 符号反転の戦争 λ0δ 比 w*（等 δ なので率比と同じ）
def mix_drop_w(wsh, r, tb, lam=1.0, d=2.0):
    return drop(r, [(lam * wsh, d, ("exp", 0.0)), (lam * (1 - wsh), d, ("exp", 1 / tb))])
def wroot(r, tb, lam=1.0, d=2.0):
    f0, f1 = mix_drop_w(0.0, r, tb, lam, d), mix_drop_w(1.0, r, tb, lam, d)
    if f0 * f1 > 0: return None
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if mix_drop_w(mid, r, tb, lam, d) * f0 > 0: lo = mid
        else: hi = mid
    return (lo + hi) / 2
# 1次近似の w*
def J(T, r, kind):
    wl, c = pre(T, r, kind); return float(np.sum(wl * c)) * 100
def wstar_lin(r, tb):
    Jc = (J(T_EVAL, r, ("exp", 0.0)), J(W, r, ("exp", 0.0))); Jd = (J(T_EVAL, r, ("exp", 1 / tb)), J(W, r, ("exp", 1 / tb)))
    q = I0(T_EVAL) / I0(W)
    # [wJc(T)+(1-w)Jd(T)] = q[wJc(W)+(1-w)Jd(W)]
    num = q * Jd[1] - Jd[0]; den = (Jc[0] - Jd[0]) - q * (Jc[1] - Jd[1])
    return num / den
h23 = {"wstar_exact_r20": {}, "wstar_lin_r20": {}}
for nm in ["main", "marani_2pt", "marani_lo", "marani_hi"]:
    tb = TB_SENS[nm]; h23["wstar_exact_r20"][nm] = wroot(20.0, tb); h23["wstar_lin_r20"][nm] = wstar_lin(20.0, tb)
h23["wstar_lin_r20_tau58_handcheck"] = wstar_lin(20.0, 58.0)
h23["wstar_exact_r20_main_by_lam_d"] = {f"lam{l}_d{d}": wroot(20.0, TAU_BIO, l, d) for l in LAMS for d in DBS}
h23["war_share_r006_d006"] = war_share("r006_d006"); h23["war_share_r006_eqd"] = war_share("r006_eqd")
res["H23_detail"] = h23

# 符号マップ（λ0=1, δ̄=2, 等 δ）: w × τ_bio @ r=20、 w × r @ τ_bio=主推定
WG = np.linspace(0, 1, 21); TBG = np.geomspace(20, 2000, 15); RG2 = np.geomspace(1, 3000, 15)
sm1 = [[mix_drop_w(w, 20.0, tb) for w in WG] for tb in TBG]
sm2 = [[mix_drop_w(w, r, TAU_BIO) for w in WG] for r in RG2]
smr = [[drop(r, [(w, 2.0, ("exp", 0.0)), (1 - w, 2.0, ("rise", 50.0, 3.0))]) for w in WG] for r in RG2]
res["sign_maps"] = {"w": WG.tolist(), "tau_bio": TBG.tolist(), "r": RG2.tolist(),
                    "decl_r20_w_by_taubio": sm1, "decl_taumain_w_by_r": sm2, "rise_w_by_r": smr}

# ============================ 判定 ============================
h20_vals = {k: v for k, v in grid.items() if k.startswith("main_wconst_brise|") and k.endswith("|r20.0")}
h21_main = h21["main"]["drops_r20"]
h21s_ok = all(all(v < 0 for v in h21[nm]["drops_r20"].values()) for nm in ["main", "marani_2pt", "marani_lo", "marani_hi"])
a20 = [v for k, v in H22["a"].items() if k.endswith("|r20.0")]; ainf = [v for k, v in H22["a"].items() if k.endswith("|rinf")]
b20 = [v for k, v in H22["b"].items() if k.endswith("|r20.0")]; binf = [v for k, v in H22["b"].items() if k.endswith("|rinf")]
ws_main = h23["wstar_exact_r20"]["main"]
res["verdicts"] = {
    "H20": bool(all(v > 0 for v in h20_vals.values())), "H20_min_drop_r20": min(h20_vals.values()),
    "H20_n_cells": len(h20_vals),
    "H21": bool(any(v < 0 for v in h21_main.values())), "H21_n_negative": sum(v < 0 for v in h21_main.values()),
    "H21_min_max_drop_r20": [min(h21_main.values()), max(h21_main.values())],
    "H21s": bool(h21s_ok),
    "H22a": bool(all(x["diff"] > 0 for x in a20) and all(x["diff"] >= -1e-9 for x in ainf)),
    "H22a_r20_diff_minmax": [min(x["diff"] for x in a20), max(x["diff"] for x in a20)],
    "H22a_rinf_diff_minmax": [min(x["diff"] for x in ainf), max(x["diff"] for x in ainf)],
    "H22b": bool(all(x["diff"] > 0 for x in b20) and all(x["diff"] >= -1e-9 for x in binf)),
    "H22b_r20_diff_minmax": [min(x["diff"] for x in b20), max(x["diff"] for x in b20)],
    "H22b_rinf_diff_minmax": [min(x["diff"] for x in binf), max(x["diff"] for x in binf)],
    "H22": None,
    "H23": bool(ws_main is not None and 0.4 < ws_main < 0.8 and abs(war_share("r006_d006") - ws_main) <= 0.3),
    "H23_wstar_main": ws_main, "H23_war_share_r006_d006": war_share("r006_d006"),
    "H23_handcalc_0.61_vs_lin_tau58": h23["wstar_lin_r20_tau58_handcheck"],
}
res["verdicts"]["H22"] = bool(res["verdicts"]["H22a"] and res["verdicts"]["H22b"])
res["runtime_seconds"] = time.time() - t0
json.dump(res, open(OUT / "results.json", "w"), indent=1, ensure_ascii=False, default=float)

# ============================ 図 ============================
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
bc = np.array([int(k[:4]) + 10 for k in bins]); br = np.array([v["rate_per_yr"] for v in bins.values()])
blo = np.array([v["ci95_per_yr"][0] for v in bins.values()]); bhi = np.array([v["ci95_per_yr"][1] for v in bins.values()])
ax[0].errorbar(bc, br, yerr=[br - blo, bhi - br], fmt="ko", ms=4, capsize=2, label="epidemic onsets / yr (20-yr bins, exact 95% CI)")
for key, col in [("1600-1945", "C0"), ("1900-2019", "C3"), ("1946-2019", "C2")]:
    f = fits[key]; a, b = map(int, key.split("-")); yy = np.arange(a, b + 1)
    sel = (YEARS >= a) & (YEARS <= b); xm = YEARS[sel].mean()
    ah = math.log(n_year[sel].sum() / np.sum(np.exp(f["g_hat"] * (YEARS[sel] - xm))))
    ax[0].plot(yy, np.exp(ah + f["g_hat"] * (yy - xm)), color=col, label=f"trend {key}: g={f['g_hat']:.4f}/yr")
ax[0].axvline(1945.5, color="gray", ls=":"); ax[0].set_yscale("log"); ax[0].set_xlabel("year")
ax[0].set_ylabel("onsets per year"); ax[0].set_title("Marani dataset: epidemic onset rate (CSV 1600-1945 + xlsx 1946-2019)", fontsize=9)
ax[0].legend(fontsize=7)
ss = np.linspace(0, 400, 401)
for nm, kind, st in [("bio_decl (tau=%.0f, data)" % TAU_BIO, ("exp", G_BIO), "C3-"),
                     ("bio_rise 3-2e^{-s/50} (Marani 'up to 3x')", ("rise", 50.0, 3.0), "C1-"),
                     ("war_const (Clauset/CoW)", ("exp", 0.0), "k-"), ("tau_s=200 (004-007 assumption)", ("exp", 1 / 200), "k--")]:
    ax[1].plot(ss, shape(ss, kind), st, label=nm)
ax[1].set_yscale("log"); ax[1].set_xlabel("years from now s"); ax[1].set_ylabel("m_k(s)")
ax[1].set_title("Shock-rate shape scenarios", fontsize=9); ax[1].legend(fontsize=7)
ax[2].bar([y + 10 for y in range(1820, 2004, 20)], [((large.start >= y) & (large.start < y + 20)).sum() / 20 for y in range(1820, 2004, 20)],
          width=18, color="C7", label="large wars (>=26,625 battle deaths) / yr")
fw = res["war_cow"]["trend_large_annual"]
ax[2].set_title(f"CoW v4.0 large interstate wars: g={fw['g_hat']:.4f}/yr, 95% [{fw['ci95'][0]:.4f},{fw['ci95'][1]:.4f}]", fontsize=9)
ax[2].set_xlabel("year"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "fig_epidemic_rate_and_shapes.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
rx = [1e4 if math.isinf(r) else r for r in RCURVE]
for key, st in [("main_wconst_brise|r006_d006", "C1-o"), ("wconst_bdecl|r006_d006", "C3-s"), ("wconst_bconst|r006_d006", "k-^"),
                ("w200_brise|r006_d006", "C1--"), ("w200_bdecl|r006_d006", "C3--"), ("single_tau200", "k:"), ("single_bio_decl", "C3:")]:
    ax[0].semilogx(rx, curves[key], st, ms=3, label=key)
ax[0].axhline(0, color="k", lw=0.7); ax[0].set_xlabel("recovery time r (years; r=inf plotted at 1e4)")
ax[0].set_ylabel("drop = S_noshock - S(1e4)"); ax[0].set_title("lambda0_tot=1/century, delta_bar=2, 006 allocation", fontsize=9)
ax[0].legend(fontsize=7)
im = ax[1].pcolormesh(WG, TBG, np.sign(np.array(sm1)) * np.abs(np.array(sm1)) ** 0.5, cmap="RdBu_r", shading="nearest",
                      vmin=-0.15, vmax=0.15)
ax[1].set_yscale("log"); ax[1].set_xlabel("war share w of lambda0*delta (war const, rest bio decline)")
ax[1].set_ylabel("tau_bio (years)"); ax[1].set_title("sign map r=20: red = shocks lower S (sqrt-scaled drop)", fontsize=9)
ax[1].contour(WG, TBG, np.array(sm1), levels=[0], colors="k")
ax[1].axhline(TAU_BIO, color="C3", ls="--"); ax[1].axvline(war_share("r006_d006"), color="g", ls=":", label="006 alloc")
ax[1].legend(fontsize=7); plt.colorbar(im, ax=ax[1])
plt.tight_layout(); plt.savefig(OUT / "fig_factor_drop_and_signmap.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for a_, M, tt in [(ax[0], np.array(sm2), f"war const + bio decline (tau_bio={TAU_BIO:.0f})"), (ax[1], np.array(smr), "war const + bio rise (3x)")]:
    im = a_.pcolormesh(WG, RG2, np.sign(M) * np.abs(M) ** 0.5, cmap="RdBu_r", shading="nearest", vmin=-0.4, vmax=0.4)
    if (M.min() < 0) and (M.max() > 0): a_.contour(WG, RG2, M, levels=[0], colors="k")
    a_.set_yscale("log"); a_.set_xlabel("war share w"); a_.set_ylabel("r (years)"); a_.set_title(tt + "; red: shocks lower S", fontsize=9)
    plt.colorbar(im, ax=a_)
plt.tight_layout(); plt.savefig(OUT / "fig_sign_map_w_by_r.png", dpi=120); plt.close()

print(json.dumps({k: v for k, v in res.items() if k not in ("grid_drop", "H22_detail", "sign_maps")}, indent=1, ensure_ascii=False, default=float))
print(f"runtime_sec={time.time() - t0:.1f}")
