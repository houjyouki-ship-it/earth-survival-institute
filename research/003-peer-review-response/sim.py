"""003: 査読対応 — Ord 1/6 再較正・Hawkes風脆弱化・ハザード構造比較・感度分析
再現: python3 sim.py  (seed=20261014) -> results.json, fig_*.png, stdout.txt
"""
from __future__ import annotations
import json, math, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261014
N = 200_000
N_HAWKES = 30_000
TAU_MED = 200.0
H_NAT = 1 / 87_000
SIGMA = 1.0
TS_MAIN = [100, 1_000, 10_000, 100_000]

# ---------- 1. Ord 1/6 較正（解析） ----------
# S(100)=exp[-h0·τ·(1-e^(-100/τ))-100·h_nat]=5/6
# => h0 = [ln(6/5) - 100·h_nat] / [τ·(1-e^(-100/τ))]
def h0_calibrated_exp(tau=TAU_MED, h_nat=H_NAT, T=100.0, p_cat=1 / 6):
    target = -math.log(1 - p_cat)  # ln(6/5)
    denom = tau * (1.0 - math.exp(-T / tau))
    return (target - T * h_nat) / denom

h0_old = 1 - (5 / 6) ** (1 / 100)  # 旧: 一定ハザード換算
h0_new = h0_calibrated_exp()

def H_peril_exp(T, h0, tau):
    return h0 * tau * (-np.expm1(-T / tau))

def S_point_exp(T, h0, tau=TAU_MED, hn=H_NAT):
    return float(math.exp(-(h0 * tau * (-math.expm1(-T / tau)) + hn * T)))

def draw_lognormal(n, h0med, tau_med, sigma, rng, rho=0.0):
    z1 = rng.normal(size=n)
    w = rng.normal(size=n)
    z2 = rho * z1 + math.sqrt(1 - rho * rho) * w
    return h0med * np.exp(sigma * z1), tau_med * np.exp(sigma * z2)

def S_mc(T, h0, tau, hn=H_NAT):
    x = np.exp(-(H_peril_exp(T, h0, tau) + hn * T))
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size))

t0 = time.time()
res = {
    "seed": SEED,
    "N": N,
    "N_hawkes": N_HAWKES,
    "tau_median": TAU_MED,
    "h_nat_upper": H_NAT,
    "event_definitions": {
        "existential_catastrophe_Ord": "人類の長期的可能性を不可逆に失う存続破局（Ord 2020）。本モデルの主終端事象。",
        "extinction_XPT": "世界人口5,000人未満（XPT）。本モデルの主終端とは別。混同しない。",
        "natural_extinction_upper_SnyderBeattie": "自然要因による人類絶滅の年率上限（点推定ではない）。",
        "belief_note": "Ordの1/6およびσ事前は研究者の信念の度合い（主観予測）であり、観測頻度ではない。",
    },
}

# --- 較正表 ---
calib = {
    "formula": "h0 = [ln(6/5) - 100*h_nat] / [tau*(1-exp(-100/tau))]",
    "tau": TAU_MED,
    "h_nat": H_NAT,
    "h0_old_constant_conversion": h0_old,
    "h0_new_decay_calibrated": h0_new,
    "h0_new_rounded_reviewer": 0.002302,
    "ratio_new_over_old": h0_new / h0_old,
    "pct_increase": 100 * (h0_new / h0_old - 1),
    "S100_point_old": S_point_exp(100, h0_old),
    "S100_point_new": S_point_exp(100, h0_new),
    "P_catastrophe_100_old": 1 - S_point_exp(100, h0_old),
    "P_catastrophe_100_new": 1 - S_point_exp(100, h0_new),
    "reviewer_claimed_h0": 0.002302,
    "abs_diff_vs_reviewer_claim": abs(h0_new - 0.002302),
}
res["calibration"] = calib

# --- 001/002 主要結果の新旧再計算 ---
rng = np.random.default_rng(SEED)
recalc = {"point_estimate": {}, "baseline_sigma1_independent": {}, "delta_point": {}, "delta_baseline": {}}
for label, h0m in [("old", h0_old), ("new", h0_new)]:
    pe = {f"S_{T}": S_point_exp(T, h0m) for T in TS_MAIN}
    h0s, taus = draw_lognormal(N, h0m, TAU_MED, SIGMA, rng, rho=0.0)
    bl = {}
    for T in TS_MAIN:
        m, se = S_mc(T, h0s, taus)
        bl[f"S_{T}"] = m
        bl[f"SE_{T}"] = se
    recalc["point_estimate"][label] = pe
    recalc["baseline_sigma1_independent"][label] = bl
# 差分（new - old）
for T in TS_MAIN:
    k = f"S_{T}"
    recalc["delta_point"][k] = recalc["point_estimate"]["new"][k] - recalc["point_estimate"]["old"][k]
    recalc["delta_baseline"][k] = (
        recalc["baseline_sigma1_independent"]["new"][k] - recalc["baseline_sigma1_independent"]["old"][k]
    )
# 001 介入 A/B（新較正・不確実性込み）
h0s, taus = draw_lognormal(N, h0_new, TAU_MED, SIGMA, np.random.default_rng(SEED), rho=0.0)
recalc["new_interventions_baseline"] = {
    "A_halve_h0": {f"S_{T}": S_mc(T, 0.5 * h0s, taus)[0] for T in TS_MAIN},
    "B_halve_tau": {f"S_{T}": S_mc(T, h0s, 0.5 * taus)[0] for T in TS_MAIN},
}
res["recalc_001_002_key"] = recalc

# --- σ事前感度（信念の度合い）---
sigma_sens = {}
for s in [0.5, 1.0, 1.5]:
    r = np.random.default_rng(SEED + 7)
    out = {}
    for label, h0m in [("old", h0_old), ("new", h0_new)]:
        h0s, taus = draw_lognormal(N, h0m, TAU_MED, s, r, rho=0.0)
        out[label] = {f"S_{T}": S_mc(T, h0s, taus)[0] for T in [100, 10_000]}
    sigma_sens[s] = out
res["sigma_prior_sensitivity"] = sigma_sens

# --- 仮説→モデルの性質（解析的恒等式の確認）---
# H2: ∫_0^∞ h0 e^{-t/τ} dt = h0 τ → 半減等価
props = {
    "H2_halve_h0_vs_halve_tau": {
        "status": "model_property_not_hypothesis",
        "identity": "∫_0^∞ h0 exp(-t/τ) dt = h0*τ; halving h0 or τ yields same infinite cumulative peril",
        "check_infinite_cum_ratio": 1.0,  # (0.5*h0)*τ / (h0*(0.5*τ))
        "finite_T100_not_identical": True,
        "S100_A_point_new": S_point_exp(100, 0.5 * h0_new),
        "S100_B_point_new": float(
            math.exp(-(h0_new * (0.5 * TAU_MED) * (-math.expm1(-100 / (0.5 * TAU_MED))) + H_NAT * 100))
        ),
    },
    "H5_var_log_h0tau": {
        "status": "model_property_not_hypothesis",
        "identity": "Var[log(h0*τ)] = 2*σ²*(1+ρ) under bivariate lognormal",
        "check_formula_at_rho0_sigma1": 2.0 * (1.0 ** 2) * (1 + 0.0),
        "note": "長期Sが h0τ のみに依存するなら同じ積分布で同じS。有限期間の差は研究対象になり得る。",
    },
    "H6_equal_area_pulse": {
        "status": "model_property_not_hypothesis",
        "identity": "additive pulse fully realized by T => S_none/S_pulse = exp(A*w) exactly",
        "example_Aw": h0_new * 100,
        "ratio_identity": math.exp(h0_new * 100),
    },
    "H3_natural_floor_dominance": {
        "status": "model_property_under_assumed_floor",
        "note": "h_nat を固定床として置くと T≫τ で人為項は消え自然床が支配。床値自体は Snyder-Beattie の上限であり仮説的。",
        "S_1e5_with_nat_point_new": S_point_exp(100_000, h0_new),
        "S_1e5_no_nat_point_new": float(
            math.exp(-(h0_new * TAU_MED * (-math.expm1(-100_000 / TAU_MED))))
        ),
    },
}
res["hypothesis_demotions"] = props

# ---------- ハザード構造比較（いずれも100年破局1/6に較正） ----------
# 1) 指数減衰: 上記 h0_new
# 2) 一定: h_c + h_nat, 100*(h_c+h_nat)=ln(6/5)
# 3) 線形増加: h(t)=a+b*t + h_nat, ここでは「100年で2倍」: h(t)=a*(1+t/100)+h_nat
# 4) べき減衰: h(t)=h0*(1+t/τ)^(-α)+h_nat, α=1
ln65 = math.log(6 / 5)

h_const = ln65 / 100 - H_NAT  # 人為項のみ

# 線形: ∫_0^100 a(1+t/100) dt = a*[t + t²/(200)]_0^100 = a*(100+50)=150a
# 150a + 100*h_nat = ln(6/5) => a = (ln65 - 100*h_nat)/150
a_lin = (ln65 - 100 * H_NAT) / 150.0
b_lin = a_lin / 100.0  # h_anth(t)=a+b*t = a(1+t/100)

# べき減衰 α=1, τ=200: ∫_0^T h0/(1+t/τ) dt = h0*τ*ln(1+T/τ)
# h0*200*ln(1.5) + 100*h_nat = ln65
h0_pow = (ln65 - 100 * H_NAT) / (TAU_MED * math.log(1 + 100 / TAU_MED))
ALPHA_POW = 1.0


def cum_anth(structure, T, params):
    """人為累積ハザード（自然床除く）"""
    if structure == "exp_decay":
        h0, tau = params["h0"], params["tau"]
        return h0 * tau * (1 - math.exp(-T / tau))
    if structure == "constant":
        return params["h"] * T
    if structure == "linear_increase":
        a, b = params["a"], params["b"]
        # ∫_0^T (a+b t) dt = aT + b T²/2
        return a * T + 0.5 * b * T * T
    if structure == "power_decay":
        h0, tau, alpha = params["h0"], params["tau"], params["alpha"]
        if abs(alpha - 1.0) < 1e-12:
            return h0 * tau * math.log(1 + T / tau)
        # ∫ h0 (1+t/τ)^{-α} dt = h0 τ/(1-α) [(1+T/τ)^{1-α} - 1]
        return h0 * tau / (1 - alpha) * ((1 + T / tau) ** (1 - alpha) - 1)
    raise ValueError(structure)


structures = {
    "exp_decay": {"h0": h0_new, "tau": TAU_MED},
    "constant": {"h": h_const},
    "linear_increase": {"a": a_lin, "b": b_lin, "note": "anthropogenic hazard doubles by year 100"},
    "power_decay_alpha1": {"h0": h0_pow, "tau": TAU_MED, "alpha": ALPHA_POW},
}
struct_map = {
    "exp_decay": "exp_decay",
    "constant": "constant",
    "linear_increase": "linear_increase",
    "power_decay_alpha1": "power_decay",
}
struct_results = {}
for name, params in structures.items():
    key = struct_map[name]
    row = {"params": params, "S": {}, "P_catastrophe": {}}
    for T in [100, 10_000, 100_000]:
        H = cum_anth(key if name != "power_decay_alpha1" else "power_decay", T, params) + H_NAT * T
        s = math.exp(-H)
        row["S"][T] = s
        row["P_catastrophe"][T] = 1 - s
    # 検証: S(100)≈5/6
    row["S100_minus_5over6"] = row["S"][100] - 5 / 6
    struct_results[name] = row
res["structure_comparison"] = struct_results

# ---------- Hawkes風脆弱化（刻み内 Poisson 複数事象） ----------
def hawkes_damage(h0_arr, tau_arr, c, delta, dt, T_end, rng, hn=H_NAT):
    """各刻みで n~Poisson(c * H_peril_step)。mult *= (1+δ)^n。
    存続ハザードは mult * peril + hn。δ=0 なら解析式に一致するはず。
    """
    n = h0_arr.size
    mult = np.ones(n)
    logS = np.zeros(n)
    ncat = np.zeros(n)
    targets = {T for T in (100, 1000, 10000) if T <= T_end + 1e-9}
    out = {}
    n_steps = int(math.ceil(T_end / dt))
    for i in range(n_steps):
        a = i * dt
        b = min((i + 1) * dt, T_end)
        # ∫_a^b h0 exp(-u/τ) du = h0*τ*(exp(-a/τ)-exp(-b/τ))
        hp_base = h0_arr * tau_arr * (np.exp(-a / tau_arr) - np.exp(-b / tau_arr))
        H_exist = mult * hp_base
        logS -= H_exist
        if delta != 0.0:
            lam = c * H_exist
            alive = logS > -50.0
            n_ev = np.zeros(n, dtype=np.int32)
            if np.any(alive):
                n_ev[alive] = rng.poisson(np.maximum(lam[alive], 0.0))
            with np.errstate(over="ignore"):
                mult = np.where(n_ev > 0, mult * np.exp(n_ev * math.log1p(delta)), mult)
            mult = np.minimum(mult, 1e12)
            ncat += n_ev
        for Trec in list(targets):
            if Trec not in out and b + 1e-9 >= Trec:
                x = np.exp(logS - hn * Trec)
                out[Trec] = {
                    "S": float(x.mean()),
                    "SE": float(x.std(ddof=1) / math.sqrt(n)),
                    "mean_cats": float(ncat.mean()),
                    "mean_cats_surv_w": float((ncat * x).sum() / max(x.sum(), 1e-300)),
                }
        if b >= T_end - 1e-12:
            break
    return out


def analytic_S_dict(h0_arr, tau_arr, Ts, hn=H_NAT):
    d = {}
    for T in Ts:
        x = np.exp(-(H_peril_exp(T, h0_arr, tau_arr) + hn * T))
        d[T] = {"S": float(x.mean()), "SE": float(x.std(ddof=1) / math.sqrt(h0_arr.size))}
    return d


# 共通乱数の世界パラメータ
rng_h = np.random.default_rng(SEED + 11)
h0_h, tau_h = draw_lognormal(N_HAWKES, h0_new, TAU_MED, SIGMA, rng_h, rho=0.0)

hawkes = {
    "note": "非致死的大災害は存続破局ハザードの c 倍の強度で発生（未較正シナリオ）。δ は1回あたりの以後ハザード倍率増分（未較正）。c・δは根拠のない仮定／感度範囲。",
    "c_values": [1.0, 3.33, 9.05],
    "delta_values": [0.0, 0.5, 1.0],
    "dt_values": [1.0, 0.1, 0.01],
    "validation_delta0": {},
    "convergence": {},
    "sensitivity_c_delta": {},
}

# δ=0 検証（解析一致）+ 収束（主要: c=3.33, δ=1）
analytic = analytic_S_dict(h0_h, tau_h, [100, 1000, 10000])
hawkes["analytic_baseline_new_h0"] = {str(k): v for k, v in analytic.items()}

for dt in [1.0, 0.1, 0.01]:
    T_end = 1000.0 if dt <= 0.01 else 10000.0
    n_use = N_HAWKES if dt >= 0.1 else min(12_000, N_HAWKES)
    h0_u, tau_u = h0_h[:n_use], tau_h[:n_use]
    anal_u = analytic_S_dict(h0_u, tau_u, [100, 1000, 10000])
    r = np.random.default_rng(SEED + 100 + int(dt * 1000))
    out0 = hawkes_damage(h0_u, tau_u, c=3.33, delta=0.0, dt=dt, T_end=T_end, rng=r)
    hawkes["validation_delta0"][str(dt)] = {
        "T_end": T_end, "N": n_use,
        "results": {str(k): v for k, v in out0.items()},
        "abs_err_vs_analytic": {
            str(k): abs(out0[k]["S"] - anal_u[k]["S"]) for k in out0 if k in anal_u
        },
    }

# 収束試験: c=3.33, δ=1
for dt in [1.0, 0.1, 0.01]:
    T_end = 1000.0 if dt <= 0.01 else 10000.0
    n_use = N_HAWKES if dt >= 0.1 else min(12_000, N_HAWKES)
    h0_u, tau_u = h0_h[:n_use], tau_h[:n_use]
    r = np.random.default_rng(SEED + 200 + int(dt * 1000))
    out = hawkes_damage(h0_u, tau_u, c=3.33, delta=1.0, dt=dt, T_end=T_end, rng=r)
    hawkes["convergence"][str(dt)] = {"T_end": T_end, "N": n_use, "results": {str(k): v for k, v in out.items()}}

# c・δ 感度（dt=1, T=10000）— 複数組み合わせ
sens_cd = {}
for c in [1.0, 3.33, 9.05]:
    for delta in [0.0, 0.5, 1.0]:
        r = np.random.default_rng(SEED + 300 + int(c * 100) + int(delta * 10))
        out = hawkes_damage(h0_h, tau_h, c=c, delta=delta, dt=1.0, T_end=10000.0, rng=r)
        sens_cd[f"c{c}_delta{delta}"] = {str(k): v for k, v in out.items()}
hawkes["sensitivity_c_delta"] = sens_cd

# 旧実装（最大1回/刻み）との比較用: -expm1 方式, dt=1
def legacy_max1(h0_arr, tau_arr, c, delta, dt, T_end, rng, hn=H_NAT):
    n = h0_arr.size
    mult = np.ones(n)
    logS = np.zeros(n)
    ncat = np.zeros(n)
    out = {}
    targets = {T for T in (100, 1000, 10000) if T <= T_end + 1e-9}
    n_steps = int(math.ceil(T_end / dt))
    for i in range(n_steps):
        a = i * dt
        b = min((i + 1) * dt, T_end)
        hp_base = h0_arr * tau_arr * (np.exp(-a / tau_arr) - np.exp(-b / tau_arr))
        H_exist = mult * hp_base
        logS -= H_exist
        if delta > 0:
            u = rng.random(n)
            alive = logS > -50.0
            ev = (u < -np.expm1(-c * H_exist)) & alive
            mult = np.where(ev, mult * (1 + delta), mult)
            ncat += ev.astype(float)
        for Trec in list(targets):
            if Trec not in out and b + 1e-9 >= Trec:
                x = np.exp(logS - hn * Trec)
                out[Trec] = {"S": float(x.mean()), "SE": float(x.std(ddof=1) / math.sqrt(n))}
        if b >= T_end - 1e-12:
            break
    return out


rleg = np.random.default_rng(SEED + 200 + 1000)
leg = legacy_max1(h0_h, tau_h, 3.33, 1.0, 1.0, 10000.0, rleg)
hawkes["legacy_max1_per_step_c3.33_delta1_dt1"] = {str(k): v for k, v in leg.items()}
res["hawkes_fragility"] = hawkes

# ---------- 不確実性込みの平均較正（査読 2.1 後半）----------
# E_σ[S(100)] = 5/6 となる h0 中央値を二分法で求める（共通乱数）。
def mean_calibrate(sigma, n=N, seed=SEED + 21):
    r = np.random.default_rng(seed)
    z1 = r.normal(size=n); z2 = r.normal(size=n)
    tau = TAU_MED * np.exp(sigma * z2)
    f = lambda m: float(np.mean(np.exp(-(H_peril_exp(100.0, m * np.exp(sigma * z1), tau) + H_NAT * 100)))) - 5 / 6
    lo, hi = 1e-5, 0.05
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if f(mid) > 0: lo = mid
        else: hi = mid
    m = math.sqrt(lo * hi)
    h0s = m * np.exp(sigma * z1)
    return {"h0_median": m, "ratio_vs_point_new": m / h0_new,
            **{f"S_{T}": S_mc(T, h0s, tau)[0] for T in TS_MAIN}}
res["mean_calibrated_uncertain"] = {s: mean_calibrate(s) for s in [0.5, 1.0, 1.5]}

# ---------- 文献取得状況 ----------
res["literature_checked"] = {
    "Thorstad_2022_Existential_Risk_Pessimism_and_Time_of_Perils": {
        "opened": True,
        "url": "https://www.globalprioritiesinstitute.org/wp-content/uploads/David-Thorstad-Existential-risk-pessimism1.pdf",
        "also": "https://reflectivealtruism.com/2022/11/30/existential-risk-pessimism-and-the-time-of-perils-part-i-the-problem/",
        "summary": "悲観的な存続リスク推定はしばしばリスク削減の「天文的価値」を弱める。天文的価値を救うには短い危険期＋極めて低いその後リスクが必要だが、その仮説は十分支持されないと論じる。",
    },
    "Hawkes_1971_Biometrika": {
        "opened": True,
        "url": "https://www.dcscience.net/Hawkes-Biometrika-1971.pdf",
        "doi": "10.1093/biomet/58.1.83",
        "summary": "自己励起・相互励起点過程のスペクトル。強度 λ(t)=ν+∫ g(t-u)dN(u)。本研究の脆弱化は「大災害が以後の存続ハザードを恒久的に上げる」離散近似であり、減衰カーネル付き古典Hawkesそのものではない。",
    },
    "Liu_Wolfe_Huang_2004_Shared_Frailty": {
        "opened_full_pdf": False,
        "opened_abstract": True,
        "url_abstract": "https://onlinelibrary.wiley.com/doi/10.1111/j.0006-341X.2004.00225.x",
        "doi": "10.1111/j.0006-341X.2004.00225.x",
        "url_repo_attempted": "https://deepblue.lib.umich.edu/items/b4e10ade-dacf-4252-8b26-f9ebff88f258",
        "summary_from_abstract": "反復事象と終端事象を共有ガンマフレイルティで結合する比例ハザードモデル。反復災害＋終端破局の同時モデルの参考。全文PDFは本環境では未取得。",
    },
}

res["runtime_seconds"] = None  # filled later
json.dump(res, open("results.json", "w"), indent=2, ensure_ascii=False)

# ---------- 図 ----------
# fig1: 新旧 h0 点推定・ベースライン生存曲線
tt = np.logspace(1, 5, 80)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for lab, h0m, ls in [("old h0=0.001822", h0_old, "--"), ("new h0=0.002302", h0_new, "-")]:
    ax[0].semilogx(tt, [S_point_exp(T, h0m) for T in tt], ls, label=lab)
ax[0].axhline(5 / 6, color="gray", lw=0.8, label="Ord 5/6 at 100y target")
ax[0].axvline(100, color="gray", lw=0.5, ls=":")
ax[0].set_xlabel("years"); ax[0].set_ylabel("S(T) point estimate")
ax[0].set_title("Recalibration of Ord 1/6 under exponential decay")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

# 構造比較
for name, style in [
    ("exp_decay", "-"),
    ("constant", "--"),
    ("linear_increase", "-."),
    ("power_decay_alpha1", ":"),
]:
    params = structures[name]
    key = struct_map[name]
    ss = []
    for T in tt:
        H = cum_anth(key if name != "power_decay_alpha1" else "power_decay", float(T), params) + H_NAT * float(T)
        ss.append(math.exp(-H))
    ax[1].semilogx(tt, ss, style, label=name)
ax[1].axhline(5 / 6, color="gray", lw=0.8)
ax[1].axvline(100, color="gray", lw=0.5, ls=":")
ax[1].set_xlabel("years"); ax[1].set_ylabel("S(T)")
ax[1].set_title("Same 100y catastrophe=1/6 calibration, different structures")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_recalibration_and_structures.png", dpi=130)

# fig2: Hawkes 収束 + c,δ 感度
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
# 収束: S vs dt at T=100,1000
dts = [1.0, 0.1, 0.01]
for Trec, mk in [(100, "o"), (1000, "s")]:
    ys = []
    for dt in dts:
        ys.append(hawkes["convergence"][str(dt)]["results"].get(str(Trec), {}).get("S", float("nan")))
    ax[0].plot(dts, ys, mk + "-", label=f"S({Trec}y), c=3.33, δ=1")
# δ=0 errors
ax0b = ax[0].twinx()
errs100 = [hawkes["validation_delta0"][str(dt)]["abs_err_vs_analytic"].get("100", float("nan")) for dt in dts]
ax0b.plot(dts, errs100, "k--", alpha=0.5, label="|S−analytic| δ=0 T=100")
ax[0].set_xscale("log"); ax[0].invert_xaxis()
ax[0].set_xlabel("dt (years)"); ax[0].set_ylabel("S")
ax[0].set_title("Hawkes-style convergence (Poisson per step)")
ax[0].legend(loc="lower left", fontsize=8); ax[0].grid(alpha=0.3)

# c,δ heatmap-like bars for S(10000)
labels = []
vals = []
for c in [1.0, 3.33, 9.05]:
    for delta in [0.0, 0.5, 1.0]:
        labels.append(f"c={c}\nδ={delta}")
        vals.append(sens_cd[f"c{c}_delta{delta}"]["10000"]["S"])
ax[1].bar(range(len(vals)), vals, color=["C0", "C1", "C2"] * 3)
ax[1].axhline(analytic[10000]["S"], color="k", ls="--", lw=1, label="no fragility (analytic)")
ax[1].set_xticks(range(len(vals)))
ax[1].set_xticklabels(labels, fontsize=7)
ax[1].set_ylabel("S(10,000 y)")
ax[1].set_title("c·δ sensitivity (unanchored scenario params; dt=1)")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("fig_hawkes_convergence_sensitivity.png", dpi=130)

# fig3: σ感度
fig, ax = plt.subplots(figsize=(7, 4))
sigs = [0.5, 1.0, 1.5]
ax.plot(sigs, [sigma_sens[s]["old"]["S_10000"] for s in sigs], "o--", label="old h0, S(1e4)")
ax.plot(sigs, [sigma_sens[s]["new"]["S_10000"] for s in sigs], "s-", label="new h0, S(1e4)")
ax.plot(sigs, [sigma_sens[s]["new"]["S_100"] for s in sigs], "^-", label="new h0, S(100)")
ax.set_xlabel("σ prior (belief uncertainty on log h0, log τ)")
ax.set_ylabel("mean S")
ax.set_title("Sensitivity to σ prior (median-anchored lognormals)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_sigma_sensitivity.png", dpi=130)

res["runtime_seconds"] = time.time() - t0
json.dump(res, open("results.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
print(f"\n# runtime_seconds={res['runtime_seconds']:.1f}", file=sys.stderr)
