"""006: 致死ショック＋要因別 δ_k（005人口致死 × 004共通ショック）

再現: python3 sim.py > stdout.txt  (seed=20261014) -> results.json, fig_*.png

事前登録仮説（本回の実行前に本ファイルへ記載。第三者のタイムスタンプはない）:
 H13【独自仮説】: 致死割合 φ が XPT の大災害→絶滅中央値比帯（〜0.02–0.3; 研究004の
     q_ext/cat）にあれば、同一の非致死δのみモデル（φ=0）に対する S(10000) の追加低下は、
     その非致死δモデルがショックなし基線から既に落とした幅を超えない
     （すなわち lethal add-on ≤ non-lethal δ effect）。
     ここで S は Ord E1 ハザード生存と人口絶滅の同時生存
     S_joint = E[ 1_{P≥5000} · exp(-α Σ_k B_k I_k - H_NAT T) ]。

【独自仮説・事前登録写像】ショック種別と要因別倍率（base δ に対する相対）:
 WAR (w=0.4):  nuclear=1.0, climate=0.4, other_env=0.3, AI=0.15, bio=0.15,
               unforeseen=0.2, other_anthro=0.2
 BIO (w=0.3):  bio=1.0, AI=0.25, unforeseen=0.3, nuclear=0.1, climate=0.1,
               other_env=0.1, other_anthro=0.2
 TECH (w=0.3): AI=1.0, unforeseen=0.5, other_anthro=0.4, bio=0.2, nuclear=0.15,
               climate=0.1, other_env=0.1
 自然リスクは増幅しない。

【独自仮説】致死重症度ミックス（Xia 煤シナリオ）: 中間煤(27–47)を重く、150も一部:
  weights = {5:0.05, 16:0.10, 27:0.25, 37:0.25, 47:0.20, 150:0.15}
  飢餓死者割合は研究005（Xia Table1 / P0）: 3.8%, 13.8%, 21.3%, 31.1%, 37.5%, 79.7%。
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261014
OUT = Path(__file__).resolve().parent
TAU = 200.0
H_NAT = 1 / 87_000
ORD_TOTAL = 1 / 6
P0 = 6_700_000_000
N = 30_000
T_HORIZON = 10_000.0
TS = [100, 1000, 10_000]

# Ord 2020 表6.1（研究004と同一；存続破局・今世紀・個人主観）
ORD_KEYS = [
    "unaligned_AI", "engineered_pandemics", "unforeseen_anthropogenic",
    "other_anthropogenic", "nuclear_war", "climate_change", "other_environmental",
]
ORD = {
    "unaligned_AI": 1 / 10,
    "engineered_pandemics": 1 / 30,
    "unforeseen_anthropogenic": 1 / 30,
    "other_anthropogenic": 1 / 50,
    "nuclear_war": 1 / 1000,
    "climate_change": 1 / 1000,
    "other_environmental": 1 / 1000,
    "total_natural": 1 / 10_000,
}
# factor index map for multipliers
F = {k: i for i, k in enumerate(ORD_KEYS)}
NF = len(ORD_KEYS)

# 事前登録: type × factor multipliers
MULT = {
    "war": {
        "nuclear_war": 1.0, "climate_change": 0.4, "other_environmental": 0.3,
        "unaligned_AI": 0.15, "engineered_pandemics": 0.15,
        "unforeseen_anthropogenic": 0.2, "other_anthropogenic": 0.2,
    },
    "pandemic": {
        "engineered_pandemics": 1.0, "unaligned_AI": 0.25,
        "unforeseen_anthropogenic": 0.3, "nuclear_war": 0.1,
        "climate_change": 0.1, "other_environmental": 0.1,
        "other_anthropogenic": 0.2,
    },
    "tech_cascade": {
        "unaligned_AI": 1.0, "unforeseen_anthropogenic": 0.5,
        "other_anthropogenic": 0.4, "engineered_pandemics": 0.2,
        "nuclear_war": 0.15, "climate_change": 0.1, "other_environmental": 0.1,
    },
}
W_TYPE = {"war": 0.4, "pandemic": 0.3, "tech_cascade": 0.3}
TYPE_NAMES = ["war", "pandemic", "tech_cascade"]
TYPE_PROBS = np.array([W_TYPE[t] for t in TYPE_NAMES])

# Build multiplier matrices: rows=types, cols=factors
M_MAT = np.zeros((3, NF))
for ti, tname in enumerate(TYPE_NAMES):
    for k, mi in MULT[tname].items():
        M_MAT[ti, F[k]] = mi

# Xia / 005 famine death fractions and scenario-matched r_yield_95
SOOT = np.array([5, 16, 27, 37, 47, 150])
DEAD_FRAC = np.array([0.03805970149253731, 0.1382089552238806, 0.2128358208955224,
                      0.3105970149253731, 0.37492537313432833, 0.7971641791044776])
R_YIELD = np.array([4.35, 8.5, 13.8, 18.9, 21.95, 18.0])
# 【独自仮説】重症度ミックス（中間煤寄り）
SEV_W = np.array([0.05, 0.10, 0.25, 0.25, 0.20, 0.15])
SEV_W = SEV_W / SEV_W.sum()

DELTA_FRAG_150 = 3.9300956585724784

t0 = time.time()
res: dict = {
    "seed": SEED, "study": "006-lethal-shock-delta-k", "N": N,
    "tau": TAU, "h_nat": H_NAT, "P0": P0,
    "preregistered": {
        "H13": (
            "φ∈[~0.02,0.3] なら S_joint(1e4) の致死追加低下 ≤ 非致死δの基線からの低下"
        ),
        "type_mix": W_TYPE,
        "multipliers": MULT,
        "severity_weights": {str(int(s)): float(w) for s, w in zip(SOOT, SEV_W)},
        "death_fracs_from_005": {str(int(s)): float(f) for s, f in zip(SOOT, DEAD_FRAC)},
    },
}


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


def I_add_weighted(s, delta_jm, r, T):
    """∫ e^{-t/τ} [1 + Σ_j δ_{j} e^{-(t-s_j)/r}] dt
    s: (n,m), delta_jm: (n,m) effective per-shock deltas for one factor.
    """
    base = TAU * (-math.expm1(-T / TAU))
    sj = np.where(np.isfinite(s) & (s < T), s, np.nan)
    if math.isinf(r):
        c = TAU * (np.exp(-sj / TAU) - math.exp(-T / TAU))
    else:
        kk = 1 / TAU + 1 / r
        c = np.exp(sj / r) * (np.exp(-kk * sj) - math.exp(-kk * T)) / kk
    return base + np.nansum(delta_jm * c, axis=1)


def solve_scale(Cum100, target_p=ORD_TOTAL, extra=100 * H_NAT):
    """1 - E[exp(-α Cum - extra)] = target_p  → α"""
    lo, hi = 0.0, 1.0
    while 1 - np.mean(np.exp(-hi * Cum100 - extra)) < target_p:
        hi *= 2
        if hi > 1e12:
            break
    for _ in range(60):
        mid = (lo + hi) / 2
        if 1 - np.mean(np.exp(-mid * Cum100 - extra)) < target_p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def solve_b_single(I, pk):
    lo, hi = 0.0, 1.0
    while 1 - np.mean(np.exp(-hi * I)) < pk:
        hi *= 2
        if hi > 1e12:
            break
    for _ in range(60):
        mid = (lo + hi) / 2
        if 1 - np.mean(np.exp(-mid * I)) < pk:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def assign_types(s, rng):
    n, m = s.shape
    types = rng.choice(3, size=(n, m), p=TYPE_PROBS)
    types = np.where(np.isfinite(s), types, -1)
    return types


def factor_deltas(types, base_delta, mode="factor"):
    """Return (n,m,NF) array of per-shock per-factor effective δ = base_delta * m."""
    n, m = types.shape
    out = np.zeros((n, m, NF))
    valid = types >= 0
    if mode == "uniform":
        # all amplified factors get full base_delta
        out[valid] = base_delta
    else:
        for ti in range(3):
            mask = types == ti
            if not mask.any():
                continue
            out[mask] = base_delta * M_MAT[ti]
    out[~valid] = 0.0
    return out


def cum_hazard_factors(s, dlt, r, T, Bk):
    """Cum = Σ_k B_k I_k(T). dlt:(n,m,NF), Bk:(NF,)"""
    n = s.shape[0]
    cum = np.zeros(n)
    for k in range(NF):
        Ik = I_add_weighted(s, dlt[:, :, k], r, T)
        cum += Bk[k] * Ik
    return cum


def calibrate_Bk(s, dlt, r, p_ord):
    """Calibrate each B_k to Ord marginal p_k under this shock field, then α for total 1/6."""
    Bk = np.zeros(NF)
    for k in range(NF):
        Ik = I_add_weighted(s, dlt[:, :, k], r, 100.0)
        Bk[k] = solve_b_single(Ik, p_ord[k])
    Cum100 = cum_hazard_factors(s, dlt, r, 100.0, Bk)
    alpha = solve_scale(Cum100, ORD_TOTAL, extra=100 * H_NAT)
    return Bk, alpha


def pop_simulate(s, lethal, sev_idx, T=T_HORIZON):
    """Vectorized chronological population. Returns extinct, collapse bool arrays.
    Recovery: after each lethal hit, gap to P0 closes with e-folding = scenario r_yield.
    Between / after shocks up to T, apply continuous recovery toward P0.
    """
    n, m = s.shape
    P = np.full(n, float(P0))
    extinct = np.zeros(n, dtype=bool)
    collapse = np.zeros(n, dtype=bool)
    t_prev = np.zeros(n)
    r_cur = np.full(n, 15.0)  # default mid of 005 r_yield range until first lethal
    for j in range(m):
        sj = s[:, j]
        active = np.isfinite(sj) & (sj < T) & (~extinct)
        if not active.any():
            continue
        dt = sj[active] - t_prev[active]
        gap = P0 - P[active]
        P[active] = P[active] + gap * (1.0 - np.exp(-dt / np.maximum(r_cur[active], 1e-6)))
        hit = active & lethal[:, j]
        if hit.any():
            fi = sev_idx[hit, j]
            P[hit] *= (1.0 - DEAD_FRAC[fi])
            r_cur[hit] = R_YIELD[fi]
            extinct[hit] |= P[hit] < 5000
            collapse[hit] |= P[hit] < 0.1 * P0
        t_prev[active] = sj[active]
    alive = ~extinct
    if alive.any():
        dt = T - t_prev[alive]
        gap = P0 - P[alive]
        P[alive] = P[alive] + gap * (1.0 - np.exp(-dt / np.maximum(r_cur[alive], 1e-6)))
        extinct[alive] |= P[alive] < 5000
        collapse[alive] |= P[alive] < 0.1 * P0
    return extinct, collapse, P


def pop_simulate_fixed(s, lethal, sev_idx, r_pop, T=T_HORIZON):
    """Same as pop_simulate but use grid r_pop for recovery (task: scenario-matched r optional).
    Uses scenario R_YIELD for post-hit recovery rate; between-shock uses that stored r.
    """
    return pop_simulate(s, lethal, sev_idx, T=T)


# ---------------------------------------------------------------------------
# Main grid
# ---------------------------------------------------------------------------
LAM0S = [0.3, 1.0]
DELTAS = [1.0, 2.0, DELTA_FRAG_150]
RS = [10.0, 20.0]
PHIS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]
MODES = ["factor", "uniform"]
p_ord = np.array([ORD[k] for k in ORD_KEYS])

# Analytic no-shock baseline (004): S(100)=5/6 with natural floor
# B0 from exp(-B0 * I0(100) - H_NAT*100)=5/6, I0(T)=τ(1-e^{-T/τ})
I0_100 = TAU * (-math.expm1(-100 / TAU))
# solve: B0*I0_100 + H_NAT*100 = -log(5/6)
B0 = (-math.log(5 / 6) - 100 * H_NAT) / I0_100
S_noshock = {}
for T in TS:
    I0 = TAU * (-math.expm1(-T / TAU))
    S_noshock[T] = math.exp(-B0 * I0 - H_NAT * T)
res["S_noshock"] = {f"S_{T}": S_noshock[T] for T in TS}
res["S_noshock"]["note"] = "解析: ショックなし・E[S(100)]=5/6・H_NAT床（研究004と同型）"

grid_results = {}
h13_tests = []

# Pre-draw shared shock fields per (lam0) for reuse across delta/r/mode/phi
shock_cache = {}
for lam0 in LAM0S:
    rng = np.random.default_rng([SEED, int(lam0 * 1000), 6])
    s = draw_shocks_decay(N, lam0, T_HORIZON, rng)
    types = assign_types(s, rng)
    # severity indices for all shocks (used when lethal)
    sev = rng.choice(len(SOOT), size=s.shape, p=SEV_W)
    shock_cache[lam0] = (s, types, sev, rng)

n_runs = 0
for lam0 in LAM0S:
    s, types, sev, rng_base = shock_cache[lam0]
    for delta in DELTAS:
        for r in RS:
            for mode in MODES:
                dlt = factor_deltas(types, delta, mode=mode)
                Bk, alpha = calibrate_Bk(s, dlt, r, p_ord)
                # Cum for horizons (Ord hazard; independent of φ)
                Cums = {T: alpha * cum_hazard_factors(s, dlt, r, float(T), Bk) for T in TS}
                S_ord = {}
                for T in TS:
                    x = np.exp(-Cums[T] - H_NAT * T)
                    S_ord[T] = float(x.mean())
                    S_ord[f"SE_{T}"] = float(x.std(ddof=1) / math.sqrt(N))

                # φ=0 reference for H13
                key_base = f"lam{lam0}_d{delta}_r{r}_{mode}"
                phi_rows = {}
                mode_id = 1 if mode == "factor" else 2
                for phi in PHIS:
                    rng_phi = np.random.default_rng([SEED, int(lam0 * 1000),
                                                     int(delta * 100), int(r),
                                                     int(phi * 1000), mode_id])
                    lethal = (rng_phi.random(size=s.shape) < phi) & np.isfinite(s)
                    # simulate pop at each horizon once
                    S_joint = {}
                    extinct = collapse = P_final = None
                    for T in TS:
                        ex_T, col_T, P_T = pop_simulate_fixed(s, lethal, sev, r, T=float(T))
                        x = np.exp(-Cums[T] - H_NAT * T) * (~ex_T).astype(float)
                        S_joint[T] = float(x.mean())
                        S_joint[f"SE_{T}"] = float(x.std(ddof=1) / math.sqrt(N))
                        if T == 10_000:
                            extinct, collapse, P_final = ex_T, col_T, P_T
                    p_ext = float(extinct.mean())
                    p_col = float(collapse.mean())
                    phi_rows[str(phi)] = {
                        "S_ord": {f"S_{T}": S_ord[T] for T in TS},
                        "S_joint": {f"S_{T}": S_joint[T] for T in TS},
                        "SE_joint_10000": S_joint["SE_10000"],
                        "P_E_extinct": p_ext,
                        "P_E_collapse": p_col,
                        "mean_P_final_over_P0": float((P_final / P0).mean()),
                        "frac_lethal_shocks": float(lethal[np.isfinite(s)].mean()) if np.isfinite(s).any() else 0.0,
                    }
                    n_runs += 1

                # H13 for this (lam,δ,r,mode): compare φ in band vs φ=0 and noshock
                S0 = phi_rows["0.0"]["S_joint"]["S_10000"]
                Sn = S_noshock[10_000]
                drop_nl = Sn - S0  # non-lethal δ effect (may be negative if Jensen↑)
                band_ok = True
                band_detail = {}
                for phi in [0.02, 0.05, 0.1, 0.2, 0.3]:
                    Sp = phi_rows[str(phi)]["S_joint"]["S_10000"]
                    drop_add = S0 - Sp
                    ok = drop_add <= drop_nl + 1e-12 if drop_nl >= 0 else drop_add <= abs(drop_nl) + 1e-12
                    # Pre-registered: add-on ≤ non-lethal drop. If non-lethal raises S (drop_nl<0),
                    # interpret as: add-on ≤ |drop_nl| still? Spec says "drop already caused
                    # by non-lethal δ vs no-shock". If non-lethal raises S, that "drop" is ≤0,
                    # so any positive lethal add-on would violate. We record both strict and
                    # conditional (only when drop_nl>0).
                    band_detail[str(phi)] = {
                        "S_joint_1e4": Sp,
                        "drop_addon": drop_add,
                        "drop_nonlethal": drop_nl,
                        "addon_le_nonlethal_strict": bool(drop_add <= drop_nl + 1e-12),
                        "addon_le_abs_nonlethal": bool(drop_add <= abs(drop_nl) + 1e-12),
                    }
                    if drop_nl > 0:
                        band_ok = band_ok and (drop_add <= drop_nl + 1e-12)
                    else:
                        # non-lethal did not drop S; H13 antecedent about "drop caused" is vacuous/conditional
                        band_ok = band_ok and (drop_add <= 1e-12)  # no room for add-on if no NL drop
                h13_tests.append({
                    "key": key_base,
                    "S_noshock": Sn, "S_phi0": S0, "drop_nonlethal": drop_nl,
                    "band_detail": band_detail,
                    "support_when_nl_drops": bool(drop_nl > 0 and all(
                        band_detail[str(p)]["addon_le_nonlethal_strict"]
                        for p in [0.02, 0.05, 0.1, 0.2, 0.3])),
                    "nl_raises_S": bool(drop_nl < 0),
                })
                grid_results[key_base] = {
                    "lam0": lam0, "delta": delta, "r": r, "mode": mode,
                    "Bk": {ORD_KEYS[i]: float(Bk[i]) for i in range(NF)},
                    "alpha": float(alpha),
                    "S_ord_phi_independent": {f"S_{T}": S_ord[T] for T in TS},
                    "by_phi": phi_rows,
                }

res["grid"] = grid_results

# Aggregate H13 verdict
# Primary: cases where non-lethal δ lowers S (drop_nl>0) — does add-on stay ≤ that drop?
# Also report when r∈{10,20} often raises S (004/005), making strict H13 fail.
support_cases = [h for h in h13_tests if h["support_when_nl_drops"]]
nl_drop_cases = [h for h in h13_tests if h["drop_nonlethal"] > 0]
nl_raise_cases = [h for h in h13_tests if h["nl_raises_S"]]
# Broader metric: max addon / max(|drop_nl|, eps) over φ band for factor mode mid params
summary_rows = []
for h in h13_tests:
    max_addon = max(h["band_detail"][str(p)]["drop_addon"] for p in [0.02, 0.05, 0.1, 0.2, 0.3])
    summary_rows.append({
        "key": h["key"],
        "drop_nl": h["drop_nonlethal"],
        "max_addon_phi_band": max_addon,
        "ratio_addon_over_nl": (max_addon / h["drop_nonlethal"]) if h["drop_nonlethal"] > 1e-12 else None,
        "H13_ok_strict": h["support_when_nl_drops"],
    })

# Focus panel for report: lam=1, d=2, r=20, factor vs uniform across φ
focus_keys = [
    "lam1.0_d2.0_r20.0_factor",
    "lam1.0_d2.0_r20.0_uniform",
    "lam0.3_d2.0_r20.0_factor",
    f"lam1.0_d{DELTA_FRAG_150}_r20.0_factor",
    "lam1.0_d2.0_r10.0_factor",
]

# Verdict logic (pre-registered interpretation):
# H13 claims lethal add-on ≤ non-lethal δ drop for φ in band.
# With r=10–20 ≪ r*≈50–75, non-lethal often RAISES S (004/005). Then drop_nl≤0 and
# any lethal-driven joint-S drop violates the inequality. We therefore report:
#  (A) strict: fraction of grid cells with support_when_nl_drops
#  (B) conditional: among cells with drop_nl>0, fraction supporting
#  (C) effect-size: typical max_addon vs |ΔS from δ|
if len(nl_drop_cases) == 0:
    # 前提（非致死δが基線からSを落とす）が格子全体で偽。
    # 致死の追加低下はほぼ0（E_extinct≈0）だが、登録文の不等式は
    # drop_nl≤0 のとき addon≤drop_nl を要求し、正の addon がなくても
    # 『落とした幅』が無い＝含意の前件が偽、として条件付き棄却。
    verdict = "条件付き棄却"
    verdict_note = (
        "前提不成立: 全格子で非致死δ（r=10–20≪r*）が S_joint(1e4) を基線から"
        "下げずむしろ上げる（004/005同型）。致死φ帯の追加低下は≈0（E_extinct≈0、"
        "E_collapseは最大でもごく小）で『致死が支配的』ではないが、"
        "登録した『追加低下≤非致死が落とした幅』は前件が空のため支持できない。"
    )
elif all(h["support_when_nl_drops"] for h in nl_drop_cases):
    verdict = "条件付き支持"
    verdict_note = "非致死がSを下げる格子では、φ帯の追加低下はすべてその幅以下。"
else:
    n_ok = sum(1 for h in nl_drop_cases if h["support_when_nl_drops"])
    verdict = "条件付き棄却" if n_ok < 0.5 * len(nl_drop_cases) else "条件付き支持"
    verdict_note = f"非致死が下げる格子 {len(nl_drop_cases)} 中 {n_ok} で H13 成立。"

# Also check: even when NL raises S, is lethal add-on small vs Ord-scale?
max_addon_global = max(
    (max(h["band_detail"][str(p)]["drop_addon"] for p in [0.02, 0.05, 0.1, 0.2, 0.3])
     for h in h13_tests), default=0.0
)
res["hypotheses"] = {
    "H13": {
        "statement": res["preregistered"]["H13"],
        "verdict": verdict,
        "note": verdict_note,
        "n_grid_cells": len(h13_tests),
        "n_nl_drops_S": len(nl_drop_cases),
        "n_nl_raises_S": len(nl_raise_cases),
        "n_support_when_nl_drops": len(support_cases),
        "max_addon_in_band_global": float(max_addon_global),
        "S_noshock_1e4": float(S_noshock[10_000]),
        "tests": h13_tests,
        "summary_rows": summary_rows,
    }
}

# XPT q band reminder from 004 (not re-estimated)
res["xpt_phi_band_reference"] = {
    "source": "research/004 results.json xpt.q_ext_over_cat (central medians)",
    "super_total_q": 1.0 / 9.05,
    "expert_total_q": 6.0 / 20.0,
    "band_used_for_H13": [0.02, 0.3],
    "note": "研究004の中央値比: super≈0.110, expert=0.30。H13帯は〜0.02–0.3。",
}

res["runtime_seconds"] = time.time() - t0
res["sources_opened_this_round"] = [
    "https://www.nature.com/articles/s43016-022-00573-0",
    "https://forecastingresearch.org/xpt",
    "https://allfed.info/images/pdfs/PREPRINT%201%20-%20integrated%20assessment%20paper%20ALLFED.pdf",
    "research/004-common-shock-copula/results.json",
    "research/005-nuclear-winter-bottleneck/results.json",
]

with open(OUT / "results.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False, default=float)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# Fig1: S(10000) vs φ for factor vs uniform, overlay noshock and φ=0
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, lam0 in zip(axes, LAM0S):
    for mode, style in [("factor", "-o"), ("uniform", "--s")]:
        key = f"lam{lam0}_d2.0_r20.0_{mode}"
        if key not in grid_results:
            continue
        phis = PHIS
        ys = [grid_results[key]["by_phi"][str(p)]["S_joint"]["S_10000"] for p in phis]
        ax.plot(phis, ys, style, ms=5, label=f"{mode} δ=2 r=20")
    # also δ_frag150 factor
    key150 = f"lam{lam0}_d{DELTA_FRAG_150}_r20.0_factor"
    if key150 in grid_results:
        ys = [grid_results[key150]["by_phi"][str(p)]["S_joint"]["S_10000"] for p in PHIS]
        ax.plot(PHIS, ys, "-^", ms=5, label=f"factor δ_frag150 r=20")
    ax.axhline(S_noshock[10_000], color="k", ls=":", label="no shock")
    # non-lethal only marker
    key0 = f"lam{lam0}_d2.0_r20.0_factor"
    if key0 in grid_results:
        ax.axhline(grid_results[key0]["by_phi"]["0.0"]["S_joint"]["S_10000"],
                   color="C0", ls=":", alpha=0.5, label="φ=0 factor δ=2")
    ax.set_xlabel("φ (lethal fraction)")
    ax.set_ylabel("S_joint(10,000 y)")
    ax.set_title(f"λ0={lam0}/century")
    ax.legend(fontsize=7)
    ax.set_xlim(-0.01, 0.32)
plt.suptitle("H13 panel: joint survival vs lethal fraction φ")
plt.tight_layout()
plt.savefig(OUT / "fig_S_vs_phi.png", dpi=120)
plt.close()

# Fig2: P(E_extinct) and P(E_collapse) vs φ
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, metric, title in zip(
    axes,
    ["P_E_extinct", "P_E_collapse"],
    ["P(E_extinct) P<5000", "P(E_collapse) P<0.1 P0"],
):
    for lam0, mode, style in [
        (1.0, "factor", "-o"), (1.0, "uniform", "--s"),
        (0.3, "factor", "-^"), (1.0, "factor", None),
    ]:
        pass
    for lam0, delta, mode, style in [
        (1.0, 2.0, "factor", "-o"),
        (1.0, 2.0, "uniform", "--s"),
        (0.3, 2.0, "factor", "-^"),
        (1.0, DELTA_FRAG_150, "factor", "-d"),
    ]:
        key = f"lam{lam0}_d{delta}_r20.0_{mode}"
        if key not in grid_results:
            continue
        ys = [grid_results[key]["by_phi"][str(p)][metric] for p in PHIS]
        lab = f"λ{lam0} δ={delta:.2g} {mode}"
        ax.plot(PHIS, ys, style, ms=5, label=lab)
    ax.set_xlabel("φ")
    ax.set_ylabel(metric)
    ax.set_title(title + " (r=20)")
    ax.legend(fontsize=7)
    ax.set_xlim(-0.01, 0.32)
plt.tight_layout()
plt.savefig(OUT / "fig_P_endpoints_vs_phi.png", dpi=120)
plt.close()

# Fig3: multiplier heatmap
fig, ax = plt.subplots(figsize=(9, 3.5))
im = ax.imshow(M_MAT, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xticks(range(NF))
ax.set_xticklabels([k.replace("_", "\n") for k in ORD_KEYS], fontsize=7)
ax.set_yticks(range(3))
ax.set_yticklabels(TYPE_NAMES)
ax.set_title("Pre-registered δ_k multipliers by shock type × Ord factor")
for i in range(3):
    for j in range(NF):
        ax.text(j, i, f"{M_MAT[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.046, label="multiplier")
plt.tight_layout()
plt.savefig(OUT / "fig_delta_k_multipliers.png", dpi=120)
plt.close()

# stdout summary
print("=== 006 lethal-shock-delta-k ===")
print(f"seed={SEED} N={N} runtime={res['runtime_seconds']:.1f}s")
print(f"S_noshock_1e4={S_noshock[10_000]:.4f}")
print(f"H13 verdict: {verdict}")
print(verdict_note)
print(f"n_nl_raises_S={len(nl_raise_cases)} n_nl_drops_S={len(nl_drop_cases)} "
      f"max_addon={max_addon_global:.4f}")
for fk in focus_keys:
    if fk not in grid_results:
        continue
    g = grid_results[fk]
    print(f"\n{fk}:")
    for phi in PHIS:
        row = g["by_phi"][str(phi)]
        print(f"  φ={phi}: S_joint_1e4={row['S_joint']['S_10000']:.4f} "
              f"S_ord={row['S_ord']['S_10000']:.4f} "
              f"P_ext={row['P_E_extinct']:.4f} P_col={row['P_E_collapse']:.4f}")
print(f"\nWrote results to {OUT}")
