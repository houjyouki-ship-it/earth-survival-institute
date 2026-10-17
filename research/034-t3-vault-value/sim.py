"""研究034：T3 社会にとっての金庫の価値 u と、S 対 T＋C の交差点（open_problems 101）。
033 の模型をそのまま写し、V に u_T3・u_frag を加えた。以下は 033 の docstring：
研究033：シナリオに依存する読者の RAM（q）と、提言4つの Shapley 値（open_problems 99）
事前登録：PREREGISTRATION.md。seed 固定、numpy＋matplotlib のみ。
先頭の乱数生成は 024 と同じ seed・同じ順序にし、GEN0（T0 出発・金庫喪失0・発見1）で 024 の P2 を再現する。
"""
import json, time, sys, platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
import os; HERE = os.path.abspath(HERE)
# ---------- 024 と同一の乱数列（seed 20261024） ----------
rng = np.random.default_rng(20261024)
N = 200_000; H = 1000.0
def logu(lo, hi, n=N, g=None):
    g = rng if g is None else g
    return np.exp(g.uniform(np.log(lo), np.log(hi), n))
G_LO, G_RDH4 = 0.725, 4.45   # 024 sources/text_sizes.json の値（0.725 MB, 4.45 MB）
S24 = json.load(open(f"{HERE}/../024-content-value-reader-guide/sources/text_sizes.json"))
G_LO = S24["wp_total"] / 1e6; G_RDH4 = S24["rdh4"]["utf8"] / 1e6; G_HI = 2 * G_RDH4
P_H = 0.9305; D_MAIN = 1.9
Z = rng.standard_normal((3, N))
T1 = 20 * np.exp(0.7 * Z[0]); T2 = T1 + 30 * np.exp(0.7 * Z[1])
claim = logu(500, 2000); f_store = logu(0.3, 1.0)
life = claim * f_store / np.log(2) ** 0.5 * (-np.log(rng.uniform(size=N))) ** 0.5
g_req = logu(G_LO, G_HI); P_max = rng.uniform(0.3, 0.9, N); kappa = logu(0.5, 2.0)
t_build = 20 * np.exp(0.7 * rng.standard_normal(N))
U_t = rng.uniform(size=N); U_h = rng.uniform(size=N); U_sep = rng.uniform(size=N)
p_sep = rng.uniform(0.1, 0.5, N)
L_core = 50 * 11 / 12 * logu(1, 10); Vv = logu(550, 2400); t_tube = logu(0.5, 5); Uup = logu(3, 30)
N_min = Uup * (L_core + Vv * 3 * t_tube / 250) / (10.0 * logu(0.1, 0.4) * logu(0.005, 0.05))
N_T2 = logu(0.5e6, 5e6); pop_ok = N_min < N_T2
d01, d12 = T1, T2 - T1

# ---------- 本回の乱数（別の生成器、seed 20261033） ----------
SEED = 20261033
g = np.random.default_rng(SEED)
STATES = ["extinct", "T3", "T3_fragile", "T2", "T1", "T0"]
S32 = {}  # 032 の SCEN をそのまま使う（平均の状態割合・喪失・発見・M・識字喪失：すべて【推測】）
exec(open(f"{HERE}/../032-collapse-scenarios/sim.py").read().split("SCEN = ")[1].split("KAPPA_DIR")[0].join(["S32['SCEN'] = ", ""]))
SCEN = S32["SCEN"]
P_SALV = {"NUC_G": 0.3, "NUC_R": 0.7, "BIO": 0.8, "AI": 0.3, "CASC": 0.5, "GEN": 0.0}   # 【推測】
KAPPA_DIR = 20.0
U_state = g.uniform(size=N); U_lost = g.uniform(size=N); U_find = g.uniform(size=N); U_lit = g.uniform(size=N)
U_acc = g.uniform(size=N); W = g.uniform(20, 80, N); p_acc = g.uniform(0.1, 0.6, N)
lit_mult = logu(1, 30, g=g); Zm = g.uniform(size=N)
U_lost2 = g.uniform(size=N); U_find2 = g.uniform(size=N)
SITE2 = {"lost": 0.06, "find": 0.75}
# 読者の RAM
M_B = logu(4 * 1024, 144 * 1024, g=g)                     # 手引きで作る読者（025 の B 級）
f_psram = g.uniform(0.1, 0.4, N); U_ps = g.uniform(size=N)
M_salv = np.where(U_ps < f_psram, logu(4 * 2**20, 16 * 2**20, g=g), logu(2 * 1024, 520 * 1024, g=g))
f_mcu = g.uniform(0.0, 0.5, N); U_mcu = g.uniform(size=N)
U_salv = g.uniform(size=N); L_e = logu(10, 60, g=g); sp_salv = g.uniform(0.2, 0.6, N)
M_C = logu(144 * 1024, 2**20, g=g)                        # 感度：C 級の読者

ANCH_M = np.log([4096, 18432, 65536, 147456, 2**20 + 16384, 9 * 2**20])
ANCH_Q = np.array([0.0, 0.004583300352870422, 0.05644917259355787, 0.17015031414137002, 0.4316764473145267, 1.0])
def q_of(Mr):
    return np.interp(np.log(Mr), ANCH_M, ANCH_Q)
OPT = {"c4_rule": "noisy_or", "reader_B": "B", "f_mcu_on": True, "salv_on": True}
def q_read(Mr, C, mode="star"):
    q = q_of(Mr)
    if mode == "star":
        if not C: return q
        return 1 - (1 - 0.76) * (1 - q) if OPT["c4_rule"] == "noisy_or" else np.maximum(0.76, q)
    qd = np.maximum(0.101, q)                       # 依存重み【推測】
    return 1 - (1 - 0.704) * (1 - qd) if C else qd  # mode=="dep": C は依存枠の有無

def draw_scenario(name):
    s = SCEN[name]; mean = np.array(s["mean"], float); mean /= mean.sum()
    probs = g.dirichlet(KAPPA_DIR * mean, size=N)
    st = (U_state[:, None] > np.cumsum(probs, axis=1)).sum(axis=1).clip(0, 5)
    p_lost = g.uniform(*s["lost"], N); p_find = g.uniform(*s["find"], N)
    M = np.exp(np.log(s["M"][0]) + Zm * (np.log(s["M"][1]) - np.log(s["M"][0])))
    return {"name": name, "probs": probs, "st": st, "p_lost": p_lost, "p_find": p_find, "M": M,
            "lit": s["lit"], "p_salv": P_SALV[name]}

def V(sc, T=False, Hh=False, C=False, S=False, ideal=False, qconst=None, mode="star", detail=False, u_T3=1.0, u_frag=1.0):
    if ideal:
        st = np.full(N, 5); alive = np.ones(N, bool); intact = np.ones(N, bool); Mm = np.ones(N)
        T2s = T2.copy(); salv_ok = np.zeros(N, bool)
    else:
        st = sc["st"]; alive = st != 0; Mm = sc["M"]
        intact = (U_lost > sc["p_lost"]) & (U_find < sc["p_find"])
        if S: intact = intact | ((U_lost2 > SITE2["lost"]) & (U_find2 < SITE2["find"]))
        lit_lost = U_lit < sc["lit"]
        T0s = np.where(lit_lost, d01 * lit_mult, d01) + d12
        acc = (st == 2) & (U_acc < p_acc)
        T2s = np.select([st == 3, st == 4, st == 5, st == 2], [np.zeros(N), d12, T0s, W], default=np.zeros(N))
        salv_ok = (U_salv < sc["p_salv"]) & np.isin(st, [2, 3, 4, 5]) & ~acc & OPT["salv_on"]
        salv_ok &= (T2s + t_build * sp_salv) < L_e
    T3t = T2s + 943 * Mm * np.exp(0.8 * Z[2])
    gen = (T3t < life) & (T3t <= H)
    cov = np.zeros(N); avail = np.zeros(N, bool)
    if Hh:
        hand_ok = U_h < P_H; cov = np.where(hand_ok, np.minimum(1.0, 0.25 * D_MAIN / g_req), cov); avail |= hand_ok
    if T:
        tab_ok = U_sep > p_sep; cov = np.where(tab_ok, 1.0, cov); avail |= tab_ok
    Tt = T2s + np.where(salv_ok, t_build * sp_salv, t_build)
    tgt = avail & pop_ok & (U_t < P_max * cov ** kappa) & (Tt < life) & (Tt <= H)
    MB = M_C if OPT["reader_B"] == "C" else M_B
    Mr = np.where(salv_ok, M_salv, MB)
    qv = np.full(N, qconst) if qconst is not None else q_read(Mr, C, mode)
    v = np.maximum(qv * tgt, gen.astype(float))
    mcu_now = np.zeros(N, bool)
    if not ideal:
        mcu_now = acc & (U_mcu < f_mcu) & OPT["f_mcu_on"]
        v = np.where(acc & ~mcu_now, u_frag, v)
        if qconst is None:
            v = np.where(mcu_now, np.maximum(q_read(M_salv, C, mode), gen), v)
        else:
            v = np.where(mcu_now, 1.0, v)
        v = np.where(st == 1, u_T3, v)
        v = np.where(alive & intact, v, 0.0)
    if detail:
        return v, {"salv_reader": salv_ok & tgt, "mcu_now": mcu_now}
    return v

PL = ["T", "H", "C", "S"]
from itertools import combinations
from math import factorial
def shapley(sc, ideal=False, **kw):
    vals = {}
    for k in range(5):
        for sub in combinations(PL, k):
            vals[frozenset(sub)] = V(sc, T="T" in sub, Hh="H" in sub, C="C" in sub, S="S" in sub, ideal=ideal, **kw)
    n = len(PL); sh = {}
    for p in PL:
        acc_ = np.zeros(N)
        others = [x for x in PL if x != p]
        for k in range(n):
            for sub in combinations(others, k):
                wgt = factorial(k) * factorial(n - k - 1) / factorial(n)
                acc_ += wgt * (vals[frozenset(sub) | {p}] - vals[frozenset(sub)])
        sh[p] = acc_
    return sh, vals

R = {"seed_024_stream": 20261024, "seed": SEED, "N": N, "python": sys.version.split()[0], "numpy": np.__version__,
     "platform": platform.platform(), "P_SALV": P_SALV, "anchors_q": ANCH_Q.tolist()}

# ---------- 検査 ----------
r24 = json.load(open(f"{HERE}/../024-content-value-reader-guide/results.json"))["grid_D1.9"]["m943_q0.3"]
chk = float((V(None, T=True, ideal=True, qconst=0.3) - V(None, ideal=True, qconst=0.3)).mean())
R["check_024"] = {"dV_P2_q0.3": chk, "024": r24["dV"]["P2_tablet"], "reproduced_exactly": abs(chk - r24["dV"]["P2_tablet"]) < 1e-12,
                  "mean_q_readerB": float(q_of(M_B).mean()), "025_readerB_mean_q": 0.03743933823833786}

# ---------- 重み（032 と同じ） ----------
r_g = g.uniform(0.3, 0.7, N)
def weights(anchor):
    if anchor == "A_Ord":
        base = {"AI": 1/10, "BIO": 1/30, "NUC": 1/1000, "CASC": 2/1000, "GEN": 1/30 + 1/50}
    elif anchor == "A_XPT_S":
        bio = 2.13 * 0.01 / 0.38; rest = max(0.0, 9.05 - 2.13 - 4.0 - bio)
        base = {"AI": 2.13, "BIO": bio, "NUC": 4.0, "CASC": rest / 2, "GEN": rest / 2}
    else:
        bio = 12 * 1 / 3; rest = max(0.0, 20 - 12 - 8 - bio)
        base = {"AI": 12, "BIO": bio, "NUC": 8, "CASC": rest / 2, "GEN": rest / 2}
    tot = sum(base.values())
    w = {k: v / tot for k, v in base.items() if k != "NUC"}
    w["NUC_G"] = base["NUC"] / tot * float(np.mean(r_g)); w["NUC_R"] = base["NUC"] / tot * (1 - float(np.mean(r_g)))
    return w
WTS = {a: weights(a) for a in ("A_Ord", "A_XPT_S", "A_XPT_E")}

DRAWS = {name: draw_scenario(name) for name in SCEN}


# ======================= 研究034 本体 =======================
R["seed_034"] = 20261034
R["note"] = "033 の模型（sim_head 部分は 033 の sim.py の写しで、V に u_T3・u_frag を追加）"

def shap_scen(sc, **kw):
    sh, vals = shapley(sc, **kw)
    o = {"shapley": {p: float(sh[p].mean()) for p in PL}}
    o["marg_S_given_TC"] = float((vals[frozenset("TCS")] - vals[frozenset("TC")]).mean())
    o["bundle_TC"] = float((vals[frozenset("TC")] - vals[frozenset()]).mean())
    return o

LIN = {}
for name, sc in DRAWS.items():
    LIN[name] = {"u1": shap_scen(sc, u_T3=1.0, u_frag=1.0), "u0": shap_scen(sc, u_T3=0.0, u_frag=1.0),
                 "u0_frag0": shap_scen(sc, u_T3=0.0, u_frag=0.0)}
# 線形性の確認（BIO と NUC_G で u=0.5）
lin_chk = {}
for name in ("BIO", "NUC_G"):
    h = shap_scen(DRAWS[name], u_T3=0.5, u_frag=1.0)
    pred = 0.5 * (LIN[name]["u0"]["shapley"]["S"] + LIN[name]["u1"]["shapley"]["S"])
    lin_chk[name] = {"S_u0.5": h["shapley"]["S"], "pred": pred, "abs_err": abs(h["shapley"]["S"] - pred)}
R["linearity_check"] = lin_chk
r33 = json.load(open(f"{HERE}/../033-scenario-reader-q/results.json"))["main"]
R["check_033"] = {n: {p: (LIN[n]["u1"]["shapley"][p], r33["scen"][n]["shapley"][p]) for p in PL} for n in DRAWS}
R["check_033_exact"] = all(abs(LIN[n]["u1"]["shapley"][p] - r33["scen"][n]["shapley"][p]) < 1e-12 for n in DRAWS for p in PL)
for p in PL:  # T,H,C が u に依存しないことの確認
    if p != "S":
        R.setdefault("TC_invariant_max_abs", 0.0)
        R["TC_invariant_max_abs"] = max(R["TC_invariant_max_abs"], max(abs(LIN[n]["u1"]["shapley"][p] - LIN[n]["u0_frag0"]["shapley"][p]) for n in DRAWS))

def S_of(name, uT, uf=1.0):
    a0 = LIN[name]["u0"]["shapley"]["S"]; a1 = LIN[name]["u1"]["shapley"]["S"]; f0 = LIN[name]["u0_frag0"]["shapley"]["S"]
    # S(uT,uf) = f0 + uT*(a1 - a0) + uf*(a0 - f0)   （u_T3 と u_frag はそれぞれ線形で、効く結果が重ならない）
    return f0 + uT * (a1 - a0) + uf * (a0 - f0)
def margS_of(name, uT, uf=1.0):
    a0 = LIN[name]["u0"]["marg_S_given_TC"]; a1 = LIN[name]["u1"]["marg_S_given_TC"]; f0 = LIN[name]["u0_frag0"]["marg_S_given_TC"]
    return f0 + uT * (a1 - a0) + uf * (a0 - f0)
TCv = {n: LIN[n]["u1"]["shapley"]["T"] + LIN[n]["u1"]["shapley"]["C"] for n in DRAWS}
BUN = {n: LIN[n]["u1"]["bundle_TC"] for n in DRAWS}

def crossing(fS, ref):
    s0, s1 = fS(0.0), fS(1.0)
    if s0 >= ref: return 0.0
    if s1 < ref: return None
    return (ref - s0) / (s1 - s0)

CROSS = {"scen": {}, "weighted": {}}
for n in DRAWS:
    CROSS["scen"][n] = {"S_u0": S_of(n, 0), "S_u1": S_of(n, 1), "TplusC": TCv[n],
                        "u_star_shapley": crossing(lambda u: S_of(n, u), TCv[n]),
                        "u_star_shapley_fragalso": crossing(lambda u: S_of(n, u, u), TCv[n]),
                        "u_star_marginal": crossing(lambda u: margS_of(n, u), BUN[n]), "bundle_TC": BUN[n]}
for a, w in WTS.items():
    Sw = lambda uT, uf=1.0: sum(w[s] * S_of(s, uT, uf) for s in w)
    Mw = lambda uT, uf=1.0: sum(w[s] * margS_of(s, uT, uf) for s in w)
    TCw = sum(w[s] * TCv[s] for s in w); Bw = sum(w[s] * BUN[s] for s in w)
    CROSS["weighted"][a] = {"S_u0": Sw(0), "S_u1": Sw(1), "S_u0_frag0": Sw(0, 0), "TplusC": TCw, "bundle_TC": Bw,
                            "u_star_shapley": crossing(lambda u: Sw(u), TCw),
                            "u_star_shapley_fragalso": crossing(lambda u: Sw(u, u), TCw),
                            "u_star_marginal": crossing(lambda u: Mw(u), Bw),
                            "S_T3_share_at_u1": (Sw(1) - Sw(0)) / Sw(1)}
R["crossing"] = CROSS

# ---------- u の見積もり（外側の不確かさ、seed 20261034） ----------
gu = np.random.default_rng(20261034)
K = 4000
PREG = {"NUC_G": (0.05, 0.5), "NUC_R": (0.9, 1.0), "BIO": (0.6, 0.98), "AI": (0.1, 0.9), "CASC": (0.4, 0.9), "GEN": (0.3, 0.9)}  # 【推測】
# 拠点（場所は sources/excerpts.json の事実。GitHub 本体の場所は【推測】）
SITES = {"SWH": [("EU", 4)], "GH": [("NA", 1)], "IA": [("NA", 4), ("EU", 1)], "HITS": [("EU", 1), ("NA", 1), ("EG", 1)]}
REG = ["EU", "NA", "EG"]
NIN = 400   # 内側：地域と拠点の生死のくじ（係数1組あたり）
def draw_coef(n):
    return {"c_SWH": gu.uniform(0.8, 0.99, n), "c_GH": gu.uniform(0.85, 0.99, n), "c_IA": gu.uniform(0.0, 0.3, n),
            "c_HITS": gu.uniform(0.2, 0.6, n), "f_orph": np.exp(gu.uniform(np.log(0.003), np.log(0.3), n)),
            "s_site": gu.uniform(0.7, 0.99, n)}
COEF = draw_coef(K)
PR = {s: gu.uniform(*PREG[s], K) for s in PREG}
UR = {s: gu.uniform(size=(K, NIN, len(REG))) for s in PREG}
USITE = {k: gu.uniform(size=(K, NIN, sum(c for _, c in v))) for k, v in SITES.items()}

def u_est(s, rule="indep", orph=True):
    alive = UR[s] < PR[s][:, None, None]                     # (K, NIN, 3)
    A = {}
    for k, lst in SITES.items():
        j = 0; anyok = np.zeros((K, NIN), bool)
        for reg, cnt in lst:
            ri = REG.index(reg)
            ok = (USITE[k][:, :, j:j + cnt] < COEF["s_site"][:, None, None]).any(axis=2) & alive[:, :, ri]
            anyok |= ok; j += cnt
        A[k] = anyok
    cov = {k: COEF["c_" + k][:, None] * A[k] for k in SITES}
    if rule == "indep":
        miss = np.ones((K, NIN))
        for k in SITES: miss = miss * (1 - cov[k])
    else:
        miss = 1 - np.max(np.stack([cov[k] for k in SITES]), axis=0)
    fo = COEF["f_orph"][:, None] if orph else 1.0
    return (fo * miss).mean(axis=1)                          # 係数1組ごとの期待 u（K,）

UEST = {}
for tag, kw in {"main": {}, "nested": {"rule": "nested"}, "no_clone": {"orph": False}, "nested_no_clone": {"rule": "nested", "orph": False}}.items():
    UEST[tag] = {s: u_est(s, **kw) for s in PREG}
def summ(x):
    return {"median": float(np.median(x)), "mean": float(np.mean(x)), "p05": float(np.quantile(x, 0.05)), "p95": float(np.quantile(x, 0.95))}
R["u_estimate"] = {tag: {s: summ(UEST[tag][s]) for s in PREG} for tag in UEST}
R["u_model"] = {"p_region": PREG, "sites": SITES, "K": K, "NIN": NIN}

# 外側の不確かさでの P(S > T+C)：シナリオごとの u を係数1組ごとに入れる
PSG = {}
for tag in UEST:
    PSG[tag] = {}
    for a, w in WTS.items():
        Sk = sum(w[s] * np.array([S_of(s, 0.0)]) + w[s] * UEST[tag][s] * (S_of(s, 1.0) - S_of(s, 0.0)) for s in w)
        Sk_f = sum(w[s] * (S_of(s, 0.0, 0.0) + UEST[tag][s] * (S_of(s, 1.0, 1.0) - S_of(s, 0.0, 0.0))) for s in w)
        TCw = CROSS["weighted"][a]["TplusC"]
        PSG[tag][a] = {"P_S_gt_TC": float(np.mean(Sk > TCw)), "S_median": float(np.median(Sk)), "S_p05": float(np.quantile(Sk, 0.05)),
                       "S_p95": float(np.quantile(Sk, 0.95)), "TplusC": TCw, "S_median_fragalso": float(np.median(Sk_f)),
                       "P_S_gt_TC_fragalso": float(np.mean(Sk_f > TCw)), "S_over_TC_median": float(np.median(Sk) / TCw)}
R["S_vs_TC_under_u"] = PSG

# ---------- 仮説 ----------
CW = CROSS["weighted"]; UE = R["u_estimate"]["main"]
ustar_ok = lambda a: (CW[a]["u_star_shapley"] is not None) and CW[a]["u_star_shapley"] <= 0.2
med = {s: UE[s]["median"] for s in UE}
Smed_main = PSG["main"]["A_XPT_S"]["S_median"]; Smed_frag = PSG["main"]["A_XPT_S"]["S_median_fragalso"]
R["hypotheses"] = {
 "H46a": {"value": {"S_u1": CW["A_XPT_S"]["S_u1"], "S_u0": CW["A_XPT_S"]["S_u0"], "TplusC": CW["A_XPT_S"]["TplusC"]},
          "supported": CW["A_XPT_S"]["S_u1"] > CW["A_XPT_S"]["TplusC"] > CW["A_XPT_S"]["S_u0"]},
 "H46b": {"value": {a: CW[a]["u_star_shapley"] for a in CW}, "supported": all(ustar_ok(a) for a in CW)},
 "H46c": {"value": med, "supported": all(v < 0.1 for v in med.values())},
 "H46d": {"value": PSG["main"]["A_XPT_S"]["P_S_gt_TC"], "supported": PSG["main"]["A_XPT_S"]["P_S_gt_TC"] < 0.5},
 "H46e": {"value": {"NUC_G": med["NUC_G"], "BIO": med["BIO"], "ratio": med["NUC_G"] / med["BIO"]}, "supported": med["NUC_G"] >= 3 * med["BIO"]},
 "H46f": {"value": {"main": Smed_main, "fragalso": Smed_frag, "drop": 1 - Smed_frag / Smed_main}, "supported": (1 - Smed_frag / Smed_main) >= 0.2},
}

# ---------- 探索（事前登録外、初回実行後に追加。判定には使わない） ----------
EXP = {"marg_S_given_TC_u0": {}, "S_alone_vs_TC_alone": {}, "f_orph_breakeven": {}}
for name, sc in DRAWS.items():
    for uT in (0.0, 1.0):
        vS = V(sc, S=True, u_T3=uT); v0 = V(sc, u_T3=uT)
        EXP["S_alone_vs_TC_alone"].setdefault(name, {})[f"S_alone_u{int(uT)}"] = float((vS - v0).mean())
    EXP["S_alone_vs_TC_alone"][name]["TC_alone"] = BUN[name]
    EXP["marg_S_given_TC_u0"][name] = margS_of(name, 0.0)
for a, w in WTS.items():
    sa0 = sum(w[s] * EXP["S_alone_vs_TC_alone"][s]["S_alone_u0"] for s in w); sa1 = sum(w[s] * EXP["S_alone_vs_TC_alone"][s]["S_alone_u1"] for s in w)
    Bw = CROSS["weighted"][a]["bundle_TC"]
    EXP["S_alone_vs_TC_alone"][a] = {"S_alone_u0": sa0, "S_alone_u1": sa1, "TC_alone": Bw,
                                     "u_star_alone": crossing(lambda u: sa0 + u * (sa1 - sa0), Bw)}
    EXP["marg_S_given_TC_u0"][a] = sum(w[s] * margS_of(s, 0.0) for s in w)
    # u = f_orph × miss。miss を clone なしの u の中央値に固定し、Shapley(S) = T+C となる f_orph を解く
    m = {s: R["u_estimate"]["no_clone"][s]["median"] for s in w}
    num = CROSS["weighted"][a]["TplusC"] - sum(w[s] * S_of(s, 0.0) for s in w)
    den = sum(w[s] * m[s] * (S_of(s, 1.0) - S_of(s, 0.0)) for s in w)
    EXP["f_orph_breakeven"][a] = num / den if den > 0 else None
R["explore"] = EXP

# ---------- 図 ----------
us = np.linspace(0, 1, 101)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for i, a in enumerate(WTS):
    w = WTS[a]
    ax[i].plot(us, [sum(w[s] * S_of(s, u) for s in w) for u in us], label="Shapley(S), u_frag=1")
    ax[i].plot(us, [sum(w[s] * S_of(s, u, u) for s in w) for u in us], ls="--", label="Shapley(S), u_frag=u")
    ax[i].axhline(CW[a]["TplusC"], color="k", label="Shapley(T)+Shapley(C)")
    lo, hi = np.quantile(np.concatenate([UEST["main"][s] for s in PREG]), [0.05, 0.95])
    ax[i].axvspan(0, float(np.quantile(np.concatenate([UEST["no_clone"][s] for s in PREG]), 0.95)), color="C2", alpha=0.08, label="u est. 5-95% (no clone)")
    ax[i].axvspan(lo, hi, color="C3", alpha=0.25, label="u est. 5-95% (main)")
    ax[i].set_xlabel("u = value vault adds to a surviving T3 society"); ax[i].set_ylabel("weighted Shapley"); ax[i].set_title(a)
    ax[i].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_crossing_u.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
names = list(PREG); x = np.arange(len(names)); wd = 0.2
for i, tag in enumerate(UEST):
    m = [R["u_estimate"][tag][s]["median"] for s in names]
    lo_ = [m[j] - R["u_estimate"][tag][s]["p05"] for j, s in enumerate(names)]
    hi_ = [R["u_estimate"][tag][s]["p95"] - m[j] for j, s in enumerate(names)]
    ax[0].bar(x + (i - 1.5) * wd, m, wd, yerr=[lo_, hi_], capsize=2, label=tag)
ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(names); ax[0].set_ylabel("u (median, 5-95%)")
ax[0].set_title("estimated value the vault adds to a T3 society"); ax[0].legend(fontsize=8)
for j, s in enumerate(names):
    v = CROSS["scen"][s]["u_star_shapley"]
    ax[1].bar(j, v if v is not None else 1.05, color="C0" if v is not None else "0.6")
ax[1].set_xticks(x); ax[1].set_xticklabels(names); ax[1].set_ylabel("u* where Shapley(S) = T + C (grey: S < T+C even at u=1)")
ax[1].set_title("crossing point by scenario")
plt.tight_layout(); plt.savefig(f"{HERE}/fig_u_estimate_and_ustar.png", dpi=120); plt.close()

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
print(json.dumps({"check_024": R["check_024"], "check_033_exact": R["check_033_exact"], "TC_invariant_max_abs": R["TC_invariant_max_abs"],
                  "linearity": lin_chk, "crossing": CROSS, "u_estimate": R["u_estimate"], "S_vs_TC": PSG, "explore": EXP, "hyp": R["hypotheses"]},
                 ensure_ascii=False, indent=1, default=str))
print("runtime", round(R["runtime_seconds"], 1))
