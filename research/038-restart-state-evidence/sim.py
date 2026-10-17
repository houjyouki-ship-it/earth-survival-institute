"""研究038：再出発の状態の外部根拠（open_problems 98、next_plan の固定の最優先指示）
事前登録：PREREGISTRATION.md。seed 固定、numpy＋matplotlib のみ。ネット接続なし。
先頭は 032 の sim.py の 6〜172 行をそのまま写し、032 の乱数列・シナリオ別の値・重み付き値を完全に再現する。
その後、別の生成器（seed 20261038）で、生存者の数・最大のつながった集団・技術段の最小人口・人口の回復を引き、
状態と T2 到達時刻を置き換えて 032 と同じ価値の式で評価する。
"""
# ===================== 032 の写し（ここから） =====================
import json, time, sys, platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
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

# ---------- 本回の乱数（別の生成器、seed 20261032） ----------
SEED = 20261032
g = np.random.default_rng(SEED)
STATES = ["extinct", "T3", "T3_fragile", "T2", "T1", "T0"]
# 平均の状態確率・金庫喪失・発見・m23 倍率・T0 での識字喪失確率：すべて【推測】
SCEN = {
 "NUC_G": {"mean": [0.01, 0.65, 0.15, 0.14, 0.04, 0.01], "lost": (0.05, 0.30), "find": (0.2, 0.6), "M": (1, 2), "lit": 0.2,
           "note": "Xia 2022: >5e9 deaths, 150 Tg, crops -90%, Australia/NZ least affected; vault in Svalbard (high N latitude)"},
 "NUC_R": {"mean": [0.001, 0.93, 0.05, 0.02, 0.001, 0.001], "lost": (0.01, 0.05), "find": (0.5, 0.9), "M": (1, 1), "lit": 0.1,
           "note": "Xia 2022: >2e9 deaths, 5-47 Tg"},
 "BIO":   {"mean": [0.15, 0.40, 0.20, 0.12, 0.08, 0.05], "lost": (0.01, 0.05), "find": (0.3, 0.9), "M": (1, 3), "lit": 0.5,
           "note": "infrastructure intact, people lost; small survivor population slows rebuild"},
 "AI":    {"mean": [0.80, 0.08, 0.04, 0.05, 0.02, 0.01], "lost": (0.2, 0.8), "find": (0.3, 0.9), "M": (1, 1), "lit": 0.3,
           "note": "mostly extinction / permanent disempowerment; vault may be destroyed"},
 "CASC":  {"mean": [0.02, 0.35, 0.20, 0.23, 0.12, 0.08], "lost": (0.05, 0.30), "find": (0.3, 0.9), "M": (1, 3), "lit": 0.5,
           "note": "compound / slow collapse (climate, war, pandemics); easy resources depleted"},
 "GEN":   {"mean": [0.05, 0.001, 0.001, 0.001, 0.001, 0.946], "lost": (0.05, 0.30), "find": (0.3, 0.9), "M": (1, 1), "lit": 0.3,
           "note": "unknown / other risks: the old single assumption (restart from T0)"},
}
KAPPA_DIR = 20.0
U_state = g.uniform(size=N); U_lost = g.uniform(size=N); U_find = g.uniform(size=N); U_lit = g.uniform(size=N)
U_acc = g.uniform(size=N); W = g.uniform(20, 80, N); p_acc = g.uniform(0.1, 0.6, N)
lit_mult = logu(1, 30, g=g); Zm = g.uniform(size=N)
# 第2拠点（南半球）の独立な乱数（探索用）
U_lost2 = g.uniform(size=N); U_find2 = g.uniform(size=N)

def draw_scenario(name, mean_override=None):
    s = SCEN[name]; mean = np.array(mean_override if mean_override is not None else s["mean"], float); mean /= mean.sum()
    probs = g.dirichlet(KAPPA_DIR * mean, size=N)
    cum = np.cumsum(probs, axis=1)
    st = (U_state[:, None] > cum).sum(axis=1).clip(0, 5)
    p_lost = g.uniform(*s["lost"], N); p_find = g.uniform(*s["find"], N)
    M = np.exp(np.log(s["M"][0]) + Zm * (np.log(s["M"][1]) - np.log(s["M"][0])))
    return {"probs": probs, "st": st, "p_lost": p_lost, "p_find": p_find, "M": M}

def V_s(sc, qv, g_hand=0.0, tablet=False, site2=False, ideal=False):
    if ideal:  # 024 の想定：T0 出発・金庫は残り必ず見つかる・識字あり
        alive = np.ones(N, bool); T2s = T2.copy(); M = np.ones(N); st = np.full(N, 5)
        intact = np.ones(N, bool)
    else:
        st = sc["st"]; alive = st != 0
        intact = (U_lost > sc["p_lost"]) & (U_find < sc["p_find"])
        if site2:
            intact = intact | ((U_lost2 > SITE2["lost"]) & (U_find2 < SITE2["find"]))
        lit_lost = U_lit < SCEN_LIT
        T0s = np.where(lit_lost, d01 * lit_mult, d01) + d12
        T2s = np.select([st == 3, st == 4, st == 5, st == 2], [np.zeros(N), d12, T0s, W], default=np.zeros(N))
        M = sc["M"]
    T3t = T2s + 943 * M * np.exp(0.8 * Z[2])
    gen = (T3t < life) & (T3t <= H)
    cov = np.zeros(N); avail = np.zeros(N, bool)
    if g_hand > 0:
        hand_ok = U_h < P_H; cov = np.where(hand_ok, np.minimum(1.0, g_hand / g_req), cov); avail |= hand_ok
    if tablet:
        tab_ok = U_sep > p_sep; cov = np.where(tab_ok, 1.0, cov); avail |= tab_ok
    Tt = T2s + t_build
    tgt = avail & pop_ok & (U_t < P_max * cov ** kappa) & (Tt < life) & (Tt <= H)
    v = np.maximum(qv * tgt, gen.astype(float))
    if not ideal:
        t3_now = (st == 1) | ((st == 2) & (U_acc < p_acc))
        v = np.where(t3_now, 1.0, v)
        v = np.where(alive & intact, v, 0.0)
    return v, gen, tgt

def recs(sc, ideal=False):
    P0 = V_s(sc, 0.3, ideal=ideal)[0]; P2 = V_s(sc, 0.3, tablet=True, ideal=ideal)[0]
    P3 = V_s(sc, 0.3, g_hand=0.25 * D_MAIN, tablet=True, ideal=ideal)[0]
    c4 = V_s(sc, 0.76, tablet=True, ideal=ideal)[0] - V_s(sc, 0.031, tablet=True, ideal=ideal)[0]
    dep = V_s(sc, 0.704, tablet=True, ideal=ideal)[0] - V_s(sc, 0.101, tablet=True, ideal=ideal)[0]
    per = {"R_tab": P2 - P0, "R_hand25": P3 - P2, "R_c4": c4, "R_dep": dep, "V_P0": P0}
    return per

R = {"seed_024_stream": 20261024, "seed": SEED, "N": N, "python": sys.version.split()[0], "platform": platform.platform(),
     "SCEN": {k: {kk: vv for kk, vv in v.items()} for k, v in SCEN.items()}, "states": STATES}

# ---------- 検査：024 の P2 を再現 ----------
SCEN_LIT = np.zeros(N)
chk = recs(None, ideal=True)
r24 = json.load(open(f"{HERE}/../024-content-value-reader-guide/results.json"))["grid_D1.9"]["m943_q0.3"]
R["check_024"] = {"dV_P2": float(chk["R_tab"].mean()), "024_dV_P2": r24["dV"]["P2_tablet"],
                  "dV_P3_minus_P2": float(chk["R_hand25"].mean()), "024_P3_minus_P2": r24["dV_P3_minus_P2"],
                  "reproduced_exactly": bool(abs(chk["R_tab"].mean() - r24["dV"]["P2_tablet"]) < 1e-12 and abs(chk["R_hand25"].mean() - r24["dV_P3_minus_P2"]) < 1e-12),
                  "dV_c4": float(chk["R_c4"].mean()), "dV_dep": float(chk["R_dep"].mean())}

# ---------- シナリオ別 ----------
SITE2 = {"lost": 0.06, "find": 0.75}   # 探索用の第2拠点（南半球）【推測】
DRAWS = {}; PER = {}; SC_OUT = {}
for name in SCEN:
    sc = draw_scenario(name); DRAWS[name] = sc
    SCEN_LIT = np.full(N, SCEN[name]["lit"])
    per = recs(sc); PER[name] = per
    # 探索：南半球の第2拠点（NUC_G では南の生存者が見つけやすい：find 0.75；他も同じ値で置く）
    v2 = V_s(sc, 0.3, site2=True)[0]
    st = sc["st"]
    SC_OUT[name] = {k: float(v.mean()) for k, v in per.items()}
    SC_OUT[name]["state_shares"] = {STATES[i]: float(np.mean(st == i)) for i in range(6)}
    SC_OUT[name]["explore_site2_dV_P0"] = float(v2.mean() - per["V_P0"].mean())
    # 探索（事前登録外、初回実行後に追加）：T3 が残る結果を除いた（再建する社会だけの）第2拠点の価値
    rebuild = np.isin(st, [3, 4, 5]) | ((st == 2) & (U_acc >= p_acc))
    SC_OUT[name]["explore_site2_dV_P0_rebuild_only"] = float(np.mean((v2 - per["V_P0"]) * rebuild))
    SC_OUT[name]["explore_tab_dV_rebuild_only"] = float(np.mean(per["R_tab"] * rebuild))
    SC_OUT[name]["ratio_c4_over_tab"] = SC_OUT[name]["R_c4"] / SC_OUT[name]["R_tab"] if SC_OUT[name]["R_tab"] > 0 else None
    SC_OUT[name]["ratio_tab_over_GEN0"] = SC_OUT[name]["R_tab"] / R["check_024"]["dV_P2"]
R["scenarios"] = SC_OUT

# ---------- 重み ----------
r_g = g.uniform(0.3, 0.7, N)   # 核のうち全面戦争の割合【推測】
def weights(anchor):
    if anchor == "A_Ord":
        base = {"AI": 1/10, "BIO": 1/30, "NUC": 1/1000, "CASC": 2/1000, "GEN": 1/30 + 1/50}
    elif anchor == "A_XPT_S":
        bio = 2.13 * 0.01 / 0.38; rest = max(0.0, 9.05 - 2.13 - 4.0 - bio)
        base = {"AI": 2.13, "BIO": bio, "NUC": 4.0, "CASC": rest / 2, "GEN": rest / 2}
    elif anchor == "A_XPT_E":
        bio = 12 * 1 / 3; rest = max(0.0, 20 - 12 - 8 - bio)
        base = {"AI": 12, "BIO": bio, "NUC": 8, "CASC": rest / 2, "GEN": rest / 2}
    tot = sum(base.values())
    w = {k: v / tot for k, v in base.items() if k != "NUC"}
    w["NUC_G"] = base["NUC"] / tot * r_g; w["NUC_R"] = base["NUC"] / tot * (1 - r_g)
    return w, base
WT = {}
for a in ("A_Ord", "A_XPT_S", "A_XPT_E"):
    w, base = weights(a)
    out = {"base": base, "mean_weights": {k: float(np.mean(v)) for k, v in w.items()}}
    for rk in ("R_tab", "R_hand25", "R_c4", "R_dep"):
        contrib = {s: w[s] * PER[s][rk] for s in w}
        tot = sum(contrib.values())
        out[rk] = float(np.mean(tot))
        out[rk + "_share_by_scen"] = {s: float(np.mean(c) / np.mean(tot)) for s, c in contrib.items()}
        out[rk + "_ratio_vs_GEN0"] = float(np.mean(tot) / chk[rk].mean())
    out["explore_site2_dV_P0"] = float(sum(np.mean(w[s]) * SC_OUT[s]["explore_site2_dV_P0"] for s in w))
    out["explore_site2_dV_P0_rebuild_only"] = float(sum(np.mean(w[s]) * SC_OUT[s]["explore_site2_dV_P0_rebuild_only"] for s in w))
    out["decomp_ratio_vs_GEN"] = out["R_tab"] / SC_OUT["GEN"]["R_tab"]
    out["_per_sample_tab"] = sum(w[s] * PER[s]["R_tab"] for s in w)
    out["_w"] = w
    WT[a] = out

# ===================== 032 の写し（ここまで） =====================

# ===================== 038 =====================
R["study"] = "038-restart-state-evidence"; R["seed_038"] = 20261038
R.pop("SCEN", None)
r32 = json.load(open(f"{HERE}/../032-collapse-scenarios/results.json"))
# ---------- 検査1：032 の再現 ----------
chk32 = {s: {"R_tab": SC_OUT[s]["R_tab"], "032": r32["scenarios"][s]["R_tab"]} for s in SCEN}
chk32w = {a: {"R_tab": WT[a]["R_tab"], "032": r32["weighted"][a]["R_tab"]} for a in WT}
R["check_032"] = {"per_scenario": chk32, "weighted": chk32w,
                  "reproduced_exactly": bool(all(abs(v["R_tab"] - v["032"]) < 1e-12 for v in chk32.values()) and
                                             all(abs(v["R_tab"] - v["032"]) < 1e-12 for v in chk32w.values()))}

# ---------- 一般化した価値の式（状態 st と T2 到達時刻 T2s を外から与える） ----------
def V_gen(st, T2s, sc, qv, g_hand=0.0, tablet=False):
    alive = st != 0
    intact = (U_lost > sc["p_lost"]) & (U_find < sc["p_find"])
    M = sc["M"]
    T3t = T2s + 943 * M * np.exp(0.8 * Z[2])
    gen = (T3t < life) & (T3t <= H)
    cov = np.zeros(N); avail = np.zeros(N, bool)
    if g_hand > 0:
        hand_ok = U_h < P_H; cov = np.where(hand_ok, np.minimum(1.0, g_hand / g_req), cov); avail |= hand_ok
    if tablet:
        tab_ok = U_sep > p_sep; cov = np.where(tab_ok, 1.0, cov); avail |= tab_ok
    Tt = T2s + t_build
    tgt = avail & pop_ok & (U_t < P_max * cov ** kappa) & (Tt < life) & (Tt <= H)
    v = np.maximum(qv * tgt, gen.astype(float))
    t3_now = (st == 1) | ((st == 2) & (U_acc < p_acc))
    v = np.where(t3_now, 1.0, v)
    return np.where(alive & intact, v, 0.0)

def T2s_032(st, lit):
    lit_lost = U_lit < lit
    T0s = np.where(lit_lost, d01 * lit_mult, d01) + d12
    return np.select([st == 3, st == 4, st == 5, st == 2], [np.zeros(N), d12, T0s, W], default=np.zeros(N))

def recs_gen(st, T2s, sc, T2s_primer=None):
    P0 = V_gen(st, T2s, sc, 0.3); P2 = V_gen(st, T2s, sc, 0.3, tablet=True)
    out = {"R_tab": P2 - P0,
           "R_c4": V_gen(st, T2s, sc, 0.76, tablet=True) - V_gen(st, T2s, sc, 0.031, tablet=True),
           "R_dep": V_gen(st, T2s, sc, 0.704, tablet=True) - V_gen(st, T2s, sc, 0.101, tablet=True), "V_P0": P0, "V_P2": P2}
    if T2s_primer is not None:
        P2p = V_gen(st, T2s_primer, sc, 0.3, tablet=True); P0p = V_gen(st, T2s_primer, sc, 0.3)
        out["R_primer_on_tab"] = P2p - P2          # 板の手引きに T0 手引きを足した増分
        out["R_primer_alone"] = P0p - P0           # 板の手引きなしで T0 手引きだけ
    return out

# ---------- 検査2：一般化した式に 032 の状態を入れると 032 と一致 ----------
chk_gen = {}
for s in SCEN:
    st = DRAWS[s]["st"]; v = recs_gen(st, T2s_032(st, SCEN[s]["lit"]), DRAWS[s])
    chk_gen[s] = float(abs(v["R_tab"].mean() - PER[s]["R_tab"].mean()))
R["check_generalized_eq_032"] = {"max_abs_diff": max(chk_gen.values()), "ok": bool(max(chk_gen.values()) < 1e-12)}

# ---------- 038 の乱数 ----------
g2 = np.random.default_rng(20261038)
POP_W = 8.232e9                      # 【事実】World 8,232,000,000（Wikipedia の国別人口一覧、UN projection 2025-06-13）
N_Fe = logu(2e3, 1e5, g=g2)
N_lens = np.maximum(logu(3e4, 3e6, g=g2), N_Fe)
N_T2e = np.maximum(N_T2, N_lens)     # N_T2 は 024 と同じ配列（pop_ok は変えない）
N_3f = np.maximum(logu(1e5, 1e7, g=g2), N_T2e)
N_3 = np.maximum(logu(1e7, 5e8, g=g2), N_3f)
r_pop = logu(0.0014, 0.0061, g=g2)   # 黒死病からの回復で較正【推測：損失30〜60%、150〜250年】
d_Fe = 20 * np.exp(0.7 * g2.standard_normal(N))
f_primer = g2.uniform(0.3, 0.8, N)
U_sv = g2.uniform(size=(8, N))

def survivors(name, var=None):
    var = var or {}
    u = U_sv
    if name == "NUC_G":
        s = np.exp(np.log(0.03) + u[0] * (np.log(0.19) - np.log(0.03)))
        au = 27.92e6 * (0.5 + 0.5 * u[1]) * (0.0 if var.get("no_australia") else 1.0)
        alt = np.exp(np.log(4.647e7) + u[2] * (np.log(2.883e8) - np.log(4.647e7))) * (0.02 + 0.23 * u[3])
        nmax = np.maximum(au, alt); mcap = 1 + 2 * u[4]
    elif name == "NUC_R":
        s = 0.60 + (0.958 - 0.60) * u[0]
        nmax = s * np.exp(np.log(3.4e8) + u[1] * (np.log(4e9) - np.log(3.4e8))); mcap = np.ones(N)
    elif name == "BIO":
        a = 0.3 + 0.65 * u[0]; c = 0.3 + 0.7 * u[1]; s = 1 - a * c
        p_tail = 0.0 if var.get("no_tail") else 0.3 * u[2]
        tail = u[3] < p_tail
        s = np.where(tail, np.exp(np.log(1e-7) + u[4] * (np.log(1e-2) - np.log(1e-7))), s)
        nmax = s * np.exp(np.log(3e8) + u[5] * (np.log(8.232e9) - np.log(3e8))); mcap = np.ones(N)
    elif name == "CASC":
        s = 0.2 + 0.7 * u[0]
        nmax = s * np.exp(np.log(1e6) + u[1] * (np.log(3e8) - np.log(1e6))); mcap = 1 + 2 * u[2]
    ntot = s * POP_W
    return {"s": s, "ntot": ntot, "nmax": np.minimum(nmax, ntot), "mcap": mcap}

def assign(sv, thr_mult=1.0):
    n = sv["nmax"]; k = thr_mult
    st = np.full(N, 6)
    st = np.where(n >= k * N_Fe, 5, st); st = np.where(n >= k * N_lens, 4, st); st = np.where(n >= k * N_T2e, 3, st)
    st = np.where(n >= k * N_3f, 2, st); st = np.where(n >= k * N_3 * sv["mcap"], 1, st)
    st = np.where(sv["ntot"] < 5000, 0, st)
    return st

def times(st, sv, lit, primer=False, r_mult=1.0, thr_mult=1.0, no_wait=False):
    n = np.maximum(sv["nmax"], 1.0); r = r_pop * r_mult; k = thr_mult
    wait = (lambda Nk: np.zeros(N)) if no_wait else (lambda Nk: np.maximum(0.0, np.log(k * Nk / n) / r))
    fp = f_primer if primer else 1.0
    d01e = np.where(U_lit < lit, d01 * lit_mult, d01) * fp
    dFe = d_Fe * fp
    tFe = np.where(st == 6, np.maximum(dFe, wait(N_Fe)), 0.0)
    t1 = np.where(st >= 5, np.maximum(tFe + d01e, wait(N_lens)), 0.0)
    t2 = np.where(st >= 4, np.maximum(t1 + d12, wait(N_T2e)), 0.0)
    stage = np.where(st == 6, dFe, 0.0) + np.where(st >= 5, d01e, 0.0) + np.where(st >= 4, d12, 0.0)
    T2s = np.select([st == 2, st == 3, st >= 4], [W, np.zeros(N), t2], default=np.zeros(N))
    return T2s, t2, stage

EVID = ["NUC_G", "NUC_R", "BIO", "CASC"]
NAMES7 = ["extinct", "T3", "T3_fragile", "T2", "T1", "T0_iron", "T00_pre_iron"]
HEAVY = (4, 5, 6)

def run_all(var=None, thr_mult=1.0, r_mult=1.0, primer=True):
    var = var or {}
    out = {}; per = {}
    for s in SCEN:
        lit = SCEN[s]["lit"]
        if s in EVID:
            sv = survivors(s, var); st = assign(sv, thr_mult)
            T2s, t2, stage = times(st, sv, lit, r_mult=r_mult, thr_mult=thr_mult)
            T2p = times(st, sv, lit, primer=True, r_mult=r_mult, thr_mult=thr_mult)[0] if primer else None
        else:   # AI・GEN は 032 のまま（外部根拠なし／定義）。T0 を 7 状態の T0_iron と同じ 5 で扱う
            sv = None; st = DRAWS[s]["st"]; T2s = T2s_032(st, lit); t2 = T2s; stage = T2s
            if primer:
                lit_lost = U_lit < lit
                T2p = np.select([st == 3, st == 4, st == 5, st == 2],
                                [np.zeros(N), d12, np.where(lit_lost, d01 * lit_mult, d01) * f_primer + d12, W], default=np.zeros(N))
            else:
                T2p = None
        rc = recs_gen(st, T2s, DRAWS[s], T2p)
        per[s] = rc
        heavy = np.isin(st, HEAVY); light = np.isin(st, (1, 2)); mid = st == 3
        o = {k: float(v.mean()) for k, v in rc.items()}
        o["state_shares"] = {NAMES7[i]: float(np.mean(st == i)) for i in range(7)}
        o["heavy_share"] = float(heavy.mean()); o["light_share"] = float(light.mean())
        o["R_tab_from_light"] = float(np.mean(rc["R_tab"] * light)); o["R_tab_from_T2"] = float(np.mean(rc["R_tab"] * mid))
        o["R_tab_from_heavy"] = float(np.mean(rc["R_tab"] * heavy))
        if primer:
            o["R_primer_on_tab_from_heavy"] = float(np.mean(rc["R_primer_on_tab"] * heavy))
            o["primer_gain_over_tab_heavy"] = (o["R_primer_on_tab_from_heavy"] / o["R_tab_from_heavy"]) if o["R_tab_from_heavy"] > 0 else None
        if s in EVID:
            hv = heavy & (t2 > 0)
            share = np.where(t2 > 0, (t2 - stage) / np.maximum(t2, 1e-9), 0.0)
            o["wait_share_median_heavy"] = float(np.median(share[hv])) if hv.any() else None
            o["t2_median_heavy"] = float(np.median(t2[hv])) if hv.any() else None
            o["stage_median_heavy"] = float(np.median(stage[hv])) if hv.any() else None
            o["nmax_quantiles"] = [float(x) for x in np.percentile(sv["nmax"], [5, 50, 95])]
            o["surv_share_quantiles"] = [float(x) for x in np.percentile(sv["s"], [5, 50, 95])]
            o["_heavy_mask"] = hv; o["_share"] = share; o["_t2"] = t2; o["_stage"] = stage
        out[s] = o
    wt = {}
    for a in WT:
        w = WT[a]["_w"]
        res = {}
        for rk in ("R_tab", "R_c4", "R_dep", "R_primer_on_tab") if primer else ("R_tab", "R_c4", "R_dep"):
            res[rk] = float(np.mean(sum(w[s] * per[s][rk] for s in w)))
        for grp, msk in (("light", (1, 2)), ("T2", (3,)), ("heavy", HEAVY)):
            res["R_tab_from_" + grp] = float(np.mean(sum(w[s] * per[s]["R_tab"] * np.isin(
                (assign(survivors(s, var), thr_mult) if s in EVID else DRAWS[s]["st"]), msk) for s in w)))
        res["heavy_share_weighted"] = float(sum(np.mean(w[s]) * out[s]["heavy_share"] for s in w))
        res["ratio_R_tab_vs_032"] = res["R_tab"] / r32["weighted"][a]["R_tab"]
        res["ratio_R_tab_vs_GEN0"] = res["R_tab"] / R["check_024"]["dV_P2"]
        wt[a] = res
    return out, wt

BASE, BASE_W = run_all()
def strip(d):
    return {s: {k: v for k, v in o.items() if not k.startswith("_")} for s, o in d.items()}
R["scenarios_038"] = strip(BASE); R["weighted_038"] = BASE_W
R["scenarios_032_state_shares"] = {s: r32["scenarios"][s]["state_shares"] for s in SCEN}
R["thresholds_quantiles"] = {k: [float(x) for x in np.percentile(v, [5, 50, 95])] for k, v in
                             {"N_Fe": N_Fe, "N_lens": N_lens, "N_T2e": N_T2e, "N_3f": N_3f, "N_3": N_3, "r_pop": r_pop}.items()}

# ---------- 感度（同じ乱数で入れ替え） ----------
SENS = {"r_x3": dict(r_mult=3.0), "thresholds_x0.3": dict(thr_mult=0.3), "thresholds_x3": dict(thr_mult=3.0),
        "NUC_G_no_australia": dict(var={"no_australia": True}), "BIO_no_tail": dict(var={"no_tail": True})}
R["sensitivity"] = {}
for k, kw in SENS.items():
    o, w = run_all(primer=True, **kw)
    R["sensitivity"][k] = {"heavy_share": {s: o[s]["heavy_share"] for s in EVID},
                           "weighted_R_tab": {a: w[a]["R_tab"] for a in w}, "weighted_R_tab_from_heavy": {a: w[a]["R_tab_from_heavy"] for a in w},
                           "wait_share_median_heavy": {s: o[s].get("wait_share_median_heavy") for s in EVID},
                           "primer_gain_over_tab_heavy": {s: o[s].get("primer_gain_over_tab_heavy") for s in SCEN}}
# 人口待ちを消した変種（知識だけが律速の場合）
R["no_pop_wait"] = {}
for s in EVID:
    sv = survivors(s); st = assign(sv); lit = SCEN[s]["lit"]
    T2a, _, _ = times(st, sv, lit, no_wait=True); T2b, _, _ = times(st, sv, lit, primer=True, no_wait=True)
    a_ = recs_gen(st, T2a, DRAWS[s], T2b); hv = np.isin(st, HEAVY)
    R["no_pop_wait"][s] = {"R_tab": float(a_["R_tab"].mean()), "R_tab_from_heavy": float(np.mean(a_["R_tab"] * hv)),
                           "primer_gain_over_tab_heavy": float(np.mean(a_["R_primer_on_tab"] * hv) / max(np.mean(a_["R_tab"] * hv), 1e-15))}

# ---------- Kline & Boyd 2010 表1 ----------
KB = [("Malekula", 1100, "low", 13, 3.2), ("Tikopia", 1500, "low", 22, 4.7), ("Santa Cruz", 3600, "low", 24, 4.0),
      ("Yap", 4791, "high", 43, 5.0), ("Lau Fiji", 7400, "high", 33, 5.0), ("Trobriand", 8000, "high", 19, 4.0),
      ("Chuuk", 9200, "high", 40, 3.8), ("Manus", 13000, "low", 28, 6.6), ("Tonga", 17500, "high", 55, 5.4),
      ("Hawaii", 275000, "low", 71, 6.6)]
lp = np.log([k[1] for k in KB]); lt = np.log([k[3] for k in KB]); lc = np.log([k[4] for k in KB])
def slope(x, y): return float(np.polyfit(x, y, 1)[0])
b_tools = slope(lp, lt); b_tu = slope(lp, lc)
std_beta = float(np.corrcoef(lp, lt)[0, 1])
g3 = np.random.default_rng(20261039); bs = []
for _ in range(20000):
    i = g3.integers(0, 10, 10)
    if np.ptp(lp[i]) > 0: bs.append(slope(lp[i], lt[i]))
bs = np.array(bs)
noH = np.array([k[0] != "Hawaii" for k in KB])
R["kline_boyd"] = {"b_log_tools_on_log_pop": b_tools, "b_ci90_bootstrap": [float(x) for x in np.percentile(bs, [5, 95])],
                   "std_beta_check_vs_paper_0.805": std_beta, "b_log_TU_on_log_pop": b_tu,
                   "b_without_Hawaii": slope(lp[noH], lt[noH]),
                   "toolkit_ratio_pop_100x_smaller": float(100 ** (-b_tools)),
                   "toolkit_ratio_Tasmania_3k_15k_vs_Hawaii": [float((3000 / 275000) ** b_tools), float((15000 / 275000) ** b_tools)]}

# ---------- 仮説 ----------
hs = lambda s: BASE[s]["heavy_share"]
h32 = lambda s: sum(r32["scenarios"][s]["state_shares"][k] for k in ("T1", "T0"))
heavy_wait = np.concatenate([BASE[s]["_share"][BASE[s]["_heavy_mask"]] for s in EVID])
heavy_wait_w = None
prim = {s: BASE[s].get("primer_gain_over_tab_heavy") for s in SCEN}
# H50f は重み付き（A_Ord・A_XPT_S・A_XPT_E）の重い崩壊分で判定
def primer_heavy_weighted(a):
    w = WT[a]["_w"]; num = 0.0; den = 0.0
    for s in w:
        num += np.mean(w[s]) * BASE[s]["R_primer_on_tab_from_heavy"]; den += np.mean(w[s]) * BASE[s]["R_tab_from_heavy"]
    return num / den if den > 0 else None
pw = {a: primer_heavy_weighted(a) for a in WT}
R["primer_gain_weighted_heavy"] = pw
R["hypotheses"] = {
 "H50a": {"value": hs("NUC_G"), "rule": "NUC_G heavy share < 0.10", "supported": hs("NUC_G") < 0.10},
 "H50b": {"value": [hs("BIO"), h32("BIO")], "rule": "BIO heavy share < 032 (T1+T0)", "supported": hs("BIO") < h32("BIO")},
 "H50c": {"value": [hs("CASC"), h32("CASC")], "rule": "CASC heavy share > 032 (T1+T0)", "supported": hs("CASC") > h32("CASC")},
 "H50d": {"value": {a: BASE_W[a]["ratio_R_tab_vs_032"] for a in BASE_W}, "rule": "0.5 <= ratio <= 2 for all anchors",
          "supported": all(0.5 <= BASE_W[a]["ratio_R_tab_vs_032"] <= 2 for a in BASE_W)},
 "H50e": {"value": float(np.median(heavy_wait)) if heavy_wait.size else None, "n_heavy": int(heavy_wait.size),
          "rule": "median pop-wait share of time-to-T2 in heavy samples (NUC_G,NUC_R,BIO,CASC pooled) > 0.5",
          "supported": bool(heavy_wait.size and np.median(heavy_wait) > 0.5)},
 "H50f": {"value": pw, "rule": "T0 primer raises heavy-collapse tablet value by < 20% (all anchors)",
          "supported": all(v is not None and v < 0.20 for v in pw.values())},
 "H50g": {"value": b_tools, "rule": "Kline-Boyd log-log slope < 0.5", "supported": b_tools < 0.5},
}

# ---------- 図 ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
names = list(SCEN); x = np.arange(len(names))
cols = ["#555", "#1f77b4", "#9ecae1", "#2ca02c", "#ff7f0e", "#d62728", "#7f0000"]
for j, (lab, getter) in enumerate((("032 (assumed)", lambda s: [r32["scenarios"][s]["state_shares"][k] for k in ("extinct", "T3", "T3_fragile", "T2", "T1", "T0")] + [0.0]),
                                    ("038 (evidence model)", lambda s: [BASE[s]["state_shares"][k] for k in NAMES7]))):
    bottom = np.zeros(len(names))
    for i in range(7):
        vals = np.array([getter(s)[i] for s in names])
        ax[0].bar(x + (j - 0.5) * 0.38, vals, 0.36, bottom=bottom, color=cols[i], label=NAMES7[i] if j == 1 else None,
                  edgecolor="k" if j == 1 else "none", linewidth=0.4); bottom += vals
ax[0].set_xticks(x); ax[0].set_xticklabels(names); ax[0].set_ylabel("share of outcomes")
ax[0].set_title("restart state: left bar 032 (assumed), right bar 038 (survivors x thresholds)"); ax[0].legend(fontsize=7, ncol=4)
anc = list(BASE_W); xx = np.arange(len(anc))
for j, (lab, vals) in enumerate((("032", [r32["weighted"][a]["R_tab"] for a in anc]),)):
    ax[1].bar(xx - 0.2, vals, 0.38, color="#bbb", label="032 total")
bot = np.zeros(len(anc))
for grp, c in (("light", "#9ecae1"), ("T2", "#2ca02c"), ("heavy", "#d62728")):
    v = np.array([BASE_W[a]["R_tab_from_" + grp] for a in anc]); ax[1].bar(xx + 0.2, v, 0.38, bottom=bot, color=c, label="038 from " + grp); bot += v
ax[1].set_xticks(xx); ax[1].set_xticklabels(anc); ax[1].set_ylabel("weighted dV of tablet guide"); ax[1].legend(fontsize=8)
ax[1].set_title("weighted value: 032 vs 038 split by restart depth")
plt.tight_layout(); plt.savefig(f"{HERE}/fig_state_shares_and_value.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for s, c in zip(EVID, ["C0", "C1", "C2", "C3"]):
    hv = BASE[s]["_heavy_mask"]
    if hv.sum() > 50:
        ax[0].hist(np.log10(BASE[s]["_t2"][hv]), bins=60, histtype="step", color=c, label=f"{s} with pop wait")
        ax[0].hist(np.log10(BASE[s]["_stage"][hv]), bins=60, histtype="step", color=c, ls="--")
ax[0].set_xlabel("log10 years to reach T2 (heavy collapse); dashed = stage time only"); ax[0].legend(fontsize=8)
ax[1].hist(heavy_wait, bins=50, color="C3"); ax[1].set_xlabel("share of time-to-T2 spent waiting for population"); ax[1].set_title("heavy-collapse samples, pooled")
ax[2].scatter(lp / np.log(10), lt / np.log(10), c=["C0" if k[2] == "low" else "C1" for k in KB])
xs = np.linspace(lp.min(), lp.max(), 10); a0, b0 = np.polyfit(lp, lt, 1)[::-1]
ax[2].plot(xs / np.log(10), (a0 + b0 * xs) / np.log(10), "k--", label=f"b={b_tools:.3f}")
for k in KB: ax[2].annotate(k[0], (np.log10(k[1]), np.log10(k[3])), fontsize=7)
ax[2].set_xlabel("log10 population"); ax[2].set_ylabel("log10 marine tool types"); ax[2].set_title("Kline & Boyd 2010 Table 1"); ax[2].legend()
plt.tight_layout(); plt.savefig(f"{HERE}/fig_time_to_T2_and_toolkit.png", dpi=120); plt.close()

for a in WT:
    WT[a].pop("_per_sample_tab", None); WT[a].pop("_w", None)
R["weighted_032_recomputed"] = WT
R["runtime_seconds"] = time.time() - t0
json.dump(R, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({"check_032": R["check_032"]["reproduced_exactly"], "check_gen": R["check_generalized_eq_032"],
                  "scen": {s: {k: v for k, v in R["scenarios_038"][s].items() if k in ("state_shares", "heavy_share", "R_tab", "R_tab_from_heavy", "wait_share_median_heavy", "t2_median_heavy", "stage_median_heavy", "nmax_quantiles", "primer_gain_over_tab_heavy")} for s in SCEN},
                  "weighted": BASE_W, "hyp": R["hypotheses"], "kb": R["kline_boyd"], "sens": R["sensitivity"], "no_wait": R["no_pop_wait"],
                  "thr": R["thresholds_quantiles"]}, ensure_ascii=False, indent=1, default=float))
print("runtime", round(R["runtime_seconds"], 1))
