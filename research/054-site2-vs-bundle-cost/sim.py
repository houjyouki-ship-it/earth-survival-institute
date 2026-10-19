"""研究054：南半球の第2拠点 S と束 T＋C の費用あたりの比較（open_problems 103＋100）。計算のみ、ネット接続なし。
事前登録：PREREGISTRATION.md。040 の sim.py を `A = all_arrays()` の前まで exec し、038〜040 の乱数列・関数・成分をそのまま使う。
新しい乱数は seed 20261054。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P40 = os.path.abspath(os.path.join(HERE, "..", "040-cohesion-vs-knowledge", "sim.py"))
src = open(P40, encoding="utf8").read(); cut = src.index("A = all_arrays()")
G40 = {"__file__": P40, "__name__": "sim040"}
exec(compile(src[:cut], P40, "exec"), G40)
ns = G40["ns"]; N = G40["N"]; SRC = G40["SRC"]; run_src = G40["run_src"]; fixed_src = G40["fixed_src"]
weighted = G40["weighted"]; HEAVY = G40["HEAVY"]; DRAWS = G40["DRAWS"]
U_lost, U_find, Z, life, H = ns["U_lost"], ns["U_find"], ns["Z"], ns["life"], ns["H"]
U_h, P_H, g_req, U_sep, p_sep, t_build = ns["U_h"], ns["P_H"], ns["g_req"], ns["U_sep"], ns["p_sep"], ns["t_build"]
pop_ok, U_t, P_max, kappa, U_acc, p_acc, U_sv = ns["pop_ok"], ns["U_t"], ns["P_max"], ns["kappa"], ns["U_acc"], ns["p_acc"], ns["U_sv"]
V_gen = ns["V_gen"]
r40 = json.load(open(os.path.join(HERE, "..", "040-cohesion-vs-knowledge", "results.json")))
R = {"study": "054-site2-vs-bundle-cost", "seed": 20261054, "N": N}

# ---------- 040 の BASE の状態と T2 到達時刻 ----------
B = {k: run_src(SRC[k], "BASE") for k in SRC}
for k in ("AI", "GEN"): B[k] = fixed_src(k, "BASE")
SC = {k: SRC[k]["sc"] for k in SRC}; SC["AI"] = DRAWS["AI"]; SC["GEN"] = DRAWS["GEN"]
COMPS = list(B)

g = np.random.default_rng(20261054)
def lu(lo, hi, n=N): return np.exp(g.uniform(np.log(lo), np.log(hi), n))

# ---------- 読者の q（033 と同じ q(M)） ----------
ANCH_M = np.log([4096, 18432, 65536, 147456, 2**20 + 16384, 9 * 2**20])
ANCH_Q = np.array([0.0, 0.004583300352870422, 0.05644917259355787, 0.17015031414137002, 0.4316764473145267, 1.0])
M_B = lu(4 * 1024, 144 * 1024)
q_B = np.interp(np.log(M_B), ANCH_M, ANCH_Q)
q_TC = 1 - (1 - 0.76) * (1 - q_B)

# ---------- u（T3 の社会にとっての金庫の価値）：034 の clone なしの網羅 × 037 の半々の f_orph ----------
r34 = json.load(open(os.path.join(HERE, "..", "034-t3-vault-value", "results.json")))["u_estimate"]["no_clone"]
r37 = json.load(open(os.path.join(HERE, "..", "037-copy-survival-a", "results.json")))["f_orph"]["main"]["mix"]
UMAP = {"NUC_G": "NUC_G", "NUC_R": "NUC_R", "BIO": "BIO", "NAT": "BIO", "CASC": "CASC", "AI": "AI",
        "GEN": "GEN", "U3": "GEN", "SV": "GEN", "AS": "GEN", "SO": "GEN"}
def lnq(qd):
    s = (np.log(qd["p95"]) - np.log(qd["p05"])) / (2 * 1.6449)
    return np.minimum(1.0, np.exp(np.log(qd["median"]) + s * g.standard_normal(N)))
U_T3 = {k: lnq(r34[UMAP[k]]) * lnq(r37[UMAP[k]]) for k in COMPS}

# ---------- 届く確率（100） ----------
AUNZ = (27.92e6 + 5.36e6) / 8.232e9                 # 038 の国別人口（豪州 2,792万、NZ 536万、世界 82.32億）
p_SH = g.uniform(0.10, 0.12, N)                       # 南半球は世界人口の 10〜12%（Wikipedia）
f_near = g.uniform(0.6, 0.95, N); f_mid = g.uniform(0.2, 0.6, N); f_far = g.uniform(0.05, 0.4, N)   # 【推測】
k_S = g.uniform(0.5, 1.0, N)                          # 【推測】知られている度合い
U_reg = g.uniform(size=N); U_lost2 = g.uniform(size=N); U_find2 = g.uniform(size=N); U_corr = g.uniform(size=N)
lost_full = g.uniform(0.02, 0.15, N); lost_tc = g.uniform(0.05, 0.40, N)
u1, u2, u3 = U_sv[1], U_sv[2], U_sv[3]
au = 27.92e6 * (0.5 + 0.5 * u1); alt = np.exp(np.log(4.647e7) + u2 * (np.log(2.883e8) - np.log(4.647e7))) * (0.02 + 0.23 * u3)
NUCG_near = au >= alt
def region(k):
    if k == "NUC_G":
        return np.where(NUCG_near, 0, 1)
    return np.where(U_reg < AUNZ, 0, np.where(U_reg < p_SH, 1, 2))
REG = {k: region(k) for k in COMPS}

def find2(k, opt):
    if opt.get("S033"): return np.full(N, 0.75)
    if "find_fixed" in opt: return np.full(N, opt["find_fixed"])
    f = np.choose(REG[k], [f_near, f_mid, f_far])
    return f * (1.0 if opt.get("kS1") else k_S)

def intact2(k, kind, opt):
    lost = np.full(N, 0.06) if opt.get("S033") else (lost_full if kind == "full" else lost_tc)
    uf = np.where(U_corr < opt.get("rho", 0.0), U_find, U_find2)
    return (U_lost2 > lost) & (uf < find2(k, opt))

# ---------- 価値の式（040 の V_gen の写しに S と u を足す） ----------
def V_ext(k, qv, tablet, S=None, u_on=True, opt=None):
    opt = opt or {}
    st, T2s, sc = B[k]["st"], B[k]["T2s"], SC[k]
    alive = st != 0
    intact1 = (U_lost > sc["p_lost"]) & (U_find < sc["p_find"])
    T3t = T2s + 943 * sc["M"] * np.exp(0.8 * Z[2])
    gen = (T3t < life) & (T3t <= H)
    cov = np.zeros(N); avail = np.zeros(N, bool)
    if tablet:
        tab_ok = U_sep > p_sep; cov = np.where(tab_ok, 1.0, cov); avail |= tab_ok
    Tt = T2s + t_build
    tgt = avail & pop_ok & (U_t < P_max * cov ** kappa) & (Tt < life) & (Tt <= H)
    uT3 = U_T3[k] if u_on else 1.0
    ufr = (U_T3[k] if opt.get("ufrag_u") else 1.0) if u_on else 1.0
    frag_acc = (st == 2) & (U_acc < p_acc)
    def finish(v, c=1.0):
        v = np.where(st == 1, uT3 * c, v)
        return np.where(frag_acc, ufr * c, v)
    v1 = finish(np.maximum(qv * tgt, gen.astype(float)))
    if S is None:
        return np.where(alive & intact1, v1, 0.0)
    i2 = intact2(k, S, opt)
    if S == "full":
        v2 = v1
    else:   # 束の写しだけ（板＋c4）
        c = opt.get("cTC", 1.0)
        v2 = finish(np.maximum(0.76 * tgt, c * gen.astype(float)), c) if tablet else np.zeros(N)
    return np.where(alive & intact1, v1, np.where(alive & i2, v2, 0.0))

# 検査：S なし・u=1 で 040 の V_gen と一致し、weighted の TAB を再現する
mx = 0.0
for k in COMPS:
    st, T2s, sc = B[k]["st"], B[k]["T2s"], SC[k]
    mx = max(mx, float(np.abs(V_ext(k, 0.3, True, u_on=False) - V_gen(st, T2s, sc, 0.3, tablet=True)).max()),
             float(np.abs(V_ext(k, 0.3, False, u_on=False) - V_gen(st, T2s, sc, 0.3)).max()))
HV = {k: np.isin(B[k]["st"], HEAVY).astype(float) for k in COMPS}
LT = {k: np.isin(B[k]["st"], (1, 2)).astype(float) for k in COMPS}
T2M = {k: (B[k]["st"] == 3).astype(float) for k in COMPS}
D_tab = {k: V_ext(k, 0.3, True, u_on=False) - V_ext(k, 0.3, False, u_on=False) for k in COMPS}
Wc = weighted({"TAB": D_tab}, HV)
chk = max(max(abs(Wc[a][u]["TAB"]["total"] - r40["weighted"][a][u]["TAB"]["total"]), abs(Wc[a][u]["TAB"]["from_heavy"] - r40["weighted"][a][u]["TAB"]["from_heavy"])) for a in Wc for u in Wc[a])
R["check_040"] = {"max_abs_V_vs_V_gen": mx, "max_abs_weighted_TAB": chk, "reproduced": bool(mx < 1e-12 and chk < 1e-12)}

def deltas(opt=None):
    opt = opt or {}
    D = {x: {} for x in ("TAB", "T_B", "TC", "S_full", "S_full_given_TC", "TC_given_S_full", "S_TC_given_TC")}
    for k in COMPS:
        V0 = V_ext(k, 0.3, False, opt=opt); VTC = V_ext(k, q_TC, True, opt=opt)
        VS = V_ext(k, 0.3, False, S="full", opt=opt); VTCS = V_ext(k, q_TC, True, S="full", opt=opt)
        D["TAB"][k] = V_ext(k, 0.3, True, opt=opt) - V0
        D["T_B"][k] = V_ext(k, q_B, True, opt=opt) - V0
        D["TC"][k] = VTC - V0
        D["S_full"][k] = VS - V0
        D["S_full_given_TC"][k] = VTCS - VTC
        D["TC_given_S_full"][k] = VTCS - VS
        D["S_TC_given_TC"][k] = V_ext(k, q_TC, True, S="tc", opt=opt) - VTC
    return D

def agg(D, anchors=("A_Ord", "A_XPT_S", "A_XPT_E")):
    tot = weighted(D, HV, anchors); lt = weighted(D, LT, anchors); t2 = weighted(D, T2M, anchors)
    out = {}
    for a in tot:
        out[a] = {}
        for u in tot[a]:
            out[a][u] = {x: {"total": tot[a][u][x]["total"], "heavy": tot[a][u][x]["from_heavy"],
                             "light": lt[a][u][x]["from_heavy"], "T2": t2[a][u][x]["from_heavy"]} for x in D}
    return out

D_main = deltas()
VAL = agg(D_main)
R["values_main"] = VAL
# 再現の確認：TAB は u に依存しない
R["check_TAB_u_invariant"] = max(abs(VAL[a][u]["TAB"]["total"] - r40["weighted"][a][u]["TAB"]["total"]) for a in VAL for u in VAL[a])
# 成分ごとの1件あたり（平均）
R["per_event_main"] = {k: {x: float(D_main[x][k].mean()) for x in D_main} for k in COMPS}
R["region_shares"] = {k: [float(np.mean(REG[k] == i)) for i in range(3)] for k in ("NUC_G", "U3")}
R["u_T3_median"] = {k: float(np.median(U_T3[k])) for k in COMPS}
# NUC_G からの寄与（H66f）
def only(D, keep):
    return {x: {k: (D[x][k] if k == keep else np.zeros(N)) for k in COMPS} for x in D}
VAL_NG = agg(only(D_main, "NUC_G"), anchors=("A_Ord", "A_XPT_S"))
R["share_from_NUC_G"] = {f"{a}/{u}": {x: (VAL_NG[a][u][x]["total"] / VAL[a][u][x]["total"] if VAL[a][u][x]["total"] > 0 else None) for x in VAL_NG[a][u]} for a in VAL_NG for u in ("U2", "U3")}

# ---------- 費用（2025年ドル） ----------
CPI = {2008: 215.3, 2011: 224.9, 2013: 233.0, 2018: 251.1, 2019: 255.7, 2025: 321.9}   # 米 CPI-U 年平均（Minneapolis Fed）
def to25(x, y): return x * CPI[2025] / CPI[y]
NC = 200_000
gc = np.random.default_rng(20261054 + 1)
def lc(lo, hi, n=NC): return np.exp(gc.uniform(np.log(lo), np.log(hi), n))
eur = gc.uniform(1.0, 1.2, NC)
def cost_T(prod_lo=1e5, prod_hi=1e7, ch_lo=1e7, ch_hi=1e8):
    chars = lc(ch_lo, ch_hi); prod = lc(prod_lo, prod_hi); wage = lc(1e5, 2e5); per_tab = lc(5e4, 5e6)
    n_tab = np.ceil(chars / per_tab)
    return chars / prod * wage + n_tab * 195 * eur * 3, n_tab, chars / prod * wage
C_T, NTAB, C_auth = cost_T()
p_reel = lc(8900 / 4, 8900 * 4) * eur                                  # 【推測】AWA の「from €8900」を中心に
n_C = lc(4, 10)
C_C = n_C * p_reel + lc(0.5, 3) * lc(1e5, 2e5)
C_TC = C_T + C_C
NOK19 = 282000 / 2.4e6                                                  # 2019 の換算（Wikipedia の 240万 kr = $28.2万）
F = {"F1_SeedVault_build_2008": to25(8.8e6, 2008) * np.ones(NC),
     "F2_SeedVault_build_plus_upgrade": to25(8.8e6, 2008) + to25(193e6 * NOK19, 2019) * np.ones(NC),
     "F3_PlantBank_2013": to25(19.8e6, 2013) * gc.uniform(0.9, 1.05, NC)}
FN = list(F); pick = gc.integers(0, len(FN), NC); fspread = lc(0.5, 2.0)
cap = np.choose(pick, [F[f] for f in FN]) * fspread
om_yr = np.exp(np.log(to25(282000, 2019)) + gc.uniform(size=NC) * (np.log(1.0e6 * 1.1) - np.log(to25(282000, 2019))))  # 年の維持費 logU(種子庫の 2019年の維持費 $28.2万→2025年ドル, NordGen の年 €100万×1.1)
years = gc.uniform(20, 100, NC)
copy_full = 186 * p_reel
C_Sfull = cap + om_yr * years + copy_full
C_Sfull_withTC = C_Sfull + NTAB * gc.uniform(195, 260, NC) * eur + n_C * p_reel
C_STC = NTAB * gc.uniform(195, 260, NC) * eur + n_C * p_reel + lc(1e4, 3e5) + lc(1e3, 5e4) * gc.uniform(20, 100, NC)
def q(x, ps=(0.05, 0.5, 0.95)): return [float(np.quantile(x, p)) for p in ps]
R["costs"] = {"C_T": q(C_T), "C_T_author_share_median": float(np.median(C_auth / C_T)), "C_C": q(C_C), "C_TC": q(C_TC),
              "C_Sfull": q(C_Sfull), "C_Sfull_capital": q(cap), "C_Sfull_om": q(om_yr * years), "C_Sfull_copy186": q(copy_full),
              "C_Sfull_withTC": q(C_Sfull_withTC), "C_STC": q(C_STC), "p_reel_usd": q(p_reel),
              "anchors_2025usd": {f: float(np.median(F[f])) for f in FN},
              "ratio_median_Sfull_over_TC": float(np.median(C_Sfull) / np.median(C_TC))}

# ---------- 費用あたりの比 CE（053 の定義：TAB=1） ----------
OPTS = [("TC", C_TC), ("T_B", C_T), ("S_full", C_Sfull), ("S_full_given_TC", C_Sfull_withTC), ("S_TC_given_TC", C_STC)]
def ce_block(val, C_TAB=C_T, Ctc=C_TC, Cs=C_Sfull, Cstc=C_STC, parts=("total", "heavy", "light", "T2")):
    out = {}
    cost = {"TC": Ctc, "T_B": C_TAB, "S_full": Cs, "S_full_given_TC": Cs + (C_Sfull_withTC - C_Sfull), "S_TC_given_TC": Cstc,
            "TC_given_S_full": Ctc}
    for p in parts:
        vt = val["TAB"][p]; out[p] = {}
        if vt <= 0: continue
        for x in ("TC", "T_B", "S_full", "S_full_given_TC", "S_TC_given_TC", "TC_given_S_full"):
            ce = val[x][p] / vt * C_TAB / cost[x]
            out[p][x] = {"value_ratio": val[x][p] / vt, "CE_q": q(ce)}
        r = (val["S_full"][p] / val["TC"][p]) * Ctc / Cs if val["TC"][p] > 0 else None
        out[p]["CE_Sfull_over_TC_q"] = q(r) if r is not None else None
        out[p]["P_CE_Sfull_gt_TC"] = float((r > 1).mean()) if r is not None else None
        out[p]["value_ratio_Sfull_over_TC"] = val["S_full"][p] / val["TC"][p] if val["TC"][p] > 0 else None
        # 束の後の S_full と S_TC の比
        if val["S_full_given_TC"][p] > 0:
            rr = (val["S_TC_given_TC"][p] / val["S_full_given_TC"][p]) * (C_Sfull_withTC) / Cstc
            out[p]["CE_STC_over_SfullGivenTC_q"] = q(rr)
    return out
CE = {}
for a in VAL:
    for u in VAL[a]:
        CE[f"{a}/{u}"] = ce_block(VAL[a][u])
R["CE_main"] = CE
M1, M2 = "A_Ord/U3", "A_XPT_S/U2"
def V2(key): a, u = key.split("/"); return VAL[a][u]

# ---------- 感度 ----------
SENS_OPT = {"S033_const": {"S033": True}, "ufrag_u": {"ufrag_u": True}, "corr_rho0.5": {"rho": 0.5}, "kS_1": {"kS1": True}, "cTC_0.3": {"cTC": 0.3}}
R["sensitivity_value"] = {}
for tag, opt in SENS_OPT.items():
    Vk = agg(deltas(opt), anchors=("A_Ord", "A_XPT_S"))
    R["sensitivity_value"][tag] = {}
    for key in (M1, M2):
        a, u = key.split("/"); v = Vk[a][u]; c = ce_block(v, parts=("total", "heavy", "light"))
        R["sensitivity_value"][tag][key] = {"values": v, "CE_Sfull_over_TC_median": {p: c[p]["CE_Sfull_over_TC_q"][1] for p in c},
                                             "P_CE_Sfull_gt_TC": {p: c[p]["P_CE_Sfull_gt_TC"] for p in c},
                                             "CE_STC_given_TC_median": {p: c[p]["S_TC_given_TC"]["CE_q"][1] for p in c}}
# find_S の走査（喪失は主設定）
FG = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
R["find_scan"] = {M1: [], M2: []}
for f in FG:
    Vk = agg(deltas({"find_fixed": f}), anchors=("A_Ord", "A_XPT_S"))
    for key in (M1, M2):
        a, u = key.split("/"); v = Vk[a][u]
        row = {"find": f}
        for p in ("total", "heavy", "light"):
            row[p] = {"S_full": v["S_full"][p], "TC": v["TC"][p], "S_full_given_TC": v["S_full_given_TC"][p], "S_TC_given_TC": v["S_TC_given_TC"][p],
                      "CE_ratio_median_costs": (v["S_full"][p] / v["TC"][p]) * float(np.median(C_TC)) / float(np.median(C_Sfull)) if v["TC"][p] > 0 else None,
                      "CE_STC_given_TC_median": float(np.median(v["S_TC_given_TC"][p] / v["TAB"][p] * C_T / C_STC)) if v["TAB"][p] > 0 else None}
        R["find_scan"][key].append(row)
def find_star(key, p="total"):
    rows = R["find_scan"][key]; xs = [r["find"] for r in rows]; ys = [r[p]["CE_ratio_median_costs"] for r in rows]
    if ys[-1] < 1: return {"find_star": None, "gt_1": True, "ratio_at_find1": ys[-1]}
    for i in range(1, len(xs)):
        if ys[i] >= 1 > ys[i - 1]:
            return {"find_star": xs[i - 1] + (1 - ys[i - 1]) / (ys[i] - ys[i - 1]) * (xs[i] - xs[i - 1]), "gt_1": False, "ratio_at_find1": ys[-1]}
    return {"find_star": xs[0] if ys[0] >= 1 else None, "gt_1": False, "ratio_at_find1": ys[-1]}
R["find_star"] = {key: {p: find_star(key, p) for p in ("total", "heavy", "light")} for key in (M1, M2)}

# 費用の感度
R["sensitivity_cost"] = {}
def cost_case(tag, C_Tk, C_Sk=C_Sfull):
    C_TCk = C_Tk + C_C
    R["sensitivity_cost"][tag] = {}
    for key in (M1, M2):
        c = ce_block(V2(key), C_TAB=C_Tk, Ctc=C_TCk, Cs=C_Sk, parts=("total", "heavy"))
        R["sensitivity_cost"][tag][key] = {p: {"CE_Sfull_over_TC_q": c[p]["CE_Sfull_over_TC_q"], "P_CE_Sfull_gt_TC": c[p]["P_CE_Sfull_gt_TC"],
                                               "CE_TC_q": c[p]["TC"]["CE_q"], "CE_STC_given_TC_q": c[p]["S_TC_given_TC"]["CE_q"]} for p in c}
    R["sensitivity_cost"][tag]["C_TC_q"] = q(C_TCk); R["sensitivity_cost"][tag]["C_S_q"] = q(C_Sk)
cost_case("prod_low_1e5_1e6", cost_T(1e5, 1e6)[0])
cost_case("prod_high_1e6_1e7", cost_T(1e6, 1e7)[0])
cost_case("chars_024_greq", cost_T(ch_lo=0.725e6, ch_hi=8.9e6)[0])
for i, f in enumerate(FN):
    cost_case("anchor_" + f, C_T, F[f] * fspread + om_yr * years + copy_full)
clock = 42e6 * gc.uniform(1.0, CPI[2025] / CPI[2011], NC)
cost_case("anchor_F4_LongNowClock", C_T, clock * fspread + om_yr * years + copy_full)
cost_case("no_om", C_T, cap + copy_full)

# ---------- 仮説 ----------
Hh = {}
Hh["H66a"] = {"value": R["costs"]["ratio_median_Sfull_over_TC"], "rule": "median C_Sfull >= 3 x median C_TC", "supported": R["costs"]["ratio_median_Sfull_over_TC"] >= 3}
def med_lt(key, p):
    c = CE[key][p]; return c["S_full"]["CE_q"][1], c["TC"]["CE_q"][1]
v = med_lt(M1, "total"); Hh["H66b"] = {"value": v, "rule": "A_Ord/U3 median CE(S_full) < median CE(TC)", "supported": v[0] < v[1]}
v = med_lt(M2, "total"); Hh["H66c"] = {"value": v, "rule": "A_XPT_S/U2 median CE(S_full) < median CE(TC)", "supported": v[0] < v[1]}
v = {k: med_lt(k, "heavy") for k in (M1, M2)}; Hh["H66d"] = {"value": v, "rule": "heavy only, both: median CE(S_full) < CE(TC)", "supported": all(a < b for a, b in v.values())}
v = {k: CE[k]["total"]["S_TC_given_TC"]["CE_q"][1] for k in (M1, M2)}; Hh["H66e"] = {"value": v, "rule": "median CE(S_TC|TC) > 1 for both", "supported": all(x > 1 for x in v.values())}
v = R["share_from_NUC_G"][M2]["S_full_given_TC"]; Hh["H66f"] = {"value": v, "rule": "A_XPT_S/U2 share of S_full|TC from NUC_G >= 0.5", "supported": v is not None and v >= 0.5}
v = R["find_star"][M1]["total"]; Hh["H66g"] = {"value": v, "rule": "A_Ord/U3 find_S* > 1 (ratio at find=1 < 1)", "supported": bool(v["gt_1"])}
v = R["sensitivity_cost"]["prod_low_1e5_1e6"][M1]["total"]["P_CE_Sfull_gt_TC"]; Hh["H66h"] = {"value": v, "rule": "prod logU(1e5,1e6): A_Ord/U3 P(CE_S>CE_TC) < 0.5", "supported": v < 0.5}
R["hypotheses"] = Hh

# ---------- 図 ----------
fig, ax = plt.subplots(figsize=(8.5, 4.8))
items = [("T (tablet guide)", C_T), ("C (c4, ~4-10 reels)", C_C), ("T+C bundle", C_TC), ("S_full: capital", cap), ("S_full: O&M", om_yr * years),
         ("S_full: copy 186 reels", copy_full), ("S_full total", C_Sfull), ("S_TC: south copy of bundle", C_STC)]
ax.boxplot([np.log10(x) for _, x in items], whis=(5, 95), showfliers=False)
ax.set_xticks(range(1, len(items) + 1)); ax.set_xticklabels([a for a, _ in items], rotation=25, fontsize=7, ha="right")
ax.set_ylabel("log10 cost (2025 USD)"); ax.set_title("Cost of the bundle, the new southern site, and a southern copy of the bundle (box 25-75%, whiskers 5-95%)", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_costs.png"), dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
xs_ = ["TC", "S_full", "S_full_given_TC", "S_TC_given_TC", "T_B"]
for ax, key in zip(axs, (M1, M2)):
    a, u = key.split("/"); val = VAL[a][u]
    cost = {"TC": C_TC, "T_B": C_T, "S_full": C_Sfull, "S_full_given_TC": C_Sfull_withTC, "S_TC_given_TC": C_STC}
    data = []; labs = []; cols = []
    for x in xs_:
        for p, col in (("heavy", "C3"), ("light", "C0")):
            if val["TAB"][p] > 0 and val[x][p] > 0:
                data.append(np.log10(val[x][p] / val["TAB"][p] * C_T / cost[x])); labs.append(f"{x}\n{p}"); cols.append(col)
    bp = ax.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True)
    for b, c in zip(bp["boxes"], cols): b.set_facecolor(c); b.set_alpha(0.4)
    ax.set_xticks(range(1, len(labs) + 1)); ax.set_xticklabels(labs, fontsize=6.5, rotation=30)
    ax.axhline(0, color="k", ls="--", lw=0.8); ax.set_title(f"{key}: CE vs tablet guide (red=heavy, blue=light)", fontsize=9)
axs[0].set_ylabel("log10 CE (value per $, TAB=1, same part)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_ce_heavy_light.png"), dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for key, c in ((M1, "C3"), (M2, "C0")):
    rows = R["find_scan"][key]
    for p, ls in (("total", "-"), ("heavy", "--"), ("light", ":")):
        ys = [r[p]["CE_ratio_median_costs"] for r in rows]
        if all(y is not None for y in ys): axs[0].semilogy(FG, ys, color=c, ls=ls, label=f"{key} {p}")
        ys2 = [r[p]["CE_STC_given_TC_median"] for r in rows]
        if all(y is not None and y > 0 for y in ys2): axs[1].semilogy(FG, ys2, color=c, ls=ls, label=f"{key} {p}")
axs[0].axhline(1, color="k", lw=0.8); axs[0].set_xlabel("find probability of southern site S"); axs[0].set_ylabel("CE(S_full alone) / CE(T+C alone), median costs")
axs[0].set_title("100: reach of S vs break-even with the bundle", fontsize=9); axs[0].legend(fontsize=7)
axs[1].axhline(1, color="k", lw=0.8); axs[1].set_xlabel("find probability of southern site S"); axs[1].set_ylabel("median CE of southern copy of bundle | T+C (TAB=1)")
axs[1].set_title("S_TC: copy of the bundle in an existing southern institution", fontsize=9); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_breakeven_find.png"), dpi=130); plt.close(fig)

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({"check": R["check_040"], "check_u": R["check_TAB_u_invariant"], "costs": R["costs"], "u_T3_median": R["u_T3_median"], "region": R["region_shares"]}, ensure_ascii=False, indent=1))
for key in (M1, M2):
    a, u = key.split("/")
    print("==", key)
    for x, v in VAL[a][u].items(): print("  value", x, {p: round(vv, 7) for p, vv in v.items()})
    for p, c in CE[key].items():
        print("  CE", p, {x: [round(z, 4) for z in c[x]["CE_q"]] for x in c if isinstance(c[x], dict) and "CE_q" in c[x]},
              "S/TC", c.get("CE_Sfull_over_TC_q"), "P", c.get("P_CE_Sfull_gt_TC"))
print("share_from_NUC_G", json.dumps(R["share_from_NUC_G"]))
print("find_star", json.dumps(R["find_star"]))
print("sens_value", json.dumps({t: {k: (v["CE_Sfull_over_TC_median"], v["CE_STC_given_TC_median"]) for k, v in d.items()} for t, d in R["sensitivity_value"].items()}))
print("sens_cost", json.dumps({t: {k: (d[k]["total"]["CE_Sfull_over_TC_q"][1], d[k]["total"]["P_CE_Sfull_gt_TC"], d[k]["heavy"]["CE_Sfull_over_TC_q"][1]) for k in (M1, M2)} for t, d in R["sensitivity_cost"].items()}))
for k, v in Hh.items(): print(k, v["supported"], json.dumps(v["value"])[:300])
print("runtime", round(R["runtime_seconds"], 1))
