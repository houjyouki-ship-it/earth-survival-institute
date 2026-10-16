"""研究016：孤立（autarky）期の速度上限 — 教師なし追いつきの代理（PREREGISTRATION.md）
(1) Maddison 2023 で孤立6エピソードの所得成長を、同時期・同所得の開放国と比べ減速倍率 r を推定。
(2) r を 014 の時代係数 γ で等価時代 e_iso に換算。
(3) 015 と同じ乱数の混合型梯子で、e_iso における Piql / M-DISC の D23* と相乗効果を再評価（015 を厳密再現して検証）。
seed 固定、numpy + matplotlib のみ。
"""
import csv, json, time, math, statistics, collections
import numpy as np
t0 = time.time()
SEED = 20261015; SEED_BOOT = 20261016
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
SRC = HERE + "/../013-empirical-catchup-a/sources/owid_gdppc_maddison.csv"
R014 = json.load(open(HERE + "/../014-ratio-vs-speedcap/results.json"))
R015 = json.load(open(HERE + "/../015-nk-calibration/results.json"))

raw = collections.defaultdict(dict); NAME = {}
for r in csv.DictReader(open(SRC, encoding="utf-8")):
    if r["GDP per capita"] and r["Code"]:
        raw[r["Code"]][int(r["Year"])] = float(r["GDP per capita"]); NAME[r["Code"]] = r["Entity"]
AGG = {c for c in raw if c.startswith("OWID_")}
OIL = {"SAU","KWT","ARE","QAT","BHR","OMN","IRQ","IRN","LBY","DZA","AGO","NGA","GAB","GNQ","COG","VEN","ECU","TTO"}

EPIS = [("JPN_sakoku", "JPN", 1639, 1853), ("ESP_autarky", "ESP", 1939, 1959), ("CHN_postsplit", "CHN", 1960, 1978),
        ("MMR_bwts", "MMR", 1962, 1988), ("ALB_postchina", "ALB", 1978, 1990), ("PRK_postsoviet", "PRK", 1991, 2022)]
G_FLOOR = 0.001

def interp_log(code, y, maxgap=30):
    d = raw[code]; ys = sorted(d)
    if y in d: return d[y]
    lo = [t for t in ys if t < y]; hi = [t for t in ys if t > y]
    if not lo or not hi: return None
    a, b = lo[-1], hi[0]
    if y - a > maxgap or b - y > maxgap: return None
    f = (y - a) / (b - a); return math.exp(math.log(d[a]) + f * (math.log(d[b]) - math.log(d[a])))

def episode(eid, code, s, e, epis, factor_seq=(1.5, 2.0)):
    ys = sorted(raw[code]); inn = [t for t in ys if s <= t <= e]
    if len(inn) < 2 or inn[-1] - inn[0] < 8: return {"id": eid, "excluded": True}
    s_, e_ = inn[0], inn[-1]; y0, y1 = raw[code][s_], raw[code][e_]
    g_iso = math.log(y1 / y0) / (e_ - s_)
    busy = {c for (_, c, a, b) in epis if not (b < s_ or a > e_)}
    pool = []
    for c in raw:
        if c in AGG or c in OIL or c in busy: continue
        a0, a1 = interp_log(c, s_), interp_log(c, e_)
        if a0 is None or a1 is None: continue
        pool.append((c, a0, math.log(a1 / a0) / (e_ - s_)))
    used = None
    for fac in factor_seq:
        ctrl = [p for p in pool if y0 / fac <= p[1] <= y0 * fac]
        if len(ctrl) >= 3: used = fac; break
    if used is None: ctrl = pool; used = "all"
    g_open = float(np.median([p[2] for p in ctrl]))
    r = max(g_open, G_FLOOR) / max(g_iso, G_FLOOR)
    return {"id": eid, "code": code, "window_registered": [s, e], "window_used": [s_, e_], "n_obs_in_window": len(inn),
            "y_start": y0, "y_end": y1, "g_iso": g_iso, "T2_iso_years": (math.log(2) / g_iso if g_iso > 0 else math.inf),
            "match_factor": used, "n_controls": len(ctrl), "controls": sorted(p[0] for p in ctrl), "g_open_median": g_open,
            "g_open_quartiles": [float(np.quantile([p[2] for p in ctrl], q)) for q in (0.25, 0.75)],
            "T2_open_years": (math.log(2) / g_open if g_open > 0 else math.inf), "r": r, "excluded": False}

EP = [episode(*x, EPIS) for x in EPIS]
EPu = [x for x in EP if not x["excluded"]]
logr = np.log([x["r"] for x in EPu]); r_med = float(np.exp(np.median(logr))); r_geo = float(np.exp(np.mean(logr)))
rb = np.random.default_rng(SEED_BOOT); B = 4000
bm = np.array([np.median(logr[rb.integers(0, len(logr), len(logr))]) for _ in range(B)])
bg = np.array([np.mean(logr[rb.integers(0, len(logr), len(logr))]) for _ in range(B)])
r_med_ci = [float(np.exp(np.percentile(bm, 2.5))), float(np.exp(np.percentile(bm, 97.5)))]
r_geo_ci = [float(np.exp(np.percentile(bg, 2.5))), float(np.exp(np.percentile(bg, 97.5)))]

A1 = R014["aft"]["D1_UK_NL_US"]; alpha, beta, gamma, s = A1["alpha"], A1["beta"], A1["gamma"], A1["s"]
e_ref = R014["e_ref_median_entry_year"]
to_era = lambda r: e_ref + 50 * math.log(r) / gamma
r_star = math.exp(gamma * (1850 - e_ref) / 50)
e_iso = to_era(r_med); e_iso_ci = [to_era(r_med_ci[1]), to_era(r_med_ci[0])]; e_iso_geo = to_era(r_geo)
LOO = {}
for i, x in enumerate(EPu):
    lr = np.delete(logr, i); rm = float(np.exp(np.median(lr))); LOO[x["id"]] = {"r_med": rm, "e_iso": to_era(rm), "H28b": bool(rm >= r_star)}
# 014 の AFT 中央値（先頭年数 118 年、e=1850）— 014 report の 57 年の再計算
cap = lambda e, P=118.0: math.exp(alpha + beta * math.log(P) + gamma * (e - 1950) / 50)
g_iso_med = float(np.median([x["g_iso"] for x in EPu])); T2_iso_med = math.log(2) / g_iso_med if g_iso_med > 0 else math.inf

# ---------- 梯子（015 と同じ乱数・同じ順序） ----------
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
def num(x): return x if isinstance(x, float) else (1e9 if str(x).startswith(">") else 0.0)
NK_INC = {int(k): v for k, v in R015["n_k_income"].items()}; NK_Y = {int(k): v for k, v in R015["n_k_pop_income"].items()}
def hybrid(nk, e):
    def f(k, d):
        n = nk[k]; return np.minimum(n * np.exp(alpha + gamma * (e - 1950) / 50 + s * ZA) * (d / n) ** beta, d)
    return f
base = {x: run_cell(x) for x in D23_GRID}
def cell(nk, e):
    g = {x: run_cell(x, hybrid(nk, e)) for x in D23_GRID}
    return {"e": e, "D23_star_primer": {m: crit([g[x]["P_survive"][m] for x in D23_GRID]) for m in ("Piql_film_QR", "MDISC_BD")},
            "syn300": {m: g[300]["P_read"][m] - base[300]["P_read"][m] for m in ("Piql_film_QR", "MDISC_BD")},
            "median_T3_m23_300": g[300]["median_T3"], "P_survive_grid_Piql": {str(x): g[x]["P_survive"]["Piql_film_QR"] for x in D23_GRID}}
# 015 再現
c15 = cell(NK_Y, 1850.0); ref = R015["cells"]["pop_income_e1850"]
check015 = {"Piql_D23": abs(num(c15["D23_star_primer"]["Piql_film_QR"]) - num(ref["D23_star_primer"]["Piql_film_QR"])) < 1e-9,
            "syn300_Piql": abs(c15["syn300"]["Piql_film_QR"] - ref["synergy_P_read_m23_300"]["Piql_film_QR"]) < 1e-9}
check015["reproduced_exactly"] = bool(all(check015.values()))
CELLS = {"Y_e1850_015": c15}
for lab, e in [("iso_point", e_iso), ("iso_ci_slow", e_iso_ci[0]), ("iso_ci_fast", e_iso_ci[1]), ("iso_geo", e_iso_geo)]:
    CELLS["Y_" + lab] = cell(NK_Y, e); CELLS["income_" + lab] = cell(NK_INC, e)
for lab, v in LOO.items(): CELLS["Y_LOO_" + lab] = {"e": v["e_iso"], "Piql_D23": cell(NK_Y, v["e_iso"])["D23_star_primer"]["Piql_film_QR"]}
# 探索的（事前登録外）：e を 1700〜2000 で走査
SCAN = {}
for e in [1700, 1750, 1800, 1850, 1900, 1950, 2000]:
    cy = cell(NK_Y, float(e)); SCAN[str(e)] = {"Piql_D23_Y": cy["D23_star_primer"]["Piql_film_QR"], "syn300_Piql_Y": cy["syn300"]["Piql_film_QR"]}
# 【探索的・事前登録外】絶対水準での等価時代：cap(e)=T とおく e_abs、および対照（開放国）の「選択なし」成長と AFT 上限の差
e_abs = lambda T: 1950 + 50 * (math.log(T) - alpha - beta * math.log(118.0)) / gamma
bgi = np.array([np.median(np.array([x["g_iso"] for x in EPu])[rb.integers(0, len(EPu), len(EPu))]) for _ in range(B)])
T2b = np.where(bgi > 0, math.log(2) / np.maximum(bgi, 1e-12), 1e9)  # 1e9 = 成長なし（倍増しない）
EXP = {"e_abs_iso_median": e_abs(T2_iso_med), "T2_iso_median_boot95": [float(np.percentile(T2b, 2.5)), float(np.percentile(T2b, 97.5))]}
EXP["e_abs_iso_boot95"] = [e_abs(EXP["T2_iso_median_boot95"][1]) if EXP["T2_iso_median_boot95"][1] < 1e8 else -math.inf, e_abs(EXP["T2_iso_median_boot95"][0])]
EXP["open_vs_cap"] = {x["id"]: {"T2_open": x["T2_open_years"], "cap_at_start": cap(x["window_used"][0]), "ratio_open_over_cap": (x["T2_open_years"] / cap(x["window_used"][0]) if np.isfinite(x["T2_open_years"]) else math.inf)} for x in EPu}
mod = [x for x in EPu if x["window_used"][0] >= 1900]
EXP["modern_open_over_cap_median"] = float(np.median([EXP["open_vs_cap"][x["id"]]["ratio_open_over_cap"] for x in mod]))
EXP["modern_iso_over_cap_median"] = float(np.median([(x["T2_iso_years"] / cap(x["window_used"][0])) if np.isfinite(x["T2_iso_years"]) else math.inf for x in mod]))
EXP["e_abs_eref_times_open_gap"] = to_era(EXP["modern_open_over_cap_median"])
for lab, e in [("e_abs_iso", EXP["e_abs_iso_median"]), ("e_abs_iso_lo", EXP["e_abs_iso_boot95"][1]), ("eref_x_open_gap", EXP["e_abs_eref_times_open_gap"])]:
    if np.isfinite(e): c_ = cell(NK_Y, e); ci_ = cell(NK_INC, e); EXP["ladder_" + lab] = {"e": e, "Piql_D23_Y": c_["D23_star_primer"]["Piql_film_QR"], "syn300_Piql_Y": c_["syn300"]["Piql_film_QR"], "Piql_D23_income": ci_["D23_star_primer"]["Piql_film_QR"], "syn300_Piql_income": ci_["syn300"]["Piql_film_QR"]}
D23_nop = crit([base[x]["P_survive"]["Piql_film_QR"] for x in D23_GRID])

Hy = {"H28a": {"r_med": r_med, "ci95": r_med_ci, "supported": bool(r_med > 1 and r_med_ci[0] > 1)},
      "H28b": {"r_med": r_med, "r_star": r_star, "e_iso": e_iso, "supported": bool(r_med >= r_star)},
      "H28c": {"Piql_D23_Y_eiso": CELLS["Y_iso_point"]["D23_star_primer"]["Piql_film_QR"], "supported": bool(num(CELLS["Y_iso_point"]["D23_star_primer"]["Piql_film_QR"]) < 1100)},
      "H28d": {"n_same_as_H28b": int(sum(v["H28b"] == bool(r_med >= r_star) for v in LOO.values())), "n": len(LOO)},
      "H28e": {"T2_iso_median_years": T2_iso_med, "cap_1850_P118": cap(1850), "supported": bool(T2_iso_med >= cap(1850))}}
Hy["H28d"]["supported"] = bool(Hy["H28d"]["n_same_as_H28b"] >= 5)

results = {"seed": SEED, "seed_boot": SEED_BOOT, "episodes": EP, "r_median": r_med, "r_median_ci95": r_med_ci, "r_geomean": r_geo, "r_geomean_ci95": r_geo_ci,
           "gamma": gamma, "e_ref": e_ref, "r_star_for_1850": r_star, "e_iso": e_iso, "e_iso_ci95": e_iso_ci, "e_iso_geomean": e_iso_geo, "LOO": LOO,
           "aft_cap_P118": {str(e): cap(e) for e in (1850, 1900, 1950, e_ref)}, "check_015": check015, "cells": CELLS,
           "D23_star_no_primer_Piql": D23_nop, "EXPLORATORY_era_scan_Y": SCAN, "EXPLORATORY_absolute": EXP, "hypotheses": Hy,
           "labels": "孤立期の減速を崩壊後の教師なし追いつきの代理とするのは【推測】。r は政策要因（計画経済）も含む。"}

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
lab = [x["id"] for x in EPu]; gi = [x["g_iso"] * 100 for x in EPu]; go = [x["g_open_median"] * 100 for x in EPu]
qlo = [x["g_open_quartiles"][0] * 100 for x in EPu]; qhi = [x["g_open_quartiles"][1] * 100 for x in EPu]; yy = np.arange(len(lab))
ax[0].barh(yy - 0.2, gi, 0.4, color="firebrick", label="isolated episode"); ax[0].barh(yy + 0.2, go, 0.4, color="steelblue", label="matched open (median, IQR)")
ax[0].errorbar(go, yy + 0.2, xerr=[np.array(go) - qlo, np.array(qhi) - go], fmt="none", ecolor="k", lw=0.8)
for i, x in enumerate(EPu): ax[0].annotate(f"r={x['r']:.2f} (n={x['n_controls']})", (max(gi[i], go[i]), i), xytext=(4, -3), textcoords="offset points", fontsize=7)
ax[0].set_yticks(yy); ax[0].set_yticklabels([f"{x['id']} {x['window_used'][0]}-{x['window_used'][1]}" for x in EPu], fontsize=7)
ax[0].axvline(0, color="k", lw=0.5); ax[0].set_xlabel("GDP per capita growth, %/yr (Maddison 2023)"); ax[0].legend(fontsize=7)
ax[0].set_title(f"isolation slowdown r: median {r_med:.2f} [{r_med_ci[0]:.2f},{r_med_ci[1]:.2f}], r* for 1850 = {r_star:.2f}", fontsize=9)
es = [int(k) for k in SCAN]; ax[1].plot(es, [min(num(SCAN[str(e)]["Piql_D23_Y"]), 1700) for e in es], "-o", color="k", label="Piql D23* (pop x income n_k)")
ax[1].axhline(num(D23_nop), color="r", ls="--", label="no primer"); ax[1].axhline(1100, color="gray", ls=":", label="1100 y")
ax[1].axvspan(e_iso_ci[0], e_iso_ci[1], color="orange", alpha=0.25, label=f"e_iso 95% [{e_iso_ci[0]:.0f},{e_iso_ci[1]:.0f}]"); ax[1].axvline(e_iso, color="orange")
ax[1].axvline(1850, color="b", ls=":", label="015 e=1850"); ax[1].set_xlabel("equivalent era e of speed cap"); ax[1].set_ylabel("Piql D23* with primer (capped 1700)")
ax[1].legend(fontsize=7); ax[1].set_title("archive critical m23 vs speed-cap era")
plt.tight_layout(); plt.savefig(HERE + "/fig_isolation_slowdown.png", dpi=130); plt.close()
fig, ax = plt.subplots(figsize=(7, 4.2))
for x in EPu:
    ys = sorted(t for t in raw[x["code"]] if x["window_used"][0] - 30 <= t <= x["window_used"][1] + 30)
    v = np.log([raw[x["code"]][t] / x["y_start"] for t in ys]); ax.plot(np.array(ys) - x["window_used"][0], v, "-", lw=1, label=x["id"])
ax.axvline(0, color="k", lw=0.5); ax.set_xlabel("years since isolation start"); ax.set_ylabel("log(GDPpc / GDPpc at start)"); ax.legend(fontsize=7); ax.set_title("isolated episodes (+-30y)")
plt.tight_layout(); plt.savefig(HERE + "/fig_episode_paths.png", dpi=130); plt.close()

results["runtime_seconds"] = time.time() - t0
json.dump(results, open(HERE + "/results.json", "w"), ensure_ascii=False, indent=1, default=str)
for x in EP: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items() if k != "controls"})
print("r_med", r_med, r_med_ci, "r_geo", r_geo, r_geo_ci, "r*", r_star, "e_iso", e_iso, e_iso_ci, "e_iso_geo", e_iso_geo)
print("LOO", LOO); print("cap", results["aft_cap_P118"]); print("check015", check015)
for k, v in CELLS.items(): print(k, {kk: vv for kk, vv in v.items() if kk not in ("P_survive_grid_Piql",)})
print("scan", SCAN, "noprimer", D23_nop); print("EXP", json.dumps(EXP, default=str)); print("H", json.dumps(Hy, default=str)); print("runtime", results["runtime_seconds"])
