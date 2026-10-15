"""研究015：n_k を所得倍増 vs 人口×所得倍増で較正し、混合型梯子の Piql/M-DISC D23* を再評価。
事前登録: PREREGISTRATION.md（本ファイル実行前に作成済み）。
seed 固定。numpy + matplotlib のみ。AFT 係数は研究014 results.json から固定入力。
"""
import csv, json, time, math, statistics
import numpy as np

t0 = time.time()
SEED = 20261015
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
SRC_GDP = HERE + "/../013-empirical-catchup-a/sources/owid_gdppc_maddison.csv"
SRC_POP = HERE + "/sources/population_england_millennium.csv"
R014 = json.load(open(HERE + "/../014-ratio-vs-speedcap/results.json"))
R013 = json.load(open(HERE + "/../013-empirical-catchup-a/results.json"))

# ---------- データ ----------
gdp = {}
for r in csv.DictReader(open(SRC_GDP, encoding="utf-8")):
    if r["Code"] == "GBR" and r["GDP per capita"]:
        gdp[int(r["Year"])] = float(r["GDP per capita"])
pop = {}
for r in csv.DictReader(open(SRC_POP, encoding="utf-8")):
    pop[int(r["Year"])] = float(r["Population (England)"])

MILESTONES = [1290, 1620, 1948, 2012]
assert all(y in gdp and y in pop for y in MILESTONES)

def n_income(a, b):
    return math.log2(gdp[b] / gdp[a])

def n_pop(a, b):
    return math.log2(pop[b] / pop[a])

def n_Y(a, b):
    return math.log2((pop[b] * gdp[b]) / (pop[a] * gdp[a]))

pairs = [(1290, 1620), (1620, 1948), (1948, 2012)]
nk_income = {i + 1: n_income(*p) for i, p in enumerate(pairs)}
nk_pop = {i + 1: n_pop(*p) for i, p in enumerate(pairs)}
nk_Y = {i + 1: n_Y(*p) for i, p in enumerate(pairs)}
# 境界感度: 1620 人口を 1700 に置換（England→UK 切替直前）
pop_sens = dict(pop); pop_sens[1620] = pop[1700]
nk_Y_boundary = {
    1: math.log2((pop_sens[1620] * gdp[1620]) / (pop[1290] * gdp[1290])),
    2: math.log2((pop[1948] * gdp[1948]) / (pop_sens[1620] * gdp[1620])),
    3: nk_Y[3],
}

# 014 再現確認（所得のみ n_k）
nk014 = {int(k): v for k, v in R014["n_doublings_GBR"].items()}
check_nk014 = {k: abs(nk_income[k] - nk014[k]) < 1e-12 for k in (1, 2, 3)}

# ---------- 梯子（014 と同じ乱数構造） ----------
rng = np.random.default_rng(SEED); N = 200_000; H = 1000.0
MEDIA = {
    "MDISC_BD": dict(tier=3, claim=(100, 500), site="room"),
    "Piql_film_QR": dict(tier=3, claim=(500, 2000), site="vault"),
    "Silica_glass": dict(tier=3, claim=(1e4, 1e5), site="room"),
    "DNA_silica": dict(tier=4, claim=(2000, 1e4), site="room"),
    "Microfilm_AgX": dict(tier=1, claim=(300, 500), site="room"),
    "Rosetta_Ni": dict(tier=2, claim=(1e3, 1e4), site="room"),
    "MOM_ceramic": dict(tier=1, claim=(1e4, 1e6), site="vault"),
}
F_RANGE = {"room": (0.1, 1.0), "vault": (0.3, 1.0)}; WK = 2.0
Z = {k: rng.standard_normal(N) for k in ["d01", "d12", "d23", "d34"]}
U_a = rng.uniform(np.log(0.2), np.log(0.7), N); U_find_p = rng.uniform(0.3, 0.9, N)
U_find = {k: rng.uniform(size=N) for k in ["primer", "corpus"]}
U_claim = {m: rng.uniform(size=N) for m in MEDIA}
U_f = {m: rng.uniform(size=N) for m in MEDIA}
U_surv = {m: rng.uniform(size=N) for m in MEDIA}
U01 = (U_a - np.log(0.2)) / (np.log(0.7) - np.log(0.2))
ND = statistics.NormalDist()
ZA = np.array([ND.inv_cdf(min(max(u, 1e-12), 1 - 1e-12)) for u in U01])

def lifetime(m):
    lo, hi = MEDIA[m]["claim"]; flo, fhi = F_RANGE[MEDIA[m]["site"]]
    claim = np.exp(np.log(lo) + U_claim[m] * (np.log(hi) - np.log(lo)))
    f = np.exp(np.log(flo) + U_f[m] * (np.log(fhi) - np.log(flo)))
    return claim * f / np.log(2) ** (1 / WK) * (-np.log(U_surv[m])) ** (1 / WK)

LIFE = {m: lifetime(m) for m in MEDIA}

A1 = R014["aft"]["D1_UK_NL_US"]
e_ref = R014["e_ref_median_entry_year"]
alpha, beta, gamma, s = A1["alpha"], A1["beta"], A1["gamma"], A1["s"]

# mutable N_K used by hybrid_step
N_K = dict(nk_income)

def tier_times(m23, primer_found=None, step=None, d12_med=30.0):
    steps = [20 * np.exp(0.7 * Z["d01"]), d12_med * np.exp(0.7 * Z["d12"]),
             m23 * np.exp(0.8 * Z["d23"]), 50 * np.exp(0.8 * Z["d34"])]
    T = np.zeros((5, N))
    for k, d in enumerate(steps):
        dd = np.where(primer_found, step(k, d), d) if (primer_found is not None and k >= 1) else d
        T[k + 1] = T[k] + dd
    return T

def run_cell(m23, step=None, d12_med=30.0):
    if step is not None:
        pf = (U_find["primer"] < U_find_p) & (LIFE["MOM_ceramic"] > tier_times(m23, d12_med=d12_med)[1])
        T = tier_times(m23, pf, step, d12_med)
    else:
        T = tier_times(m23, d12_med=d12_med)
    cf = U_find["corpus"] < U_find_p
    out = {"median_T3": float(np.median(T[3])), "P_survive": {}, "P_read": {}}
    for m, sp in MEDIA.items():
        tk = T[sp["tier"]]; alive = LIFE[m] > tk
        out["P_survive"][m] = float(np.mean(alive))
        out["P_read"][m] = float(np.mean(alive & (tk <= H) & cf))
    return out

D23_GRID = [50, 100, 200, 300, 400, 600, 800, 1200, 1600]

def crit(vals):
    xs = np.array(D23_GRID, float); ys = np.array(vals)
    if ys[0] < 0.5: return "<%d" % D23_GRID[0]
    if ys[-1] >= 0.5: return ">%d" % D23_GRID[-1]
    i = int(np.where(ys < 0.5)[0][0])
    return float(np.exp(np.log(xs[i - 1]) + (0.5 - ys[i - 1]) * (np.log(xs[i]) - np.log(xs[i - 1])) / (ys[i] - ys[i - 1])))

def hybrid_step(e):
    def f(k, d):
        n = N_K[k]
        dur = n * np.exp(alpha + gamma * (e - 1950) / 50 + s * ZA) * (d / n) ** beta
        return np.minimum(dur, d)
    return f

def num(x):
    if isinstance(x, float): return x
    s_ = str(x)
    if s_.startswith(">"): return 1e9
    if s_.startswith("<"): return 0.0
    return float(x)

def eval_nk(nk_dict, e, label):
    global N_K
    N_K = dict(nk_dict)
    st = hybrid_step(e)
    base = {x: run_cell(x) for x in D23_GRID}  # no primer independent of N_K for steps, but still needed
    # actually no-primer does not use N_K; compute once outside ideally
    g = {x: run_cell(x, st) for x in D23_GRID}
    return {
        "label": label, "e": e, "n_k": {str(k): nk_dict[k] for k in (1, 2, 3)},
        "D23_star_primer": {m: crit([g[x]["P_survive"][m] for x in D23_GRID]) for m in ("Piql_film_QR", "MDISC_BD")},
        "synergy_P_read_m23_300": {
            "Piql_film_QR": g[300]["P_read"]["Piql_film_QR"] - base[300]["P_read"]["Piql_film_QR"],
            "MDISC_BD": g[300]["P_read"]["MDISC_BD"] - base[300]["P_read"]["MDISC_BD"],
        },
        "median_T3_m23_300": g[300]["median_T3"],
        "P_survive_grid_Piql": {str(x): g[x]["P_survive"]["Piql_film_QR"] for x in D23_GRID},
        "P_survive_grid_MDISC": {str(x): g[x]["P_survive"]["MDISC_BD"] for x in D23_GRID},
    }

# no-primer baseline once
base_nop = {x: run_cell(x) for x in D23_GRID}
D23_noprimer = {m: crit([base_nop[x]["P_survive"][m] for x in D23_GRID]) for m in ("Piql_film_QR", "MDISC_BD")}

# 014 hybrid 再現チェック（所得 n_k + e_ref）
N_K = dict(nk_income)
g014 = {x: run_cell(x, hybrid_step(e_ref)) for x in D23_GRID}
d23_piql_014rep = crit([g014[x]["P_survive"]["Piql_film_QR"] for x in D23_GRID])
d23_mdisc_014rep = crit([g014[x]["P_survive"]["MDISC_BD"] for x in D23_GRID])
syn_piql_014rep = g014[300]["P_read"]["Piql_film_QR"] - base_nop[300]["P_read"]["Piql_film_QR"]
ref014 = R014["reeval"]["hybrid_AFT_D1_eref"]
same = lambda a, b: (a == b) if isinstance(a, str) or isinstance(b, str) else abs(a - b) < 1e-9
check_014_ladder = {
    "Piql_D23": same(d23_piql_014rep, ref014["D23_star_primer"]["Piql_film_QR"]),
    "MDISC_D23": same(d23_mdisc_014rep, ref014["D23_star_primer"]["MDISC_BD"]),
    "syn300_Piql": abs(syn_piql_014rep - ref014["synergy_P_read_m23_300"]["Piql_film_QR"]) < 1e-9,
}
check_014_ladder["reproduced_exactly"] = bool(all(check_014_ladder.values()))

# 主比較
cells = {}
for key, nk, e in [
    ("income_eref", nk_income, e_ref),
    ("pop_income_eref", nk_Y, e_ref),
    ("income_e1850", nk_income, 1850.0),
    ("pop_income_e1850", nk_Y, 1850.0),
    ("boundary_pop_income_eref", nk_Y_boundary, e_ref),
]:
    # reuse base_nop inside eval — rewrite eval to accept external base
    N_K = dict(nk)
    st = hybrid_step(e)
    g = {x: run_cell(x, st) for x in D23_GRID}
    cells[key] = {
        "label": key, "e": e, "n_k": {str(k): nk[k] for k in (1, 2, 3)},
        "D23_star_primer": {
            "Piql_film_QR": crit([g[x]["P_survive"]["Piql_film_QR"] for x in D23_GRID]),
            "MDISC_BD": crit([g[x]["P_survive"]["MDISC_BD"] for x in D23_GRID]),
        },
        "synergy_P_read_m23_300": {
            "Piql_film_QR": g[300]["P_read"]["Piql_film_QR"] - base_nop[300]["P_read"]["Piql_film_QR"],
            "MDISC_BD": g[300]["P_read"]["MDISC_BD"] - base_nop[300]["P_read"]["MDISC_BD"],
        },
        "median_T3_m23_300": g[300]["median_T3"],
        "P_survive_grid_Piql": {str(x): g[x]["P_survive"]["Piql_film_QR"] for x in D23_GRID},
        "P_survive_grid_MDISC": {str(x): g[x]["P_survive"]["MDISC_BD"] for x in D23_GRID},
    }

# 探索的: 全段同一 c 倍（014 対照）
NK_SCAN = {}
for e_ in (e_ref, 1850.0):
    for cmul in (1, 2, 4, 8):
        nk = {k: nk_income[k] * cmul for k in (1, 2, 3)}
        N_K = dict(nk)
        st = hybrid_step(e_)
        g = {x: run_cell(x, st) for x in D23_GRID}
        NK_SCAN[f"e{int(e_)}_x{cmul}"] = {
            "Piql_D23_star_primer": crit([g[x]["P_survive"]["Piql_film_QR"] for x in D23_GRID]),
            "MDISC_D23_star_primer": crit([g[x]["P_survive"]["MDISC_BD"] for x in D23_GRID]),
            "syn300_Piql": g[300]["P_read"]["Piql_film_QR"] - base_nop[300]["P_read"]["Piql_film_QR"],
            "syn300_MDISC": g[300]["P_read"]["MDISC_BD"] - base_nop[300]["P_read"]["MDISC_BD"],
        }

# 実効 a（pop×income, e_ref）
N_K = dict(nk_Y)
eff_a = {}
for k, nm in [(1, "d12"), (2, "d23"), (3, "d34")]:
    d = [None, 30 * np.exp(0.7 * Z["d12"]), 300 * np.exp(0.8 * Z["d23"]), 50 * np.exp(0.8 * Z["d34"])][k]
    a_ = hybrid_step(e_ref)(k, d) / d
    eff_a[nm] = {str(q): float(np.quantile(a_, q)) for q in (0.25, 0.5, 0.75)}

# ---------- 仮説判定 ----------
ratio_23 = nk_Y[2] / nk_income[2]
Hy = {}
Hy["H27a"] = {
    "n23_income": nk_income[2], "n23_pop_income": nk_Y[2], "ratio": ratio_23,
    "supported": bool(nk_Y[2] >= 2.0 * nk_income[2]),
}
piql_Y = cells["pop_income_eref"]["D23_star_primer"]["Piql_film_QR"]
Hy["H27b"] = {"Piql_D23_star": piql_Y, "supported": bool(num(piql_Y) < 1100)}
mdisc_inc = cells["income_eref"]["D23_star_primer"]["MDISC_BD"]
mdisc_Y = cells["pop_income_eref"]["D23_star_primer"]["MDISC_BD"]
# 悪化しない: 数値比較（>1600 等は num で）— D23* が大きいほど primer が効いているので「悪化」= D23* が下がる
Hy["H27c"] = {
    "MDISC_income": mdisc_inc, "MDISC_pop_income": mdisc_Y,
    "supported": bool(num(mdisc_Y) >= num(mdisc_inc) - 1e-9) or (str(mdisc_inc).startswith("<") and str(mdisc_Y).startswith("<")),
}
# 注: M-DISC が両方 <50 なら「悪化しない」を支持。D23* 低下は primer 弱化＝悪化。
# 再定義: 「悪化しない」= pop×income の D23* が income の D23* 以上（効きが弱まらない）、または両方フロア
if str(mdisc_inc).startswith("<") and str(mdisc_Y).startswith("<"):
    Hy["H27c"]["supported"] = True
else:
    Hy["H27c"]["supported"] = bool(num(mdisc_Y) >= num(mdisc_inc) - 1e-9)

piql_Y_1850 = cells["pop_income_e1850"]["D23_star_primer"]["Piql_film_QR"]
Hy["H27d"] = {"Piql_D23_star_e1850": piql_Y_1850, "supported": bool(num(piql_Y_1850) < 1100)}

milestone_table = []
for y in MILESTONES:
    milestone_table.append({
        "year": y, "gdppc": gdp[y], "population": pop[y], "Y_pop_x_gdppc": pop[y] * gdp[y],
    })

results = {
    "seed": SEED, "N": N,
    "preregistration_file": "PREREGISTRATION.md",
    "population_series_note": "OWID population-of-england-millennium: pre-1700 England, post-1700 UK (Bank of England 2016).",
    "milestones": milestone_table,
    "n_k_income": {str(k): nk_income[k] for k in (1, 2, 3)},
    "n_k_pop_only": {str(k): nk_pop[k] for k in (1, 2, 3)},
    "n_k_pop_income": {str(k): nk_Y[k] for k in (1, 2, 3)},
    "n_k_pop_income_boundary1700": {str(k): nk_Y_boundary[k] for k in (1, 2, 3)},
    "ratio_pop_income_over_income": {str(k): nk_Y[k] / nk_income[k] for k in (1, 2, 3)},
    "check_nk_vs_014": check_nk014,
    "check_014_hybrid_ladder": check_014_ladder,
    "aft_fixed_from_014": {"alpha": alpha, "beta": beta, "gamma": gamma, "s": s, "e_ref": e_ref},
    "D23_star_no_primer": D23_noprimer,
    "cells": cells,
    "EXPLORATORY_uniform_c_scan": NK_SCAN,
    "hybrid_effective_a_pop_income_eref": eff_a,
    "hypotheses": Hy,
    "labels": "n_k を英国の人口×一人あたり所得の倍増回数とみなす対応は【推測】。技術段マイルストーン年の対応も【推測】。人口系列の England/UK 切替は【事実（OWID注記）】。",
}

# ---------- 図 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.6))
ks = [1, 2, 3]; labs = ["n12 (1290-1620)", "n23 (1620-1948)", "n34 (1948-2012)"]
x = np.arange(3); w = 0.35
ax[0].bar(x - w / 2, [nk_income[k] for k in ks], w, label="income only (014)", color="steelblue")
ax[0].bar(x + w / 2, [nk_Y[k] for k in ks], w, label="pop × income", color="darkorange")
ax[0].axhline(2 * nk_income[2], color="red", ls="--", lw=0.8, label=f"2× n23_income={2*nk_income[2]:.2f}")
ax[0].set_xticks(x); ax[0].set_xticklabels(labs, fontsize=8)
ax[0].set_ylabel("doublings of scale proxy")
ax[0].set_title("n_k: income vs population×income (GBR)")
ax[0].legend(fontsize=7)

# survival curves
for key, col, ls in [("income_eref", "steelblue", "-"), ("pop_income_eref", "darkorange", "-"),
                     ("pop_income_e1850", "green", "--")]:
    ys = [cells[key]["P_survive_grid_Piql"][str(x)] for x in D23_GRID]
    ax[1].plot(D23_GRID, ys, ls + "o", color=col, ms=3, label=key)
ax[1].axhline(0.5, color="k", ls=":", lw=0.8)
ax[1].set_xscale("log"); ax[1].set_xlabel("m23 (years, no primer)")
ax[1].set_ylabel("Piql P_survive (with hybrid primer)")
ax[1].set_title("Piql survival vs m23 under n_k definitions")
ax[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(HERE + "/fig_nk_income_vs_popincome.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.4))
order = ["income_eref", "pop_income_eref", "boundary_pop_income_eref", "income_e1850", "pop_income_e1850"]
vals_p = [num(cells[k]["D23_star_primer"]["Piql_film_QR"]) for k in order]
vals_m = [num(cells[k]["D23_star_primer"]["MDISC_BD"]) for k in order]
ax[0].barh(range(len(order)), [min(v, 1700) for v in vals_p], color="slategray")
ax[0].axvline(1100, color="r", ls="--", label="H27b threshold 1100")
ax[0].axvline(num(D23_noprimer["Piql_film_QR"]), color="purple", ls=":", label="no primer")
for i, v in enumerate(vals_p):
    ax[0].annotate(f"{v:.0f}" if v < 1e8 else ">1600", (min(v, 1700), i), fontsize=7, xytext=(3, -3), textcoords="offset points")
ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels(order, fontsize=8)
ax[0].set_xlabel("Piql D23* with hybrid primer")
ax[0].set_title("Piql D23* by n_k definition"); ax[0].legend(fontsize=7)

syn_p = [cells[k]["synergy_P_read_m23_300"]["Piql_film_QR"] for k in order]
syn_m = [cells[k]["synergy_P_read_m23_300"]["MDISC_BD"] for k in order]
ax[1].bar(np.arange(len(order)) - 0.2, syn_p, 0.4, label="Piql", color="slategray")
ax[1].bar(np.arange(len(order)) + 0.2, syn_m, 0.4, label="M-DISC", color="tan")
ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels(order, rotation=20, ha="right", fontsize=7)
ax[1].set_ylabel("ΔP_read at m23=300"); ax[1].set_title("primer synergy"); ax[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(HERE + "/fig_D23_and_synergy.png", dpi=130); plt.close()

results["runtime_seconds"] = time.time() - t0
json.dump(results, open(HERE + "/results.json", "w"), ensure_ascii=False, indent=1, default=str)

# stdout
lines = []
def p(s):
    print(s); lines.append(s)
p("check_nk_vs_014: %s" % check_nk014)
p("check_014_hybrid_ladder: %s" % check_014_ladder)
p("milestones: %s" % milestone_table)
p("n_k_income: %s" % nk_income)
p("n_k_pop_income: %s" % nk_Y)
p("n_k_boundary: %s" % nk_Y_boundary)
p("ratio Y/income: %s" % {k: nk_Y[k]/nk_income[k] for k in (1,2,3)})
for k, c in cells.items():
    p("[%s] Piql D23*=%s MDISC=%s syn300 Piql=%.4f MDISC=%.4f medT3=%.1f" % (
        k, c["D23_star_primer"]["Piql_film_QR"], c["D23_star_primer"]["MDISC_BD"],
        c["synergy_P_read_m23_300"]["Piql_film_QR"], c["synergy_P_read_m23_300"]["MDISC_BD"],
        c["median_T3_m23_300"]))
p("D23 no primer: %s" % D23_noprimer)
p("NK scan: %s" % json.dumps(NK_SCAN, default=str))
p("eff a pop×income: %s" % eff_a)
p("hypotheses: %s" % json.dumps({k: (v["supported"], {kk: vv for kk, vv in v.items() if kk != "supported"}) for k, v in Hy.items()}, default=str))
p("runtime_sec=%.1f" % results["runtime_seconds"])
open(HERE + "/stdout.txt", "w").write("\n".join(lines) + "\n")
