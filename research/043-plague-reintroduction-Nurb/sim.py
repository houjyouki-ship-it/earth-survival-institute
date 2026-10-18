"""研究043：ペストが繰り返し始める集団の大きさ N_urb の外部根拠（open_problems 125）と E の持続（126 の一部）。
事前登録：PREREGISTRATION.md。A はデータの計算（sources/excerpts.json の数値のみ）、B は 042 の sim.py を exec して
038〜042 の乱数・関数をそのまま使う。新しい乱数は seed 20261043。numpy＋matplotlib のみ、ネット接続なし。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import exp, lgamma, log

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
R = {"study": "043-plague-reintroduction-Nurb", "seed": 20261043}
rng = np.random.default_rng(20261043)

# ================= A. データ =================
# A1：大北方戦争期の町（人口、死亡割合）。derived=人口を死者と割合から逆算した町
TOWNS = [  # name, population, mortality, note
 ("Lviv 1704-05", 10000 / 0.40, 0.40, "pop derived"), ("Poznan 1707-09", 14000, 9000 / 14000, ""),
 ("Konigsberg 1709-10", 9500 / 0.25, 0.25, "pop derived"), ("Elbing 1709-10", 1200 / 0.15, 0.15, "per year; 40-day quarantine"),
 ("Damm 1709", 900, 500 / 900, ""), ("Reval 1710", 20000, 15000 / 20000, "incl. soldiers/refugees"),
 ("Stockholm 1710-11", 55000, 22000 / 55000, ""), ("Helsingfors 1710", 1185 / (2 / 3), 2 / 3, "pop derived; incl. refugees"),
 ("Abo 1710-11", 6000, 2000 / 6000, "incl. refugees"), ("Visby 1710-12", 450 / 0.2, 0.2, "pop derived"),
 ("Helsingor 1711", 3000, 1800 / 3000, ""), ("Copenhagen 1711", 60000, 17500 / 60000, "12,000-23,000 deaths, midpoint"),
 ("Groepelingen 1712", 360, 56 / 360, "isolation"), ("Bremen 1712-13", 28000, 192 / 28000, "isolation"),
 ("Rosenberg 1708", 1700, 860 / 1700, ""),
]
QUAR = {"Elbing 1709-10", "Groepelingen 1712", "Bremen 1712-13"}

def rankdata(x):
    o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
    for v in np.unique(x):  # ties -> average
        m = x == v
        if m.sum() > 1: r[m] = r[m].mean()
    return r
def spearman(x, y): return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
def spear_boot(x, y, B=10000):
    n = len(x); out = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if np.unique(x[i]).size > 2 and np.unique(y[i]).size > 2: out.append(spearman(x[i], y[i]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
lp = np.log10([t[1] for t in TOWNS]); mm = np.array([t[2] for t in TOWNS])
keep = np.array([t[0] not in QUAR for t in TOWNS])
R["A1_towns"] = {"table": [{"town": t[0], "pop": t[1], "mortality": t[2], "note": t[3]} for t in TOWNS],
                 "spearman_all": spearman(lp, mm), "ci95_all": spear_boot(lp, mm),
                 "spearman_no_quarantine": spearman(lp[keep], mm[keep]), "ci95_no_quarantine": spear_boot(lp[keep], mm[keep]),
                 "median_m_lt_5000": float(np.median(mm[(10 ** lp < 5000) & keep])), "median_m_ge_5000": float(np.median(mm[(10 ** lp >= 5000) & keep])),
                 "smallest_pop_with_m_ge_0.5": float(min(t[1] for t in TOWNS if t[2] >= 0.5))}

# A2：再来の率
SE_YEARS = [1350, 1359, 1368, 1413, 1421, 1439, 1451, 1455, 1464, 1472, 1495, 1548, 1565, 1572, 1576, 1580, 1603, 1623, 1629, 1638, 1653, 1657, 1710]
WIN = 1713 - 1350  # 363 年（Sweden の最初の流行から最後の流行の終わりまで）
def pois_ci(k, a=0.05):
    def cdf(k_, mu): return sum(exp(-mu + i * log(mu) - lgamma(i + 1)) for i in range(k_ + 1)) if mu > 0 else 1.0
    def solve(f, lo, hi):
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0: lo = mid
            else: hi = mid
        return 0.5 * (lo + hi)
    up = solve(lambda mu: cdf(k, mu) - a / 2, 1e-9, 100.0)
    lo_ = 0.0 if k == 0 else solve(lambda mu: a / 2 - (1 - cdf(k - 1, mu)), 1e-9, 100.0)  # P(X>=k) は mu で増えるので符号を反転（初回後の修正）
    return lo_, up
k_se = len(SE_YEARS) - 1; gaps = np.diff(SE_YEARS)
lam = {"Sweden": {"k": k_se, "years": WIN, "lam": k_se / WIN, "ci": [c / WIN for c in pois_ci(k_se)], "gap_median": float(np.median(gaps)), "gap_max": int(gaps.max())},
       "England": {"lam": 1 / 8.5, "range": [1 / 12, 1 / 5], "note": "intervals of 5-12 years after 1361 (source)"},
       "Russia": {"lam": 1 / 5.5, "note": "every 5-6 years 1350-1490 (042 sources)"},
       "Paris": {"lam": 1 / 3.0, "note": "about every 3 years, 16th-17th c. (042 sources)"},
       "Iceland": {"k": 2, "years": WIN, "lam": 2 / WIN, "ci": [c / WIN for c in pois_ci(2)], "pop": [40000, 60000]}}
R["A2_recurrence"] = lam

# A3：再来による人口の減り
GAP41 = 0.0250 - 0.0022
def drag(l, m): return l * -np.log(1 - m)
A3 = {}
for k in ("Sweden", "England", "Russia", "Paris", "Iceland"):
    A3[k] = {f"m={m}": drag(lam[k]["lam"], m) for m in (0.10, 0.15, 0.20)}
    A3[k]["ratio_to_041_gap_m0.15"] = A3[k]["m=0.15"] / GAP41
A3["Iceland"]["with_own_m(0.55,0.40)"] = (-np.log(1 - 0.55) - np.log(1 - 0.40)) / WIN
A3["England"]["range_lam_m0.15"] = [drag(1 / 12, 0.15), drag(1 / 5, 0.15)]
A3["gap_041"] = GAP41
R["A3_drag"] = A3

# A4：Iceland からの N_c の下限（λ(n) = λmax n/(n+Nc) で λ(n_IS) ≤ λ_IS）
A4 = {}
for src_, lmax in (("Sweden", lam["Sweden"]["lam"]), ("England", lam["England"]["lam"])):
    for nis in (40000, 50000, 60000):
        for lab, lis in (("point", lam["Iceland"]["lam"]), ("up95", lam["Iceland"]["ci"][1])):
            A4[f"{src_}_n{nis}_{lab}"] = float(nis * max(lmax / lis - 1, 0))
N_LO = min(v for k, v in A4.items() if k.endswith("up95"))
A4["N_lo_conservative"] = N_LO
A4["N_lo_point_Sweden_n50000"] = A4["Sweden_n50000_point"]
R["A4_Nc_bound"] = A4

# ================= B. 042 の模型の更新 =================
P42 = os.path.abspath(os.path.join(HERE, "..", "042-scenario-qF-plague", "sim.py"))
src = open(P42, encoding="utf8").read()
G = {"__file__": P42, "__name__": "sim042"}
exec(compile(src[:src.index("D, HV, PH, INFO = evaluate()")], P42, "exec"), G)
exec(compile(src[src.index("def sweep2"):src.index("SWm = sweep2()")], P42, "exec"), G)
T = G["T"]; N = G["N"]; lu = G["lu"]; U7 = G["U7"]; weighted = G["weighted"]; breakeven = G["breakeven"]; evaluate = G["evaluate"]
r42 = json.load(open(os.path.join(HERE, "..", "042-scenario-qF-plague", "results.json")))
times2_orig = G["times2"]
OPT = {"soft": False, "DE": np.inf}

def times3(st, sv, lit, rF, rE, Nurb):
    if not OPT["soft"] and np.isinf(OPT["DE"]): return times2_orig(st, sv, lit, rF, rE, Nurb)
    n = np.maximum(sv["nmax"], 1.0); rF = np.broadcast_to(rF, n.shape); rE = np.broadcast_to(rE, n.shape)
    c = np.broadcast_to(np.asarray(Nurb, dtype=float), n.shape)
    def wait(Nk):
        top = np.maximum(Nk, n)
        if OPT["soft"]:
            Bc = 1 - rE / rF; cf = np.where(np.isinf(c), 0.0, c)
            Wf = lambda x: np.log(x) / rF + Bc / rE * np.log(rE * x + rF * cf)
            return np.where(np.isinf(c), np.log(top / n) / rF, Wf(top) - Wf(n))
        a = np.clip(c, n, top); up = np.log(top / a)
        tE = up / rE; DE = OPT["DE"]
        return np.log(a / n) / rF + np.where(tE <= DE, tE, DE + (up - rE * DE) / rF)
    d01e = np.where(T["U_lit"] < lit, T["d01"] * T["lit_mult"], T["d01"])
    tFe = np.where(st == 6, np.maximum(T["d_Fe"], wait(T["N_Fe"])), 0.0)
    t1 = np.where(st >= 5, np.maximum(tFe + d01e, wait(T["N_lens"])), 0.0)
    t2 = np.where(st >= 4, np.maximum(t1 + T["d12"], wait(T["N_T2e"])), 0.0)
    wF = np.log(np.clip(c, n, np.maximum(T["N_T2e"], n)) / n) / rF
    return np.select([st == 2, st == 3, st >= 4], [T["W"], np.zeros(N), t2], default=np.zeros(N)), wait(T["N_T2e"]), wF
G["times2"] = times3

Nurb042 = G["N_urb_main"].copy()
def run(prior_lo=1e4, soft=False, DE=np.inf, keys=("U3", "CASC", "BIO"), cfg=None):
    OPT["soft"] = soft; OPT["DE"] = DE
    G["N_urb_main"] = lu(U7[6], prior_lo, 1e7)
    out = evaluate(cfg, keys=keys)
    OPT["soft"] = False; OPT["DE"] = np.inf; G["N_urb_main"] = Nurb042
    return out

# 042 の再現
D0, H0, P0, I0 = run(1e4)
ref42 = r42["main"]["per_heavy_event"]["U3"]["TAB"]
R["check_042"] = {"U3_TAB": P0["U3"]["TAB"], "U3_TAB_042": ref42, "rel_err": abs(P0["U3"]["TAB"] - ref42) / ref42,
                  "reproduced": bool(abs(P0["U3"]["TAB"] - ref42) / ref42 < 1e-9)}

def summary(D, HV, PH, INFO):
    W = weighted(D, HV, anchors=("A_Ord", "A_XPT_S"))
    return {"U3_TAB": PH["U3"]["TAB"], "named_TAB": PH["named_pooled"]["TAB"], "CASC_TAB": PH["CASC"]["TAB"], "BIO_TAB": PH["BIO"]["TAB"],
            "be_U3": breakeven(PH), "qF_eff_U3": INFO["U3"]["qF_eff_heavy"], "waitF_share_U3": INFO["U3"]["wait_F_share_median"],
            "w_AOrd_U3_TAB": W["A_Ord"]["U3"]["TAB"]["total"], "w_AXPTS_U2_TAB": W["A_XPT_S"]["U2"]["TAB"]["total"],
            "w_AOrd_U3_GROW0.5_over_TAB": W["A_Ord"]["U3"]["GROW_0.5"]["total"] / W["A_Ord"]["U3"]["TAB"]["total"]}

S = {}
S["B0_042_prior_threshold"] = summary(D0, H0, P0, I0)
S["B1_Nlo_threshold"] = summary(*run(N_LO))
S["B1p_Nlo_point_threshold"] = summary(*run(A4["N_lo_point_Sweden_n50000"]))
S["B2_Nlo_soft"] = summary(*run(N_LO, soft=True))
S["B2_042prior_soft"] = summary(*run(1e4, soft=True))
for de in (600, 370, 200):
    S[f"B3_Nlo_DE{de}"] = summary(*run(N_LO, DE=de))
S["B3_042prior_DE370"] = summary(*run(1e4, DE=370))
R["B"] = S

# n*（U3、B1 の事前）
def nstar(lo, soft=False, DE=np.inf):
    OPT["soft"] = soft; OPT["DE"] = DE; G["N_urb_main"] = lu(U7[6], lo, 1e7)
    sw = G["sweep2"](); OPT["soft"] = False; OPT["DE"] = np.inf; G["N_urb_main"] = Nurb042
    return sw
SW0 = nstar(1e4); SW1 = nstar(N_LO); SW1s = nstar(N_LO, soft=True); SW1d = nstar(N_LO, DE=370)
R["n_star"] = {"042_prior": SW0["n_star"], "042_reported": r42["n_star"]["main"], "B1": SW1["n_star"], "B2_soft": SW1s["n_star"], "B3_DE370": SW1d["n_star"]}

# 固定の N_urb での U3 の TAB（042 と同じ固定値、閾値と連続）
FIX = {}
for nu in (1e4, 1e5, 3e5, 1e6, 1e7):
    FIX[str(nu)] = {"threshold": run(1e4, keys=("U3",), cfg={"Nurb_fixed": nu})[2]["U3"]["TAB"],
                    "soft": run(1e4, soft=True, keys=("U3",), cfg={"Nurb_fixed": nu})[2]["U3"]["TAB"]}
R["fixed_Nurb_U3_TAB"] = FIX

# ================= 仮説 =================
b0 = S["B0_042_prior_threshold"]; b1 = S["B1_Nlo_threshold"]; b2 = S["B2_Nlo_soft"]; b3 = S["B3_Nlo_DE370"]
dS = A3["Sweden"]["m=0.15"]; dE = A3["England"]["m=0.15"]
R["hypotheses"] = {
 "H55a": {"value": [R["A1_towns"]["spearman_all"], R["A1_towns"]["ci95_all"]], "rule": "Spearman(log pop, mortality) <= 0", "supported": bool(R["A1_towns"]["spearman_all"] <= 0)},
 "H55b": {"value": [dS, dE, GAP41], "rule": "drag(m=0.15) in [0.5,2]x gap for Sweden AND England", "supported": bool(all(0.5 * GAP41 <= d <= 2 * GAP41 for d in (dS, dE)))},
 "H55c": {"value": N_LO, "rule": "N_lo >= 1e5", "supported": bool(N_LO >= 1e5)},
 "H55d": {"value": [b1["U3_TAB"], ref42, b1["U3_TAB"] / ref42], "rule": "B1 U3 TAB >= 1.15 x 042", "supported": bool(b1["U3_TAB"] >= 1.15 * ref42)},
 "H55e": {"value": [b2["U3_TAB"], b1["U3_TAB"], abs(b2["U3_TAB"] - b1["U3_TAB"]) / b1["U3_TAB"]], "rule": "|soft-threshold|/threshold < 0.15", "supported": bool(abs(b2["U3_TAB"] - b1["U3_TAB"]) / b1["U3_TAB"] < 0.15)},
 "H55f": {"value": [b3["U3_TAB"], b1["U3_TAB"], b3["U3_TAB"] / b1["U3_TAB"]], "rule": "DE=370 raises U3 TAB by >=10%", "supported": bool(b3["U3_TAB"] >= 1.10 * b1["U3_TAB"])},
 "H55g": {"value": [b1["w_AOrd_U3_TAB"], r42["main"]["weighted"]["A_Ord"]["U3"]["TAB"], b1["w_AOrd_U3_TAB"] / r42["main"]["weighted"]["A_Ord"]["U3"]["TAB"]], "rule": "weighted A_Ord U3 TAB within +-30% of 042", "supported": bool(0.7 <= b1["w_AOrd_U3_TAB"] / r42["main"]["weighted"]["A_Ord"]["U3"]["TAB"] <= 1.3)},
}

# ================= 図 =================
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for t in TOWNS:
    c_ = "C3" if t[0] in QUAR else "C0"
    ax[0].scatter(t[1], t[2], color=c_); ax[0].annotate(t[0].split(" ")[0], (t[1], t[2]), fontsize=6, xytext=(3, 2), textcoords="offset points")
ax[0].set_xscale("log"); ax[0].set_xlabel("town population"); ax[0].set_ylabel("plague mortality share")
ax[0].set_title(f"A1: 1704-1713 towns (red = isolation/quarantine)\nSpearman all {R['A1_towns']['spearman_all']:.2f}, no-quarantine {R['A1_towns']['spearman_no_quarantine']:.2f}", fontsize=9)
ks = ["Iceland", "Sweden", "England", "Russia", "Paris"]
x = np.arange(len(ks))
ax[1].bar(x, [A3[k]["m=0.15"] * 100 for k in ks], color="C1", label="drag at m=0.15")
ax[1].errorbar(x, [A3[k]["m=0.15"] * 100 for k in ks], yerr=[[100 * (A3[k]["m=0.15"] - A3[k]["m=0.1"]) for k in ks], [100 * (A3[k]["m=0.2"] - A3[k]["m=0.15"]) for k in ks]], fmt="none", color="k")
ax[1].axhline(GAP41 * 100, color="k", ls="--", label="041: r_F - r_E = 2.28 pt")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"{k}\nλ={lam[k]['lam']:.3f}/yr" for k in ks], fontsize=8)
ax[1].set_ylabel("growth-rate loss from recurrences (%/yr)"); ax[1].legend(fontsize=8); ax[1].set_title("A3: recurrence drag vs the E-F gap", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_towns_drag.png"), dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
labs = ["B0_042_prior_threshold", "B1_Nlo_threshold", "B1p_Nlo_point_threshold", "B2_Nlo_soft", "B3_Nlo_DE600", "B3_Nlo_DE370", "B3_Nlo_DE200"]
ax[0].barh(range(len(labs)), [S[l]["U3_TAB"] for l in labs], color="C2")
ax[0].axvline(ref42, color="k", ls=":", label="042")
ax[0].set_yticks(range(len(labs))); ax[0].set_yticklabels([l.replace("_", " ") for l in labs], fontsize=7)
ax[0].set_xlabel("tablet-guide value per U3 heavy event"); ax[0].legend(fontsize=8); ax[0].set_title("B: N_urb prior, soft curve, E persistence", fontsize=9)
nus = [float(k) for k in FIX]
ax[1].semilogx(nus, [FIX[k]["threshold"] for k in FIX], "k-o", label="threshold (042)")
ax[1].semilogx(nus, [FIX[k]["soft"] for k in FIX], "C1-s", label="soft r(n)")
ax[1].axvline(N_LO, color="C3", ls="--", label=f"Iceland lower bound N_lo={N_LO:,.0f}")
ax[1].set_xlabel("N_urb (fixed)"); ax[1].set_ylabel("U3 TAB"); ax[1].legend(fontsize=8); ax[1].set_title("U3 TAB vs N_urb", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_model_update.png"), dpi=120); plt.close()

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=1, default=float)
out = {k: R[k] for k in R if k not in ("A1_towns",)}
out["A1"] = {k: v for k, v in R["A1_towns"].items() if k != "table"}
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
print("runtime", round(R["runtime_seconds"], 1))
