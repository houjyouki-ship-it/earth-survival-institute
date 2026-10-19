"""研究059：書かれた手順が伝達の損失を減らす割合 s を伝達の連鎖の実験で置き直す（open_problems 157）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md。
058 の sim.py を「# ---------- 仮説 ----------」の前まで exec し（058→057→056→055→054→… の乱数をそのまま使う）、
s の分布だけを差し替えて F を出し直す。新しい乱数は seed 20261059。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P58 = os.path.abspath(os.path.join(HERE, "..", "058-literacy-threshold", "sim.py"))
src = open(P58, encoding="utf8").read(); cut = src.index("# ---------- 仮説 ----------")
G = {"__file__": P58, "__name__": "sim058"}
exec(compile(src[:cut], P58, "exec"), G)
r58 = json.load(open(os.path.join(HERE, "..", "058-literacy-threshold", "results.json")))
R = {"study": "059-written-transmission-s", "seed": 20261059}
N, NK, EPS = G["N"], G["NK"], G["EPS"]
F3, thr_arrays, psi_main, s_main, p_early_main, U_st = G["F3"], G["thr_arrays"], G["psi_main"], G["s_main"], G["p_early_main"], G["U_st"]
KEYS = G["KEYS"]
R["check_058"] = max(abs(G["RES"][v][k]["F_total"] - r58["F"][v][k]["F_total"]) for v in r58["F"] for k in r58["F"][v])

EV = {s["id"]: s for s in json.load(open(os.path.join(HERE, "sources", "evidence.json")))["studies"]}
lg = lambda p: np.log(p / (1 - p))
g = np.random.default_rng(20261059)

# ---------- 1. Morgan 2015：対面の言葉の教示（VT）と模倣・エミュレーション（IE）の位置1 ----------
def lognorm_ci(med, lo, hi, n):   # 95% CI から logit 正規で近似
    m = lg(med); sd = (lg(hi) - lg(lo)) / 3.92
    return m + sd * g.standard_normal(n)
M = EV["Morgan2015"]["data"]
VT_hit = lognorm_ci(0.10, 0.07, 0.16, N); IE_hit = lognorm_ci(0.04, 0.03, 0.06, N)
E_hit = g.uniform(lg(0.15), lg(0.60), N)                      # 専門家の値【推測】
sM_hit = np.clip(1 - (E_hit - VT_hit) / np.maximum(E_hit - IE_hit, 1e-9), 0, 1)
VT_v = lognorm_ci(0.73, 0.68, 0.78, N); IE_v = lognorm_ci(0.58, 0.52, 0.64, N)
E_v = lg(g.uniform(0.80, 0.97, N))
sM_v = np.clip(1 - (E_v - VT_v) / np.maximum(E_v - IE_v, 1e-9), 0, 1)
Egrid = np.linspace(lg(0.12), lg(0.7), 60)
sE_point = 1 - (Egrid - lg(0.10)) / (Egrid - lg(0.04))
q = lambda x: [float(np.quantile(x, a)) for a in (0.05, 0.5, 0.95)]
R["morgan"] = {"s_M_hit_q": q(sM_hit), "s_M_viable_q": q(sM_v),
               "s_point_E0.3": float(1 - (lg(0.3) - lg(0.10)) / (lg(0.3) - lg(0.04))),
               "pos5_VT_logit_check": float(1 / (1 + np.exp(-(lg(0.10) + 4 * M["grad_p_hit"]["VT"])))),
               "share_of_gap_above_floor_lost_per_pos": {k: float(-M["grad_p_hit"][k] / (lg(M["p_hit_pos1"][k]) - lg(M["p_hit_pos1"]["RE"])))
                                                        for k in ("IE", "BT", "GT", "VT")}}

# ---------- 2. 書いた・言葉の経路が実演だけより良かった実験の割合 w ----------
codes = {k: EV[k]["code"] for k in ("Morgan2015", "Zwirner2015", "HSSC2021", "Derex2019", "Bietti2019")}
a_w = 0.5 + sum(codes.values()); b_w = 0.5 + len(codes) - sum(codes.values())
w = g.beta(a_w, b_w, N)
s_low = g.uniform(0, 0.1, N)
s_mix = np.where(g.uniform(size=N) < w, sM_hit, s_low)
R["w"] = {"codes": codes, "beta": [a_w, b_w], "w_q": q(w)}
R["s_q"] = {"s_058": q(s_main), "s_M_hit": q(sM_hit), "s_M_viable": q(sM_v), "s_mix": q(s_mix), "s_low": q(s_low),
            "mean": {"s_058": float(s_main.mean()), "s_mix": float(s_mix.mean()), "s_M_hit": float(sM_hit.mean())}}

# ---------- 3. Derex 2019：書いた理論の効き ----------
D = EV["Derex2019"]["data"]
R["derex"] = {"gen_x_theory": D["gen_x_theory"], "gen_config": D["gen_config"],
              "ci_includes_0": D["gen_x_theory"]["ci"][0] < 0 < D["gen_x_theory"]["ci"][1],
              "upper_over_gen": D["gen_x_theory"]["ci"][1] / D["gen_config"]["mean"],
              "mean_over_gen": D["gen_x_theory"]["mean"] / D["gen_config"]["mean"]}

# ---------- 4. Muthukrishna 2014：一定の α の検証と床への戻り ----------
S = EV["Muthukrishna2014"]["data"]["exp2_slope_SD"]
ratio = {k: S[f"{k}_4_10"][0] / S[f"{k}_1_3"][0] for k in ("one", "five")}
def ols_slope(x, y): x = np.asarray(x, float); return float(np.polyfit(x, y, 1)[0])
def ratio_rho(rho):
    gg = np.arange(1, 11); z = rho ** (gg - 1)
    return ols_slope(gg[3:], z[3:]) / ols_slope(gg[:3], z[:3])
def solve_rho(target):
    lo, hi = 1e-6, 1 - 1e-6
    for _ in range(200):
        mid = (lo + hi) / 2
        if ratio_rho(mid) < target: lo = mid
        else: hi = mid
    return (lo + hi) / 2
rho_one = solve_rho(ratio["one"]); rho_five = solve_rho(ratio["five"])
# 標準誤差で揺らした比
e1 = g.normal(S["one_1_3"][0], S["one_1_3"][1], N); l1 = g.normal(S["one_4_10"][0], S["one_4_10"][1], N)
e5 = g.normal(S["five_1_3"][0], S["five_1_3"][1], N); l5 = g.normal(S["five_4_10"][0], S["five_4_10"][1], N)
P_late_flatter = float(np.mean(np.abs(l1) < np.abs(e1)))
# Henrich の式での α/β と実験の閾値 N*（前半）：N* = 5^(−Δ1/(Δ5−Δ1))
d1, d5 = S["one_1_3"][0], S["five_1_3"][0]
Nstar_point = 5 ** (-d1 / (d5 - d1))
dd = e5 - e1; ok = dd > 0
with np.errstate(over="ignore"):
    Nstar = np.where(ok, np.minimum(5.0 ** np.minimum(-e1 / np.where(ok, dd, 1), 400), 1e12), 1e12)   # 傾きが逆なら閾値なし→1e12 として数える
x1 = EV["Muthukrishna2014"]["data"]["exp1_slope_SD"]
Nstar_exp1 = 5 ** (-x1["one"] / (x1["five"] - x1["one"]))
R["muthukrishna"] = {"ratio_late_early": ratio, "rho_one": rho_one, "a_one": 1 - rho_one, "rho_five": rho_five, "a_five": 1 - rho_five,
                     "P_late_flatter_one": P_late_flatter, "Nstar_exp2_point": Nstar_point,
                     "Nstar_exp2_q": [float(np.quantile(Nstar, a)) for a in (0.05, 0.5, 0.95)], "P_Nstar_gt_1000": float(np.mean(Nstar > 1000)), "share_Nstar_undefined": float(1 - ok.mean()),
                     "alpha_over_beta_exp2": float(np.log(Nstar_point) + EPS), "Nstar_exp1_point_written_notes": Nstar_exp1}

# ---------- 5. 床からの差の模型でも閾値の比は同じ形か（H71d） ----------
aa, Dd, bb, ss = g.uniform(0.1, 1, 1000), g.uniform(0.5, 5, 1000), g.uniform(0.05, 1, 1000), g.uniform(0, 1, 1000)
lnN = aa * Dd / bb - EPS; lnN2 = (1 - ss) * aa * Dd / bb - EPS
rat_gap = np.exp(lnN2 - lnN); rat_hen = (np.exp(lnN) * np.exp(EPS)) ** (-ss)
R["gap_model_check_max_rel"] = float(np.max(np.abs(rat_gap / rat_hen - 1)))

# ---------- 6. F の出し直し ----------
def thr_linear(psi, s):
    out = []
    for j in range(3):
        hit = U_st[j] < psi
        out.append(np.where(hit, np.maximum(1 - s, 1e-9), 1.0))
    return out
PE = {"main": p_early_main, "one": 1.0}
VERS = {"s_058": s_main, "s_M_hit": sM_hit, "s_M_viable": sM_v, "s_mix": s_mix, "s_low": s_low}
RES = {}
for vn, s in VERS.items():
    RES[f"{vn}/psi_main/main"] = F3(thr_arrays(psi_main, s), PE["main"])
    RES[f"{vn}/psi1/one"] = F3(thr_arrays(1.0, s), PE["one"])
for vn in ("s_058", "s_mix", "s_M_hit"):
    RES[f"linear_{vn}/psi_main/main"] = F3(thr_linear(psi_main, VERS[vn]), PE["main"])
    RES[f"linear_{vn}/psi1/one"] = F3(thr_linear(1.0, VERS[vn]), PE["one"])
R["F"] = RES
R["check_F_s058_vs_058"] = max(abs(RES["s_058/psi_main/main"][k]["F_total"] - r58["F"]["main/main"][k]["F_total"]) for k in RES["s_058/psi_main/main"])
BE = r58["CE_with_F"]["A_Ord/U3"]["breakeven_F"]
CE0 = {k: r58["CE_with_F"][k]["CH_heavy_TC_054value"] for k in r58["CE_with_F"]}
R["CE_with_F"] = {k: {"base": CE0[k], "x_F_s_mix": CE0[k] * RES["s_mix/psi_main/main"][k]["F_total"],
                      "x_F_s_M_psi1_one": CE0[k] * RES["s_M_hit/psi1/one"][k]["F_total"],
                      "CH_heavy_STC": r58["CE_with_F"][k]["CH_heavy_STC"], "breakeven_F": BE} for k in CE0}

# ---------- 仮説 ----------
Hh = {}
v = R["morgan"]["s_M_hit_q"][1]; Hh["H71a"] = {"value": v, "rule": "median s_M (p_hit) < 0.5", "supported": v < 0.5}
Hh["H71b"] = {"value": ratio, "rule": "late/early slope ratio < 0.3 in both treatments", "supported": all(r < 0.3 for r in ratio.values())}
Hh["H71c"] = {"value": 1 - rho_one, "rule": "a = 1 - rho (one-model) >= 0.3", "supported": 1 - rho_one >= 0.3}
v = R["gap_model_check_max_rel"]; Hh["H71d"] = {"value": v, "rule": "gap-model threshold ratio equals Henrich ratio (rel < 1e-9)", "supported": v < 1e-9}
v = RES["s_mix/psi_main/main"]["A_Ord/U3"]["F_total"]; Hh["H71e"] = {"value": v, "rule": "F (s_mix, psi main, p_early main, A_Ord/U3) < 058 main", "supported": v < r58["F"]["main/main"]["A_Ord/U3"]["F_total"]}
v = RES["linear_s_mix/psi_main/main"]["A_Ord/U3"]["F_total"]; Hh["H71f"] = {"value": v, "rule": "linear thr, s_mix: F < 1.05", "supported": v < 1.05}
v = RES["s_M_hit/psi1/one"]["A_Ord/U3"]["F_total"]; Hh["H71g"] = {"value": v, "rule": "s_M, psi1, p_early 1: F < 5.0", "supported": v < 5.0}
Hh["H71h"] = {"value": R["derex"], "rule": "Derex theory effect CI includes 0 and upper < generation effect", "supported": R["derex"]["ci_includes_0"] and R["derex"]["upper_over_gen"] < 1}
R["hypotheses"] = Hh
R["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
bins = np.linspace(0, 1, 41)
for nm, x, c in (("058 prior Beta(3.5,4.5)", s_main, "gray"), ("s_M p_hit (Morgan VT vs IE)", sM_hit, "C0"), ("s_M viable", sM_v, "C2"), ("s_mix (main)", s_mix, "C3")):
    axs[0].hist(x, bins=bins, density=True, histtype="step", lw=1.5, label=nm, color=c)
axs[0].set_xlabel("s = share of per-generation transmission loss removed"); axs[0].set_ylabel("density"); axs[0].legend(fontsize=7)
axs[0].set_title("059: s priors from transmission-chain experiments", fontsize=9)
axs[1].plot(1 / (1 + np.exp(-Egrid)), sE_point, "C0"); axs[1].axvspan(0.15, 0.6, color="C0", alpha=0.1)
axs[1].set_xlabel("expert's P(viable flake per hit) (not reported; assumed range shaded)"); axs[1].set_ylabel("s = 1 - (E-VT)/(E-IE), logit")
axs[1].set_title("Morgan 2015 Table 1 position 1: VT 0.10, IE 0.04", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_s_priors.png"), dpi=120); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
names = list(RES.keys())
for ax, (a, u) in zip(axs, KEYS):
    key = f"{a}/{u}"; y = np.arange(len(names))
    ax.barh(y - 0.2, [RES[n][key]["F_total"] for n in names], height=0.4, label="F total", color="C0")
    ax.barh(y + 0.2, [RES[n][key]["F_heavy"] or 0 for n in names], height=0.4, label="F heavy part", color="C3")
    ax.axvline(BE, color="k", lw=1); ax.axvline(1, color="gray", lw=0.6); ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7); ax.set_title(key, fontsize=9); ax.set_xlabel("F = value(T+C with manual lowering thresholds) / value(T+C, 054)")
axs[0].legend(fontsize=7)
fig.suptitle("059: F with s re-estimated from transmission chains (black line: 056 break-even 5.0)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_F_versions.png"), dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
gg = np.arange(1, 11)
for k, rho, c in (("one", rho_one, "C1"), ("five", rho_five, "C2")):
    e = S[f"{k}_1_3"][0]
    # 前半の OLS 傾きが e になるように振幅を合わせた床への戻り
    amp = e / ols_slope(gg[:3], rho ** (gg[:3] - 1))
    ax.plot(gg, amp * rho ** (gg - 1) - amp * rho ** 9, "-", color=c, label=f"{k}-model: floor-reversion fit rho={rho:.2f}")
    ax.plot(gg, e * (gg - 1) - e * 9, ":", color=c, label=f"{k}-model: constant alpha (slope {e})")
ax.set_xlabel("generation"); ax.set_ylabel("skill relative to gen-10 level (SD)"); ax.legend(fontsize=7)
ax.set_title("Muthukrishna 2014 exp.2 (knots): early vs late slopes", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_constant_alpha_test.png"), dpi=120); plt.close(fig)

json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: R[k] for k in ("check_058", "check_F_s058_vs_058", "morgan", "w", "s_q", "derex", "muthukrishna", "gap_model_check_max_rel", "CE_with_F")}, indent=1, ensure_ascii=False, default=float))
for k, v in RES.items():
    print(k, {kk: (round(vv["F_total"], 4), round(vv["F_heavy"], 4) if vv["F_heavy"] else None) for kk, vv in v.items()})
for k, h in Hh.items(): print(k, h["supported"], json.dumps(h["value"], default=float)[:200])
print("runtime", round(R["runtime_s"], 1))
