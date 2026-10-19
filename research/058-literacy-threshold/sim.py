"""研究058：書かれた知識は段の閾値（N_Fe・N_lens・N_T2e）を下げるか、下げるなら T＋C の価値を何倍にするか（open_problems 155）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md。
057 の sim.py を「FS = [1.0」の前まで exec し（057→056→055→054→040→039→038 の乱数と 057 の p_early・f_ev をそのまま使う）、
段ごとの閾値の倍率を Henrich 2004 の式 (3) で置く。新しい乱数は seed 20261058。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P57 = os.path.abspath(os.path.join(HERE, "..", "057-pre-reader-knowledge", "sim.py"))
src = open(P57, encoding="utf8").read(); cut = src.index("FS = [1.0")
G = {"__file__": P57, "__name__": "sim057"}
exec(compile(src[:cut], P57, "exec"), G)
r57 = json.load(open(os.path.join(HERE, "..", "057-pre-reader-knowledge", "results.json")))
R = {"study": "058-literacy-threshold", "seed": 20261058}
B, COMPS, SRC, SCEN, FIXED, ns = G["B"], G["COMPS"], G["SRC"], G["SCEN"], G["FIXED"], G["ns"]
N = G["N"]; agg = G["agg"]; V_m = G["V_m"]; V0 = G["V0"]; VTC = G["VTC"]; base_val = G["base_val"]; KEYS = G["KEYS"]
tab_ok = G["tab_ok"]; U_early = G["U_early"]; p_early_main = G["p_early_main"]; F_EV = G["F_EV"]; q_TC = G["q_TC"]
U_lit, d01, d12, lit_mult, d_Fe, W = (ns[k] for k in ("U_lit", "d01", "d12", "lit_mult", "d_Fe", "W"))
N_Fe, N_lens, N_T2e, r_pop = (ns[k] for k in ("N_Fe", "N_lens", "N_T2e", "r_pop"))

# 057 の値を差0で再現するか
chk = {}
for name, args in (("extreme_f0_thr1_one", (0.0, 0.0, 0.0, 1.0, 1.0)), ("extreme_f0_thr0.1_one", (0.0, 0.0, 0.0, 0.1, 1.0)),
                   ("extreme_f0_thr0.01_one", (0.0, 0.0, 0.0, 0.01, 1.0))):
    o = G["F_of"](*args)
    chk[name] = max(abs(o[k]["F_total"] - r57[name][k]["F_total"]) for k in o)
o = G["F_of"](F_EV["fFe"], F_EV["f01"], F_EV["f12"], 1.0, p_early_main)
chk["F_ev/pre12/main"] = max(abs(o[k]["F_total"] - r57["F_ev"]["pre12/main"][k]["F_total"]) for k in o)
R["check_057"] = chk

# ---------- 事例（sources/cases.json） ----------
CS = json.load(open(os.path.join(HERE, "sources", "cases.json")))["cases"]
Lf = [c for c in CS if c["group"] == "literate" and c["kind"] == "failure"]
Ls = [c for c in CS if c["group"] == "literate" and c["kind"] == "success"]
Nf = [c for c in CS if c["group"] == "nonliterate"]
kL = float(np.mean([c["k"] for c in Lf])); kN = float(np.mean([c["k"] for c in Nf]))
pers = sum(c["carrier"] in ("person", "person+book") for c in Ls) / len(Ls)
book_only = sum(c["carrier"] == "book" for c in Ls)
R["cases"] = {"k_literate_failures": {c["id"]: c["k"] for c in Lf}, "k_mean_literate": kL,
              "k_nonliterate_losses": {c["id"]: c["k"] for c in Nf}, "k_mean_nonliterate": kN,
              "literate_success_carrier": {c["id"]: c["carrier"] for c in Ls}, "share_person_carried": pers, "n_book_only_success": book_only,
              "sens_k_mean_literate_Iceland0": float(np.mean([0.0 if c["id"] == "Iceland_iron" else c["k"] for c in Lf])),
              "sens_k_mean_literate_Iceland1": float(np.mean([1.0 if c["id"] == "Iceland_iron" else c["k"] for c in Lf]))}
kN_sum = sum(c["k"] for c in Nf); nN = len(Nf)
c57 = json.load(open(os.path.join(HERE, "..", "057-pre-reader-knowledge", "sources", "cases.json")))["cases"]
bk = [c for c in c57 if c["mode"] == "book"]; s_full = sum(c["outcome"] == "full" for c in bk); s_n = len(bk)
R["priors"] = {"psi_beta": [kN_sum + 0.5, nN - kN_sum + 0.5], "s_beta": [s_full + 0.5, s_n - s_full + 0.5]}

# ---------- 新しい乱数 ----------
g = np.random.default_rng(20261058)
EPS = 0.5772156649
psi_main = g.beta(kN_sum + 0.5, nN - kN_sum + 0.5, N)
psi_unif = g.uniform(size=N)
s_main = g.beta(s_full + 0.5, s_n - s_full + 0.5, N)
U_st = g.uniform(size=(3, N))        # 段ごとの当たり
U_one = g.uniform(size=N)            # 3段で共有する当たり
NK = [N_Fe, N_lens, N_T2e]

def thr_arrays(psi, s, lam=0.2, corr=False, only=None):
    out = []
    for j, Nk in enumerate(NK):
        hit = (U_one if corr else U_st[j]) < psi
        t = np.minimum(1.0, (lam * Nk * np.exp(EPS)) ** (-s))
        t = np.where(hit, t, 1.0)
        if only is not None and j != only: t = np.ones(N)
        out.append(t)
    return out

def T2s_m3(k, thrs, fFe=1.0, f01=1.0, f12=1.0):
    st = B[k]["st"]
    if k in FIXED:   # 人口の模型がない成分：閾値は効かない（057 と同じ）
        lit = SCEN[k]["lit"]; d01e = np.where(U_lit < lit, d01 * lit_mult, d01) * f01
        return np.select([st == 3, st == 4, st == 5, st == 2], [np.zeros(N), d12 * f12, d01e + d12 * f12, W], default=np.zeros(N))
    sv = SRC[k]["sv"]; lit = SRC[k]["lit"]
    n = np.maximum(sv["nmax"], 1.0); r = r_pop * (r_pop / r_pop)
    wait = lambda Nk, t: np.maximum(0.0, np.log(t * Nk / n) / r)
    d01e = np.where(U_lit < lit, d01 * lit_mult, d01) * f01
    tFe = np.where(st == 6, np.maximum(d_Fe * fFe, wait(N_Fe, thrs[0])), 0.0)
    t1 = np.where(st >= 5, np.maximum(tFe + d01e, wait(N_lens, thrs[1])), 0.0)
    t2 = np.where(st >= 4, np.maximum(t1 + d12 * f12, wait(N_T2e, thrs[2])), 0.0)
    return np.select([st == 2, st == 3, st >= 4], [W, np.zeros(N), t2], default=np.zeros(N))
ONE = [np.ones(N)] * 3
R["check_T2s_m3"] = max(float(np.abs(T2s_m3(k, ONE) - B[k]["T2s"]).max()) for k in COMPS)

def F3(thrs, p_early, f=(1.0, 1.0, 1.0)):
    man = tab_ok & (U_early < p_early)
    D = {}; binds = {}
    for k in COMPS:
        T2 = np.where(man, T2s_m3(k, thrs, *f), B[k]["T2s"])
        D[k] = V_m(k, q_TC, True, T2) - V0[k]
    val = agg({"x": D}, anchors=("A_Ord", "A_XPT_S"))
    out = {}
    for a, u in KEYS:
        b = base_val[a][u]["TC"]; v = val[a][u]["x"]
        out[f"{a}/{u}"] = {"F_total": v["total"] / b["total"], "F_heavy": v["heavy"] / b["heavy"] if b["heavy"] > 0 else None,
                           "F_light": (v["total"] - v["heavy"]) / (b["total"] - b["heavy"])}
    return out

VERS = {"main": dict(psi=psi_main, s=s_main), "corr": dict(psi=psi_main, s=s_main, corr=True), "psi_unif": dict(psi=psi_unif, s=s_main),
        "psi1": dict(psi=1.0, s=s_main), "psi1_s1": dict(psi=1.0, s=1.0), "lam1": dict(psi=psi_main, s=s_main, lam=1.0)}
PE = {"main": p_early_main, "one": 1.0}
RES = {}; THRQ = {}
for vn, kw in VERS.items():
    th = thr_arrays(**kw)
    THRQ[vn] = {nm: [float(np.quantile(t, q)) for q in (0.05, 0.5, 0.95)] for nm, t in zip(("N_Fe", "N_lens", "N_T2e"), th)}
    for pn, pv in PE.items():
        RES[f"{vn}/{pn}"] = F3(th, pv)
th = thr_arrays(psi_main, s_main)
for pn, pv in PE.items():
    RES[f"main_fev/{pn}"] = F3(th, pv, f=(F_EV["fFe"], F_EV["f01"], F_EV["f12"]))
R["F"] = RES; R["thr_quantiles"] = THRQ
# 知識が縛りの段だけの thr（H70d）
t_hit = np.minimum(1.0, (0.2 * N_Fe * np.exp(EPS)) ** (-s_main))
R["thr_Fe_if_knowledge_bound_q"] = [float(np.quantile(t_hit, q)) for q in (0.05, 0.5, 0.95)]
# 段ごとの分解（psi1, p_early=1）
DEC = {}
for j, nm in enumerate(("N_Fe", "N_lens", "N_T2e")):
    DEC[nm] = F3(thr_arrays(1.0, s_main, only=j), 1.0)
R["decompose_psi1_one"] = DEC
# 閾値を下げた後の人口の待ち（U3 の重い崩壊、psi1・s 主）
th1 = thr_arrays(1.0, s_main)
WA = {}
for k in ("U3", "BIO", "CASC"):
    if k not in COMPS: continue
    st = B[k]["st"]; heavy = st >= 4
    a0 = T2s_m3(k, ONE); a1 = T2s_m3(k, th1); a2 = T2s_m3(k, th1, 0.0, 0.0, 0.0)
    WA[k] = {"T2_median_base": float(np.median(a0[heavy])), "T2_median_thr": float(np.median(a1[heavy])),
             "T2_median_thr_and_f0": float(np.median(a2[heavy])), "share_heavy": float(heavy.mean())}
R["T2_heavy_medians"] = WA

# ---------- 費用あたり（057 の表に F を掛ける） ----------
CEC = r57["CE_compare_median"]; BE = r57["breakeven_056"]["V_mix"]
R["CE_with_F"] = {key: {"CH_heavy_TC_054value": CEC[key]["CH_heavy_TC_054value"],
                        "x_F_main": CEC[key]["CH_heavy_TC_054value"] * RES["main/main"][key]["F_total"],
                        "x_F_psi1_s1_one": CEC[key]["CH_heavy_TC_054value"] * RES["psi1_s1/one"][key]["F_total"],
                        "CH_heavy_STC": CEC[key]["CH_heavy_STC"], "breakeven_F": BE} for key in CEC}

# ---------- 仮説 ----------
Hh = {}
Hh["H70a"] = {"value": kL, "rule": "mean k of literate failures < 0.25", "supported": kL < 0.25}
Hh["H70b"] = {"value": kN, "rule": "mean k of nonliterate isolate losses >= 0.30", "supported": kN >= 0.30}
Hh["H70c"] = {"value": pers, "rule": "share of literate successes carried by a skilled person >= 0.75", "supported": pers >= 0.75}
v = R["thr_Fe_if_knowledge_bound_q"][1]
Hh["H70d"] = {"value": v, "rule": "median thr (N_Fe, knowledge-bound, lam 0.2, s main) < 0.1", "supported": v < 0.1}
v = RES["main/main"]["A_Ord/U3"]["F_total"]
Hh["H70e"] = {"value": v, "rule": "main, p_early main: F (A_Ord/U3) < 1.2", "supported": v < 1.2}
mx = max(r["A_Ord/U3"]["F_total"] for r in RES.values())
Hh["H70f"] = {"value": mx, "rule": "all versions: F (A_Ord/U3) < 5", "supported": mx < 5}
v = RES["psi1/one"]["A_Ord/U3"]["F_heavy"]
Hh["H70g"] = {"value": v, "rule": "psi1, p_early=1: F_heavy (A_Ord/U3) >= 1.5", "supported": v >= 1.5}
gFe = DEC["N_Fe"]["A_Ord/U3"]["F_total"] - 1; gAll = RES["psi1/one"]["A_Ord/U3"]["F_total"] - 1
Hh["H70h"] = {"value": {"gain_Fe_only": gFe, "gain_all": gAll, "ratio": gFe / gAll if gAll > 0 else None},
              "rule": "psi1, p_early=1: gain from N_Fe alone < 0.5 of total gain", "supported": gAll > 0 and gFe / gAll < 0.5}
R["hypotheses"] = Hh
R["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
names = list(RES.keys())
for ax, (a, u) in zip(axs, KEYS):
    key = f"{a}/{u}"
    y = np.arange(len(names))
    ax.barh(y - 0.2, [RES[n][key]["F_total"] for n in names], height=0.4, label="F total", color="C0")
    ax.barh(y + 0.2, [RES[n][key]["F_heavy"] or 0 for n in names], height=0.4, label="F heavy part", color="C3")
    ax.axvline(BE, color="k", lw=1); ax.text(BE, len(names) - 0.5, " break-even 5.0", fontsize=7)
    ax.axvline(1, color="gray", lw=0.6); ax.set_xscale("log"); ax.set_title(key, fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7); ax.set_xlabel("value(T+C with threshold-lowering manual) / value(T+C, 054)")
axs[0].legend(fontsize=7)
fig.suptitle("058: if written knowledge lowers stage population thresholds (Henrich eq.3, psi = knowledge-bound share)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_F_versions.png"), dpi=120); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
grp = [("literate failures", Lf, "k"), ("non-literate isolate losses", Nf, "k")]
ids = [c["id"] for c in Lf] + [c["id"] for c in Nf]; ks = [c["k"] for c in Lf] + [c["k"] for c in Nf]
cols = ["C1"] * len(Lf) + ["C2"] * len(Nf)
axs[0].barh(range(len(ids)), ks, color=cols); axs[0].set_yticks(range(len(ids))); axs[0].set_yticklabels(ids, fontsize=7)
axs[0].set_xlabel("k = loss attributed to lost knowledge (1) / mixed (0.5) / materials, fuel, manpower, environment (0)")
axs[0].set_title(f"mean k: literate {kL:.2f} (orange), non-literate {kN:.2f} (green)", fontsize=9)
Ngrid = np.logspace(2, 7, 200)
for s, ls in ((0.1, ":"), (0.43, "-"), (0.8, "--")):
    axs[1].plot(Ngrid, np.minimum(1, (0.2 * Ngrid * np.exp(EPS)) ** (-s)), ls, label=f"s={s}")
axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_xlabel("stage threshold N (people)"); axs[1].set_ylabel("threshold multiplier thr")
axs[1].set_title("Henrich 2004 eq.(3): thr = (0.2 N e^eps)^(-s) if a manual removes share s of transmission loss", fontsize=8); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_cases_thr.png"), dpi=120); plt.close(fig)

json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: R[k] for k in ("check_057", "check_T2s_m3", "cases", "priors", "thr_Fe_if_knowledge_bound_q", "thr_quantiles", "T2_heavy_medians", "CE_with_F")}, indent=1, ensure_ascii=False, default=float))
for k, v in RES.items():
    print(k, {kk: (round(vv["F_total"], 4), round(vv["F_heavy"], 4) if vv["F_heavy"] else None) for kk, vv in v.items()})
for k, v in DEC.items(): print("only", k, {kk: round(vv["F_total"], 4) for kk, vv in v.items()})
for k, h in Hh.items(): print(k, h["supported"], json.dumps(h["value"], default=float)[:200])
print("runtime", round(R["runtime_s"], 1))
