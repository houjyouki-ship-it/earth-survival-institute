"""研究139：輸入なしで無傷の国が計算機の社会（T3）を何年保つか（open_problems 321 (a)）。
注：L_T3・状態の割合・ΔV は模型の中の量で、生存確率でも復興確率でもない。"""
import json, math, os, re, shutil, subprocess, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = {}
X = json.load(open(f"{HERE}/../136-conquest-pressure/xia_tables.json"))["S4"]["rows"]
POP = {"Argentina": X["Argentina"][0] * 1e6, "Australia": X["Australia"][0] * 1e6}   # Xia 表 S4（2010 年）
# ---- 資料からの部品の寿命（sources/README_sources.txt）
def wb_fit(t1, s1, t2, s2):
    b = (math.log(-math.log(s2)) - math.log(-math.log(s1))) / math.log(t2 / t1)
    eta = t1 / (-math.log(s1)) ** (1 / b); return eta, b
HDD_eta, HDD_b = wb_fit(4, 0.90, 6, 0.65)          # Backblaze 2021：4 年 90%、6 年 65%
ARR_HDD = 0.0301                                     # Schroeder & Gibson 加重平均
SSD_l = 0.0092                                       # Backblaze SSD 生涯 AFR（Q2 2022）
# Schroeder Table 3：IC（CPU・メモリ・基板）と電源の交換の、HDD に対する比
T3tab = {"HPC1": dict(hdd=30.6, ic=28.5 + 12.4 + 4.9 + 1.0, psu=1.6),
         "COM1": dict(hdd=18.1, ic=20.1 + 2.0, psu=34.8),
         "COM2": dict(hdd=49.1, ic=23.4 + 3.4 + 2.2, psu=10.1)}
r_ic = {k: v["ic"] / v["hdd"] for k, v in T3tab.items()}; r_psu = {k: v["psu"] / v["hdd"] for k, v in T3tab.items()}
CAP_eta = 15.0                                       # 2000h/105°C → 45°C で約 15 年（Wikipedia の例）
R["inputs"] = dict(POP=POP, HDD_weibull=dict(eta=HDD_eta, beta=HDD_b), ARR_HDD=ARR_HDD, SSD_afr=SSD_l,
                   r_ic=r_ic, r_psu=r_psu, CAP_eta=CAP_eta)
print("HDD Weibull eta %.2f beta %.2f" % (HDD_eta, HDD_b), "r_ic", {k: round(v, 2) for k, v in r_ic.items()}, "r_psu", {k: round(v, 3) for k, v in r_psu.items()})

class Comp:
    def __init__(s, kind, eta, beta=1.0): s.kind, s.eta, s.beta = kind, eta, beta
    def mu(s): return s.eta * math.gamma(1 + 1 / s.beta)
    def H(s, t): return (np.maximum(t, 0) / s.eta) ** s.beta    # 累積ハザード
def exp_c(l): return Comp("exp", 1 / l, 1.0)
def wb_match5(l, beta=3.0):   # 5 年の累積の故障を指数と合わせた摩耗型
    return Comp("wb", 5 / (l * 5) ** (1 / beta), beta)

HMAX, DT = 1000.0, 0.25
def pool_life(c, N0, Nthr, a):
    """予備の山 P：保管で a 倍の速さで年を取り、使用中の N_thr 個の故障（更新の率 N_thr/μ）を埋める。P<0 で尽きる。"""
    P, t = N0 - Nthr, 0.0; rate = Nthr / c.mu()
    while t < HMAX:
        dec = 1 - math.exp(-(c.H(a * (t + DT)) - c.H(a * t))) if a > 0 else 0.0
        P = P * (1 - dec) - rate * DT; t += DT
        if P < 0: return t
    return HMAX
def storage_life(cs, ch, N0, Nthr, a, s_ssd):
    """SSD を先に使い、尽きたら HDD（HDD の山も同じ時間だけ保管で年を取る）。"""
    Ps, Ph, t = s_ssd * N0, (1 - s_ssd) * N0, 0.0
    while t < HMAX:
        for c, nm in ((cs, "s"), (ch, "h")):
            dec = 1 - math.exp(-(c.H(a * (t + DT)) - c.H(a * t))) if a > 0 else 0.0
            if nm == "s": Ps *= (1 - dec)
            else: Ph *= (1 - dec)
        need = Nthr / (cs.mu() if Ps > Nthr else ch.mu()) * DT
        if Ps > Nthr: Ps -= need
        else: Ph -= need
        t += DT
        if Ps <= Nthr and Ph <= Nthr: return t
    return HMAX
def L_T3(p):
    N0 = p["pc"] * p["pop"]
    ic = exp_c(p["r_ic"] * ARR_HDD) if p["ic_beta"] == 1 else wb_match5(p["r_ic"] * ARR_HDD, p["ic_beta"])
    ssd = exp_c(SSD_l) if p["ssd_beta"] == 1 else wb_match5(SSD_l, p["ssd_beta"])
    hdd = Comp("wb", HDD_eta, HDD_b) if p["hdd"] == "bb" else exp_c(ARR_HDD)
    cap = Comp("wb", CAP_eta, 3.0)
    Ls = {"IC": pool_life(ic, N0, p["Nthr"], p["a"]),
          "storage": storage_life(ssd, hdd, N0, p["Nthr"], p["a"], p["s_ssd"]),
          "PSU_cap": pool_life(cap, N0, p["Nthr"], p["a_cap"]) if not p["cap_repair"] else HMAX}
    k = min(Ls, key=Ls.get)
    return Ls[k], (k if Ls[k] < HMAX else "none(>=1000)"), Ls

MAIN = dict(pop=POP["Argentina"], pc=0.5, Nthr=1e4, a=0.0, a_cap=0.0, r_ic=r_ic["COM1"], ic_beta=1, ssd_beta=1,
            hdd="bb", s_ssd=0.5, cap_repair=False)
L, k, Ls = L_T3(MAIN); R["main"] = dict(L=L, binding=k, by_comp=Ls); print("main", L, k, Ls)
# ---- 1 つずつ動かす表
grid = []
for nm, vals in [("a", [0, 0.01, 0.1, 0.3, 1.0]), ("a_cap", [0, 0.3, 1.0, 1.5]), ("Nthr", [1e3, 1e4, 1e5]), ("pc", [0.3, 1.0]),
                 ("ic_beta", [1, 2, 3]), ("ssd_beta", [1, 3]), ("hdd", ["bb", "exp"]), ("s_ssd", [0.0, 0.5]), ("pop", [POP["Australia"]])]:
    for v in vals:
        p = dict(MAIN); p[nm] = v
        if nm == "a": p["a_cap"] = v
        L, k, Ls = L_T3(p); grid.append(dict(var=nm, val=v, L=L, binding=k, by=Ls))
        print(f"{nm}={v}: L={L:.1f} ({k})", {q: round(w, 1) for q, w in Ls.items()})
# 悲観の角：全部が保管でも同じ速さで年を取る・摩耗型
for nm, p in [("pess_all_age", dict(MAIN, a=1.0, a_cap=1.0, ic_beta=3, ssd_beta=3, s_ssd=0.0)),
              ("pess_all_age_Nthr1e5", dict(MAIN, a=1.0, a_cap=1.0, ic_beta=3, ssd_beta=3, s_ssd=0.0, Nthr=1e5, pc=0.3)),
              ("a1_cap_repair", dict(MAIN, a=1.0, a_cap=1.0, cap_repair=True))]:
    L, k, Ls = L_T3(p); grid.append(dict(var=nm, val=None, L=L, binding=k, by=Ls)); print(nm, round(L, 1), k, {q: round(w, 1) for q, w in Ls.items()})
R["grid"] = grid
# ---- 不確かさの MC（模型の版の幅、確率の主張ではない）
rng = np.random.default_rng(20261139); NMC = 3000; out = []
for i in range(NMC):
    p = dict(pop=POP["Argentina"] if rng.random() < 0.5 else POP["Australia"], pc=math.exp(rng.uniform(math.log(0.3), 0)),
             Nthr=10 ** rng.uniform(3, 5), a=rng.uniform(0, 1), a_cap=rng.uniform(0, 1.5),
             r_ic=list(r_ic.values())[rng.integers(3)], ic_beta=[1, 2, 3][rng.integers(3)], ssd_beta=[1, 3][rng.integers(2)],
             hdd=["bb", "exp"][rng.integers(2)], s_ssd=rng.uniform(0, 0.7), cap_repair=bool(rng.random() < 0.5))
    L, k, _ = L_T3(p); out.append((L, k, p["a"], p["a_cap"], p["cap_repair"]))
Larr = np.array([o[0] for o in out]); ks = [o[1] for o in out]
frac = dict(heavy=float(np.mean(Larr < 20)), fragile=float(np.mean((Larr >= 20) & (Larr < 80))), T3=float(np.mean(Larr >= 80)))
R["mc"] = dict(n=NMC, median=float(np.median(Larr)), q10=float(np.quantile(Larr, 0.1)), frac=frac,
               binding={k: ks.count(k) / NMC for k in set(ks)},
               binding_when_lt80={k: sum(1 for o in out if o[0] < 80 and o[1] == k) / max(1, int(np.sum(Larr < 80))) for k in set(ks)})
a_lt80 = [o[2] for o in out if o[0] < 80]; R["mc"]["a_median_when_lt80"] = float(np.median(a_lt80)) if a_lt80 else None
print("MC", R["mc"])
# ---- 032 の NUC_G へ写像（138 の Q：主 0.9998 独立、0.984 完全相関）
def nuc_mean(Q, fr):
    alive = 0.99; T3 = alive * Q * fr["T3"]; fg = alive * Q * fr["fragile"]; hv = alive * (1 - Q + Q * fr["heavy"])
    return [0.01, T3, fg, hv * 14 / 19, hv * 4 / 19, hv * 1 / 19]
fr_main = dict(heavy=float(R["main"]["L"] < 20), fragile=float(20 <= R["main"]["L"] < 80), T3=float(R["main"]["L"] >= 80))
VERS = {"orig032": [0.01, 0.65, 0.15, 0.14, 0.04, 0.01],
        "main_Qind": nuc_mean(0.9998, fr_main), "main_Qcom": nuc_mean(0.984, fr_main),
        "mc_Qcom": nuc_mean(0.984, frac)}
R["nuc_means"] = VERS; print({k: [round(x, 4) for x in v] for k, v in VERS.items()})
# ---- 032 を一時の写しで再実行（NUC_G の平均だけ置き換え）
src = f"{HERE}/../032-collapse-scenarios"; W = {}
for nm, mean in VERS.items():
    tmp = f"/tmp/r139_032_{nm}"; shutil.rmtree(tmp, ignore_errors=True); shutil.copytree(src, tmp)
    os.makedirs("/tmp/024-content-value-reader-guide", exist_ok=True)
    shutil.copytree(f"{HERE}/../024-content-value-reader-guide", "/tmp/024-content-value-reader-guide", dirs_exist_ok=True)
    s = open(f"{tmp}/sim.py").read()
    s = s.replace('"NUC_G": {"mean": [0.01, 0.65, 0.15, 0.14, 0.04, 0.01]', '"NUC_G": {"mean": ' + json.dumps(mean))
    open(f"{tmp}/sim.py", "w").write(s)
    subprocess.run(["python3", f"{tmp}/sim.py"], cwd=tmp, capture_output=True, check=True)
    r = json.load(open(f"{tmp}/results.json"))["weighted"]
    W[nm] = {a: dict(R_tab=r[a]["R_tab"], ratio_GEN0=r[a]["R_tab_ratio_vs_GEN0"], NUC_G_share=r[a]["R_tab_share_by_scen"]["NUC_G"]) for a in r}
orig = json.load(open(f"{src}/results.json"))["weighted"]
assert abs(W["orig032"]["A_Ord"]["R_tab"] - orig["A_Ord"]["R_tab"]) < 1e-12, "032 の再現に失敗"
assert abs(W["orig032"]["A_XPT_S"]["R_tab"] - orig["A_XPT_S"]["R_tab"]) < 1e-12
R["reproduce_032"] = True; R["weighted"] = W
for nm in W: print(nm, {a: (round(v["R_tab"], 5), round(v["R_tab"] / W["orig032"][a]["R_tab"] - 1, 4)) for a, v in W[nm].items()})
heavy_new = {nm: sum(v[3:]) for nm, v in VERS.items()}; R["heavy_share"] = heavy_new
R["hyp"] = {"H151a": dict(val=HDD_b, ok=HDD_b > 1), "H151b": dict(val=R["main"]["L"], ok=R["main"]["L"] >= 20),
            "H151c": dict(val=R["mc"]["binding_when_lt80"], ok=None),
            "H151d": dict(val=heavy_new["main_Qcom"], ok=heavy_new["main_Qcom"] < 0.19)}
json.dump(R, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
# ---- 図
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
av = [0, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0]
for nt in (1e3, 1e4, 1e5):
    ax[0].plot(av, [L_T3(dict(MAIN, a=x, a_cap=x, Nthr=nt))[0] for x in av], "o-", label=f"N_thr={nt:.0e}")
ax[0].set_xscale("symlog", linthresh=0.01); ax[0].set_yscale("log"); ax[0].axhline(80, ls=":", c="k"); ax[0].axhline(20, ls=":", c="r")
ax[0].set_xlabel("shelf ageing rate a (stored spares vs in use)"); ax[0].set_ylabel("L_T3 (years, capped 1000)"); ax[0].legend(fontsize=7)
ax[0].set_title("import-free computer stock (model)")
ax[1].hist(np.minimum(Larr, 1000), bins=40); ax[1].axvline(80, ls=":", c="k"); ax[1].axvline(20, ls=":", c="r")
ax[1].set_xlabel("L_T3 (years), MC over model versions"); ax[1].set_ylabel("count")
plt.tight_layout(); plt.savefig(f"{HERE}/fig_import_free_T3.png", dpi=120)
