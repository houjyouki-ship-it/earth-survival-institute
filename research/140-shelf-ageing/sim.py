"""研究140：計算機の部品の保管の老化 a を型ごとに資料の機構から置き、139 の L_T3 と 032 の NUC_G を置き直す（open_problems 322）。
注：a・L_T3・状態の割合・ΔV は模型の中の量で、生存確率でも復興確率でもない。"""
import json, math, os, shutil, subprocess, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = {}
P139 = f"{HERE}/../139-import-free-T3"
src139 = open(f"{P139}/sim.py").read()
ns = {"__file__": f"{P139}/sim.py"}; exec(src139.split("MAIN = dict(")[0], ns)   # 139 の資料の値と部品の関数だけ取り込む
Comp, exp_c, wb_match5, pool_life, storage_life, L_T3_139 = (ns[k] for k in ("Comp", "exp_c", "wb_match5", "pool_life", "storage_life", "L_T3"))
POP, ARR_HDD, SSD_l, HDD_eta, HDD_b, CAP_eta, r_ic, HMAX, DT = (ns[k] for k in ("POP", "ARR_HDD", "SSD_l", "HDD_eta", "HDD_b", "CAP_eta", "r_ic", "HMAX", "DT"))
MAIN139 = dict(pop=POP["Argentina"], pc=0.5, Nthr=1e4, a=0.0, a_cap=0.0, r_ic=r_ic["COM1"], ic_beta=1, ssd_beta=1, hdd="bb", s_ssd=0.5, cap_repair=False)
# ---- 再現：139 の主（1000）と a=1 の行（32.5）
r139 = json.load(open(f"{P139}/results.json"))
assert L_T3_139(MAIN139)[0] == r139["main"]["L"]
g1 = [g for g in r139["grid"] if g["var"] == "a" and g["val"] == 1.0][0]
assert abs(L_T3_139(dict(MAIN139, a=1.0, a_cap=1.0))[0] - g1["L"]) < 1e-9; R["reproduce_139"] = True
# ---- 機構からの a（sources/README_sources.txt）
def a_arrhenius(Ts, Tu): return 2 ** ((Ts - Tu) / 10)       # 10 度で寿命 2 倍（Wikipedia アルミ電解コンデンサ）
A = dict(cap=a_arrhenius(25, 45), ic=0.0, ssd=0.0, hdd=0.0)    # 主：IC・SSD の摩耗は電界・電流が要る、HDD はランプ式で停止中の摩耗なし
R["a_main"] = A; print("a main", A)
def storage_life2(cs, ch, N0, Nthr, a_s, a_h, s_ssd):
    Ps, Ph, t = s_ssd * N0, (1 - s_ssd) * N0, 0.0
    while t < HMAX:
        for c, a, nm in ((cs, a_s, "s"), (ch, a_h, "h")):
            dec = 1 - math.exp(-(c.H(a * (t + DT)) - c.H(a * t))) if a > 0 else 0.0
            if nm == "s": Ps *= (1 - dec)
            else: Ph *= (1 - dec)
        need = Nthr / (cs.mu() if Ps > Nthr else ch.mu()) * DT
        if Ps > Nthr: Ps -= need
        else: Ph -= need
        t += DT
        if Ps <= Nthr and Ph <= Nthr: return t
    return HMAX
def cap_life_noreform(c, N0, Nthr, t_shelf):
    """悲観の版：再化成をしない社会では保管 t_shelf 年を過ぎた予備は使えない（段の関数）。使用中の N_thr 個は η の寿命で入れ替わる。"""
    P, t = N0 - Nthr, 0.0; rate = Nthr / c.mu()
    while t < HMAX:
        P -= rate * DT; t += DT
        if t >= t_shelf: return t      # 予備が使えなくなった時点で使用中の N_thr を割り始める（悲観の段）
        if P < 0: return t
    return HMAX
def L_T3(p):
    N0 = p["pc"] * p["pop"]
    ic = exp_c(p["r_ic"] * ARR_HDD) if p["ic_beta"] == 1 else wb_match5(p["r_ic"] * ARR_HDD, p["ic_beta"])
    ssd = exp_c(SSD_l) if p["ssd_beta"] == 1 else wb_match5(SSD_l, p["ssd_beta"])
    hdd = Comp("wb", HDD_eta, HDD_b) if p["hdd"] == "bb" else exp_c(ARR_HDD)
    cap = Comp("wb", CAP_eta, 3.0)
    if p["cap_repair"]: Lc = HMAX
    elif p.get("t_shelf"): Lc = cap_life_noreform(cap, N0, p["Nthr"], p["t_shelf"])
    else: Lc = pool_life(cap, N0, p["Nthr"], p["a_cap"])
    Ls = {"IC": pool_life(ic, N0, p["Nthr"], p["a_ic"]),
          "storage": storage_life2(ssd, hdd, N0, p["Nthr"], p["a_ssd"], p["a_hdd"], p["s_ssd"]), "PSU_cap": Lc}
    k = min(Ls, key=Ls.get); return Ls[k], (k if Ls[k] < HMAX else "none(>=1000)"), Ls
BASE = dict(MAIN139, a_cap=A["cap"], a_ic=A["ic"], a_ssd=A["ssd"], a_hdd=A["hdd"]); BASE.pop("a")
assert abs(L_T3(dict(BASE, a_cap=1, a_ic=1, a_ssd=1, a_hdd=1))[0] - g1["L"]) < 1e-9     # 型ごとの版が 139 と一致
L, k, Ls = L_T3(BASE); R["main"] = dict(L=L, binding=k, by=Ls); print("main", round(L, 1), k, {q: round(w, 1) for q, w in Ls.items()})
rows = []
def row(nm, **kw):
    L, k, Ls = L_T3(dict(BASE, **kw)); rows.append(dict(name=nm, L=L, binding=k, by=Ls, kw={a: (b if not isinstance(b, float) else round(b, 4)) for a, b in kw.items()}))
    print(f"{nm}: L={L:.1f} ({k})", {q: round(w, 1) for q, w in Ls.items()})
for Ts in (10, 15, 35): row(f"cap_Ts{Ts}", a_cap=a_arrhenius(Ts, 45))
for Tu in (55,): row(f"cap_Tu{Tu}", a_cap=a_arrhenius(25, Tu))
for f in (0.1, 0.3): row(f"ic_whisker{f}", a_ic=f)
row("hdd_0.2", a_hdd=0.2); row("ic_beta3_whisk0.3", a_ic=0.3, ic_beta=3)
for ts in (5, 10): row(f"noreform_tshelf{ts}", t_shelf=ts)
row("cap_repair", cap_repair=True)
for nt in (1e3, 1e5): row(f"Nthr{nt:.0e}", Nthr=nt)
row("pc0.3_Nthr1e5", pc=0.3, Nthr=1e5)
R["rows"] = rows
# ---- 版の MC（139 と同じ版の幅、a だけ機構から引く。確率の主張ではない）
rng = np.random.default_rng(20261140); NMC = 3000; out = []
for i in range(NMC):
    Ts, Tu = rng.uniform(10, 35), rng.uniform(40, 55)
    p = dict(pop=POP["Argentina"] if rng.random() < 0.5 else POP["Australia"], pc=math.exp(rng.uniform(math.log(0.3), 0)),
             Nthr=10 ** rng.uniform(3, 5), a_cap=a_arrhenius(Ts, Tu), a_ic=rng.uniform(0, 0.3), a_ssd=rng.uniform(0, 0.05), a_hdd=rng.uniform(0, 0.2),
             r_ic=list(r_ic.values())[rng.integers(3)], ic_beta=[1, 2, 3][rng.integers(3)], ssd_beta=[1, 3][rng.integers(2)],
             hdd=["bb", "exp"][rng.integers(2)], s_ssd=rng.uniform(0, 0.7), cap_repair=bool(rng.random() < 0.5),
             t_shelf=None)   # 再化成なしの段の版は MC に入れず表の角だけ（資料は現代品に前処理不要とする）
    L, k, _ = L_T3(p); out.append((L, k, p["a_cap"]))
Larr = np.array([o[0] for o in out]); ks = [o[1] for o in out]
frac = dict(heavy=float(np.mean(Larr < 20)), fragile=float(np.mean((Larr >= 20) & (Larr < 80))), T3=float(np.mean(Larr >= 80)))
R["mc"] = dict(n=NMC, median=float(np.median(Larr)), q10=float(np.quantile(Larr, 0.1)), frac=frac,
               binding_when_lt80={kk: sum(1 for o in out if o[0] < 80 and o[1] == kk) / max(1, int(np.sum(Larr < 80))) for kk in set(ks)},
               a_cap_q=[float(np.quantile([o[2] for o in out], q)) for q in (0.1, 0.5, 0.9)])
print("MC", R["mc"])
# ---- 032 の NUC_G へ（139 と同じ写像・同じ Q）
def nuc_mean(Q, fr):
    alive = 0.99; T3 = alive * Q * fr["T3"]; fg = alive * Q * fr["fragile"]; hv = alive * (1 - Q + Q * fr["heavy"])
    return [0.01, T3, fg, hv * 14 / 19, hv * 4 / 19, hv * 1 / 19]
fr_main = dict(heavy=float(L < 20), fragile=float(20 <= R["main"]["L"] < 80), T3=float(R["main"]["L"] >= 80))
VERS = {"orig032": [0.01, 0.65, 0.15, 0.14, 0.04, 0.01], "main_Qcom": nuc_mean(0.984, fr_main), "mc_Qcom": nuc_mean(0.984, frac)}
R["nuc_means"] = VERS; R["heavy_share"] = {k: sum(v[3:]) for k, v in VERS.items()}
src = f"{HERE}/../032-collapse-scenarios"; W = {}
for nm, mean in VERS.items():
    tmp = f"/tmp/r140_032_{nm}"; shutil.rmtree(tmp, ignore_errors=True); shutil.copytree(src, tmp)
    shutil.copytree(f"{HERE}/../024-content-value-reader-guide", "/tmp/024-content-value-reader-guide", dirs_exist_ok=True)
    s = open(f"{tmp}/sim.py").read(); key = '"NUC_G": {"mean": [0.01, 0.65, 0.15, 0.14, 0.04, 0.01]'; assert key in s
    open(f"{tmp}/sim.py", "w").write(s.replace(key, '"NUC_G": {"mean": ' + json.dumps(mean)))
    subprocess.run(["python3", f"{tmp}/sim.py"], cwd=tmp, capture_output=True, check=True)
    r = json.load(open(f"{tmp}/results.json"))["weighted"]; W[nm] = {a: r[a]["R_tab"] for a in r}
orig = json.load(open(f"{src}/results.json"))["weighted"]
assert abs(W["orig032"]["A_Ord"] - orig["A_Ord"]["R_tab"]) < 1e-12; R["reproduce_032"] = True
R["weighted"] = W; R["dV_rel"] = {nm: {a: W[nm][a] / W["orig032"][a] - 1 for a in W[nm]} for nm in W}
print("heavy", R["heavy_share"]); print({nm: {a: round(v, 4) for a, v in d.items()} for nm, d in R["dV_rel"].items()})
w139 = r139["weighted"]; R["vs139_mc"] = {a: w139["mc_Qcom"][a]["R_tab"] / w139["orig032"][a]["R_tab"] - 1 for a in w139["mc_Qcom"]}
R["hyp"] = {"H152a": dict(val=A["cap"], ok=A["cap"] < 0.3), "H152b": dict(val=A["ic"], ok=A["ic"] < 0.1),
            "H152c": dict(val=[R["main"]["L"], R["heavy_share"]["main_Qcom"]], ok=R["main"]["L"] >= 80 and R["heavy_share"]["main_Qcom"] < 0.05),
            "H152d": dict(val=R["main"]["binding"], ok=R["main"]["binding"] == "PSU_cap")}
print("hyp", R["hyp"])
json.dump(R, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
Tss = np.arange(0, 46, 5)
for nt in (1e3, 1e4, 1e5): ax[0].plot(Tss, [L_T3(dict(BASE, a_cap=a_arrhenius(t, 45), Nthr=nt))[0] for t in Tss], "o-", label=f"N_thr={nt:.0e}")
ax[0].set_yscale("log"); ax[0].axhline(80, ls=":", c="k"); ax[0].axhline(20, ls=":", c="r")
ax[0].set_xlabel("storage temperature of spare PSUs (C), in-use case 45C"); ax[0].set_ylabel("L_T3 (years, capped 1000)"); ax[0].legend(fontsize=7)
ax[0].set_title("capacitor shelf ageing by 10-degree rule (model)")
ax[1].hist(np.minimum(Larr, 1000), bins=40); ax[1].axvline(80, ls=":", c="k"); ax[1].axvline(20, ls=":", c="r")
ax[1].set_xlabel("L_T3 (years), MC over model versions, mechanism-based a"); ax[1].set_ylabel("count")
plt.tight_layout(); plt.savefig(f"{HERE}/fig_shelf_ageing.png", dpi=120)
