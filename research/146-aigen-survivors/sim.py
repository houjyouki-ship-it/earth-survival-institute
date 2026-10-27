"""研究146：AI・GEN の重い標本に生き残りの割合 s と群の数を与える（open_problems 335）。
計算のみ・ネット接続なし。入力は 038 の sim.py（写しで実行）、070・145 の results.json。
事前登録：PREREGISTRATION.md（2026-10-27 09:41 JST、計算の前）。seed 20261146。
注：F_heavy・TC は生存確率や復興確率ではない（模型の中の量）。"""
import json, os, runpy, shutil, time, contextlib, io
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
R145 = json.load(open(f"{RES}/145-resettlement-rule/results.json", encoding="utf8"))
R070 = json.load(open(f"{RES}/070-group-deposit-overlap/results.json"))
R038 = json.load(open(f"{RES}/038-restart-state-evidence/results.json"))
# ---- 038 を写しで実行 ----
W = "/tmp/rep146/research"; os.makedirs(f"{W}/038-restart-state-evidence", exist_ok=True)
for d in os.listdir(RES):
    if d != "038-restart-state-evidence" and not os.path.exists(f"{W}/{d}"): os.symlink(f"{RES}/{d}", f"{W}/{d}")
shutil.copy(f"{RES}/038-restart-state-evidence/sim.py", f"{W}/038-restart-state-evidence/sim.py")
_cwd = os.getcwd(); os.chdir(f"{W}/038-restart-state-evidence")
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(f"{W}/038-restart-state-evidence/sim.py", run_name="m146")
os.chdir(_cwd)
N = G["N"]; EVID = G["EVID"]; HEAVY = G["HEAVY"]; SCEN = G["SCEN"]; DRAWS = G["DRAWS"]; POP_W = G["POP_W"]
survivors, assign, times, recs_gen, T2s_032, weights = G["survivors"], G["assign"], G["times"], G["recs_gen"], G["T2s_032"], G["weights"]
ANCH = ("A_Ord", "A_XPT_S", "A_XPT_E"); THS = (0.01, 0.05, 0.10)
CHECK = {}
# ---- EVID の側（038 のまま） ----
SV = {}; ST = {}; PER = {}
for s in EVID:
    sv = survivors(s); st = assign(sv); ST[s] = st; SV[s] = sv
    T2s = times(st, sv, SCEN[s]["lit"])[0]; PER[s] = recs_gen(st, T2s, DRAWS[s])["R_tab"]
for s in ("AI", "GEN"):
    st = DRAWS[s]["st"]; ST[s] = st; PER[s] = recs_gen(st, T2s_032(st, SCEN[s]["lit"]), DRAWS[s])["R_tab"]
def rtab_heavy(per, a):
    w = weights(a)[0]; return float(np.mean(sum(w[s] * per[s] * np.isin(ST[s], HEAVY) for s in w)))
for a in ANCH:
    d = abs(rtab_heavy(PER, a) - R038["weighted_038"][a]["R_tab_from_heavy"]); CHECK[f"038_Rtab_heavy_{a}"] = d; assert d < 1e-12, (a, d)
def evid_parts(a, th):
    w = weights(a)[0]; num = den = 0.0
    for s in EVID:
        hv = np.isin(ST[s], HEAVY); den += float(np.mean(w[s] * hv)); num += float(np.mean(w[s] * hv * (SV[s]["s"] < th)))
    return num, den
for a in ANCH:
    n_, d_ = evid_parts(a, 0.05); ref = R145["model_survival_fraction"]["weighted"][a]["f_deep_EVID"]["0.05"]
    CHECK[f"145_fdeep_{a}"] = abs(n_ / d_ - ref); assert CHECK[f"145_fdeep_{a}"] < 1e-12
# ---- 引き直しの元（状態ごと） ----
POOL = {}
for s in EVID:
    for tgt, ok in ((4, (4,)), (5, (5, 6))):
        idx = np.where(np.isin(ST[s], ok))[0]
        if len(idx) == 0: idx = np.where(np.isin(ST[s], HEAVY))[0]
        POOL[(s, tgt)] = idx
POOLN = {f"{s}|{t}": int(len(v)) for (s, t), v in POOL.items()}
def resample(target, probs, rng):
    """target の重い標本（4・5）に、probs（元のシナリオ→確率）で (s, ntot, nmax, mcap) を組で引き直す"""
    out = {k: np.ones(N) for k in ("s", "mcap")}; out["ntot"] = np.full(N, POP_W); out["nmax"] = np.full(N, POP_W)
    st = ST[target]; srcs = [k for k, v in probs.items() if v > 0 and len(POOL[(k, 4)]) + len(POOL[(k, 5)]) > 0]
    pp = np.array([probs[k] for k in srcs]); pp = pp / pp.sum()
    for tgt in (4, 5):
        ii = np.where(st == tgt)[0]
        if len(ii) == 0: continue
        ch = rng.choice(len(srcs), size=len(ii), p=pp)
        for j, src in enumerate(srcs):
            sel = ii[ch == j]; pool = POOL[(src, tgt)]
            if len(sel) == 0 or len(pool) == 0: continue
            pick = pool[rng.integers(0, len(pool), len(sel))]
            for k in ("s", "ntot", "nmax", "mcap"): out[k][sel] = SV[src][k][pick]
    return out
def geff(sv): return np.clip(np.round(sv["ntot"] / np.maximum(sv["nmax"], 1.0)), 1, 10)
SG = R070["scan_G"]; GG = np.log([r["G"] for r in SG]); TCG = np.array([r["nospread_mix"] for r in SG])
def tc_of(gv): return np.interp(np.log(gv), GG, TCG)
F_DEEP = R145["deep_set_7of7"]["F"]
F_IN = {th: R145["rule_R333"][f"{th:.2f}"]["insitu_F"] for th in THS}
F_A145 = {a: R145["combined"][f"{a}|th=0.05"]["F_readA"] for a in ANCH}
# ---- 版 ----
OUT = {}; FIGD = {}
for a in ANCH:
    w = weights(a)[0]
    evw = {s: float(np.mean(w[s] * np.isin(ST[s], HEAVY))) for s in EVID}
    VAR = {"P_BIO": {"BIO": 1.0}, "P_CASC": {"CASC": 1.0}, "P_MIX": {"BIO": 0.5, "CASC": 0.5}, "P_EVIDW": evw}
    for vn, probs in VAR.items():
        rng = np.random.default_rng(20261146)
        per = dict(PER); svx = {}
        for s in ("AI", "GEN"):
            sv = resample(s, probs, rng); svx[s] = sv; st = ST[s]; hv = np.isin(st, HEAVY)
            T2h = times(st, sv, SCEN[s]["lit"])[0]
            T2s = np.where(hv, T2h, T2s_032(st, SCEN[s]["lit"]))
            per[s] = recs_gen(st, T2s, DRAWS[s])["R_tab"]
        row = {"pool_probs": probs}
        for th in THS:
            n_, d_ = evid_parts(a, th); na = da = 0.0
            for s in ("AI", "GEN"):
                hv = np.isin(ST[s], HEAVY); da += float(np.mean(w[s] * hv)); na += float(np.mean(w[s] * hv * (svx[s]["s"] < th)))
            f = (n_ + na) / (d_ + da)
            row[f"th={th:.2f}"] = {"f_deep_all": f, "f_deep_AIGEN": na / da, "f_deep_EVID": n_ / d_, "share_AIGEN": da / (d_ + da),
                                   "F_readA": f * F_DEEP + (1 - f) * F_IN[th], "F_readB": F_IN[th]}
        row["dF_readA_vs_145(th=0.05)"] = row["th=0.05"]["F_readA"] - F_A145[a]
        rh = rtab_heavy(per, a); row["R_tab_from_heavy"] = rh; row["ratio_R_tab_heavy_vs_038"] = rh / R038["weighted_038"][a]["R_tab_from_heavy"]
        # 群の数（推測）：重い標本全体（EVID＋AI・GEN）の重み付き
        num = den = 0.0; gs = []; gw = []
        for s in list(EVID) + ["AI", "GEN"]:
            sv = SV[s] if s in EVID else svx[s]; hv = np.isin(ST[s], HEAVY); ge = geff(sv)
            ws = np.broadcast_to(np.asarray(w[s], float), (N,)) * hv
            num += float(np.sum(ws * tc_of(ge))); den += float(np.sum(ws)); gs.append(ge[hv]); gw.append(ws[hv])
        gs = np.concatenate(gs); gw = np.concatenate(gw); o = np.argsort(gs); cw = np.cumsum(gw[o]) / gw.sum()
        row["G_eff_weighted_median"] = float(gs[o][np.searchsorted(cw, 0.5)]); row["G_eff_share_G1"] = float(gw[gs == 1].sum() / gw.sum())
        row["TC_070_interp_heavy"] = num / den
        row["G_eff_AIGEN_median"] = float(np.median(np.concatenate([geff(svx[s])[np.isin(ST[s], HEAVY)] for s in ("AI", "GEN")])))
        OUT[f"{a}|{vn}"] = row
        if a == "A_XPT_S": FIGD[vn] = np.concatenate([svx[s]["s"][np.isin(ST[s], HEAVY)] for s in ("AI", "GEN")])
# 038 の G の主（070 の平均 3.9 の主抽出）の TC 0.819 との比較用
TC070_main_draw_mean = float(np.mean([r["nospread_mix"] for r in SG]))
# ---- 仮説 ----
dA = {a: OUT[f"{a}|P_MIX"]["dF_readA_vs_145(th=0.05)"] for a in ANCH}
rngF = {a: max(OUT[f"{a}|{v}"]["th=0.05"]["F_readA"] for v in ("P_BIO", "P_CASC", "P_MIX", "P_EVIDW")) -
           min(OUT[f"{a}|{v}"]["th=0.05"]["F_readA"] for v in ("P_BIO", "P_CASC", "P_MIX", "P_EVIDW")) for a in ANCH}
rc = OUT["A_Ord|P_BIO"]["ratio_R_tab_heavy_vs_038"]
H = {"H158a": {"class": "confirmatory", "value": dA, "supported": all(abs(v) < 0.05 for v in dA.values())},
     "H158b": {"class": "confirmatory", "value": rngF, "supported": rngF["A_Ord"] >= 0.05 and rngF["A_XPT_S"] >= 0.05},
     "H158c": {"class": "prediction", "value": rc, "supported": rc <= 0.8}}
RESULT = {"study": "146-aigen-survivors", "note": "F_heavy・TC は生存確率や復興確率ではない", "check": CHECK, "pool_sizes": POOLN,
          "F_deep_7of7": F_DEEP, "F_insitu": {f"{k:.2f}": v for k, v in F_IN.items()}, "F_readA_145": F_A145,
          "variants": OUT, "hypotheses": H, "runtime_s": time.time() - T0}
json.dump(RESULT, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
# ---- 図 ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for vn, x in FIGD.items():
    x = np.sort(x); ax[0].plot(x, np.arange(1, len(x) + 1) / len(x), label=vn)
ax[0].axvline(0.05, c="r", ls="--", lw=.8); ax[0].set_xscale("log"); ax[0].set_xlabel("assigned survival fraction s (AI+GEN heavy, A_XPT_S)"); ax[0].set_ylabel("CDF"); ax[0].legend(fontsize=8)
vs = ("P_BIO", "P_CASC", "P_MIX", "P_EVIDW"); xx = np.arange(len(vs))
for k, a in enumerate(ANCH):
    ax[1].plot(xx, [OUT[f"{a}|{v}"]["th=0.05"]["F_readA"] for v in vs], "o-", label=f"{a} read A"); ax[1].axhline(F_A145[a], ls=":", c=f"C{k}", lw=.8)
ax[1].axhline(F_IN[0.05], c="k", lw=.8, label="read B (in-situ)"); ax[1].set_xticks(xx); ax[1].set_xticklabels(vs); ax[1].set_ylabel("F_heavy (model quantity)"); ax[1].legend(fontsize=7)
ax[1].set_title("dotted: 145 EVID-only", fontsize=9)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_aigen_s.png", dpi=120)
print("check", CHECK); print("pools", POOLN)
for k, v in OUT.items():
    print(k, {"fA5": round(v["th=0.05"]["F_readA"], 4), "f_all": round(v["th=0.05"]["f_deep_all"], 3), "f_AIGEN": round(v["th=0.05"]["f_deep_AIGEN"], 3),
              "shareAIGEN": round(v["th=0.05"]["share_AIGEN"], 3), "dF": round(v["dF_readA_vs_145(th=0.05)"], 4), "Rh_ratio": round(v["ratio_R_tab_heavy_vs_038"], 3),
              "Gmed": v["G_eff_weighted_median"], "G1": round(v["G_eff_share_G1"], 3), "TC": round(v["TC_070_interp_heavy"], 4), "GmedAIGEN": v["G_eff_AIGEN_median"]})
print({k: (v["supported"], v["value"]) for k, v in H.items()}); print("runtime", round(time.time() - T0, 1))
