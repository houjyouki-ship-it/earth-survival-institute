"""研究145：停滞の規則に再入植の型を入れる（open_problems 333）＋停滞の判定の r 依存（334）。
計算のみ・ネット接続なし（新しい資料なし。入力は 038・128・143・144 の results.json と 038 の sim）。
事前登録：PREREGISTRATION.md（2026-10-27 07:42 JST、計算の前）。
注：F_heavy・p|lost は生存確率や復興確率ではない（模型の中の量）。"""
import json, os, math, runpy, shutil, time, contextlib, io
import numpy as np
from scipy import stats, integrate
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
R143 = json.load(open(f"{RES}/143-plost-new-cases/results.json", encoding="utf8"))
R144 = json.load(open(f"{RES}/144-copan-trough/results.json", encoding="utf8"))
GR = json.load(open(f"{RES}/128-stag-given-stageloss/grid.json"))
PS = np.array(GR["PS"]); SL = "slow_Tasmania"
FC = np.array([GR["GRID"][f"{p:.6f}|{SL}|D2rz|none"]["F_heavy"] for p in PS])
def Fx(x):
    x = np.atleast_1d(np.asarray(x, float)); y = np.interp(x, PS, FC); lo = x < PS[0]
    y[lo] = FC[0] + (x[lo] - PS[0]) * (FC[1] - FC[0]) / (PS[1] - PS[0]); return y
A0 = B0 = 0.5  # Jeffreys
def mix_post(k_fix, n_fix, w_list):
    """固定の k/n に、確率 w_i で停滞の事例を足した混合の事後（停滞しなければ非停滞の 1 件）。
    混合の成分は 2^m 通り。返り値：p の平均、F_heavy の平均（格子の補間を事後で積分）。"""
    comps = [(1.0, k_fix, n_fix)]
    for w in w_list:
        comps = [(c * w, k + 1, n + 1) for c, k, n in comps] + [(c * (1 - w), k, n + 1) for c, k, n in comps]
    pm = 0.0; Fm = 0.0
    for c, k, n in comps:
        if c == 0: continue
        d = stats.beta(A0 + k, B0 + n - k); pm += c * d.mean()
        Fm += c * integrate.quad(lambda x: Fx(x)[0] * d.pdf(x), 0, 1, limit=200)[0]
    return pm, Fm
def post_kn(k, n): return mix_post(k, n, [])
CHECK = {}
# ---- 再現：144 の主（その場、Copán w=0.070375）と 143 の 3/3・2/3、144 の登録外 2/2 ----
w144 = R144["variants"]["main_reg[200,1000]"]["w"]
pm, Fm = mix_post(2, 2, [w144]); ref = R144["variants"]["main_reg[200,1000]"]
CHECK["144_main"] = {"p": abs(pm - ref["p_mean"]), "F": abs(Fm - ref["F_mean"])}
pm2, Fm2 = post_kn(2, 2); ref2 = R144["variants"]["unreg_Copan_as_abandon(insitu 2/2)"]
CHECK["144_2of2"] = {"p": abs(pm2 - ref2["p_mean"]), "F": abs(Fm2 - ref2["F_mean"])}
for v in CHECK.values(): assert v["p"] < 1e-9 and v["F"] < 1e-6, CHECK
# ---- 事例（143・144 の値。Pmin/P0 は谷の割合） ----
P0_COP = 26500.0
CASES = {  # lost の 9 件。frac=None は放棄（消滅）
 "Greece_LBA": {"frac": 300000 / 600000, "stag": True},
 "Britain_postRoman": {"frac": 1e6 / 3e6, "stag": True},
 "Norse_Greenland": {"frac": 0.0, "stag": True, "abandon": True},
 "Cahokia": {"frac": None, "stag": True, "abandon": True},
 "Tikal": {"frac": None, "stag": True, "abandon": True},
 "Tiwanaku": {"frac": None, "stag": True, "abandon": True},
 "Mesa_Verde": {"frac": None, "stag": True, "abandon": True},
 "Great_Zimbabwe": {"frac": None, "stag": True, "abandon": True},
 "Copan": {"frac": "logU[200,1000]/26500", "stag": "w(r)"},
}
def p_cop_deep(th):  # Copán の Pmin が th·P0 未満の確率（Pmin は [200,1000] の対数一様）
    x = th * P0_COP
    return float(np.clip(math.log(x / 200) / math.log(5), 0, 1))
# ---- 334：r による Copán の停滞の確率 w（144 の w_stag と同じ格子・同じ式を独立に再計算） ----
r_lo, r_hi = 0.0014, 0.0061
rq = {q: math.exp(math.log(r_lo) + q * (math.log(r_hi) - math.log(r_lo))) for q in (0.25, 0.5, 0.75)}
R038q = json.load(open(f"{RES}/038-restart-state-evidence/results.json"))["thresholds_quantiles"]["r_pop"]
def w_stag(r, lo=200, hi=1000, n=400001, P0s=(26000, 27000), tss=(800, 900), trecs=(1998, 2001)):
    pm = np.exp(np.linspace(math.log(lo), math.log(hi), n)); out = []
    for p0 in P0s:
        for ts in tss:
            for tr in trecs:
                out.append(float(np.mean((tr - ts) - np.log(p0 / pm) / r > 0)))
    return float(np.mean(out))
CHECK["144_w_regrid"] = abs(w_stag(R144["r_median_038"]) - w144)  # 144 は MC（n=2e5）なので MC 誤差の範囲で一致
assert CHECK["144_w_regrid"] < 0.004, CHECK
RDEP = {}
for q, r in list(rq.items()) + [("038_median_exact", R144["r_median_038"])]:
    w = w_stag(r); pm, Fm = mix_post(2, 2, [w]); RDEP[str(q)] = {"r": r, "w_Copan": w, "insitu_p": pm, "insitu_F": Fm}
# r を 038 の事前（対数一様 [0.0014,0.0061]）で周辺化した w（登録外の追加）
_rg = np.exp(np.linspace(math.log(r_lo), math.log(r_hi), 241)); _wg = np.array([w_stag(x, n=20001) for x in _rg])
w_marg = float(np.mean(_wg)); pmM, FmM = mix_post(2, 2, [w_marg])
# w の混合は成分に線形なので、w を周辺化してから混合しても r ごとの F の平均と一致する
FmM2 = float(np.mean([mix_post(2, 2, [x])[1] for x in _wg[::8]])); 
RDEP["marginal_over_r_prior"] = {"w_Copan": w_marg, "insitu_p": pmM, "insitu_F": FmM, "F_check_coarse": FmM2}
# 損益分岐の r（P0 26,500、ts 800、Trec 2001 で Pmin=1000 と 200 のとき停滞になる最小の r）
RDEP["r_breakeven_Pmin1000"] = math.log(26500 / 1000) / (2001 - 800)
RDEP["r_breakeven_Pmin200"] = math.log(26500 / 200) / (2001 - 800)
# ---- R333：閾値ごとのその場の集合 ----
RULE = {}
for th in (0.01, 0.02, 0.05, 0.10):
    pd = p_cop_deep(th)
    # Copán は確率 pd で再入植の型（その場から外れる）。外れない場合、その場の 1 件として w の停滞。
    # その場の事後 = pd·(2/2) + (1-pd)·(2/2 + Copán[w])
    p_out, F_out = post_kn(2, 2); p_in, F_in = mix_post(2, 2, [w144])
    RULE[f"{th:.2f}"] = {"P_Copan_deep": pd, "insitu_p": pd * p_out + (1 - pd) * p_in, "insitu_F": pd * F_out + (1 - pd) * F_in,
                         "deep_set_n": 7 + pd, "shallow_set": ["Greece_LBA", "Britain_postRoman"] + (["Copan(partial)"] if pd < 1 else [])}
# ---- 模型の側：038 の重い標本の生き残りの割合 s ----
W = "/tmp/rep145/research"; os.makedirs(f"{W}/038-restart-state-evidence", exist_ok=True)
for d in os.listdir(RES):
    if d != "038-restart-state-evidence" and not os.path.exists(f"{W}/{d}"): os.symlink(f"{RES}/{d}", f"{W}/{d}")
shutil.copy(f"{RES}/038-restart-state-evidence/sim.py", f"{W}/038-restart-state-evidence/sim.py")
_cwd = os.getcwd(); os.chdir(f"{W}/038-restart-state-evidence")
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(f"{W}/038-restart-state-evidence/sim.py", run_name="m145")
os.chdir(_cwd)
R038 = json.load(open(f"{RES}/038-restart-state-evidence/results.json"))
EVID = G["EVID"]; HEAVY = G["HEAVY"]; WT = G["WT"]
MOD = {"per_scen": {}, "weighted": {}}
SV = {}
for s in EVID:
    sv = G["survivors"](s); st = G["assign"](sv); hv = np.isin(st, HEAVY); SV[s] = (sv, hv)
    d = abs(float(hv.mean()) - R038["scenarios_038"][s]["heavy_share"]); CHECK[f"038_heavy_share_{s}"] = d; assert d < 1e-12
    ss = sv["s"][hv]
    MOD["per_scen"][s] = {"heavy_share": float(hv.mean()), "n_heavy": int(hv.sum()),
        "s_q05_50_95_heavy": [float(x) for x in np.percentile(ss, [5, 50, 95])] if hv.any() else None,
        "f_deep": {f"{th:.2f}": (float(np.mean(ss < th)) if hv.any() else None) for th in (0.01, 0.02, 0.05, 0.10)}}
for a in ("A_Ord", "A_XPT_S", "A_XPT_E"):
    w = G["weights"](a)[0]
    num = {f"{th:.2f}": 0.0 for th in (0.01, 0.02, 0.05, 0.10)}; den = 0.0; den_all = 0.0
    for s in EVID:
        sv, hv = SV[s]; ws = w[s]
        den += float(np.mean(ws * hv))
        for th in num: num[th] += float(np.mean(ws * hv * (sv["s"] < float(th))))
    # AI・GEN の重い分（s なし）の重みも並べる
    hv_other = 0.0
    for s in ("AI", "GEN"):
        st = G["DRAWS"][s]["st"]; hv_other += float(np.mean(w[s] * np.isin(st, HEAVY)))
    MOD["weighted"][a] = {"f_deep_EVID": {th: num[th] / den for th in num}, "heavy_weight_EVID": den, "heavy_weight_AI_GEN_no_s": hv_other,
                          "share_of_heavy_without_s": hv_other / (den + hv_other)}
# ---- 結合 ----
COMB = {}
p_deep_all, F_deep_all = post_kn(7, 7)  # 再入植の型 7 件（放棄 6＋Copán）、規則で停滞
for a in ("A_Ord", "A_XPT_S", "A_XPT_E"):
    for th in ("0.05", "0.01", "0.10"):
        f = MOD["weighted"][a]["f_deep_EVID"][th]; ru = RULE[th]
        # 読み A：深い側に再入植の型の p を当てる（F は格子が p にほぼ線形なので F の混合で近似）
        F_A = f * F_deep_all + (1 - f) * ru["insitu_F"]
        # 読み B：深い側は群の喪失として停滞の集計から外す＝その場の p だけ
        F_B = ru["insitu_F"]
        COMB[f"{a}|th={th}"] = {"f_deep": f, "F_readA": F_A, "F_readB": F_B, "p_readA": f * p_deep_all + (1 - f) * ru["insitu_p"]}
CHECK["Fx_linearity_maxdev_0.5_1"] = float(np.max(np.abs(Fx(np.linspace(0.5, 1, 51)) - np.interp(np.linspace(0.5, 1, 51), [0.5, 1], [Fx(0.5)[0], Fx(1.0)[0]]))))
# ---- 仮説 ----
fdS = MOD["weighted"]["A_XPT_S"]["f_deep_EVID"]["0.05"]
H = {
 "H157a": {"class": "confirmatory", "rule": "A_XPT_S の重い標本（EVID）で s<5% の割合 >= 0.5", "value": fdS, "supported": fdS >= 0.5},
 "H157b": {"class": "confirmatory", "rule": "r 上の四分位で w>=0.5 かつ下の四分位で w<0.1", "value": [RDEP["0.75"]["w_Copan"], RDEP["0.25"]["w_Copan"]],
           "supported": RDEP["0.75"]["w_Copan"] >= 0.5 and RDEP["0.25"]["w_Copan"] < 0.1},
 "H157c": {"class": "prediction", "rule": "R333(5%) のその場の F が 2.12 と差 <0.001", "value": RULE["0.05"]["insitu_F"],
           "supported": abs(RULE["0.05"]["insitu_F"] - ref2["F_mean"]) < 1e-3},
 "H157d": {"class": "prediction", "rule": "結合の F が全版で 1.94〜2.24", "value": [min(v[k] for v in COMB.values() for k in ("F_readA", "F_readB")), max(v[k] for v in COMB.values() for k in ("F_readA", "F_readB"))]},
}
H["H157d"]["supported"] = 1.94 <= H["H157d"]["value"][0] and H["H157d"]["value"][1] <= 2.24
OUT = {"study": "145-resettlement-rule", "note": "F_heavy・p|lost は生存確率や復興確率ではない", "check": CHECK, "cases": CASES, "r_quartiles": rq,
       "r_quartiles_038_results": R038q, "r_dependence": RDEP, "rule_R333": RULE, "model_survival_fraction": MOD,
       "deep_set_7of7": {"p": p_deep_all, "F": F_deep_all}, "combined": COMB, "hypotheses": H, "runtime_s": None}
# ---- 図 ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
rr = np.exp(np.linspace(math.log(r_lo), math.log(r_hi), 60)); ww = [w_stag(x, n=20001) for x in rr]
ax[0].plot(rr * 1000, ww, label="w (Copán stagnant)"); ax[0].plot(rr * 1000, [mix_post(2, 2, [x])[1] - 1 for x in ww], label="in-situ F_heavy − 1")
for q, r in rq.items(): ax[0].axvline(r * 1000, c="gray", ls=":", lw=.8)
ax[0].set_xscale("log"); ax[0].set_xlabel("r (per 1000 yr)  [038 prior range, quartiles dotted]"); ax[0].legend(fontsize=8); ax[0].set_title("334: Copán verdict vs r", fontsize=9)
sc = [s for s in EVID if SV[s][1].any()]
for s in sc:
    sv, hv = SV[s]; x = np.sort(sv["s"][hv]); ax[1].plot(x, np.arange(1, len(x) + 1) / len(x), label=f"{s} (heavy {hv.mean():.3f})")
ax[1].axvline(0.05, c="r", ls="--", lw=.8, label="R333 5%"); ax[1].set_xscale("log"); ax[1].set_xlabel("global survival fraction s (heavy samples, 038)"); ax[1].set_ylabel("CDF"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_resettlement.png", dpi=120)
OUT["runtime_s"] = time.time() - T0
json.dump(OUT, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
print("check", CHECK); print("rq", rq, "038", R038q)
for k, v in RDEP.items(): print("rdep", k, v)
for k, v in RULE.items(): print("rule", k, v)
print(json.dumps(MOD, indent=1, default=float)); print("deep7/7", p_deep_all, F_deep_all)
for k, v in COMB.items(): print("comb", k, {a: round(b, 4) for a, b in v.items()})
print({k: (v["supported"], v["value"]) for k, v in H.items()}); print("runtime", round(OUT["runtime_s"], 1))
