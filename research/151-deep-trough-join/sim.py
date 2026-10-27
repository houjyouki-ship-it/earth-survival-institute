"""研究151：深い谷の標本を停滞と群の喪失のどちらで数えるか（open_problems 336）。
計算のみ（入力は 038 の sim.py の写し、070・128・145・146 の results.json、146 の sim.py の定義部）。
λ の資料は sources/ の Wikipedia 本文（2026-10-27 取得）。事前登録：PREREGISTRATION.md（19:45 JST）。
注：F_heavy・TC・V は生存確率や復興確率ではない（模型の中の量）。"""
import json, os, time, shutil
import numpy as np
from scipy import stats, integrate
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
R146 = json.load(open(f"{RES}/146-aigen-survivors/results.json", encoding="utf8"))
R145 = json.load(open(f"{RES}/145-resettlement-rule/results.json", encoding="utf8"))
# ---- 146 の定義部を写しで実行（「# ---- 版 ----」の前まで） ----
W = "/tmp/rep151/research"; os.makedirs(f"{W}/146-aigen-survivors", exist_ok=True)
for d in os.listdir(RES):
    if d != "146-aigen-survivors" and not os.path.exists(f"{W}/{d}"): os.symlink(f"{RES}/{d}", f"{W}/{d}")
src = open(f"{RES}/146-aigen-survivors/sim.py", encoding="utf8").read(); src = src[:src.index("# ---- 版 ----")]
ns = {"__file__": f"{W}/146-aigen-survivors/sim.py", "__name__": "m151"}
exec(compile(src, "146head", "exec"), ns)
N, EVID, HEAVY, ST, SV, weights, resample, geff, tc_of = (ns[k] for k in ("N", "EVID", "HEAVY", "ST", "SV", "weights", "resample", "geff", "tc_of"))
ANCH = ("A_Ord", "A_XPT_S", "A_XPT_E"); VARS = ("P_BIO", "P_CASC", "P_MIX", "P_EVIDW"); TH = 0.05
CHECK = {}
# ---- 128 の格子 ----
GR = json.load(open(f"{RES}/128-stag-given-stageloss/grid.json")); PS = np.array(GR["PS"])
def Fgrid(key):
    FC = np.array([GR["GRID"][f"{p:.6f}|{key}"]["F_heavy"] for p in PS])
    def f(x):
        x = np.atleast_1d(np.asarray(x, float)); y = np.interp(x, PS, FC); lo = x < PS[0]
        y[lo] = FC[0] + (x[lo] - PS[0]) * (FC[1] - FC[0]) / (PS[1] - PS[0]); return y
    return f
DES = {"D2rz": Fgrid("slow_Tasmania|D2rz|none"), "D1": Fgrid("slow_Tasmania|D1|main"), "D4": Fgrid("slow_Tasmania|D4|main")}
FK = {d: {} for d in DES}
def F_k(d, k):  # 停滞 k 件／k 件（Jeffreys）の事後で F を積分
    if k not in FK[d]:
        b = stats.beta(0.5 + k, 0.5); FK[d][k] = integrate.quad(lambda x: DES[d](x)[0] * b.pdf(x), 0, 1, limit=200)[0]
    return FK[d][k]
F_IN = {d: F_k(d, 2) for d in DES}
CHECK["F_in_vs_145"] = abs(F_IN["D2rz"] - R145["rule_R333"]["0.05"]["insitu_F"]); assert CHECK["F_in_vs_145"] < 1e-9
CHECK["F_7of7_vs_145"] = abs(F_k("D2rz", 7) - R145["deep_set_7of7"]["F"]); assert CHECK["F_7of7_vs_145"] < 1e-9
def F_deep(d, lam):
    return sum(stats.binom.pmf(j, 7, 1 - lam) * F_k(d, 2 + j) for j in range(8))
# ---- 146 の引き直しを再現し、重い標本の (重み, 深い, G) を集める ----
SAMP = {}
for a in ANCH:
    w = weights(a)[0]
    evw = {s: float(np.mean(w[s] * np.isin(ST[s], HEAVY))) for s in EVID}
    VV = {"P_BIO": {"BIO": 1.0}, "P_CASC": {"CASC": 1.0}, "P_MIX": {"BIO": 0.5, "CASC": 0.5}, "P_EVIDW": evw}
    for vn in VARS:
        rng = np.random.default_rng(20261146); svx = {}
        for s in ("AI", "GEN"):
            svx[s] = resample(s, VV[vn], rng)
            _ = rng  # 146 はこの後に乱数を使わない（times は決定論）
        ws_, dp_, g_ = [], [], []
        for s in list(EVID) + ["AI", "GEN"]:
            sv = SV[s] if s in EVID else svx[s]; hv = np.isin(ST[s], HEAVY)
            ws = np.broadcast_to(np.asarray(w[s], float), (N,)) * hv
            ws_.append(ws[hv]); dp_.append(sv["s"][hv] < TH); g_.append(geff(sv)[hv])
        ws_, dp_, g_ = np.concatenate(ws_), np.concatenate(dp_), np.concatenate(g_).astype(int)
        f = float(np.sum(ws_ * dp_) / np.sum(ws_)); tc = float(np.sum(ws_ * tc_of(g_)) / np.sum(ws_))
        ref = R146["variants"][f"{a}|{vn}"]
        CHECK[f"f_deep|{a}|{vn}"] = abs(f - ref["th=0.05"]["f_deep_all"]); CHECK[f"TC|{a}|{vn}"] = abs(tc - ref["TC_070_interp_heavy"])
        assert CHECK[f"f_deep|{a}|{vn}"] < 1e-9 and CHECK[f"TC|{a}|{vn}"] < 1e-9, (a, vn, f, tc)
        FA = f * F_k("D2rz", 7) + (1 - f) * F_IN["D2rz"]; CHECK[f"FA|{a}|{vn}"] = abs(FA - ref["th=0.05"]["F_readA"]); assert CHECK[f"FA|{a}|{vn}"] < 1e-9
        SAMP[(a, vn)] = (ws_, dp_, g_, f, FA)
TCG = {g: float(tc_of(np.array([g]))[0]) for g in range(0, 11) if g > 0}; TCG[0] = 0.0
def eval_lam(a, vn, lam, ver):
    ws, dp, g, f, FA = SAMP[(a, vn)]
    # 深い標本の G' 〜 Bin(G, 1-λ)
    Gu = np.unique(g[dp]) if dp.any() else np.array([], int)
    eTC = {}; pext = {}
    for G in Gu:
        pm = stats.binom.pmf(np.arange(G + 1), G, 1 - lam)
        gg = np.arange(G + 1) if ver == "death" else np.maximum(np.arange(G + 1), 1)
        eTC[G] = float(np.sum(pm * np.array([TCG[int(x)] for x in gg]))); pext[G] = float(pm[0]) if ver == "death" else 0.0
    tc0 = np.array([TCG[int(x)] for x in g]); tc1 = tc0.copy(); surv = np.ones(len(g))
    for G in Gu:
        m = dp & (g == G); tc1[m] = eTC[G]; surv[m] = 1 - pext[G]
    V = float(np.sum(ws * tc1) / np.sum(ws * tc0)); Pext = float(np.sum(ws * (1 - surv)) / np.sum(ws))
    fs = float(np.sum(ws * dp * surv) / np.sum(ws * surv)) if np.sum(ws * surv) > 0 else 0.0  # 生き残った重い標本の中の深い割合
    out = {"V_rel_lam0": V, "P_ext_heavy": Pext, "f_deep_surv": fs}
    for d in DES: out[f"F_{d}"] = fs * F_deep(d, lam) + (1 - fs) * F_IN[d]
    FB = F_IN["D2rz"]; out["pos_B0_A1"] = (out["F_D2rz"] - FB) / (FA - FB) if FA - FB > 1e-12 else None
    return out
LAMS = [0.0, 0.1, 0.143, 0.25, 0.357, 0.5, 0.75, 0.9, 1.0]
TAB = {}
for (a, vn) in SAMP:
    for ver in ("merge", "death"):
        for lam in LAMS: TAB[f"{a}|{vn}|{ver}|{lam}"] = eval_lam(a, vn, lam, ver)
        # λ〜U(0,1)：F と V を λ で平均（F は f_deep_surv が λ に依る death 版でもそのまま平均）
        lg = np.linspace(0, 1, 101); rr = [eval_lam(a, vn, x, ver) for x in lg]
        U = {k: float(np.mean([r[k] for r in rr])) for k in ("F_D2rz", "F_D1", "F_D4", "V_rel_lam0", "P_ext_heavy")}
        ws, dp, g, f, FA = SAMP[(a, vn)]; FB = F_IN["D2rz"]
        U["pos_B0_A1"] = (U["F_D2rz"] - FB) / (FA - FB) if FA - FB > 1e-12 else None
        TAB[f"{a}|{vn}|{ver}|U01"] = U
# λ=1・merge は読み B と差 0
CHECK["lam1_merge_eq_readB_max"] = max(abs(TAB[f"{a}|{vn}|merge|1.0"]["F_D2rz"] - F_IN["D2rz"]) for (a, vn) in SAMP); assert CHECK["lam1_merge_eq_readB_max"] < 1e-9
# ---- λ の資料（sources/ の Wikipedia 本文、2026-10-27 取得） ----
LAMSRC = {"Mesa_Verde": (1.0, 1.0, "they migrated south to locations in Arizona and New Mexico"),
          "Tikal": (0.0, 0.5, "Some of Tikal's population may have migrated to the Peten Lakes region"),
          "Cahokia": (0.0, 0.5, "The immigrant populations inhabiting upland villages ... were some of the first to leave the city"),
          "Great_Zimbabwe": (0.0, 0.5, "political and economic power had shifted away from Great Zimbabwe to the north and west"),
          "Tiwanaku": (0.0, 0.0, "collapsed around AD 1000; the reason for this is still open to debate"),
          "Norse_Greenland": (0.0, 0.0, "the Norse language most likely disappeared with the ethnic group that spoke it"),
          "Copan": (0.0, 0.0, "登録どおり 0（144 で人口が残った）")}
lam_strict = sum(v[0] for v in LAMSRC.values()) / 7; lam_len = sum(v[1] for v in LAMSRC.values()) / 7
for nm, lam in (("strict", lam_strict), ("lenient", lam_len)):
    for (a, vn) in SAMP:
        for ver in ("merge", "death"): TAB[f"{a}|{vn}|{ver}|src_{nm}"] = eval_lam(a, vn, lam, ver)
# ---- 仮説 ----
k0 = "A_XPT_S|P_MIX"
H = {}
v = TAB[f"{k0}|merge|0.0"]["F_D2rz"]; H["H163a"] = {"class": "confirmatory", "value": v, "ref_readA": SAMP[("A_XPT_S", "P_MIX")][4], "supported": abs(v - SAMP[("A_XPT_S", "P_MIX")][4]) < 0.02}
v = TAB[f"{k0}|merge|U01"]["pos_B0_A1"]; H["H163b"] = {"class": "prediction", "value": v, "supported": v >= 0.5}
v = TAB[f"{k0}|death|0.5"]["V_rel_lam0"]; H["H163c"] = {"class": "prediction", "value": 1 - v, "supported": (1 - v) >= 0.05}
ok = all(r["F_D2rz"] >= r["F_D1"] >= r["F_D4"] for r in TAB.values()); H["H163d"] = {"class": "confirmatory", "value": ok, "supported": ok}
n_mig = sum(1 for k, v_ in LAMSRC.items() if k != "Copan" and v_[0] >= 1)
pos_s = TAB[f"{k0}|merge|src_strict"]["pos_B0_A1"]
H["H163e"] = {"class": "prediction", "value": {"n_migration_explicit": n_mig, "lam_strict": lam_strict, "lam_lenient": lam_len, "pos_at_strict": pos_s},
              "supported": n_mig >= 4 and pos_s < 0.5}
OUT = {"study": "151-deep-trough-join", "note": "F_heavy・TC・V は生存確率や復興確率ではない", "check": CHECK, "F_in": F_IN,
       "F_k_D2rz": {k: v for k, v in FK["D2rz"].items()}, "lambda_sources": LAMSRC, "lam_strict": lam_strict, "lam_lenient": lam_len,
       "table": TAB, "hypotheses": H}
# ---- 図 ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4)); lg = np.linspace(0, 1, 51)
for vn in VARS:
    ax[0].plot(lg, [eval_lam("A_XPT_S", vn, x, "merge")["F_D2rz"] for x in lg], label=f"A_XPT_S {vn}")
    ax[1].plot(lg, [eval_lam("A_XPT_S", vn, x, "death")["V_rel_lam0"] for x in lg], label=f"death {vn}")
for x, ls in ((lam_strict, ":"), (lam_len, "--")): ax[0].axvline(x, c="k", ls=ls, lw=.8); ax[1].axvline(x, c="k", ls=ls, lw=.8)
ax[0].set_xlabel("λ (share of abandonment cases counted as group loss)"); ax[0].set_ylabel("F_heavy (D2rz), joined"); ax[0].legend(fontsize=7)
ax[1].set_xlabel("λ (per-group loss in deep samples)"); ax[1].set_ylabel("heavy value rel. to λ=0 (TC of 070)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_join.png", dpi=120)
OUT["runtime_s"] = time.time() - T0
json.dump(OUT, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
print("check max", max(v for v in CHECK.values())); print("F_in", F_IN); print("lam", lam_strict, lam_len)
for k in ("merge|0.0", "merge|src_strict", "merge|src_lenient", "merge|U01", "merge|1.0", "death|src_strict", "death|0.5", "death|U01"):
    for a in ANCH:
        for vn in VARS:
            r = TAB[f"{a}|{vn}|{k}"]; print(k, a, vn, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in r.items()})
print(json.dumps(H, ensure_ascii=False, default=float)); print("runtime", round(OUT["runtime_s"], 1))
